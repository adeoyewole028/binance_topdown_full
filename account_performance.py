"""Rebuildable full-history account performance, never a financial write."""
from __future__ import annotations
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import Any
from paper_policy import decimal_text,financial_arithmetic
from operations_policy import OperationsPolicy

REASON_MESSAGES={
    'setup_required':'Configure starting capital and trading assumptions.',
    'initial_equity_unavailable':'Starting account equity is unavailable.',
    'imported_history_unknown':'Imported records lack a complete financial history.',
    'conversion_unavailable':'No current currency conversion is available for {asset}.',
    'valuation_stale':'The {asset} valuation is stale.',
    'valuation_invalid':'The {asset} price observation is invalid.',
    'reconciliation_required':'Account reconciliation is required.',
    'account_observation_stale':'The account observation is stale.',
    'fee_history_unknown':'Recorded fees are incomplete.',
    'historical_quote_conversion_unavailable':'A historical currency conversion is unavailable for {asset}.',
    'historical_fee_conversion_unavailable':'A historical fee conversion is unavailable for {asset}.',
    'cost_basis_unknown':'Cost basis is unknown for {asset}.',
    'fill_cost_unknown':'A recorded fill lacks confirmed cost.',
    'holding_balance_mismatch':'The {asset} holding does not match account balances.',
    'balance_history_mismatch':'Balances do not match the recorded financial history.',
    'historical_equity_incomplete':'Historical equity observations are incomplete.',
}


def ledger_value(value: Any) -> Decimal:
    """Derived stored money may exceed the precision of operator inputs."""
    try:
        result=Decimal(str(value))
        if isinstance(value,bool) or not result.is_finite():raise ValueError('Invalid ledger value')
        return result
    except InvalidOperation as error:
        raise ValueError('Invalid ledger value') from error


@financial_arithmetic
def weighted_entry(held: Decimal,basis: Decimal,received: Decimal,cost: Decimal) -> Decimal:
    if received<=0 or min(held,basis,cost)<0:raise ValueError('Invalid weighted entry')
    return (held*basis+cost)/(held+received)


@financial_arithmetic
def realized_exit(consumed: Decimal,basis: Decimal,proceeds: Decimal) -> Decimal:
    if min(consumed,basis,proceeds)<0:raise ValueError('Invalid realized exit')
    return proceeds-consumed*basis


@financial_arithmetic
def project(*,snapshot: dict[str,Any],mode: str,now: float,
            opening: dict[str,Any],checkpoints: list[dict[str,Any]],
            imported: bool) -> dict[str,Any]:
    config=snapshot['config']
    policy=config.get('paper_policy' if mode=='paper' else 'execution_policy')
    currency=policy['quote'] if policy else None
    result: dict[str,Any]={'state':'unavailable','reasons':[],'reason_messages':[], 'currency':currency,
        'realized':None,'unrealized':None,'equity':None,'initial_equity':None,'peak_equity':None,
        'drawdown':None,'return_fraction':None,'win_rate':None,'completed_trades':None,
        'win_rate_denominator':'completed_full_position_exits','cost_basis':'fee_inclusive_weighted_average',
        'total_profit':None,'valuation':'estimated_from_fresh_observed_prices',
        'fees_by_asset':{},'realized_by_currency':{},'positions':[],
        'valuation_sources':{},'fee_conversion':'native_fill_asset; no historical conversion invented',
        'history_fills':len(snapshot['fills']),'historical_price_coverage':'recorded_fresh_equity_checkpoints'}
    reasons: set[str]=set()
    if not policy:
        result['reasons']=['setup_required']
        result['reason_messages']=[REASON_MESSAGES['setup_required']]
        return result
    assert isinstance(currency,str)
    def conversion(asset: str) -> Decimal | None:
        if asset==currency:return Decimal(1)
        row=snapshot['market_prices'].get(asset+'/'+currency)
        if row is None:
            reasons.add('conversion_unavailable:'+asset)
            result['valuation_sources'][asset]={'state':'unavailable','symbol':asset+'/'+currency}
            return None
        try:
            stamp=datetime.fromisoformat(row['fetched_at'])
        except (ValueError,TypeError):
            reasons.add('valuation_invalid:'+asset)
            result['valuation_sources'][asset]={'state':'invalid','symbol':asset+'/'+currency}
            return None
        age=now-stamp.timestamp()
        result['valuation_sources'][asset]={'state':'fresh' if stamp.tzinfo is not None and 0<=age<OperationsPolicy().price_expiry else 'stale',
            'symbol':asset+'/'+currency,'price':row['price'],'fetched_at':row['fetched_at'],'age_seconds':age}
        if stamp.tzinfo is None or not 0<=age<OperationsPolicy().price_expiry:
            reasons.add('valuation_stale:'+asset)
            return None
        return ledger_value(row['price'])
    account=snapshot['paper_account' if mode=='paper' else 'exchange_account']
    account_fresh=True
    if mode=='exchange':
        if account.get('state')!='ready':
            reasons.add('reconciliation_required')
            account_fresh=False
        stamp=account.get('observed_at')
        if isinstance(stamp,bool) or not isinstance(stamp,(int,float)) or not 0<=now-stamp<OperationsPolicy().price_expiry:
            reasons.add('account_observation_stale')
            account_fresh=False
    equity=Decimal(0)
    equity_known=bool(account.get('balances')) and account_fresh
    for asset,values in account.get('balances',{}).items():
        quantity=ledger_value(values['total'])
        if not quantity:continue
        rate=conversion(asset)
        if rate is None:equity_known=False
        else:equity+=quantity*rate
    if equity_known:result['equity']=decimal_text(equity)
    initial=opening.get('equity')
    seed=ledger_value(initial) if initial is not None else None
    if mode=='exchange' and opening.get('history_fill_count',0)>0:seed=None
    if seed is None or seed<=0:reasons.add('initial_equity_unavailable')
    else:result['initial_equity']=decimal_text(seed)
    holdings: dict[str,dict[str,Any]]={}
    reconstructed={asset:ledger_value(values['total']) for asset,values in opening.get('balances',{}).items()}
    if mode=='paper' and seed is not None:reconstructed[currency]=seed
    realized: dict[str,Decimal]={}
    fees: dict[str,Decimal]={}
    closed: list[Decimal]=[]
    seen: set[str]=set()
    unknown=imported
    if imported:reasons.add('imported_history_unknown')
    for fill in snapshot['fills']:
        identity=fill['fill_id']
        if identity in seen:continue
        seen.add(identity)
        symbol=fill['symbol']
        base,quote=symbol.split('/')
        fee=ledger_value(fill['fee_amount']) if fill.get('fee_amount') is not None else None
        asset=fill.get('fee_asset')
        if fee is None or not asset:
            unknown=True
            reasons.add('fee_history_unknown')
            fee=Decimal(0)
        else:fees[asset]=fees.get(asset,Decimal(0))+fee
        if fill.get('qty') is None or fill.get('quote_cost') is None:
            unknown=True
            reasons.add('fill_cost_unknown')
            continue
        quantity,cost=(ledger_value(fill[k]) for k in ('qty','quote_cost'))
        direction=Decimal(1) if fill['side']=='buy' else Decimal(-1)
        reconstructed[base]=reconstructed.get(base,Decimal(0))+direction*quantity
        reconstructed[quote]=reconstructed.get(quote,Decimal(0))-direction*cost
        if asset:reconstructed[asset]=reconstructed.get(asset,Decimal(0))-fee
        if quote!=currency:
            unknown=True
            reasons.add('historical_quote_conversion_unavailable:'+quote)
        if fee and asset not in {base,quote}:
            unknown=True
            reasons.add('historical_fee_conversion_unavailable:'+asset)
        held=holdings.setdefault(symbol,{'quantity':Decimal(0),'basis':Decimal(0),'cycle':Decimal(0)})
        if fill['side']=='buy':
            received=quantity-(fee if asset==base else Decimal(0))
            if received<=0:
                unknown=True
                reasons.add('cost_basis_unknown:'+symbol)
                continue
            held['basis']=weighted_entry(held['quantity'],held['basis'],received,cost+(fee if asset==quote else Decimal(0)))
            held['quantity']+=received
        else:
            consumed=quantity+(fee if asset==base else Decimal(0))
            if consumed>held['quantity']:
                unknown=True
                reasons.add('cost_basis_unknown:'+symbol)
                continue
            profit=realized_exit(consumed,held['basis'],cost-(fee if asset==quote else Decimal(0)))
            realized[quote]=realized.get(quote,Decimal(0))+profit
            held['cycle']+=profit
            held['quantity']-=consumed
            if not held['quantity']:
                if quote==currency:closed.append(held['cycle'])
                held.update(basis=Decimal(0),cycle=Decimal(0))
    result['fees_by_asset']={asset:decimal_text(value) for asset,value in sorted(fees.items())}
    result['realized_by_currency']={asset:decimal_text(value) for asset,value in sorted(realized.items())}
    if not imported and (mode=='paper' or opening.get('balances')):
        if any(reconstructed.get(asset,Decimal(0))!=ledger_value(account.get('balances',{}).get(asset,{}).get('total','0'))
               for asset in set(reconstructed)|set(account.get('balances',{}))):
            equity_known=False
            result['equity']=None
            reasons.add('balance_history_mismatch')
    unrealized=Decimal(0)
    unrealized_known=account_fresh
    for position in snapshot['positions']:
        symbol=position['symbol']
        base,quote=symbol.split('/')
        current_holding=holdings.get(symbol)
        quantity=ledger_value(position['qty'])
        if ledger_value(account.get('balances',{}).get(base,{}).get('total','0'))!=quantity:
            equity_known=False
            result['equity']=None
            reasons.add('holding_balance_mismatch:'+symbol)
        if current_holding is None or current_holding['quantity']!=quantity or position['basis_status'].startswith(('legacy-opening','provider_basis_unknown')):
            unknown=True
            unrealized_known=False
            reasons.add('cost_basis_unknown:'+symbol)
            result['positions'].append({'symbol':symbol,'quantity':decimal_text(quantity),'unrealized':None,'currency':currency})
            continue
        rate=conversion(base)
        if rate is None or quote!=currency or not account_fresh:
            unrealized_known=False
            value=None
        else:
            value=quantity*(rate-current_holding['basis'])
            unrealized+=value
        result['positions'].append({'symbol':symbol,'quantity':decimal_text(quantity),
            'entry':decimal_text(current_holding['basis']),'unrealized':decimal_text(value) if value is not None else None,'currency':currency})
    if not unknown:
        result.update(realized=decimal_text(realized.get(currency,Decimal(0))),completed_trades=len(closed),
            win_rate=decimal_text(Decimal(sum(value>0 for value in closed))/len(closed)) if closed else None)
        if unrealized_known:result['unrealized']=decimal_text(unrealized)
        if equity_known and seed is not None and seed>0:
            result['total_profit']=decimal_text(equity-seed)
            result['return_fraction']=decimal_text((equity-seed)/seed)
    # Every confirmed fill prefix needs a qualified observation. A batch-final
    # observation cannot prove the equity path between its individual fills.
    counts=[checkpoint.get('history_fill_count') for checkpoint in checkpoints]
    valid_counts=[count for count in counts if isinstance(count,int) and not isinstance(count,bool) and 0<=count<=len(seen)]
    historical_equity_known=not seen or (len(valid_counts)==len(counts) and
        valid_counts==sorted(valid_counts) and set(valid_counts)==set(range(len(seen)+1)))
    if not historical_equity_known:reasons.add('historical_equity_incomplete')
    if seed is not None and seed>0 and historical_equity_known:
        peak=seed
        maximum=Decimal(0)
        for checkpoint in checkpoints:
            value=ledger_value(checkpoint['equity'])
            peak=max(peak,value)
            maximum=max(maximum,(peak-value)/peak)
        if equity_known:
            peak=max(peak,equity)
            maximum=max(maximum,(peak-equity)/peak)
            result.update(peak_equity=decimal_text(peak),drawdown=decimal_text(maximum))
    result['reasons']=sorted(reasons)
    result['reason_messages']=[REASON_MESSAGES[reason.split(':',1)[0]].format(asset=reason.partition(':')[2]) for reason in result['reasons']]
    result['realized_by_currency_status']='partial_before_unknown_basis_or_fees' if unknown else 'confirmed_net_native_currency'
    result['history_fills']=len(seen)
    result['state']='partial' if reasons else 'complete'
    if result['equity'] is None and result['realized'] is None and not (fees or realized):result['state']='unavailable'
    return result
