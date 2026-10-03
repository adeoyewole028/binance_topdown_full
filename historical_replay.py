"""Pure causal single-symbol portfolio replay; no transport or execution owner."""
from __future__ import annotations

from bisect import bisect_right
from datetime import datetime, timezone
from decimal import Decimal, ROUND_FLOOR
import hashlib
import json
from typing import Any

from configuration import validated_config
from historical_data import validate_history
from paper_policy import validated_policy, number, decimal_text, financial_arithmetic, fill_terms, entry_quantity
from position_protection import exit_price
from strategy_profiles import evaluate, PROFILES, DURATIONS
from account_performance import weighted_entry, realized_exit


@financial_arithmetic
def replay(dataset: dict[str,Any], *, config: dict[str,Any], policy: dict[str,Any], sizing: str,
           quantity_step: str, namespace: str, config_version: int) -> dict[str,Any]:
    """Decide from completed bars; execute only at the next open event.

    Lot precision is an explicit research assumption, not a claim of historical
    exchange filter availability or live order admissibility.
    """
    config=validated_config(config)
    policy=validated_policy(policy)
    config=validated_config({**config,'paper_policy':policy})
    step=number(quantity_step)
    if (sizing not in {'fixed','risk'} or step<=0 or not namespace.startswith('research:')
            or type(config_version) is not int or config_version<1
            or dataset['profile']!=config['strategy_profile'] or dataset['symbol'].split('/')[1]!=policy['quote']
            or dataset.get('profile_version')!=PROFILES[dataset['profile']].version):
        raise ValueError('Explicit consistent research namespace/profile/configuration/precision/quote required')
    history=validate_history(dataset['bars'],symbol=dataset['symbol'],venue=dataset['venue'],
        profile=dataset['profile'],start=dataset['start'],end=dataset['end'],as_of=dataset['as_of'])
    assumptions={'policy':policy,'sizing':sizing,'quantity_step':decimal_text(step),
        'quantity_precision':'explicit_research_assumption; historical/live exchange filters are not certified',
        'entry':'next_open_after_completed_signal','fill':'deterministic_full',
        'protection':'stop_first_same_bar; adverse_stop_gap; conservative_target',
        'valuation':'closed_bar_and_simulated_fill_events','end':'retain_open_positions; no_forced_liquidation'}
    result: dict[str,Any]={'state':history['state'],'coverage':history['coverage'],'format_version':1,
        'namespace':namespace,'symbol':history['symbol'],'venue':history['venue'],
        'profile':history['profile'],'profile_version':history['profile_version'],
        'config_version':config_version,'config':config,'start':history['start'],'end':history['end'],'as_of':history['as_of'],
        'assumptions':assumptions,'data_sha256':history['data_sha256'],
        'config_sha256':hashlib.sha256(json.dumps(config,sort_keys=True,allow_nan=False).encode()).hexdigest(),
        'fills':None,'positions':None,'metrics':None,'decisions':None,
        'limitations':'OHLC simulation; no profitability guarantee, historical filter certification or live fill fidelity'}
    if history['state']!='complete' or dataset.get('state')=='incomplete':
        result['state']='incomplete'
        result['reason']='incomplete_history'
        return result
    rows=history['bars']
    execution=[r for r in rows['15m'] if r[0]>=history['start']]
    if not execution:
        result.update(state='incomplete',reason='no_complete_execution_interval')
        return result
    selected=PROFILES[history['profile']]
    windows={selected.trend:selected.trend_window,selected.bias:selected.bias_window,'15m':selected.execution_window}
    timestamps={tf:[r[0] for r in values] for tf,values in rows.items()}
    cash=number(policy['initial_cash'])
    quantity=Decimal(0)
    basis=Decimal(0)
    realized=Decimal(0)
    peak=cash
    drawdown=Decimal(0)
    halted=False
    last_entry: int | None=None
    pending: dict[str,Any] | None=None
    fills: list[dict[str,Any]]=[]
    decisions: list[dict[str,Any]]=[]
    equity_curve: list[dict[str,Any]]=[]
    trades: list[Decimal]=[]

    def mark(at: int, price: Decimal) -> None:
        nonlocal peak,drawdown,halted
        equity=cash+quantity*price
        peak=max(peak,equity)
        loss=(peak-equity)/peak
        drawdown=max(drawdown,loss)
        if loss>=number(policy['drawdown_fraction']):halted=True
        equity_curve.append({'at':at,'cash':decimal_text(cash),'exposure':decimal_text(quantity*price),
                             'equity':decimal_text(equity),'drawdown':decimal_text(loss),'entry_halted':halted})

    def decide(at: int, price: Decimal) -> dict[str,Any]:
        available={}
        for tf,values in rows.items():
            index=bisect_right(timestamps[tf],at-DURATIONS[tf]*1000)
            available[tf]=values[max(0,index-windows[tf]):index]
        decision=evaluate(namespace=namespace,venue=history['venue'],symbol=history['symbol'],
            profile=history['profile'],config_version=config_version,config=config,bars=available,
            as_of=datetime.fromtimestamp(at/1000,timezone.utc),
            position={'entry':basis,'qty':quantity,'price':price} if quantity else None)
        decisions.append(decision)
        return decision

    def fill(side: str, amount: Decimal, price: Decimal, at: int, reason: str, signal: str | None) -> None:
        nonlocal cash,quantity,basis,realized,last_entry
        terms=fill_terms(amount,price,side,policy)
        cost,fee=terms['cost'],terms['fee']
        if side=='buy':
            basis=weighted_entry(quantity,basis,amount,cost+fee)
            cash-=cost+fee
            quantity+=amount
            last_entry=at
        else:
            profit=realized_exit(amount,basis,cost-fee)
            realized+=profit
            cash+=cost-fee
            quantity-=amount
            if not quantity:
                trades.append(profit)
                basis=Decimal(0)
        fills.append({'at':at,'side':side,'quantity':decimal_text(amount),'price':decimal_text(terms['price']),
            'quote_cost':decimal_text(cost),'fee':decimal_text(fee),'fee_asset':policy['quote'],
            'reason':reason,'signal_id':signal,'profile':history['profile'],'config_version':config_version})

    opening=number(execution[0][1])
    first=int(execution[0][0])
    decision=decide(first,opening)
    if decision['state']=='ready' and decision['entry']:pending={'side':'buy','signal':decision['signal_id']}
    for bar in execution:
        at=int(bar[0])
        opened,high,low,closed=(number(v) for v in bar[1:5])
        mark(at,opened)
        # Existing protection observes the opening event before queued actions.
        # Later highs/lows cannot override an indicator close already filled here.
        if quantity:
            opening_hit=exit_price(basis*(1-number(config['stop_loss_pct'])),
                basis*(1+number(config['take_profit_pct'])),opened=opened,high=opened,low=opened)
            if opening_hit:
                reason,price=opening_hit
                fill('sell',quantity,price,at,reason,None)
                pending=None
        if pending:
            if pending['side']=='sell' and quantity:
                fill('sell',quantity,opened,at,'indicator_exit',pending['signal'])
            elif pending['side']=='buy' and not quantity and not halted:
                cooldown=max(policy['cooldown_seconds'],config['cooldown'])*1000
                if last_entry is None or at-last_entry>=cooldown:
                    amount=entry_quantity(cash=cash,equity=cash,exposure=Decimal(0),market_price=opened,
                        policy=policy,config=config,mode=sizing)
                    amount=(amount/step).to_integral_value(rounding=ROUND_FLOOR)*step
                    if amount>0:
                        terms=fill_terms(amount,opened,'buy',policy)
                        loss=amount*terms['price']*(number(config['stop_loss_pct'])+2*number(policy['fee_fraction'])+number(policy['slippage_fraction']))
                        if loss<=cash*number(policy['risk_fraction']) and terms['cost']+terms['fee']<=cash:
                            fill('buy',amount,opened,at,'completed_signal',pending['signal'])
            pending=None
            mark(at,opened)
        if quantity:
            hit=exit_price(basis*(1-number(config['stop_loss_pct'])),basis*(1+number(config['take_profit_pct'])),
                opened=opened,high=high,low=low)
            if hit:
                reason,price=hit
                fill('sell',quantity,price,at+900000,reason,None)
        mark(at+900000,closed)
        decision=decide(at+900000,closed)
        if decision['state']=='ready':
            if quantity and decision['exit_reasons']:pending={'side':'sell','signal':decision['signal_id']}
            elif not quantity and decision['entry']:pending={'side':'buy','signal':decision['signal_id']}
    last_price=number(execution[-1][4])
    exposure=quantity*last_price
    equity=cash+exposure
    positions=[] if not quantity else [{'symbol':history['symbol'],'quantity':decimal_text(quantity),
        'entry':decimal_text(basis),'unrealized':decimal_text(quantity*(last_price-basis))}]
    result.update(fills=fills,positions=positions,decisions=decisions,equity_curve=equity_curve,pending=pending,
        metrics={'cash':decimal_text(cash),'exposure':decimal_text(exposure),'equity':decimal_text(equity),
            'realized':decimal_text(realized),'unrealized':decimal_text(quantity*(last_price-basis)),
            'return_fraction':decimal_text((equity-number(policy['initial_cash']))/number(policy['initial_cash'])),
            'drawdown':decimal_text(drawdown),'entry_halted':halted,'completed_trades':len(trades),
            'win_rate':decimal_text(Decimal(sum(t>0 for t in trades))/Decimal(len(trades))) if trades else None,
            'win_rate_denominator':'completed_full_position_exits'})
    return result
