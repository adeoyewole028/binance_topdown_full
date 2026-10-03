"""Named, causal strategy contracts. No transport, persistence or mutable config."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import pandas as pd
from utils import ema, rsi, is_higher_highs_lows, detect_bullish_engulfing, detect_hammer


@dataclass(frozen=True)
class Profile:
    trend: str
    bias: str
    indicator_entry: bool
    version: int = 1
    bias_window: int = 200
    fixed_bias: Mapping[str,float] | None = None
    trend_window: int = 200
    execution_window: int = 100


PROFILES: Mapping[str, Profile] = MappingProxyType({
    'dashboard_intraday': Profile('4h', '1h', True),
    'daemon_daily': Profile('1d', '4h', False, fixed_bias=MappingProxyType({'near_ema_pct':0.02,'rsi_threshold':30.0}), trend_window=300, execution_window=200),
})
DURATIONS = {'1d': 86400, '4h': 14400, '1h': 3600, '15m': 900}


def trend_check(frame: Any) -> dict[str, Any]:
    close = frame['close']
    above = bool(close.iloc[-1] > ema(close, 21).iloc[-1] and close.iloc[-1] > ema(close, 50).iloc[-1])
    pivots = bool(is_higher_highs_lows(close))
    return {'bullish': above and pivots, 'above21': above, 'hhll': pivots}


def bias_check(frame: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    close = frame['close']
    value = float(rsi(close,14).iloc[-1])
    near = abs(close.iloc[-1]-ema(close,21).iloc[-1])/close.iloc[-1] <= config['near_ema_pct']
    return {'near_ma': bool(near), 'momentum_ok': bool(value > config['rsi_threshold']), 'rsi': value}


def execution_check(frame: Any, config: Mapping[str, Any], *, indicator_entry: bool) -> dict[str, Any]:
    if detect_bullish_engulfing(frame):
        return {'signal': True, 'type': 'bullish_engulfing'}
    if detect_hammer(frame):
        return {'signal': True, 'type': 'hammer'}
    if indicator_entry and len(frame) >= 2:
        close = frame['close']
        values = rsi(close,14)
        if close.iloc[-1] > ema(close,21).iloc[-1] and values.iloc[-2] <= config['rsi_exec_threshold'] < values.iloc[-1]:
            return {'signal': True, 'type': 'ema21_rsi_cross'}
    return {'signal': False}


def exit_reasons(execution: Any, bias: Any, config: Mapping[str, Any], position: Mapping[str, Any],
                 *, indicator_exit: bool) -> list[str]:
    price, entry, quantity = (float(position[key]) for key in ('price','entry','qty'))
    if any(not math.isfinite(value) or value <= 0 for value in (price,entry,quantity)):
        raise ValueError('position_unavailable')
    reasons = []
    if price >= entry*(1+config['take_profit_pct']):
        reasons.append('take_profit')
    if price <= entry*(1-config['stop_loss_pct']):
        reasons.append('stop_loss')
    if indicator_exit:
        close, momentum = execution['close'], rsi(execution['close'],14)
        average = ema(close,21)
        if close.iloc[-2] >= average.iloc[-2] and close.iloc[-1] < average.iloc[-1]:
            reasons.append('ema21_cross_down_15m')
        if momentum.iloc[-2] >= config['rsi_overbought'] > momentum.iloc[-1]:
            reasons.append('rsi_overbought_cross_down_15m')
        close, momentum = bias['close'], rsi(bias['close'],14)
        average = ema(close,21)
        if close.iloc[-2] >= average.iloc[-2] and close.iloc[-1] < average.iloc[-1] and momentum.iloc[-1] < momentum.iloc[-2]:
            reasons.append('ema21_break_1h_rsi_down')
    return reasons


def completed_frame(rows: Sequence[Sequence[float]], timeframe: str, cutoff: float, minimum: int, window: int) -> Any:
    duration = DURATIONS[timeframe]
    completed = []
    previous = None
    for row in rows:
        if not row or isinstance(row[0],bool) or not math.isfinite(float(row[0])):
            raise ValueError('invalid_timestamp')
        stamp = float(row[0])
        # Forming/future values never participate in the past decision.
        if stamp/1000+duration > cutoff:
            continue
        if len(row) != 6 or any(isinstance(value,bool) or not math.isfinite(float(value)) for value in row):
            raise ValueError('invalid_bar')
        stamp, opening, high, low, closing, volume = map(float,row)
        if stamp % (duration*1000):
            raise ValueError('unaligned_history')
        if previous is not None and stamp-previous != duration*1000:
            raise ValueError('gapped_history')
        if min(opening,high,low,closing) <= 0 or volume < 0 or not low <= min(opening,closing) <= max(opening,closing) <= high:
            raise ValueError('invalid_ohlc')
        completed.append([stamp,opening,high,low,closing,volume])
        previous = stamp
    if len(completed) < minimum:
        raise ValueError('insufficient_history')
    if completed[-1][0] != (math.floor(cutoff/duration)*duration-duration)*1000:
        raise ValueError('stale_history')
    return pd.DataFrame(completed[-window:], columns=['timestamp','open','high','low','close','volume'])


def evaluate(*, namespace: str, venue: str, symbol: str, profile: str,
             config_version: int, config: Mapping[str, Any], bars: Mapping[str, Sequence[Sequence[float]]],
             as_of: datetime, position: Mapping[str, Any] | None = None) -> dict[str, Any]:
    selected = PROFILES[profile]
    if as_of.tzinfo is None:
        raise ValueError('Timezone-aware decision time required')
    cutoff = math.floor(as_of.timestamp()/900)*900
    identity = [namespace,venue,profile,selected.version,config_version,symbol,int(cutoff*1000)]
    result: dict[str, Any] = {'state': 'unavailable', 'reason': None, 'entry': False, 'exit_reasons': [],
        'symbol': symbol, 'profile': profile, 'profile_version': selected.version,
        'config_version': config_version, 'trend_timeframe':selected.trend, 'bias_timeframe':selected.bias, 'execution_close': int(cutoff*1000),
        'time': datetime.fromtimestamp(cutoff,timezone.utc).isoformat(),
        'signal_id': hashlib.sha256(json.dumps(identity,separators=(',',':')).encode()).hexdigest()}
    try:
        trend_frame = completed_frame(bars[selected.trend],selected.trend,cutoff,50,selected.trend_window)
        bias_frame = completed_frame(bars[selected.bias],selected.bias,cutoff,21,selected.bias_window)
        execution_frame = completed_frame(bars['15m'],'15m',cutoff,21,selected.execution_window)
        trend = trend_check(trend_frame)
        bias = bias_check(bias_frame,selected.fixed_bias if selected.fixed_bias is not None else config)
        execution = execution_check(execution_frame,config,indicator_entry=selected.indicator_entry)
        exits = exit_reasons(execution_frame,bias_frame.tail(100),config,position,indicator_exit=selected.indicator_entry) if position else []
    except (ValueError,KeyError,TypeError,IndexError) as error:
        result['reason'] = str(error)[:80]
        return result
    result.update(state='ready',trend=trend,bias=bias,execution=execution,
                  exit_reasons=exits, entry=bool(position is None and trend['bullish'] and bias['near_ma'] and bias['momentum_ok'] and execution['signal']))
    return result


def evaluate_snapshot(*, namespace: str, venue: str, config_version: int, config: Mapping[str, Any],
                      cache: Mapping[str, Any], as_of: datetime, symbols: Sequence[str],
                      positions: Sequence[Mapping[str, Any]], rules: Mapping[str, Any],
                      prices: Mapping[str, Any]) -> dict[str, Any]:
    """Shared adapter projection; callers supply all observed position/config data."""
    signals, exits, decisions = [], [], []
    held = {p['symbol']:p for p in positions}
    for symbol in sorted(set(symbols)|set(held)):
        position = held.get(symbol)
        rule = rules.get(symbol,{})
        settings = rule['config'] if position and rule.get('state') == 'bound' else config
        version = rule['config_version'] if position and rule.get('state') == 'bound' else config_version
        selected = settings['strategy_profile']
        contract = PROFILES[selected]
        data = {t:cache.get(f'{venue}:{symbol}:{t}',{}).get('rows',[]) for t in (contract.trend,contract.bias,'15m')}
        context = None
        if position:
            if rule.get('state') != 'bound' or position.get('basis_status') == 'provider_basis_unknown':
                decisions.append({'symbol':symbol,'state':'unavailable','reason':'position_rules_unavailable'})
                continue
            context = {**position,'price':prices.get(symbol,{}).get('last')}
        result = evaluate(namespace=namespace,venue=venue,symbol=symbol,profile=selected,
            config_version=version,config=settings,bars=data,as_of=as_of,position=context)
        decisions.append(result)
        if result['state'] != 'ready':
            continue
        if result['entry'] and symbol in symbols:
            signals.append({**result,'type':result['execution']['type'],
                            'trend4h':result['trend'],'bias1h':result['bias']})
        if result['exit_reasons'] and context:
            price, entry, quantity = (float(context[k]) for k in ('price','entry','qty'))
            exits.append({**result,'reasons':result['exit_reasons'],'price':price,'pnl':(price-entry)*quantity})
    return {'signals':signals,'exit_signals':exits,'strategy_decisions':decisions,
            'strategy_profile':config['strategy_profile'],'strategy_version':config['strategy_version']}


def evaluate_observation(view: Any, symbols: Sequence[str], config: Mapping[str, Any]) -> dict[str, Any]:
    """Adapters read explicit cached fields only; this never calls view transports."""
    return evaluate_snapshot(namespace=view.context['namespace'],venue=view.venue,
        config_version=view.context['config_version'],config=config,cache=view.cache,
        as_of=view.context['as_of'],symbols=symbols,positions=view.context['positions'],
        rules=view.context['position_rules'],prices=view.prices)
