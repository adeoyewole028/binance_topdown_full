"""Backtesting engine for multi-timeframe strategy.

Cleanup additions:
- Module-level constants for magic numbers.
- Docstrings for public methods.
"""

import csv
import math
import json
import argparse
import os
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import pandas as pd
from utils import ema, rsi, atr, is_higher_highs_lows, detect_bullish_engulfing, detect_hammer, detect_bearish_engulfing, detect_shooting_star
import statistics
try:
    import ccxt  # type: ignore
except ImportError:  # pragma: no cover
    ccxt = None

LOOKBACK_GATING = 200  # candles of each higher TF to consider for gating slice
MIN_HISTORY = 50       # minimum rows required per higher timeframe slice
EPS = 1e-9             # numerical stability epsilon


@dataclass
class Trade:
    symbol: str
    entry_time: int
    exit_time: int
    entry: float
    exit: float
    side: str
    qty: float
    pnl: float
    r_multiple: Optional[float]
    exit_reason: str
    equity_after: Optional[float] = None

@dataclass
class BacktestConfig:
    ema_fast: int = 21
    ema_slow: int = 50
    rsi_threshold: float = 30
    rsi_exec_threshold: float = 45
    atr_period: int = 14
    atr_stop_mult: float = 1.5
    atr_tp_mult: float = 2.5
    min_stop_pct: float = 0.004
    partial_tp_r_multiple: float = 1.0
    partial_tp_fraction: float = 0.5
    trail_activate_r_multiple: float = 1.5
    trail_atr_mult: float = 1.0
    min_atr_rel: float = 0.001
    near_ema_pct: float = 0.02  # proximity threshold for 1h price to fast EMA
    # Extensions
    enable_shorts: bool = True
    base_capital: float = 1000.0
    risk_per_trade_pct: float = 0.01
    adaptive_near_multiplier: float = 2.0
    atr_compression_threshold: float = 0.0008
    trend_tp_boost: float = 0.5  # extra TP range multiplier when higher TF trends align
    use_bullish_patterns: bool = True
    use_bearish_patterns: bool = True

    def validate(self):
        """Validate configuration values; raise ValueError on invalid settings."""
        if self.ema_fast <= 0 or self.ema_slow <= 0:
            raise ValueError('EMA periods must be positive')
        if self.ema_fast >= self.ema_slow:
            raise ValueError('ema_fast should be < ema_slow for trend structure')
        if not (0 < self.risk_per_trade_pct <= 0.2):
            raise ValueError('risk_per_trade_pct must be in (0,0.2]')
        if self.partial_tp_fraction <= 0 or self.partial_tp_fraction > 1:
            raise ValueError('partial_tp_fraction must be in (0,1]')
        if self.trail_atr_mult <= 0:
            raise ValueError('trail_atr_mult must be > 0')
        if self.near_ema_pct <= 0:
            raise ValueError('near_ema_pct must be > 0')

class Backtester:
    def __init__(self, df_15m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame, df_1d: pd.DataFrame, symbol: str, cfg: BacktestConfig):
        """Initialize backtester with prepared timeframe DataFrames.

        DataFrames must have datetime index (UTC) & OHLC columns.
        """
        self.df15 = df_15m.copy()
        self.df1h = df_1h.copy()
        self.df4h = df_4h.copy()
        self.df1d = df_1d.copy()
        self.symbol = symbol
        self.cfg = cfg
        self.trades: List[Trade] = []
        self.position: Optional[Dict[str, Any]] = None
        self.equity = cfg.base_capital
        self.equity_curve: List[Dict[str, Any]] = []
        # Gate diagnostics
        self.gate_stats = {
            'checks': 0,
            'passes': 0,
            'fail_insufficient_history': 0,
            'fail_trend1d': 0,
            'fail_trend4h': 0,
            'fail_near_ma': 0,
            'fail_momentum': 0,
        }
        self.verbose = False

    def _prep(self):
        """Compute indicator columns (EMA, RSI, ATR)."""
        f, s = self.cfg.ema_fast, self.cfg.ema_slow
        for df in (self.df15, self.df1h, self.df4h, self.df1d):
            df['ema_fast'] = ema(df['close'], f)
            df['ema_slow'] = ema(df['close'], s)
            if len(df) >= 14:
                df['rsi'] = rsi(df['close'], 14)
        # ATR only on 15m for risk sizing
        self.df15['atr'] = atr(self.df15, self.cfg.atr_period)

    def _daily_trend(self, df):
        """Return True if price above both EMAs and higher highs / higher lows structure."""
        above = (df['close'] > df['ema_fast']) & (df['close'] > df['ema_slow'])
        hhll = is_higher_highs_lows(df['close'])
        return above.iloc[-1] and hhll

    def _gate(self, idx15):
        """Higher timeframe alignment & momentum gate with diagnostics.

        Returns dict if gate passes else None while updating gate_stats.
        """
        self.gate_stats['checks'] += 1
        ts = self.df15.index[idx15]
        df1d_sub = self.df1d[self.df1d.index <= ts].tail(LOOKBACK_GATING)
        df4h_sub = self.df4h[self.df4h.index <= ts].tail(LOOKBACK_GATING)
        df1h_sub = self.df1h[self.df1h.index <= ts].tail(LOOKBACK_GATING)
        if len(df1d_sub) < MIN_HISTORY or len(df4h_sub) < MIN_HISTORY or len(df1h_sub) < MIN_HISTORY:
            self.gate_stats['fail_insufficient_history'] += 1
            return None
        trend1d = self._daily_trend(df1d_sub)
        if not trend1d:
            self.gate_stats['fail_trend1d'] += 1
        trend4h = self._daily_trend(df4h_sub)
        if trend1d and (not trend4h):
            self.gate_stats['fail_trend4h'] += 1
        last1h = df1h_sub.iloc[-1]
        near_thresh = self.cfg.near_ema_pct
        last_atr = self.df15.iloc[idx15].get('atr')
        if last_atr and self.df15.iloc[idx15]['close'] > 0:
            rel = last_atr / self.df15.iloc[idx15]['close']
            if rel < self.cfg.atr_compression_threshold:
                near_thresh *= self.cfg.adaptive_near_multiplier
        near_ma = abs(last1h['close'] - last1h['ema_fast']) / max(last1h['close'], EPS) <= near_thresh
        if trend1d and trend4h and (not near_ma):
            self.gate_stats['fail_near_ma'] += 1
        momentum_ok = last1h['rsi'] > self.cfg.rsi_threshold
        if trend1d and trend4h and near_ma and (not momentum_ok):
            self.gate_stats['fail_momentum'] += 1
        passed = trend1d and trend4h and near_ma and momentum_ok
        if not passed:
            return None
        self.gate_stats['passes'] += 1
        return {'trend1d': trend1d, 'trend4h': trend4h, 'last1h': last1h}

    def run(self):
        self._prep()
        for i in range(200, len(self.df15)):
            row = self.df15.iloc[i]
            gate = self._gate(i)
            if self.position is None:
                # Entry logic similar to live
                if not gate:
                    continue
                prev = self.df15.iloc[i-1]
                atr_val = row.get('atr')
                atr_ok = True
                if atr_val and row['close'] > 0:
                    atr_ok = (atr_val / row['close']) >= self.cfg.min_atr_rel
                # Pattern confirmation
                recent_slice = self.df15.iloc[i-5:i]
                bull_pattern = self.cfg.use_bullish_patterns and (detect_bullish_engulfing(recent_slice) or detect_hammer(recent_slice))
                bear_pattern = self.cfg.use_bearish_patterns and (detect_bearish_engulfing(recent_slice) or detect_shooting_star(recent_slice))
                # Long
                price_above = row['close'] > row['ema_fast']
                rsi_cross_up = (prev.get('rsi', 0) <= self.cfg.rsi_exec_threshold) and (row.get('rsi', 0) > self.cfg.rsi_exec_threshold)
                long_ok = price_above and rsi_cross_up and atr_ok and (bull_pattern or not self.cfg.use_bullish_patterns)
                # Short mirror
                price_below = row['close'] < row['ema_fast']
                rsi_cross_down = (prev.get('rsi', 100) >= (100 - self.cfg.rsi_exec_threshold)) and (row.get('rsi', 100) < (100 - self.cfg.rsi_exec_threshold))
                short_ok = self.cfg.enable_shorts and price_below and rsi_cross_down and atr_ok and (bear_pattern or not self.cfg.use_bearish_patterns)
                direction = None
                if long_ok:
                    direction = 'long'
                elif short_ok:
                    direction = 'short'
                if direction:
                    atr_stop_pct = None
                    atr_tp_pct = None
                    if atr_val and row['close'] > 0:
                        atr_stop_pct = max(atr_val * self.cfg.atr_stop_mult / row['close'], self.cfg.min_stop_pct)
                        atr_tp_pct = atr_stop_pct * (self.cfg.atr_tp_mult / max(self.cfg.atr_stop_mult, 1e-9))
                    if direction == 'long':
                        sl = row['close'] * (1 - (atr_stop_pct or self.cfg.min_stop_pct))
                        tp_base = row['close'] * (1 + (atr_tp_pct or self.cfg.atr_tp_mult * self.cfg.min_stop_pct))
                        risk_unit = row['close'] - sl
                        if gate['trend1d'] and gate['trend4h']:
                            tp = row['close'] + (tp_base - row['close']) * (1 + self.cfg.trend_tp_boost)
                        else:
                            tp = tp_base
                    else:
                        sl = row['close'] * (1 + (atr_stop_pct or self.cfg.min_stop_pct))
                        tp_base = row['close'] * (1 - (atr_tp_pct or self.cfg.atr_tp_mult * self.cfg.min_stop_pct))
                        risk_unit = sl - row['close']
                        if gate['trend1d'] and gate['trend4h']:
                            tp = row['close'] - (row['close'] - tp_base) * (1 + self.cfg.trend_tp_boost)
                        else:
                            tp = tp_base
                    qty = 1.0
                    if risk_unit > 0:
                        risk_amount = self.equity * self.cfg.risk_per_trade_pct
                        qty = risk_amount / risk_unit
                    self.position = {
                        'side': direction,
                        'entry_price': row['close'], 'sl': sl, 'tp': tp, 'qty': qty, 'open_index': i,
                        'partial_done': False, 'trail_active': False, 'risk_unit': risk_unit,
                        'partial_target': (row['close'] + self.cfg.partial_tp_r_multiple * risk_unit) if direction=='long' else (row['close'] - self.cfg.partial_tp_r_multiple * risk_unit),
                        'atr_at_entry': atr_val
                    }
            else:
                # Manage position
                price = row['close']
                pos = self.position
                # Partial
                if pos['side']=='long' and (not pos['partial_done']) and price >= pos['partial_target']:
                    pos['partial_done'] = True
                    pos['sl'] = max(pos['sl'], pos['entry_price'])
                if pos['side']=='short' and (not pos['partial_done']) and price <= pos['partial_target']:
                    pos['partial_done'] = True
                    pos['sl'] = min(pos['sl'], pos['entry_price'])
                # Trail activation
                if pos['side']=='long' and (not pos['trail_active']) and pos['risk_unit'] > 0 and (price - pos['entry_price']) / pos['risk_unit'] >= self.cfg.trail_activate_r_multiple:
                    pos['trail_active'] = True
                if pos['side']=='short' and (not pos['trail_active']) and pos['risk_unit'] > 0 and (pos['entry_price'] - price) / pos['risk_unit'] >= self.cfg.trail_activate_r_multiple:
                    pos['trail_active'] = True
                if pos['trail_active']:
                    atr_val2 = row.get('atr') or pos.get('atr_at_entry')
                    if atr_val2 and atr_val2 > 0:
                        if pos['side']=='long':
                            new_sl = price - atr_val2 * self.cfg.trail_atr_mult
                            if new_sl > pos['sl']:
                                pos['sl'] = new_sl
                        else:
                            new_sl = price + atr_val2 * self.cfg.trail_atr_mult
                            if new_sl < pos['sl']:
                                pos['sl'] = new_sl
                # Exit conditions
                exit_reason = None
                if pos['side']=='long':
                    if price <= pos['sl']:
                        exit_reason = 'stop'
                    elif price >= pos['tp']:
                        exit_reason = 'target'
                else:
                    if price >= pos['sl']:
                        exit_reason = 'stop'
                    elif price <= pos['tp']:
                        exit_reason = 'target'
                # Optional EMA fast break
                if exit_reason is None:
                    prev = self.df15.iloc[i-1]
                    if pos['side']=='long':
                        if (prev['close'] >= prev['ema_fast']) and (price < row['ema_fast']):
                            exit_reason = 'ema_fast_break'
                    else:
                        if (prev['close'] <= prev['ema_fast']) and (price > row['ema_fast']):
                            exit_reason = 'ema_fast_break'
                if exit_reason:
                    pnl = (price - pos['entry_price']) * pos['qty'] if pos['side']=='long' else (pos['entry_price'] - price) * pos['qty']
                    r_mult = None
                    if pos['risk_unit'] > 0:
                        if pos['side']=='long':
                            r_mult = (price - pos['entry_price']) / pos['risk_unit']
                        else:
                            r_mult = (pos['entry_price'] - price) / pos['risk_unit']
                    self.equity += pnl
                    self.trades.append(Trade(
                        symbol=self.symbol,
                        entry_time=int(self.df15.index[pos['open_index']].timestamp()),
                        exit_time=int(self.df15.index[i].timestamp()),
                        entry=pos['entry_price'], exit=price, side=pos['side'], qty=pos['qty'], pnl=pnl, r_multiple=r_mult, exit_reason=exit_reason, equity_after=self.equity
                    ))
                    self.position = None
            # Record equity point each iteration
            self.equity_curve.append({'timestamp': int(self.df15.index[i].timestamp()), 'equity': self.equity})
        # Force close last
        if self.position is not None:
            row = self.df15.iloc[-1]
            price = row['close']
            pos = self.position
            if pos['side']=='long':
                pnl = (price - pos['entry_price']) * pos['qty']
                r_mult = (price - pos['entry_price']) / pos['risk_unit'] if pos['risk_unit']>0 else None
            else:
                pnl = (pos['entry_price'] - price) * pos['qty']
                r_mult = (pos['entry_price'] - price) / pos['risk_unit'] if pos['risk_unit']>0 else None
            self.equity += pnl
            self.trades.append(Trade(
                symbol=self.symbol,
                entry_time=int(self.df15.index[pos['open_index']].timestamp()),
                exit_time=int(self.df15.index[-1].timestamp()),
                entry=pos['entry_price'], exit=price, side=pos['side'], qty=pos['qty'], pnl=pnl, r_multiple=r_mult, exit_reason='eod', equity_after=self.equity
            ))
            self.position = None
        return self.trades

    def metrics(self) -> Dict[str, Any]:
        if not self.trades:
            return { 'trades': 0, 'total_pnl': 0, 'avg_r': 0, 'win_rate': 0, 'profit_factor': 0, 'max_drawdown': 0, 'sharpe': 0, 'median_r': 0, 'best_r': 0, 'worst_r': 0, 'expectancy_r': 0 }
        pnl_series = [t.pnl for t in self.trades]
        # Drawdown from equity curve (more robust with variable sizing)
        max_dd = 0.0
        if self.equity_curve:
            peak = self.equity_curve[0]['equity']
            for pt in self.equity_curve:
                eq = pt['equity']
                if eq > peak:
                    peak = eq
                dd = peak - eq
                if dd > max_dd:
                    max_dd = dd
        wins = [p for p in pnl_series if p > 0]
        losses = [abs(p) for p in pnl_series if p < 0]
        profit_factor = (sum(wins)/sum(losses)) if wins and losses else (float('inf') if wins and not losses else 0)
        rs = [t.r_multiple for t in self.trades if t.r_multiple is not None]
        avg_r = sum(rs)/len(rs) if rs else 0
        win_rate = (len(wins)/len(pnl_series))*100
        # Simple Sharpe: mean(daily PnL)/std(PnL) * sqrt(n) with n = trades (proxy)
        sharpe = 0
        if len(pnl_series) > 1:
            try:
                sharpe = (statistics.mean(pnl_series) / (statistics.pstdev(pnl_series) or 1e-9)) * math.sqrt(len(pnl_series))
            except Exception:
                sharpe = 0
        median_r = statistics.median(rs) if rs else 0
        best_r = max(rs) if rs else 0
        worst_r = min(rs) if rs else 0
        # Expectancy in R
        avg_win_r = statistics.mean([t.r_multiple for t in self.trades if t.r_multiple and t.r_multiple > 0]) if any(t.r_multiple and t.r_multiple>0 for t in self.trades) else 0
        avg_loss_r = statistics.mean([t.r_multiple for t in self.trades if t.r_multiple is not None and t.r_multiple < 0]) if any(t.r_multiple is not None and t.r_multiple<0 for t in self.trades) else 0
        expectancy_r = (win_rate/100)*avg_win_r + (1-win_rate/100)*avg_loss_r
        return {
            'trades': len(self.trades),
            'total_pnl': sum(pnl_series),
            'avg_r': avg_r,
            'median_r': median_r,
            'best_r': best_r,
            'worst_r': worst_r,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_dd,
            'sharpe': sharpe,
            'expectancy_r': expectancy_r
        }

def load_csv_to_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.set_index('timestamp')
    return df


def fetch_ohlcv_dataframe(exchange, symbol: str, timeframe: str, since_ms: int) -> pd.DataFrame:
    """Fetch OHLCV data (all available from since_ms to now) and return as DataFrame."""
    all_rows = []
    limit = 1000
    since = since_ms
    now_ms = int(time.time() * 1000)
    while True:
        try:
            batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        except Exception as e:  # pragma: no cover - network
            print(f"Fetch error {timeframe}: {e}")
            break
        if not batch:
            break
        all_rows.extend(batch)
        if len(batch) < limit:
            break
        since = batch[-1][0] + 1
        if since >= now_ms:
            break
        # Safety: avoid excessively long loops
        if len(all_rows) > 200000:
            break
    if not all_rows:
        return pd.DataFrame(columns=['timestamp','open','high','low','close','volume']).set_index('timestamp')
    df = pd.DataFrame(all_rows, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.set_index('timestamp')
    return df


def ensure_data(args: argparse.Namespace):
    """Ensure data CSVs exist; optionally auto-fetch missing ones if --auto-fetch provided."""
    required = {
        'data_15m': '15m',
        'data_1h': '1h',
        'data_4h': '4h',
        'data_1d': '1d'
    }
    missing = [k for k in required if (not os.path.exists(getattr(args, k))) or getattr(args, 'force_fetch', False)]
    if not missing:
        return
    if not getattr(args, 'auto_fetch', False):
        raise FileNotFoundError(f"Missing data files: {missing}. Provide CSVs or use --auto-fetch.")
    if ccxt is None:
        raise RuntimeError('ccxt not installed; cannot auto-fetch. pip install ccxt or create files manually.')
    exchange_id = getattr(args, 'exchange', 'binance')
    ex_class = getattr(ccxt, exchange_id, None)
    if ex_class is None:
        raise RuntimeError(f'Exchange {exchange_id} not found in ccxt.')
    exchange = ex_class({'enableRateLimit': True})
    since_days = getattr(args, 'since_days', 180)
    since_ms = int(time.time() * 1000) - since_days * 86400 * 1000
    print(f"Auto-fetching missing data since {since_days} days ago ({missing}) ...")
    for attr in missing:
        tf = required[attr]
        out_path = getattr(args, attr)
        df = fetch_ohlcv_dataframe(exchange, args.symbol, tf, since_ms)
        if df.empty:
            raise RuntimeError(f"Fetched no data for {args.symbol} {tf}")
        df.reset_index().to_csv(out_path, index=False)
        print(f"Saved {len(df)} rows to {out_path}")


def run_single(args: argparse.Namespace):
    ensure_data(args)
    cfg = BacktestConfig(
        ema_fast=args.ema_fast, ema_slow=args.ema_slow, rsi_threshold=args.rsi_threshold,
        rsi_exec_threshold=args.rsi_exec_threshold, atr_period=args.atr_period,
        atr_stop_mult=args.atr_stop_mult, atr_tp_mult=args.atr_tp_mult, min_stop_pct=args.min_stop_pct,
        partial_tp_r_multiple=args.partial_tp_r_multiple, partial_tp_fraction=args.partial_tp_fraction,
        trail_activate_r_multiple=args.trail_activate_r_multiple, trail_atr_mult=args.trail_atr_mult,
    min_atr_rel=args.min_atr_rel, near_ema_pct=args.near_ema_pct,
    enable_shorts=not args.disable_shorts,
    base_capital=args.base_capital, risk_per_trade_pct=args.risk_per_trade_pct,
    adaptive_near_multiplier=args.adaptive_near_multiplier,
    atr_compression_threshold=args.atr_compression_threshold,
    trend_tp_boost=args.trend_tp_boost,
    use_bullish_patterns=not args.disable_bullish_patterns,
    use_bearish_patterns=not args.disable_bearish_patterns
    )
    cfg.validate()
    df15 = load_csv_to_df(args.data_15m)
    df1h = load_csv_to_df(args.data_1h)
    df4h = load_csv_to_df(args.data_4h)
    df1d = load_csv_to_df(args.data_1d)
    bt = Backtester(df15, df1h, df4h, df1d, args.symbol, cfg)
    bt.verbose = args.verbose
    trades = bt.run()
    out_rows = [asdict(t) for t in trades]
    if args.out_trades:
        with open(args.out_trades, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=out_rows[0].keys())
            writer.writeheader(); writer.writerows(out_rows)
    if getattr(args, 'equity_out', None) and bt.equity_curve:
        with open(args.equity_out, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=['timestamp','equity'])
            w.writeheader(); w.writerows(bt.equity_curve)
    # Summary
    m = bt.metrics()
    summary = {'metrics': m, 'gate_stats': bt.gate_stats}
    print(json.dumps(summary if args.verbose else m, indent=2))
    if getattr(args, 'json_out', None):
        with open(args.json_out, 'w', encoding='utf-8') as f:
            json.dump({'config': asdict(cfg), 'metrics': m, 'gate_stats': bt.gate_stats}, f, indent=2)


def build_arg_parser():
    p = argparse.ArgumentParser(description='Simple strategy backtester')
    p.add_argument('--symbol', required=True)
    p.add_argument('--data-15m', dest='data_15m', required=True)
    p.add_argument('--data-1h', dest='data_1h', required=True)
    p.add_argument('--data-4h', dest='data_4h', required=True)
    p.add_argument('--data-1d', dest='data_1d', required=True)
    p.add_argument('--out-trades', dest='out_trades')
    p.add_argument('--equity-out', dest='equity_out', help='CSV file to write equity curve for single run')
    p.add_argument('--json-out', dest='json_out', help='Write single-run metrics + gate stats JSON file')
    p.add_argument('--verbose', action='store_true', help='Print extended diagnostics (gate stats)')
    # Auto-fetch options
    p.add_argument('--auto-fetch', action='store_true', help='Automatically fetch missing timeframe CSVs via ccxt')
    p.add_argument('--exchange', default='binance', help='Exchange id for ccxt (default: binance)')
    p.add_argument('--since-days', dest='since_days', type=int, default=180, help='Days of history to fetch for auto-fetch (default 180)')
    p.add_argument('--force-fetch', action='store_true', help='Re-fetch and overwrite existing CSVs')
    # Params
    p.add_argument('--ema-fast', type=int, default=21)
    p.add_argument('--ema-slow', type=int, default=50)
    p.add_argument('--rsi-threshold', type=float, default=30)
    p.add_argument('--rsi-exec-threshold', type=float, default=45)
    p.add_argument('--atr-period', type=int, default=14)
    p.add_argument('--atr-stop-mult', type=float, default=1.5)
    p.add_argument('--atr-tp-mult', type=float, default=2.5)
    p.add_argument('--min-stop-pct', type=float, default=0.004)
    p.add_argument('--partial-tp-r-multiple', type=float, default=1.0)
    p.add_argument('--partial-tp-fraction', type=float, default=0.5)
    p.add_argument('--trail-activate-r-multiple', type=float, default=1.5)
    p.add_argument('--trail-atr-mult', type=float, default=1.0)
    p.add_argument('--min-atr-rel', type=float, default=0.001)
    p.add_argument('--near-ema-pct', type=float, default=0.02, help='Proximity threshold (fraction) for 1h price vs fast EMA (entry gate)')
    # Extended behavior
    p.add_argument('--disable-shorts', action='store_true')
    p.add_argument('--base-capital', type=float, default=1000.0)
    p.add_argument('--risk-per-trade-pct', type=float, default=0.01)
    p.add_argument('--adaptive-near-multiplier', type=float, default=2.0)
    p.add_argument('--atr-compression-threshold', type=float, default=0.0008)
    p.add_argument('--trend-tp-boost', type=float, default=0.5)
    p.add_argument('--disable-bullish-patterns', action='store_true')
    p.add_argument('--disable-bearish-patterns', action='store_true')
    # Sweep parameters (comma-separated lists)
    p.add_argument('--sweep-ema-fast')
    p.add_argument('--sweep-ema-slow')
    p.add_argument('--sweep-atr-stop-mult')
    p.add_argument('--sweep-atr-tp-mult')
    p.add_argument('--sweep-partial-r')
    p.add_argument('--sweep-trail-r')
    p.add_argument('--sweep-near-ema-pct')
    p.add_argument('--sweep-risk-per-trade-pct')
    p.add_argument('--sweep-trend-tp-boost')
    p.add_argument('--sweep-out', help='CSV file to write sweep summary')
    return p

if __name__ == '__main__':
    parser = build_arg_parser()
    args = parser.parse_args()
    sweep_fields = [
        ('ema_fast','sweep_ema_fast', int),
        ('ema_slow','sweep_ema_slow', int),
        ('atr_stop_mult','sweep_atr_stop_mult', float),
        ('atr_tp_mult','sweep_atr_tp_mult', float),
        ('partial_tp_r_multiple','sweep_partial_r', float),
        ('trail_activate_r_multiple','sweep_trail_r', float),
    ('near_ema_pct','sweep_near_ema_pct', float),
    ('risk_per_trade_pct','sweep_risk_per_trade_pct', float),
    ('trend_tp_boost','sweep_trend_tp_boost', float),
    ]
    sweep_lists = {}
    for base, argname, caster in sweep_fields:
        val = getattr(args, argname, None)
        if val:
            try:
                parts = [caster(x.strip()) for x in val.split(',') if x.strip()]
                if parts:
                    sweep_lists[base] = parts
            except Exception:
                pass
    if not sweep_lists:
        run_single(args)
    else:
        # Build cartesian product
        import itertools
        keys = list(sweep_lists.keys())
        combos = list(itertools.product(*[sweep_lists[k] for k in keys]))
        results = []
        # Cache dataframes once
        base_df15 = load_csv_to_df(args.data_15m)
        base_df1h = load_csv_to_df(args.data_1h)
        base_df4h = load_csv_to_df(args.data_4h)
        base_df1d = load_csv_to_df(args.data_1d)
        for combo in combos:
            # Override args temporarily
            local_kwargs = {k:v for k,v in zip(keys, combo)}
            for k,v in local_kwargs.items():
                setattr(args, k, v)
            cfg = BacktestConfig(
                ema_fast=args.ema_fast, ema_slow=args.ema_slow, rsi_threshold=args.rsi_threshold,
                rsi_exec_threshold=args.rsi_exec_threshold, atr_period=args.atr_period,
                atr_stop_mult=args.atr_stop_mult, atr_tp_mult=args.atr_tp_mult, min_stop_pct=args.min_stop_pct,
                partial_tp_r_multiple=args.partial_tp_r_multiple, partial_tp_fraction=args.partial_tp_fraction,
                trail_activate_r_multiple=args.trail_activate_r_multiple, trail_atr_mult=args.trail_atr_mult,
                min_atr_rel=args.min_atr_rel, near_ema_pct=args.near_ema_pct,
                enable_shorts=not args.disable_shorts,
                base_capital=args.base_capital, risk_per_trade_pct=args.risk_per_trade_pct,
                adaptive_near_multiplier=args.adaptive_near_multiplier,
                atr_compression_threshold=args.atr_compression_threshold,
                trend_tp_boost=args.trend_tp_boost,
                use_bullish_patterns=not args.disable_bullish_patterns,
                use_bearish_patterns=not args.disable_bearish_patterns
            )
            try:
                cfg.validate()
            except Exception as e:
                print(f"Config validation failed for combo {local_kwargs}: {e}")
                continue
            bt = Backtester(base_df15, base_df1h, base_df4h, base_df1d, args.symbol, cfg)
            trades = bt.run()
            m = bt.metrics()
            rec = dict(m)
            for k,v in local_kwargs.items():
                rec[k] = v
            results.append(rec)
            print(json.dumps({'combo': local_kwargs, 'summary': rec}, indent=2))
        if args.sweep_out and results:
            fieldnames = sorted({fn for r in results for fn in r.keys()})
            with open(args.sweep_out, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader(); w.writerows(results)
        # Print best by avg_r then by total_pnl
        if results:
            best = sorted(results, key=lambda r: (r['avg_r'], r['total_pnl']), reverse=True)[:5]
            print('Top 5 combos:')
            for b in best:
                print(b)
