import csv
import math
import json
import argparse
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import pandas as pd
from utils import ema, rsi, atr, is_higher_highs_lows, detect_bullish_engulfing, detect_hammer
import statistics

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

class Backtester:
    def __init__(self, df_15m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame, df_1d: pd.DataFrame, symbol: str, cfg: BacktestConfig):
        self.df15 = df_15m.copy()
        self.df1h = df_1h.copy()
        self.df4h = df_4h.copy()
        self.df1d = df_1d.copy()
        self.symbol = symbol
        self.cfg = cfg
        self.trades: List[Trade] = []
        self.position: Optional[Dict[str, Any]] = None

    def _prep(self):
        f, s = self.cfg.ema_fast, self.cfg.ema_slow
        for df in (self.df15, self.df1h, self.df4h, self.df1d):
            df['ema_fast'] = ema(df['close'], f)
            df['ema_slow'] = ema(df['close'], s)
            if len(df) >= 14:
                df['rsi'] = rsi(df['close'], 14)
        # ATR only on 15m for risk sizing
        self.df15['atr'] = atr(self.df15, self.cfg.atr_period)

    def _daily_trend(self, df):
        above = (df['close'] > df['ema_fast']) & (df['close'] > df['ema_slow'])
        hhll = is_higher_highs_lows(df['close'])
        return above.iloc[-1] and hhll

    def _gate(self, idx15):
        # Map 15m index timestamp to nearest lower timeframe indexes
        ts = self.df15.index[idx15]
        df1d_sub = self.df1d[self.df1d.index <= ts].tail(200)
        df4h_sub = self.df4h[self.df4h.index <= ts].tail(200)
        df1h_sub = self.df1h[self.df1h.index <= ts].tail(200)
        if len(df1d_sub) < 50 or len(df4h_sub) < 50 or len(df1h_sub) < 50:
            return None
        trend1d = self._daily_trend(df1d_sub)
        trend4h = self._daily_trend(df4h_sub)
        last1h = df1h_sub.iloc[-1]
        near_ma = abs(last1h['close'] - last1h['ema_fast']) / max(last1h['close'], 1e-9) <= 0.02
        momentum_ok = last1h['rsi'] > self.cfg.rsi_threshold
        if not (trend1d and trend4h and near_ma and momentum_ok):
            return None
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
                price_above = row['close'] > row['ema_fast']
                rsi_cross_up = (prev.get('rsi', 0) <= self.cfg.rsi_exec_threshold) and (row.get('rsi', 0) > self.cfg.rsi_exec_threshold)
                # Vol regime
                atr_val = row.get('atr')
                atr_ok = True
                if atr_val and row['close'] > 0:
                    atr_ok = (atr_val / row['close']) >= self.cfg.min_atr_rel
                if price_above and rsi_cross_up and atr_ok:
                    # Size using ATR stop
                    atr_stop_pct = None
                    atr_tp_pct = None
                    if atr_val and row['close'] > 0:
                        atr_stop_pct = max(atr_val * self.cfg.atr_stop_mult / row['close'], self.cfg.min_stop_pct)
                        atr_tp_pct = atr_stop_pct * (self.cfg.atr_tp_mult / max(self.cfg.atr_stop_mult, 1e-9))
                    sl = row['close'] * (1 - (atr_stop_pct or self.cfg.min_stop_pct))
                    tp = row['close'] * (1 + (atr_tp_pct or self.cfg.atr_tp_mult * self.cfg.min_stop_pct))
                    risk_unit = row['close'] - sl
                    self.position = {
                        'entry_price': row['close'], 'sl': sl, 'tp': tp, 'qty': 1.0, 'open_index': i,
                        'partial_done': False, 'trail_active': False, 'risk_unit': risk_unit,
                        'partial_target': row['close'] + self.cfg.partial_tp_r_multiple * risk_unit,
                        'atr_at_entry': atr_val
                    }
            else:
                # Manage position
                price = row['close']
                pos = self.position
                # Partial
                if (not pos['partial_done']) and price >= pos['partial_target']:
                    pos['partial_done'] = True
                    # Move stop to breakeven
                    pos['sl'] = max(pos['sl'], pos['entry_price'])
                # Trail activation
                if (not pos['trail_active']) and pos['risk_unit'] > 0 and (price - pos['entry_price']) / pos['risk_unit'] >= self.cfg.trail_activate_r_multiple:
                    pos['trail_active'] = True
                if pos['trail_active']:
                    atr_val = row.get('atr') or pos.get('atr_at_entry')
                    if atr_val and atr_val > 0:
                        new_sl = price - atr_val * self.cfg.trail_atr_mult
                        if new_sl > pos['sl']:
                            pos['sl'] = new_sl
                # Exit conditions
                exit_reason = None
                if price <= pos['sl']:
                    exit_reason = 'stop'
                elif price >= pos['tp']:
                    exit_reason = 'target'
                # Optional EMA fast break
                if exit_reason is None:
                    prev = self.df15.iloc[i-1]
                    if (prev['close'] >= prev['ema_fast']) and (price < row['ema_fast']):
                        exit_reason = 'ema_fast_break'
                if exit_reason:
                    pnl = (price - pos['entry_price']) * pos['qty']
                    r_mult = None
                    if pos['risk_unit'] > 0:
                        r_mult = (price - pos['entry_price']) / pos['risk_unit']
                    self.trades.append(Trade(
                        symbol=self.symbol,
                        entry_time=int(self.df15.index[pos['open_index']].timestamp()),
                        exit_time=int(self.df15.index[i].timestamp()),
                        entry=pos['entry_price'], exit=price, side='long', qty=pos['qty'], pnl=pnl, r_multiple=r_mult, exit_reason=exit_reason
                    ))
                    self.position = None
        # Force close last
        if self.position is not None:
            row = self.df15.iloc[-1]
            price = row['close']
            pos = self.position
            pnl = (price - pos['entry_price']) * pos['qty']
            r_mult = (price - pos['entry_price']) / pos['risk_unit'] if pos['risk_unit']>0 else None
            self.trades.append(Trade(
                symbol=self.symbol,
                entry_time=int(self.df15.index[pos['open_index']].timestamp()),
                exit_time=int(self.df15.index[-1].timestamp()),
                entry=pos['entry_price'], exit=price, side='long', qty=pos['qty'], pnl=pnl, r_multiple=r_mult, exit_reason='eod'
            ))
            self.position = None
        return self.trades

    def metrics(self) -> Dict[str, Any]:
        if not self.trades:
            return { 'trades': 0, 'total_pnl': 0, 'avg_r': 0, 'win_rate': 0, 'profit_factor': 0, 'max_drawdown': 0, 'sharpe': 0 }
        pnl_series = [t.pnl for t in self.trades]
        cum = []
        running = 0
        for p in pnl_series:
            running += p
            cum.append(running)
        peak = cum[0]
        max_dd = 0
        for v in cum:
            if v > peak:
                peak = v
            dd = peak - v
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
        return {
            'trades': len(self.trades),
            'total_pnl': sum(pnl_series),
            'avg_r': avg_r,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_dd,
            'sharpe': sharpe
        }

def load_csv_to_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.set_index('timestamp')
    return df

def run_single(args: argparse.Namespace):
    cfg = BacktestConfig(
        ema_fast=args.ema_fast, ema_slow=args.ema_slow, rsi_threshold=args.rsi_threshold,
        rsi_exec_threshold=args.rsi_exec_threshold, atr_period=args.atr_period,
        atr_stop_mult=args.atr_stop_mult, atr_tp_mult=args.atr_tp_mult, min_stop_pct=args.min_stop_pct,
        partial_tp_r_multiple=args.partial_tp_r_multiple, partial_tp_fraction=args.partial_tp_fraction,
        trail_activate_r_multiple=args.trail_activate_r_multiple, trail_atr_mult=args.trail_atr_mult,
        min_atr_rel=args.min_atr_rel
    )
    df15 = load_csv_to_df(args.data_15m)
    df1h = load_csv_to_df(args.data_1h)
    df4h = load_csv_to_df(args.data_4h)
    df1d = load_csv_to_df(args.data_1d)
    bt = Backtester(df15, df1h, df4h, df1d, args.symbol, cfg)
    trades = bt.run()
    out_rows = [asdict(t) for t in trades]
    if args.out_trades:
        with open(args.out_trades, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=out_rows[0].keys())
            writer.writeheader(); writer.writerows(out_rows)
    # Summary
    m = bt.metrics()
    print(json.dumps(m, indent=2))


def build_arg_parser():
    p = argparse.ArgumentParser(description='Simple strategy backtester')
    p.add_argument('--symbol', required=True)
    p.add_argument('--data-15m', dest='data_15m', required=True)
    p.add_argument('--data-1h', dest='data_1h', required=True)
    p.add_argument('--data-4h', dest='data_4h', required=True)
    p.add_argument('--data-1d', dest='data_1d', required=True)
    p.add_argument('--out-trades', dest='out_trades')
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
    # Sweep parameters (comma-separated lists)
    p.add_argument('--sweep-ema-fast')
    p.add_argument('--sweep-ema-slow')
    p.add_argument('--sweep-atr-stop-mult')
    p.add_argument('--sweep-atr-tp-mult')
    p.add_argument('--sweep-partial-r')
    p.add_argument('--sweep-trail-r')
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
                min_atr_rel=args.min_atr_rel
            )
            df15 = load_csv_to_df(args.data_15m)
            df1h = load_csv_to_df(args.data_1h)
            df4h = load_csv_to_df(args.data_4h)
            df1d = load_csv_to_df(args.data_1d)
            bt = Backtester(df15, df1h, df4h, df1d, args.symbol, cfg)
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
