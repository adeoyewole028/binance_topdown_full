"""Multi-symbol batch backtesting utility.

Uses the existing Backtester/BacktestConfig from backtest.py to run the same
strategy parameters across multiple symbols, fetching OHLCV data (via ccxt)
for required timeframes (15m,1h,4h,1d) when not provided on disk.

Outputs per-symbol metrics plus an aggregate metrics block computed by
concatenating all trades and re-calculating distribution stats.

Initial version: focuses on simplicity (sequential fetch/backtest). Future
improvements could parallelize data retrieval and support walk-forward.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, Any, List

import pandas as pd

try:  # pragma: no cover - runtime network dependency
    import ccxt  # type: ignore
except ImportError:  # pragma: no cover
    ccxt = None

from backtest import (
    BacktestConfig,
    Backtester,
    fetch_ohlcv_dataframe,
)


TIMEFRAMES = ["15m", "1h", "4h", "1d"]


def fetch_timeframes(exchange, symbol: str, since_ms: int) -> Dict[str, pd.DataFrame]:
    """Fetch all required timeframes for a symbol.

    Returns mapping timeframe -> DataFrame (indexed by timestamp).
    """
    data = {}
    for tf in TIMEFRAMES:
        df = fetch_ohlcv_dataframe(exchange, symbol, tf, since_ms)
        if df.empty:
            raise RuntimeError(f"Fetched no data for {symbol} {tf}")
        data[tf] = df
    return data


def aggregate_metrics(all_trades: List[Dict[str, Any]], per_symbol_equity: Dict[str, List[Dict[str, Any]]], base_capital: float) -> Dict[str, Any]:
    """Compute aggregate metrics across all trades from multiple symbols.

    Approach: trades are sorted by exit_time to build a synthetic equity curve
    starting with base_capital * num_symbols. This is an approximation because
    simultaneous trades across symbols are simply ordered by close time.
    """
    import math, statistics
    if not all_trades:
        return { 'trades': 0, 'total_pnl': 0, 'avg_r': 0, 'win_rate': 0, 'profit_factor': 0, 'max_drawdown': 0, 'sharpe': 0, 'median_r': 0, 'best_r': 0, 'worst_r': 0, 'expectancy_r': 0 }
    # Sort by exit (close) time
    trades_sorted = sorted(all_trades, key=lambda t: t['exit_time'])
    starting_equity = base_capital * len(per_symbol_equity)
    equity = starting_equity
    equity_curve = []
    for t in trades_sorted:
        equity += t['pnl']
        equity_curve.append({'timestamp': t['exit_time'], 'equity': equity})
    # Drawdown
    peak = equity_curve[0]['equity']
    max_dd = 0.0
    for pt in equity_curve:
        eq = pt['equity']
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd
    pnls = [t['pnl'] for t in trades_sorted]
    wins = [p for p in pnls if p > 0]
    losses = [abs(p) for p in pnls if p < 0]
    profit_factor = (sum(wins)/sum(losses)) if wins and losses else (float('inf') if wins and not losses else 0)
    rs = [t['r_multiple'] for t in trades_sorted if t['r_multiple'] is not None]
    avg_r = sum(rs)/len(rs) if rs else 0
    win_rate = (len(wins)/len(pnls))*100 if pnls else 0
    sharpe = 0
    if len(pnls) > 1:
        try:
            import math, statistics
            sharpe = (statistics.mean(pnls) / (statistics.pstdev(pnls) or 1e-9)) * math.sqrt(len(pnls))
        except Exception:
            sharpe = 0
    import statistics as stats
    median_r = stats.median(rs) if rs else 0
    best_r = max(rs) if rs else 0
    worst_r = min(rs) if rs else 0
    avg_win_r = stats.mean([r for r in rs if r is not None and r > 0]) if any(r for r in rs if r is not None and r > 0) else 0
    avg_loss_r = stats.mean([r for r in rs if r is not None and r < 0]) if any(r for r in rs if r is not None and r < 0) else 0
    expectancy_r = (win_rate/100)*avg_win_r + (1-win_rate/100)*avg_loss_r
    return {
        'trades': len(trades_sorted),
        'total_pnl': sum(pnls),
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


def run_multi(args: argparse.Namespace):
    if ccxt is None:
        raise RuntimeError("ccxt not installed; install with 'pip install ccxt' for multi backtest auto-fetch")
    exchange_id = args.exchange
    ex_class = getattr(ccxt, exchange_id, None)
    if ex_class is None:
        raise RuntimeError(f"Exchange {exchange_id} not found in ccxt")
    exchange = ex_class({'enableRateLimit': True})
    since_ms = int(time.time() * 1000) - args.since_days * 86400 * 1000

    symbols = [s.strip() for s in args.symbols.split(',') if s.strip()]
    if not symbols:
        raise ValueError('--symbols produced empty list')

    # Build config (shared across symbols)
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

    per_symbol_metrics: Dict[str, Any] = {}
    per_symbol_gate_stats: Dict[str, Any] = {}
    all_trades: List[Dict[str, Any]] = []
    per_symbol_equity: Dict[str, List[Dict[str, Any]]] = {}

    for sym in symbols:
        print(f"Running backtest for {sym} ...")
        data = fetch_timeframes(exchange, sym, since_ms)
        bt = Backtester(data['15m'], data['1h'], data['4h'], data['1d'], sym, cfg)
        trades = bt.run()
        m = bt.metrics()
        per_symbol_metrics[sym] = m
        per_symbol_gate_stats[sym] = bt.gate_stats
        # Trades to serializable dicts
        for t in trades:
            all_trades.append({
                'symbol': t.symbol,
                'entry_time': t.entry_time,
                'exit_time': t.exit_time,
                'entry': t.entry,
                'exit': t.exit,
                'side': t.side,
                'qty': t.qty,
                'pnl': t.pnl,
                'r_multiple': t.r_multiple,
                'exit_reason': t.exit_reason,
            })
        per_symbol_equity[sym] = bt.equity_curve

    aggregate = aggregate_metrics(all_trades, per_symbol_equity, cfg.base_capital)

    result = {
        'symbols': symbols,
        'config': cfg.__dict__,
        'per_symbol': per_symbol_metrics,
        'per_symbol_gate_stats': per_symbol_gate_stats,
        'aggregate': aggregate
    }

    print(json.dumps(result if args.verbose else {'aggregate': aggregate, 'per_symbol': per_symbol_metrics}, indent=2))

    if args.out_json:
        with open(args.out_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
    if args.out_csv:
        import csv
        rows = []
        for sym, metrics in per_symbol_metrics.items():
            row = {'symbol': sym}
            row.update(metrics)
            rows.append(row)
        fieldnames = sorted({k for r in rows for k in r.keys()})
        with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(rows)


def build_parser():
    p = argparse.ArgumentParser(description='Multi-symbol batch backtester (auto-fetch)')
    p.add_argument('--symbols', required=True, help='Comma separated list e.g. BTC/USDT,ETH/USDT,SOL/USDT')
    p.add_argument('--since-days', type=int, default=180, help='History window size in days (default 180)')
    p.add_argument('--exchange', default='binance', help='ccxt exchange id (default binance)')
    # Strategy params (mirroring backtest.py)
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
    p.add_argument('--near-ema-pct', type=float, default=0.02)
    p.add_argument('--disable-shorts', action='store_true')
    p.add_argument('--base-capital', type=float, default=1000.0, help='Per-symbol starting equity')
    p.add_argument('--risk-per-trade-pct', type=float, default=0.01)
    p.add_argument('--adaptive-near-multiplier', type=float, default=2.0)
    p.add_argument('--atr-compression-threshold', type=float, default=0.0008)
    p.add_argument('--trend-tp-boost', type=float, default=0.5)
    p.add_argument('--disable-bullish-patterns', action='store_true')
    p.add_argument('--disable-bearish-patterns', action='store_true')
    # Output
    p.add_argument('--out-json', help='Write full per-symbol + aggregate metrics JSON')
    p.add_argument('--out-csv', help='Write per-symbol metrics CSV')
    p.add_argument('--verbose', action='store_true')
    return p


if __name__ == '__main__':  # pragma: no cover
    parser = build_parser()
    args = parser.parse_args()
    run_multi(args)
