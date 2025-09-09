Binance Top-Down Full Project (Daemon + Backtester + Parameter Sweep)

Quick Start:
- Copy .env.example -> .env and fill keys (use testnet for testing)
- Install: pip install -r requirements.txt
- Run dashboard (web UI bot): python dashboard.py
- Run daemon (headless loop): python topdown_daemon.py
- Single backtest run: python backtest.py --symbol BTC/USDT --data-15m data/BTCUSDT_15m.csv --data-1h data/BTCUSDT_1h.csv --data-4h data/BTCUSDT_4h.csv --data-1d data/BTCUSDT_1d.csv

### Backtester Parameters
Core logic mirrors live strategy (multi-timeframe gating, ATR sizing, partial profit, trailing stop, volatility regime filter).

Key args:
- --ema-fast / --ema-slow : Moving averages for trend & execution alignment
- --rsi-threshold : Higher TF momentum filter (1h)
- --rsi-exec-threshold : Execution RSI cross level (15m)
- --atr-period : ATR length (15m sizing)
- --atr-stop-mult / --atr-tp-mult : Stop & take-profit ATR multiples
- --min-stop-pct : Floor stop size when ATR too small
- --partial-tp-r-multiple : R multiple to trigger partial profit + BE stop
- --partial-tp-fraction : Fraction closed at partial (modelled position-level; current PnL assumes unit size)
- --trail-activate-r-multiple : R multiple to start trailing
- --trail-atr-mult : ATR multiple below price for trailing stop
- --min-atr-rel : Minimum ATR/price relative threshold to allow entries (volatility regime)

### Parameter Sweep
Provide comma-separated lists for any of:
- --sweep-ema-fast
- --sweep-ema-slow
- --sweep-atr-stop-mult
- --sweep-atr-tp-mult
- --sweep-partial-r
- --sweep-trail-r

Example sweep (writes CSV):
python backtest.py --symbol BTC/USDT \
	--data-15m data/BTCUSDT_15m.csv --data-1h data/BTCUSDT_1h.csv --data-4h data/BTCUSDT_4h.csv --data-1d data/BTCUSDT_1d.csv \
	--sweep-ema-fast 13,21,34 --sweep-ema-slow 50,89 \
	--sweep-atr-stop-mult 1.0,1.5 --sweep-atr-tp-mult 2.0,2.5 \
	--sweep-partial-r 0.8,1.0 --sweep-trail-r 1.2,1.5 \
	--sweep-out sweep_results.csv

Each combination runs independently; results print JSON per combo plus a Top 5 summary sorted by avg_r then total_pnl. CSV rows include parameters + metrics.

### Reported Metrics
- trades : Total closed trades
- total_pnl : Sum of raw PnL (unit position sizing)
- avg_r : Mean R multiple across trades with defined R
- win_rate : Percentage of profitable trades
- profit_factor : Gross profit / gross loss (inf if no losses)
- max_drawdown : Peak-to-trough equity decline (raw units)
- sharpe : Mean trade PnL / std trade PnL * sqrt(trade_count) (simple proxy)

### Advanced Features (Latest Update)
Added support for:
1. Short selling (disable via `--disable-shorts`).
2. Risk-based position sizing using `--base-capital` and `--risk-per-trade-pct`.
3. Adaptive near-EMA gating: `--near-ema-pct` base threshold expanded by `--adaptive-near-multiplier` when ATR compression (`ATR/price < --atr-compression-threshold`).
4. Dynamic TP extension when 1D & 4H trends align using `--trend-tp-boost`.
5. Candlestick pattern confirmation:
	 - Bullish: engulfing, hammer (disable via `--disable-bullish-patterns`).
	 - Bearish: bearish engulfing, shooting star (disable via `--disable-bearish-patterns`).
6. Equity curve export via `--equity-out equity_curve.csv` (single run).
7. Expanded metrics: `median_r`, `best_r`, `worst_r`, `expectancy_r`.

Example single run with equity export:
```
python backtest.py --symbol SOL/USDT \
	--data-15m data_15m.csv --data-1h data_1h.csv --data-4h data_4h.csv --data-1d data_1d.csv \
	--risk-per-trade-pct 0.01 --trend-tp-boost 0.5 --equity-out equity_curve.csv
```

Example sweep including new params:
```
python backtest.py --symbol SOL/USDT \
	--data-15m data_15m.csv --data-1h data_1h.csv --data-4h data_4h.csv --data-1d data_1d.csv \
	--sweep-ema-fast 13,21 --sweep-ema-slow 34,55 \
	--sweep-atr-stop-mult 1.0,1.5 --sweep-atr-tp-mult 2.0,3.0 \
	--sweep-near-ema-pct 0.02,0.03 --sweep-trend-tp-boost 0.3,0.5 \
	--sweep-risk-per-trade-pct 0.005,0.01 \
	--sweep-out sweep.csv
```

Sweep CSV now includes expectancy and distribution metrics for better ranking.

### Cleanup Enhancements
Recent refactor added:
- Config validation (rejects invalid EMA order, risk %, etc.).
- Gate diagnostics with `--verbose` showing counts of each rejection reason.
- JSON export via `--json-out run.json` including config, metrics, and gate stats.
- Equity-curve based drawdown calculation (more accurate with variable sizing).
- Cached DataFrame reuse during parameter sweeps for speed.

Example verbose JSON run:
```
python backtest.py --symbol SOL/USDT \
	--data-15m data_15m.csv --data-1h data_1h.csv --data-4h data_4h.csv --data-1d data_1d.csv \
	--verbose --json-out run.json
```

Gate stats fields:
- checks: total gating evaluations
- passes: number allowing an entry evaluation
- fail_insufficient_history / fail_trend1d / fail_trend4h / fail_near_ma / fail_momentum: reason counts

### Multi-Symbol Batch Backtesting (New)
`multi_backtest.py` lets you run the same configuration across multiple symbols (auto-fetching OHLCV via ccxt) and produces:
- Per-symbol metrics
- Per-symbol gate diagnostics (if `--verbose`)
- Aggregate metrics (all trades merged) with combined drawdown approximation

Example:
```
python multi_backtest.py --symbols BTC/USDT,ETH/USDT,SOL/USDT \
	--since-days 120 --ema-fast 21 --ema-slow 50 \
	--risk-per-trade-pct 0.01 --trend-tp-boost 0.5 \
	--out-json multi_results.json --out-csv multi_metrics.csv
```

Outputs (stdout):
- Default: compact JSON containing aggregate + per_symbol metrics
- With `--verbose`: full JSON including gate stat breakdowns

Aggregate Equity Approximation:
Starts with `base_capital * number_of_symbols`; trades are ordered by exit time to build a synthetic equity curve (simple sequencing). This ignores true concurrency but is adequate for coarse portfolio-level expectancy diagnostics.

CSV Columns (per-symbol rows): symbol + all strategy metrics (trades, total_pnl, avg_r, win_rate, profit_factor, max_drawdown, sharpe, median_r, best_r, worst_r, expectancy_r).

Notes:
- All symbols share the same parameter set.
- Data is always fetched fresh (future enhancement: on-disk caching for reuse).
- If you see zero trades across symbols, broaden gating (e.g. increase `--near-ema-pct`, reduce `--rsi-exec-threshold`, or relax volatility filters).

Planned Extensions:
- Parallel fetch & backtest execution
- Long vs short separated aggregate stats
- Walk-forward (chunked) multi-symbol runs
- Correlation matrix of per-symbol equity returns

### Notes / Future Ideas
- Position scaling and variable quantity not yet modelled in PnL (assumes qty=1). Partial profit logic marks state (breakeven shift) but does not reduce quantity for simplicity.
- Equity curve CSV export & per-trade R distribution histogram can be added later.
- Multi-symbol batch sweeps could wrap current runner externally.

---

## Deploy to Render (auto-deploy from Git)

This app runs a Flask web server and a background bot thread. Use Render for a long-lived process.

1) Push this repo to GitHub (or GitLab/Bitbucket).
2) In Render, create a new Web Service from your repo.
3) It will auto-detect `render.yaml`.

Key settings (in render.yaml):
- Runtime: Python 3.11
- Build: `pip install -r requirements.txt`
- Start: `gunicorn -w 2 -k gthread -b 0.0.0.0:$PORT dashboard:app`
- Persistent disk: mounted at `/opt/render/project/src/data` (used by `config.json` and `positions.json`).

Environment variables (set in Render dashboard):
- API_KEY, API_SECRET (Binance)
- PAPER_TRADING (true/false)
- NEAR_EMA_PCT, RSI_THRESHOLD, RSI_EXEC_THRESHOLD, RSI_OVERBOUGHT
- TAKE_PROFIT_PCT, STOP_LOSS_PCT, TRAILING_PCT
- UNIVERSE_SIZE, BASE_QUOTE, MIN_24H_QUOTE_VOLUME
- EQUITY_PROXY (for risk sizing)

Optional:
- DATA_DIR (defaults to repo directory; on Render we set to mounted disk path)
- FLASK_SECRET_KEY (for session/flash)

Autoscaling/Health:
- Plan can be Free/Starter for testing. Keep in mind free instances may spin down.
- Ensure `PORT` is provided by the platform (render.yaml sets a default), and that no debug reloader runs in production.

Access:
- After deploy, open the Render-provided URL. The dashboard runs and scans continuously.
# binance_topdown_full
