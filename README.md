Binance Top-Down Full Project (Daemon + Backtester)
- Copy .env.example -> .env and fill keys (use testnet for testing)
- Install: pip install -r requirements.txt
- Run daemon: python topdown_daemon.py
- Backtest: python backtester.py

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
