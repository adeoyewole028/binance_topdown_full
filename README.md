# Binance Topdown

Flask operator dashboard, optional daemon and historical research tool. The approved remediation is tracked in [specification #2](https://github.com/adeoyewole028/binance_topdown_full/issues/2) and [the execution ledger](docs/planning/trading-remediation/execution-ledger.md). Local code, provider acceptance and deployment are separate milestones.

Install requirements.txt and requirements-dev.txt. Configure the operator and environment using [the secure launch guide](docs/operator-launch.md) before running the dashboard. Existing .env files need operator credentials, a fresh session secret and explicit venue/execution settings. Never commit them.

- Isolated local HTTP server: `python dashboard.py` with APP_ENV=development. No scanner starts automatically; add `--scan` to explicitly run the paper scanner in this same process.
- Explicit isolated local paper scanner: `python run_scanner.py`. It owns its loop; it does not start an HTTP server. Shared durable production ownership is BT03 work.
- Production HTTP entry point: `gunicorn -c gunicorn.conf.py -w 2 -k gthread -b 0.0.0.0:$PORT wsgi:app` on the supported Linux host, behind HTTPS. Each worker initializes access controls; none starts a scanner.
- Optional legacy paper automation: `python topdown_daemon.py`, disabled unless AUTOMATION_ENABLED=true. Exchange automation awaits the BT09 shared reconciled executor.
- Historical research: `python backtester.py`; causality/accounting corrections remain BT10 work.

Ordinary application imports do not read .env, load financial snapshots, start threads or construct exchange clients. wsgi.py is explicitly a launch module: it loads configuration and access-control storage. Debug/reload are disabled. The obsolete dashboard:app entry point leaves access unconfigured; migrate to wsgi:app.

Run the credential-cleared offline suite with `python tools/check_baseline.py`, type checks with `python -m mypy` and lint with `python -m pyflakes .`. Strict expected failures identify remaining ticket defects and become passing ordinary tests under their owners. Approved rollout/provider gates remain open; no deployment or live trading is implied by these commands.
