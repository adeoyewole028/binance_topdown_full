# Operator access and explicit launch contract (BT02)

## Configure one operator

Set `OPERATOR_USERNAME`, `OPERATOR_PASSWORD_HASH` and `FLASK_SECRET_KEY` in private deployment configuration or an untracked `.env`. Use Werkzeug's password hash, never a plaintext password. Generate it interactively without putting a password on the command line:

```powershell
python -c "from getpass import getpass; from werkzeug.security import generate_password_hash; print(generate_password_hash(getpass('Operator password: ')))"
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

The second output is a new session secret. Startup rejects absent operator configuration or a session secret shorter than 32 characters. There is one configured account; no registration or password-reset product. Change the configured hash to rotate its password, and restart HTTP workers. Rotation invalidates prior sessions.

Production uses `APP_ENV=production`, HTTPS and `wsgi:app`. Cookies are Secure, HttpOnly and SameSite=Strict. Sessions expire 30 minutes after sign-in, with no sliding extension; sign-out revokes that session in access-control storage. Five login attempts in 15 minutes throttle the configured account globally, including unknown usernames, fresh cookies, concurrent workers and process restarts. Existing signed-in sessions remain usable during a throttle window. Authentication storage errors fail closed with a redacted unavailable response. `DATA_DIR/operator-auth.sqlite3` contains only login attempt times and opaque expiring session records, never passwords or provider credentials. Keep it on private writable storage shared by all HTTP processes of this host.

All mutation routes, including login/logout, require a session-bound unpredictable CSRF token. Forms include it; an authenticated same-origin client can send `X-CSRF-Token` instead. A supplied Origin must match the request origin. `/` and `/api/status` require operator authentication; account CORS is same-origin. TLS terminates at a trusted reverse proxy; the app does not trust arbitrary forwarded client addresses. No account or password details enter public API responses or routine authentication errors.

`APP_ENV=development` permits non-Secure cookies only on isolated loopback HTTP. `python dashboard.py` binds to 127.0.0.1, rejects production, and disables debug/reload. BT03 retires its same-process `--scan` option; launch the separately owned scanner using `python run_worker.py --scan`. `FLASK_DEBUG`, `FLASK_RUN_DEBUG`, `FLASK_RUN_RELOAD` `GUNICORN_CMD_ARGS=--reload` or `--reload` settings fail validation; the mandatory gunicorn.conf.py launcher hooks reject effective reload values from configuration and overrides. Never use the Flask CLI as a production launcher.

## Mode, venue and server gates

| Setting | Values / default | Meaning |
|---|---|---|
| `TRADING_MODE` | paper / exchange; paper | Initial execution mode. Persisted legacy paper flags cannot override it. |
| `BINANCE_VENUE` | testnet / mainnet; testnet | Explicit venue for public market data and any private client. |
| `ALLOW_EXCHANGE_EXECUTION` | strict boolean; false | Immutable server gate; no HTTP field can enable it. |
| `ALLOW_MAINNET_EXECUTION` | strict boolean; false | Additional mainnet gate, not rollout authorization. |
| `AUTOMATION_ENABLED` | strict boolean; false | Automation remains stopped pending BT09 shared-owner integration. |
| `BINANCE_TESTNET_API_KEY/SECRET` | optional pair | Keys belonging to Spot testnet. |
| `BINANCE_MAINNET_API_KEY/SECRET` | optional pair | Keys belonging to Spot mainnet. |

Public paper clients receive no private credentials. Exchange mode requires the selected credential pair and applicable gates before any private call. Testnet construction calls `set_sandbox_mode(True)` immediately after construction, before any other provider method. There is no custom URL override. Server validation checks configured identity declarations; actual venue/account proof remains provider acceptance work. Require only necessary Spot read/trade permissions, no withdrawal permission, and provider-side restrictions during that later verification.

Compatibility: `PAPER_TRADING` maps strict true to paper and false to exchange; `TESTNET` maps strict true to testnet and false to mainnet. Boolean spellings true/false, 1/0, yes/no, y/n, on/off are accepted; unknown values fail. `FLASK_ENV` aliases `APP_ENV`. Generic `API_KEY/API_SECRET` and `BINANCE_API_KEY/BINANCE_API_SECRET` are supported only with matching `CREDENTIAL_VENUE`. Different nonempty canonical/alias keys, contradictory mode/venue aliases, missing credential halves and invalid enums fail startup. Scoped keys avoid the former ambiguous venue interpretation.

Ordinary settings preserve omitted fields and cannot switch mode. Reset preserves mode and never changes immutable venue, credentials or gates, or deletes holdings/history. The dynamic-universe control explicitly submits Enabled/Disabled. `/change-mode` requires CSRF, expected current mode and exact confirmation; actual namespace-changing transitions return a review conflict until BT03 supplies isolated durable recovery and reconciliation. Existing exposure cannot become a real sell by toggling a checkbox. An exchange startup finding legacy positions fails for review. Actual venue/account transitions with holdings are outside BT02.

## Ownership and upgrade boundary

Importing dashboard/daemon/backtester is inert: no background lifecycle or `.env` reads. Explicit launchers load `.env` without overriding supplied environment. The HTTP WSGI launcher never owns a scanner. BT03 requires explicitly initialized SQLite state, queues HTTP orders, and holds a kernel account/venue lock in the separately launched owner. Both HTTP workers read shared positions/configuration/scans. `run_scanner.py` remains restricted to isolated development paper mode with durable ownership. See [migration and recovery](durable-state.md). Exchange dispatch/reconciliation and automation remain future BT06/BT09 work. The current Render command points to `wsgi:app`; single-host hosting/topology validation, supervisor installation and operational backup/restore acceptance remain BT14. No active service, hosting purchase or rollout was changed.

Sources checked on 2026-10-02: [Flask security](https://flask.palletsprojects.com/en/stable/web-security/), [Flask configuration](https://flask.palletsprojects.com/en/stable/config/), [Werkzeug password hashing](https://werkzeug.palletsprojects.com/en/stable/utils/), [CCXT sandbox contract](https://docs.ccxt.com/docs/manual).
