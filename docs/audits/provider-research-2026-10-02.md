# Provider and deployment contract audit

Date: 2026-10-02. Scope: static repository inspection plus current primary documentation. No Binance account, credentials, trading requests, deployment, or application network execution were used. Findings concern implementation contracts; provider sandbox acceptance and actual deployed configuration remain unverified.

## Conclusion

The repository should not be treated as ready for unattended live trading. Its declared testnet control is bypassed, daemon sizing reads a nonexistent CCXT field, protective orders use a futures order type on Spot, and the dashboard deployment combines independent worker state with a free-plan disk declaration that Render does not support. These are source-level conclusions, not a claim that the current service has lost money or that a particular deployment failed.

## Findings

### PR01 — Critical: TESTNET does not select testnet

`topdown_daemon.py:119` reads TESTNET, but `:126` calls `make_exchange(testnet=False)` unconditionally. `dashboard.py:334-339` creates a mainnet Spot exchange without sandbox selection. Setting TESTNET=true therefore does not isolate trading; PAPER_TRADING=false makes order paths real-money paths. This contradicts the testing instructions in `README.md:2`.

CCXT requires sandbox selection immediately after constructing an exchange and requires separate sandbox credentials. [CCXT manual](https://github.com/ccxt/ccxt/wiki/manual). The unused daemon sandbox branch at `topdown_daemon.py:20-21` also overwrites URLs with `/api`, whereas current CCXT Spot test URLs already contain `/api/v3`; this overwrite is incompatible with the adapter's endpoint layout. [CCXT Binance adapter](https://raw.githubusercontent.com/ccxt/ccxt/master/python/ccxt/binance.py).

Required acceptance: prove environment identity before any private operation; reject contradictory credentials/settings; exercise sandbox buys and protective orders with explicit sandbox evidence. Binance Spot Testnet periodically resets orders and balances, so test persistence must reconcile after resets. [Binance Testnet](https://developers.binance.com/en/docs/products/spot/testnet/general-info).

### PR02 — High: sizing reads the wrong market field and does not enforce trading filters

`topdown_daemon.py:81,167` indexes `market['limits']['amount']['step']`. Current CCXT Binance parsing stores stepSize in `precision.amount`; amount limits contain min/max. The expression fails before its `or 1` fallback can run. [CCXT Binance adapter](https://raw.githubusercontent.com/ccxt/ccxt/master/python/ccxt/binance.py).

`topdown_daemon.py:92-93` rounds prices to eight decimal places regardless of symbol rules. `dashboard.py:483,508` accepts a submitted float quantity without application-level amount/notional/balance validation. CCXT supplies precision-formatting methods after markets are loaded; precision and minimum limits are distinct. [CCXT manual](https://docs.ccxt.com/docs/manual). Binance checks price ticks, lot quantities, market lot quantities and applicable min/max notional rules. [Binance filters](https://developers.binance.com/en/docs/products/spot/filters).

Required acceptance: fixtures for multiple amount/price steps and market-notional rules; reject zero, negative, nonfinite, below-minimum and over-balance orders; validate against sandbox exchange metadata. Adapter formatting alone is not evidence that all filters or balance constraints pass.

### PR03 — Critical: daemon protective-order contract is incorrect and non-atomic

`topdown_daemon.py:96` submits a full-quantity limit sell, then `:97-98` submits another sell with `type=STOP_MARKET`. Spot documents STOP_LOSS/STOP_LOSS_LIMIT, not STOP_MARKET. The first sell can succeed before the second fails; `:181-182` labels the composite action as failed without recovery. No OCO linkage, cancellation, protective-order verification or fill reconciliation exists.

Binance supports OCO order lists with linked take-profit and stop-loss legs and cancellation when the competing leg activates. [Binance trading endpoints](https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/trade). The current CCXT adapter distinguishes Spot STOP_LOSS from futures STOP_MARKET. [CCXT Binance adapter](https://raw.githubusercontent.com/ccxt/ccxt/master/python/ccxt/binance.py).

`dashboard.py:453-464` merely publishes exit signals; live `/trade` submits a market order without attaching an exchange-native stop. The screen's stop-loss setting does not prove executable protection.

Required acceptance: select and specify a supported Spot protective-order design; reconcile actual filled quantity and fees; prove protection installation and cancellation; stop further entries and alert on unprotected exposure. Model partial success explicitly rather than treating a multi-order sequence as all-or-nothing.

### PR04 — High: ambiguous order outcomes have no reconciliation

Daemon order exceptions flow to `topdown_daemon.py:181-182` without persisted order identity or exchange status queries. Dashboard trade/close exceptions likewise just flash/log an error at `dashboard.py:486-488,511-513`. Binance states that timeouts and 5xx responses can have unknown execution status and must not automatically be interpreted as failed execution. [Binance REST general information](https://developers.binance.com/en/docs/products/spot/rest-api).

Required acceptance: durable order intent and client identity, status lookup on ambiguous response, startup reconciliation of balances/orders/fills and safe handling of retries. Do not assume repeating a request is safe merely because the first response was an error.

### PR05 — High: deployment creates multiple independent scanner/state owners

`render.yaml:19` starts two Gunicorn workers. `dashboard.py:1022-1028` starts a bot thread on module import; `_bot_started` is a process-local flag. Gunicorn uses separate worker processes. [Gunicorn design](https://gunicorn.org/design/). Therefore each worker has its own status, broker state, exchange limiter and scanning thread, while JSON files are shared. UI requests can land on inconsistent workers and concurrent file writers can overwrite state. A restarted worker also starts another fresh scanner.

Required acceptance: one authoritative owner for execution and durable state, process-independent dashboard reads/control, defined locking/transaction rules, and restart/concurrency verification. Render provides continuously running background workers as a deployment primitive; whether to adopt that primitive is an architecture decision. [Render background workers](https://render.com/docs/background-workers).

### PR06 — High: Render Blueprint contradicts current service limits

`render.yaml:6` declares plan free, while `:20-23` declares a persistent disk. Current Render documentation says free web services cannot have persistent disks; they also spin down after 15 minutes without incoming traffic and may restart at any time. The Blueprint does not establish an always-running durable scanner. [Render free services](https://render.com/docs/free).

Only data under a mounted disk survives when disk storage is actually provisioned. `topdown_daemon.py:177` writes trade history to relative `logs/`, outside the declared DATA_DIR. Persistent disks impose deployment/scaling restrictions and are not shared across separate services. [Render persistent disks](https://render.com/docs/disks).

Required acceptance: validate an actually supported Blueprint, choose uptime/storage requirements, persist the authoritative ledger, document graceful shutdown and restart reconciliation, and prove the deployed journey. A successful build or healthy homepage does not prove continuous trading or data durability.

### PR07 — Medium: process-local throttling lacks aggregate rate-limit handling

Both exchange constructors enable CCXT throttling. Dashboard OHLCV retries (`dashboard.py:59-75`) catch every error and sleep 0.5/1 seconds; no application handling of Retry-After or an account/IP-wide request budget is visible. Independent worker exchange instances increase aggregate traffic. Binance request limits are IP-based; 429/418 responses require waiting as directed. [Binance REST general information](https://developers.binance.com/en/docs/products/spot/rest-api).

Required acceptance: central request budgeting where multiple processes share egress, differentiated transient/authentication/rate-limit failures, provider-directed backoff, bounded stale-data behavior and observable scan health.

## Versions and proof boundaries

`requirements.txt` uses lower bounds only (`ccxt>=4.3.53`, `gunicorn>=21.2.0` and other dependencies). There is no reproducible dependency lock in the inspected tree. Current adapter source establishes current contracts; its exact compatibility with an already deployed dependency version must be checked separately. Record installed versions, Python version and provider metadata in acceptance evidence without secrets.

This research does not approve an architecture, implement fixes, or authorize activation. Planning should preserve paper behavior, make environment selection fail-safe, choose a single execution/state owner and supported protection contract, then specify dependency-ordered remediation and sandbox/restart/failure acceptance gates.

