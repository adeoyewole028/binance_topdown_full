# Historical research (BT10)

`backtester.py` replaces the unvalidated entry counter with format-version-1 portfolio replay. Old entry-only output is unvalidated; no retrospective fills are fabricated. Both `dashboard_intraday` version 1 and `daemon_daily` version 1 are explicit choices. Importing research modules never loads credentials, starts a worker or requests a provider.

## Offline study

Supply captured dataset JSON, a complete strategy configuration JSON and a complete explicitly selected paper policy JSON. Policy fields are the same mandatory fields documented in `paper-execution.md`; there are no research capital, fee, slippage or risk defaults. Select an explicit lot step for the research assumption. It does not certify historical or current exchange filters.

```powershell
python backtester.py --data captured-history.json --config study-config.json --policy study-policy.json --symbol BTC/USDT --venue testnet --profile dashboard_intraday --start 2026-10-03T12:00:00Z --end 2026-10-03T13:15:00Z --sizing fixed --quantity-step 0.001 --config-version 1 --output study-result.json
```

These dates, symbol and precision are an illustrative fixture, not recommended trading or capital settings. Date-only values mean UTC midnight; timestamps must explicitly indicate UTC. Ranges are `[start,end)`. The dataset contains `symbol`, `venue`, `profile`, `profile_version`, UTC millisecond `start`, `end`, `as_of` capture time, `state`, `coverage`, `bars` and `data_sha256`. Bars are six-value UTC-open-time/OHLC/volume rows keyed by timeframe. Replay revalidates bars and recomputes coverage/hash; changing a declared hash cannot make incomplete data valid. Wrong study identity/version or missing capture time is rejected.

Warmup is profile-specific: intraday 200 4h/200 1h/100 15m bars; daily 300 1d/200 4h/200 15m bars. Sorted unique aligned completed bars, valid positive OHLC and nonnegative volume are required. Forming/future bars are excluded using the recorded capture time. Missing warmup or study bars produces incomplete output with null fills, positions and metrics; CLI exit 2 distinguishes incomplete output from complete zero-trade output (exit 0). Input and output paths must differ. Public-fetch outputs must remain outside the shared trading state directory.

## Optional public acquisition

Replace `--data` with explicit `--fetch --state-directory EXISTING_SHARED_HOST_STATE --account CONFIGURED_ACCOUNT --state-mode paper` (or the existing exchange namespace mode). `--save-data captured-history.json` optionally preserves the dataset. The state must already be provisioned and valid; research does not provision, fund or change operator configuration. Use the same host-wide directory as the scanner/executor to share IP/venue counters. No `.env` or private credentials are read; only a public Spot transport is constructed. A server-time capture bound precedes pagination. Discovery costs 21 weight/two raw requests; every kline page costs 2 weight/one raw request. Research leaves the existing urgent reserve intact, propagates provider failure classes and stops rather than bypassing budget or synthesizing history. A deferred page makes its dataset incomplete. Retry later as a new study; there is no unbounded retry loop.

Primary contract checked 2026-10-03: [Binance Spot REST market data](https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/market). `GET /api/v3/klines` identifies candles by open time, permits at most 1000 per page and costs weight 2. Both startTime/endTime are UTC regardless of timeZone. Adapter sends timeZone 0 and inclusive provider endTime equal to the study's exclusive upper bound minus one millisecond. Source claims are documentation evidence; local tests use fake raw transports and do not prove actual provider availability or historical retention.

## Causal portfolio contract

Each decision uses only bars closed at its 15m boundary and the same versioned pure evaluator as runtime. Entries and indicator exits execute at the next available bar open. A single symbol/position cannot overlap another entry. Financial accounting shares paper sizing/fill arithmetic: adverse slippage, full deterministic fills, quote-currency fees and weighted fee-inclusive basis. Allocation/exposure/cash/risk, cooldown and sticky drawdown entry stop apply. Held positions can still reduce after the drawdown stop.

Stop/profit levels bind entry configuration. Existing protection observes opening-only prices before a queued indicator exit; later intrabar highs/lows cannot change an exit already filled at the open. A stop gap uses the worse opening price; when both stop and target touch one bar, stop wins. Targets use the conservative target price. OHLC cannot establish intrabar sequence: protection fills use the enclosing candle close timestamp as a reporting convention, not a claim of actual execution time. Equity/drawdown samples are opening, simulated fill and closed-bar events; unobserved intrabar peaks are not invented. At study end holdings remain marked at the last close, and any next-open signal remains unfilled. Win rate counts only complete position exits and is null when its denominator is zero.

Results record format/profile/config versions, canonical study configuration with its selected financial policy, policy/precision/execution assumptions, UTC bounds/capture time, validated coverage and SHA-256 data/config provenance, fills, decisions, cash/exposure/equity curve, positions, realized/unrealized return, drawdown and win-rate denominator. Identical offline inputs yield identical output. No profitability guarantee, historical filter certification or live execution fidelity is implied. Actual provider certification and rollout remain later tickets.
