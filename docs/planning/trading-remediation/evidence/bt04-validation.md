# BT04 validation in progress

Sole current ticket: [BT04 #6](https://github.com/adeoyewole028/binance_topdown_full/issues/6). Review fixed point: `b7fa4993ec98df43c7a7c2604f3bc1ae3633dfcc`. All D07 targets were selected by the user; no D05 policy was chosen.

Full offline suite on 2026-10-03: **139 cases, 137 pass, 2 strict BT06 expected failures, zero other skips/failures**, 46.11 seconds. [Machine-readable evidence](bt04-market-data.json) captures source hashes, preserved golden fixture hash, environment/dependencies and boundary limitations. Strict mypy:29 sources; repository Pyflakes and diff whitespace check passed. Review/closure still pending; source revision in the JSON is the pre-commit HEAD, with exact tested working-source hashes.

| Criterion | Observable evidence |
|---|---|
| Completed/fresh validated inputs | `tests/test_market_runtime.py`: duplicate/gap/unsorted/OHLC/nonfinite/insufficient histories degrade; forming candles never reach evaluator; new close blocks until refetch; expired price/failed refresh cannot fill and preserve fetched-at; controlled-clock scanner expiry blocks dispatch. |
| Independent required coverage | Quote/whitelist/blacklist/static fallback test retains legitimate held and pending symbols; active durable protection fixture remains covered after position/universe changes. No BT07 protection lifecycle was activated. |
| Supervised health | HTTP sees durable last-completed scan, owner state and fail-closed readiness; owner restart reuses/revalidates slow cache; approved cadence coalesces missed slots; existing real-process kernel exclusion/crash handover and unresolved-outcome regression pass. |
| Atomic validated/versioned configuration |16 invalid value cases, atomic multi-field rejection, optimistic conflict; omitted flags/settings unchanged; equity_proxy regression promoted from strict expected failure. Unsupported trailing UI label updated. |
| Budget/failure behavior | Discovered weight/raw windows leave urgent capacity despite research; a second account namespace cannot reset shared IP usage; rate Retry-After survives scanner restart; auth has one attempt until journaled review; failed transient refresh retains age. Public raw Spot adapter plus existing evaluator/fake transport reaches HTTP paper projection/exit observation without real network. |

No npm/native frontend build exists: Flask's inline HTML is covered by authenticated HTTP renders. Python source compilation is recorded before commit. Required independent Standards/Spec reviews will be added before closure. Remaining strict expected failures are BT06 exchange idempotency/fill observation; exchange dispatch is still inactive. This validation is local disposable simulation, not provider sandbox/live or deployment acceptance. Actual account migration, alert configuration, backup operations and deployment are later gates.
