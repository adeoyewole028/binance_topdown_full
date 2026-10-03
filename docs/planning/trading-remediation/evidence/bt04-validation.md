# BT04 completed validation

Completed ticket: [BT04 #6](https://github.com/adeoyewole028/binance_topdown_full/issues/6). Review fixed point: `b7fa4993ec98df43c7a7c2604f3bc1ae3633dfcc`. All D07 targets were selected by the user; no D05 policy was chosen.

Full offline suite on 2026-10-03 at source revision `d34b004f96f266c5fb85505edbf62467cdde11e3`: **144 cases, 142 pass, 2 strict BT06 expected failures, zero other skips/failures**, 49.01 seconds. [Machine-readable evidence](bt04-market-data.json) records exact source/fixture hashes and environment boundaries. Strict mypy:29 sources; repository Pyflakes, diff whitespace and20-source compilation without imports passed. Implementation/correction commits: `5a9d4b6`, `1ac1db2`, `d34b004`.

| Criterion | Observable evidence |
|---|---|
| Completed/fresh validated inputs | `tests/test_market_runtime.py`: duplicate/gap/unsorted/OHLC/nonfinite/insufficient histories degrade; forming candles never reach evaluator; new close blocks until refetch; expired price/failed refresh cannot fill and preserve fetched-at; controlled-clock scanner expiry blocks dispatch. |
| Independent required coverage | Quote/whitelist/blacklist/static fallback test retains legitimate held and pending symbols; active durable protection fixture remains covered after position/universe changes. No BT07 protection lifecycle was activated. |
| Supervised health | HTTP sees durable last-completed scan, owner state and fail-closed readiness; owner restart reuses/revalidates slow cache; approved cadence coalesces missed slots; existing real-process kernel exclusion/crash handover and unresolved-outcome regression pass. |
| Atomic validated/versioned configuration |16 invalid value cases, atomic multi-field rejection, optimistic conflict; omitted flags/settings unchanged; equity_proxy regression promoted from strict expected failure. Unsupported trailing UI label updated. |
| Budget/failure behavior | Discovered weight/raw windows leave urgent capacity despite research; a second account namespace cannot reset shared IP usage; rate Retry-After survives scanner restart; auth has one attempt until journaled review; failed transient refresh retains age. Public raw Spot adapter plus existing evaluator/fake transport reaches HTTP paper projection/exit observation without real network. |

No npm/native frontend build exists: Flask's inline HTML is covered by authenticated HTTP renders. Python source compilation is recorded before commit. Final independent Standards and Spec reviews both report zero actionable findings. Remaining strict expected failures are BT06 exchange idempotency/fill observation; exchange dispatch is still inactive. This validation is local disposable simulation, not provider sandbox/live or deployment acceptance. Actual account migration, alert configuration, backup operations and deployment are later gates.


## Review and corrections

Standards:0 actionable findings, worst none. Spec initially found P1 stale ticker timestamp after slow history and P2 reported budget usage lost at rollover. Red-to-green regressions now cover63-second slow history, ordinary rollover, and known/unknown2-second metadata timing uncertainty. Independent Spec recheck ran four correction cases successfully and reports0 remaining findings, worst none. Both reviewers compared the fixed pre-ticket revision through `d34b004`; no later-ticket scope was implemented. Shared policy replaced duplicated readiness thresholds. All five criteria are satisfied within the offline/local BT04 scope.
