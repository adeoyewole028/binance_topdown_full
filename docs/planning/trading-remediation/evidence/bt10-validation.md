# BT10 historical replay validation

Sole active ticket [#12](https://github.com/adeoyewole028/binance_topdown_full/issues/12), fixedpoint9853710e3da1111a0f4ea0e3e0c96a04abe623c4, source15d7fb3/ab5f9a9. D04 both named/versioned profiles and both D05 rules retained. All five local acceptance criteria passed atab5f9a9; closure publication/read-back is next.

| Criterion | Observable local evidence | Gate |
|---|---|---|
| Historical pagination and warmup/coverage | 1200 requested execution bars plus100 warmup cross two1000-bar pages; explicit UTC bounds; missing/duplicate/invalid OHLC histories incomplete; broad empty range has arithmetic coverage without timestamp allocation | Passed; final full327pass |
| Completed daily/bias bars and no overlapping entry | Capture-time excludes forming bars; both profiles compare each boundary to runtime evaluator; future daily final close cannot change earlier decisions; one held position suppresses entries | Passed; final full327pass |
| Causal perturbation and runtime parity | Future execution bar changes preserve earlier decisions/fills; daily future mutation and identical as-of decisions for both profiles; inputs immutable | Passed; final full327pass |
| D05 portfolio and protection | Exact quote fees/full fills/adverse slippage/weighted basis, cash/exposure caps, sticky drawdown reduction, adverse stop gaps, stop-first ambiguous bars, retained end holdings/null zero-denominator win rate; opening protection outranks queued exit without using later intrabar prices | Passed; final full327pass |
| CLI/provenance/assumptions | Repeated offline CLI output byte-identical; explicit UTC date/range/profile/version/quote/config/policy/precision; capture/coverage/data-config hashes; public range transport and research reserve; shared-state output overwrite guard; no profitability/livefidelity claim | Passed; final full327pass |

## TDD and review corrections

Initial RED tests showed missing capture-time/CLI APIs. Mandatory recorded capture time now bounds acquired and offline completed bars; CLI is offline by explicit dataset or explicitly public-fetch with existing shared state. A mistaken gap-P&L expected literal was corrected to verified arithmetic (-13.801 realized,986.199 cash); product math unchanged. An external test fixture attempted to create a nested sandbox outside its active disposable root; setup is now obtained before entering the guarded provider fixture, retaining the write guard. Pure/runtime parity and repeatable CLI tests preserve financial configuration without invented operator values.

Initial frozen full322pass/0fail/skips/xfail228.13s at15d7fb3 is preserved in [initial review snapshot](bt10-initial-review-snapshot.json). Standards P2 identified enumeration of every expected timestamp during empty preflight; corrected missing-count arithmetic uses already bounded/aligned/unique rows. Broad UTC1970–9999 regression exposed Windows-specific fromtimestamp limits; pure epoch/timedelta date validation preserves coverage without platform-dependent failures or fabricated caps. Spec P2 independently reproduced queued indicator exit receiving128.7 despite bound conservative target111.08889 at opening gap. RED target/stop gap cases now pass after opening-only protection precedes pending actions; normal opening closes at99 even if later same-bar low reaches70. Runtime paper ordering already correct and unchanged.

31focusedpass18.90s, strict type48, repository pyflakes/diff and31application compilation pass. Shared paper fill/sizing arithmetic extraction is the necessary BT10 integration change; initial fullsuite includes all previous paper/automation/protection regressions. No schema migration, financial state provisioning, capital defaults, profile redesign or future-ticket source.

## Standards

Final independent review atab5f9a9:0hard violations/0heuristic findings. Original resource-allocation P2 resolved. Protection/output guards stay within BT10 and approved external/pure seams.

## Spec

Final independent review atab5f9a9:0remaining actionable findings. Opening protection P2 resolved; independent target/stop/normal-open/broad-coverage/output-guard validation5passed4.73s. No missing requirements or scope creep identified.

## Evidence and limitations

[Deterministic explicit financial fixture](bt10-deterministic-example.json), [CLI/contract and primary-provider references](../../../historical-replay.md). [Final327case machine result and31source hashes](bt10-historical-replay.json). Build is Python source compilation; no separate binary asset/build pipeline applies to this research ticket. No database migration or compatibility backfill; old entry-only counts labelled unvalidated and no retrospective fills fabricated. No actual provider, credentials, funding, trade, activation, deployment or push. OHLC simulation does not certify historical/live exchange filters or fill fidelity; actual provider/rollout gates remain later tickets. All five local acceptance gates passed. Final full327pass/0failures/skips/xfails261.03s, aggregatebudget600s; type48/repositorylint/diff/31sourcecompile pass; Standards0hard/0heuristic and Spec0remaining. Source15d7fb3/ab5f9a9; fixedpoint9853710. Parent#2 and futureprovider/rollout gates remain open.
