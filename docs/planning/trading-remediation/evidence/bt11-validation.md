# BT11 full-history account performance validation

Ticket [#13](https://github.com/adeoyewole028/binance_topdown_full/issues/13), fixed point3e47b2f3bcdecc5044c85b27950288a139de9654; source24115d1/58ecb4a/2d8b08c. All five local acceptance criteria passed at2d8b08c; closure publication/read-back is next.

| Criterion | Observable local evidence | Gate |
|---|---|---|
| Full history, pagination, restart and rebuild |24 completed positions/48 fills retain exact -9.6 USDT realized and990.4 equity across20/20/8 fill pages and runtime reinitialization; metrics use one committed snapshot | Passed; final344fullpass |
| Seeded equity and selected win denominator | Initial1000 capital/initial-loss drawdown.0096; pure loss/rebound retains.021 maximum drawdown; fully exited positions form the win denominator; partial closes do not count as completed; incomplete fill-prefix checkpoints withhold drawdown | Passed; final344fullpass |
| Partial fills, fees and freshness | Duplicate observations count once; quote fees in basis/proceeds; base fees in actual quantity; third-asset fee units retained with unknown historical conversion; fake Spot partial closes retain exact-.404 net USDT/.404 fees; stale prices/account and unresolved account withhold affected valuations | Passed; final344fullpass |
| Confirmed/estimated values and currency | Authenticated API and existing HTML separate confirmed realized, estimated unrealized and estimated account change, explicit USDT; incomplete fields null with human-readable reasons; unauthorized401 and invalid pagination422 | Passed; final344fullpass |
| Journal integrity and coherent rebuilding | Immutable seed change409 without altered metrics; changed cash mismatches fill reconstruction and withholds equity while retaining confirmed realized; one committed transaction reads fills/config/accounts/seed/checkpoints; old/imported history never receives invented opening basis | Passed; final344fullpass |

## TDD and prerequisite corrections

First behavior tests failed on the former scan-only placeholder. A new full-history projection now shares finite Decimal weighted entry/realized exit arithmetic with paper/replay; derived stored values retain their precision rather than being revalidated against operator-input digit limits. An early90case focused run exposed two original tiny-fee/large-capital regressions (88pass/2fail); both original tests passed after the ledger-value correction, without changing their expectations. A stale-price test initially mixed frozen owner and wall-clock public fixture times; the external fake public provider now follows the controlled clock and the real owned runtime, preserving existing product freshness/admission gates.

Initial frozen full342pass341.88s at58ecb4a is retained in [initial review snapshot](bt11-initial-review-snapshot.json). Spec review independently reproduced twoP2s: stale/unresolved exchange observations still published valuation-derived metrics, and an opening/final checkpoint gap falsely certified zero drawdown. Two added tests failed before correction and passed after2d8b08c. Current aggregate and position valuation now requires fresh reconciled account state; confirmed realized/native fees remain. Every unique confirmed fill prefix0..N requires qualified integer/nonbool/bounded/nondecreasing checkpoint coverage. Missing intermediate batch observations remain partial/null historical drawdown rather than invented valuations. Pure complete fee examples contain explicit observed intermediate equity values.

Corrected17performance tests passed25.99s; performance/replay48passed49.84s during corrections. Strict type50, repository pyflakes, diff check and32application source compilations passed. No separate binary/frontend asset build applies. No schema migration, retrospective journal fabrication, mutable balance editor or new capital/risk choices.

## Standards

Independent final review at2d8b08c:0 documented breaches/0 meaningful heuristic findings. Approved pure portfolio and authenticated owned fake-provider seams retained; no own-collaborator mocks or SQL acceptance assertions.

## Spec

Independent final review at2d8b08c:0 remaining actionable findings; bothP2s resolved. Five independent focused accounting/freshness/coverage checks passed34.77s. All five local acceptance criteria passed. Corrected frozen344case run returned342pass/2child-process timeouts in511.42s (15s separate HTTP process and30s normal-import subprocess). The failed snapshot is retained in bt11-timeout-review-snapshot.json. No accounting assertion failed; no timeout or safety guard has been relaxed. Both unchanged original timeout checks passed8.66s in isolation; the serial final run passed344/0failures/skips/xfails520.07s, retaining600s aggregate budget and unchanged individual timeout/safety guards.

## Evidence and boundaries

[Contract](../../../account-performance.md), [machine result/source hashes](bt11-account-performance.json). Realized profit is qualified confirmed accounting; current holdings are estimated from fresh observations, and drawdown describes recorded qualified equity samples rather than unobserved price excursions. Incomplete imported/provider history or historical currency conversions remain disclosed. No real provider, credentials, account funding, trade, activation, migration, deployment or push. Later browser/provider/operational/rollout acceptance remains separately assigned. Parent#2 remains open.

Final local gates:344fullpass520.07s;17performancepass25.99s; strict type50, repository lint/diff and32application source compilation pass; Standards0hard/0heuristic and Spec0remaining. Corrected source2d8b08cbf279f190b3b844ce02428c65966958e8; no source changes during final fullsuite.
