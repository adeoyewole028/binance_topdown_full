# BT08 acceptance progress

Sole active ticketBT08#10; fixedpointfeed17e1c8f499ee990ff3e08335ec1ab164e748. D04 both profiles retained. All five acceptance criteria remain open until final review/full checks.

| Criterion | Current seam evidence | Gate |
|---|---|---|
| Golden named profile/reason parity |Literal BT01 trend/bias/cross fixtures; both full-profile engulfing/hammer branches; ordered indicator exit examples |Passed; final acceptance below |
| Causal completed/as-of data |Both profiles future/forming perturbation with nonfinite future values, timezone parity, warmup/gap/duplicate/unsorted/OHLC/stale negative paths, UTC daily/execution close boundary |Passed; final acceptance below |
| Exact documented definitions |Version1catalog preserves daily300/200/200 and dashboard200/200/100 windows; original EMA/rolling-RSI/close pivots/pattern and exit ordering |Passed; final acceptance below |
| Stable identity/provenance |Repeated owned scans/restart HTTP signal IDs; original profile/config version retained on queued intent after configchange; explicit canonical profile chooser |Passed; final acceptance below |
| Pure evaluator/adapter isolation |Same explicit daily profile produces identical dashboard/daemon adapter outputs; readonly config/unchanged input; socket/provider guards show no execution |Passed; final acceptance below |

TDD tracers:red missing evaluator then golden green; gapped history incorrectlyready red then explicitunavailable green; missing bound exit red then ordered reasons green; HTTP signal identity missing red then repeat/restart green; daemon adapter missing red then parity green; intent profile provenance missing red then immutable projection green. Focused73baseline/market/strategy cases passed before later additional checks; newest20versioned cases passed and added ordered indicator-exit proof passed. Final exact-revision checks supersede intermediates. [Profile contract](../../../strategy-profiles.md). No actual provider, trade, funding, migration, deployment or push. Automation/replay remain future tickets, not implemented early.

## Standards

Required independent review pending.

## Spec

Required independent review pending.


## Initial reviews and corrections

Standards at8dbfd76:0documented violations,1optional P3 possible duplication of the bias-window200 across runtime/evaluation. Bias window now lives in the immutable Profile catalog and both consumers read it.

Spec at8dbfd76:1actionableP2. Daily v1 had inherited configurable dashboard near-EMA/RSI thresholds rather than preserving characterized fixed0.02/30. Independent guarded golden comparison and root red parity check reproduced bothconditionsFalse instead ofTrue at dashboard near0/RSI99. The daily immutable catalog now owns its fixed thresholds; dashboard remains configurable; no mathematical definition or separately approved version change is introduced. Documentation corrected. Final rechecks/full validation pending; ticket remains open.


## Full-suite runner budget correction

First complete run at52b5c65 reached about79percent with no reported test failure, then the pre-existing aggregate300s runner budget expired and produced no complete machine acceptance artifact. A standalone20case protection rerun passed50.70s (slowest4.04s); Python startup probe0.11s. No stalled process or source test defect was found. The suite now has271cases and this host showed slower execution than the earlier249case151.71s run. The baseline runner now offers an explicit positive --timeout budget, retaining default300s; final BT08 run uses bounded600s. Individual subprocess15s and all clock/freshness/provider/socket/credential/write/assertion gates are unchanged; no failure is waived or claimed complete. This narrowly scoped harness correction is necessary for complete evidence, not a product or risk-policy change. Independent rechecks and final complete run remain required.


## Final independent reviews

Standards at933c9f9:0hard violations/0remaining heuristic findings. Original optionalP3 bias-window duplication resolved in the immutable catalog; final runner-budget correction preserves default/individual gates and records chosen resource limit. Prior product review remains valid.

Spec at933c9f9:0remaining actionables. OriginalP2 daily threshold drift resolved; independent2case golden/non-default bias parity passed. No additional requirement gaps or unapproved scope expansion; harness budget correction does not change product rules or waive a failure. Both axes read-only with no external calls.

## Final acceptance

All five BT08 criteria passed at933c9f9f76ac1a6b9f1ef2367de81d064c2c6859. [Exact machine results/27 source hashes](bt08-versioned-strategies.json):271passed,0failures/skips/xfails,235.30s, bounded aggregate budget600s. Strict type40sources, repository lint/diff and27application-source compilation pass.22versioned cases in the final full suite cover golden/profile/exit parity, fixed daily thresholds, invalid data/as-of/future isolation, adapter parity, scan restart identity and immutable intent provenance. Original incomplete300s attempt remains historical, superseded by this complete run.

Source commits8dbfd76/52b5c65; harness correction933c9f9; fixedpointfeed17e. Existing immutable config/journal/scan storage used; no schema/data migration. Legacy dashboard profile defaults are compatibility reads, not immutable-history rewrites or reissued orders. No actual provider/trade/funding/migration/deployment/push. Automation lifecycle remainsBT09, causal portfolio replayBT10, full browser/accessibilityBT12/BT13, provider acceptanceBT15. Publish/close/read-back#10 next; parent#2 remains open.
