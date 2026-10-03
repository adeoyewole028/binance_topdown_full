# BT08 acceptance progress

Sole active ticketBT08#10; fixedpointfeed17e1c8f499ee990ff3e08335ec1ab164e748. D04 both profiles retained. All five acceptance criteria remain open until final review/full checks.

| Criterion | Current seam evidence | Gate |
|---|---|---|
| Golden named profile/reason parity |Literal BT01 trend/bias/cross fixtures; both full-profile engulfing/hammer branches; ordered indicator exit examples |Final review/full pending |
| Causal completed/as-of data |Both profiles future/forming perturbation with nonfinite future values, timezone parity, warmup/gap/duplicate/unsorted/OHLC/stale negative paths, UTC daily/execution close boundary |Final review/full pending |
| Exact documented definitions |Version1catalog preserves daily300/200/200 and dashboard200/200/100 windows; original EMA/rolling-RSI/close pivots/pattern and exit ordering |Final review/full pending |
| Stable identity/provenance |Repeated owned scans/restart HTTP signal IDs; original profile/config version retained on queued intent after configchange; explicit canonical profile chooser |Final review/full pending |
| Pure evaluator/adapter isolation |Same explicit daily profile produces identical dashboard/daemon adapter outputs; readonly config/unchanged input; socket/provider guards show no execution |Final review/full pending |

TDD tracers:red missing evaluator then golden green; gapped history incorrectlyready red then explicitunavailable green; missing bound exit red then ordered reasons green; HTTP signal identity missing red then repeat/restart green; daemon adapter missing red then parity green; intent profile provenance missing red then immutable projection green. Focused73baseline/market/strategy cases passed before later additional checks; newest20versioned cases passed and added ordered indicator-exit proof passed. Final exact-revision checks supersede intermediates. [Profile contract](../../../strategy-profiles.md). No actual provider, trade, funding, migration, deployment or push. Automation/replay remain future tickets, not implemented early.

## Standards

Required independent review pending.

## Spec

Required independent review pending.


## Initial reviews and corrections

Standards at8dbfd76:0documented violations,1optional P3 possible duplication of the bias-window200 across runtime/evaluation. Bias window now lives in the immutable Profile catalog and both consumers read it.

Spec at8dbfd76:1actionableP2. Daily v1 had inherited configurable dashboard near-EMA/RSI thresholds rather than preserving characterized fixed0.02/30. Independent guarded golden comparison and root red parity check reproduced bothconditionsFalse instead ofTrue at dashboard near0/RSI99. The daily immutable catalog now owns its fixed thresholds; dashboard remains configurable; no mathematical definition or separately approved version change is introduced. Documentation corrected. Final rechecks/full validation pending; ticket remains open.
