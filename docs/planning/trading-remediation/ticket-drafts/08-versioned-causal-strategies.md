# BT08: Evaluate selected strategy profiles from completed candles

**GitHub issue:** https://github.com/adeoyewole028/binance_topdown_full/issues/10

## Parent

[Implementation specification #2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)


**Status:** open; BT08 implementation in progress; all acceptance gates remain open. The ready-for-agent label is triage only: resolve the prerequisites and decision blockers before execution.
**What to build:** Produce reproducible strategy signals from the same as-of inputs across runtime and replay.
**Blocked by:** [BT04 #6](https://github.com/adeoyewole028/binance_topdown_full/issues/6).
**Decision blockers:** None additional; applicable choices approved by the user on 2026-10-02.
**Specification:** [#2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)
**Requirement coverage:** R12, R06.

## Objective and context

Produce reproducible strategy signals from the same as-of inputs across runtime and replay. This is slice 8 of the audit-remediation initiative. Its upstream requirements and decision IDs are authoritative; future slices must not be implemented early.

## Scope and implementation requirements

- Implement D04-selected named/versioned strategy contracts without silently changing indicator math or choosing a timeframe.
- Route dashboard/daemon signal evaluation through pure inputs/config/position context; stable candle identity deduplicates signals.
- Document warmup, completed-bar alignment, insufficient/gap behavior, entry/exit reasons and legacy-correction differences.

## Affected components

Shared deterministic evaluator, indicator definitions/config profiles, dashboard/daemon signal adapters.

## Acceptance criteria

- [ ] Selected profile signal/reason outputs match characterized valid legacy fixtures at identical as-of completed-bar inputs.
- [ ] Future bars or forming-candle changes cannot alter previously emitted decision; insufficient/gapped history returns explicit unavailable state.
- [ ] Daily/4h/1h/15m roles, EMA/rolling-RSI/close-pivot semantics and RSI-cross/candlestick branches are documented and versioned.
- [ ] Same namespace/profile/config/symbol/execution candle yields stable signal identity across scans and restart.
- [ ] No strategy evaluator invokes network, mutates state or depends on global mutable config.

## Testing and validation

Golden strategy fixtures, future-data perturbation, timezone/candle-boundary, warmup/missing-bar and repeated signal tests at pure evaluator seam.

Run focused behavioral checks during development, relevant type/static/lint/build checks and the full applicable suite at closure. Record any genuinely inapplicable check with reason. Perform code-review, fix actionable findings and commit only scoped completed work, preserving unrelated changes.

## Regression requirements

Preserve exact approved indicator/profile definitions; timing fixes have documented before/after rationale.

## Security requirements

Evaluator receives no credentials and cannot execute financial tools.

## Migration and rollout considerations

Record profile/config version on new signals/intents; legacy signals are not reissued as new orders.

## Completion evidence

Profile contract, golden/causal/parity results, decision D04 record and scoped commit.

Record criterion-by-criterion evidence links, test outputs, review outcome, commit, environment/revision and remaining limitations. Do not mark complete when a required gate is unavailable or fails. Stop dependent work, retain open status, and record exact blocker and next allowed step.

## Out of scope

Strategy optimization, new indicators, silently deprecating a profile.

## Execution record

Implementation: not started. Checks: not run for this ticket. Acceptance evidence: none. Review: test boundaries and breakdown approved by the user on 2026-10-02. Commit: none. Current blockers: incomplete prerequisite tickets and unresolved policy decisions above, where applicable. Publication is complete; no feature code changes are authorized by the current planning task. Next permitted implementation step requires explicit implementation instruction and satisfied blockers.


## Approved decision update

APPROVED on 2026-10-02: preserve both existing strategies as named, versioned profiles; no automatic selection


## BT08 start (2026-10-03)

BT07#9 CLOSED and parent#2 OPEN read-back verified. Live#10/comments inspected (OPEN/0comments), native prerequisite#6 CLOSED. Sole active ticket. Fixed review point`feed17e1c8f499ee990ff3e08335ec1ab164e748`. Approved D04 preserves both named/versioned profiles without automatic selection; pure evaluator and completed-bar seam preserved. No automation intake/replay implementation early; no actual provider/trade/deployment/push. Next: golden/causal pure-evaluator tracer, runtime/daemon adapters, focused/full checks and required two-axis review.
