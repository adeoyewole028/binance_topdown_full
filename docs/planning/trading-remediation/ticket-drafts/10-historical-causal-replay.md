# BT10: Backtest complete historical ranges with causal execution

**GitHub issue:** https://github.com/adeoyewole028/binance_topdown_full/issues/12

## Parent

[Implementation specification #2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)


**Status:** open; approved breakdown; implementation not started. The ready-for-agent label is triage only: resolve the prerequisites and decision blockers before execution.
**What to build:** Produce reproducible historical trade/equity results that cannot see future bars or silently omit requested data.
**Blocked by:** [BT07 #9](https://github.com/adeoyewole028/binance_topdown_full/issues/9), [BT08 #10](https://github.com/adeoyewole028/binance_topdown_full/issues/10).
**Decision blockers:** D05.
**Specification:** [#2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)
**Requirement coverage:** R14, R09, R12.

## Objective and context

Produce reproducible historical trade/equity results that cannot see future bars or silently omit requested data. This is slice 10 of the audit-remediation initiative. Its upstream requirements and decision IDs are authoritative; future slices must not be implemented early.

## Scope and implementation requirements

- Fetch UTC start/end plus warmup with coverage/gap report, bounded chronological replay and deduplicated entries.
- Use selected runtime profiles and approved fees/slippage/entry-next-event/exits/capital/intrabar policy.
- Report fills, positions, exposure, equity and result provenance instead of only entry count.

## Affected components

Historical data pagination/replay CLI, shared evaluator and paper fill/accounting, research result provenance.

## Acceptance criteria

- [ ] Range older than last 1000 bars paginates correctly with required warmup; missing/gapped history reports incomplete rather than valid zero result.
- [ ] No decision uses final daily close before its availability, future bias bar, unfinished candle or overlapping iteration duplicate.
- [ ] Future-data mutation cannot change earlier decisions; identical as-of runtime/replay inputs produce identical signals.
- [ ] Fees/slippage/capital, gap exits, same-bar TP/SL ambiguity and partial-fill assumptions follow recorded D05 policy.
- [ ] CLI range/profile inputs and reproducible result provenance include coverage, version and selected assumptions; output makes no profitability guarantee.

## Testing and validation

Local paginated adapter fixtures, timezone/range limits, causal perturbation/runtime parity, deterministic portfolio/loss/exit accounting and gap/ambiguity cases.

Run focused behavioral checks during development, relevant type/static/lint/build checks and the full applicable suite at closure. Record any genuinely inapplicable check with reason. Perform code-review, fix actionable findings and commit only scoped completed work, preserving unrelated changes.

## Regression requirements

Keep research workflow offline-capable and selected shared indicator behavior; incorrect previous trade counts are not preserved.

## Security requirements

Public-only historical adapter and strict research rate lane; no order-capable calls from replay.

## Migration and rollout considerations

Version outputs; label old entry-only results unvalidated without attempting retrospective fabricated fills.

## Completion evidence

Coverage and causal/parity test reports, deterministic example results and scoped commit.

Record criterion-by-criterion evidence links, test outputs, review outcome, commit, environment/revision and remaining limitations. Do not mark complete when a required gate is unavailable or fails. Stop dependent work, retain open status, and record exact blocker and next allowed step.

## Out of scope

Historical market-data purchase, profit optimization or live fill fidelity guarantee.

## Execution record

Implementation: not started. Checks: not run for this ticket. Acceptance evidence: none. Review: test boundaries and breakdown approved by the user on 2026-10-02. Commit: none. Current blockers: incomplete prerequisite tickets and unresolved policy decisions above, where applicable. Publication is complete; no feature code changes are authorized by the current planning task. Next permitted implementation step requires explicit implementation instruction and satisfied blockers.
