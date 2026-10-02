# BT11: Report performance from full history and account equity

**GitHub issue:** https://github.com/adeoyewole028/binance_topdown_full/issues/13

## Parent

[Implementation specification #2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)


**Status:** open; approved breakdown; implementation not started. The ready-for-agent label is triage only: resolve the prerequisites and decision blockers before execution.
**What to build:** Let the operator trust metrics across refresh, history pagination and restart, with incomplete data clearly labelled.
**Blocked by:** [BT10 #12](https://github.com/adeoyewole028/binance_topdown_full/issues/12).
**Decision blockers:** D05.
**Specification:** [#2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)
**Requirement coverage:** R15, R05.

## Objective and context

Let the operator trust metrics across refresh, history pagination and restart, with incomplete data clearly labelled. This is slice 11 of the audit-remediation initiative. Its upstream requirements and decision IDs are authoritative; future slices must not be implemented early.

## Scope and implementation requirements

- Reuse approved replay cost-basis/fee/win-rate definitions for durable paper and exchange projections.
- Compute actual asset-qualified equity/peak drawdown with price and conversion freshness; retain realized/unrealized separation.
- Mark imported unknown basis/fees and missing conversions unavailable or partial rather than inventing totals.

## Affected components

Performance projection service, durable fills/equity snapshots, status API and replay summary.

## Acceptance criteria

- [ ] More than 20 closed trades, pagination, restart and rebuilding projections retain identical full-history totals.
- [ ] Drawdown uses positive seeded account equity and handles an initial loss correctly; win-rate denominator follows selected method.
- [ ] Partial fills/multi-asset fees are accounted once; unconvertible fee/basis or stale prices are visible as incomplete metrics.
- [ ] Displayed estimated/unrealized figures are distinguishable from confirmed realized values and currency is explicit.
- [ ] No mutable balance editing silently rewrites journal history; rebuild and snapshot consistency checks pass.

## Testing and validation

Deterministic known-capital/win-loss fixtures beyond display cap, reconstruction/restart, multi-asset fee/partial close and missing-history/conversion tests.

Run focused behavioral checks during development, relevant type/static/lint/build checks and the full applicable suite at closure. Record any genuinely inapplicable check with reason. Perform code-review, fix actionable findings and commit only scoped completed work, preserving unrelated changes.

## Regression requirements

Preserve performance display intent, remove false profit-relative drawdown and retained-history truncation.

## Security requirements

Authenticated projections only; sanitized research/sample account data.

## Migration and rollout considerations

Legacy metrics remain unavailable/partial until provenance exists; no invented capital or complete history.

## Completion evidence

Expected versus observed metrics, projection rebuild parity and scoped commit.

Record criterion-by-criterion evidence links, test outputs, review outcome, commit, environment/revision and remaining limitations. Do not mark complete when a required gate is unavailable or fails. Stop dependent work, retain open status, and record exact blocker and next allowed step.

## Out of scope

New risk limits, tax reporting or return promises.

## Execution record

Implementation: not started. Checks: not run for this ticket. Acceptance evidence: none. Review: test boundaries and breakdown approved by the user on 2026-10-02. Commit: none. Current blockers: incomplete prerequisite tickets and unresolved policy decisions above, where applicable. Publication is complete; no feature code changes are authorized by the current planning task. Next permitted implementation step requires explicit implementation instruction and satisfied blockers.
