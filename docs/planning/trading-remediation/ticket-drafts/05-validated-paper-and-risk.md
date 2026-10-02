# BT05: Execute credible paper orders within capital and risk limits

**GitHub issue:** https://github.com/adeoyewole028/binance_topdown_full/issues/7

## Parent

[Implementation specification #2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)


**Status:** open; approved breakdown; implementation not started. The ready-for-agent label is triage only: resolve the prerequisites and decision blockers before execution.
**What to build:** Make a valid paper buy or sell change durable cash and holdings exactly once, and reject unsafe inputs without effects.
**Blocked by:** [BT04 #6](https://github.com/adeoyewole028/binance_topdown_full/issues/6).
**Decision blockers:** D05.
**Specification:** [#2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)
**Requirement coverage:** R08, R09.

## Objective and context

Make a valid paper buy or sell change durable cash and holdings exactly once, and reject unsafe inputs without effects. This is slice 5 of the audit-remediation initiative. Its upstream requirements and decision IDs are authoritative; future slices must not be implemented early.

## Scope and implementation requirements

- Validate canonical side/symbol, finite positive qty, fresh reviewed price/config, balance, open/reserved exposure, cooldown and exchange metadata.
- Implement atomic available/reserved paper capital and holdings, approved seed/fee/slippage/fill models, fixed/risk sizing and durable cooldown releases.
- Make paper success/rejection visible via durable operation status; idempotency covers repeats and semantic conflicts.

## Affected components

Shared order admission/sizing, paper broker/accounting, reservations/cooldowns, operator order endpoints.

## Acceptance criteria

- [ ] Negative/zero/NaN qty, arbitrary side, unsupported symbol, missing/stale price, overspend/oversell or risk-limit breach leaves balances/holdings unchanged.
- [ ] Same key/request yields one paper effect; changed request with same key conflicts; concurrent reservations cannot overdraw assets or position slots.
- [ ] Valid buy/sell records fees/cost basis and cash/holding movements atomically; restart retains all effects and cooldowns.
- [ ] Server validates/recomputes sizing regardless of UI estimate; initial capital/quote are explicit and equity_proxy cannot authorize spend.
- [ ] Rejected paper order is displayed as rejection, never success; entry cooldown does not block approved risk reduction.

## Testing and validation

HTTP-to-store order journey, deterministic paper accounting, simultaneous buy/sell/duplicate requests, fee/seed/cooldown/reservation and write-failure cases.

Run focused behavioral checks during development, relevant type/static/lint/build checks and the full applicable suite at closure. Record any genuinely inapplicable check with reason. Perform code-review, fix actionable findings and commit only scoped completed work, preserving unrelated changes.

## Regression requirements

Preserve manual buy/sell and risk/fixed sizing intent; replace phantom funds and zero-price behavior explicitly.

## Security requirements

Paper namespace cannot emit private exchange order calls; all mutations remain authenticated and CSRF-protected.

## Migration and rollout considerations

Imported positions with unknown basis/seed remain flagged; no invented initial capital or fee history.

## Completion evidence

Before/after capital/holding invariant outputs, concurrency rejection cases, policy version and scoped commit.

Record criterion-by-criterion evidence links, test outputs, review outcome, commit, environment/revision and remaining limitations. Do not mark complete when a required gate is unavailable or fails. Stop dependent work, retain open status, and record exact blocker and next allowed step.

## Out of scope

Real exchange dispatch, guaranteed profit, selecting unapproved risk/fee assumptions.

## Execution record

Implementation: not started. Checks: not run for this ticket. Acceptance evidence: none. Review: test boundaries and breakdown approved by the user on 2026-10-02. Commit: none. Current blockers: incomplete prerequisite tickets and unresolved policy decisions above, where applicable. Publication is complete; no feature code changes are authorized by the current planning task. Next permitted implementation step requires explicit implementation instruction and satisfied blockers.

