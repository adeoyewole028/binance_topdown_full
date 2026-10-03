# BT05: Execute credible paper orders within capital and risk limits

**GitHub issue:** https://github.com/adeoyewole028/binance_topdown_full/issues/7

## Parent

[Implementation specification #2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)


**Status:** in progress; BT04 closed and native dependency verified; D05 policy approved. Sole active ticket.
**What to build:** Make a valid paper buy or sell change durable cash and holdings exactly once, and reject unsafe inputs without effects.
**Blocked by:** [BT04 #6](https://github.com/adeoyewole028/binance_topdown_full/issues/6).
**Decision blockers:** none for the selected D05 policy; explicit runtime setup/provider acceptance remain required as applicable.
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


## Current handoff (2026-10-03)

BT04#6 CLOSED/read-back and native prerequisite verified. BT05 remains unstarted. D05 capital input contract selected: explicit operator setup required, execution blocked while seed/fee/slippage/allocation/exposure/cooldown/drawdown inputs are unset; no invented numerical values. Accounting/replay and protection selections remain pending. This supersedes the preliminary execution record only for prerequisite/decision status; no BT05 effects or acceptance claimed.

## D05 approved contract (2026-10-03)

APPROVED on 2026-10-03: explicit operator setup for all capital/risk/fee/slippage/cooldown/drawdown inputs before paper trading; deterministic full paper fills, quote-currency fees, weighted-average cost basis, adverse slippage and gap prices, stop first when both stop/profit targets touch one candle. Exchange-supported stop-market protection; refuse unsupported symbols; re-protect confirmed remaining holdings after partial closes; failed/uncertain protection blocks entries; automatic emergency liquidation and trailing remain disabled.
Implement only this ticket’s scope; this approval does not authorize future tickets early, account funding or live/provider/rollout actions.

BT05 started after live issue/comments/native dependency inspection; pre-ticket review point `2b77b6dbb786fbd6e7162b84d4c4e0c1d7b8d6bf`. All five criteria remain open pending implementation, checks, review, evidence and scoped commit. Approved HTTP/owned-executor/fixture-provider seam carries forward.
