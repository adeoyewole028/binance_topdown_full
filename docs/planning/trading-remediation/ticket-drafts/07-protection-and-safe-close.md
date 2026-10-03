# BT07: Protect confirmed positions and close without overselling

**GitHub issue:** https://github.com/adeoyewole028/binance_topdown_full/issues/9

## Parent

[Implementation specification #2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)


**Status:** open; BT07 implementation in progress; all acceptance gates remain open. The ready-for-agent label is triage only: resolve the prerequisites and decision blockers before execution.
**What to build:** Give filled positions confirmed protective orders and a safe manual close, with explicit partial-failure recovery.
**Blocked by:** [BT06 #8](https://github.com/adeoyewole028/binance_topdown_full/issues/8).
**Decision blockers:** none for the selected D05 policy; explicit runtime setup/provider acceptance remain required as applicable.
**Specification:** [#2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)
**Requirement coverage:** R11, R03, R09.

## Objective and context

Give filled positions confirmed protective orders and a safe manual close, with explicit partial-failure recovery. This is slice 7 of the audit-remediation initiative. Its upstream requirements and decision IDs are authoritative; future slices must not be implemented early.

## Scope and implementation requirements

- Implement selected Spot protection using sellable confirmed quantity after fees; track installation/verification/list-leg states.
- Provide the corresponding isolated paper protection/exit lifecycle under the approved simulation policy; simulated protection is labelled separately from exchange-confirmed protection.
- Handle protection failure/unknown outcome, halt new entries and expose/alert unprotected exposure without erasing successful entry.
- Coordinate cancellation, partial fills and manual close/reprotection; serial reservation and reconciliation prevent oversells.

## Affected components

Protection groups/Spot order lists, executor reconciliation, close endpoint/operation status.

## Acceptance criteria

- [ ] Position is labelled protected only after provider confirms selected supported Spot order/list; no STOP_MARKET Spot request.
- [ ] Paper TP/SL/protection lifecycle and partial-close accounting are exercised under the same selected rules without private exchange calls; status clearly identifies simulation.
- [ ] Failed/unknown protection leaves entry/fills recorded and halts new entries; operator sees next recovery action.
- [ ] Protection-fill/manual-close race, duplicate close and cancellation timeout cannot submit beyond confirmed remaining quantity.
- [ ] Partial close retains remaining holdings and selected protection policy; stop/take-profit counterpart lifecycle is verified.
- [ ] Kill switch and graceful stop preserve exchange protection and allow authorized reconciliation/risk reduction.

## Testing and validation

Fake order-list contracts and buy/protection partial failure; cancel-fill race, fee-in-base, duplicate/partial close and restore tests; sandbox proof in BT15.

Run focused behavioral checks during development, relevant type/static/lint/build checks and the full applicable suite at closure. Record any genuinely inapplicable check with reason. Perform code-review, fix actionable findings and commit only scoped completed work, preserving unrelated changes.

## Regression requirements

Keep exit reasons/manual close available independent of signal occurrence; no paper-to-real close path.

## Security requirements

Reviewed authenticated closes and no automatic emergency liquidation without explicit policy.

## Migration and rollout considerations

Reconcile any pre-existing protective orders before modifying/canceling; record unmanaged exposure for review.

## Completion evidence

Entry/protection state traces, race invariants, operator recovery examples and scoped commit.

Record criterion-by-criterion evidence links, test outputs, review outcome, commit, environment/revision and remaining limitations. Do not mark complete when a required gate is unavailable or fails. Stop dependent work, retain open status, and record exact blocker and next allowed step.

## Out of scope

Unapproved trailing stops/emergency liquidation, claims of guaranteed stop execution price.

## Execution record

Implementation: not started. Checks: not run for this ticket. Acceptance evidence: none. Review: test boundaries and breakdown approved by the user on 2026-10-02. Commit: none. Current blockers: incomplete prerequisite tickets and unresolved policy decisions above, where applicable. Publication is complete; no feature code changes are authorized by the current planning task. Next permitted implementation step requires explicit implementation instruction and satisfied blockers.

## D05 approved contract (2026-10-03)

APPROVED on 2026-10-03: explicit operator setup for all capital/risk/fee/slippage/cooldown/drawdown inputs before paper trading; deterministic full paper fills, quote-currency fees, weighted-average cost basis, adverse slippage and gap prices, stop first when both stop/profit targets touch one candle. Exchange-supported stop-market protection; refuse unsupported symbols; re-protect confirmed remaining holdings after partial closes; failed/uncertain protection blocks entries; automatic emergency liquidation and trailing remain disabled.
Implement only this ticket’s scope; this approval does not authorize future tickets early, account funding or live/provider/rollout actions.

## BT07 start (2026-10-03)

Live #9/comments and native #8 CLOSED verified. Fixed review point `000913cbc8c6860fba9750cf62764a79cf87adac`. Only BT07 starts under carried-forward strict implementation authorization and approved D05. No actual provider/activation/funding/deployment/push. Next: offline protection/close lifecycle tracers, full validation and required two-axis review before closure.
