# BT13: Review orders and recover pending or unprotected positions

**GitHub issue:** https://github.com/adeoyewole028/binance_topdown_full/issues/15

## Parent

[Implementation specification #2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)


**Status:** open; BT13 sole active implementation ticket; five local acceptance gates open. The ready-for-agent label is triage only: resolve the prerequisites and decision blockers before execution.
**What to build:** Let an authenticated operator submit a reviewed intent and understand confirmed, rejected, unknown and unprotected outcomes.
**Blocked by:** [BT12 #14](https://github.com/adeoyewole028/binance_topdown_full/issues/14).
**Decision blockers:** None additional; strict sequential implementation authorized on2026-10-02 and carried forward.
**Specification:** [#2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)
**Requirement coverage:** R16, R17, R10, R11.

## Objective and context

Let an authenticated operator submit a reviewed intent and understand confirmed, rejected, unknown and unprotected outcomes. This is slice 13 of the audit-remediation initiative. Its upstream requirements and decision IDs are authoritative; future slices must not be implemented early.

## Scope and implementation requirements

- Review venue/mode, quantity, funds/exposure, fee estimate and config/price version; submit with reusable operation token.
- Render durable operation/fill/protection status and clear recovery instructions; disable unsafe submission while preserving safe close/reconcile actions.
- Expose close action for every applicable position independent of exit signals; reflect cancel/fill race and partial close.

## Affected components

Manual order/close/review UI, operation lookup/status, protection/reconciliation recovery views.

## Acceptance criteria

- [ ] Changed/stale reviewed inputs require refreshed review; server-side sizing never trusts a browser estimate.
- [ ] Repeated clicks/refresh/retry reference the same intent; accepted/pending/unknown are never described as confirmed fill.
- [ ] Actual fills, partial closure, protection failure and unresolved provider outcomes have distinct recoverable views.
- [ ] Operator can initiate a reviewed position close without waiting for an exit signal; unknown cancellation cannot authorize an oversell.
- [ ] Paper/live mismatch and permission interruptions preserve form context without executing action; keyboard/error focus is usable.

## Testing and validation

Browser and HTTP manual buy/partial-fill/unknown/reject/protection-fail/close journey with fake provider; duplicate token, conflict and auth interruption cases.

Run focused behavioral checks during development, relevant type/static/lint/build checks and the full applicable suite at closure. Record any genuinely inapplicable check with reason. Perform code-review, fix actionable findings and commit only scoped completed work, preserving unrelated changes.

## Regression requirements

Keep manual symbol/side/quantity and risk-size paths while removing misleading raw response success messages.

## Security requirements

Reviewed authenticated CSRF-protected live-sensitive actions; operation IDs authorized by namespace.

## Migration and rollout considerations

Quarantined positions are clearly read-only and cannot become exchange sells.

## Completion evidence

End-to-end browser status/recovery screenshots or recordings with operation traces and scoped commit.

Record criterion-by-criterion evidence links, test outputs, review outcome, commit, environment/revision and remaining limitations. Do not mark complete when a required gate is unavailable or fails. Stop dependent work, retain open status, and record exact blocker and next allowed step.

## Out of scope

Mainnet execution, bypassing safety gates for UI convenience.

## Execution record

Implementation: not started. Checks: not run for this ticket. Acceptance evidence: none. Review: test boundaries and breakdown approved by the user on 2026-10-02. Commit: none. Current blockers: incomplete prerequisite tickets and unresolved policy decisions above, where applicable. Publication is complete; no feature code changes are authorized by the current planning task. Next permitted implementation step requires explicit implementation instruction and satisfied blockers.


## BT13 start (2026-10-03)

Live#15OPEN/0comments/native#14CLOSED verified. BT01-BT12 complete; BT13 soleactive/fivecriteriaopen. Fixedpointdf35364e87345ff4f9d462e9a7c4a630121c7c6f. Preserve approved HTTP->transactional StateStore->kernel-owned execution->fake public/private provider and actual browser seams. Add side-effect-free reviewed order preview, signed semantic request/config/quote/holding context, stable namespace idempotency through retries/refresh, authenticated namespace operation lookup with actual fills/protection/recovery, and reviewed partial/full close for every applicable current holding independently of exit signals. Server canonical sizing/admission/protection cancellation/reconciliation remain authoritative; no numeric capital/risk policies invented. Forms and auth interruption preserve drafts; unresolved outcomes never imply fills or authorize another sell. Quarantined legacy data read-only. Existing safe /trade and /close adapters remain shared-intake compatibility routes; actual first-class UI review replaces their raw one-click forms. No real provider/keys/trades/mainnet/activation/deployment/push or futureticket.
