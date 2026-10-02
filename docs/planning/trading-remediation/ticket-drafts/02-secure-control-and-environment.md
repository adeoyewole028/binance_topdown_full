# BT02: Secure control access and explicit environment transitions

**GitHub issue:** https://github.com/adeoyewole028/binance_topdown_full/issues/4

## Parent

[Implementation specification #2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)


**Status:** open; approved breakdown; implementation not started. The ready-for-agent label is triage only: resolve the prerequisites and decision blockers before execution.
**What to build:** Make unauthorized control and accidental real-money activation impossible through the existing entry points.
**Blocked by:** [BT01 #3](https://github.com/adeoyewole028/binance_topdown_full/issues/3).
**Decision blockers:** D01.
**Specification:** [#2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)
**Requirement coverage:** R01, R02, R03.

## Objective and context

Make unauthorized control and accidental real-money activation impossible through the existing entry points. This is slice 2 of the audit-remediation initiative. Its upstream requirements and decision IDs are authoritative; future slices must not be implemented early.

## Scope and implementation requirements

- Implement approved operator login, secure session, CSRF, throttling and account telemetry access restrictions.
- Separate immutable execution allow gate from paper/exchange venue config; partial updates/reset preserve safety fields; guarded mode transitions reject unresolved exposure.
- Remove scanner/order side effects from imports; harmonize documented .env/compatibility parsing, sandbox construction and production-secret/debug behavior.

## Affected components

Flask operator routes/session, shared environment/config parser, explicit launchers.

## Acceptance criteria

- [ ] Anonymous/unauthorized or invalid-CSRF trade, close, config and reset requests produce no order or state effects; account GET data requires auth.
- [ ] Omitted paper/mode/boolean fields retain prior values; reset cannot enable exchange execution or alter venue credentials.
- [ ] Server live gate defaults off and cannot be enabled through HTTP config; contradictory venue/credential settings fail before private calls.
- [ ] Normal imports and offline CI start no threads or exchange requests; explicit development and production launch paths have documented ownership.
- [ ] Production default/missing secret and debug/reloader startup are rejected; successful authenticated paper access remains available.

## Testing and validation

HTTP auth/CSRF/partial/reset tests; fake constructor/venue assertions; subprocess import and safe-launch checks; relevant lint/type/full suite.

Run focused behavioral checks during development, relevant type/static/lint/build checks and the full applicable suite at closure. Record any genuinely inapplicable check with reason. Perform code-review, fix actionable findings and commit only scoped completed work, preserving unrelated changes.

## Regression requirements

Preserve authenticated versions of existing routes and valid environment knobs; deprecated aliases need conflict rules and upgrade notes.

## Security requirements

Approved D01 auth, session cookie flags, throttling and redacted errors; trading-only permissions documented for later provider verification.

## Migration and rollout considerations

Existing persisted paper flag cannot override immutable gate. No live deployment here; operational disablement is documented, not silently applied.

## Completion evidence

Negative/positive HTTP outputs, no-side-effect import proof, environment matrix, auth review and scoped commit.

Record criterion-by-criterion evidence links, test outputs, review outcome, commit, environment/revision and remaining limitations. Do not mark complete when a required gate is unavailable or fails. Stop dependent work, retain open status, and record exact blocker and next allowed step.

## Out of scope

Durable financial store, actual venue/account transition with holdings, user registration/password-reset product.

## Execution record

Implementation: not started. Checks: not run for this ticket. Acceptance evidence: none. Review: test boundaries and breakdown approved by the user on 2026-10-02. Commit: none. Current blockers: incomplete prerequisite tickets and unresolved policy decisions above, where applicable. Publication is complete; no feature code changes are authorized by the current planning task. Next permitted implementation step requires explicit implementation instruction and satisfied blockers.

