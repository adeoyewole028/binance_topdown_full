# BT02: Secure control access and explicit environment transitions

**GitHub issue:** https://github.com/adeoyewole028/binance_topdown_full/issues/4

## Parent

[Implementation specification #2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)


**Status:** complete; all scoped acceptance and review gates passed. The ready-for-agent label is triage only: resolve the prerequisites and decision blockers before execution.
**What to build:** Make unauthorized control and accidental real-money activation impossible through the existing entry points.
**Blocked by:** [BT01 #3](https://github.com/adeoyewole028/binance_topdown_full/issues/3).
**Decision blockers:** None additional; D01 was approved by the user on 2026-10-02.
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

- [x] Anonymous/unauthorized or invalid-CSRF trade, close, config and reset requests produce no order or state effects; account GET data requires auth.
- [x] Omitted paper/mode/boolean fields retain prior values; reset cannot enable exchange execution or alter venue credentials.
- [x] Server live gate defaults off and cannot be enabled through HTTP config; contradictory venue/credential settings fail before private calls.
- [x] Normal imports and offline CI start no threads or exchange requests; explicit development and production launch paths have documented ownership.
- [x] Production default/missing secret and debug/reloader startup are rejected; successful authenticated paper access remains available.

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

Complete on 2026-10-03. Implementation commit 31173c7adff0794dcf807c7179cfe4e38e9236d4; review corrections 1a5b740. All five criteria mapped in docs/planning/trading-remediation/evidence/bt02-validation.md and sanitized bt02-security.json. Full offline suite: 80 cases, 73 passed, 7 strict known defects owned by later tickets; zero unexpected failures/skips. Strict mypy passed14 sources; repository Pyflakes passed; nine source modules compile without import. Independent Standards and Spec reviewers each found one P2, both reproduced and corrected (effective production reload rejection; configured reset defaults). Final rechecks: Standards0 and Spec0 actionable findings; Spec independently passed13 focused cases. Session revocation/expiry/rotation, shared throttle, CSRF/auth denial, immutable gates/venue conflicts and unmodified import/explicit launch checks passed. Existing authenticated paper and golden strategies preserved. No source push, real provider order, hosting activation or deployment; later financial/recovery/provider/rollout gates remain open. Next BT03 after closure verification; D02/D06 approved.


## Approved decision D01

The user selected one configured operator with a password, secure session and CSRF protection on 2026-10-02. Use password hashing and login throttling; no registration/password-reset product or upstream identity substitution. Prerequisite tickets and required verification still apply.
