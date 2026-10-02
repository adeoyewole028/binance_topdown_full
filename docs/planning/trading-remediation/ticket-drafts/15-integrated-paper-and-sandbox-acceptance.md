# BT15: Prove integrated paper and Spot testnet acceptance

**GitHub issue:** https://github.com/adeoyewole028/binance_topdown_full/issues/17

## Parent

[Implementation specification #2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)


**Status:** open; approved breakdown; implementation not started. The ready-for-agent label is triage only: resolve the prerequisites and decision blockers before execution.
**What to build:** Collect the independent evidence needed to distinguish local correctness from deployed paper and real sandbox financial behavior.
**Blocked by:** [BT14 #16](https://github.com/adeoyewole028/binance_topdown_full/issues/16).
**Decision blockers:** D07.
**Specification:** [#2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)
**Requirement coverage:** R01, R02, R03, R04, R05, R06, R07, R08, R09, R10, R11, R12, R13, R14, R15, R16, R17, R18, R19, R20.

## Objective and context

Collect the independent evidence needed to distinguish local correctness from deployed paper and real sandbox financial behavior. This is slice 15 of the audit-remediation initiative. Its upstream requirements and decision IDs are authoritative; future slices must not be implemented early.

## Scope and implementation requirements

- Execute full offline suite and actual deployed isolated paper/operator/security/accessibility journeys at recorded revision.
- Use separately authorized testnet credentials for real sandbox buy/fill/protection/cancel/close/status reconciliation, including controlled failure paths.
- Run process restart/state restore/testnet-reset recovery and health/permissions/budget checks; resolve findings before canary.

## Affected components

Integrated deployed paper/testnet journeys, provider adapter/protection, runtime ownership/restore, browser/operator handoff.

## Acceptance criteria

- [ ] Full suite and requirement matrix have linked passing evidence; unresolved defects retain this ticket open.
- [ ] Deployed paper login/config/order/exit/metrics and browser recovery match spec under multiple HTTP workers.
- [ ] Sandbox actual orders/fills/list protection/close query evidence matches durable local holdings; test-order validation alone is insufficient.
- [ ] Restart/ambiguous response/partial protection/duplicate intent recovery and testnet-reset cases preserve isolation and no duplicate exposure.
- [ ] Provider metadata/adapter version, trading-only/no-withdrawal permissions, configuration/monitoring and restore evidence are recorded without secrets.
- [ ] No mainnet call or activation occurs; operator accepts sandbox handoff and remaining production gate.

## Testing and validation

Full integrated security/regression/browser/multi-process/restore suites plus actual Spot testnet lifecycle; controlled failure injection must not conceal simulated versus provider evidence.

Run focused behavioral checks during development, relevant type/static/lint/build checks and the full applicable suite at closure. Record any genuinely inapplicable check with reason. Perform code-review, fix actionable findings and commit only scoped completed work, preserving unrelated changes.

## Regression requirements

Repeat key existing authenticated manual/automation/paper journeys after integration; reopen only demonstrated regressions.

## Security requirements

Sandbox account and controlled credentials/budget required; provider least privilege and audit redaction verified; never publish sensitive payloads.

## Migration and rollout considerations

Use disposable or backed-up isolated environment; prove approved migration/restore before any production cutover.

## Completion evidence

Revision/environment/version manifest, per-requirement evidence links, redacted provider IDs/outcomes, browser/restore records and acceptance signoff.

Record criterion-by-criterion evidence links, test outputs, review outcome, commit, environment/revision and remaining limitations. Do not mark complete when a required gate is unavailable or fails. Stop dependent work, retain open status, and record exact blocker and next allowed step.

## Out of scope

Mainnet rollout, claims that mocks/HTTP 200 certify provider operation.

## Execution record

Implementation: not started. Checks: not run for this ticket. Acceptance evidence: none. Review: test boundaries and breakdown approved by the user on 2026-10-02. Commit: none. Current blockers: incomplete prerequisite tickets and unresolved policy decisions above, where applicable. Publication is complete; no feature code changes are authorized by the current planning task. Next permitted implementation step requires explicit implementation instruction and satisfied blockers.

