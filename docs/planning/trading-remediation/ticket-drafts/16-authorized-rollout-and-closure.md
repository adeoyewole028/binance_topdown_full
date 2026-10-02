# BT16: Run authorized canary and close with production evidence

**GitHub issue:** https://github.com/adeoyewole028/binance_topdown_full/issues/18

## Parent

[Implementation specification #2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)


**Status:** open; approved breakdown; implementation not started. The ready-for-agent label is triage only: resolve the prerequisites and decision blockers before execution.
**What to build:** Activate only approved scope in a controlled environment and close only after real journeys and operational gates are satisfied.
**Blocked by:** [BT15 #17](https://github.com/adeoyewole028/binance_topdown_full/issues/17).
**Decision blockers:** D07.
**Specification:** [#2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)
**Requirement coverage:** R20, R19.

## Objective and context

Activate only approved scope in a controlled environment and close only after real journeys and operational gates are satisfied. This is slice 16 of the audit-remediation initiative. Its upstream requirements and decision IDs are authoritative; future slices must not be implemented early.

## Scope and implementation requirements

- Obtain explicit deployment/mainnet/symbol/amount/time budget and monitoring authorization; verify prerequisites and migration/restore snapshot.
- Deploy recorded revision, reconcile before enablement, run approved canary and observe provider/local outcomes and alert/health signals.
- Stop/rollback on defined triggers while retaining protective/reconciliation responsibility; update all ticket evidence and final operator handoff.

## Affected components

Approved hosting/account rollout, reconciliation/protection monitoring, operational handoff and tracker closure.

## Acceptance criteria

- [ ] Written rollout authorization and approved limits/environment/owner are recorded before activation; absent authorization keeps ticket blocked.
- [ ] Deployed revision/environment and migration/reconciliation evidence are recorded; first authorized journey proves actual fill/protection/close/local accounting.
- [ ] Monitoring covers approved duration with no unexplained financial differences/duplicate effects/unprotected exposure or unresolved critical errors.
- [ ] Rollback rehearsal or triggered rollback blocks new writes and retains provider protection/unknown-outcome recovery; never claims to reverse fills.
- [ ] All accepted scope and requirement evidence are closed, documentation/handoff current and traceability complete; exclusions require explicit scope change.
- [ ] If mainnet is excluded later, record approved exclusion and final paper/testnet boundary rather than claiming production proof.

## Testing and validation

Authorized canary real journey, metric/error observation and protection-preserving rollback/restore verification in approved environment.

Run focused behavioral checks during development, relevant type/static/lint/build checks and the full applicable suite at closure. Record any genuinely inapplicable check with reason. Perform code-review, fix actionable findings and commit only scoped completed work, preserving unrelated changes.

## Regression requirements

Preserve all verified operator/paper/sandbox contracts and deployed provider state; code rollback cannot revert economics.

## Security requirements

Mainnet/provider permissions and all secrets handled outside ticket text; activation approved independently of mutable UI.

## Migration and rollout considerations

Pause/backup/migrate/reconcile/canary/monitor; keep compatibility reader for rollback and never resume legacy JSON writers.

## Completion evidence

Authorization, deployed revision, redacted provider/local journey proof, monitoring/rollback/handoff and closure records.

Record criterion-by-criterion evidence links, test outputs, review outcome, commit, environment/revision and remaining limitations. Do not mark complete when a required gate is unavailable or fails. Stop dependent work, retain open status, and record exact blocker and next allowed step.

## Out of scope

Autonomous funding/trading outside budget, production claims without actual journey evidence.

## Execution record

Implementation: not started. Checks: not run for this ticket. Acceptance evidence: none. Review: test boundaries and breakdown approved by the user on 2026-10-02. Commit: none. Current blockers: incomplete prerequisite tickets and unresolved policy decisions above, where applicable. Publication is complete; no feature code changes are authorized by the current planning task. Next permitted implementation step requires explicit implementation instruction and satisfied blockers.

