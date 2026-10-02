# BT14: Package a supported deployment with recovery and CI gates

**GitHub issue:** https://github.com/adeoyewole028/binance_topdown_full/issues/16

## Parent

[Implementation specification #2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)


**Status:** open; approved breakdown; implementation not started. The ready-for-agent label is triage only: resolve the prerequisites and decision blockers before execution.
**What to build:** Produce a reproducible, continuously supervised release with tested restore and one verification-gated deployment owner.
**Blocked by:** [BT09 #11](https://github.com/adeoyewole028/binance_topdown_full/issues/11), [BT13 #15](https://github.com/adeoyewole028/binance_topdown_full/issues/15).
**Decision blockers:** D07
**Specification:** [#2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)
**Requirement coverage:** R18, R19, R05, R04.

## Objective and context

Produce a reproducible, continuously supervised release with tested restore and one verification-gated deployment owner. This is slice 14 of the audit-remediation initiative. Its upstream requirements and decision IDs are authoritative; future slices must not be implemented early.

## Scope and implementation requirements

- Pin tested runtime/dependencies; full offline suite/lint/type/build with side-effect-free imports; resolve scoped lint issues.
- Correct Render/topology configuration under approved budget and state storage, ensure sole scanner/executor ownership and readiness.
- Define redacted observability/alerts, secret/permissions/rotation docs, backup retention/RPO/RTO and fault-tested restore/runbooks; eliminate competing deploy triggers.

## Affected components

Dependency/runtime packaging, CI/release configuration, supervisor/store backup, logs/metrics/runbooks.

## Acceptance criteria

- [ ] Clean dependency installation yields tested versions and full applicable checks pass without external order/data side effects.
- [ ] Deployment manifest matches supported service/disk/store model; multiple HTTP workers do not start multiple scanner/executor owners.
- [ ] One CI-gated release path exists with migration/readiness/rollback checks; hosted configuration evidence is recorded when authorized.
- [ ] Backups restore complete journal/state within approved RPO/RTO; disk-full/corruption/restart/owner-loss drills do not resume blind execution.
- [ ] Operation-correlated redacted metrics/alerts expose stale scans, unknown outcomes, unprotected exposure and mismatches within approved intervals.
- [ ] Runbooks define kill, protection-preserving shutdown, reconcile, restore, upgrade and rollback; operator/destination/budget are recorded.

## Testing and validation

Clean install/full suite/package build, multi-process readiness, fault/backup/restore/rollback drills, manifest/release-path validation and secret-redaction checks.

Run focused behavioral checks during development, relevant type/static/lint/build checks and the full applicable suite at closure. Record any genuinely inapplicable check with reason. Perform code-review, fix actionable findings and commit only scoped completed work, preserving unrelated changes.

## Regression requirements

Maintain intended long-lived operator workflow and persistent export provenance; no healthy-homepage-only readiness claim.

## Security requirements

No credential values in config/report/logs; least-privilege provider permissions verified at BT15; no alert sending without authorization.

## Migration and rollout considerations

Deployment changes/configuration are executed only within granted rollout authorization; local manifest readiness can precede external changes.

## Completion evidence

Version/CI/build outputs, supported blueprint review, restore timing/integrity, runbook/alert matrix and scoped commit.

Record criterion-by-criterion evidence links, test outputs, review outcome, commit, environment/revision and remaining limitations. Do not mark complete when a required gate is unavailable or fails. Stop dependent work, retain open status, and record exact blocker and next allowed step.

## Out of scope

Spending/hosting activation without approval, customer custody/hyperscale infrastructure.

## Execution record

Implementation: not started. Checks: not run for this ticket. Acceptance evidence: none. Review: test boundaries and breakdown approved by the user on 2026-10-02. Commit: none. Current blockers: incomplete prerequisite tickets and unresolved policy decisions above, where applicable. Publication is complete; no feature code changes are authorized by the current planning task. Next permitted implementation step requires explicit implementation instruction and satisfied blockers.

## Approved decision update

APPROVED on 2026-10-02: one host with transactional SQLite and one supervised execution owner; hosting purchase/activation is not authorized
