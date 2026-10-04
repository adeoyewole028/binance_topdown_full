# BT14: Package a supported deployment with recovery and CI gates

**GitHub issue:** https://github.com/adeoyewole028/binance_topdown_full/issues/16

## Parent

[Implementation specification #2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)


**Status:** open; BT14 is the sole active ticket; local operations preparation in progress. The ready-for-agent label is triage only: resolve the prerequisites and decision blockers before execution.
**What to build:** Produce a reproducible, continuously supervised release with tested restore and one verification-gated deployment owner.
**Blocked by:** [BT09 #11](https://github.com/adeoyewole028/binance_topdown_full/issues/11), [BT13 #15](https://github.com/adeoyewole028/binance_topdown_full/issues/15).
**Decision blockers:** D07 numerical targets/alert ownership approved; actual hosting budget/access, alert destination and deployment authorization remain external gates.
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

Implementation started 2026-10-03 after live issue/comments and native BT09#11/BT13#15 closure verification. Clean fixed point: 4619f3d9227a1a496c635c48138185e49d5b0600. All six acceptance gates remain open. Strict sequential implementation authorization carried forward. Local packaging, supervision, readiness, backup/restore drills and release preparation are authorized. Actual host budget/access, private alert destination and deployment/activation remain external gates. No source push, hosting purchase, real notification or provider action. Next: supported topology and offline recovery preparation, then focused/full validation and two-axis review before closure.

## Local operations checkpoint (2026-10-04)

Current source12600a641c96bbbeb7eea45d38fa2302c7056451, committed locally on main; source has not been pushed. Both proposed D05 rules remain approved and preserved. Local single-host supervisor/readiness, verified backup/retention, durable redacted incident outbox, exact-revision release guard and runbook prepared. Clean isolated Windows Python3.11.17 installation has46 compatible pinned packages. Focused operations/release25passed66.23s; expanded safe imports passed; strict63-source types, project lint/68-file compile and CSS build passed. Independent final Standards0 actionable/Spec0 remaining.

Three full396-test Windows attempts exceeded the unchanged600s aggregate budget; no full-suite acceptance. Earlier repeated marker at the separate-HTTP case passed in isolation and in the final diagnostic, but no complete report establishes the original cause. No individual deadline relaxed. Linux clean CI/browser/process-group/durability proof remains required. Actual host/monthly cap/capacity, private alert destination and authorized delivery, independent off-host recovery, initial mounted-disk provisioning, protected release configuration/legacy trigger retirement and rollout authorization remain unresolved. All six criteria remain unchecked; BT14 sole active, BT15/BT16 unstarted. No provider/notification/trade/spend/deployment action.

Durable local evidence (not source-published): docs/planning/trading-remediation/evidence/bt14-validation.md, bt14-runtime.json, bt14-focused.json, bt14-intermediate-timeout.json, bt14-serial-timeout.json and bt14-diagnostic-timeout.json; docs/operations-runbook.md. Budget/intended-host question pending. Next: required authorized Linux/hosted verification and secure external configuration; keep this issue OPEN until every gate passes.

## Approved decision update

APPROVED on 2026-10-02: one host with transactional SQLite and one supervised execution owner; hosting purchase/activation is not authorized


## Approved D07 operating targets

APPROVED2026-10-03: scans300s, scan-health600s, price-refresh30s/expiry60s, reconciliation15s, backups7days/RPO15min/RTO1hour; configured operator owns private webhook alerts configured before activation. Hosting spend, deployment/provider access, actual alert destination and live-canary budget/duration remain separate gates.
