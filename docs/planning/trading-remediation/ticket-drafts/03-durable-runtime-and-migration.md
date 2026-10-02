# BT03: Recover isolated state under one execution owner

**GitHub issue:** https://github.com/adeoyewole028/binance_topdown_full/issues/5

## Parent

[Implementation specification #2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)


**Status:** open; approved breakdown; implementation not started. The ready-for-agent label is triage only: resolve the prerequisites and decision blockers before execution.
**What to build:** Give dashboard workers and the daemon one consistent durable state namespace that survives restart without duplicate ownership.
**Blocked by:** [BT02 #4](https://github.com/adeoyewole028/binance_topdown_full/issues/4).
**Decision blockers:** None additional; D02 and D06 were selected by the user.
**Specification:** [#2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)
**Requirement coverage:** R03, R04, R05.

## Objective and context

Give dashboard workers and the daemon one consistent durable state namespace that survives restart without duplicate ownership. This is slice 3 of the audit-remediation initiative. Its upstream requirements and decision IDs are authoritative; future slices must not be implemented early.

## Scope and implementation requirements

- Implement approved transactional store/schema and account/venue/mode namespacing with intent/fill/config/protection/audit identities and constraints.
- Make one supervised owner authoritative for order work; workers read durable snapshots and enqueue only, with safe handover of unresolved work.
- Create repeatable backup/checksum/import/quarantine for legacy positions/config/CSV; block execution on corruption/write failure; retain rollback-compatible journals.

## Affected components

Runtime supervision/ownership, transactional store, migration/import, config/operation/status reads.

## Acceptance criteria

- [ ] Two HTTP workers read the same committed state; second execution owner cannot dispatch while first owns account.
- [ ] Config/intent/projection updates are atomic; store unavailability or ownership uncertainty blocks new dispatch without losing recoverable intent.
- [ ] Legacy repeat import has no duplicate effects; confirmed paper quantities and quarantine counts match approved manifest.
- [ ] Unknown legacy origin never becomes mainnet holdings; paper/testnet/mainnet records cannot be queried/mutated across namespace.
- [ ] Restart restores pending operations, positions/config and ownership state; corrupted or absent required state never yields a tradable empty account.
- [ ] Rollback/restore preserves journal effects and forbids old JSON writers after cutover.

## Testing and validation

Transactional/concurrent-worker/owner-takeover integration tests; repeat/corrupt migration fixtures; crash/disk-full/write-failure and rollback simulation.

Run focused behavioral checks during development, relevant type/static/lint/build checks and the full applicable suite at closure. Record any genuinely inapplicable check with reason. Perform code-review, fix actionable findings and commit only scoped completed work, preserving unrelated changes.

## Regression requirements

Preserve valid legacy paper quantities through approved import; no invented fills, fees or complete-history metrics.

## Security requirements

Credentials stay out of store; namespace authorization and immutable provenance tested.

## Migration and rollout considerations

Pause -> backup/checksum -> schema/import -> integrity -> acquire/reconcile -> approved resume; no real migration during planning.

## Completion evidence

Schema/invariant checks, import manifest, concurrency/restart/rollback evidence, reviewed operational topology and scoped commit.

Record criterion-by-criterion evidence links, test outputs, review outcome, commit, environment/revision and remaining limitations. Do not mark complete when a required gate is unavailable or fails. Stop dependent work, retain open status, and record exact blocker and next allowed step.

## Out of scope

Mainnet reconciliation implementation, distributed platform or new custody model.

## Execution record

Implementation: not started. Checks: not run for this ticket. Acceptance evidence: none. Review: test boundaries and breakdown approved by the user on 2026-10-02. Commit: none. Current blockers: incomplete prerequisite tickets and unresolved policy decisions above, where applicable. Publication is complete; no feature code changes are authorized by the current planning task. Next permitted implementation step requires explicit implementation instruction and satisfied blockers.


## Approved decision update

APPROVED on 2026-10-02: one host with transactional SQLite and one supervised execution owner; hosting purchase/activation is not authorized

## Approved decision D06

APPROVED on 2026-10-03: quarantine all legacy snapshots read-only until the operator confirms each paper-account origin; exchange holdings always come from provider reconciliation
