# BT03 validation and handoff

Date 2026-10-03. Ticket [BT03 #5](https://github.com/adeoyewole028/binance_topdown_full/issues/5), prerequisite BT02 closure verified, fixed point `dd950b8f01565eba61b5e79e9b5beb0d2484b338`. User-selected D02/D06 are authoritative: one host/transactional SQLite/single owner; every legacy snapshot quarantined until its paper origin is confirmed. No additional comments superseded these decisions.

## Criterion evidence

| Criterion | Observable proof |
|---|---|
| Shared committed HTTP state; exclusive owner | Two independently loaded HTTP workers see config/fills; a separate HTTP subprocess reads quantity `0.125` and updates shared risk to `0.025`. A real owner subprocess excludes a contender, is forcibly killed, and its replacement processes the queued quantity once at generation2. Request-local SQLite read snapshots prevent mixed observations during an interleaved fill. |
| Atomic updates and fail-closed uncertainty | Config commit failure retains prior config. Fill commit failure leaves no position and one queued intent; restart consumes once. Stable operation/config/journal references are transactional. Changed ownership token/generation refuses work; unresolved prior dispatch keeps replacement recovering and leaves new intent queued. |
| Repeat import and manifest agreement | Fixture BTC quantity `0.125` imports once after exact namespace/checksum attestation; one negative ETH row stays quarantined. Repeating attestation/quarantine does not duplicate the opening. Duplicate BTC rows and NaN ETH produce 0 imports/3 quarantines. Raw backup checksum is verified before import. Counts and provenance are exposed through authenticated status; no invented order/fill/capital/fee event. |
| Origin and namespace isolation | Before attestation, all snapshots are read-only quarantine with zero position effects. Incorrect/general attestation and exchange-mode attestation refuse import. Testnet/mainnet namespaces share a physical DB while retaining distinct config/positions; forged HTTP namespace does not affect another. Mode is immutable and server-selected. |
| Restart, pending state and corruption | Existing paper/config reload regressions pass; queued operations and persisted quotes survive restart. Abrupt owner death recovers under a new generation; unknown prior provider outcome blocks new effects. Missing/corrupted DB returns unavailable and refuses intake/ownership rather than creating an empty account. |
| Journal-preserving rollback/restore; JSON cutover | Online-backup candidate recovers quantity2 and filled intent. After newer partial close to quantity1, older backup restoration refuses and retains current DB. Corrupt DB/checksum or inflated manifest high-water refuses candidates. Source snapshots remain unchanged/read-only; old dashboard JSON writers are removed and standalone daemon main cannot execute independently. No restore overwrites authoritative state. Compatible readers retain newer journals. |

Tests are at the approved operator/state/owner/provider seam. Disposable SQLite DB-API fault injection models disk-full and read interleaving; assertions observe HTTP behavior. Subprocesses use disposable paths, cleared real credentials, no dotenv/provider construction; the HTTP subprocess uses the same dummy operator hash to preserve the genuine multi-worker security contract. Real provider outcomes/reconciliation remain BT06/BT15 proof layers.

## TDD and verification

Red-to-green tracers established stale second-worker config, unavailable state/failed commits, queued HTTP intake, owned paper projection, exact fractional close, FIFO intent ordering, quarantine/import, online restore, corrupted restore metadata and mixed-commit HTTP status. Three later BT05 baseline regressions became required projection preconditions here (negative buy/sell and unpriced fill); they are ordinary passing regressions. This does not satisfy BT05 capital, risk, fees, slippage, reservations or protection policy and does not select D05 values.

Full applicable suite at final implementation revision `b60b4b2e3f23c5909051ee4f2dfe269977da0bb2`: **101 cases, 97 passed, 4 strict future xfails, zero unexpected failures/skips**, 28.94s. Machine evidence: [bt03-durable-state.json](bt03-durable-state.json). The four known failures retain their BT04 (equity save/negative risk) and BT06 (semantic dedup/exchange fill visibility) owners. Strict mypy: **23 sources pass**. Repository Pyflakes and diff checks pass. **15 application sources compile** without imports, the applicable Python build check. Golden strategy fixture hash is unchanged. Focused operator/control/durable checks:43pass; Spec independently checked the three final correction scenarios.

Scoped implementation/review-correction commits: `96b3887`, `a1f13a7`, `b60b4b2`. No unrelated files were included. Final evidence/bookkeeping commit does not modify application source.

## Standards

Final **0 actionable findings; worst severity none**. Optional duplicated ownership comparison and display conversion were centralized during review. Ownership still checks the token/generation inside each write transaction, and also verifies the held kernel handle against the lock pathname. Read helpers reuse one connection without implicit shared request state. Required closure evidence is recorded here.

## Spec

Initial P2 restore metadata finding reproduced: changing only manifest high-water could allow a stale candidate. Correction verifies actual backup DB identity/schema/integrity/journal high-water against the manifest before current-state comparison; regression now rejects metadata corruption.

Second P2 HTTP finding reproduced: a fill between separate reads produced empty display positions alongside a canonical holding and filled operation. Correction reads config/scan/positions/operations/migration/ownership in one explicit transaction and retains it in Flask request-local state; HTML and API use the same observation. Mutation routes do not replace shared config globals. Interleaving regression passes. Final **0 remaining actionable findings; worst issue none**, with independent focused correction verification. No premature later-ticket policies.

## Operational boundary and next step

[Durable-state runbook](../../../durable-state.md) covers explicit init, stopped-writer quarantine, namespace-bound confirmation, manifest/integrity review, online backup, isolated restore candidates and compatible-reader rollback. Config/CSV remain review-only inputs; source files are retained. The owner begins recovering and never infers a free lock means an earlier provider call had no effect. Schema includes later lifecycle identities without activating their logic. Scanner/owner are separately launched; old same-process HTTP scanner and independent daemon dispatch paths are retired.

All checks used disposable fixtures. No existing account files were migrated, no real exchange or hosting state was changed, and no source push, deployment, production activation or spending occurred. Linux host/supervisor installation, restore schedules/RPO/RTO, actual provider reconciliation and deployed acceptance remain BT14/BT15; exchange dispatch stays disabled in this foundation. Interim `legacy-paper-v1` is explicitly incomplete in capital/fees/history.

BT03 #5 CLOSED/read-back verified after evidence commit91ccf89 and tracker update; specification #2 remains OPEN. Next sole ticket is BT04 #6, whose D07 operating targets remain pending the existing user question; do not implement it until those values are selected and recorded. D05 also remains unresolved. The overall initiative stays open.
