# BT07 acceptance record

Only BT07 #9 is active. Fixed point `000913cbc8c6860fba9750cf62764a79cf87adac`; start record f63d638. D05 is approved. This progress snapshot does not close the ticket. Full applicable checks, independent two-axis review and final exact-revision evidence remain required.

| Criterion | Implementation and offline seam evidence | Gate |
|---|---|---|
| Confirmed supported Spot protection | Raw OCO TAKE_PROFIT/STOP_LOSS, stable identities, list/both-leg queries, precision/capability preflight; unsupported refuses entry, pinned CCXT substituted transports | Pending final checks/review |
| Isolated simulation | Paired durable groups, partial close preserves original rules, adverse quote gap, completed-bar stop-first, terminal dedup; no private transport | Pending final checks/review |
| Failed/unknown retains entry and blocks entries | Lost-list response/query recovery, pre-engine rejection retains successful fill, durable alert and next action in authenticated status | Pending final checks/review |
| Serialized close/race/timeout | Full/partial cancel-fill race, distinct duplicate full closes, cancellation timeout across restart, changed client aliases use verified provider IDs | Pending final checks/review |
| Remaining holding reprotected | Verified pair replacement after partial close; confirmed partial entry protected while unfilled quote stays reserved; base fees reduce pair quantity | Pending final checks/review |
| Pause/stop preserves native protection | Stop records owner status without cancel; paused restart verifies existing group and allows reviewed reduction/reprotection | Pending final checks/review |

Focused initial tracer: 2 red assertions before implementation, then both pass. Subsequent exchange partial-close tracer red before implementation and green after. Broad accounting/protection intermediate88 cases87 pass/1 base-fee availability expectation updated to actual provider locked protection with explicit coverage assertions. Paper drawdown and weighted-average fixtures now set supported stop/target ranges before entry so their intended accounting observations remain independent of newly implemented automatic protection. Final checks supersede these intermediate outputs.

Required integration corrections within BT07: entry-rule quantity accounts for actual base fees; protective child intents are exposed under the group/legs rather than reordered canonical manual operations; healthy protection legs do not count as unresolved ordinary market orders; provider cancel aliases preserve verified provider-ID ownership; full-close dispatch records reconciled reduced quantity while preserving original reviewed intent. These are demonstrated protection/accounting seams, not unrelated completed-ticket rewrites.

Runbook: [position protection](../../../position-protection.md). Actual provider sandbox behavior is a BT15 gate; external alerts/deployment/mainnet remain later authorized gates. No real private provider call, funding, trade, migration, deployment or push occurred.

## Standards

Initial independent review at b90f30d:0hard violations/1optional P3 duplicated fresh group lookup. A centralized fresh group reader now handles missing state explicitly; no stale cache crosses provider observations. Correction review pending.

## Spec

Initial independent review at b90f30d:1P1, request-budget deferral incorrectly became transmitted uncertainty for placement and cancellation. Two independent guarded HTTP reproductions showed no calls and permanently blocked reductions, contrary to R03 risk reduction and R11 resolved/canceled remaining-quantity close. Root red checks reproduced the failed placement state and deferred cancel path. Dispatch markers/base reservations and canceling now commit only after budget admission; proven no-attempt states fail locally and allow reviewed reduction, while actual transmitted ambiguity remains unresolved. Two correction checks pass; a separate crash-before-admission proof and final independent recheck remain required.

Initial full suite245cases242passed/3failed at b90f30d (see initial machine artifact). Separate HTTP subprocess15s startup timeout reran alone successfully3.53s without changing its gate. Two old fixtures crossed newly active paper take-profit or used pre-contract creating state; fixture policy/state updated while preserving original accounting/coverage assertions. Paper maintenance explicitly ignores non-simulation/unmanaged groups. Final full run and re-review supersede initial output; ticket stays open meanwhile.

## Second review and correction

Spec re-review at a72da9f:0remaining actionables; independently reran3 budget/prepare recovery checks. Standards re-review: original optional duplication resolved;1P3 private-method test interceptor violated the agreed TDD seam. Replaced it with approved SQLite prepare-write fault injection and retained HTTP/provider-effect assertions; focused proof passes. No private collaborator mock remains.

A subsequent owned idle-callback integration check found that the15-second reconcile callback made process_pending return before reprotecting a terminal partial trigger remainder. Reproduced red (remaining0.7/groupclosed), corrected by servicing protection maintenance during idle without repeating ordinary reconciliation. Idle reprotection and the original approved15-second reconciliation check both pass. Final re-review/full suite required; no closure yet.
