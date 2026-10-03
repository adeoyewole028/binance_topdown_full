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
