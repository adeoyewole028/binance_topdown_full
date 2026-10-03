# BT06 validation and acceptance record

Status: all six local BT06 acceptance criteria complete at69716fb6bf89e014b2dbf402ca21d6dc86059285. Final229cases229passed,0failures/skips/xfails,124.02seconds; type35/lint/diff/24source compilation pass. Both independent reviews0remaining findings. Historical pre-review records below are superseded by this final acceptance. Provider/deployed acceptance remains BT15/BT16. Fixed point: `5c3ea9186400e1c8f5b8c19a9a0d63f6efa9a444`.

| Criterion | Current local evidence | Remaining gate |
| --- | --- | --- |
| Ordinary pinned precision / current filters / funds | Actual CCXT4.5.3 market parsing and amount precision with replaced raw endpoints; current lot/notional admission; server fixed/risk sizing and caps | Passed; final evidence and reviews below |
| Stable operation / timeout recovery | Retired duplicate-order expectation; success followed by timeout and delayed lookup; crash before/after transport; failed fill transaction recovery | Passed; final evidence and reviews below |
| Partial cancel / reservations / old observations | Partial-fill remaining reservation, confirmed cancellation, older NEW observation, pagination and conflicting duplicate economic identity | Passed; final evidence and reviews below |
| Actual fills / positions / balances / API | Retired visible-fill expectation; actual quote cost/average/quote fee, base fee sellable quantity, third-asset fee holding projection | Passed; final evidence and reviews below |
| Recovery before new exposure | Startup crash recovery, account-wide foreign open order, unowned trade history, failed persistence, entry pause with reduction allowed, 15-second idle scheduling | Passed; final evidence and reviews below |
| Reset / account / namespace identity | Provider balance reset retains confirmed facts; changed account identity before merge; unresolved other namespace blocks dispatch; explicit adapter venue host validation | Passed; final evidence and reviews below |

Tests use authenticated HTTP, temporary transactional state, kernel-held worker ownership and controlled public/private providers. Actual pinned CCXT parsing is tested with replaced raw transports. Socket/DNS, credential, file and thread guards remain active. No real private call, trade, funding, migration, deployment or push occurred. Real sandbox acceptance remains BT15.

Focused checks during implementation caught and corrected pre-dispatch budget reservations, definitive-rejection release, pinned CCXT Spot margin-cache initialization, 1000-trade pagination, duplicate economic identity, identity-before-merge, account-wide open orders, unexpected trade history, third-asset holdings, periodic polling and paper setup message regression. Initial full suite: 223 passed, no failures/skips/xfails, 104.85 seconds. Five later focused lot/notional/funds/venue/malformed-fill cases pass; final full evidence must include them. Type checking: 35 sources pass. Repository pyflakes: pass. Independent review: pending. [Provider contracts](bt06-provider-contracts.md); [operator runbook](../../../exchange-execution.md).


## Standards

Initial review at998698b:0actionable findings. Recheck at69716fb:0actionable findings. No documented-standard breaches or meaningful Fowler heuristic smells. The response normalization, exact baseline identity tracking and atomic dispatch-readiness changes remain within the approved seams and ticket scope. Both reviews were read-only with no credential/provider access.

## Spec

Initial review at998698b:2actionable findings, worstP1. R10 “Store partial fills, costs and fees independently” was violated by assuming updateTime in a raw FULL submission response; the actual endpoint supplies transactTime. R10 “trade history with overlap and deduplication, tolerates delayed/out-of-order observations” was violated by subtracting1000from sparse global trade IDs and accepting newly seen lower IDs as pre-baseline history.

Both were reproduced red and corrected green at69716fb. The pinned CCXT raw FULL fixture now matches the documented timestamp shape; the adapter normalizes it. History uses complete shared-budget paginated overlap and exact, separately persisted baseline IDs. A cursor without those IDs requires review. Dispatch changes account/owner readiness in the same transaction as the reservation, before transport. Independent Spec recheck:0remaining actionables;35exchange tests passed independently.

Review summary: Standards0remaining (worst:none); Spec0remaining (initial worst:P1, resolved). Final full-suite evidence and closure metadata remain pending.


## Final acceptance

All six criteria verified at69716fb. [Exact tested hashes and machine output](bt06-exchange-execution.json). Full229passed, no skips/xfails/failures,124.02s; strict type35, repository lint, git diff check and24source compilation pass. Standards0/Spec0 after both corrections; scoped source commits998698b/69716fb. No actual provider/funding/migration/trade/deploy/push. Real provider history volume/permissions/fill/protection acceptance remains BT15; complete overlap is budgeted and fails closed rather than truncating. Larger UX and browser acceptance remain BT12/BT13. Publish/close/read-back#8 next; parent#2 stays open.
