# BT06 validation and acceptance record

Status: in progress. All six acceptance criteria remain open until full validation, independent Standards/Spec reviews, actionable corrections and final evidence are complete. Fixed point: `5c3ea9186400e1c8f5b8c19a9a0d63f6efa9a444`.

| Criterion | Current local evidence | Remaining gate |
| --- | --- | --- |
| Ordinary pinned precision / current filters / funds | Actual CCXT4.5.3 market parsing and amount precision with replaced raw endpoints; current lot/notional admission; server fixed/risk sizing and caps | Final full checks and reviews |
| Stable operation / timeout recovery | Retired duplicate-order expectation; success followed by timeout and delayed lookup; crash before/after transport; failed fill transaction recovery | Final full checks and reviews |
| Partial cancel / reservations / old observations | Partial-fill remaining reservation, confirmed cancellation, older NEW observation, pagination and conflicting duplicate economic identity | Final full checks and reviews |
| Actual fills / positions / balances / API | Retired visible-fill expectation; actual quote cost/average/quote fee, base fee sellable quantity, third-asset fee holding projection | Final full checks and reviews |
| Recovery before new exposure | Startup crash recovery, account-wide foreign open order, unowned trade history, failed persistence, entry pause with reduction allowed, 15-second idle scheduling | Final full checks and reviews |
| Reset / account / namespace identity | Provider balance reset retains confirmed facts; changed account identity before merge; unresolved other namespace blocks dispatch; explicit adapter venue host validation | Final full checks and reviews |

Tests use authenticated HTTP, temporary transactional state, kernel-held worker ownership and controlled public/private providers. Actual pinned CCXT parsing is tested with replaced raw transports. Socket/DNS, credential, file and thread guards remain active. No real private call, trade, funding, migration, deployment or push occurred. Real sandbox acceptance remains BT15.

Focused checks during implementation caught and corrected pre-dispatch budget reservations, definitive-rejection release, pinned CCXT Spot margin-cache initialization, 1000-trade pagination, duplicate economic identity, identity-before-merge, account-wide open orders, unexpected trade history, third-asset holdings, periodic polling and paper setup message regression. Initial full suite: 223 passed, no failures/skips/xfails, 104.85 seconds. Five later focused lot/notional/funds/venue/malformed-fill cases pass; final full evidence must include them. Type checking: 35 sources pass. Repository pyflakes: pass. Independent review: pending. [Provider contracts](bt06-provider-contracts.md); [operator runbook](../../../exchange-execution.md).
