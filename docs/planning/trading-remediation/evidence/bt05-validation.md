# BT05 validation in progress

Sole active ticket: [BT05 #7](https://github.com/adeoyewole028/binance_topdown_full/issues/7). Review fixed point `2b77b6dbb786fbd6e7162b84d4c4e0c1d7b8d6bf`. D05 selections are approved; no runtime account was funded outside temporary fixtures.

Initial full suite:175 cases,173 passes,2 strict future BT06 expected failures,zero other failures/skips in71.49s. [Machine evidence](bt05-paper-execution.json) records exact tested working-source hashes and pre-commit revision2259fc0. It precedes additional setup/security/fault and equal-seed-spelling tests/canonicalization; final full evidence must be refreshed after review. Latest focused40 paper cases passed, plus one equal-seed-spelling regression; mypy32 sources, repository Pyflakes and22-source compilation without imports passed. All acceptance boxes remain open pending independent review/final evidence.

| Acceptance criterion | Evidence at the approved seam |
|---|---|
| Invalid inputs have no economic effect | Explicit setup/invalid quantity matrix; canonical side/symbol guard; fresh reviewed quote expiration, changed-price rejection, provider lot/notional constraints, allocation/exposure/risk caps; no phantom seed. |
| Idempotency and reservations | Canonical same-key requests/semantic409, retry close-all after holding gone, simultaneous authenticated unique/repeated requests, pending quote/holding and distinct slot reservations, cancel/reject releases. |
| Atomic fees/basis/cash and durability | Worked102.01 buy/quote-fee case, multi-entry153.015 average basis and partial-sale889.99 cash, observable fills; restart cooldown/reduction; BT03 full-disk regression now verifies cash/reservation/fill invariants; failed-seed-commit rollback. |
| Server sizing and explicit seed | Required setup inputs, one immutable seed, equal decimal spelling/reseed refusal; fixed/risk sizing uses funded cash/equity and rounds down to metadata precision; equity_proxy cannot authorize spend. |
| Truthful outcomes and reduction | Authenticated HTML setup/order states; rejection message instead of queued-success claim, durable drawdown entry stop and manual reduction, cooldown exemption for reductions; atomic paper cancellation of unfilled requests only. |

Shared provider filter input support is necessary BT05 integration; urgent weight/raw declarations now include reference/average fetches. The existing market budget cap test correspondingly leaves one research request under its16-request fixture. Prior queue-order fixture now expects sells against unfilled buys to reject; input ordering remains asserted. Exact-decimal close uses provider matching18-place precision. Expired-review test now requires a newly reviewed request after quote refresh; pending coverage uses a legitimate funded-quote symbol. These are documented corrections to earlier assumptions under approved R08, not loss of ownership, coverage or decimal guarantees.

No private exchange dispatch/protection/replay/automation was implemented. Tests remain disposable/fake-provider with socket/DNS/write guards; explicit concurrent test-client threads are the only thread-start exception. HTTP rendering/labels/native controls are checked; live browser, assistive technology, device, provider sandbox and deployment acceptance are not claimed. Those later gates remain open. Flask has no separate frontend build; source compilation and authenticated renders are relevant local checks.


## First independent review and corrections

Standards: zero hard/actionable violations; one optional duplicated-code observation in reference/average refresh logic, resolved with a shared per-symbol observation helper preserving each lane and timestamp. Spec: three actionable findings (P1 submitted review binding, P1 lost tiny fees, P2 ignored MAX_POSITION). Authenticated regression cases first failed, then passed after signed review admission, deterministic financial arithmetic/exact serialization, and confirmed-plus-reserved base position limits. Broad preliminary111 paper/market/durable cases passed48.78s; this preceded final helper/scale corrections and is not closure evidence. Final reviewed source, full suite and review read-back still pending.


Re-review found and fixed a P2 extreme-zero intermediate formatting allocation (zero returns canonical0 before fixed formatting) and a P1 post-fill drawdown latch gap (fee/slippage-induced crossings now persist a risk checkpoint/entry stop within the fill transaction). The fee-crossing/rebound HTTP case failed before correction, then passed. The extreme-zero HTTP case passed without dangerous pre-fix allocation. The full pre-correction run exceeded the runner's120-second bound at roughly75%; it produced no final evidence and is not counted as passing. The growing authenticated suite now has a bounded300-second runner allowance; final elapsed/output remains required.
