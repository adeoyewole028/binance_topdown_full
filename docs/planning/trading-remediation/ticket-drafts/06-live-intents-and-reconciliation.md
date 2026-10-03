# BT06: Dispatch exchange orders with durable fill reconciliation

**GitHub issue:** https://github.com/adeoyewole028/binance_topdown_full/issues/8

## Parent

[Implementation specification #2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)


**Status:** local implementation and all six acceptance criteria complete; final verification and independent reviews passed. Closure publication/read-back pending.
**What to build:** Make each approved exchange intent traceable from submission through partial/confirmed fill and recovery from unknown outcomes.
**Blocked by:** [BT05 #7](https://github.com/adeoyewole028/binance_topdown_full/issues/7).
**Decision blockers:** none for the selected D05 policy; explicit runtime setup/provider acceptance remain required as applicable.
**Specification:** [#2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)
**Requirement coverage:** R03, R08, R10.

## Objective and context

Make each approved exchange intent traceable from submission through partial/confirmed fill and recovery from unknown outcomes. This is slice 6 of the audit-remediation initiative. Its upstream requirements and decision IDs are authoritative; future slices must not be implemented early.

## Scope and implementation requirements

- Map validated quantities/prices/notional to pinned CCXT precision and current Spot filters, enforce free assets and serial risk reservation.
- Persist client identity before dispatch; implement ambiguous-outcome lookup and deduplicated provider order/fill observations, startup/periodic reconciliation.
- Project actual fills, costs and fee assets into holdings and operation status; isolate venue/testnet reset and unexpected balance/order differences.

## Affected components

Shared intent executor, CCXT Spot adapter, provider observations/fills/reservations, operation status.

## Acceptance criteria

- [x] Ordinary CCXT market metadata without amount.step is supported; tick/lot/market-notional and available-fund fixtures reject invalid orders.
- [x] Repeated same operation and recovery after timeout-after-provider-success creates no second economic order in controlled fault tests.
- [x] Partially filled then canceled orders retain fills and release only confirmed unfilled reservations; out-of-order observations cannot erase fills.
- [x] Live-confirmed fills update durable positions/history/balances and API status using actual quantities/average/cost/fees.
- [x] Unknown dispatch/status or failed reconciliation blocks conflicting/new exposure until resolved; startup recovers pending intent before new execution.
- [x] Testnet reset/mainnet identity mismatch creates a review state without crossing namespaces.

## Testing and validation

Fake-provider HTTP-to-fill, timeout/crash-before-and-after-submit, delayed lookup/partial/cancel/out-of-order/duplicate fill cases; focused adapter contract tests. Real sandbox journey reserved for BT15.

Run focused behavioral checks during development, relevant type/static/lint/build checks and the full applicable suite at closure. Record any genuinely inapplicable check with reason. Perform code-review, fix actionable findings and commit only scoped completed work, preserving unrelated changes.

## Regression requirements

Keep paper orders offline; preserve historical provider observations and account namespace, never infer rejection from timeout.

## Security requirements

Trading-only configured credentials, no withdrawal calls, stable references and provider redaction; authenticated status.

## Migration and rollout considerations

Initialize live holdings from approved provider reconciliation, not legacy paper files; remain gated off by default.

## Completion evidence

Redacted order lifecycle/fill/reservation fault traces, schema constraints, pinned adapter metadata and scoped commit.

Record criterion-by-criterion evidence links, test outputs, review outcome, commit, environment/revision and remaining limitations. Do not mark complete when a required gate is unavailable or fails. Stop dependent work, retain open status, and record exact blocker and next allowed step.

## Out of scope

Provider protection/close coordination, live activation or automatic resubmission based only on immediate not-found.

## Execution record

Implementation: not started. Checks: not run for this ticket. Acceptance evidence: none. Review: test boundaries and breakdown approved by the user on 2026-10-02. Commit: none. Current blockers: incomplete prerequisite tickets and unresolved policy decisions above, where applicable. Publication is complete; no feature code changes are authorized by the current planning task. Next permitted implementation step requires explicit implementation instruction and satisfied blockers.


## D05 approved contract (2026-10-03)

APPROVED on 2026-10-03: explicit operator setup for all capital/risk/fee/slippage/cooldown/drawdown inputs before paper trading; deterministic full paper fills, quote-currency fees, weighted-average cost basis, adverse slippage and gap prices, stop first when both stop/profit targets touch one candle. Exchange-supported stop-market protection; refuse unsupported symbols; re-protect confirmed remaining holdings after partial closes; failed/uncertain protection blocks entries; automatic emergency liquidation and trailing remain disabled.
Implement only this ticket’s scope; this approval does not authorize future tickets early, account funding or live/provider/rollout actions.


## BT06 start (2026-10-03)

Live#8 OPEN, no superseding comments; native prerequisite#7 CLOSED verified. Pre-ticket fixed point `5c3ea9186400e1c8f5b8c19a9a0d63f6efa9a444`. D05 approved policy preserved; real sandbox validation remains BT15. All six criteria remain open; implement only exchange admission, durable dispatch/observations/reconciliation at the agreed authenticated HTTP/owned worker/fake provider seam. No real provider calls, credential reads, activation, source push or deployment. Next: inspect pinned CCXT/Spot contracts and implement the durable identity/fill tracer test-first.


## BT06 current validation (2026-10-03)

Sole active ticket; all six criteria remain open. Owned private adapter, reviewed capital sizing/risk, explicit operator setup and entry pause, durable dispatch/actual fill projections, account identity-first reconciliation, account-wide open orders, overlapping paginated trade history, third-asset fees, crash/write recovery and 15-second idle scheduling implemented. Two BT06 expected failures promoted to passing owned-provider HTTP tests. Latest focused broad run88passed/1paper-message regression, corrected and its focused check passed; later fault cases passed individually. Type35 and repository pyflakes pass. Initial full suite is running; independent Standards/Spec review and final corrected evidence remain required. [Acceptance record](../evidence/bt06-validation.md); source is uncommitted BT06 work against5c3ea91. No future ticket, real provider/funding/migration/trade/deploy/push. Next permitted step: complete current full suite, remaining venue/response fault checks, scoped pre-review commit, two-axis review/corrections and refreshed full evidence before closure.


## BT06 review snapshot (2026-10-03)

Source review commit998698be36430cabec2b5783d967c4228808bdbd. Full228cases228passed,0failures/skips/xfails,112.29seconds; strict type35, repository lint and24source compilation pass. Two BT06 expected failures retired with owned-provider HTTP proofs. Standards review:0actionable findings at998698b. Independent Spec review pending; source frozen during review. All criteria remain open. Review commands: git diff5c3ea9186400e1c8f5b8c19a9a0d63f6efa9a444...HEAD; commitsb229ee9/998698b. No real provider/funding/migration/trade/deployment/push or future-ticket implementation. Next permitted step: receive Spec findings, correct actionables and repeat affected/final gates before closure.


## Final acceptance (supersedes historical planning/progress records)

All six criteria complete at69716fb6bf89e014b2dbf402ca21d6dc86059285. Full229cases229passed,0failures/skips/xfails,124.02seconds; type35/lint/diff/24source compilation pass. Standards0/Spec0 after real FULL timestamp and sparse/delayed history corrections. Scoped source commits998698b/69716fb. [Criterion/review evidence](../evidence/bt06-validation.md), [machine evidence](../evidence/bt06-exchange-execution.json), [operator runbook](../../../exchange-execution.md). No real provider/funding/migration/trade/deploy/push. Actual provider/load acceptance remains BT15; protection stays BT07. Parent#2 remains open. Publish/close/read-back next before starting dependent work.
