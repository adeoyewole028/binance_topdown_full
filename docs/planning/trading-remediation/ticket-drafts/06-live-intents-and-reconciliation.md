# BT06: Dispatch exchange orders with durable fill reconciliation

**GitHub issue:** https://github.com/adeoyewole028/binance_topdown_full/issues/8

## Parent

[Implementation specification #2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)


**Status:** open; approved breakdown; implementation not started. The ready-for-agent label is triage only: resolve the prerequisites and decision blockers before execution.
**What to build:** Make each approved exchange intent traceable from submission through partial/confirmed fill and recovery from unknown outcomes.
**Blocked by:** [BT05 #7](https://github.com/adeoyewole028/binance_topdown_full/issues/7).
**Decision blockers:** D05.
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

- [ ] Ordinary CCXT market metadata without amount.step is supported; tick/lot/market-notional and available-fund fixtures reject invalid orders.
- [ ] Repeated same operation and recovery after timeout-after-provider-success creates no second economic order in controlled fault tests.
- [ ] Partially filled then canceled orders retain fills and release only confirmed unfilled reservations; out-of-order observations cannot erase fills.
- [ ] Live-confirmed fills update durable positions/history/balances and API status using actual quantities/average/cost/fees.
- [ ] Unknown dispatch/status or failed reconciliation blocks conflicting/new exposure until resolved; startup recovers pending intent before new execution.
- [ ] Testnet reset/mainnet identity mismatch creates a review state without crossing namespaces.

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

