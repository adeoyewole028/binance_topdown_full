# BT11: Report performance from full history and account equity

**GitHub issue:** https://github.com/adeoyewole028/binance_topdown_full/issues/13

## Parent

[Implementation specification #2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)


**Status:** complete locally; all five acceptance gates passed; GitHub closure/read-back pending. The ready-for-agent label is triage only: resolve the prerequisites and decision blockers before execution.
**What to build:** Let the operator trust metrics across refresh, history pagination and restart, with incomplete data clearly labelled.
**Blocked by:** [BT10 #12](https://github.com/adeoyewole028/binance_topdown_full/issues/12).
**Decision blockers:** none for the selected D05 policy; explicit runtime setup/provider acceptance remain required as applicable.
**Specification:** [#2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)
**Requirement coverage:** R15, R05.

## Objective and context

Let the operator trust metrics across refresh, history pagination and restart, with incomplete data clearly labelled. This is slice 11 of the audit-remediation initiative. Its upstream requirements and decision IDs are authoritative; future slices must not be implemented early.

## Scope and implementation requirements

- Reuse approved replay cost-basis/fee/win-rate definitions for durable paper and exchange projections.
- Compute actual asset-qualified equity/peak drawdown with price and conversion freshness; retain realized/unrealized separation.
- Mark imported unknown basis/fees and missing conversions unavailable or partial rather than inventing totals.

## Affected components

Performance projection service, durable fills/equity snapshots, status API and replay summary.

## Acceptance criteria

- [x] More than 20 closed trades, pagination, restart and rebuilding projections retain identical full-history totals.
- [x] Drawdown uses positive seeded account equity and handles an initial loss correctly; win-rate denominator follows selected method.
- [x] Partial fills/multi-asset fees are accounted once; unconvertible fee/basis or stale prices are visible as incomplete metrics.
- [x] Displayed estimated/unrealized figures are distinguishable from confirmed realized values and currency is explicit.
- [x] No mutable balance editing silently rewrites journal history; rebuild and snapshot consistency checks pass.

## Testing and validation

Deterministic known-capital/win-loss fixtures beyond display cap, reconstruction/restart, multi-asset fee/partial close and missing-history/conversion tests.

Run focused behavioral checks during development, relevant type/static/lint/build checks and the full applicable suite at closure. Record any genuinely inapplicable check with reason. Perform code-review, fix actionable findings and commit only scoped completed work, preserving unrelated changes.

## Regression requirements

Preserve performance display intent, remove false profit-relative drawdown and retained-history truncation.

## Security requirements

Authenticated projections only; sanitized research/sample account data.

## Migration and rollout considerations

Legacy metrics remain unavailable/partial until provenance exists; no invented capital or complete history.

## Completion evidence

Expected versus observed metrics, projection rebuild parity and scoped commit.

Record criterion-by-criterion evidence links, test outputs, review outcome, commit, environment/revision and remaining limitations. Do not mark complete when a required gate is unavailable or fails. Stop dependent work, retain open status, and record exact blocker and next allowed step.

## Out of scope

New risk limits, tax reporting or return promises.

## Execution record

Implementation source24115d1/58ecb4a/2d8b08c, fixedpoint3e47b2f3bcdecc5044c85b27950288a139de9654. Full344pass/0failure/skips/xfails520.07s; type50/repository lint/diff/32source compilation pass. Standards0hard/0heuristic; Spec0remaining after both reproducedP2s corrected. Allfive criteria passed. [Criterion-by-criterion acceptance/review evidence](../evidence/bt11-validation.md), [machine result/source hashes](../evidence/bt11-account-performance.json), [contract](../../../account-performance.md). Confirmed accounting preserved; stale/unresolved account valuations and missing intermediate historical equity are unavailable. Initial342pass and intermediate342pass/2process-timeout snapshots retained; unchanged timeout checks passed8.66s and final fullsuite passed. No schema migration, realprovider/credentials/funding/trade/activation/deploy/push; browser/provider/rollout gates stay later tickets. Publish/close/read-back#13 then inspectBT12#14 before starting; parent#2 remainsOPEN.

## D05 approved contract (2026-10-03)

APPROVED on 2026-10-03: explicit operator setup for all capital/risk/fee/slippage/cooldown/drawdown inputs before paper trading; deterministic full paper fills, quote-currency fees, weighted-average cost basis, adverse slippage and gap prices, stop first when both stop/profit targets touch one candle. Exchange-supported stop-market protection; refuse unsupported symbols; re-protect confirmed remaining holdings after partial closes; failed/uncertain protection blocks entries; automatic emergency liquidation and trailing remain disabled.
Implement only this ticket’s scope; this approval does not authorize future tickets early, account funding or live/provider/rollout actions.


## BT11 start (2026-10-03)

Live#13OPEN/0comments/native#12CLOSED verified; parent#2OPEN. Fixedpoint3e47b2f3bcdecc5044c85b27950288a139de9654. D05 fee-inclusive weighted basis/fullposition-exit denominator shared with replay; unknown imported basis/conversions remainpartial/unavailable. Existing authenticatedHTTP→ownedworker→fakepublic/privateprovider→HTTP and pure evaluator/portfolio projection seams apply. Next:fullhistory>20trades reconstruction/restart tracer, then asset-qualified fresh equity/fees/availability and shared definitions. No futureticket or actualprovider/funding/trade/activation/deployment/push.
