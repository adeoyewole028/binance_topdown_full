# BT04: Expose fresh market data and supervised scan health

**GitHub issue:** https://github.com/adeoyewole028/binance_topdown_full/issues/6

## Parent

[Implementation specification #2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)


**Status:** open; approved breakdown; implementation not started. The ready-for-agent label is triage only: resolve the prerequisites and decision blockers before execution.
**What to build:** Let the operator trust scan age and held-position data, with stale or failed inputs preventing execution.
**Blocked by:** [BT03 #5](https://github.com/adeoyewole028/binance_topdown_full/issues/5).
**Decision blockers:** D07.
**Specification:** [#2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)
**Requirement coverage:** R06, R07, R19.

## Objective and context

Let the operator trust scan age and held-position data, with stale or failed inputs preventing execution. This is slice 4 of the audit-remediation initiative. Its upstream requirements and decision IDs are authoritative; future slices must not be implemented early.

## Scope and implementation requirements

- Store closed-bar provenance, fetched-at timestamps, warmup/gap validation and held/pending/protected market coverage.
- Coordinate request budgets across market, reconciliation and research lanes; classify failure and respect provider-directed backoff; risk-reduction traffic has priority.
- Supervise scanner and publish heartbeat, last completed scan, degraded reason and config version; validate/version all settings atomically.

## Affected components

Scanner/market-data adapter, universe and price/candle cache, durable health/status.

## Acceptance criteria

- [ ] Unfinished/duplicate/malformed/gapped/insufficient candles cannot yield execution-ready inputs; failed refresh never resets age.
- [ ] Universe/quote changes retain coverage for every held/pending/protected symbol; whitelist/blacklist and valid fallback remain documented.
- [ ] Scanner death or staleness appears within approved target; web liveness cannot report execution readiness by itself.
- [ ] Invalid/nonfinite/out-of-range or version-conflicting config changes fail without mutation; equity_proxy saves if retained; omitted values stay unchanged.
- [ ] Rate-limit/auth/transient failures have distinct bounded behavior and no retry storm; execution/reconciliation cannot be starved by research.

## Testing and validation

Controlled-clock/market fixtures, owner restart, config conflict/invalid-value HTTP tests, request-budget/backoff and stale-held-symbol integration cases.

Run focused behavioral checks during development, relevant type/static/lint/build checks and the full applicable suite at closure. Record any genuinely inapplicable check with reason. Perform code-review, fix actionable findings and commit only scoped completed work, preserving unrelated changes.

## Regression requirements

Retain valid volume-ranked/static universe and market controls; configuration snapshots never rewrite prior operation rules.

## Security requirements

No private credentials needed for public paper data; health/account telemetry stays authenticated and redacted.

## Migration and rollout considerations

Version existing config and label unsupported trailing setting; operational numerical thresholds must be supplied under D07.

## Completion evidence

Freshness/heartbeat fault outputs, config validation matrix, request-budget traces and scoped commit.

Record criterion-by-criterion evidence links, test outputs, review outcome, commit, environment/revision and remaining limitations. Do not mark complete when a required gate is unavailable or fails. Stop dependent work, retain open status, and record exact blocker and next allowed step.

## Out of scope

Choosing a new strategy, automatic orders or a fabricated universal rate-limit value.

## Execution record

Implementation: not started. Checks: not run for this ticket. Acceptance evidence: none. Review: test boundaries and breakdown approved by the user on 2026-10-02. Commit: none. Current blockers: incomplete prerequisite tickets and unresolved policy decisions above, where applicable. Publication is complete; no feature code changes are authorized by the current planning task. Next permitted implementation step requires explicit implementation instruction and satisfied blockers.

