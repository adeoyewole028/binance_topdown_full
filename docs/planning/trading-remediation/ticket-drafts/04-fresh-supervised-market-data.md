# BT04: Expose fresh market data and supervised scan health

**GitHub issue:** https://github.com/adeoyewole028/binance_topdown_full/issues/6

## Parent

[Implementation specification #2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)


**Status:** completed locally; all acceptance/check/review/evidence gates passed on 2026-10-03. Tracker closure pending read-back.
**What to build:** Let the operator trust scan age and held-position data, with stale or failed inputs preventing execution.
**Blocked by:** [BT03 #5](https://github.com/adeoyewole028/binance_topdown_full/issues/5).
**Decision blockers:** none for BT04; D07 targets selected by the user.
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

- [x] Unfinished/duplicate/malformed/gapped/insufficient candles cannot yield execution-ready inputs; failed refresh never resets age.
- [x] Universe/quote changes retain coverage for every held/pending/protected symbol; whitelist/blacklist and valid fallback remain documented.
- [x] Scanner death or staleness appears within approved target; web liveness cannot report execution readiness by itself.
- [x] Invalid/nonfinite/out-of-range or version-conflicting config changes fail without mutation; equity_proxy saves if retained; omitted values stay unchanged.
- [x] Rate-limit/auth/transient failures have distinct bounded behavior and no retry storm; execution/reconciliation cannot be starved by research.

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

Started2026-10-03 after BT03 #5 closure/native blocker verification and user's “use these targets” selection. D07 approved: scans300s, scan-health600s, price-refresh30s/expiry60s, reconciliation15s, backups7days/RPO15min/RTO1hour, configured operator/private webhook alerts configured before activation. Backup operations belong to BT14; private reconciliation behavior belongs to BT06. Hosting spend, deployment, actual alert destination and live rollout remain separate gates. Checks/acceptance/review are not yet complete. No future ticket started and no source push, service change, provider call or deployment.

Pre-review implementation commit `5a9d4b6` on local main (not pushed). Freshness/cache/health, complete atomic config validation, retained coverage, public owned adapter and shared priority budget implemented. Initial full offline suite137pass/2strictBT06xfails; strict types29/lint pass. Subsequent spec completeness correction binds unambiguous entry-config exit rules; focused17+mixed-entry checks pass. Independent Standards/Spec reviews are active and final full evidence will follow. Local artifacts: `docs/planning/trading-remediation/evidence/bt04-validation.md`, `bt04-market-data.json`; runbook `docs/market-data.md`. All criteria remain open until review/final validation and closure evidence. Actual provider certification, migration and deployment were not performed.


## Final completion evidence (supersedes preliminary execution records)

Source revision `d34b004f96f266c5fb85505edbf62467cdde11e3`; scoped commits `5a9d4b6`, `1ac1db2`, `d34b004`. Full144cases142pass2strictBT06expected failures,0otherfailures/skips in49.01s. Strict mypy29, repository Pyflakes, whitespace check and20-source compilation pass. Standards0/Spec0 actionable findings after original P1 quote-age/P2 uncertain-window corrections and four independent correction regressions. All five acceptance criteria are checked; detailed evidence at `docs/planning/trading-remediation/evidence/bt04-validation.md` and `bt04-market-data.json`; operational runbook `docs/market-data.md`. Tests use temporary offline fixtures and fake transport; source remains local, no real provider/account/deployment acceptance claimed. BT05 implementation has not started.
