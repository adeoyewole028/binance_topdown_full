# BT10: Backtest complete historical ranges with causal execution

**GitHub issue:** https://github.com/adeoyewole028/binance_topdown_full/issues/12

## Parent

[Implementation specification #2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)


**Status:** local implementation complete; all five acceptance gates passed; tracker closure/read-back pending. The ready-for-agent label is triage only: resolve the prerequisites and decision blockers before execution.
**What to build:** Produce reproducible historical trade/equity results that cannot see future bars or silently omit requested data.
**Blocked by:** [BT07 #9](https://github.com/adeoyewole028/binance_topdown_full/issues/9), [BT08 #10](https://github.com/adeoyewole028/binance_topdown_full/issues/10).
**Decision blockers:** none for the selected D05 policy; explicit runtime setup/provider acceptance remain required as applicable.
**Specification:** [#2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)
**Requirement coverage:** R14, R09, R12.

## Objective and context

Produce reproducible historical trade/equity results that cannot see future bars or silently omit requested data. This is slice 10 of the audit-remediation initiative. Its upstream requirements and decision IDs are authoritative; future slices must not be implemented early.

## Scope and implementation requirements

- Fetch UTC start/end plus warmup with coverage/gap report, bounded chronological replay and deduplicated entries.
- Use selected runtime profiles and approved fees/slippage/entry-next-event/exits/capital/intrabar policy.
- Report fills, positions, exposure, equity and result provenance instead of only entry count.

## Affected components

Historical data pagination/replay CLI, shared evaluator and paper fill/accounting, research result provenance.

## Acceptance criteria

- [x] Range older than last 1000 bars paginates correctly with required warmup; missing/gapped history reports incomplete rather than valid zero result.
- [x] No decision uses final daily close before its availability, future bias bar, unfinished candle or overlapping iteration duplicate.
- [x] Future-data mutation cannot change earlier decisions; identical as-of runtime/replay inputs produce identical signals.
- [x] Fees/slippage/capital, gap exits, same-bar TP/SL ambiguity and partial-fill assumptions follow recorded D05 policy.
- [x] CLI range/profile inputs and reproducible result provenance include coverage, version and selected assumptions; output makes no profitability guarantee.

## Testing and validation

Local paginated adapter fixtures, timezone/range limits, causal perturbation/runtime parity, deterministic portfolio/loss/exit accounting and gap/ambiguity cases.

Run focused behavioral checks during development, relevant type/static/lint/build checks and the full applicable suite at closure. Record any genuinely inapplicable check with reason. Perform code-review, fix actionable findings and commit only scoped completed work, preserving unrelated changes.

## Regression requirements

Keep research workflow offline-capable and selected shared indicator behavior; incorrect previous trade counts are not preserved.

## Security requirements

Public-only historical adapter and strict research rate lane; no order-capable calls from replay.

## Migration and rollout considerations

Version outputs; label old entry-only results unvalidated without attempting retrospective fabricated fills.

## Completion evidence

Coverage and causal/parity test reports, deterministic example results and scoped commit.

Record criterion-by-criterion evidence links, test outputs, review outcome, commit, environment/revision and remaining limitations. Do not mark complete when a required gate is unavailable or fails. Stop dependent work, retain open status, and record exact blocker and next allowed step.

## Out of scope

Historical market-data purchase, profit optimization or live fill fidelity guarantee.

## Execution record

Implementation: complete at ab5f9a9ea88d622e142a284cd65be7e691b5f0fe (source15d7fb3/ab5f9a9). All five criteria satisfy explicit capture-bound UTC pagination/warmup/gap coverage, both versioned profiles/shared causal evaluator, next-open financial replay/protection and reproducible public-only/offline CLI. Full327pass/0failures/skips/xfails261.03s; strict type48, repositorylint/diff/31sourcecompilation pass; Standards0hard/0heuristic, Spec0remaining after original twoP2 corrections. [Criterion/review evidence](../evidence/bt10-validation.md), [machine hashes](../evidence/bt10-historical-replay.json), [deterministic explicit fixture](../evidence/bt10-deterministic-example.json), [research contract](../../../historical-replay.md). No schema migration, fabricated capital, actual provider/credentials/trade/activation/deployment/push. Historical/livefilter and fillfidelity not certified; laterprovider/rollout gates remain. Currentblockers:none for localBT10. Next:publish/close/read-back#12, then inspectBT11 alone.

## D05 approved contract (2026-10-03)

APPROVED on 2026-10-03: explicit operator setup for all capital/risk/fee/slippage/cooldown/drawdown inputs before paper trading; deterministic full paper fills, quote-currency fees, weighted-average cost basis, adverse slippage and gap prices, stop first when both stop/profit targets touch one candle. Exchange-supported stop-market protection; refuse unsupported symbols; re-protect confirmed remaining holdings after partial closes; failed/uncertain protection blocks entries; automatic emergency liquidation and trailing remain disabled.
Implement only this ticket’s scope; this approval does not authorize future tickets early, account funding or live/provider/rollout actions.


## BT10 start (2026-10-03)

Live#12 OPEN/0comments/native#9/#10 bothCLOSED verified afterBT09 closure. Sole active ticket; fixedreviewpoint`9853710e3da1111a0f4ea0e3e0c96a04abe623c4`. D05 bothrules/explicitallcapitalinputs approved; D04 bothprofiles. Approved pure completedbar/asof/config/position evaluator/replay seam and fake paginated public adapter; no profit optimization or livefidelity claim. All fivecriteriaopen. Next:paginated warmup/range/coverage tracers, then causal next-event portfolio/exits/provenance and strict CLI/research budget gate. No later-ticket code or actual provider/credential/funding/trade/deployment/push.


## Final local acceptance (2026-10-03)

All five criteria passed atab5f9a9. 31historical focusedpass; full327pass261.03s/type48/lint/diff/31sourcecompile; two-axis finalreviews0remaining. Initial322pass snapshot preserved honestly; corrected finalresult supersedes it. No futureticket source. Parent#2 staysOPEN; actualprovider and rollout gates staylater.
