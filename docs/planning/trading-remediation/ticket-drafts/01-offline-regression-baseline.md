# BT01: Establish safe regression and acceptance baselines

**GitHub issue:** https://github.com/adeoyewole028/binance_topdown_full/issues/3

## Parent

[Implementation specification #2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)


**Status:** open; approved breakdown; implementation not started. The ready-for-agent label is triage only: resolve the prerequisites and decision blockers before execution.
**What to build:** Give remediation a reproducible offline baseline for legitimate journeys and known failures before behavior changes.
**Blocked by:** None (baseline has no ticket prerequisite).
**Decision blockers:** None additional; implementation requires a separate instruction.
**Specification:** [#2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)
**Requirement coverage:** R01, R02, R08, R12, R14, R18.

## Objective and context

Give remediation a reproducible offline baseline for legitimate journeys and known failures before behavior changes. This is slice 1 of the audit-remediation initiative. Its upstream requirements and decision IDs are authoritative; future slices must not be implemented early.

## Scope and implementation requirements

- Capture current valid indicator and both runtime-profile outputs with timestamped fixtures; record the audit defects as remediation assertions.
- Build the proposed HTTP/disposable-store/fake-exchange test boundary without starting normal imports or using real credentials during characterization; retain provenance of audit AST probes.
- Record environment versions, safe offline commands, full applicable suite gates and documented behavior corrections.

## Affected components

Dashboard HTTP journey, strategy utilities, daemon/replay entry points, CI.

## Acceptance criteria

- [ ] Offline baseline and fixture generation make zero exchange requests and create no production state.
- [ ] Golden inputs establish exact legacy EMA, rolling RSI, close-pivot, hammer and engulfing outputs and valid profile differences.
- [ ] Anonymous control, partial-mode update, invalid qty, missing price, duplicate submission and live-position divergence have reproducible defect cases; remediation expectations are explicit.
- [ ] Source/CI side effects and missing behavioral-suite baseline are documented; fixture/result provenance and baseline review are recorded.

## Testing and validation

Run characterization fixtures, HTTP audit reproductions and syntax/static checks without application startup or network; distinguish baseline defects from passing future acceptance.

Run focused behavioral checks during development, relevant type/static/lint/build checks and the full applicable suite at closure. Record any genuinely inapplicable check with reason. Perform code-review, fix actionable findings and commit only scoped completed work, preserving unrelated changes.

## Regression requirements

Preserve valid strategy mathematics and legitimate paper/control journeys as characterized; never encode unsafe access or look-ahead as compatibility.

## Security requirements

Use fake keys/adapter and temporary state only; prevent fixture/log capture of actual credentials.

## Migration and rollout considerations

Read-only legacy sample fixtures only; no import/cutover or provider calls.

## Completion evidence

Versioned fixture manifest, offline baseline outputs, defect-to-test mapping, seam review and code-review/commit evidence.

Record criterion-by-criterion evidence links, test outputs, review outcome, commit, environment/revision and remaining limitations. Do not mark complete when a required gate is unavailable or fails. Stop dependent work, retain open status, and record exact blocker and next allowed step.

## Out of scope

Behavior remediation, live/sandbox acceptance, architecture selection.

## Execution record

Implementation: not started. Checks: not run for this ticket. Acceptance evidence: none. Review: test boundaries and breakdown approved by the user on 2026-10-02. Commit: none. Current blockers: incomplete prerequisite tickets and unresolved policy decisions above, where applicable. Publication is complete; no feature code changes are authorized by the current planning task. Next permitted implementation step requires explicit implementation instruction and satisfied blockers.

