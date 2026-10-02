# BT09: Run automation through the same reconciled order lifecycle

**GitHub issue:** https://github.com/adeoyewole028/binance_topdown_full/issues/11

## Parent

[Implementation specification #2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)


**Status:** open; approved breakdown; implementation not started. The ready-for-agent label is triage only: resolve the prerequisites and decision blockers before execution.
**What to build:** Allow explicitly enabled automation to enter and exit safely without bypassing operator controls or duplicating ownership.
**Blocked by:** [BT07 #9](https://github.com/adeoyewole028/binance_topdown_full/issues/9), [BT08 #10](https://github.com/adeoyewole028/binance_topdown_full/issues/10).
**Decision blockers:** D03.
**Specification:** [#2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)
**Requirement coverage:** R13, R03, R08.

## Objective and context

Allow explicitly enabled automation to enter and exit safely without bypassing operator controls or duplicating ownership. This is slice 9 of the audit-remediation initiative. Its upstream requirements and decision IDs are authoritative; future slices must not be implemented early.

## Scope and implementation requirements

- Replace daemon's direct broker/counter path with same intent, strategy, risk, protection and recovery contracts.
- Persist automated status, open slots/cooldown and startup reconciliation; bind signal identity and immutable config to intent.
- Gracefully stop intake, retain unresolved intents/protection and reject simultaneous daemon launch.

## Affected components

Explicit daemon automation launcher, shared signals/admission/executor/protection, durable cooldown/status.

## Acceptance criteria

- [ ] Selected approved automation produces an intent from a completed-bar signal through the same admission/executor as manual orders.
- [ ] Duplicate scan/restart cannot create a second automatic intent for one signal identity; max open/reserved positions and cooldown survive restart.
- [ ] Confirmed exits free open slots; protection/order partial failure does not erase holdings or incorrectly release risk.
- [ ] Automation remains disabled absent explicit setting/authorization and cannot start from imports/reset or unresolved startup state.
- [ ] Second launcher cannot dispatch; graceful termination preserves operation recovery and confirmed protection.

## Testing and validation

End-to-end fake-signal-to-fill/protection/close, duplicate owners/restart/cooldown/max-slots and shutdown fault tests.

Run focused behavioral checks during development, relevant type/static/lint/build checks and the full applicable suite at closure. Record any genuinely inapplicable check with reason. Perform code-review, fix actionable findings and commit only scoped completed work, preserving unrelated changes.

## Regression requirements

Retain approved daily/intraday profile and whitelist/blacklist/fixed-risk behavior; no endless monotonic position counter.

## Security requirements

Same venue/live/kill gates as HTTP; secrets only within selected adapter.

## Migration and rollout considerations

Disable old direct-dispatch entry path with explicit upgrade instructions; all tracked effects use namespace/version.

## Completion evidence

Automation lifecycle demo/trace, ownership shutdown results, scope/profile approval and scoped commit.

Record criterion-by-criterion evidence links, test outputs, review outcome, commit, environment/revision and remaining limitations. Do not mark complete when a required gate is unavailable or fails. Stop dependent work, retain open status, and record exact blocker and next allowed step.

## Out of scope

Selecting autonomous mainnet authorization, opportunistic strategy redesign.

## Execution record

Implementation: not started. Checks: not run for this ticket. Acceptance evidence: none. Review: test boundaries and breakdown approved by the user on 2026-10-02. Commit: none. Current blockers: incomplete prerequisite tickets and unresolved policy decisions above, where applicable. Publication is complete; no feature code changes are authorized by the current planning task. Next permitted implementation step requires explicit implementation instruction and satisfied blockers.

