# BT12: Keep operator status and settings truthful and accessible

**GitHub issue:** https://github.com/adeoyewole028/binance_topdown_full/issues/14

## Parent

[Implementation specification #2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)


**Status:** open; approved breakdown; implementation not started. The ready-for-agent label is triage only: resolve the prerequisites and decision blockers before execution.
**What to build:** Give the operator stable forms, clear mode/health/metrics and usable recovery from stale, offline or invalid configuration.
**Blocked by:** [BT11 #13](https://github.com/adeoyewole028/binance_topdown_full/issues/13).
**Decision blockers:** None additional; D01 was approved by the user on 2026-10-02.
**Specification:** [#2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)
**Requirement coverage:** R16, R17, R07.

## Objective and context

Give the operator stable forms, clear mode/health/metrics and usable recovery from stale, offline or invalid configuration. This is slice 12 of the audit-remediation initiative. Its upstream requirements and decision IDs are authoritative; future slices must not be implemented early.

## Scope and implementation requirements

- Replace whole-page timed refresh with status updates preserving input/focus and selected namespace/profile.
- Show mode/venue, scan success/data age, degraded/unknown states, partial metrics and clear config errors; canonical controls for enums.
- Make live announcements, keyboard/focus/reflow/contrast/targets usable; bundle production assets and stable candle-based optional alerts.

## Affected components

Dashboard status/settings/navigation/polling, authenticated status API, production styles.

## Acceptance criteria

- [ ] Typed settings/quantities and focus persist during polling and recoverable failures; submitted invalid fields remain visible with useful errors.
- [ ] Mode/venue and data age are clear; unavailable/no-signal/permission-limited/degraded states are distinct and no false Running badge.
- [ ] Settings include version conflicts, equity estimate labels and unsupported trailing behavior; reset cannot weaken safety.
- [ ] Signal alerts deduplicate by candle identity and sound is optional; dynamic errors/status have screen-reader announcements.
- [ ] Keyboard, narrow/wide layout, zoom/reflow, contrast and touch-target manual/browser checks meet selected WCAG 2.2 AA criteria.
- [ ] Production view does not require runtime Tailwind build CDN.

## Testing and validation

HTTP rendering plus browser form-preservation, offline/stale/auth interruption, keyboard/focus/announcements/reflow and asset-build checks.

Run focused behavioral checks during development, relevant type/static/lint/build checks and the full applicable suite at closure. Record any genuinely inapplicable check with reason. Perform code-review, fix actionable findings and commit only scoped completed work, preserving unrelated changes.

## Regression requirements

Retain recognizable existing dashboard layout and optional sound, replace internal labels without a brand redesign.

## Security requirements

Same-origin authenticated polling; no stack/provider objects or untrusted HTML in toast messages.

## Migration and rollout considerations

User sound preference retained safely; namespace/profile controls expose migrated quarantine/incomplete data.

## Completion evidence

Browser journey/viewport/accessibility recordings, field/error examples, asset checks and scoped commit.

Record criterion-by-criterion evidence links, test outputs, review outcome, commit, environment/revision and remaining limitations. Do not mark complete when a required gate is unavailable or fails. Stop dependent work, retain open status, and record exact blocker and next allowed step.

## Out of scope

New design system/visual redesign, order review and risk-reduction interactions delivered in BT13.

## Execution record

Implementation: not started. Checks: not run for this ticket. Acceptance evidence: none. Review: test boundaries and breakdown approved by the user on 2026-10-02. Commit: none. Current blockers: incomplete prerequisite tickets and unresolved policy decisions above, where applicable. Publication is complete; no feature code changes are authorized by the current planning task. Next permitted implementation step requires explicit implementation instruction and satisfied blockers.

## Approved decision D01

The user selected one configured operator with a password, secure session and CSRF protection on 2026-10-02. Use password hashing and login throttling; no registration/password-reset product or upstream identity substitution. Prerequisite tickets and required verification still apply.
