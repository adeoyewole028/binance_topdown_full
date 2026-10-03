# BT12: Keep operator status and settings truthful and accessible

**GitHub issue:** https://github.com/adeoyewole028/binance_topdown_full/issues/14

## Parent

[Implementation specification #2](https://github.com/adeoyewole028/binance_topdown_full/issues/2)


**Status:** locally complete; all six acceptance gates passed; GitHub closure/read-back pending. The ready-for-agent label is triage only: resolve the prerequisites and decision blockers before execution.
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

- [x] Typed settings/quantities and focus persist during polling and recoverable failures; submitted invalid fields remain visible with useful errors.
- [x] Mode/venue and data age are clear; unavailable/no-signal/permission-limited/degraded states are distinct and no false Running badge.
- [x] Settings include version conflicts, equity estimate labels and unsupported trailing behavior; reset cannot weaken safety.
- [x] Signal alerts deduplicate by candle identity and sound is optional; dynamic errors/status have screen-reader announcements.
- [x] Keyboard, narrow/wide layout, zoom/reflow, contrast and touch-target manual/browser checks meet selected WCAG 2.2 AA criteria.
- [x] Production view does not require runtime Tailwind build CDN.

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

Implementation: all six criteria complete locally. Final source6ae3b965b20ed2e039b56d480a1a908b07fa517c (applicationfc47de9); final350fullpass/0failures/skips/xfails436.09seconds, type52/lint/32app compilation/diff/node syntax/static CSS build/audit0, both review axes0remaining. Actual Chromium145 browser3widths0axe/keyboard/focus/reflow/targets/200percent CSS scaling and first setup/typed drafts/fresh-stale metrics/history/offline/auth/422/409/reset/sound/candle journeys pass. Source hashes match full and browser evidence. Initial/intermediate failed snapshots retained; original process safety/time limits unchanged. [Criterion evidence](../evidence/bt12-validation.md), [machine results](../evidence/bt12-operator-status.json), [browser journey](../evidence/bt12-browser/journey.json). Physical-device/native zoom preference/actual screen-reader-audio and provider/deployed acceptance remain bounded as documented, assigned to later gates. No real provider/credentials/funds/trades/activation/deployment/push. Closure/read-back next; BT13 still unstarted.

## Approved decision D01

The user selected one configured operator with a password, secure session and CSRF protection on 2026-10-02. Use password hashing and login throttling; no registration/password-reset product or upstream identity substitution. Prerequisite tickets and required verification still apply.


## BT12 start (2026-10-03)

Live#14OPEN/0comments/native#13CLOSED verified; BT11closed; fixedpointaa5b434b49439b804088a56e10fad818de39d6f9. Existing HTTP/ownedfakeprovider seam plus mandatory real browser acceptance. R16/R17/R07: preserve existinglayout/typedforms/focus; same-origin statuspolling, truthful namespace/mode/venue/health/age/availability, versionedsettings/errors, candleidentity opt-in sound, semantic/native accessible controls, self-hosted bundledstyles. SelectedWCAG2.2AA gates:1.3.1,1.4.3,1.4.4,1.4.10,1.4.11,2.1.1,2.1.2,2.4.7,2.4.11,2.5.8,3.3.1/2/3,4.1.2/3. W3C primaryguidance checked2026-10-03. No visualredesign/futureorder-review/risk-reduction controls, realprovider/credentials/trade/funding/activation/deployment/push. Next:first authenticated rendering/invalid-field tracer, then targetedpolling/assets and actual guarded local browserjourneys; allsix criteria open.
