# BT09 automation acceptance

Sole active ticket BT09 #11. Fixed point d4900456c4dfa6b36eb0c02e8850d3f22af576a1. D03 explicit disabled-by-default automation and D04 both versioned profiles retained. All five local acceptance gates passed at4f83d79; final acceptance below supersedes intermediate pending notes.

| Criterion | Observable evidence | Gate |
|---|---|---|
| Completed-bar signal uses manual admission/executor | Both profile/fixed-risk entry traces, actual paper fills and active simulated protection; fake Spot entry/protection; shared signed/automatic review context | Passed; final acceptance below |
| Restart idempotency, position/cooldown limits | Repeat/restart one signal/intent; confirmed close cannot reenter same bar; cap on two reserved automatic signals; restart cooldown refusal and later slot reuse | Passed; final acceptance below |
| Confirmed exits and partial/unknown safety | Automatic indicator close retains entry config with current dispatch review; entry-paused fake Spot cancels OCO and confirms close; lost protection response keeps holding/reservation and refuses entries through restart | Passed; final acceptance below |
| Explicit default-disabled lifecycle | Default controller/advisory/import/reset remain disabled even envflagtrue; launcher needs explicit flag+server gate; missing setup/stale/configchanged/unknownstartup refuse intake | Passed; final acceptance below |
| Single owner and graceful/fault shutdown | Real child automation owner fences actual competing launcher; durable graceful termination; stop during public scan prevents intake; shutdown journal write failure releases lock and retains confirmed protection | Passed; final acceptance below |

## TDD and review corrections

Shutdown state was initially incorrectly running after owner exit; owned cleanup now records stopped/disabled and preserves protection. Missing automatic indicator close reproduced at HTTP/owned-market seam, then shared full-close path passed with immutable entry-rule provenance. Cooldown/cap refusals initially reported running; status now reports the actual rejected admission without journal churn from intermediate states. Clock fixture rounding was corrected to explicit whole seconds, retaining strict negative/future freshness checks rather than weakening them. External fault fixtures and fake providers only; no internal collaborator mocking or SQL acceptance assertions.

Standards at a542148:0 documented violations,1 optional P3 review-context duplication. Shared canonical StateStore.review_context now serves both automation and signed HTTP, resolving the integration concern. Final Standards at e889870:0hard/0heuristic.

Spec independently reproduced P2 at a542148: accepted provider clock skew crossed the 15m boundary but automation compared to unadjusted local time, incorrectly skipping a healthy completed signal. Corrected e889870 derives current completed boundary from the accepted measured provider-clock observation for both entries and entry-bound exits. New +/-30s boundary HTTP/fill cases pass. Final Spec pending.

Initial full293cases at a542148 passed212.33s,0fail/skips/xfail. Corrected full295case run at e889870 pending, aggregate budget600s with individual safety/process gates unchanged. Type43/lint/diff pass; final 28source compilation pending. [Runbook](../../../automation.md). No schema migration, fabricated capital, profile redesign, emergency liquidation or trailing contract. Actual provider/trade/funding/migration/deployment/push remains absent; BT10+ and production/provider/rollout gates remain outstanding.


## Full-suite boundary fixture correction

Corrected e889870 full run:294pass/1failure/0skips/xfail212.70s, preserved in bt09-clock-correction-failed.json. The failure was the existing BT07 deferred-cancel recovery test; isolated original case passed4.32s. It used a frozen local clock but an actual wall-clock public provider, then assumed advancing to the next local minute always expired the server budget window. A near-boundary provider observation could already move into that window. Conservative production budget/review expiry behavior correctly retained the holding instead of the test's unconditional expected close. The necessary prerequisite fixture correction aligns fake public metadata/bars with the explicitly controlled owner clock and parameterizes seconds1/59 to exercise both review-age extremes. Both corrected cases passed6.46s with the original close/retained-protection/private-call-count assertions unchanged; no production budget, expiry, financial/protection code or safety gate changed. Final corrected full296case run and independent delta review are required before closure.


## Standards

Final independent review at4f83d79:0hard violations/0remaining heuristic findings. P3 canonical review context resolved; the deterministic external fixture correction retains original behavioral assertions and changes no production gate.

## Spec

Final independent review at4f83d79:0remaining actionable findings. P2 accepted-provider clock correction independently passed both boundary cases. Final fixture delta independently passed both recovery cases8.14s, preserves original gates and is a necessary prerequisite correction. No future-ticket scope added. Final full-suite completion is still required.


## Final acceptance

All five BT09 local criteria passed at4f83d796222de1d242fe944fa16768570fa7f2f3. [Exact machine result and28source hashes](bt09-automation.json):296passed,0failures/skips/xfails217.77s, bounded aggregate600s. Type43sources, repository lint/diff and28application-source compilation pass.24newautomation cases cover both profiles/sizing, measured provider clock boundaries, automatic paper/Spot entry/fill/protection/exit, original-rule exits/currentreview, repeated/restartidempotency, persisted limits/cooldown, explicit defaults/reset/setup/health/startup gates, real process exclusion, stopduringpublicscan and statuswritefault recovery. Existing BT07 boundary fixture now has two strict controlled-clock cases; no production protection/timing change. Standards0/Spec0 final independent reviews, both original concerns resolved and necessary fixture correction independently rechecked.

Source commits a542148/e889870; prerequisite regression-fixture correction4f83d79; fixedpointd490045. Existing journal/config/intent/position/cooldown/reservation/protection storage used with additive source/provenance/status projections; no schema/data migration or implicit capital setup. [Runbook](../../../automation.md). Earlier full failure remains honestly preserved in [failed validation snapshot](bt09-clock-correction-failed.json), superseded by the complete passing run. No actual provider, funding, trade, migration, deployment, activation or push. ReplayBT10 and latermetrics/UI/ops/provider/rollout gates remain unstarted and outstanding. Publish/close/read-back#11 next; parent#2 remainsOPEN.
