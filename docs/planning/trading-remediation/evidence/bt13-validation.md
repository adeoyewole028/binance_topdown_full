# BT13 reviewed manual order and recovery evidence

Status: all five local acceptance criteria complete. Full offline suite: 371 passed, zero failures/skips/expected failures, 471.20 seconds. BT14–BT16 have not started.

Fixed point: `df35364e87345ff4f9d462e9a7c4a630121c7c6f`. Application: `73060ffe7a894ed97cbb3e2a20fe8359d7d6827e`. Test launcher checkpoint: `e0447a24bd6f2835a2c39e46ef666498a669cd8a`. No external provider, credentials, funding, actual exchange trade, activation, migration, deployment or push was used.

## Criterion map

1. Changed/stale reviews and authoritative sizing: `tests/test_manual_orders.py` tests side-effect-free fixed allocation despite a browser quantity of 99999, signed price/config/age/namespace/quantity conflicts, concurrent funding reservation changes, invalid/tampered/reused tokens, exact derived balance after a tiny approved fee, and fresh supervised market/protection gates. Real Chromium delays the actual HTTP review response while quantity changes from 1 to 2; the old response cannot enable confirmation. The captured draft must still match, and confirmation sends current semantics for independent server comparison.

2. One intent across repeat/refresh/retry: HTTP repeated confirmation returns the same operation before and after execution. Chromium loses the actual response after committed intake, verifies replacement is blocked before refresh, refreshes with the same token/draft, and finds exactly one operation through authenticated token lookup. Confirmation requires retained browser storage; blocked storage creates no intent. Queued/unknown/partial states show no invented fill and do not offer a replacement. Lookup and status checks make no provider request.

3. Recoverable actual outcomes: HTTP and seven browser scenarios exercise paper fill/close, Spot partial fill, unknown dispatch with original-identity reconciliation, provider rejection, expiry, failed protection and unknown protective cancellation. Confirmed native fees are displayed separately from estimates. Expired orders preserve zero or partial actual fill histories and allow a new reviewed draft under unchanged risk/protection gates. Failed protection preserves the confirmed position; new entries remain gated, and reductions wait for fresh supervised account reconciliation. Saved operation lookup is independently readable without replacing the active draft.

4. Close independently of signals: current position buttons load reviewed partial or full close drafts. HTTP proves a 0.4 close without an exit signal, 0.6 confirmed remainder and matching simulated protection. Chromium performs paper partial/full close and a Spot reduction with failed protection. An unknown protective cancellation leaves the close queued, records no sell fill, and repeated intake cannot create a second sell. Only current confirmed holdings supply these controls; legacy quarantine is read-only.

5. Mode/permission/draft and keyboard behavior: HTTP refuses wrong mode/venue/namespace before intake and enforces authentication/CSRF/namespace lookup. Chromium signs out after a full-close review, preserves the signed intent and form, signs in in a separate tab, refreshes CSRF and confirms the same reviewed close. User-triggered errors/review messages receive keyboard focus; current position close focuses its quantity input. Seven journeys have zero automated WCAG A/AA findings and mobile page reflow. Original dashboard regression also passes desktop/mobile/320px checks, polling, drafts, setup/settings errors, stale valuations, offline/auth recovery and reset protection.

## Checks and review

- 102 initial focused HTTP/order/protection/security checks passed in 150.54 seconds.
- Final 21 focused manual-order cases passed in 24.59 seconds; strict type checking passed for 55 source files, repository lint, script syntax, CSS build and whitespace checks passed.
- [Final manual browser traces and application hashes](bt13-browser/journey.json): seven actual Chromium scenarios against guarded loopback Flask, transactional SQLite, kernel-owned execution and fake public/private providers. Browser-only storage and response-loss conditions are exercised at their external boundaries.
- [Original dashboard regression traces](bt13-browser/bt12-regression/journey.json): three widths, zero automated accessibility findings, keyboard/reflow, 200 percent CSS scaling and original status/settings journeys. Historical BT12 artifacts remain unchanged.
- Standards and Spec reviews report zero remaining actionable findings at `e0447a2`. Initial Spec findings for derived precision, expiry and delayed draft response were reproduced by regressions and fixed; the optional shared freshness heuristic was addressed. A final clean first-attempt lost-response regression found and fixed replacement gating before HTTP.
- [Full offline suite](bt13-manual-orders.json): 371/371 passed in 471.20 seconds at e0447a2; zero failures/skips/expected failures. All 40 source hashes match current files; both seven-file browser hash sets match the full suite. All 33 application Python files compile. Final strict55 type checks, repository lint, script syntax, CSS build and whitespace checks pass. Aggregate600-second budget and original individual safety gates unchanged.

Intermediate browser snapshots remain alongside final evidence. Fixture startup timeouts were investigated; the harness now waits for each guarded fixture before launching Chromium and closes that owned browser per scenario. The original 30-second startup limit, request bounds and original subprocess safety limits remain unchanged.

## Limits and next step

These are local offline and real desktop Chromium checks. They do not certify real provider behavior, deployment, native mobile devices, screen-reader use or audible alerts. Those required external/rollout milestones remain separate. Publish and read back BT13 closure before beginning BT14. Application commits 0ee08c6, 1ac07cc and 73060ff; final test checkpoint e0447a2. Source is committed locally; nothing has been pushed or deployed.
