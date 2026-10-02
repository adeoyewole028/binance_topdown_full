# BT02 validation and handoff

Date: 2026-10-03. Ticket: [BT02 #4](https://github.com/adeoyewole028/binance_topdown_full/issues/4). Review fixed point: `f7db424437fe35d3f531ba9991a56ae91c7b0d57`. BT01 #3 closure and BT02 native blocker state were read-back verified; no additional GitHub comments changed scope. D01 was selected by the user. D02/D03/D04 and D06 selections are preserved as planning change control, with no dependent implementation started.

## Criterion map

| Criterion | Acceptance evidence |
|---|---|
| Anonymous/unauthorized/invalid-CSRF control has no effects; GET account access authenticated | `test_anonymous_account_and_control_access_is_denied`, four route cases in `test_invalid_or_cross_site_requests_have_no_state_effects`, preserved anonymous-effect regression and authenticated paper buy/partial sell/close/reload tests. Login itself requires CSRF; logout revokes replayed cookies; expired/rotated sessions fail. Auth-store loss returns 503 before financial writes. |
| Omitted booleans retained; reset cannot elevate or change credentials | Partial settings preserve paper/dynamic values; attempted ordinary mode change returns 409; reset preserves even an exchange flag while server gate denies it; HTTP immutable-field injection is ignored with unchanged venue/gate/config. |
| Immutable gate defaults off; contradictory environments fail early | HTTP trade/close deny even in-memory exchange flags; old persisted exchange flag is replaced by configured paper startup. Environment matrix rejects invalid enums/booleans, mode/venue conflicts, credential aliases/mismatches, missing gates and mainnet without its additional gate. Paper construction sends no credentials; private testnet uses its scoped fake keys and immediate sandbox setter. |
| No normal import/CI startup; explicit launch ownership | Fresh subprocess imports unmodified dashboard/daemon/backtester under blocked network/thread/credential boundaries with no startup stripping. Explicit WSGI initializes only HTTP/access storage. Development launcher binds loopback, debug/reload off, default scanner off. Separate development scanner/optional same-process `--scan` are explicit paper-only actions. CI uses the offline runner and configured strict types. |
| Production secrets/debug rejected; authenticated paper preserved | Missing/default production secret fails; explicit debug/reload flags fail before calls. Secure/HttpOnly/Strict cookies and 30-minute expiry tested; password hash rotation and global throttle across fresh clients/module restart tested. Valid authenticated paper workflows and 15 golden strategy cases pass. |

## Red/green and checks

Initial anonymous-account test failed with HTTP 200 before the security boundary was added, then passed. The authenticated paper test failed because dashboard forms lacked CSRF, then passed after real form tokens were added. Partial config failed with paper=false, then passed with omitted-value preservation and guarded transitions. Venue construction failed on the old manually overridden testnet URLs, then passed using the common constructor. Cookie replay remained authenticated after logout, then failed after opaque session revocation was added. A Windows locked-file failure exposed that SQLite context managers do not close connections; explicit closing corrected it. Strict types exposed response-type mismatches; those were corrected.

Full applicable suite: `python tools/check_baseline.py --ticket 'BT02 / GitHub #4' --output docs/planning/trading-remediation/evidence/bt02-security.json`: **76 cases, 69 passed, 7 strict known-defect xfails, zero unexpected failures/skips**. `python -m mypy`: no issues in 13 sources, including all new security/environment/launcher modules. `python -m pyflakes .`: pass. Eight application modules compile without imports. Source/fixture hashes and exact dependency versions are in [machine evidence](bt02-security.json). Three BT02-owned xfails were promoted to ordinary passing tests. Seven future-ticket defects still reproduce through authenticated/CSRF-protected requests; none is concealed by denial of anonymous access. Strategy fixtures are unchanged.

## Boundaries and upgrade

No real credentials, network/provider orders, active hosting/service changes, source push or deployment occurred. Python source compilation is the applicable BT02 build check; Linux hosting/container/release compatibility belongs to BT14, deployed/browser/provider acceptance to BT15. Auth SQLite holds only login/session records, not future financial state. Actual namespace transitions with holdings remain BT03; `/change-mode` fails closed pending isolated recovery. Daemon exchange automation remains withheld for the reconciled executor in BT09. Existing JSON/process-local limitations are recorded; no durable financial-state claim is made here. The explicit development same-process scanner is opt-in only; normal imports/HTTP production workers stay inert.

See [operator launch/upgrade contract](../../../operator-launch.md) for credentials, compatibility conflicts, server gates, .env precedence, HTTPS/session settings, login throttle and process ownership. The Render source launcher was changed to `wsgi:app` so configured production startup actually validates access; no active service was touched.

## Review

Standards and Spec reviews pending. Do not close BT02 or begin BT03 until findings are resolved, checks pass and scoped commit/evidence are recorded.
