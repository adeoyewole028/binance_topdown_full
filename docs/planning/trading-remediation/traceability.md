# Requirement, finding and acceptance traceability

Version 0.3; 2026-10-02. Test boundaries and 16-ticket breakdown: **approved by the user**. Evidence state: **all future acceptance pending**. Mapping means planned coverage, not successful implementation.

Inputs: [audit](../../audits/end-to-end-audit-2026-10-02.md), [provider research](../../audits/provider-research-2026-10-02.md). Target: [implementation specification](implementation-spec.md). Ticket detail/status: [execution ledger](execution-ledger.md).

## Source finding coverage

| Finding | Requirements | Spec sections | Draft owners | Required observable evidence |
|---|---|---|---|---|
| A01 | R01, R02 | S1, S9 | [BT02 #4](https://github.com/adeoyewole028/binance_topdown_full/issues/4), [BT13 #15](https://github.com/adeoyewole028/binance_topdown_full/issues/15) | Unauthorized/CSRF request and omitted-mode tests show zero provider/state effects. |
| A02 | R03, R05, R10, R11 | S1, S3, S5-S6 | [BT03 #5](https://github.com/adeoyewole028/binance_topdown_full/issues/5), [BT06 #8](https://github.com/adeoyewole028/binance_topdown_full/issues/8), [BT07 #9](https://github.com/adeoyewole028/binance_topdown_full/issues/9) | Actual fill projections and paper/live isolation; paper records cannot submit a real close. |
| A03 | R03, R10, R11 | S5-S6 | [BT06 #8](https://github.com/adeoyewole028/binance_topdown_full/issues/8), [BT07 #9](https://github.com/adeoyewole028/binance_topdown_full/issues/9) | Duplicate/timeout/entry-protection partial failure leaves one traceable intent/effect and recoverable exposure. |
| A04 | R08, R10, R11 | S5-S6 | [BT06 #8](https://github.com/adeoyewole028/binance_topdown_full/issues/8), [BT07 #9](https://github.com/adeoyewole028/binance_topdown_full/issues/9) | Pinned metadata precision/filters and verified Spot contingent/list protection. |
| A05 | R02 | S1 | [BT02 #4](https://github.com/adeoyewole028/binance_topdown_full/issues/4) | Environment matrix selects sandbox before private calls; imports and omitted mode never select mainnet. |
| A06 | R04, R05 | S2-S3 | [BT03 #5](https://github.com/adeoyewole028/binance_topdown_full/issues/5), [BT14 #16](https://github.com/adeoyewole028/binance_topdown_full/issues/16) | Multi-worker shared-state, single-owner restart/handover and transactional write/restore tests. |
| A07 | R08, R09 | S5 | [BT05 #7](https://github.com/adeoyewole028/binance_topdown_full/issues/7) | Invalid negative/nonfinite/missing-price/overspend/oversell leaves balances/holdings unchanged. |
| A08 | R06, R12, R14 | S4, S7-S8 | [BT04 #6](https://github.com/adeoyewole028/binance_topdown_full/issues/6), [BT08 #10](https://github.com/adeoyewole028/binance_topdown_full/issues/10), [BT10 #12](https://github.com/adeoyewole028/binance_topdown_full/issues/12) | Historical pagination/coverage and future-perturbation/as-of causality. |
| A09 | R12, R14 | S7-S8 | [BT08 #10](https://github.com/adeoyewole028/binance_topdown_full/issues/10), [BT10 #12](https://github.com/adeoyewole028/binance_topdown_full/issues/12) | Approved profile definitions, runtime/replay parity and fill/exit/capital results. |
| A10 | R05, R08, R10, R13 | S3, S5, S7 | [BT03 #5](https://github.com/adeoyewole028/binance_topdown_full/issues/5), [BT05 #7](https://github.com/adeoyewole028/binance_topdown_full/issues/7), [BT06 #8](https://github.com/adeoyewole028/binance_topdown_full/issues/8), [BT09 #11](https://github.com/adeoyewole028/binance_topdown_full/issues/11) | Available-capital sizing, actual fills, persistent cooldown/exposure release and daemon recovery. |
| A11 | R07, R11, R13, R16 | S4, S6-S7, S9 | [BT04 #6](https://github.com/adeoyewole028/binance_topdown_full/issues/6), [BT07 #9](https://github.com/adeoyewole028/binance_topdown_full/issues/9), [BT09 #11](https://github.com/adeoyewole028/binance_topdown_full/issues/11), [BT12 #14](https://github.com/adeoyewole028/binance_topdown_full/issues/14) | Heartbeat death detection, protection/manual semantics, unsupported trailing label and graceful recovery. |
| A12 | R18, R19, R20 | S10 | [BT14 #16](https://github.com/adeoyewole028/binance_topdown_full/issues/16), [BT15 #17](https://github.com/adeoyewole028/binance_topdown_full/issues/17), [BT16 #18](https://github.com/adeoyewole028/binance_topdown_full/issues/18) | Supported paid deployment model, deployed paper/testnet proof, authorized production gate. |
| A13 | R05, R07 | S3-S4 | [BT03 #5](https://github.com/adeoyewole028/binance_topdown_full/issues/5), [BT04 #6](https://github.com/adeoyewole028/binance_topdown_full/issues/6) | Config validation, equity-proxy save, missing-directory/write/corruption fault and atomic restore. |
| A14 | R09, R14, R15 | S5, S8 | [BT05 #7](https://github.com/adeoyewole028/binance_topdown_full/issues/7), [BT10 #12](https://github.com/adeoyewole028/binance_topdown_full/issues/12), [BT11 #13](https://github.com/adeoyewole028/binance_topdown_full/issues/13) | Full history, actual seeded equity drawdown, fees, restart and projection parity. |
| A15 | R06, R07, R12 | S4, S7 | [BT04 #6](https://github.com/adeoyewole028/binance_topdown_full/issues/6), [BT08 #10](https://github.com/adeoyewole028/binance_topdown_full/issues/10) | Immutable config snapshots, held-symbol freshness, completed candles and stable signal identity. |
| A16 | R01, R02, R18, R19 | S1, S10 | [BT02 #4](https://github.com/adeoyewole028/binance_topdown_full/issues/4), [BT14 #16](https://github.com/adeoyewole028/binance_topdown_full/issues/16) | Required production secret/debug policy, authenticated CORS telemetry, redacted order/errors. |
| A17 | R02, R18 | S1, S10 | [BT01 #3](https://github.com/adeoyewole028/binance_topdown_full/issues/3), [BT02 #4](https://github.com/adeoyewole028/binance_topdown_full/issues/4), [BT14 #16](https://github.com/adeoyewole028/binance_topdown_full/issues/16) | Safe imports, behavioral full-suite CI, pinned install and one gated deployment path. |
| A18 | R16, R17 | S9 | [BT12 #14](https://github.com/adeoyewole028/binance_topdown_full/issues/14), [BT13 #15](https://github.com/adeoyewole028/binance_topdown_full/issues/15) | Browser form-preservation/mode/stale/review/status/recovery, keyboard/AA evidence. |
| A19 | R07, R12, R18, R19 | S4, S7, S10 | [BT04 #6](https://github.com/adeoyewole028/binance_topdown_full/issues/6), [BT08 #10](https://github.com/adeoyewole028/binance_topdown_full/issues/10), [BT12 #14](https://github.com/adeoyewole028/binance_topdown_full/issues/14), [BT14 #16](https://github.com/adeoyewole028/binance_topdown_full/issues/16) | Indicator definitions, scoped evaluator/presentation seams, reproducible dependencies and runbooks. |
| PR01 | R02 | S1 | [BT02 #4](https://github.com/adeoyewole028/binance_topdown_full/issues/4), [BT15 #17](https://github.com/adeoyewole028/binance_topdown_full/issues/17) | Explicit venue identity, sandbox environment/proof and testnet reset isolation. |
| PR02 | R08, R10 | S5 | [BT05 #7](https://github.com/adeoyewole028/binance_topdown_full/issues/7), [BT06 #8](https://github.com/adeoyewole028/binance_topdown_full/issues/8), [BT15 #17](https://github.com/adeoyewole028/binance_topdown_full/issues/17) | Finite input and current exchange precision/lot/notional fixtures plus real testnet metadata. |
| PR03 | R11 | S6 | [BT07 #9](https://github.com/adeoyewole028/binance_topdown_full/issues/9), [BT15 #17](https://github.com/adeoyewole028/binance_topdown_full/issues/17) | Verified supported Spot protection/list lifecycle and partial-failure/cancel recovery. |
| PR04 | R10 | S5 | [BT06 #8](https://github.com/adeoyewole028/binance_topdown_full/issues/8), [BT15 #17](https://github.com/adeoyewole028/binance_topdown_full/issues/17) | Durable identity, timeout-after-success reconciliation, no blind retry. |
| PR05 | R04, R05 | S2-S3 | [BT03 #5](https://github.com/adeoyewole028/binance_topdown_full/issues/5), [BT14 #16](https://github.com/adeoyewole028/binance_topdown_full/issues/16), [BT15 #17](https://github.com/adeoyewole028/binance_topdown_full/issues/17) | Multi-process execution-owner and consistent committed state proof. |
| PR06 | R18, R19 | S10 | [BT14 #16](https://github.com/adeoyewole028/binance_topdown_full/issues/16), [BT15 #17](https://github.com/adeoyewole028/binance_topdown_full/issues/17) | Actually supported persistent topology and restore/deployed journey evidence. |
| PR07 | R06, R19 | S4, S10 | [BT04 #6](https://github.com/adeoyewole028/binance_topdown_full/issues/6), [BT14 #16](https://github.com/adeoyewole028/binance_topdown_full/issues/16), [BT15 #17](https://github.com/adeoyewole028/binance_topdown_full/issues/17) | Aggregate request-weight control, backoff/priority and observable failures. |

## Requirement coverage

| Requirement | Accepted-behavior goal (draft pending decisions) | Implementation owners | Integration/closure owners |
|---|---|---|---|
| R01 | Auth/CSRF and account privacy | [BT02 #4](https://github.com/adeoyewole028/binance_topdown_full/issues/4) | [BT15 #17](https://github.com/adeoyewole028/binance_topdown_full/issues/17) |
| R02 | Explicit mode/venue and safe startup | [BT02 #4](https://github.com/adeoyewole028/binance_topdown_full/issues/4) | [BT15 #17](https://github.com/adeoyewole028/binance_topdown_full/issues/17) |
| R03 | Unified guarded mutations/kill safety | [BT02 #4](https://github.com/adeoyewole028/binance_topdown_full/issues/4), [BT03 #5](https://github.com/adeoyewole028/binance_topdown_full/issues/5), [BT06 #8](https://github.com/adeoyewole028/binance_topdown_full/issues/8), [BT07 #9](https://github.com/adeoyewole028/binance_topdown_full/issues/9), [BT09 #11](https://github.com/adeoyewole028/binance_topdown_full/issues/11) | [BT15 #17](https://github.com/adeoyewole028/binance_topdown_full/issues/17), [BT16 #18](https://github.com/adeoyewole028/binance_topdown_full/issues/18) |
| R04 | Single ownership and consistent workers | [BT03 #5](https://github.com/adeoyewole028/binance_topdown_full/issues/5), [BT14 #16](https://github.com/adeoyewole028/binance_topdown_full/issues/16) | [BT15 #17](https://github.com/adeoyewole028/binance_topdown_full/issues/17) |
| R05 | Transactional provenance, migration/restore | [BT03 #5](https://github.com/adeoyewole028/binance_topdown_full/issues/5), [BT11 #13](https://github.com/adeoyewole028/binance_topdown_full/issues/13), [BT14 #16](https://github.com/adeoyewole028/binance_topdown_full/issues/16) | [BT15 #17](https://github.com/adeoyewole028/binance_topdown_full/issues/17) |
| R06 | Completed/fresh market data and budgets | [BT04 #6](https://github.com/adeoyewole028/binance_topdown_full/issues/6), [BT08 #10](https://github.com/adeoyewole028/binance_topdown_full/issues/10) | [BT15 #17](https://github.com/adeoyewole028/binance_topdown_full/issues/17) |
| R07 | Validated config/supervision | [BT04 #6](https://github.com/adeoyewole028/binance_topdown_full/issues/6), [BT12 #14](https://github.com/adeoyewole028/binance_topdown_full/issues/14) | [BT15 #17](https://github.com/adeoyewole028/binance_topdown_full/issues/17) |
| R08 | Admission/exposure/server sizing | [BT05 #7](https://github.com/adeoyewole028/binance_topdown_full/issues/7), [BT06 #8](https://github.com/adeoyewole028/binance_topdown_full/issues/8), [BT09 #11](https://github.com/adeoyewole028/binance_topdown_full/issues/11) | [BT15 #17](https://github.com/adeoyewole028/binance_topdown_full/issues/17) |
| R09 | Paper capital/fees/simulation invariants | [BT05 #7](https://github.com/adeoyewole028/binance_topdown_full/issues/7), [BT07 #9](https://github.com/adeoyewole028/binance_topdown_full/issues/9), [BT10 #12](https://github.com/adeoyewole028/binance_topdown_full/issues/12) | [BT15 #17](https://github.com/adeoyewole028/binance_topdown_full/issues/17) |
| R10 | Durable provider order/fill reconciliation | [BT06 #8](https://github.com/adeoyewole028/binance_topdown_full/issues/8), [BT13 #15](https://github.com/adeoyewole028/binance_topdown_full/issues/15) | [BT15 #17](https://github.com/adeoyewole028/binance_topdown_full/issues/17) |
| R11 | Protection and close races | [BT07 #9](https://github.com/adeoyewole028/binance_topdown_full/issues/9), [BT13 #15](https://github.com/adeoyewole028/binance_topdown_full/issues/15) | [BT15 #17](https://github.com/adeoyewole028/binance_topdown_full/issues/17) |
| R12 | Shared versioned causal strategy profiles | [BT08 #10](https://github.com/adeoyewole028/binance_topdown_full/issues/10), [BT10 #12](https://github.com/adeoyewole028/binance_topdown_full/issues/12) | [BT15 #17](https://github.com/adeoyewole028/binance_topdown_full/issues/17) |
| R13 | Daemon shared lifecycle/restart | [BT09 #11](https://github.com/adeoyewole028/binance_topdown_full/issues/11) | [BT15 #17](https://github.com/adeoyewole028/binance_topdown_full/issues/17) |
| R14 | Historical causal portfolio replay | [BT10 #12](https://github.com/adeoyewole028/binance_topdown_full/issues/12) | [BT15 #17](https://github.com/adeoyewole028/binance_topdown_full/issues/17) |
| R15 | Full-history capital-aware metrics | [BT11 #13](https://github.com/adeoyewole028/binance_topdown_full/issues/13) | [BT15 #17](https://github.com/adeoyewole028/binance_topdown_full/issues/17) |
| R16 | Honest preserved-input operator flows | [BT12 #14](https://github.com/adeoyewole028/binance_topdown_full/issues/14), [BT13 #15](https://github.com/adeoyewole028/binance_topdown_full/issues/15) | [BT15 #17](https://github.com/adeoyewole028/binance_topdown_full/issues/17) |
| R17 | Accessibility/responsive/asset acceptance | [BT12 #14](https://github.com/adeoyewole028/binance_topdown_full/issues/14), [BT13 #15](https://github.com/adeoyewole028/binance_topdown_full/issues/15) | [BT15 #17](https://github.com/adeoyewole028/binance_topdown_full/issues/17) |
| R18 | Reproducible CI/gated deployment | [BT14 #16](https://github.com/adeoyewole028/binance_topdown_full/issues/16) | [BT15 #17](https://github.com/adeoyewole028/binance_topdown_full/issues/17) |
| R19 | Redacted observability/backup operations | [BT04 #6](https://github.com/adeoyewole028/binance_topdown_full/issues/6), [BT14 #16](https://github.com/adeoyewole028/binance_topdown_full/issues/16) | [BT15 #17](https://github.com/adeoyewole028/binance_topdown_full/issues/17), [BT16 #18](https://github.com/adeoyewole028/binance_topdown_full/issues/18) |
| R20 | Independent sandbox/authorized rollout/closure | [BT15 #17](https://github.com/adeoyewole028/binance_topdown_full/issues/17), [BT16 #18](https://github.com/adeoyewole028/binance_topdown_full/issues/18) | [BT16 #18](https://github.com/adeoyewole028/binance_topdown_full/issues/18) |

All 19 audit findings and seven provider findings are accounted for; all 20 requirements have implementation/validation owners. [BT01 #3](https://github.com/adeoyewole028/binance_topdown_full/issues/3) supplies the baseline; it is not the only acceptance owner of any remediated behavior.

## Completeness review against source

| Review concern | Preserved in draft | Outstanding gate |
|---|---|---|
| Current manual/paper/daemon/research journeys | Existing System, S5-S9 | D03/D04 may explicitly change profiles or automation scope |
| Existing architecture, no wholesale rewrite | Solution, S2 | D02 topology/store approval |
| Auth, CSRF, secrets, telemetry, abuse | S1/S9/S10 | D01 auth selection and real permissions proof |
| Money/quantity precision, filters, available capital | S3/S5 | D05 exact risk/seed/fee policy and pinned metadata |
| Retry, unknown/duplicate/partial/out-of-order effects | S5/S6 | Provider fault/sandbox proof |
| Persisted ownership, concurrency and restart | S2/S3 | Selected store implementation and takeover proof |
| Legacy import, quarantine, rollback/restore | S3/S10 | D06 origin confirmation, D07 recovery targets |
| Market clock/freshness/warmup and rate budgets | S4 | D07 numerical thresholds and current provider metadata |
| Protection, safe close, unused trailing behavior | S6/S4 | D05 stop/OCO/partial protection policy |
| Strategy preservation and timing | S7 | D04 profiles and causal golden evidence |
| Historical range, look-ahead and capital simulation | S8 | D05 intrabar/fill/cost-basis assumptions |
| Full-history equity and performance disclosure | S8 | Data availability and shared accounting proof |
| Operator states, form preservation, accessibility | S9 | Browser AA/manual acceptance |
| Dependency order and narrow slice scope | Ledger + 16 unpublished tickets | User granularity/blocking-edge approval recorded |
| Reproducible checks, provider and rollout boundaries | Testing Decisions/S10 | External testnet/deployment/mainnet authorization |
| Diagnosability, alert/backup ownership, closure | S10 | D07 owner/destination/retention/canary policy |
| Durable tracker publication and handoff | Ledger | D08 resolved: GitHub publication verified; seam/breakdown review approved |

## Review outcome

No known source requirement is omitted. No implementation ticket is marked complete; no mocks, syntax checks or prior audit evidence are relabelled as provider/production acceptance. Approved breakdown coverage does **not** establish implementation readiness. D02-D07 and exact policy targets remain explicit gaps; D08 publication is verified; the skill-required seam/breakdown review is satisfied. Resolve remaining gaps by revising these same authoritative artifacts before executing affected tickets. Preserve the original audit as immutable evidence; record new findings or approved scope changes with affected requirement/ticket links.

## BT04 acceptance update (2026-10-03)

R06/R07/R19 map to completed BT04 local checks atd34b004:144cases142pass2strictBT06xfails; type29/lint/20sourcecompile pass; both reviews0 remaining findings. Original source completeness entries above are historical planning gates, superseded by approved D01-D04/D06-D08 records and current execution ledger. D07 operating thresholds approved; actual activation/canary/destination/access remain separate. D05 runtime explicit setup selected, remaining accounting/protection answers pending. Later provider/rollout acceptance is not satisfied by these offline results.

D05 policy resolved2026-10-03 by explicit setup selection and “use both proposed rules”. R08/R09 proceed in BT05; R11/R12 protection stays BT07; R14/R15 replay/accounting stays BT10/BT11. Numeric setup remains mandatory at runtime; provider and rollout acceptance remain distinct.


## BT05 acceptance update (2026-10-03)

R08/R09 paper admission/accounting criteria map to completed BT05 source12bfbdd,50paperbehaviorcases within194fullcases192pass2BT06strictxfails. Signed review/context freshness, invalid inputs/caps/capital, idempotency/reservations/concurrency, modeledfees/basis/fill atomicity/diskfailure, seed/security/sizing, cooldown/drawdown restart and manual reduction have evidence in `evidence/bt05-validation.md`. Standards0/Spec0 final review. Protection/replay/exchangeprovider/activation acceptance stays with later tickets, not silently satisfied here.


## BT06 local completion trace

R03/R08/R10 map to owned exchange admission/pause, reconciled capital sizing and actual fill/asset fee projections, stable identity/recovery and account-wide/order/overlapping history reconciliation at69716fb. Full229pass0xfail/skip/failure;35exchange tests include official FULL response shape, sparse delayed history, timeout/crash/write failures, cancellation/old observations, fees, reset/venue/account/namespace isolation and idle scheduling. Type35/lint/24source compile and both reviews0remaining findings. Evidence: `evidence/bt06-validation.md` and `evidence/bt06-exchange-execution.json`. Protection, provider/load certification, deployed acceptance and release closure remain assigned to their later tickets.


## BT07 local completion trace

R11/R03/R09 map to completed owned protection/serialized close lifecycle at18b1b3f. Confirmed native list+leg ownership, base-fee net quantity, isolated deterministic paper exits, budget/prepare failure recovery, cancel-fill/duplicate-close races, partial entry/remainder reprotection and pause/graceful-stop preservation are covered by20 focused protection cases within249fullpass. Standards0/Spec0 and type38/lint/26source compilation pass. Evidence:`evidence/bt07-validation.md` and`evidence/bt07-position-protection.json`. Provider certification, deployed journeys, external alert activation and initiative closure remain separate later gates.


## BT08 local completion trace

R12/R06 map to versioned pure causal profiles/runtime adapters at933c9f9. Original indicator/pivot/pattern math and profile windows/fixed daily thresholds preserved; completed UTC as-of/gap/warmup/future isolation, position exit precedence, shared adapter parity, restart-stable identity and immutable intent config/profile provenance covered by22newversioned cases within271fullpass. Type40/lint/27source compile, Standards0/Spec0. Evidence:`evidence/bt08-validation.md` and`evidence/bt08-versioned-strategies.json`. Automatic-intake dedup/lifecycle remainsBT09; real replay/coverageBT10; deployed/provider/closure gates remain later owners.


## BT09 local completion trace

R13/R03/R08 map to explicit shared owned automation at4f83d79. Bothprofile/fixed-risk completed entry traces and measured provider-clock boundaries, original-rule/current-review automatic closes, restart signal identity/cooldown/slot reservations, unset/configstale/unresolved/default/import/reset gates, fake Spot unknown protection/paused reductions, actual child process exclusion and graceful/fault stop cover all five criteria.24newautomation cases plus one necessary existing recovery-fixture boundary case within296fullpass; type43/lint/28sourcecompile, Standards0/Spec0. Evidence:`evidence/bt09-validation.md`, `evidence/bt09-automation.json`. Actual provider/deployed acceptance and initiative closure remain separate later gates.


BT10 local acceptance R14/R09/R12 completed atab5f9a9ea88d622e142a284cd65be7e691b5f0fe:327fullpass/0failures/skips/xfails261.03s,type48/lint/diff/31sourcecompile,Standards0/Spec0. [Criterion map](evidence/bt10-validation.md), [source hashes](evidence/bt10-historical-replay.json), [explicit fixture](evidence/bt10-deterministic-example.json), [public/offline replay contract](../../historical-replay.md). No actualprovider/activation or later metrics/UI/ops/acceptance/rollout implementation.


BT11 local acceptance R15/R05 at2d8b08c: full durable fill history/pagination/restart, shared fee-inclusive basis/fullposition-exit win denominator, actual seeded equity/drawdown, unique partial/base/quote/thirdassetfee qualification, stale/unresolved account and missing fill-prefix valuation gates, immutable seed and snapshot/history consistency.344fullpass520.07s/type50/lint/diff/32compile,Standards0/Spec0. [Criterion evidence](evidence/bt11-validation.md), [source hashes](evidence/bt11-account-performance.json). Later browser/provider/operational/rollout gates remain assigned; no actualprovider/activation claim.
