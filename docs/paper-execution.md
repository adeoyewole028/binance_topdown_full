# Explicit paper-account operation (BT05)

This is local simulation, isolated by account/venue/mode. No private provider dispatch, protection placement or automatic strategy orders are implemented here. Those remain BT06/BT07/BT09; replay rules remain BT10. “Filled” means a simulated full fill under the recorded model, not provider execution.

## Setup and seed

Authenticated, CSRF-protected `/paper/setup` requires quote, initial_cash, fee_fraction, slippage_fraction, risk_fraction, fixed_quote, max_allocation_fraction, max_exposure_fraction, max_positions, cooldown_seconds and drawdown_fraction, plus expected_config_version. No numeric defaults or funding are inferred. Fractions use0.01 for1%. Values are finite decimal strings, with bounded28-digit input representation; rates are nonnegative and below1, caps positive and at most1, cash/allocation positive, capacities positive integral, cooldown nonnegative integral. Setup atomically appends an immutable configuration version, one namespace-qualified seed movement and audit event. Seed amount/quote cannot be replaced or repeated to create money; use a new isolated paper account for a new seed. Updating legacy strategy configuration cannot directly replace paper_policy. Resetting strategy settings does not reseed or remove paper assumptions.

Configuration and paper setup remain independent of exchange balances. Legacy snapshots stay quarantined under D06. Imported positions with unknown basis/history keep their review flags; they do not invent asset movements, funding or fee history, and block new paper entries until reconciled through approved migration work.

## Orders and reservations

`/trade` accepts canonical symbol, buy/sell side, quantity, optional sizing (`quantity`, `fixed`, `risk`) and Idempotency-Key/header or idempotency_key/form. `/close` uses a positive quantity or an explicit close-all contract when omitted. The rendered forms carry a unique retry key. Key scope is account/venue/mode; canonical decimal requests replay one operation, semantic differences conflict409. A retry after a close-all fill still returns the original operation. Clients that omit a key create a new independent request.

BEGIN IMMEDIATE serializes admission and reserves available quote cash for buys, confirmed available base holdings for sells, and distinct open/pending symbol slots. Pending buys are not sellable holdings. Duplicate submissions cannot reserve twice; fills/rejections/cancellations release reservations. `/paper/cancel` cancels only queued/setup/market-data-waiting paper intents, idempotently; it cannot undo fills or affect provider orders.

Fresh reviewed prices and immutable configuration bind intent admission. Changed prices/settings require a new review/request; an expired or missing reviewed quote rejects without a financial reservation. Scanner failure can defer execution, but it cannot authorize an old reviewed quote later. Account holdings outside the active strategy universe remain included in market coverage.

The exclusive execution owner rechecks public-data health, symbol/market status, MARKET support, required quantity rules, current filter inputs, capital, holdings, risk, exposure, position slots and cooldown before effects. Oversells reject completely; no quantity is silently clipped. Cash/holding movements, fees, fill, weighted-average position basis, cooldown and operation/journal state commit in one transaction. Crash or failed commit cannot preserve a partial economic effect. Exchange namespaces remain inactive until BT06.

## Sizing and simulation semantics

Fixed sizing treats fixed_quote as the total funding-asset budget including modeled fees. Risk sizing uses funded account equity, the smaller of strategy risk and the explicit paper cap, and the stop distance plus adverse slippage/round-trip fee allowance. Both cap to available cash, per-order allocation and total exposure, then round downward to provider quantity precision. Explicit quantities are validated rather than clipped. equity_proxy is display-only and never authorizes spend.

Buy price is observed price times(1+slippage); sell price times(1-slippage). Fee is quantity times simulated price times fee_fraction, paid in the quote asset. Buy fees enter weighted-average cost basis. Partial reductions retain the remaining average basis and can proceed during entry cooldown or a drawdown stop. The paper setup selects an entry cooldown; stricter current strategy cooldown and position ceilings also apply.

Drawdown admission uses marked funded equity and durable equity checkpoints. Crossing the configured drawdown threshold records a persistent entry stop, while reductions remain available. A stopped account is visible in paper_account; it cannot be silently resumed by changing display equity or reseeding. Reviewed recovery belongs to the later recovery workflow, not an automatic liquidation. Data/history that cannot support equity is marked unavailable/review-required.

## Provider rules and timing

Fresh venue-qualified exchangeInfo supplies Spot status, order type, LOT_SIZE/MARKET_LOT_SIZE and MIN_NOTIONAL/NOTIONAL rules. Required reference-price inputs are fetched through the shared IP budget; a recognized provider no-reference result permits last-price or matching average-window fallback according to metadata. Missing, stale or mismatched-window inputs reject; a24-hour ticker average never substitutes for a required5-minute window. Independent quote refresh also refreshes these inputs, and failed refresh preserves the original quote receipt time. Urgent budget declarations include reference/average requests so research leaves their headroom.

Primary sources verified2026-10-03: [Spot filters](https://developers.binance.com/en/docs/products/spot/filters), [Spot public market endpoints](https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/market). This verifies documentation, not provider/testnet acceptance. CCXT4.5.3 raw average endpoint and generic public referencePrice request are used without implicit market loading or private credentials.

## Observations, migration and rollback

Authenticated `/api/status` returns coherent paper_account, available/reserved/total balances, fills with quote fees, positions/basis flags and durable operation states. Queued/rejected orders are not labelled successful fills. These observations do not claim BT11 full-history performance metrics or BT12/BT13 browser/device acceptance.

The existing financial schema remainsv1; paper_policy is an immutable configuration value and financial history uses existing journal/movement/fill/reservation/cooldown tables. No startup migration, auto seed, actual account import, backup or provider action occurs. Existing required state remains identity-checked; malformed durable accounting stops execution. Old binary rollback after BT05 financial effects is unsafe because it could apply legacy phantom-capital behavior: stop writers, keep a verified backup, and roll forward/review rather than run an old executor against new financial state. BT14 owns packaged rollout/restore acceptance.
