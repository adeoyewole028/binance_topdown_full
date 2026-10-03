# Owned Spot execution and reconciliation

BT06 source behavior; actual provider certification is reserved for BT15. No real account, credential, order or deployment was used for local evidence.

## Setup and launch

The runtime defaults to paper. An exchange namespace requires venue-specific credentials and the server execution gates; mainnet has its additional separate allow gate. Configure the authenticated Exchange risk setup before submitting an order. Every numeric field is required. This records estimates and caps, adds no funds, and uses no paper cash or display-only equity proxy. Its funding quote cannot subsequently change.

Use the explicit supervised `run_worker.py --scan` launcher for combined public scanning and exchange execution. The private adapter is constructed only by this gated launcher; HTTP handlers enqueue intents, never submit orders. Imports start neither network work nor threads. A second owner fails the kernel lock. A worker without scanning requires an already valid persisted market observation; `--once` cannot supply fresh data by itself. Supervisor installation and provider/deployed acceptance remain later tickets.

## Orders and money

Review the authenticated status before submission. A signed review binds the current namespace, configuration, price and holding observation; a changed review is refused. Fixed and risk sizing use reconciled available capital and current quantity filters. The worker rechecks completed market data, precision, funds, allocation, exposure, drawdown, cooldown and capacity under ownership immediately before dispatch. Buy entries can be paused independently of reductions and reconciliation. Clearing that pause does not clear a drawdown latch or provider review state.

The durable semantic intent and client order identity precede transport. The shared request budget must admit the call before its dispatch reservation is committed. A request deferred before transport remains queued without an uncertain reservation. Only a classified authoritative pre-engine rejection releases its reservation. Other failed submits retain their identity and funds for reconciliation; a missing order lookup never authorizes a replacement submission.

Confirmed provider trades, costs and original fee assets commit with positions and immutable asset movements. Base-asset fees reduce sellable holdings. Third-asset fees reduce that asset's confirmed holding; no exchange rate or historical basis is fabricated. Partial fills reserve only the unfilled remainder. Provider-confirmed terminal status releases the remainder without discarding fills. Repeated or older observations cannot erase terminal facts. Conflicting economic data for an existing fill identity halts observation rather than replacing history.

## Recovery

Startup checks account and venue identity before merging orders or fills. Account-wide open orders include symbols outside scanner coverage. Order status and paginated trade history overlap and deduplicate; baseline history establishes cursors without inventing imported basis. Existing provider holdings are recorded with unknown basis and disclosed as such. New unowned trades, unowned open orders, account changes and unexplained balance resets latch review. The worker never merges another venue or unresolved namespace into this account.

Idle reconciliation follows the approved 15-second interval. Pending entries require reconciliation before dispatch; an unresolved/failed account cannot use new intent keys to bypass recovery. A failed write after a provider effect leaves the original dispatch identity available for recovery. Restore durable write availability and reconcile the same identity; do not delete history, release uncertain funds, change accounts, or create a replacement intent to guess an outcome. A review latch needs a later approved recovery procedure; no automatic acceptance of divergent provider balances exists.

Protection and protection-aware closes belong to BT07. Automation, performance analytics, deployment and controlled sandbox/live acceptance retain their separate ticket gates. Local fake-provider evidence is not proof of provider execution, protection, or production readiness.
