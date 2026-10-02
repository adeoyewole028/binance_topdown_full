# Issue tracker and execution rules

Tracker: GitHub Issues in `adeoyewole028/binance_topdown_full`.
Specification: https://github.com/adeoyewole028/binance_topdown_full/issues/2
Ordered implementation tickets: BT01-BT16, GitHub issues #3-#18.
Triage label: `ready-for-agent`; it does not override native dependencies or decision blockers.

Read the live issue body/comments and native blocking links before starting a ticket. The configured local mirrors and publication manifest are under `docs/planning/trading-remediation`; GitHub remains authoritative for ticket status. The execution ledger records local evidence and the current permitted step.

Strict implementation was authorized on 2026-10-02. Work on exactly one ticket, in the approved dependency order. Satisfy its observable acceptance criteria, appropriate TDD/focused and full applicable suite checks, relevant type/lint/build checks, two-axis code-review, documentation, evidence and scoped commit before closing. Preserve unrelated work. Do not begin dependent work while a required decision or external gate remains unresolved. Do not close the specification just because child tickets exist.

Commit completed work on the current branch; publishing source or deploying is a separate action. No current instruction authorizes real trades, provider configuration changes, production activation or spending. Review fixed point for each ticket is the captured pre-ticket commit, so reviewers compare only that ticket's work.
