"""Explicit owned signal intake; manual and automatic effects share admission."""
from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any
from execution_worker import ExecutionWorker
from execution_owner import OwnershipUnavailable
from configuration import ConfigurationError, ConfigurationConflict
from market_runtime import observed_health
from strategy_profiles import decision_identity, PROFILES


class Automation:
    def __init__(self, owner: ExecutionWorker, *, enabled: bool = False, sizing: str = 'risk',
                 intake_allowed: Callable[[], bool] = lambda: True) -> None:
        if type(enabled) is not bool or sizing not in {'fixed','risk'}:
            raise ConfigurationError('Explicit automation setting and fixed/risk sizing required')
        self.owner,self.store,self.enabled,self.sizing = owner,owner.store,enabled,sizing
        self.stopped = False
        self.intake_allowed = intake_allowed
        if owner.automation is not None:
            raise ConfigurationConflict('An automation controller is already attached')
        self._status('waiting_for_reconciliation' if enabled else 'disabled',None)
        owner.automation = self

    def _status(self, state: str, reason: str | None, **details: Any) -> None:
        values = {'state':state,'enabled':self.enabled,'reason':reason,'sizing':self.sizing,
                  'owner_generation':self.owner.generation,**details}
        with self.store.connection(write=True) as connection:
            if not self.owner._matches_owner(connection):
                raise OwnershipUnavailable('Automation ownership uncertain')
            if self.store._automation(connection) == values:
                return
            connection.execute('INSERT INTO journal(namespace,event,payload) VALUES (?,?,?)',
                               (self.store.namespace,'automation_status',json.dumps(values,allow_nan=False)))

    def stop(self) -> None:
        self.stopped = True
        self._status('stopped' if self.enabled else 'disabled','explicit_stop' if self.enabled else None)

    def tick(self) -> None:
        if not self.owner.ownership_confirmed():
            raise OwnershipUnavailable('Automation ownership uncertain')
        if self.stopped:
            return
        if not self.intake_allowed():
            self.stop()
            return
        if not self.enabled:
            self._status('disabled',None)
            return
        snapshot = self.store.snapshot(now=self.owner.clock())
        scan = snapshot['scan']
        policy = snapshot['config'].get('paper_policy' if self.store.mode == 'paper' else 'execution_policy')
        if not policy:
            self._status('blocked','operator_setup_required')
            return
        if snapshot['ownership']['state'] != 'ready' or (self.store.mode == 'exchange' and snapshot['exchange_account']['state'] != 'ready'):
            self._status('waiting_for_reconciliation','startup_recovery')
            return
        health = observed_health({**scan,**snapshot},self.owner.clock())
        if not health['execution_ready']:
            self._status('blocked',health['reason'])
            return
        outcome: dict[str, Any] = {'state':'running','reason':None}
        held = {p['symbol']:p for p in snapshot['positions']}
        review = self.store.review_context(snapshot)
        server_now = self.owner.clock() + health['clock_skew_seconds']
        # Reductions remain admissible under entry pauses/drawdown, using the
        # entry-bound evaluator policy and a freshly reviewed confirmed holding.
        for signal in scan.get('exit_signals',[]):
            symbol = signal.get('symbol')
            rule = snapshot['position_rules'].get(symbol,{})
            if symbol not in held or rule.get('state') != 'bound':
                continue
            if self._valid_signal(signal,rule['config'],rule['config_version'],server_now):
                outcome = self._submit(signal,review,closing=True)
        if self.stopped:
            return
        risk = snapshot['paper_account'] if self.store.mode == 'paper' else snapshot['exchange_risk']
        if risk.get('entry_blocked'):
            self._status('blocked',risk.get('entry_block_reason') or risk.get('reason'))
            return
        for signal in scan.get('signals',[]):
            if signal.get('symbol') in held:
                continue
            if self._valid_signal(signal,snapshot['config'],snapshot['config_version'],server_now):
                result = self._submit(signal,review,closing=False)
                if outcome['state'] != 'blocked':
                    outcome = result
        if self.stopped:
            return
        self._status(**outcome)

    def _valid_signal(self, signal: dict[str, Any], config: dict[str, Any], version: int, server_now: float) -> bool:
        profile = config['strategy_profile']
        close = int(server_now // 900) * 900000
        if (signal.get('state') != 'ready' or signal.get('config_version') != version
                or signal.get('profile') != profile or signal.get('profile_version') != PROFILES[profile].version
                or signal.get('execution_close') != close or not isinstance(signal.get('symbol'),str)):
            return False
        identity = decision_identity(self.store.namespace,self.store.venue,profile,version,signal['symbol'],close)
        return signal.get('signal_id') == identity

    def _submit(self, signal: dict[str, Any], review: dict[str, Any], *, closing: bool) -> dict[str, Any]:
        identity = signal['signal_id']
        if self.stopped or not self.intake_allowed():
            self.stop()
            return {'state':'stopped','reason':'explicit_stop'}
        try:
            operation = self.store.enqueue('close' if closing else 'trade',signal['symbol'],'sell' if closing else 'buy','',
                sizing='full_close' if closing else self.sizing,
                idempotency_key=('auto_exit_' if closing else 'auto_entry_')+identity,
                now=self.owner.clock(),review=review,automatic_signal=signal,
                expected_owner=(self.owner.token,self.owner.generation))
            observed = next(o for o in self.store.operations() if o['operation_id'] == operation)
            return {'state':'blocked' if observed['state'] == 'rejected' else 'running',
                    'reason':observed.get('error_class') if observed['state'] == 'rejected' else None,
                    'last_signal':identity,'last_operation':operation}
        except (ConfigurationError,ConfigurationConflict):
            return {'state':'blocked','reason':'signal_admission_rejected','last_signal':identity}
