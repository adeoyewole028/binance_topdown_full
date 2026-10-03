"""Redacted local operational observations; no provider calls or hidden startup."""
from __future__ import annotations

import json
import math
import os
import re
import time
from typing import Any

from execution_owner import OwnershipUnavailable
from execution_worker import ExecutionWorker
from market_runtime import observed_health
from operations_policy import OperationsPolicy
from state_store import StateStore, StateUnavailable


class OwnerMaintenance:
    def __init__(self, owner: ExecutionWorker) -> None:
        self.owner = owner

    def step(self) -> None:
        if not self.owner.ownership_confirmed():
            raise OwnershipUnavailable('Supervised owner unavailable')
        now = self.owner.clock()
        with self.owner.store.connection() as connection:
            identity = connection.execute('SELECT id FROM store_identity').fetchone()[0]
        if not backup_ready(self.owner.store, now):
            raise StateUnavailable('Verified backup unavailable; execution must stop')
        path = self.owner.store.path.parent / 'owner-health.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps({'store_id': identity, 'token': self.owner.token,
            'generation': self.owner.generation, 'pid': os.getpid(), 'as_of': now}), encoding='utf-8')
        temporary.replace(path)


def backup_ready(store: StateStore, now: float) -> bool:
    try:
        metadata = json.loads((store.path.parent / 'backup-health.json').read_text(encoding='utf-8'))
        stamp = metadata['as_of']
        if not isinstance(metadata['snapshot'], str) or not re.fullmatch(r'snapshot-[a-f0-9]{32}', metadata['snapshot']):
            return False
        artifact = store.path.parent / 'backups' / metadata['snapshot'] / 'trading.sqlite3'
        if artifact.is_symlink() or artifact.parent.is_symlink() or artifact.parent.parent.is_symlink():
            return False
        details = artifact.stat()
        if details.st_size != metadata['size'] or details.st_mtime_ns != metadata['modified_ns']:
            return False
        with store.connection() as connection:
            identity = connection.execute('SELECT id FROM store_identity').fetchone()[0]
        return bool(metadata['store_id'] == identity and not isinstance(stamp, bool)
                    and isinstance(stamp, (int, float)) and math.isfinite(stamp)
                    and 0 <= now - stamp < OperationsPolicy().recovery_point_seconds)
    except (OSError, ValueError, KeyError, TypeError, StateUnavailable):
        return False


def control_ready(store: StateStore, now: float) -> bool:
    """Routing health for recovery UI; never a claim that trading is permitted."""
    try:
        store.validate()
        if not backup_ready(store, now):
            return False
        marker: dict[str, Any] = json.loads((store.path.parent / 'owner-health.json').read_text(encoding='utf-8'))
        stamp = marker['as_of']
        if (isinstance(stamp, bool) or not isinstance(stamp, (int, float))
                or not math.isfinite(now) or not math.isfinite(stamp)
                or not 0 <= now - stamp < OperationsPolicy().scan_health_timeout):
            return False
        if os.name != 'nt':
            pid = marker['pid']
            if type(pid) is not int or pid <= 0:
                return False
            os.kill(pid, 0)  # Linux signal zero observes liveness; never sends a terminating signal.
        with store.connection() as connection:
            identity = connection.execute('SELECT id FROM store_identity').fetchone()[0]
            owner = connection.execute('SELECT token,generation,state FROM execution_owners WHERE account=? AND venue=?',
                                       (store.account, store.venue)).fetchone()
        return bool(owner is not None and marker['store_id'] == identity and marker['token'] == owner[0]
                    and marker['generation'] == owner[1] and owner[2] != 'stopped')
    except (OSError, ValueError, KeyError, TypeError, StateUnavailable):
        return False


def observations(store: StateStore, now: float | None = None) -> dict[str, Any]:
    """Allowlisted operational metrics and operation IDs, excluding raw messages."""
    stamp = time.time() if now is None else now
    snapshot = store.snapshot(now=stamp)
    health = observed_health({**snapshot['scan'], **snapshot}, stamp)
    unresolved = [o['operation_id'] for o in snapshot['operations']
                  if o['state'] in {'unknown', 'dispatching', 'canceling', 'processing'}]
    unsafe = [g['group_id'] for g in snapshot['protection']
              if g['state'] in {'failed', 'unknown', 'canceling', 'installing'}]
    protected = {g['symbol'] for g in snapshot['protection'] if g['state'] == 'active'}
    unprotected = [p['symbol'] for p in snapshot['positions'] if p['symbol'] not in protected] if store.mode == 'exchange' else []
    history_review = any(p['basis_status'] == 'provider_basis_unknown' for p in snapshot['positions'])
    account = snapshot['exchange_account']
    risk = snapshot['paper_account'] if store.mode == 'paper' else snapshot['exchange_risk']
    conditions = []
    if not control_ready(store, stamp):
        conditions.append('owner_unavailable')
    if not health['execution_ready']:
        conditions.append('market_unavailable')
    if unresolved:
        conditions.append('unresolved_operations')
    if unsafe:
        conditions.append('protection_review_required')
    if unprotected:
        conditions.append('unprotected_exposure')
    if history_review:
        conditions.append('history_review_required')
    budget = snapshot['market_budget']
    if budget.get('state') == 'auth_blocked':
        conditions.append('provider_access_blocked')
    if budget.get('blocked_until', 0) > stamp:
        conditions.append('provider_rate_limited')
    if store.mode == 'exchange' and account.get('state') != 'ready':
        conditions.append('account_reconciliation_required')
    return {'as_of': stamp, 'config_version': snapshot['config_version'],
            'strategy_profile': snapshot['config']['strategy_profile'],
            'control_ready': control_ready(store, stamp),
            'market_ready': bool(health['execution_ready']),
            'entry_ready': bool(control_ready(store, stamp) and health['execution_ready'] and not unresolved and not unsafe
                                and not unprotected and not history_review
                                and not risk.get('entry_blocked', True) and not snapshot['entry_control']['paused']
                                and (store.mode == 'paper' or account.get('state') == 'ready')),
            'operation_ids': unresolved, 'protection_ids': unsafe,
            'unprotected_symbols': unprotected, 'conditions': conditions}
