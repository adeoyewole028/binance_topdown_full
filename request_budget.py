"""Redacted provider failure contract, shared by market and future executor lanes."""
from __future__ import annotations
from collections.abc import Callable
from typing import Any, TypeVar
import json
import math
import random
import time
from contextlib import contextmanager
from collections.abc import Iterator
from state_store import StateStore, StateUnavailable
from execution_owner import ExecutionOwner, OwnershipUnavailable

T = TypeVar('T')
SECONDS = {'SECOND': 1, 'MINUTE': 60, 'HOUR': 3600, 'DAY': 86400}


class ProviderFailure(RuntimeError):
    def __init__(self, kind: str, *, retry_after: float = 0) -> None:
        if kind not in {'rate_limit', 'auth', 'transient', 'invalid_market', 'data_quality'}:
            raise ValueError('Unknown provider failure class')
        super().__init__(kind)  # Never retain provider payloads or credentials.
        self.kind, self.retry_after = kind, retry_after


class BudgetDeferred(ProviderFailure):
    def __init__(self, retry_at: float) -> None:
        super().__init__('rate_limit')
        self.retry_at = retry_at


class RequestBudget:
    """One database per host/egress IP. All accounts share the venue row.

    Discovered limits are authoritative. Background work leaves a known 107-weight
    public-data burst (metadata/time, tickers, three klines) for urgent traffic; future clients must
    declare larger bursts via reserve_weight rather than inventing provider limits.
    """
    def __init__(self, store: StateStore, *, clock: Callable[[], float] = time.time,
                 reserve_weight: int = 107, reserve_requests: int = 6) -> None:
        if type(reserve_weight) is not int or type(reserve_requests) is not int or min(reserve_weight, reserve_requests) <= 0:
            raise ValueError('Positive urgent burst declaration required')
        self.store, self.clock = store, clock
        self.reserve_weight, self.reserve_requests = reserve_weight, reserve_requests

    def resume(self, owner: ExecutionOwner, *, actor: str) -> None:
        if not actor or owner.store.namespace != self.store.namespace or owner.store.path != self.store.path:
            raise OwnershipUnavailable('Reviewed operator and matching owner required')
        with self.store.connection(write=True) as connection:
            if not owner._matches_owner(connection):
                raise OwnershipUnavailable('Reviewed exclusive owner required')
            row = connection.execute('SELECT values_json FROM market_budget_v1 WHERE venue=?', (self.store.venue,)).fetchone()
            if row is None:
                raise StateUnavailable('Market budget unavailable')
            state = json.loads(row[0])
            state.update(auth_blocked=False, blocked_until=0, failures=0, last_failure=None)
            connection.execute('UPDATE market_budget_v1 SET values_json=? WHERE venue=?', (json.dumps(state), self.store.venue))
            connection.execute('INSERT INTO journal(namespace,event,payload) VALUES (?,?,?)',
                               (self.store.namespace, 'market_access_reviewed', json.dumps({'actor': actor, 'venue': self.store.venue})))
            scan = self.store._scan(connection)
            if scan:
                health = scan.get('market_health', {})
                health.update(state='degraded', reason='scanner_uninitialized', failures=0, retry_at=0)
                health.pop('price_refresh_failure', None)
                scan['market_health'] = health
                connection.execute('UPDATE scan_snapshots SET values_json=? WHERE namespace=?', (json.dumps(scan), self.store.namespace))

    @contextmanager
    def _state(self) -> Iterator[dict[str, Any]]:
        with self.store.connection(write=True) as connection:
            row = connection.execute('SELECT values_json FROM market_budget_v1 WHERE venue=?', (self.store.venue,)).fetchone()
            if row is None:
                raise StateUnavailable('Explicit market extension initialization required')
            values = json.loads(row[0])
            yield values
            connection.execute('UPDATE market_budget_v1 SET values_json=? WHERE venue=?',
                               (json.dumps(values, allow_nan=False), self.store.venue))

    def configure(self, metadata: dict[str, Any]) -> None:
        limits = []
        now = self.clock()
        for raw in metadata['rateLimits']:
            if raw['rateLimitType'] not in {'REQUEST_WEIGHT', 'RAW_REQUESTS'}:
                continue
            duration = int(raw['intervalNum']) * SECONDS[raw['interval']]
            limit = int(raw['limit'])
            if duration <= 0 or limit <= 0:
                raise ProviderFailure('data_quality')
            limits.append({'kind': raw['rateLimitType'], 'duration': duration, 'limit': limit})
        if not any(r['kind'] == 'REQUEST_WEIGHT' for r in limits):
            raise ProviderFailure('data_quality')
        with self._state() as state:
            # Last measured server timestamp is a conservative lower bound at
            # receipt, so local clock skew cannot open a request window early.
            state['clock_skew'] = float(metadata['serverTime']) / 1000 - now
            now += state['clock_skew']
            previous = {(r['kind'], r['duration']): r for r in state['limits']}
            for item in limits:
                old = previous.get((item['kind'], item['duration']))
                start = int(now // item['duration']) * item['duration']
                item.update(start=start, used=old['used'] if old and old['start'] == start else
                            (21 if item['kind'] == 'REQUEST_WEIGHT' else 2))
            state['limits'] = limits
            state['reserve_weight'] = max(self.reserve_weight, state.get('reserve_weight', 0))
            state['reserve_requests'] = max(self.reserve_requests, state.get('reserve_requests', 0))
            state['discovery_until'] = 0

    def _claim(self, lane: str, weight: int, raw_requests: int, discovery: bool) -> None:
        if lane not in {'reduction', 'reconciliation', 'entry', 'market', 'research'} or any(type(v) is not int or v <= 0 for v in (weight, raw_requests)):
            raise ValueError('Invalid request-budget declaration')
        now = self.clock()
        refusal: float | None = None
        with self._state() as state:
            state['reserve_weight'] = max(self.reserve_weight, state.get('reserve_weight', 0))
            state['reserve_requests'] = max(self.reserve_requests, state.get('reserve_requests', 0))
            if state.get('auth_blocked'):
                raise ProviderFailure('auth')
            if now < state.get('blocked_until', 0):
                raise BudgetDeferred(state['blocked_until'])
            if not state['limits']:
                if not discovery or now < state.get('discovery_until', 0):
                    raise BudgetDeferred(max(now + 30, state.get('discovery_until', 0)))
                state['discovery_until'] = now + 30
                return
            now += state.get('clock_skew', 0)
            for limit in state['limits']:
                start = int(now // limit['duration']) * limit['duration']
                if start != limit['start']:
                    limit.update(start=start, used=0)
                cost = weight if limit['kind'] == 'REQUEST_WEIGHT' else raw_requests
                reserve = state['reserve_weight'] if limit['kind'] == 'REQUEST_WEIGHT' else state['reserve_requests']
                capacity = limit['limit'] if lane in {'reduction', 'reconciliation'} else max(0, limit['limit'] - reserve)
                if limit['used'] + cost > capacity:
                    refusal = max(refusal or 0, start + limit['duration'])
            if refusal is None:
                for limit in state['limits']:
                    limit['used'] += weight if limit['kind'] == 'REQUEST_WEIGHT' else raw_requests
        if refusal is not None:
            raise BudgetDeferred(refusal)

    def request(self, lane: str, weight: int, call: Callable[[], T], *, raw_requests: int = 1,
                discovery: bool = False) -> T:
        self._claim(lane, weight, raw_requests, discovery)
        try:
            return call()
        except (TimeoutError, ConnectionError) as error:
            failure = ProviderFailure('transient')
            self.failed(failure)
            raise failure from error
        except ProviderFailure as error:
            self.failed(error)
            raise

    def failed(self, error: ProviderFailure) -> None:
        with self._state() as state:
            failures = min(int(state.get('failures', 0)) + 1, 6)
            retry = error.retry_after if math.isfinite(error.retry_after) and error.retry_after > 0 else 0
            state.update(failures=failures, last_failure=error.kind, blocked_until=self.clock() + max(retry, min(2 ** failures, 60)) + random.random())
            if error.kind == 'auth':
                state['auth_blocked'] = True

    def observe(self, headers: dict[str, Any]) -> None:
        with self._state() as state:
            lowered = {k.lower(): v for k, v in headers.items()}
            for item in state['limits']:
                if item['kind'] != 'REQUEST_WEIGHT':
                    continue
                duration = item['duration']
                for unit, seconds in reversed(tuple(SECONDS.items())):
                    if duration % seconds == 0:
                        value = lowered.get(f'x-mbx-used-weight-{duration // seconds}{unit[0].lower()}')
                        if value is not None:
                            item['used'] = max(item['used'], int(value))
                        break
