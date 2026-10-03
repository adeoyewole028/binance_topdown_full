"""Durable redacted incident transitions and explicitly authorized delivery."""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any
from urllib import request
from urllib.parse import urlsplit
import uuid

from operational_health import observations
from operations_policy import OperationsPolicy
from operations_recovery import BackupSchedule
from tools.http_transport import NoRedirect
from runtime_environment import parse_boolean
from state_store import StateStore, StateUnavailable


class Monitor:
    def __init__(self, store: StateStore) -> None:
        self.store = store
        self.path = store.path.parent / 'operations-alert.json'
        self.next_check = 0.0
        self.backups = BackupSchedule(store)
        self.destination = os.environ.get('OPERATOR_ALERT_WEBHOOK', '')
        self.authorized = parse_boolean(os.environ.get('OPERATOR_ALERT_DELIVERY_AUTHORIZED', 'false'))
        if self.authorized:
            parsed = urlsplit(self.destination)
            if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.fragment:
                raise ValueError('An explicitly authorized HTTPS private webhook is required')

    def step(self) -> None:
        now = time.time()
        if now < self.next_check:
            return
        self.next_check = now + OperationsPolicy().reconciliation_interval
        self.backups.step(now)
        report = observations(self.store, now)
        # No balances, credentials, URL, raw exception/provider messages or
        # arbitrary journal fields enter logs or outbound payloads.
        print(json.dumps({'event': 'operational_health', **report}), flush=True)
        incident = {key: report[key] for key in ('conditions', 'operation_ids', 'protection_ids', 'unprotected_symbols')}
        fingerprint = hashlib.sha256(json.dumps(incident, sort_keys=True).encode()).hexdigest()
        try:
            saved = json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else {}
            if saved.get('fingerprint') != fingerprint:
                saved = {'fingerprint': fingerprint, 'delivered': False,
                         'event_id': uuid.uuid4().hex, 'as_of': now, **incident}
                self._save(saved)
            if self.authorized and saved.get('delivered') is not True:
                try:
                    payload = json.dumps({key: saved[key] for key in
                        ('event_id', 'as_of', 'conditions', 'operation_ids', 'protection_ids', 'unprotected_symbols')}).encode()
                    req = request.Request(self.destination, data=payload,
                                          headers={'Content-Type': 'application/json'}, method='POST')
                    with request.build_opener(NoRedirect()).open(req, timeout=5) as response:
                        if not 200 <= response.status < 300:
                            raise StateUnavailable('Alert delivery unavailable')
                    saved['delivered'] = True
                    self._save(saved)
                except Exception:
                    print(json.dumps({'event': 'alert_delivery_unavailable', 'event_id': saved['event_id']}), flush=True)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise StateUnavailable('Operational incident persistence unavailable') from error

    def _save(self, value: dict[str, Any]) -> None:
        temporary = self.path.with_suffix('.tmp')
        with temporary.open('w', encoding='utf-8') as output:
            output.write(json.dumps(value))
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(self.path)
