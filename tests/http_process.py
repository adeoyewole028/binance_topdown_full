"""Disposable second HTTP process at the same approved operator boundary."""
import json
import os
from pathlib import Path
import sys

from tests.legacy_sandbox import LegacySandbox


def main() -> None:
    with LegacySandbox(Path(sys.argv[1]), environment={'OPERATOR_PASSWORD_HASH': os.environ['BT03_OFFLINE_HASH']}) as sandbox:
        response = sandbox.post('/update-config', data={'risk': sys.argv[2]})
        assert response.status_code == 302
        payload = sandbox.client.get('/api/status').get_json()
        print(json.dumps(payload))
        assert sandbox.network_attempts == 0
        assert sandbox.thread_start_attempts == 0


if __name__ == '__main__':
    main()
