"""Explicit CI release of one verified revision. No automatic rollback/data reset."""
from __future__ import annotations

import os
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlsplit

if __package__ in {None, ''}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.http_transport import json_response


def main() -> int:
    token = os.environ.get('RENDER_API_KEY', '')
    service = os.environ.get('RENDER_SERVICE_ID', '')
    revision = os.environ.get('RELEASE_REVISION', '')
    health = os.environ.get('RELEASE_HEALTH_URL', '')
    parsed = urlsplit(health)
    if (not token or not re.fullmatch(r'srv-[a-z0-9]{20}', service)
            or not re.fullmatch(r'[a-f0-9]{40}', revision) or parsed.scheme != 'https'
            or not parsed.hostname or not parsed.hostname.endswith('.onrender.com')
            or parsed.path != '/healthz' or parsed.query or parsed.fragment
            or parsed.username or parsed.password or parsed.port not in {None, 443}):
        raise ValueError('Reviewed release configuration required')
    base = 'https://api.render.com/v1/services/' + service
    configured = json_response(base, token=token)
    if configured.get('autoDeploy') != 'no':
        raise ValueError('Disable competing automatic deployment before release')
    deployed = json_response(base + '/deploys', token=token,
                             data={'commitId': revision, 'clearCache': 'do_not_clear'})
    deployment = deployed.get('id')
    if not isinstance(deployment, str) or not re.fullmatch(r'dep-[a-z0-9]{20}', deployment):
        raise ValueError('Release identity unavailable; inspect provider before retry')
    for _ in range(90):
        observed = json_response(base + '/deploys/' + deployment, token=token)
        status = observed.get('status')
        if status == 'live':
            if observed.get('commit', {}).get('id') != revision:
                raise ValueError('Deployed revision differs from verified source')
            for _ in range(12):
                try:
                    if json_response(health).get('control_ready') is True:
                        print('Verified revision deployed; recovery service ready. Operator trading acceptance remains separate.')
                        return 0
                except Exception:
                    pass
                time.sleep(5)
            raise ValueError('Recovery service readiness unavailable')
        if status not in {'created', 'build_in_progress', 'update_in_progress', 'pre_deploy_in_progress'}:
            raise ValueError('Release did not become live; retain journal and inspect provider')
        time.sleep(5)
    raise ValueError('Release outcome unresolved; inspect existing deployment before retry')


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception:
        print('Release unavailable; preserve authoritative state and inspect the existing deployment before retry.', file=sys.stderr)
        raise SystemExit(1) from None
