"""CI release contract at the fake external HTTPS transport boundary."""
from __future__ import annotations

import io
import json
import os
from typing import Any, Self
from unittest.mock import patch

import pytest

from tools.release_verified import main


SERVICE = 'srv-' + 'a' * 20
DEPLOYMENT = 'dep-' + 'b' * 20
REVISION = 'c' * 40
HEALTH = 'https://offline-reviewed.onrender.com/healthz'
ENV = {'RENDER_API_KEY': 'fake-release-key', 'RENDER_SERVICE_ID': SERVICE,
       'RELEASE_REVISION': REVISION, 'RELEASE_HEALTH_URL': HEALTH,
       'RELEASE_APPROVED_PLAN': '1c-2g'}


class Response(io.BytesIO):
    status = 200
    def __enter__(self) -> Self:
        return self
    def __exit__(self, *args: Any) -> None:
        self.close()


@pytest.mark.parametrize('conflict', [None, 'competing_deploy', 'wrong_host', 'multiple_instances', 'wrong_commit', 'unknown_reply'])
def test_release_uses_verified_revision_once_and_refuses_conflicting_external_contract(conflict: str | None) -> None:
    calls = []
    class Provider:
        def open(self, req: Any, **kwargs: Any) -> Response:
            calls.append(req)
            value: dict[str, Any]
            if req.get_method() == 'POST':
                assert json.loads(req.data)['commitId'] == REVISION
                if conflict == 'unknown_reply':
                    raise TimeoutError('fake reply lost after accepted deploy')
                value = {'id': DEPLOYMENT}
            elif req.full_url == HEALTH:
                assert req.get_header('Authorization') is None
                value = {'control_ready': True}
            elif '/deploys/' in req.full_url:
                value = {'status': 'live', 'commit': {'id': 'd' * 40 if conflict == 'wrong_commit' else REVISION}}
            else:
                value = {'id': SERVICE, 'type': 'web_service',
                         'autoDeploy': 'yes' if conflict == 'competing_deploy' else 'no',
                         'serviceDetails': {'plan': '1c-2g', 'numInstances': 2 if conflict == 'multiple_instances' else 1,
                            'disk': {'mountPath': '/var/data/trading'}, 'healthCheckPath': '/healthz',
                            'url': 'https://different.onrender.com' if conflict == 'wrong_host' else HEALTH.removesuffix('/healthz')}}
            return Response(json.dumps(value).encode())
    with patch.dict(os.environ, ENV, clear=True), patch('urllib.request.build_opener', return_value=Provider()):
        if conflict is None:
            assert main() == 0
        else:
            with pytest.raises((ValueError, TimeoutError)):
                main()
    posts = [req for req in calls if req.get_method() == 'POST']
    assert len(posts) == (0 if conflict in {'competing_deploy', 'wrong_host', 'multiple_instances'} else 1)


def test_missing_budget_or_release_authority_never_contacts_provider() -> None:
    with patch.dict(os.environ, {}, clear=True), patch('urllib.request.build_opener') as transport:
        with pytest.raises(ValueError):
            main()
        transport.assert_not_called()
