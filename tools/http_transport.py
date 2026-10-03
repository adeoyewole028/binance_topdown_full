"""Bounded HTTPS JSON transport; redirects cannot forward private destinations."""
from __future__ import annotations

import json
from typing import Any
from urllib import request


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str,
                         headers: Any, newurl: str) -> None:
        raise ValueError('Redirect requires configuration review')


def json_response(url: str, *, token: str = '', data: dict[str, Any] | None = None) -> dict[str, Any]:
    headers = {'Accept': 'application/json'}
    if token:
        headers['Authorization'] = 'Bearer ' + token
    body = None
    if data is not None:
        headers['Content-Type'] = 'application/json'
        body = json.dumps(data).encode()
    req = request.Request(url, data=body, headers=headers)
    with request.build_opener(NoRedirect()).open(req, timeout=10) as response:
        raw = response.read(1048577)
        if len(raw) > 1048576 or not 200 <= response.status < 300:
            raise ValueError('Response unavailable')
        decoded = json.loads(raw)
        if not isinstance(decoded, dict):
            raise ValueError('Response contract unavailable')
        return decoded
