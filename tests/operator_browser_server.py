"""Loopback-only browser fixture; fake providers, guarded disposable state."""
from __future__ import annotations
from pathlib import Path
import socket
import sys
from unittest.mock import patch
from typing import Any
from flask import jsonify,request
from werkzeug.serving import make_server
from tests.legacy_sandbox import LegacySandbox
from tests.test_market_runtime import ControlledClock,MarketFixture
from execution_worker import ExecutionWorker
from market_runtime import MarketRuntime
from tests.test_automation import SignalMarket


def main() -> None:
    real_getaddrinfo=socket.getaddrinfo
    clock=ControlledClock()
    directory=Path(sys.argv[1])
    with LegacySandbox(directory,authenticated=False) as sandbox:
        sandbox.module.initialize_runtime(clock=clock)
        with ExecutionWorker(sandbox.module.state_store,clock=clock) as owner:
            def fixture() -> Any:
                action=request.form['action']
                if action in {'scan','empty'}:
                    if action=='empty':clock.value+=900
                    provider=SignalMarket(clock()) if action=='scan' else MarketFixture(clock())
                    MarketRuntime(owner,provider,clock=clock,evaluate=sandbox.module.evaluate_market_snapshot).scan_once()
                elif action=='stale':clock.value+=61
                elif action=='process':owner.process_pending()
                else: return jsonify(error='Unknown fixture action'),422
                return jsonify(ok=True)
            sandbox.module.app.add_url_rule('/__browser_fixture','browser_fixture',fixture,methods=['POST'])
            def local_address(host: str,port: int,*args: Any,**kwargs: Any) -> Any:
                if host!='127.0.0.1':raise AssertionError('Only the loopback fixture may resolve')
                return real_getaddrinfo(host,port,*args,**kwargs)
            with patch('socket.getaddrinfo',local_address),patch('socket.getfqdn',lambda *args:'localhost'):
                server=make_server('127.0.0.1',int(sys.argv[2]),sandbox.module.app,threaded=False)
                try:
                    print(f'Offline browser fixture ready: http://127.0.0.1:{server.server_port}',flush=True)
                    server.serve_forever()
                finally:server.server_close()


if __name__=='__main__':main()
