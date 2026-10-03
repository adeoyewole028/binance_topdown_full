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
from tests.manual_browser_provider import ManualBrowserProvider
from tests.test_exchange_execution import configure
import time


def main() -> None:
    real_getaddrinfo=socket.getaddrinfo
    clock=ControlledClock()
    scenario=sys.argv[3] if len(sys.argv)>3 else 'paper'
    private=ManualBrowserProvider() if scenario!='paper' else None
    if private is not None:
        private.outcome=scenario
        clock.value=float(int(time.time()))
    directory=Path(sys.argv[1])
    environment={'PAPER_TRADING':'false','ALLOW_EXCHANGE_EXECUTION':'true'} if private else {}
    with LegacySandbox(directory,authenticated=False,paper_setup=False,environment=environment) as sandbox:
        sandbox.module.initialize_runtime(clock=clock)
        def public_market() -> MarketFixture:
            provider=MarketFixture(clock())
            if private:
                for symbol in provider.metadata()['symbols']:
                    symbol.update(ocoAllowed=True,orderTypes=['MARKET','STOP_LOSS','TAKE_PROFIT'])
                    symbol['filters'].append({'filterType':'PRICE_FILTER','minPrice':'0.01','maxPrice':'1000000','tickSize':'0.01'})
            return provider
        def fixture() -> Any:
            action=request.form['action']
            if action in {'scan','empty'}:
                if action=='empty':clock.value+=900
                provider=public_market() if private else SignalMarket(clock()) if action=='scan' else MarketFixture(clock())
                MarketRuntime(owner,provider,clock=clock,evaluate=sandbox.module.evaluate_market_snapshot).scan_once()
            elif action=='stale':clock.value+=61
            elif action=='process':owner.process_pending()
            elif action=='mark':owner.publish_prices({'BTC/USDT':101})
            elif action=='reconcile':
                clock.value+=15
                if private:private.visible=True
                owner.reconcile_exchange()
            elif action=='cancel-unknown':
                if private:private.timeout_cancel=True
            else:return jsonify(error='Unknown fixture action'),422
            return jsonify(ok=True)
        sandbox.module.app.add_url_rule('/__browser_fixture','browser_fixture',fixture,methods=['POST'])
        if private:
            sandbox.login();configure(sandbox)
        with ExecutionWorker(sandbox.module.state_store,clock=clock,exchange_provider=private) as owner:
            if private:
                MarketRuntime(owner,public_market(),clock=clock).scan_once()
                owner.reconcile_exchange()
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
