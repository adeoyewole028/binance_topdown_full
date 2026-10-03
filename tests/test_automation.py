"""Explicit automation through owned public data/intake and HTTP observations."""
from pathlib import Path
from typing import Any
import pytest
from tests.legacy_sandbox import LegacySandbox
from tests.test_paper_execution import setup
from tests.test_market_runtime import MarketFixture, ControlledClock
from tests.test_strategy_baseline import FIXTURES


class SignalMarket(MarketFixture):
    declining = False
    profile = 'dashboard_intraday'
    pattern = False
    def candles(self, symbol: str, timeframe: str, limit: int) -> list[list[float]]:
        self.calls.append((symbol,timeframe))
        keys={'1d':'trend_closes','4h':'bias_closes','15m':'cross_closes'} if self.profile == 'daemon_daily' else {'4h':'trend_closes','1h':'bias_closes','15m':'cross_closes'}
        key=keys[timeframe]
        closes=FIXTURES['profiles'][key]
        if self.declining and timeframe == '15m':
            closes=[100.0]*limit
            closes[-1]=99.0
        closes=[closes[0]]*(limit-len(closes))+closes
        duration={'1d':86400,'4h':14400,'1h':3600,'15m':900}[timeframe]
        boundary=int(self.now//duration)*duration
        rows=[[(boundary-(len(closes)-i)*duration)*1000,c,c+1,c-1,c,10] for i,c in enumerate(closes)]
        if self.pattern and timeframe == '15m':
            pattern=next(f for f in FIXTURES['candlesticks'] if f['name']=='hammer')
            for i,values in enumerate(reversed(pattern['rows']),start=1):rows[-i][1:5]=values
        return rows


@pytest.mark.parametrize('skew',[-30,30])
def test_verified_provider_clock_boundary_matches_automatic_intake(skew: int) -> None:
    from automation import Automation
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime
    clock=ControlledClock()
    clock.value+=15 if skew<0 else -15
    with LegacySandbox(paper_setup=False) as sandbox:
        setup(sandbox,fee_fraction='0',slippage_fraction='0',cooldown_seconds='0')
        sandbox.module.initialize_runtime(clock=clock)
        with ExecutionWorker(sandbox.module.state_store,clock=clock) as owner:
            control=Automation(owner,enabled=True,sizing='fixed')
            MarketRuntime(owner,SignalMarket(clock()+skew),clock=clock,evaluate=sandbox.module.evaluate_market_snapshot,automation=control).step()
            state=sandbox.client.get('/api/status').get_json()
            assert state['market_health']['execution_ready'] is True
            assert state['signals'][0]['execution_close'] == int((clock()+skew)//900)*900000
            assert state['operations'][0]['state'] == 'filled'
            assert len(state['positions']) == 1


def test_explicit_automation_uses_shared_fill_protection_and_restart_signal_dedup(tmp_path: Path) -> None:
    from automation import Automation
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime
    clock=ControlledClock()
    signal_identity = None
    for iteration in range(2):
        with LegacySandbox(tmp_path,paper_setup=False) as sandbox:
            if iteration == 0:
                assert setup(sandbox,fee_fraction='0',slippage_fraction='0',cooldown_seconds='0') == 302
            sandbox.module.initialize_runtime(clock=clock)
            with ExecutionWorker(sandbox.module.state_store,clock=clock) as owner:
                automation=Automation(owner,enabled=True,sizing='fixed')
                scanner=MarketRuntime(owner,SignalMarket(clock()),clock=clock,
                    evaluate=sandbox.module.evaluate_market_snapshot,automation=automation)
                scanner.step()
                automation.tick()
                owner.process_pending()
                state: dict[str,Any]=sandbox.client.get('/api/status').get_json()
                assert state['automation']['state'] == 'running'
                assert len(state['operations']) == 1
                assert state['operations'][0]['source'] == 'automation'
                if iteration == 0:
                    signal_identity = state['signals'][0]['signal_id']
                assert state['operations'][0]['signal_id'] == signal_identity
                assert state['operations'][0]['state'] == 'filled'
                assert state['positions'][0]['qty'] == '1'
                assert state['protection'][-1]['state'] == 'active'
                assert len(state['fills']) == 1
                assert sandbox.exchange.orders == []
                assert sandbox.network_attempts == 0


def test_confirmed_close_and_restart_cannot_reenter_the_same_execution_signal(tmp_path: Path) -> None:
    from automation import Automation
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime
    clock=ControlledClock()
    for iteration in range(2):
        with LegacySandbox(tmp_path,paper_setup=False) as sandbox:
            if iteration == 0:
                setup(sandbox,fee_fraction='0',slippage_fraction='0',cooldown_seconds='0')
            sandbox.module.initialize_runtime(clock=clock)
            with ExecutionWorker(sandbox.module.state_store,clock=clock) as owner:
                control=Automation(owner,enabled=True,sizing='fixed')
                scanner=MarketRuntime(owner,SignalMarket(clock()),clock=clock,
                    evaluate=sandbox.module.evaluate_market_snapshot,automation=control)
                scanner.step()
                if iteration == 0:
                    state=sandbox.client.get('/api/status').get_json()
                    sandbox.client.post('/close',data={'symbol':'BTC/USDT','csrf_token':sandbox.csrf,'order_review':state['order_review']})
                    owner.process_pending()
                    clock.value+=1
                    scanner.scan_once()
                control.tick()
                owner.process_pending()
                state=sandbox.client.get('/api/status').get_json()
                assert state['positions'] == []
                assert len([o for o in state['operations'] if o['source']=='automation']) == 1
                assert len(state['fills']) == 2


@pytest.mark.parametrize('enabled',[False,True])
def test_default_disabled_and_graceful_exit_have_truthful_status_without_canceling_protection(enabled: bool) -> None:
    from automation import Automation
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime
    clock=ControlledClock()
    with LegacySandbox(paper_setup=False) as sandbox:
        setup(sandbox,fee_fraction='0',slippage_fraction='0',cooldown_seconds='0')
        sandbox.module.initialize_runtime(clock=clock)
        with ExecutionWorker(sandbox.module.state_store,clock=clock) as owner:
            control=Automation(owner,enabled=enabled,sizing='fixed')
            scanner=MarketRuntime(owner,SignalMarket(clock()),clock=clock,
                evaluate=sandbox.module.evaluate_market_snapshot,automation=control)
            scanner.step()
            state=sandbox.client.get('/api/status').get_json()
            assert state['automation']['state'] == ('running' if enabled else 'disabled')
            assert len(state['fills']) == (1 if enabled else 0)
        state=sandbox.client.get('/api/status').get_json()
        assert state['automation']['state'] == ('stopped' if enabled else 'disabled')
        assert state['ownership']['state'] == 'stopped'
        if enabled:assert state['protection'][-1]['state'] == 'active'


def test_automatic_indicator_exit_uses_shared_close_and_original_profile_rules() -> None:
    from automation import Automation
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime
    clock=ControlledClock()
    with LegacySandbox(paper_setup=False) as sandbox:
        setup(sandbox,fee_fraction='0',slippage_fraction='0',cooldown_seconds='0')
        sandbox.module.initialize_runtime(clock=clock)
        market=SignalMarket(clock())
        with ExecutionWorker(sandbox.module.state_store,clock=clock) as owner:
            control=Automation(owner,enabled=True,sizing='fixed')
            scanner=MarketRuntime(owner,market,clock=clock,evaluate=sandbox.module.evaluate_market_snapshot,automation=control)
            scanner.step()
            before=sandbox.client.get('/api/status').get_json()
            original_version=before['config_version']
            sandbox.post('/update-config',data={'rsi_overbought':'99'})
            clock.value+=900
            market.now=clock()
            market.declining=True
            scanner.step()
            after=sandbox.client.get('/api/status').get_json()
            assert after['positions'] == []
            close=[o for o in after['operations'] if o['kind']=='close'][0]
            assert close['source'] == 'automation'
            assert close['signal_config_version'] == original_version
            assert close['config_version'] == after['config_version']
            assert close['state'] == 'filled'
            assert close['requested_qty'] == '1'
            assert len(after['fills']) == 2
            assert after['protection'][-1]['state'] == 'closed'
            assert sandbox.exchange.orders == []


@pytest.mark.parametrize('problem',['unset_setup','stale_scan','changed_config'])
def test_automation_does_not_intake_without_current_setup_and_completed_signal(problem: str) -> None:
    from automation import Automation
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime
    clock=ControlledClock()
    with LegacySandbox(paper_setup=False) as sandbox:
        if problem != 'unset_setup':setup(sandbox,fee_fraction='0',slippage_fraction='0')
        sandbox.module.initialize_runtime(clock=clock)
        with ExecutionWorker(sandbox.module.state_store,clock=clock) as owner:
            control=Automation(owner,enabled=True,sizing='fixed')
            scanner=MarketRuntime(owner,SignalMarket(clock()),clock=clock,evaluate=sandbox.module.evaluate_market_snapshot)
            scanner.step()
            if problem == 'stale_scan':clock.value+=900
            if problem == 'changed_config':sandbox.post('/update-config',data={'rsi_overbought':'99'})
            control.tick()
            owner.process_pending()
            state=sandbox.client.get('/api/status').get_json()
            assert state['fills'] == []
            assert state['operations'] == []
            assert sandbox.network_attempts == 0


@pytest.mark.parametrize(('profile','sizing'),[('dashboard_intraday','risk'),('daemon_daily','fixed')])
def test_explicit_profiles_and_sizing_generate_confirmed_owned_entries(profile: str, sizing: str) -> None:
    from automation import Automation
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime
    clock=ControlledClock()
    with LegacySandbox(paper_setup=False) as sandbox:
        setup(sandbox,fee_fraction='0',slippage_fraction='0',cooldown_seconds='0')
        sandbox.post('/update-config',data={'strategy_profile':profile})
        sandbox.module.initialize_runtime(clock=clock)
        market=SignalMarket(clock())
        market.profile=profile
        market.pattern=profile=='daemon_daily'
        with ExecutionWorker(sandbox.module.state_store,clock=clock) as owner:
            control=Automation(owner,enabled=True,sizing=sizing)
            MarketRuntime(owner,market,clock=clock,evaluate=sandbox.module.evaluate_market_snapshot,automation=control).step()
            state=sandbox.client.get('/api/status').get_json()
            assert len(state['positions']) == 1
            assert state['operations'][0]['signal_profile'] == profile
            assert state['operations'][0]['state'] == 'filled'
            assert state['protection'][-1]['state'] == 'active'


def test_durable_cooldown_and_confirmed_exit_slots_survive_automation_restart(tmp_path: Path) -> None:
    from automation import Automation
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime
    clock=ControlledClock()
    for iteration in range(3):
        with LegacySandbox(tmp_path,paper_setup=False) as sandbox:
            if iteration == 0:
                setup(sandbox,fee_fraction='0',slippage_fraction='0',cooldown_seconds='3600',max_positions='1')
                sandbox.post('/update-config',data={'cooldown':'0'})
            sandbox.module.initialize_runtime(clock=clock)
            with ExecutionWorker(sandbox.module.state_store,clock=clock) as owner:
                control=Automation(owner,enabled=True,sizing='fixed')
                MarketRuntime(owner,SignalMarket(clock()),clock=clock,evaluate=sandbox.module.evaluate_market_snapshot,automation=control).step()
                state=sandbox.client.get('/api/status').get_json()
                if iteration == 0:
                    assert len(state['positions']) == 1
                    sandbox.client.post('/close',data={'symbol':'BTC/USDT','csrf_token':sandbox.csrf,'order_review':state['order_review']})
                    owner.process_pending()
                    assert sandbox.client.get('/api/status').get_json()['positions'] == []
                elif iteration == 1:
                    assert state['positions'] == []
                    assert state['operations'][-1]['state'] == 'rejected'
                    assert state['operations'][-1]['error_class'] == 'entry_cooldown'
                    assert state['automation']['state'] == 'blocked'
                else:
                    assert len(state['positions']) == 1
                    assert len(state['fills']) == 3
            clock.value+=900 if iteration == 0 else 3600


def test_reserved_position_cap_applies_to_a_batch_of_automatic_signals() -> None:
    from automation import Automation
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime
    clock=ControlledClock()
    with LegacySandbox(paper_setup=False) as sandbox:
        setup(sandbox,fee_fraction='0',slippage_fraction='0',cooldown_seconds='0',max_positions='1')
        sandbox.module.initialize_runtime(clock=clock)
        market=SignalMarket(clock())
        market.symbols.append({**market.symbols[0],'symbol':'ETHUSDT','baseAsset':'ETH'})
        market.prices['ETH/USDT']=dict(market.prices['BTC/USDT'])
        with ExecutionWorker(sandbox.module.state_store,clock=clock) as owner:
            control=Automation(owner,enabled=True,sizing='fixed')
            MarketRuntime(owner,market,clock=clock,evaluate=sandbox.module.evaluate_market_snapshot,automation=control).step()
            state=sandbox.client.get('/api/status').get_json()
            assert len(state['positions']) == 1
            assert [o['state'] for o in state['operations']] == ['filled','rejected']
            assert len(state['fills']) == 1
            assert state['automation']['state'] == 'blocked'


@pytest.mark.parametrize('exclusion',['whitelist','blacklist'])
def test_automation_preserves_entry_symbol_exclusions(exclusion: str) -> None:
    from automation import Automation
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime
    clock=ControlledClock()
    with LegacySandbox() as sandbox:
        sandbox.module.initialize_runtime(clock=clock)
        with ExecutionWorker(sandbox.module.state_store,clock=clock) as owner:
            control=Automation(owner,enabled=True)
            MarketRuntime(owner,SignalMarket(clock()),clock=clock,evaluate=sandbox.module.evaluate_market_snapshot,automation=control,
                whitelist={'ETH/USDT'} if exclusion=='whitelist' else None,
                blacklist={'BTC/USDT'} if exclusion=='blacklist' else None).step()
            assert sandbox.client.get('/api/status').get_json()['operations'] == []


def test_graceful_launcher_stop_during_public_scan_prevents_automatic_intake() -> None:
    from execution_worker import ExecutionWorker
    from run_worker import run_owned_loop
    clock=ControlledClock()
    running=[True]
    class StoppingMarket(SignalMarket):
        def metadata(self) -> dict[str,Any]:
            running[0]=False
            return super().metadata()
    with LegacySandbox() as sandbox:
        sandbox.module.initialize_runtime(clock=clock)
        with ExecutionWorker(sandbox.module.state_store,clock=clock) as owner:
            run_owned_loop(owner,StoppingMarket(clock()),automate=True,should_run=lambda: running[0])
            state=sandbox.client.get('/api/status').get_json()
            assert state['automation']['state'] == 'stopped'
            assert state['operations'] == []


def test_env_flag_and_reset_do_not_implicitly_start_automation() -> None:
    from run_worker import main
    from unittest.mock import patch
    with LegacySandbox(environment={'AUTOMATION_ENABLED':'true'}) as sandbox:
        sandbox.post('/reset-config',data={})
        assert sandbox.client.get('/api/status').get_json()['automation']['state'] == 'disabled'
        with patch('dotenv.load_dotenv'):
            main(['--once'])
        state=sandbox.client.get('/api/status').get_json()
        assert state['automation']['state'] == 'disabled'
        assert sandbox.network_attempts == 0


def test_explicit_launcher_rejects_automation_without_server_enable_gate() -> None:
    from run_worker import main
    from unittest.mock import patch
    with LegacySandbox() as sandbox:
        with patch('dotenv.load_dotenv'), pytest.raises(ValueError,match='disabled'):
            main(['--automate'])
        assert sandbox.client.get('/api/status').get_json()['automation']['state'] == 'disabled'
        assert sandbox.exchange.orders == []


def test_unresolved_startup_keeps_automation_waiting_and_preserves_unknown_intent(tmp_path: Path) -> None:
    import sqlite3
    from automation import Automation
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime
    clock=ControlledClock()
    with LegacySandbox(tmp_path) as sandbox:
        sandbox.seed_price('BTC/USDT',100)
        sandbox.client.post('/trade',data={'symbol':'BTC/USDT','side':'buy','qty':'1','csrf_token':sandbox.csrf})
        operation=sandbox.client.get('/api/status').get_json()['operations'][0]['operation_id']
        with sqlite3.connect(tmp_path/'trading.sqlite3') as fixture:
            fixture.execute("UPDATE intents SET state='unknown' WHERE operation_id=?",(operation,))
        sandbox.module.initialize_runtime(clock=clock)
        with ExecutionWorker(sandbox.module.state_store,clock=clock) as owner:
            control=Automation(owner,enabled=True)
            MarketRuntime(owner,SignalMarket(clock()),clock=clock,evaluate=sandbox.module.evaluate_market_snapshot,automation=control).step()
            state=sandbox.client.get('/api/status').get_json()
            assert state['automation']['state'] == 'waiting_for_reconciliation'
            assert len(state['operations']) == 1
            assert state['operations'][0]['state'] == 'unknown'
            assert state['positions'] == []
        assert sandbox.client.get('/api/status').get_json()['operations'][0]['state'] == 'unknown'


class ProtectedSignalMarket(SignalMarket):
    def metadata(self) -> dict[str,Any]:
        values=super().metadata()
        for symbol in values['symbols']:
            symbol.update(ocoAllowed=True,orderTypes=['MARKET','STOP_LOSS','TAKE_PROFIT'])
            if not any(f['filterType']=='PRICE_FILTER' for f in symbol['filters']):
                symbol['filters'].append({'filterType':'PRICE_FILTER','minPrice':'0.01','maxPrice':'1000000','tickSize':'0.01'})
        return values


def test_automatic_exchange_entry_and_unknown_protection_preserve_confirmed_holding(tmp_path: Path) -> None:
    import time
    from tests.test_exchange_execution import SpotFixture, configure
    from automation import Automation
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime
    class LostList(SpotFixture):
        def query_list(self, list_id: str) -> dict[str,Any] | None:
            return None
    provider=LostList()
    provider.timeout_install=True
    clock=ControlledClock()
    clock.value=float(int(time.time()))
    for iteration in range(2):
        with LegacySandbox(tmp_path,paper_setup=False,environment={'PAPER_TRADING':'false','ALLOW_EXCHANGE_EXECUTION':'true'}) as sandbox:
            if iteration == 0:configure(sandbox,fee_fraction='0.01',slippage_fraction='0',cooldown_seconds='0')
            sandbox.module.initialize_runtime(clock=clock)
            with ExecutionWorker(sandbox.module.state_store,clock=clock,exchange_provider=provider) as owner:
                control=Automation(owner,enabled=True,sizing='fixed')
                scanner=MarketRuntime(owner,ProtectedSignalMarket(clock()),clock=clock,evaluate=sandbox.module.evaluate_market_snapshot,
                    reconcile=owner.poll_exchange,automation=control)
                scanner.step()
                if iteration == 0:
                    # Initial identity reconciliation waits for public metadata;
                    # the next owned reconciliation is required before intake.
                    assert sandbox.client.get('/api/status').get_json()['operations'] == []
                    owner.reconcile_exchange()
                control.tick()
                owner.process_pending()
                control.tick()
                state=sandbox.client.get('/api/status').get_json()
                assert len(state['positions']) == 1, (iteration,state['automation'],state['operations'],state['exchange_risk'])
                assert state['protection'][-1]['state'] == 'unknown'
                assert state['exchange_risk']['entry_blocked'] is True
                assert len(provider.submissions) == 1
                assert len(provider.protect_submissions) == 1
                assert len(state['fills']) == 1
                assert state['automation']['state'] in {'waiting_for_reconciliation','blocked'}
                assert sandbox.network_attempts == 0
            clock.value+=15


def test_exchange_entry_pause_still_allows_automatic_confirmed_indicator_exit() -> None:
    import time
    from tests.test_exchange_execution import SpotFixture, configure
    from automation import Automation
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime
    clock=ControlledClock()
    clock.value=float(int(time.time()))
    provider=SpotFixture()
    with LegacySandbox(paper_setup=False,environment={'PAPER_TRADING':'false','ALLOW_EXCHANGE_EXECUTION':'true'}) as sandbox:
        configure(sandbox,fee_fraction='0.01',slippage_fraction='0',cooldown_seconds='0')
        sandbox.module.initialize_runtime(clock=clock)
        market=ProtectedSignalMarket(clock())
        with ExecutionWorker(sandbox.module.state_store,clock=clock,exchange_provider=provider) as owner:
            control=Automation(owner,enabled=True,sizing='fixed')
            scanner=MarketRuntime(owner,market,clock=clock,evaluate=sandbox.module.evaluate_market_snapshot,
                reconcile=owner.poll_exchange,automation=control)
            scanner.step()
            assert sandbox.client.get('/api/status').get_json()['operations'] == []
            owner.reconcile_exchange()
            control.tick()
            owner.process_pending()
            assert len(sandbox.client.get('/api/status').get_json()['positions']) == 1
            sandbox.module.state_store.set_entry_pause(True,actor='configured-operator')
            clock.value+=900
            market.now=clock()
            market.declining=True
            scanner.step()
            state=sandbox.client.get('/api/status').get_json()
            assert state['positions'] == []
            assert len(provider.submissions) == 2
            close=[o for o in state['operations'] if o['kind']=='close'][0]
            assert close['source'] == 'automation'
            assert close['state'] == 'filled'
            assert len(provider.cancel_submissions) == 1
            assert state['entry_control']['paused'] is True


def test_actual_automation_process_fences_second_launcher_and_stops_durably(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys
    from unittest.mock import patch
    from run_worker import main
    from execution_owner import OwnershipUnavailable
    environment={key:value for key,value in os.environ.items()
        if not key.startswith(('BINANCE_','OPERATOR_','ALLOW_','TRADING_'))
        and key not in {'API_KEY','API_SECRET','FLASK_SECRET_KEY','DATA_DIR'}}
    with LegacySandbox(tmp_path,environment={'AUTOMATION_ENABLED':'true'}) as sandbox:
        child=subprocess.Popen([sys.executable,'-m','tests.automation_process',str(tmp_path)],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,env=environment)
        try:
            assert child.stdout is not None and child.stdin is not None
            assert child.stdout.readline().strip() == 'owned'
            with patch('dotenv.load_dotenv'),pytest.raises(OwnershipUnavailable):main(['--automate'])
            assert sandbox.client.get('/api/status').get_json()['operations'] == []
            child.stdin.write('\n')
            child.stdin.flush()
            assert child.wait(timeout=15) == 0
            state=sandbox.client.get('/api/status').get_json()
            assert state['automation']['state'] == 'stopped'
            assert state['ownership']['state'] == 'stopped'
            with patch('dotenv.load_dotenv'):main(['--once'])
            assert sandbox.client.get('/api/status').get_json()['automation']['state'] == 'disabled'
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=15)
            for stream in (child.stdin,child.stdout,child.stderr):
                if stream is not None:stream.close()


def test_shutdown_status_write_failure_still_releases_owner_and_retains_protection(tmp_path: Path) -> None:
    import sqlite3
    from automation import Automation
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime
    from state_store import StateUnavailable
    clock=ControlledClock()
    with LegacySandbox(tmp_path,paper_setup=False) as sandbox:
        setup(sandbox,fee_fraction='0',slippage_fraction='0',cooldown_seconds='0')
        sandbox.module.initialize_runtime(clock=clock)
        with pytest.raises(StateUnavailable):
            with ExecutionWorker(sandbox.module.state_store,clock=clock) as owner:
                control=Automation(owner,enabled=True,sizing='fixed')
                MarketRuntime(owner,SignalMarket(clock()),clock=clock,evaluate=sandbox.module.evaluate_market_snapshot,automation=control).step()
                with sqlite3.connect(tmp_path/'trading.sqlite3') as fixture:
                    fixture.execute("CREATE TRIGGER stop_write_fault BEFORE INSERT ON journal WHEN NEW.event='automation_status' BEGIN SELECT RAISE(ABORT,'offline shutdown write failure'); END")
        state=sandbox.client.get('/api/status').get_json()
        assert state['ownership']['state'] == 'stopped'
        assert state['automation']['state'] == 'stopped'
        assert state['positions'][0]['qty'] == '1'
        assert state['protection'][-1]['state'] == 'active'
        with ExecutionWorker(sandbox.module.state_store,clock=clock):
            assert sandbox.client.get('/api/status').get_json()['automation']['state'] == 'waiting_for_reconciliation'
