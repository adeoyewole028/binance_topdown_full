"""Versioned strategy decisions at the approved pure completed-bar seam."""
from datetime import datetime, timezone
from typing import Any
from pathlib import Path
import pytest
from tests.test_strategy_baseline import FIXTURES
from tests.legacy_sandbox import LegacySandbox

AS_OF = datetime(2026, 10, 3, 12, tzinfo=timezone.utc)

def bars(closes: list[float], seconds: int) -> list[list[float]]:
    boundary = int(AS_OF.timestamp() // seconds) * seconds
    return [[(boundary-(len(closes)-i)*seconds)*1000, c, c+1, c-1, c, 10] for i,c in enumerate(closes)]

def inputs(profile: str) -> dict[str, list[list[float]]]:
    f = FIXTURES['profiles']
    trend, bias = ('1d', '4h') if profile == 'daemon_daily' else ('4h', '1h')
    durations = {'1d': 86400, '4h': 14400, '1h': 3600, '15m': 900}
    return {trend: bars(f['trend_closes'], durations[trend]), bias: bars(f['bias_closes'], durations[bias]),
            '15m': bars(f['cross_closes'],900)}

def test_named_profiles_preserve_literal_legacy_cross_and_bias_outputs() -> None:
    from strategy_profiles import evaluate
    with LegacySandbox() as sandbox:
        config = sandbox.module.state_store.config()[1]
        for profile, expected in [('dashboard_intraday', True), ('daemon_daily', False)]:
            result: dict[str, Any] = evaluate(namespace='paper:testnet:test',venue='testnet',symbol='BTC/USDT',
                profile=profile,config_version=1,config=config,bars=inputs(profile),as_of=AS_OF)
            assert result['state'] == 'ready'
            assert result['trend'] == FIXTURES['profiles']['trend_expected']
            assert result['bias'] == FIXTURES['profiles']['bias_expected']
            assert result['entry'] is expected
            assert result['execution'] == FIXTURES['profiles']['dashboard_cross_expected' if expected else 'daemon_cross_expected']
            assert result['profile'] == profile
            assert result['profile_version'] == 1
        assert sandbox.exchange.orders == []
        assert sandbox.network_attempts == 0


def test_missing_completed_bar_is_unavailable_instead_of_a_valid_signal() -> None:
    from strategy_profiles import evaluate
    with LegacySandbox() as sandbox:
        rows = inputs('dashboard_intraday')
        del rows['15m'][-3]
        result = evaluate(namespace='paper:testnet:test',venue='testnet',symbol='BTC/USDT',
            profile='dashboard_intraday',config_version=1,config=sandbox.module.state_store.config()[1],
            bars=rows,as_of=AS_OF)
        assert result['state'] == 'unavailable'
        assert result['reason'] == 'gapped_history'
        assert result['entry'] is False


def test_bound_position_exit_precedes_entry_and_retains_take_profit_reason() -> None:
    from strategy_profiles import evaluate
    with LegacySandbox() as sandbox:
        result = evaluate(namespace='paper:testnet:test',venue='testnet',symbol='BTC/USDT',
            profile='dashboard_intraday',config_version=1,config=sandbox.module.state_store.config()[1],
            bars=inputs('dashboard_intraday'),as_of=AS_OF,position={'entry':100,'qty':1,'price':120})
        assert result['state'] == 'ready'
        assert result['exit_reasons'] == ['take_profit']
        assert result['entry'] is False


def test_owned_scans_publish_stable_profile_identity_across_repeat_and_restart(tmp_path: Path) -> None:
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime
    from tests.test_market_runtime import MarketFixture, ControlledClock
    class GoldenMarket(MarketFixture):
        def candles(self, symbol: str, timeframe: str, limit: int) -> list[list[float]]:
            f = FIXTURES['profiles']
            closes = f[{'4h':'trend_closes','1h':'bias_closes','15m':'cross_closes'}[timeframe]]
            duration = {'4h':14400,'1h':3600,'15m':900}[timeframe]
            padded = [closes[0]]*(limit-len(closes))+closes
            return bars(padded,duration)
    clock = ControlledClock()
    identities = []
    for _ in range(2):
        with LegacySandbox(tmp_path) as sandbox:
            sandbox.module.initialize_runtime(clock=clock)
            with ExecutionWorker(sandbox.module.state_store,clock=clock) as owner:
                scanner = MarketRuntime(owner,GoldenMarket(clock()),clock=clock,evaluate=sandbox.module.evaluate_market_snapshot)
                for _ in range(2):
                    scanner.scan_once()
                    state = sandbox.client.get('/api/status').get_json()
                    signal = state['signals'][0]
                    identities.append(signal['signal_id'])
                    assert signal['profile'] == 'dashboard_intraday'
                    assert signal['profile_version'] == 1
                    assert signal['config_version'] == state['config_version']
                assert sandbox.network_attempts == 0
    assert len(set(identities)) == 1


@pytest.mark.parametrize('profile',['dashboard_intraday','daemon_daily'])
def test_future_forming_values_timezone_and_global_changes_cannot_rewrite_past_decision(profile: str) -> None:
    from copy import deepcopy
    from datetime import timedelta
    from types import MappingProxyType
    from strategy_profiles import evaluate
    with LegacySandbox() as sandbox:
        config = MappingProxyType(sandbox.module.state_store.config()[1])
        rows = inputs(profile)
        before = deepcopy(rows)
        kwargs: dict[str, Any] = dict(namespace='paper:testnet:test',venue='testnet',symbol='BTC/USDT',
                      profile=profile,config_version=1,config=config)
        original = evaluate(**kwargs,bars=rows,as_of=AS_OF)
        assert rows == before
        for timeframe,series in rows.items():
            duration = {'1d':86400,'4h':14400,'1h':3600,'15m':900}[timeframe]
            boundary = int(AS_OF.timestamp() // duration)*duration
            series.extend([[boundary*1000,float('nan'),-1,-1,float('inf'),-1],
                           [(boundary+duration)*1000,0,0,0,0,0]])
        sandbox.module.status['config']['rsi_exec_threshold'] = 99
        offset = AS_OF.astimezone(timezone(timedelta(hours=1)))
        assert evaluate(**kwargs,bars=rows,as_of=offset) == original
        assert sandbox.network_attempts == 0
        assert sandbox.exchange.orders == []


def test_dashboard_and_daemon_adapters_use_the_same_explicit_daily_profile() -> None:
    from market_runtime import CachedMarketView
    with LegacySandbox() as sandbox:
        daemon = sandbox.load_module('topdown_daemon')
        config = {**sandbox.module.state_store.config()[1],'strategy_profile':'daemon_daily'}
        cache = {f'testnet:BTC/USDT:{t}':{'rows':r} for t,r in inputs('daemon_daily').items()}
        view = CachedMarketView('testnet',cache,{'BTC/USDT':{'last':100}},context={
            'namespace':'paper:testnet:test','config_version':7,'as_of':AS_OF,'positions':[],'position_rules':{}})
        left = sandbox.module.evaluate_market_snapshot(view,['BTC/USDT'],config)
        right = daemon.evaluate_market_snapshot(view,['BTC/USDT'],config)
        assert left == right
        assert left['strategy_decisions'][0]['profile'] == 'daemon_daily'
        assert left['strategy_decisions'][0]['entry'] is False
        assert sandbox.network_attempts == 0


def test_explicit_profile_config_and_new_intent_retain_original_version_on_later_change() -> None:
    with LegacySandbox() as sandbox:
        before = sandbox.client.get('/api/status').get_json()
        assert sandbox.post('/update-config',data={'strategy_profile':'invented'}).status_code == 400
        assert sandbox.client.get('/api/status').get_json()['config_version'] == before['config_version']
        assert sandbox.post('/update-config',data={'strategy_profile':'daemon_daily'}).status_code == 302
        sandbox.seed_price('BTC/USDT',100)
        sandbox.client.post('/trade',data={'symbol':'BTC/USDT','side':'buy','qty':'1','csrf_token':sandbox.csrf})
        sandbox.post('/update-config',data={'strategy_profile':'dashboard_intraday'})
        state = sandbox.client.get('/api/status').get_json()
        assert state['operations'][0]['strategy_profile'] == 'daemon_daily'
        assert state['operations'][0]['strategy_version'] == 1
        assert state['operations'][0]['config_version'] < state['config_version']


@pytest.mark.parametrize(('damage','reason'),[('warmup','insufficient_history'),('stale','stale_history'),
    ('duplicate','gapped_history'),('unsorted','gapped_history'),('ohlc','invalid_ohlc'),('nonfinite','invalid_bar')])
def test_invalid_completed_history_never_emits_a_signal(damage: str, reason: str) -> None:
    from strategy_profiles import evaluate
    with LegacySandbox() as sandbox:
        data = inputs('dashboard_intraday')
        series = data['15m']
        if damage == 'warmup':data['15m']=series[-20:]
        elif damage == 'stale':series.pop()
        elif damage == 'duplicate':series[-2][0]=series[-3][0]
        elif damage == 'unsorted':series[-3],series[-2]=series[-2],series[-3]
        elif damage == 'ohlc':series[-1][2]=series[-1][3]-1
        elif damage == 'nonfinite':series[-1][4]=float('nan')
        result = evaluate(namespace='paper:testnet:test',venue='testnet',symbol='BTC/USDT',
            profile='dashboard_intraday',config_version=1,config=sandbox.module.state_store.config()[1],bars=data,as_of=AS_OF)
        assert result['state'] == 'unavailable'
        assert result['reason'] == reason
        assert result['entry'] is False
        assert result['exit_reasons'] == []


def test_exact_execution_boundary_and_no_daily_final_close_before_utc_day_close() -> None:
    from datetime import timedelta
    from strategy_profiles import evaluate
    with LegacySandbox() as sandbox:
        data = inputs('daemon_daily')
        config = sandbox.module.state_store.config()[1]
        kwargs: dict[str, Any] = dict(namespace='paper:testnet:test',venue='testnet',symbol='BTC/USDT',profile='daemon_daily',config_version=1,config=config)
        past = evaluate(**kwargs,bars=data,as_of=AS_OF-timedelta(seconds=1))
        assert past['state'] == 'ready'
        assert past['execution_close'] == int(AS_OF.timestamp()*1000)-900000
        data['1d'].append([int(AS_OF.replace(hour=0).timestamp()*1000),-1,-1,-1,-1,-1])
        assert evaluate(**kwargs,bars=data,as_of=AS_OF-timedelta(seconds=1)) == past
        with pytest.raises(ValueError,match='Timezone-aware'):
            evaluate(**kwargs,bars=data,as_of=AS_OF.replace(tzinfo=None))


@pytest.mark.parametrize('profile',['dashboard_intraday','daemon_daily'])
@pytest.mark.parametrize(('fixture_name','expected'),[('engulfing','bullish_engulfing'),('hammer','hammer')])
def test_full_profiles_preserve_candlestick_priority(profile: str, fixture_name: str, expected: str) -> None:
    from strategy_profiles import evaluate
    with LegacySandbox() as sandbox:
        data = inputs(profile)
        pattern = next(f for f in FIXTURES['candlesticks'] if f['name'] == fixture_name)
        for i,values in enumerate(reversed(pattern['rows']),start=1):
            data['15m'][-i][1:5] = values
        result = evaluate(namespace='paper:testnet:test',venue='testnet',symbol='BTC/USDT',profile=profile,
            config_version=1,config=sandbox.module.state_store.config()[1],bars=data,as_of=AS_OF)
        assert result['state'] == 'ready'
        assert result['entry'] is True
        assert result['execution']['type'] == expected


def test_daily_runtime_fetches_documented_windows_and_exposes_profile_decisions() -> None:
    from execution_worker import ExecutionWorker
    from market_runtime import MarketRuntime
    from tests.test_market_runtime import MarketFixture, ControlledClock
    class DailyMarket(MarketFixture):
        def candles(self, symbol: str, timeframe: str, limit: int) -> list[list[float]]:
            self.calls.append((timeframe,str(limit)))
            return bars([100.0]*limit,{'1d':86400,'4h':14400,'15m':900}[timeframe])
    clock=ControlledClock()
    with LegacySandbox() as sandbox:
        sandbox.module.initialize_runtime(clock=clock)
        sandbox.post('/update-config',data={'strategy_profile':'daemon_daily'})
        provider=DailyMarket(clock())
        with ExecutionWorker(sandbox.module.state_store,clock=clock) as owner:
            MarketRuntime(owner,provider,clock=clock,evaluate=sandbox.module.evaluate_market_snapshot).scan_once()
            state=sandbox.client.get('/api/status').get_json()
            assert state['market_health']['execution_ready'] is True
            assert state['strategy_decisions'][0]['profile'] == 'daemon_daily'
            assert set(provider.calls) == {('1d','301'),('4h','201'),('15m','201')}
            assert state['signals'] == []
            assert sandbox.exchange.orders == []


def test_dashboard_indicator_exit_reasons_keep_legacy_order_and_daily_has_no_indicator_exit() -> None:
    from strategy_profiles import evaluate
    with LegacySandbox() as sandbox:
        config = sandbox.module.state_store.config()[1]
        for profile in ('dashboard_intraday','daemon_daily'):
            data = inputs(profile)
            data['15m'] = bars([100.0]*25+[101.0,90.0],900)
            bias_tf = '1h' if profile == 'dashboard_intraday' else '4h'
            data[bias_tf] = bars([100.0]*25+[101.0,90.0],3600 if bias_tf == '1h' else 14400)
            result=evaluate(namespace='paper:testnet:test',venue='testnet',symbol='BTC/USDT',profile=profile,
                config_version=1,config=config,bars=data,as_of=AS_OF,position={'entry':100,'qty':1,'price':100})
            assert result['state'] == 'ready'
            assert result['exit_reasons'] == (['ema21_cross_down_15m','rsi_overbought_cross_down_15m','ema21_break_1h_rsi_down'] if profile == 'dashboard_intraday' else [])
