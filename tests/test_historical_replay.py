"""BT10 public pagination/pure completed-data replay; no execution transports."""
from datetime import datetime, timezone
from typing import Any
import pytest

BASE=int(datetime(2026,10,3,12,tzinfo=timezone.utc).timestamp()*1000)
DURATIONS={'1d':86400000,'4h':14400000,'1h':3600000,'15m':900000}


class Pages:
    def __init__(self) -> None:
        self.calls: list[dict[str,Any]]=[]
        self.omit: int | None=None

    def candles_range(self, symbol: str, timeframe: str, start: int, end: int, limit: int) -> list[list[float]]:
        self.calls.append({'symbol':symbol,'timeframe':timeframe,'start':start,'end':end,'limit':limit})
        duration=DURATIONS[timeframe]
        timestamps=list(range(start,end,duration))[:limit]
        return [[float(stamp),100,102,98,100,10] for stamp in timestamps if stamp != self.omit]


def test_historical_request_older_than_one_page_includes_explicit_range_and_profile_warmup() -> None:
    from historical_data import fetch_history
    source=Pages()
    result=fetch_history(source,symbol='BTC/USDT',venue='mainnet',profile='dashboard_intraday',start=BASE,end=BASE+1200*900000,as_of=BASE+1200*900000)
    assert result['state'] == 'complete'
    assert len(result['bars']['15m']) == 1300
    assert len([c for c in source.calls if c['timeframe']=='15m']) == 2
    first=next(c for c in source.calls if c['timeframe']=='15m')
    assert first['start'] == BASE-100*900000
    assert first['end'] == BASE+1200*900000
    assert first['limit'] == 1000
    assert result['coverage']['15m']['missing'] == 0


def test_missing_historical_bar_is_incomplete_and_never_a_valid_zero_trade_report() -> None:
    from historical_data import fetch_history
    source=Pages()
    source.omit=BASE+900000
    result=fetch_history(source,symbol='BTC/USDT',venue='testnet',profile='dashboard_intraday',start=BASE,end=BASE+5*900000,as_of=BASE+5*900000)
    assert result['state'] == 'incomplete'
    assert result['coverage']['15m']['missing'] == 1
    assert result['coverage']['15m']['reason'] == 'missing_history'


@pytest.mark.parametrize('problem',['duplicate','gap','invalid_ohlc'])
def test_bad_offline_history_cannot_be_presented_as_complete(problem: str) -> None:
    from historical_data import fetch_history,validate_history
    data=fetch_history(Pages(),symbol='BTC/USDT',venue='testnet',profile='dashboard_intraday',start=BASE,end=BASE+5*900000,as_of=BASE+5*900000)
    if problem=='duplicate':data['bars']['15m'].insert(-1,data['bars']['15m'][-1])
    if problem=='gap':del data['bars']['15m'][-2]
    if problem=='invalid_ohlc':data['bars']['15m'][-1][3]=150
    result=validate_history(data['bars'],symbol='BTC/USDT',venue='testnet',profile='dashboard_intraday',start=BASE,end=BASE+5*900000,as_of=BASE+5*900000)
    assert result['state'] == 'incomplete'


def golden_dataset() -> dict[str,Any]:
    from historical_data import validate_history
    from tests.test_automation import SignalMarket
    source=SignalMarket(BASE/1000)
    rows={tf:source.candles('BTC/USDT',tf,count) for tf,count in [('4h',200),('1h',200),('15m',100)]}
    rows['1h'].append([BASE,100,101,99,100,10])
    rows['15m'] += [[BASE+i*900000,100,101,99,100,10] for i in range(5)]
    rows['15m'][-4][1:5]=[100,112,90,100]
    return validate_history(rows,symbol='BTC/USDT',venue='testnet',profile='dashboard_intraday',start=BASE,end=BASE+5*900000,as_of=BASE+5*900000)


def inputs() -> dict[str,Any]:
    from configuration import validated_config
    from tests.test_paper_execution import PAPER_INPUTS
    from tests.legacy_sandbox import LegacySandbox
    with LegacySandbox() as sandbox:
        config=validated_config(sandbox.module.state_store.config()[1])
    return {'config':config,'policy':{**PAPER_INPUTS,'fixed_quote':'102.01','cooldown_seconds':'86400'},
            'sizing':'fixed','quantity_step':'0.001','namespace':'research:testnet:offline','config_version':1}


def test_replay_uses_next_open_full_fills_quote_fees_weighted_basis_and_stop_first() -> None:
    from historical_replay import replay
    result=replay(golden_dataset(),**inputs())
    assert result['state'] == 'complete'
    assert [f['side'] for f in result['fills']] == ['buy','sell']
    assert result['fills'][0]['price'] == '101'
    assert result['fills'][0]['quantity'] == '1'
    assert result['fills'][0]['fee'] == '1.01'
    assert result['fills'][1]['reason'] == 'stop_loss'
    assert result['fills'][1]['price'] == '97.960203'
    assert result['positions'] == []
    assert result['metrics']['cash'] == '994.97060097'
    assert result['metrics']['realized'] == '-5.02939903'
    assert result['metrics']['completed_trades'] == 1
    assert result['metrics']['win_rate'] == '0'


def test_incomplete_history_returns_unavailable_finances_and_cannot_replay() -> None:
    from historical_replay import replay
    data=golden_dataset()
    del data['bars']['15m'][-2]
    result=replay(data,**inputs())
    assert result['state'] == 'incomplete'
    assert result['metrics'] is None
    assert result['fills'] is None


def test_future_bar_mutation_preserves_earlier_decisions_and_fills() -> None:
    import copy
    from historical_replay import replay
    data=golden_dataset()
    modified=copy.deepcopy(data)
    changed_at=BASE+4*900000
    modified['bars']['15m'][-1][1:5]=[100,200,50,150]
    before,after=(replay(d,**inputs()) for d in (data,modified))
    assert [d for d in before['decisions'] if d['execution_close']<=changed_at] == [d for d in after['decisions'] if d['execution_close']<=changed_at]
    assert [f for f in before['fills'] if f['at']<=changed_at] == [f for f in after['fills'] if f['at']<=changed_at]
    assert before['data_sha256'] != after['data_sha256']


def test_replay_first_decision_matches_runtime_evaluator_at_identical_asof() -> None:
    from historical_replay import replay
    from strategy_profiles import evaluate
    data=golden_dataset()
    settings=inputs()
    actual=replay(data,**settings)['decisions'][0]
    expected=evaluate(namespace=settings['namespace'],venue='testnet',symbol='BTC/USDT',profile='dashboard_intraday',
        config_version=settings['config_version'],config=settings['config'],bars=data['bars'],
        as_of=datetime.fromtimestamp(BASE/1000,timezone.utc))
    assert actual == expected


def test_adverse_stop_gap_uses_open_instead_of_the_unavailable_stop_price() -> None:
    from historical_replay import replay
    data=golden_dataset()
    data['bars']['15m'][-4][1:5]=[90,93,88,90]
    result=replay(data,**inputs())
    assert result['fills'][1]['price'] == '89.1'
    assert result['metrics']['realized'] == '-13.801'
    assert result['metrics']['cash'] == '986.199'


def test_open_end_position_is_valued_without_inventing_liquidation_or_win_rate() -> None:
    from historical_replay import replay
    data=golden_dataset()
    data['bars']['15m'][-4][1:5]=[100,101,99,100]
    result=replay(data,**inputs())
    assert len(result['fills']) == 1
    assert result['positions'][0]['quantity'] == '1'
    assert result['positions'][0]['entry'] == '102.01'
    assert result['metrics']['unrealized'] == '-2.01'
    assert result['metrics']['completed_trades'] == 0
    assert result['metrics']['win_rate'] is None


def test_request_beyond_capture_time_cannot_treat_a_forming_bar_as_history() -> None:
    from historical_data import fetch_history
    result=fetch_history(Pages(),symbol='BTC/USDT',venue='testnet',profile='dashboard_intraday',
        start=BASE,end=BASE+5*900000,as_of=BASE+2*900000-1)
    assert result['state'] == 'incomplete'
    assert result['coverage']['15m']['received'] == 101
    assert result['coverage']['15m']['missing'] == 4


def test_explicit_offline_cli_writes_versioned_reproducible_result(tmp_path: Any) -> None:
    import json
    from backtester import main
    settings=inputs()
    for name,value in [('data',golden_dataset()),('config',settings['config']),('policy',settings['policy'])]:
        (tmp_path/f'{name}.json').write_text(json.dumps(value),encoding='utf-8')
    args=['--data',str(tmp_path/'data.json'),'--config',str(tmp_path/'config.json'),
          '--policy',str(tmp_path/'policy.json'),'--symbol','BTC/USDT','--venue','testnet',
          '--profile','dashboard_intraday','--start','2026-10-03T12:00:00Z',
          '--end','2026-10-03T13:15:00Z','--sizing','fixed','--quantity-step','0.001',
          '--config-version','1','--output',str(tmp_path/'result.json')]
    assert main(args)==0
    first=(tmp_path/'result.json').read_bytes()
    assert main(args)==0
    assert (tmp_path/'result.json').read_bytes()==first
    result=json.loads(first)
    assert result['format_version']==1
    assert result['as_of']==BASE+5*900000
    assert result['profile_version']==1
    assert result['metrics']['cash']=='994.97060097'


@pytest.mark.parametrize('value',['2026-10-03T12:00:00','2026-10-03T12:00:00+01:00'])
def test_cli_rejects_non_utc_datetime(value: str) -> None:
    from backtester import utc_milliseconds
    with pytest.raises(ValueError):utc_milliseconds(value)


def test_cli_accepts_date_as_utc_midnight() -> None:
    from backtester import utc_milliseconds
    assert utc_milliseconds('2026-10-03')==BASE-12*3600000


@pytest.mark.parametrize('profile',['dashboard_intraday','daemon_daily'])
def test_both_profiles_share_causal_runtime_parity_and_do_not_mutate_inputs(profile: str) -> None:
    import copy
    from historical_data import fetch_history
    from historical_replay import replay
    from strategy_profiles import evaluate
    data=fetch_history(Pages(),symbol='BTC/USDT',venue='testnet',profile=profile,
        start=BASE,end=BASE+5*900000,as_of=BASE+5*900000)
    settings=inputs()
    settings['config']['strategy_profile']=profile
    original=copy.deepcopy((data,settings))
    result=replay(data,**settings)
    assert (data,settings)==original
    assert result['state']=='complete'
    assert len(result['decisions'])==6
    for decision in result['decisions']:
        expected=evaluate(namespace=settings['namespace'],venue='testnet',symbol='BTC/USDT',profile=profile,
            config_version=1,config=settings['config'],bars=data['bars'],
            as_of=datetime.fromtimestamp(decision['execution_close']/1000,timezone.utc))
        assert decision==expected
    assert result['fills']==[]
    assert result['metrics']['cash']=='1000'


def test_unavailable_daily_close_cannot_change_earlier_replay_decisions() -> None:
    import copy
    from historical_data import fetch_history
    from historical_replay import replay
    start=BASE-12*3600000
    end=start+2*86400000
    data=fetch_history(Pages(),symbol='BTC/USDT',venue='testnet',profile='daemon_daily',
        start=start,end=end,as_of=end)
    changed=copy.deepcopy(data)
    changed['bars']['1d'][-1][1:5]=[100,200,50,150]
    settings=inputs()
    settings['config']['strategy_profile']='daemon_daily'
    before,after=(replay(d,**settings) for d in (data,changed))
    assert [d for d in before['decisions'] if d['execution_close']<end]==[d for d in after['decisions'] if d['execution_close']<end]


def test_historical_public_adapter_sends_explicit_utc_exclusive_bounds_and_reserves_urgent_budget() -> None:
    from tests.legacy_sandbox import LegacySandbox
    from tests.test_market_runtime import ControlledClock,MarketFixture
    from market_adapter import BinancePublicMarket
    from historical_source import BudgetedHistory
    from historical_data import fetch_history
    from request_budget import RequestBudget
    clock=ControlledClock()
    settings=inputs()
    class Raw:
        last_response_headers: dict[str,Any] | None=None
        def __init__(self) -> None:self.pages: list[dict[str,Any]]=[]
        def public_get_exchangeinfo(self,params: dict[str,Any]) -> dict[str,Any]:
            metadata=MarketFixture(clock()).metadata()
            metadata['rateLimits'][0]['limit']=132
            return metadata
        def public_get_time(self,params: dict[str,Any]) -> dict[str,int]:return {'serverTime':BASE}
        def public_get_klines(self,params: dict[str,Any]) -> list[list[float]]:
            self.pages.append(params)
            duration=DURATIONS[params['interval']]
            return [[float(t),100,102,98,100,10] for t in list(range(params['startTime'],params['endTime']+1,duration))[:params['limit']]]
    with LegacySandbox() as sandbox:
        sandbox.module.initialize_runtime(clock=clock)
        raw=Raw()
        source=BudgetedHistory(BinancePublicMarket(raw),RequestBudget(sandbox.module.state_store,clock=clock))
        assert source.capture_time('BTC/USDT')==BASE
        data=fetch_history(source,symbol='BTC/USDT',venue='testnet',profile='dashboard_intraday',
            start=BASE-900000,end=BASE,as_of=BASE)
        assert len(raw.pages)==2
        assert raw.pages[0]['startTime']==(BASE-900000)//14400000*14400000-200*14400000
        assert raw.pages[0]['endTime']==BASE-1
        assert raw.pages[0]['timeZone']=='0'
        assert raw.pages[0]['limit']==1000
        assert data['state']=='incomplete'
        assert data['coverage']['15m']['reason']=='rate_limit'
        from historical_replay import replay
        result=replay(data,**settings)
        assert result['metrics'] is None
        assert source.budget.request('reconciliation',2,lambda:'urgent')=='urgent'


@pytest.mark.parametrize('field',['max_allocation_fraction','max_exposure_fraction'])
def test_replay_respects_explicit_cash_allocation_and_exposure_caps(field: str) -> None:
    from historical_replay import replay
    settings=inputs()
    settings['policy'][field]='0.01'
    result=replay(golden_dataset(),**settings)
    assert result['fills'][0]['quantity']=='0.098'
    assert result['fills'][0]['quote_cost']=='9.898'
    assert result['fills'][0]['fee']=='0.09898'
    assert all(float(e['cash'])>=0 for e in result['equity_curve'])


def test_drawdown_stop_is_sticky_and_blocks_new_entries_without_blocking_exit() -> None:
    from historical_replay import replay
    settings=inputs()
    settings['policy']['drawdown_fraction']='0.001'
    settings['policy']['cooldown_seconds']='0'
    settings['config']['cooldown']=0
    result=replay(golden_dataset(),**settings)
    assert result['metrics']['entry_halted'] is True
    assert [fill['side'] for fill in result['fills']]==['buy','sell']
    halted=[e['entry_halted'] for e in result['equity_curve']]
    assert all(halted[halted.index(True):])


@pytest.mark.parametrize('bad',['missing_capture','unknown_version','missing_policy'])
def test_missing_or_unknown_explicit_replay_contract_is_rejected(bad: str) -> None:
    from historical_replay import replay
    from configuration import ConfigurationError
    data=golden_dataset()
    settings=inputs()
    if bad=='missing_capture':del data['as_of']
    elif bad=='unknown_version':data['profile_version']=2
    else:del settings['policy']['fee_fraction']
    with pytest.raises((KeyError,ValueError,ConfigurationError)):replay(data,**settings)


@pytest.mark.parametrize(('opening','high','low','closing','reason','price'),[
    (130,131,129,130,'take_profit','111.08889'),
    (70,72,69,70,'stop_loss','69.3'),
    (100,110,70,100,'indicator_exit','99'),
])
def test_opening_protection_precedes_queued_exit_without_using_later_intrabar_prices(
    opening: int,high: int,low: int,closing: int,reason: str,price: str,
) -> None:
    from historical_replay import replay
    data=golden_dataset()
    settings=inputs()
    settings['config'].update(stop_loss_pct=0.2,take_profit_pct=0.1)
    settings['policy']['risk_fraction']='1'
    for row in data['bars']['15m'][-5:]:row[1:5]=[100,101,99,100]
    data['bars']['15m'][-4][1:5]=[100,101,90,95]
    data['bars']['15m'][-3][1:5]=[opening,high,low,closing]
    result=replay(data,**settings)
    assert result['decisions'][2]['exit_reasons']
    assert result['fills'][1]['at']==BASE+2*900000
    assert result['fills'][1]['reason']==reason
    assert result['fills'][1]['price']==price



def test_empty_history_reports_broad_range_without_enumerating_expected_timestamps() -> None:
    from historical_data import validate_history
    end=int((datetime(9999,1,1,tzinfo=timezone.utc)-datetime(1970,1,1,tzinfo=timezone.utc)).total_seconds()*1000)
    result=validate_history({},symbol='BTC/USDT',venue='testnet',profile='daemon_daily',start=0,end=end,as_of=end)
    assert result['state']=='incomplete'
    assert result['bars']=={'1d':[],'4h':[],'15m':[]}
    assert result['coverage']['15m']['expected']==end//900000+200
    assert result['coverage']['15m']['missing']==result['coverage']['15m']['expected']



def test_fetch_outputs_cannot_overwrite_shared_trading_state(tmp_path: Any) -> None:
    import json
    from backtester import main
    settings=inputs()
    (tmp_path/'config.json').write_text(json.dumps(settings['config']),encoding='utf-8')
    (tmp_path/'policy.json').write_text(json.dumps(settings['policy']),encoding='utf-8')
    with pytest.raises(ValueError,match='outside the shared trading state'):
        main(['--fetch','--state-directory',str(tmp_path/'state'),'--account','operator','--state-mode','paper',
            '--config',str(tmp_path/'config.json'),'--policy',str(tmp_path/'policy.json'),
            '--symbol','BTC/USDT','--venue','testnet','--profile','dashboard_intraday',
            '--start','2026-10-03','--end','2026-10-04','--sizing','fixed','--quantity-step','0.001',
            '--config-version','1','--output',str(tmp_path/'state'/'state-identity.json')])
    assert not (tmp_path/'state').exists()
