"""BT11 full-history financial projections at the authenticated owned seam."""
from pathlib import Path
from tests.legacy_sandbox import LegacySandbox
from tests.test_paper_execution import setup


def test_full_history_beyond_display_cap_survives_restart_and_has_seeded_loss_drawdown(tmp_path: Path) -> None:
    with LegacySandbox(tmp_path,paper_setup=False) as sandbox:
        assert setup(sandbox,cooldown_seconds='0',drawdown_fraction='1')==302
        sandbox.seed_price('BTC/USDT',100)
        for _ in range(24):
            assert sandbox.post('/trade',data={'symbol':'BTC/USDT','side':'buy','qty':'0.1'}).status_code==302
            assert sandbox.post('/trade',data={'symbol':'BTC/USDT','side':'sell','qty':'0.1'}).status_code==302
        observed=sandbox.client.get('/api/status').get_json()
        metrics=observed['performance']
        assert len(observed['fills'])==48
        assert metrics['state']=='complete'
        assert metrics['currency']=='USDT'
        assert metrics['realized']=='-9.6'
        assert metrics['unrealized']=='0'
        assert metrics['equity']=='990.4'
        assert metrics['completed_trades']==24
        assert metrics['win_rate']=='0'
        assert metrics['win_rate_denominator']=='completed_full_position_exits'
        assert metrics['drawdown']=='0.0096'
        assert metrics['cost_basis']=='fee_inclusive_weighted_average'
        pages=[sandbox.client.get('/api/performance',query_string={'offset':offset,'limit':20}).get_json() for offset in (0,20,40)]
        assert [len(page['fills']) for page in pages]==[20,20,8]
        assert all(page['performance']==metrics for page in pages)
        assert [fill['fill_id'] for page in pages for fill in page['fills']]==[fill['fill_id'] for fill in observed['fills']]
        sandbox.module.initialize_runtime()
        assert sandbox.client.get('/api/status').get_json()['performance']==metrics


def test_unseeded_account_is_unavailable_not_zero_profit() -> None:
    with LegacySandbox(paper_setup=False) as sandbox:
        response=sandbox.client.get('/api/status')
        metrics=response.get_json()['performance']
        assert metrics['state']=='unavailable'
        assert metrics['realized'] is None
        assert metrics['equity'] is None
        assert metrics['win_rate'] is None
        assert 'setup_required' in metrics['reasons']



def test_partial_closes_accumulate_one_completed_position_profit_and_explicit_currency() -> None:
    with LegacySandbox(paper_setup=False) as sandbox:
        assert setup(sandbox,cooldown_seconds='0',drawdown_fraction='1')==302
        sandbox.seed_price('BTC/USDT',100)
        sandbox.post('/trade',data={'symbol':'BTC/USDT','side':'buy','qty':'1'})
        sandbox.post('/trade',data={'symbol':'BTC/USDT','side':'sell','qty':'0.4'})
        partial=sandbox.client.get('/api/status').get_json()['performance']
        assert partial['realized']=='-1.6'
        assert partial['unrealized']=='-1.206'
        assert partial['completed_trades']==0
        assert partial['win_rate'] is None
        sandbox.post('/trade',data={'symbol':'BTC/USDT','side':'sell','qty':'0.6'})
        complete=sandbox.client.get('/api/status').get_json()['performance']
        assert complete['realized']=='-4'
        assert complete['completed_trades']==1
        assert complete['equity']=='996'
        html=sandbox.client.get('/').get_data(as_text=True)
        assert 'Confirmed realized: -4 USDT' in html
        assert 'Estimated unrealized: 0 USDT' in html


def test_stale_position_value_is_partial_and_does_not_erase_confirmed_realized() -> None:
    from tests.test_market_runtime import ControlledClock
    from execution_worker import ExecutionWorker
    clock=ControlledClock()
    with LegacySandbox(paper_setup=False) as sandbox:
        sandbox.module.initialize_runtime(clock=clock)
        assert setup(sandbox,cooldown_seconds='0',drawdown_fraction='1')==302
        with ExecutionWorker(sandbox.module.state_store,clock=clock) as owner:
            from market_runtime import MarketRuntime
            from tests.test_market_runtime import MarketFixture
            MarketRuntime(owner,MarketFixture(clock()),clock=clock).scan_once()
            owner.publish_prices({'BTC/USDT':100})
            sandbox.client.post('/trade',data={'symbol':'BTC/USDT','side':'buy','qty':'1','csrf_token':sandbox.csrf})
            owner.process_pending()
            status=sandbox.client.get('/api/status').get_json()
            assert status['operations'][-1]['state']=='filled', status['market_health']
            fresh=status['performance']
            assert fresh['state']=='complete'
            assert fresh['unrealized']=='-2.01'
            clock.value+=60
            stale=sandbox.client.get('/api/status').get_json()['performance']
            assert stale['state']=='partial'
            assert stale['equity'] is None
            assert stale['unrealized'] is None
            assert stale['realized']=='0'
            assert 'valuation_stale:BTC' in stale['reasons']


def test_repeated_seed_change_is_rejected_without_rewriting_metrics() -> None:
    with LegacySandbox(paper_setup=False) as sandbox:
        assert setup(sandbox)==302
        before=sandbox.client.get('/api/status').get_json()['performance']
        assert setup(sandbox,initial_cash='2000')==409
        assert sandbox.client.get('/api/status').get_json()['performance']==before



def pure_snapshot(fills: list[dict[str,object]],cash: str) -> dict[str,object]:
    from tests.test_historical_replay import inputs
    settings=inputs()
    config={**settings['config'],'paper_policy':settings['policy']}
    return {'config':config,'fills':fills,'positions':[],'market_prices':{},
        'paper_account':{'balances':{'USDT':{'total':cash}}},'exchange_account':{}}


def fill(identity: str,side: str,quantity: str,cost: str,fee: str,asset: str) -> dict[str,object]:
    return {'fill_id':identity,'symbol':'BTC/USDT','side':side,'qty':quantity,
            'quote_cost':cost,'fee_amount':fee,'fee_asset':asset}


def test_pure_partial_fills_and_duplicate_observations_count_once() -> None:
    from account_performance import project
    values=[fill('1','buy','1','100','1','USDT'),fill('2','sell','0.4','44','0.44','USDT'),
            fill('3','sell','0.6','66','0.66','USDT')]
    snapshot=pure_snapshot(values+values,'1007.9')
    result=project(snapshot=snapshot,mode='paper',now=0,opening={'equity':'1000'},checkpoints=[{'equity':'1000','history_fill_count':0}],imported=False)
    assert result['state']=='complete'
    assert result['realized']=='7.9'
    assert result['fees_by_asset']=={'USDT':'2.1'}
    assert result['history_fills']==3
    assert result['completed_trades']==1
    assert result['win_rate']=='1'


def test_base_asset_fees_consume_actual_quantity_and_weighted_basis_once() -> None:
    from account_performance import project
    values=[fill('1','buy','1','100','0.01','BTC'),fill('2','sell','0.4','44','0.001','BTC'),
            fill('3','sell','0.588','64.68','0.001','BTC')]
    result=project(snapshot=pure_snapshot(values,'1008.68'),mode='paper',now=0,
        opening={'equity':'1000'},checkpoints=[{'equity':'1000','history_fill_count':0}],imported=False)
    assert result['state']=='complete'
    assert abs(float(result['realized'])-8.68)<1e-12
    assert result['fees_by_asset']=={'BTC':'0.012'}
    assert result['completed_trades']==1


def test_unconvertible_third_asset_fee_keeps_confirmed_units_and_withholds_net_profit() -> None:
    from account_performance import project
    values=[fill('1','buy','1','100','0.01','BNB'),fill('2','sell','1','110','0.001','BNB')]
    result=project(snapshot=pure_snapshot(values,'1010'),mode='paper',now=0,
        opening={'equity':'1000'},checkpoints=[{'equity':'1000','history_fill_count':0}],imported=False)
    assert result['state']=='partial'
    assert result['realized'] is None
    assert result['win_rate'] is None
    assert result['fees_by_asset']=={'BNB':'0.011'}
    assert 'historical_fee_conversion_unavailable:BNB' in result['reasons']


def test_imported_history_never_manufactures_zero_realized_or_win_rate() -> None:
    from account_performance import project
    result=project(snapshot=pure_snapshot([],'1000'),mode='paper',now=0,
        opening={'equity':'1000'},checkpoints=[],imported=True)
    assert result['state']=='partial'
    assert result['realized'] is None
    assert result['win_rate'] is None
    assert 'imported_history_unknown' in result['reasons']



def test_actual_fake_exchange_partial_closes_preserve_native_fee_totals_and_one_win_denominator() -> None:
    from execution_worker import ExecutionWorker
    from tests.test_exchange_execution import SpotFixture,configure
    with LegacySandbox(paper_setup=False,environment={'PAPER_TRADING':'false','ALLOW_EXCHANGE_EXECUTION':'true'}) as sandbox:
        configure(sandbox,cooldown_seconds='0')
        sandbox.seed_price('BTC/USDT',100)
        provider=SpotFixture()
        with ExecutionWorker(sandbox.module.state_store,exchange_provider=provider) as owner:
            owner.reconcile_exchange()
            for side,quantity in [('buy','1'),('sell','0.4'),('sell','0.6')]:
                assert sandbox.client.post('/trade',data={'symbol':'BTC/USDT','side':side,'qty':quantity,'csrf_token':sandbox.csrf}).status_code==302
                owner.process_pending()
            observed=sandbox.client.get('/api/status').get_json()
            metrics=observed['performance']
            assert len(provider.submissions)==3
            assert metrics['state']=='complete',metrics['reasons']
            assert metrics['initial_equity']=='1000'
            assert metrics['realized']=='-0.404'
            assert metrics['equity']=='999.596'
            assert metrics['fees_by_asset']=={'USDT':'0.404'}
            assert metrics['completed_trades']==1
            assert metrics['win_rate']=='0'
            owner.reconcile_exchange()
            again=sandbox.client.get('/api/status').get_json()['performance']
            assert again['realized']==metrics['realized']
            assert again['fees_by_asset']==metrics['fees_by_asset']
            assert again['history_fills']==3


def test_performance_history_requires_authentication_and_valid_page() -> None:
    with LegacySandbox(authenticated=False) as sandbox:
        assert sandbox.client.get('/api/performance').status_code==401
        sandbox.login()
        for query in [{'offset':-1},{'limit':0},{'limit':1001},{'limit':'invalid'}]:
            assert sandbox.client.get('/api/performance',query_string=query).status_code==422


def test_old_history_without_qualified_equity_points_does_not_invent_max_drawdown() -> None:
    from account_performance import project
    values=[fill('1','buy','1','100','1','USDT'),fill('2','sell','1','110','1.1','USDT')]
    result=project(snapshot=pure_snapshot(values,'1007.9'),mode='paper',now=0,
        opening={'equity':'1000'},checkpoints=[],imported=False)
    assert result['realized']=='7.9'
    assert result['drawdown'] is None
    assert result['state']=='partial'
    assert 'historical_equity_incomplete' in result['reasons']



def test_full_rebuild_detects_balance_change_without_rewriting_confirmed_fill_history() -> None:
    from account_performance import project
    values=[fill('1','buy','1','100','1','USDT'),fill('2','sell','1','110','1.1','USDT')]
    result=project(snapshot=pure_snapshot(values,'2000'),mode='paper',now=0,
        opening={'equity':'1000'},checkpoints=[{'equity':'1000','history_fill_count':0}],imported=False)
    assert result['state']=='partial'
    assert result['realized']=='7.9'
    assert result['equity'] is None
    assert 'balance_history_mismatch' in result['reasons']
    assert result['history_fills']==2


def test_drawdown_retains_initial_loss_after_equity_rebounds() -> None:
    from account_performance import project
    snapshot=pure_snapshot([fill('1','buy','1','100','1','USDT')],'899')
    snapshot['paper_account']={'balances':{'USDT':{'total':'899'},'BTC':{'total':'1'}}}
    snapshot['positions']=[{'symbol':'BTC/USDT','qty':'1','basis_status':'paper-full-fill-weighted-average-v1'}]
    snapshot['market_prices']={'BTC/USDT':{'price':'120','fetched_at':'2026-10-03T12:00:00+00:00'}}
    from datetime import datetime,timezone
    now=datetime(2026,10,3,12,tzinfo=timezone.utc).timestamp()
    result=project(snapshot=snapshot,mode='paper',now=now,opening={'equity':'1000'},
        checkpoints=[{'equity':'1000','history_fill_count':0},{'equity':'979','history_fill_count':1}],imported=False)
    assert result['state']=='complete'
    assert result['equity']=='1019'
    assert result['peak_equity']=='1019'
    assert result['drawdown']=='0.021'
    assert result['unrealized']=='19'
    assert result['realized']=='0'



def test_missing_fill_cost_is_partial_with_native_fees_preserved() -> None:
    from account_performance import project
    value=fill('legacy','buy','1','100','1','USDT')
    value['quote_cost']=None
    result=project(snapshot=pure_snapshot([value],'1000'),mode='paper',now=0,
        opening={'equity':'1000'},checkpoints=[],imported=False)
    assert result['state']=='partial'
    assert result['realized'] is None
    assert result['fees_by_asset']=={'USDT':'1'}
    assert 'fill_cost_unknown' in result['reasons']
