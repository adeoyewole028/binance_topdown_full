"""Safe outcomes that fail today; strict xfails must be retired by their owner.

Never assert the insecure behavior as desirable compatibility. A later fix causes
XPASS (suite failure), forcing that ticket to promote its expectation to a test.
"""
import pytest

from tests.legacy_sandbox import LegacySandbox


def test_anonymous_trade_has_no_exchange_effect() -> None:
    with LegacySandbox(authenticated=False) as sandbox:
        sandbox.module.status['config']['paper_trading'] = False
        sandbox.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1'})
        assert sandbox.exchange.holdings.get('BTC/USDT', 0) == 0


def test_partial_config_preserves_paper_mode() -> None:
    with LegacySandbox() as sandbox:
        sandbox.post('/update-config', data={'risk': '0.02'})
        assert sandbox.client.get('/api/status').get_json()['config']['paper_trading'] is True


@pytest.mark.xfail(raises=AssertionError, reason='A07 / BT05: negative paper buys currently create negative holdings')
def test_negative_buy_leaves_positions_unchanged() -> None:
    with LegacySandbox() as sandbox:
        sandbox.seed_price('BTC/USDT', 100.0)
        sandbox.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '-1'})
        assert sandbox.client.get('/api/status').get_json()['open_positions'] == []


@pytest.mark.xfail(raises=AssertionError, reason='A07 / BT05: negative sells currently increase holdings')
def test_negative_sell_does_not_increase_holdings() -> None:
    with LegacySandbox() as sandbox:
        sandbox.seed_price('BTC/USDT', 100.0)
        sandbox.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1'})
        sandbox.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'sell', 'qty': '-2'})
        assert sandbox.client.get('/api/status').get_json()['open_positions'][0]['qty'] == 1.0


@pytest.mark.xfail(raises=AssertionError, reason='A07 / BT05: absent market price currently fills at zero')
def test_missing_price_does_not_create_position() -> None:
    with LegacySandbox() as sandbox:
        sandbox.post('/trade', data={'symbol': 'UNKNOWN/USDT', 'side': 'buy', 'qty': '1'})
        assert sandbox.client.get('/api/status').get_json()['open_positions'] == []


@pytest.mark.xfail(raises=AssertionError, reason='A03 / BT06: repeated request currently creates repeated economic effect')
def test_duplicate_operation_has_one_exchange_effect() -> None:
    with LegacySandbox(environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        sandbox.module.status['config']['paper_trading'] = False
        request = {'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1', 'idempotency_key': 'offline-repeat'}
        sandbox.post('/trade', data=request)
        sandbox.post('/trade', data=request)
        assert sandbox.exchange.holdings.get('BTC/USDT') == 1.0


@pytest.mark.xfail(raises=AssertionError, reason='A02 / BT06: confirmed exchange fills currently do not update operator positions')
def test_confirmed_exchange_fill_is_visible_to_operator() -> None:
    with LegacySandbox(environment={'PAPER_TRADING': 'false', 'ALLOW_EXCHANGE_EXECUTION': 'true'}) as sandbox:
        sandbox.module.status['config']['paper_trading'] = False
        sandbox.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1'})
        positions = sandbox.client.get('/api/status').get_json()['open_positions']
        assert any(position['symbol'] == 'BTC/USDT' and position['qty'] == 1.0 for position in positions)


@pytest.mark.xfail(raises=AssertionError, reason='A13 / BT04: equity estimate field is currently discarded')
def test_equity_estimate_setting_is_saved() -> None:
    with LegacySandbox() as sandbox:
        sandbox.post('/update-config', data={'paper_trading': '1', 'equity_proxy': '2000'})
        assert sandbox.client.get('/api/status').get_json()['config']['equity_proxy'] == 2000.0


@pytest.mark.xfail(raises=AssertionError, reason='A13 / BT04: negative risk setting currently passes validation')
def test_negative_risk_cannot_replace_valid_setting() -> None:
    with LegacySandbox() as sandbox:
        sandbox.post('/update-config', data={'paper_trading': '1', 'risk': '-2'})
        assert sandbox.client.get('/api/status').get_json()['config']['risk'] == 0.01


def test_normal_dashboard_import_has_no_background_startup() -> None:
    with LegacySandbox() as sandbox:
        sandbox.load_module('dashboard', normal_import=True)
        assert sandbox.thread_start_attempts == 0
