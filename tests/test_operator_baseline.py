"""Valid behavior at the approved operator HTTP seam, before remediation."""
from tests.legacy_sandbox import LegacySandbox
from pathlib import Path

import pytest


def test_paper_buy_is_visible_in_operator_status() -> None:
    with LegacySandbox() as sandbox:
        sandbox.seed_price('BTC/USDT', 100.0)
        response = sandbox.post('/trade', data={
            'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '2',
        })
        assert response.status_code == 302
        positions = sandbox.client.get('/api/status').get_json()['open_positions']
        assert positions == [{'symbol': 'BTC/USDT', 'entry': 100.0, 'qty': 2.0, 'pnl': 0, 'unrealized': '0', 'currency': 'USDT'}]


def test_paper_partial_sell_preserves_remaining_quantity() -> None:
    with LegacySandbox() as sandbox:
        sandbox.post('/update-config', data={'take_profit_pct': '0.5'})
        sandbox.seed_price('BTC/USDT', 100.0)
        sandbox.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '2'})
        sandbox.seed_price('BTC/USDT', 125.0)
        sandbox.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'sell', 'qty': '1'})
        positions = sandbox.client.get('/api/status').get_json()['open_positions']
        assert positions == [{'symbol': 'BTC/USDT', 'entry': 100.0, 'qty': 1.0, 'pnl': 25, 'unrealized': '25', 'currency': 'USDT'}]


def test_paper_full_close_removes_position() -> None:
    with LegacySandbox() as sandbox:
        sandbox.seed_price('BTC/USDT', 100.0)
        sandbox.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '2'})
        sandbox.post('/close', data={'symbol': 'BTC/USDT', 'qty': '2'})
        assert sandbox.client.get('/api/status').get_json()['open_positions'] == []


def test_paper_positions_reappear_after_reload(tmp_path: Path) -> None:
    with LegacySandbox(tmp_path) as first:
        first.seed_price('BTC/USDT', 100.0)
        first.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '2'})
    with LegacySandbox(tmp_path) as restarted:
        positions = restarted.client.get('/api/status').get_json()['open_positions']
        assert positions == [{'symbol': 'BTC/USDT', 'entry': 100.0, 'qty': 2.0, 'pnl': 0.0, 'unrealized': '0', 'currency': 'USDT'}]


def test_valid_explicit_paper_config_survives_reload(tmp_path: Path) -> None:
    with LegacySandbox(tmp_path) as first:
        first.post('/update-config', data={'paper_trading': '1', 'risk': '0.02', 'max_positions': '3'})
    with LegacySandbox(tmp_path) as restarted:
        config = restarted.client.get('/api/status').get_json()['config']
        assert config['paper_trading'] is True
        assert config['risk'] == pytest.approx(0.02)
        assert config['max_positions'] == 3
