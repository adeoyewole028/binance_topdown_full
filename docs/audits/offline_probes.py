"""Audit reproductions only. No application import startup, credentials, or network."""
import ast
import copy
import json
import os
from pathlib import Path
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    with tempfile.TemporaryDirectory() as scratch:
        os.environ['DATA_DIR'] = scratch
        os.environ['PAPER_TRADING'] = 'true'
        tree = ast.parse((ROOT / 'dashboard.py').read_text(encoding='utf-8'))
        # Remove only the final __name__ startup branch (both dev and WSGI).
        assert isinstance(tree.body[-1], ast.If)
        assert ast.unparse(tree.body[-1].test) == "__name__ == '__main__'"
        tree.body.pop()
        import sys
        sys.path.insert(0, str(ROOT))
        ns = {'__name__': 'audit_dashboard', '__file__': str(ROOT / 'dashboard.py')}
        exec(compile(tree, str(ROOT / 'dashboard.py'), 'exec'), ns)
        ns['local_broker'].trade_log_path = str(Path(scratch) / 'trades.csv')
        app, status = ns['app'], ns['status']
        client = app.test_client()
        observations = {}
        observations['anonymous_status_http'] = client.get('/api/status').status_code
        observations['anonymous_dashboard_http'] = client.get('/').status_code
        client.post('/update-config', data={'risk': '-2'})
        observations['anonymous_partial_config_disables_paper'] = not status['config']['paper_trading']
        observations['negative_risk_accepted'] = status['config']['risk'] == -2
        status['config']['paper_trading'] = True
        status['live_prices'] = {'BTC/USDT': 100.0}
        client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '-1'})
        observations['negative_buy_position_qty'] = status['open_positions'][0]['qty']
        status['open_positions'] = [{'symbol': 'BTC/USDT', 'entry': 100.0, 'qty': 1.0, 'pnl': 0.0}]
        client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'sell', 'qty': '-2'})
        observations['negative_sell_increases_holdings_to'] = status['open_positions'][0]['qty']
        status['open_positions'] = []
        client.post('/trade', data={'symbol': 'MISSING/USDT', 'side': 'buy', 'qty': '1'})
        observations['missing_price_entry'] = status['open_positions'][0]['entry']
        observations['equity_proxy_discarded'] = 'equity_proxy' not in ns['coerce_config_types']({'equity_proxy': '2000'})

        class FakeExchange:
            def __init__(self):
                self.orders = []
            def create_order(self, *args, **kwargs):
                self.orders.append((args, kwargs))
                return {'id': 'offline-order', 'status': 'closed', 'filled': 1}
        fake = FakeExchange()
        ns['exchange'] = fake
        status['config']['paper_trading'] = False
        before = copy.deepcopy(status['open_positions'])
        for _ in range(2):
            client.post('/trade', data={'symbol': 'BTC/USDT', 'side': 'buy', 'qty': '1'})
        observations['duplicate_live_requests_mock_order_count'] = len(fake.orders)
        observations['live_fill_leaves_positions_unchanged'] = before == status['open_positions']
        ns['save_positions'](status['open_positions'], str(Path(scratch) / 'missing' / 'positions.json'))
        observations['missing_data_dir_save_returns_false'] = ns['save_persisted_config']({}, str(Path(scratch) / 'missing' / 'config.json')) is False

        daemon_tree = ast.parse((ROOT / 'topdown_daemon.py').read_text())
        daemon_tree.body = [n for n in daemon_tree.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and ast.unparse(n.value.func) == 'signal.signal')]
        daemon_tree.body.pop()
        dn = {'__name__': 'audit_daemon'}
        exec(compile(daemon_tree, 'topdown_daemon.py', 'exec'), dn)
        fake.fetch_ticker = lambda symbol: {'last': 100}
        try:
            dn['Broker'](fake).market_buy('BTC/USDT', 20, {'BTC/USDT': {'limits': {'amount': {'min': .00001, 'max': 100}}}})
        except KeyError as error:
            observations['standard_market_shape_buy_keyerror'] = str(error)
        for file in ROOT.glob('*.py'):
            compile(file.read_text(encoding='utf-8'), str(file), 'exec')
        observations['source_syntax_compile'] = 'all four modules passed'
        print(json.dumps(observations, indent=2))


if __name__ == '__main__':
    main()
