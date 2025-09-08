
from flask import Flask, render_template_string, request, redirect, url_for, flash, jsonify, make_response
import threading
import time
import os
import json
import ccxt
import pandas as pd
from utils import ema, rsi, is_higher_highs_lows, detect_bullish_engulfing, detect_hammer, log_trade_csv
from datetime import datetime

# --- Strategy helpers (mirrors topdown_daemon) ---
def daily_trend_check(df_daily: pd.DataFrame):
    df = df_daily.copy()
    df['ema21'] = ema(df['close'], 21)
    df['ema50'] = ema(df['close'], 50)
    above_ma = bool((df['close'].iloc[-1] > df['ema21'].iloc[-1]) and (df['close'].iloc[-1] > df['ema50'].iloc[-1]))
    hhll = bool(is_higher_highs_lows(df['close']))
    return {'bullish': bool(above_ma and hhll), 'above21': above_ma, 'hhll': hhll}

def bias_check_4h(df_4h: pd.DataFrame):
    df = df_4h.copy()
    df['ema21'] = ema(df['close'], 21)
    df['rsi'] = rsi(df['close'], 14)
    last = df.iloc[-1]
    near_ma = bool(abs(last['close'] - df['ema21'].iloc[-1]) / max(float(last['close']), 1e-12) <= status['config']['near_ema_pct'])
    momentum_ok = bool(last['rsi'] > status['config']['rsi_threshold'])
    return {'near_ma': near_ma, 'momentum_ok': momentum_ok, 'rsi': float(last['rsi'])}

def bias_check_1h(df_1h: pd.DataFrame):
    df = df_1h.copy()
    df['ema21'] = ema(df['close'], 21)
    df['rsi'] = rsi(df['close'], 14)
    last = df.iloc[-1]
    near_ma = bool(abs(last['close'] - df['ema21'].iloc[-1]) / max(float(last['close']), 1e-12) <= status['config']['near_ema_pct'])
    momentum_ok = bool(last['rsi'] > status['config']['rsi_threshold'])
    return {'near_ma': near_ma, 'momentum_ok': momentum_ok, 'rsi': float(last['rsi'])}

def execution_check_15m(df_15m: pd.DataFrame):
    # Candlestick entries
    if detect_bullish_engulfing(df_15m):
        return {'signal': True, 'type': 'bullish_engulfing'}
    if detect_hammer(df_15m):
        return {'signal': True, 'type': 'hammer'}

    # Indicator-based entry
    df = df_15m.copy()
    df['ema21'] = ema(df['close'], 21)
    df['rsi'] = rsi(df['close'], 14)
    if len(df) >= 2:
        last, prev = df.iloc[-1], df.iloc[-2]
        price_above = bool(last['close'] > last['ema21'])
        rsi_cross = bool((prev['rsi'] <= status['config']['rsi_exec_threshold']) and (last['rsi'] > status['config']['rsi_exec_threshold']))
        if price_above and rsi_cross:
            return {'signal': True, 'type': 'ema21_rsi_cross'}
    return {'signal': False}

def fetch_ohlcv_df(exchange, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
    delay = 0.5
    last_err = None
    for attempt in range(3):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            return df
        except Exception as e:
            last_err = e
            if attempt == 2:
                break
            time.sleep(delay)
            delay *= 2
    raise last_err

app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'supersecretkey123')

# Defaults for config (used for reset)
DEFAULT_CONFIG = {
    'risk': 0.01,
    'max_positions': 5,
    'cooldown': 120,
    'paper_trading': os.getenv('PAPER_TRADING', 'true').lower() in ('1','true','yes','y'),
    'near_ema_pct': float(os.getenv('NEAR_EMA_PCT', '0.02')),
    'rsi_threshold': float(os.getenv('RSI_THRESHOLD', '30')),
    'rsi_exec_threshold': float(os.getenv('RSI_EXEC_THRESHOLD', '45')),
    'rsi_overbought': float(os.getenv('RSI_OVERBOUGHT', '70')),
    'take_profit_pct': float(os.getenv('TAKE_PROFIT_PCT', '0.06')),
    'stop_loss_pct': float(os.getenv('STOP_LOSS_PCT', '0.03')),
    'trailing_pct': float(os.getenv('TRAILING_PCT', '0.03')),
    'universe_size': int(os.getenv('UNIVERSE_SIZE', '12')),
    'base_quote': os.getenv('BASE_QUOTE', 'USDT'),
    'use_dynamic_universe': os.getenv('USE_DYNAMIC_UNIVERSE', 'true').lower() in ('1','true','yes','y'),
    'min_quote_vol': float(os.getenv('MIN_24H_QUOTE_VOLUME', '10000000')),
    # proxy account equity for risk sizing
    'equity_proxy': float(os.getenv('EQUITY_PROXY', '1000')),
}

status = {
    'last_scan': '',
    'universe': [],
    'logs': [],
    'open_positions': [],
    'balances': {'USDT': 1000, 'BTC': 0.1},
    'recent_trades': [],
    'signals': [],
    'exit_signals': [],
    'bot_status': 'Running',
    'config': DEFAULT_CONFIG.copy(),
    'performance': {
        'win_rate': 0.0,
        'total_profit': 0.0,
        'drawdown': 0.0
    },
    'live_prices': {},
    'cooldowns': {},
    'equity_history': [],
    'peak_equity': 0.0
}

# Exclude stablecoin-vs-quote pairs from the dynamic universe
STABLE_BASES = {
    'USDT', 'USDC', 'BUSD', 'FDUSD', 'TUSD', 'USDP', 'DAI', 'SUSD', 'UST', 'PAX'
}

# --- Config persistence helpers ---
DATA_DIR = os.getenv('DATA_DIR', os.path.dirname(__file__))
CONFIG_PATH = os.path.join(DATA_DIR, 'config.json')
POSITIONS_PATH = os.path.join(DATA_DIR, 'positions.json')

def load_persisted_config(path: str = CONFIG_PATH):
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data
    except Exception as e:
        status['logs'].append(f"Config load error: {e}")
    return {}

def save_persisted_config(cfg: dict, path: str = CONFIG_PATH):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=2)
        return True
    except Exception as e:
        status['logs'].append(f"Config save error: {e}")
        return False

def coerce_config_types(incoming: dict) -> dict:
    # Keep a type map for safe casting
    type_map = {
        'risk': float,
        'max_positions': int,
        'cooldown': int,
        'paper_trading': bool,
        'near_ema_pct': float,
        'rsi_threshold': float,
        'rsi_exec_threshold': float,
        'rsi_overbought': float,
        'take_profit_pct': float,
        'stop_loss_pct': float,
        'trailing_pct': float,
        'universe_size': int,
        'base_quote': str,
        'use_dynamic_universe': bool,
        'min_quote_vol': float,
    }
    out = {}
    for k, caster in type_map.items():
        if k not in incoming:
            continue
        v = incoming[k]
        try:
            if caster is bool:
                # Accept truthy strings or existing bool
                if isinstance(v, str):
                    out[k] = v.strip().lower() in ('1','true','on','yes','y')
                else:
                    out[k] = bool(v)
            else:
                out[k] = caster(v)
        except Exception:
            # If casting fails, skip to preserve old value
            continue
    return out

# Load persisted config and merge over defaults
_persisted = load_persisted_config()
if _persisted:
    status['config'].update(coerce_config_types(_persisted))

# --- Positions persistence ---
def load_positions(path: str = POSITIONS_PATH):
    try:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    # ensure numeric
                    out = []
                    for p in data:
                        try:
                            out.append({'symbol': p['symbol'], 'entry': float(p['entry']), 'qty': float(p['qty']), 'pnl': float(p.get('pnl', 0))})
                        except Exception:
                            continue
                    return out
    except Exception as e:
        status['logs'].append(f"Positions load error: {e}")
    return []

def save_positions(positions: list, path: str = POSITIONS_PATH):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(positions, f, indent=2)
        return True
    except Exception as e:
        status['logs'].append(f"Positions save error: {e}")
        return False

# Initialize positions from disk
_pos = load_positions()
if _pos:
    status['open_positions'] = _pos

# --- Exit strategy helpers ---
def compute_exit_reasons(symbol: str, entry: float, qty: float, exchange) -> dict:
    reasons = []
    try:
        price = exchange.fetch_ticker(symbol)['last']
    except Exception:
        price = status['live_prices'].get(symbol)
    if not isinstance(price, (int, float)) or price is None:
        return {'should_exit': False, 'reasons': [], 'price': None}

    tp_price = entry * (1 + status['config']['take_profit_pct'])
    sl_price = entry * (1 - status['config']['stop_loss_pct'])
    if price >= tp_price:
        reasons.append('take_profit')
    if price <= sl_price:
        reasons.append('stop_loss')

    # Indicator-based exits on 15m
    try:
        df15 = fetch_ohlcv_df(exchange, symbol, '15m', 100)
        df15['ema21'] = ema(df15['close'], 21)
        df15['rsi'] = rsi(df15['close'], 14)
        if len(df15) >= 2:
            last, prev = df15.iloc[-1], df15.iloc[-2]
            if (prev['close'] >= prev['ema21']) and (last['close'] < last['ema21']):
                reasons.append('ema21_cross_down_15m')
            if (prev['rsi'] >= status['config']['rsi_overbought']) and (last['rsi'] < status['config']['rsi_overbought']):
                reasons.append('rsi_overbought_cross_down_15m')
    except Exception as e:
        status['logs'].append(f"Exit check 15m error {symbol}: {e}")

    # 1h momentum weakening
    try:
        df1h = fetch_ohlcv_df(exchange, symbol, '1h', 100)
        df1h['ema21'] = ema(df1h['close'], 21)
        df1h['rsi'] = rsi(df1h['close'], 14)
        if len(df1h) >= 2:
            l1, p1 = df1h.iloc[-1], df1h.iloc[-2]
            # 1h close below EMA21 with RSI falling
            if (p1['close'] >= p1['ema21']) and (l1['close'] < l1['ema21']) and (l1['rsi'] < p1['rsi']):
                reasons.append('ema21_break_1h_rsi_down')
    except Exception as e:
        status['logs'].append(f"Exit check 1h error {symbol}: {e}")

    return {'should_exit': len(reasons) > 0, 'reasons': reasons, 'price': price}

class LocalBroker:
    def __init__(self, status_ref):
        self.status = status_ref
        self.trade_log_path = os.path.join('logs', 'trades.csv')

    def _price(self, symbol):
        p = self.status['live_prices'].get(symbol)
        if isinstance(p, (int, float)):
            return float(p)
        return None

    def _find_position(self, symbol):
        for pos in self.status['open_positions']:
            if pos['symbol'] == symbol:
                return pos
        return None

    def create_order(self, symbol: str, side: str, qty: float):
        qty = float(qty)
        price = self._price(symbol) or 0.0
        ts = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        pos = self._find_position(symbol)

        if side.lower() == 'buy':
            if pos:
                new_qty = pos['qty'] + qty
                # Weighted average entry
                pos['entry'] = (pos['entry'] * pos['qty'] + price * qty) / max(new_qty, 1e-12)
                pos['qty'] = new_qty
            else:
                self.status['open_positions'].append({'symbol': symbol, 'entry': price, 'qty': qty, 'pnl': 0})
            trade = {'time': ts, 'symbol': symbol, 'side': 'buy', 'qty': qty, 'price': price, 'status': 'filled', 'pnl': 0}
        else:  # sell
            if pos:
                sell_qty = min(qty, pos['qty'])
                pnl = (price - pos['entry']) * sell_qty
                pos['qty'] -= sell_qty
                if pos['qty'] <= 1e-12:
                    # close position fully
                    self.status['open_positions'] = [p for p in self.status['open_positions'] if p['symbol'] != symbol]
                trade = {'time': ts, 'symbol': symbol, 'side': 'sell', 'qty': sell_qty, 'price': price, 'status': 'filled', 'pnl': pnl}
            else:
                trade = {'time': ts, 'symbol': symbol, 'side': 'sell', 'qty': 0, 'price': price, 'status': 'rejected', 'pnl': 0}

        # Record trade in memory and CSV
        self.status.setdefault('recent_trades', []).insert(0, trade)
        self.status['recent_trades'] = self.status['recent_trades'][:20]
        try:
            log_trade_csv(self.trade_log_path, [trade['time'], trade['symbol'], trade['side'], trade['qty'], price, '', '', 'local', 'paper'])
        except Exception:
            pass
        # Persist positions
        try:
            save_positions(self.status['open_positions'])
        except Exception:
            pass
        return trade

    def close_position(self, symbol: str, qty: float):
        return self.create_order(symbol, 'sell', qty)

exchange = None
local_broker = LocalBroker(status)
def ensure_exchange():
    """Create the global exchange instance if missing, with expected options."""
    global exchange
    if exchange is None:
        status['logs'].append('Initializing exchange...')
        print('Initializing exchange...')
        exchange = ccxt.binance({
            'apiKey': os.getenv('API_KEY'),
            'secret': os.getenv('API_SECRET'),
            'enableRateLimit': True,
            'timeout': int(os.getenv('CCXT_TIMEOUT_MS', '10000')),
            'options': {'defaultType': 'spot', 'adjustForTimeDifference': True}
        })
        try:
            exchange.timeout = int(os.getenv('CCXT_TIMEOUT_MS', '10000'))
        except Exception:
            pass

def recompute_universe_and_prices():
    """Rebuild the trading universe based on current config and refresh live prices immediately.
    Returns the list of symbols in the refreshed universe.
    """
    ensure_exchange()
    # Mark immediate refresh time
    status['last_scan'] = time.strftime('%Y-%m-%d %H:%M:%S')
    symbols = status.get('universe') or ['ETH/USDT', 'BTC/USDT', 'SOL/USDT', 'BNB/USDT']
    try:
        if status['config']['use_dynamic_universe']:
            status['logs'].append('Refreshing dynamic universe...')
            tickers_all = exchange.fetch_tickers()
            candidates = []
            for sym, data in tickers_all.items():
                if not sym.endswith('/' + status['config']['base_quote']):
                    continue
                if any(x in sym for x in ['UP/', 'DOWN/', 'BULL/', 'BEAR/']):
                    continue
                try:
                    base, _quote = sym.split('/')
                    if base.upper() in STABLE_BASES:
                        continue
                except Exception:
                    continue
                qvol = data.get('quoteVolume') or 0
                if qvol >= status['config']['min_quote_vol']:
                    candidates.append((sym, qvol))
            candidates.sort(key=lambda x: x[1], reverse=True)
            size = max(int(status['config']['universe_size']), 1)
            symbols = [s for s, _ in candidates[:size]] or symbols
            symbols = [s for s in symbols if s.split('/')[0].upper() not in STABLE_BASES]
            if not symbols:
                symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT']
            status['logs'].append(f"Universe size now {len(symbols)}")
        else:
            # Static mode: just respect size over existing/default pool
            pool = symbols or ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT']
            size = max(int(status['config']['universe_size']), 1)
            symbols = pool[:size]
        status['universe'] = symbols
    except Exception as e:
        status['logs'].append(f"Universe error: {e}")
        status['universe'] = symbols

    # Fetch live prices for the refreshed set (with retries and per-symbol fallback)
    try:
        def _fetch_bulk():
            return exchange.fetch_tickers(symbols)
        tickers = None
        delay = 0.5
        for attempt in range(3):
            try:
                tickers = _fetch_bulk()
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(delay)
                delay *= 2
        live = {}
        if isinstance(tickers, dict):
            for s in symbols:
                try:
                    live[s] = tickers.get(s, {}).get('last', 'N/A')
                except Exception:
                    live[s] = 'N/A'
        else:
            for s in symbols:
                try:
                    t = exchange.fetch_ticker(s)
                    live[s] = t.get('last')
                except Exception:
                    live[s] = 'N/A'
                    continue
                time.sleep(0.1)
        status['live_prices'] = live
        ok = sum(1 for v in live.values() if isinstance(v, (int, float)))
        status['logs'].append(f"Prices fetched {ok}/{len(symbols)}")
        # Prepare display order: sort symbols by highest price first
        def _pval(x):
            v = live.get(x)
            return float(v) if isinstance(v, (int, float)) else float('-inf')
        status['universe_sorted'] = sorted(symbols, key=_pval, reverse=True)
    except Exception as e:
        status['live_prices'] = {s: 'N/A' for s in symbols}
        status['logs'].append(f"Error fetching prices: {e}")
        status['universe_sorted'] = symbols

    return symbols

def bot_loop():
    symbols = ['ETH/USDT', 'BTC/USDT', 'SOL/USDT', 'BNB/USDT']
    global exchange
    status['logs'].append('Bot thread starting...')
    print('Bot thread starting...')
    while True:
        try:
            # Lazy init/retryable exchange creation
            ensure_exchange()

            now = time.strftime('%Y-%m-%d %H:%M:%S')
            status['last_scan'] = now
            # Recompute universe and prices each loop so config changes reflect immediately
            symbols = recompute_universe_and_prices()

            # Update P&L for local open positions
            for pos in status['open_positions']:
                sym = pos['symbol']
                price = status['live_prices'].get(sym)
                try:
                    if isinstance(price, (int, float)):
                        pos['pnl'] = (float(price) - float(pos['entry'])) * float(pos['qty'])
                except Exception:
                    pos['pnl'] = 0
            # recent_trades retained in memory (trim occasionally)
            if len(status.get('recent_trades', [])) > 50:
                status['recent_trades'] = status['recent_trades'][:50]

            # Update performance metrics (live)
            try:
                closed = [t for t in status.get('recent_trades', []) if t.get('side') == 'sell' and t.get('status') == 'filled']
                wins = sum(1 for t in closed if (t.get('pnl') or 0) > 0)
                total = len(closed)
                win_rate = (wins / total) if total else 0.0

                realized = sum(float(t.get('pnl') or 0.0) for t in closed)
                unreal = 0.0
                for pos in status.get('open_positions', []):
                    price = status['live_prices'].get(pos['symbol'])
                    if isinstance(price, (int, float)):
                        try:
                            unreal += (float(price) - float(pos['entry'])) * float(pos['qty'])
                        except Exception:
                            pass
                total_profit = realized + unreal

                # equity tracking (relative)
                equity = total_profit
                status['equity_history'].append(equity)
                if len(status['equity_history']) > 1000:
                    status['equity_history'] = status['equity_history'][-1000:]
                if equity > status['peak_equity']:
                    status['peak_equity'] = equity
                peak = status['peak_equity'] or 0.0
                drawdown = ((peak - equity) / peak * 100.0) if peak > 0 else 0.0

                status['performance']['win_rate'] = win_rate
                status['performance']['total_profit'] = round(total_profit, 4)
                status['performance']['drawdown'] = round(drawdown, 2)
            except Exception as e:
                status['logs'].append(f"Perf calc error: {e}")

            # Multi-timeframe signal gating: 4h trend + 1h bias + 15m execution
            detected_signals = []
            for symbol in symbols:
                try:
                    df_4h_trend = fetch_ohlcv_df(exchange, symbol, '4h', 200)
                    df_1h = fetch_ohlcv_df(exchange, symbol, '1h', 200)
                    df_15m = fetch_ohlcv_df(exchange, symbol, '15m', 100)
                    # Use the same structure logic on 4h for trend
                    trend4h = daily_trend_check(df_4h_trend)
                    bias1h = bias_check_1h(df_1h)
                    exec15 = execution_check_15m(df_15m)
                    if trend4h['bullish'] and bias1h['near_ma'] and bias1h['momentum_ok'] and exec15['signal']:
                        detected_signals.append({
                            'symbol': symbol,
                            'type': exec15.get('type', 'entry'),
                            'time': now,
                            'trend4h': trend4h,
                            'bias1h': bias1h
                        })
                    else:
                        status['logs'].append(
                            f"NoSignal {symbol}: 4hTrend={trend4h['bullish']} nearEMA1h={bias1h['near_ma']} RSI1h={bias1h['rsi']:.2f} exec={exec15['signal']}"
                        )
                except Exception as e:
                    status['logs'].append(f"Signal error for {symbol}: {e}")
            status['signals'] = detected_signals

            # Build exit signals for open positions
            exit_sigs = []
            for pos in status['open_positions']:
                sym = pos['symbol']
                entry = float(pos['entry'])
                qty = float(pos['qty'])
                res = compute_exit_reasons(sym, entry, qty, exchange)
                if res['should_exit']:
                    pnl = (res['price'] - entry) * qty
                    exit_sigs.append({'symbol': sym, 'reasons': res['reasons'], 'price': res['price'], 'pnl': pnl})
            status['exit_signals'] = exit_sigs

            # Simulate cooldowns
            status['cooldowns'] = {s: 0 for s in symbols}
            # Logs
            status['logs'].append(f"Scanned at {now}")
            if len(status['logs']) > 20:
                status['logs'] = status['logs'][-20:]
        except Exception as e:
            # Surface errors to UI and stdout, then backoff and retry
            msg = f"Bot error: {e}"
            status['logs'].append(msg)
            print(msg)
            # Reset exchange so we re-init on next loop if needed
            exchange = None
            time.sleep(5)
            continue
        time.sleep(10)

@app.route('/trade', methods=['POST'])
def trade():
    symbol = request.form.get('symbol')
    side = request.form.get('side')
    qty = request.form.get('qty')
    try:
        if status['config']['paper_trading']:
            order = local_broker.create_order(symbol, side, float(qty))
        else:
            order = exchange.create_order(symbol, 'market', side, float(qty))
        flash(f"Trade executed: {order}", "success")
        status['logs'].append(f"Trade executed: {order}")
    except Exception as e:
        flash(f"Trade error: {e}", "error")
        status['logs'].append(f"Trade error: {e}")
    return redirect('/')

@app.route('/close', methods=['POST'])
def close_position():
    symbol = request.form.get('symbol')
    qty = request.form.get('qty')
    try:
        q = float(qty)
    except Exception:
        # Fallback: find from open_positions
        q = 0.0
        for pos in status['open_positions']:
            if pos['symbol'] == symbol:
                q = float(pos['qty'])
                break
    try:
        if status['config']['paper_trading']:
            order = local_broker.close_position(symbol, q)
        else:
            order = exchange.create_order(symbol, 'market', 'sell', q)
        flash(f"Closed {symbol}: {order}", 'success')
        status['logs'].append(f"Closed {symbol}: {order}")
    except Exception as e:
        flash(f"Close error for {symbol}: {e}", 'error')
        status['logs'].append(f"Close error for {symbol}: {e}")
    return redirect('/')

@app.route('/api/status', methods=['GET'])
def api_status():
    # Minimal JSON for static frontends; includes signals, positions, performance, config (non-secret)
    payload = {
        'last_scan': status['last_scan'],
        'universe': status['universe'],
        'signals': status['signals'],
        'exit_signals': status['exit_signals'],
        'open_positions': status['open_positions'],
        'performance': status['performance'],
        'config': status['config'],
        'live_prices': status['live_prices'],
    }
    resp = make_response(jsonify(payload), 200)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

@app.route('/update-config', methods=['POST'])
def update_config():
    # Build dict from form fields, then coerce types
    form = request.form.to_dict(flat=True)
    # Only accept known keys, but handle booleans even if unchecked (missing => False)
    updates = {k: v for k, v in form.items() if k in status['config']}
    for bk in ('paper_trading', 'use_dynamic_universe'):
        if bk in status['config']:
            updates[bk] = '1' if form.get(bk) is not None else '0'
    coerced = coerce_config_types(updates)
    # Merge into status
    status['config'].update(coerced)
    ok = save_persisted_config(status['config'])
    if ok:
        flash('Configuration saved.', 'success')
    else:
        flash('Failed to save configuration; using in-memory values.', 'error')
    # Apply config immediately to universe and prices so UI reflects changes without waiting
    try:
        recompute_universe_and_prices()
    except Exception as e:
        status['logs'].append(f"Immediate refresh after config failed: {e}")
    return redirect(url_for('dashboard'))

@app.route('/reset-config', methods=['POST'])
def reset_config():
    # Reset to DEFAULT_CONFIG and persist
    status['config'] = DEFAULT_CONFIG.copy()
    ok = save_persisted_config(status['config'])
    if ok:
        flash('Configuration reset to defaults.', 'success')
    else:
        flash('Reset failed to save; defaults applied in memory.', 'error')
    # Also refresh universe/prices immediately
    try:
        recompute_universe_and_prices()
    except Exception as e:
        status['logs'].append(f"Immediate refresh after reset failed: {e}")
    return redirect(url_for('dashboard'))

@app.route('/', methods=['GET'])
def dashboard():
    return render_template_string('''
        <!doctype html>
        <html lang="en" class="h-full bg-slate-950">
        <head>
            <meta charset="utf-8" />
            <meta name="viewport" content="width=device-width, initial-scale=1" />
            <meta http-equiv="refresh" content="15" />
            <title>Binance Topdown Bot Dashboard</title>
            <script src="https://cdn.tailwindcss.com"></script>
        </head>
        <body class="h-full text-slate-100">
            <header class="bg-gradient-to-r from-slate-900 to-slate-800 border-b border-slate-800">
                <div class="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
                    <h1 class="text-xl font-semibold">Binance Topdown Bot</h1>
                    <div class="flex items-center gap-2">
                        <button id="soundToggle" class="px-2 py-1 rounded border border-slate-700 text-xs text-slate-200 hover:bg-slate-800">Sound: Off</button>
                        <span class="inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium {{ 'bg-emerald-600 text-slate-900' if status['bot_status']=='Running' else 'bg-rose-600 text-white' }}">
                            {{ status['bot_status'] }}
                        </span>
                    </div>
                </div>
            </header>

            <!-- Toasts root -->
            <div id="toast-root" class="fixed top-4 right-4 z-50 space-y-2"></div>

            <main class="max-w-7xl mx-auto p-4 space-y-4">
                <section class="grid grid-cols-1 xl:grid-cols-4 gap-4">
                    <div class="col-span-1 bg-slate-900/50 rounded-lg border border-slate-800 p-4">
                        <h2 class="text-sm font-semibold text-slate-300">Overview</h2>
                        <div class="text-xs text-slate-400 mt-1">Last scan: {{ status['last_scan'] }}</div>
                        <div class="mt-3 space-y-1 text-sm">
                            <div class="flex items-center justify-between">
                                <span class="text-slate-400">Universe size</span>
                                <span class="px-2 py-0.5 rounded bg-slate-700 text-slate-200 text-xs">{{ status['universe']|length }}</span>
                            </div>
                            <div class="flex items-center justify-between">
                                <span class="text-slate-400">Signals</span>
                                <span class="px-2 py-0.5 rounded {{ 'bg-emerald-600 text-slate-900' if status['signals'] else 'bg-slate-700 text-slate-200' }} text-xs">{{ status['signals']|length }}</span>
                            </div>
                            <div class="flex items-center justify-between">
                                <span class="text-slate-400">Exit signals</span>
                                <span class="px-2 py-0.5 rounded {{ 'bg-amber-500 text-slate-900' if status['exit_signals'] else 'bg-slate-700 text-slate-200' }} text-xs">{{ status['exit_signals']|length }}</span>
                            </div>
                        </div>
                    </div>

                    <div class="col-span-1 xl:col-span-3 bg-slate-900/50 rounded-lg border border-slate-800 p-4">
                        <h2 class="text-sm font-semibold text-slate-300">Universe</h2>
                        {% set uni = status['universe_sorted'] if 'universe_sorted' in status else status['universe'] %}
                        <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-3 mt-3">
                            {% for symbol in uni[:status['config']['universe_size']] %}
                            <div class="rounded-lg border border-slate-800 bg-slate-900/60 p-3">
                                <div class="flex items-center justify-between">
                                    <div class="font-semibold text-sm">{{ symbol }}</div>
                                    <span class="px-2 py-0.5 rounded bg-slate-700 text-slate-200 text-xs">{{ status['cooldowns'].get(symbol, 0) }}s</span>
                                </div>
                                <div class="mt-2 text-sm">
                                    {% set p = status['live_prices'].get(symbol, 'N/A') %}
                                    {% if p is string or p is none %}
                                        <span class="italic text-slate-400">N/A</span>
                                    {% else %}
                                        <span class="text-slate-200">{{ p|round(6) }}</span>
                                    {% endif %}
                                </div>
                            </div>
                            {% endfor %}
                        </div>
                    </div>
                </section>

                <section class="grid grid-cols-1 gap-4">
                    <div class="bg-slate-900/50 rounded-lg border border-slate-800 p-4">
                        <h2 class="text-sm font-semibold text-slate-300">Open Positions</h2>
                        <div class="overflow-x-auto mt-2">
                            <table class="min-w-full text-sm">
                                <thead class="text-slate-400">
                                    <tr class="border-b border-slate-800">
                                        <th class="text-left py-2">Symbol</th>
                                        <th class="text-left py-2">Entry</th>
                                        <th class="text-left py-2">Qty</th>
                                        <th class="text-left py-2">P&L</th>
                                        <th class="text-left py-2">Action</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for pos in status['open_positions'] %}
                                    {% set pnlpos = 'text-emerald-400' if pos['pnl']>0 else ('text-rose-400' if pos['pnl']<0 else 'text-slate-200') %}
                                    <tr class="border-b border-slate-800/60">
                                        <td class="py-2">{{ pos['symbol'] }}</td>
                                        <td class="py-2">{{ pos['entry'] }}</td>
                                        <td class="py-2">{{ pos['qty'] }}</td>
                                        <td class="py-2 {{ pnlpos }}">{{ pos['pnl']|round(2) }}</td>
                                        <td class="py-2">
                                            <form method="post" action="/close" class="inline-flex">
                                                <input type="hidden" name="symbol" value="{{ pos['symbol'] }}">
                                                <input type="hidden" name="qty" value="{{ pos['qty'] }}">
                                                <button type="submit" class="px-2 py-1 rounded border border-slate-600 text-slate-200 hover:bg-slate-800 text-xs">Close</button>
                                            </form>
                                        </td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </section>

                <section class="grid grid-cols-1 xl:grid-cols-3 gap-4">
                    <div class="xl:col-span-2 bg-slate-900/50 rounded-lg border border-slate-800 p-4">
                        <h2 class="text-sm font-semibold text-slate-300">Signals</h2>
                        {% if status['signals'] %}
                        <div class="overflow-x-auto mt-2">
                            <table class="min-w-full text-sm">
                                <thead class="text-slate-400">
                                    <tr class="border-b border-slate-800">
                                        <th class="text-left py-2">Time</th>
                                        <th class="text-left py-2">Symbol</th>
                                        <th class="text-left py-2">Type</th>
                                        <th class="text-left py-2">4h Trend</th>
                                        <th class="text-left py-2">1h Bias</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {% for sig in status['signals'] %}
                                    <tr class="border-b border-slate-800/60">
                                        <td class="py-2 text-slate-400">{{ sig['time'] }}</td>
                                        <td class="py-2 font-medium">{{ sig['symbol'] }}</td>
                                        <td class="py-2">
                                            <span class="inline-flex items-center rounded-full bg-emerald-600 text-slate-900 px-2 py-0.5 text-xs font-medium">{{ sig['type'] }}</span>
                                        </td>
                                        <td class="py-2 space-x-1">
                                            <span class="inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium {{ 'bg-emerald-600 text-slate-900' if sig['trend4h']['bullish'] else 'bg-slate-700 text-slate-200' }}">bullish</span>
                                            <span class="inline-flex items-center rounded-full bg-slate-700 text-slate-200 px-2 py-0.5 text-xs font-medium">21/{{ 'Y' if sig['trend4h']['above21'] else 'N' }}</span>
                                            <span class="inline-flex items-center rounded-full bg-slate-700 text-slate-200 px-2 py-0.5 text-xs font-medium">HHLL/{{ 'Y' if sig['trend4h']['hhll'] else 'N' }}</span>
                                        </td>
                                        <td class="py-2 space-x-1">
                                            <span class="inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium {{ 'bg-emerald-600 text-slate-900' if sig['bias1h']['near_ma'] else 'bg-slate-700 text-slate-200' }}">near EMA21</span>
                                            <span class="inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium {{ 'bg-emerald-600 text-slate-900' if sig['bias1h']['momentum_ok'] else 'bg-slate-700 text-slate-200' }}">RSI ok</span>
                                            <span class="inline-flex items-center rounded-full bg-slate-700 text-slate-200 px-2 py-0.5 text-xs font-medium">RSI {{ sig['bias1h']['rsi']|round(2) }}</span>
                                        </td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                        {% else %}
                            <div class="text-slate-400 mt-2">No signals detected on the current scan.</div>
                        {% endif %}
                    </div>

                    <div class="bg-slate-900/50 rounded-lg border border-slate-800 p-4">
                        <h2 class="text-sm font-semibold text-slate-300">Exit Signals</h2>
                        {% if status['exit_signals'] %}
                        <div class="overflow-x-auto mt-2">
                            <table class="min-w-full text-sm">
                                <thead class="text-slate-400">
                                    <tr class="border-b border-slate-800"><th class="text-left py-2">Symbol</th><th class="text-left py-2">Price</th><th class="text-left py-2">P&L</th><th class="text-left py-2">Reasons</th><th class="py-2"></th></tr>
                                </thead>
                                <tbody>
                                    {% for es in status['exit_signals'] %}
                                    {% set pnlpos = 'text-emerald-400' if es['pnl']>0 else ('text-rose-400' if es['pnl']<0 else 'text-slate-200') %}
                                    <tr class="border-b border-slate-800/60">
                                        <td class="py-2">{{ es['symbol'] }}</td>
                                        <td class="py-2">{{ es['price'] }}</td>
                                        <td class="py-2 {{ pnlpos }}">{{ es['pnl']|round(2) }}</td>
                                        <td class="py-2 space-x-1">
                                            {% for r in es['reasons'] %}
                                                <span class="inline-flex items-center rounded-full bg-amber-500 text-slate-900 px-2 py-0.5 text-xs font-medium">{{ r }}</span>
                                            {% endfor %}
                                        </td>
                                        <td class="py-2">
                                            <form method="post" action="/close" class="inline-flex">
                                                <input type="hidden" name="symbol" value="{{ es['symbol'] }}">
                                                <input type="hidden" name="qty" value="{% for pos in status['open_positions'] %}{% if pos['symbol']==es['symbol'] %}{{ pos['qty'] }}{% endif %}{% endfor %}">
                                                <button type="submit" class="px-2 py-1 rounded border border-slate-600 text-slate-200 hover:bg-slate-800 text-xs">Close</button>
                                            </form>
                                        </td>
                                    </tr>
                                    {% endfor %}
                                </tbody>
                            </table>
                        </div>
                        {% else %}
                            <div class="text-slate-400 mt-2">No exit signals right now.</div>
                        {% endif %}
                    </div>
                </section>

                <section class="grid grid-cols-1 xl:grid-cols-3 gap-4">
                    <div class="xl:col-span-2 bg-slate-900/50 rounded-lg border border-slate-800 p-4">
                        <h2 class="text-sm font-semibold text-slate-300">Place Trade</h2>
                                    <form method="post" action="/trade" class="mt-2 grid grid-cols-1 sm:grid-cols-4 gap-2 items-end">
                                        <label class="text-sm">Symbol
                                            {% if status['universe'] %}
                                                <select class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 text-sm" name="symbol">
                                                    {% for s in status['universe'] %}
                                                        <option value="{{ s }}">{{ s }}</option>
                                                    {% endfor %}
                                                </select>
                                            {% else %}
                                                <input class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 text-sm" name="symbol" value="BTC/USDT" />
                                            {% endif %}
                                        </label>
                            <label class="text-sm">Side
                                <select class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 text-sm" name="side">
                                    <option value="buy">Buy</option>
                                    <option value="sell">Sell</option>
                                </select>
                            </label>
                            <label class="text-sm">Quantity
                                <div class="flex gap-2 items-center">
                                    <input class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1 text-sm" name="qty" id="qtyInput" type="number" step="any" value="0.01" />
                                    <button type="button" id="riskSizeBtn" class="mt-1 px-2 py-1 rounded border border-slate-600 text-slate-200 hover:bg-slate-800 text-xs">Size by risk</button>
                                </div>
                                <div class="text-xs text-slate-500 mt-1">Uses equity_proxy and stop_loss_pct to estimate qty.</div>
                            </label>
                            <button type="submit" class="w-full sm:w-auto px-3 py-2 rounded bg-indigo-600 hover:bg-indigo-500 text-sm font-medium">Submit</button>
                        </form>
                        <div class="mt-3 space-y-1">
                            {% with messages = get_flashed_messages(with_categories=true) %}
                                {% if messages %}
                                    {% for category, message in messages %}
                                        <div class="px-3 py-2 rounded text-sm {{ 'bg-emerald-600 text-slate-900' if category=='success' else 'bg-rose-600 text-white' }}">{{ message }}</div>
                                    {% endfor %}
                                {% endif %}
                            {% endwith %}
                        </div>
                        <form method="post" action="/reset-config" class="mt-3">
                            <button type="submit" class="px-3 py-2 rounded border border-slate-700 text-slate-200 hover:bg-slate-800 text-xs">Reset to defaults</button>
                        </form>
                    </div>

                    <div class="bg-slate-900/50 rounded-lg border border-slate-800 p-4">
                        <h2 class="text-sm font-semibold text-slate-300">Performance</h2>
                        <div class="grid grid-cols-2 gap-4 mt-2">
                            <div>
                                <div class="text-xs text-slate-400">Win Rate</div>
                                <div class="text-lg font-semibold">{{ status['performance']['win_rate']*100|round(2) }}%</div>
                            </div>
                            <div>
                                <div class="text-xs text-slate-400">Total Profit</div>
                                <div class="text-lg font-semibold">{{ status['performance']['total_profit'] }}</div>
                            </div>
                        </div>
                        <div class="mt-3">
                            <div class="text-xs text-slate-400">Drawdown</div>
                            <div class="text-base">{{ status['performance']['drawdown'] }}</div>
                        </div>
                        <div class="mt-4">
                            <h3 class="text-sm font-semibold text-slate-300">Config</h3>
                            <form method="post" action="/update-config" class="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-2 text-sm">
                                <label>risk
                                    <input name="risk" type="number" step="0.001" min="0" max="1" value="{{ status['config']['risk'] }}" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1" />
                                </label>
                                <label>max_positions
                                    <input name="max_positions" type="number" step="1" min="1" value="{{ status['config']['max_positions'] }}" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1" />
                                </label>
                                <label>cooldown
                                    <input name="cooldown" type="number" step="1" min="0" value="{{ status['config']['cooldown'] }}" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1" />
                                </label>
                                <label class="flex items-center gap-2">paper_trading
                                    <input name="paper_trading" type="checkbox" value="1" {% if status['config']['paper_trading'] %}checked{% endif %} class="mt-1 rounded border-slate-700" />
                                </label>
                                <label>near_ema_pct
                                    <input name="near_ema_pct" type="number" step="0.001" min="0" value="{{ status['config']['near_ema_pct'] }}" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1" />
                                </label>
                                <label>rsi_threshold
                                    <input name="rsi_threshold" type="number" step="1" min="0" max="100" value="{{ status['config']['rsi_threshold'] }}" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1" />
                                </label>
                                <label>rsi_exec_threshold
                                    <input name="rsi_exec_threshold" type="number" step="1" min="0" max="100" value="{{ status['config']['rsi_exec_threshold'] }}" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1" />
                                </label>
                                <label>rsi_overbought
                                    <input name="rsi_overbought" type="number" step="1" min="0" max="100" value="{{ status['config']['rsi_overbought'] }}" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1" />
                                </label>
                                <label>take_profit_pct
                                    <input name="take_profit_pct" type="number" step="0.001" min="0" value="{{ status['config']['take_profit_pct'] }}" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1" />
                                </label>
                                <label>stop_loss_pct
                                    <input name="stop_loss_pct" type="number" step="0.001" min="0" value="{{ status['config']['stop_loss_pct'] }}" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1" />
                                </label>
                                <label>trailing_pct
                                    <input name="trailing_pct" type="number" step="0.001" min="0" value="{{ status['config']['trailing_pct'] }}" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1" />
                                </label>
                                <label>universe_size
                                    <input name="universe_size" type="number" step="1" min="1" value="{{ status['config']['universe_size'] }}" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1" />
                                </label>
                                <label>base_quote
                                    <input name="base_quote" type="text" value="{{ status['config']['base_quote'] }}" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1" />
                                </label>
                                <label class="flex items-center gap-2">use_dynamic_universe
                                    <input name="use_dynamic_universe" type="checkbox" value="1" {% if status['config']['use_dynamic_universe'] %}checked{% endif %} class="mt-1 rounded border-slate-700" />
                                </label>
                                <label>min_quote_vol
                                    <input name="min_quote_vol" type="number" step="1000" min="0" value="{{ status['config']['min_quote_vol'] }}" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1" />
                                </label>
                                <label>equity_proxy
                                    <input name="equity_proxy" type="number" step="0.01" min="0" value="{{ status['config']['equity_proxy'] }}" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1" />
                                </label>

                                <div class="sm:col-span-2 flex items-center justify-end gap-2">
                                    <button type="submit" class="px-3 py-2 rounded bg-emerald-600 hover:bg-emerald-500 text-slate-900 font-medium">Save</button>
                                </div>
                            </form>
                        </div>
                    </div>
                </section>

                            <section>
                                <div class="bg-slate-900/50 rounded-lg border border-slate-800 p-4">
                                    <h2 class="text-sm font-semibold text-slate-300">Recent Trades</h2>
                                    <div class="overflow-x-auto mt-2">
                                        <table class="min-w-full text-sm">
                                            <thead class="text-slate-400">
                                                <tr class="border-b border-slate-800">
                                                    <th class="text-left py-2">Time</th>
                                                    <th class="text-left py-2">Symbol</th>
                                                    <th class="text-left py-2">Side</th>
                                                    <th class="text-left py-2">Qty</th>
                                                    <th class="text-left py-2">Price</th>
                                                    <th class="text-left py-2">P&L</th>
                                                    <th class="text-left py-2">Status</th>
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {% for t in status['recent_trades'] %}
                                                    {% set pnlc = 'text-emerald-400' if t['pnl']>0 else ('text-rose-400' if t['pnl']<0 else 'text-slate-200') %}
                                                    <tr class="border-b border-slate-800/60">
                                                        <td class="py-2 text-slate-400">{{ t['time'] }}</td>
                                                        <td class="py-2">{{ t['symbol'] }}</td>
                                                        <td class="py-2"><span class="px-2 py-0.5 rounded {{ 'bg-emerald-600 text-slate-900' if t['side']=='buy' else 'bg-rose-600 text-white' }} text-xs">{{ t['side'] }}</span></td>
                                                        <td class="py-2">{{ t['qty'] }}</td>
                                                        <td class="py-2">{{ t['price'] }}</td>
                                                        <td class="py-2 {{ pnlc }}">{{ t['pnl'] }}</td>
                                                        <td class="py-2 text-slate-300">{{ t['status'] }}</td>
                                                    </tr>
                                                {% endfor %}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            </section>

                <section>
                    <div class="bg-slate-900/50 rounded-lg border border-slate-800 p-4">
                        <h2 class="text-sm font-semibold text-slate-300">Logs</h2>
                        <div class="mt-2 max-h-64 overflow-auto rounded border border-slate-800 bg-slate-900 px-3 py-2 font-mono text-xs">
                            {% for log in status['logs'] %}
                                <div>{{ log }}</div>
                            {% endfor %}
                        </div>
                    </div>
                </section>
            </main>
                    <script>
                        // Persist and toggle sound preference
                        let soundEnabled = (localStorage.getItem('soundEnabled') === '1');
                        const toggleBtn = document.getElementById('soundToggle');
                        function refreshSoundLabel(){ if(toggleBtn){ toggleBtn.textContent = 'Sound: ' + (soundEnabled ? 'On' : 'Off'); } }
                        refreshSoundLabel();
                        let audioCtx = null;
                        function ensureAudio(){
                            if (!soundEnabled) return;
                            try { audioCtx = audioCtx || new (window.AudioContext || window.webkitAudioContext)(); } catch(e){}
                        }
                        function playBeep(){
                            if (!soundEnabled) return;
                            ensureAudio();
                            if (!audioCtx) return;
                            const o = audioCtx.createOscillator();
                            const g = audioCtx.createGain();
                            o.type = 'sine';
                            o.frequency.value = 880;
                            g.gain.value = 0.08; // slightly louder
                            o.connect(g); g.connect(audioCtx.destination);
                            o.start();
                            setTimeout(()=>{ o.stop(); }, 250); // a bit longer
                        }
                        if (toggleBtn) {
                            toggleBtn.addEventListener('click', async () => {
                                soundEnabled = !soundEnabled;
                                localStorage.setItem('soundEnabled', soundEnabled ? '1' : '0');
                                refreshSoundLabel();
                                if (soundEnabled) { try { ensureAudio(); if (audioCtx && audioCtx.state !== 'running') { await audioCtx.resume(); } playBeep(); } catch(e){} }
                            });
                        }

                        // Re-arm audio on first user interaction after each page load (autoplay policy)
                        if (soundEnabled) {
                            const onFirstInput = async () => {
                                try {
                                    ensureAudio();
                                    if (audioCtx && audioCtx.state !== 'running') { await audioCtx.resume(); }
                                    playBeep();
                                } catch(e) {}
                                window.removeEventListener('pointerdown', onFirstInput);
                            };
                            window.addEventListener('pointerdown', onFirstInput, { once: true });
                        }

                        // Toast helper
                        function showToast(message){
                            const root = document.getElementById('toast-root');
                            if(!root) return;
                            const el = document.createElement('div');
                            el.className = 'rounded border border-slate-700 bg-slate-900/90 text-slate-100 px-3 py-2 shadow-lg text-sm transform transition-all duration-300 translate-y-[-6px] opacity-0';
                            el.innerHTML = message;
                            root.appendChild(el);
                            requestAnimationFrame(()=>{ el.classList.remove('translate-y-[-6px]','opacity-0'); el.classList.add('translate-y-0','opacity-100'); });
                            setTimeout(()=>{
                                el.classList.add('opacity-0');
                                setTimeout(()=>{ el.remove(); }, 300);
                            }, 6000);
                        }

                        // Diff signals vs previous to raise toasts
                        const currentSignals = {{ status['signals'] | tojson }};
                        const ids = currentSignals.map(s => `${s.symbol}|${s.time}|${s.type || 'entry'}`);
                        let prev = [];
                        try { prev = JSON.parse(localStorage.getItem('signalIds') || '[]'); } catch(e) { prev = []; }
                        const newOnes = ids.filter(id => !prev.includes(id));
                        if (newOnes.length > 0) {
                            // Show each as toast and optional beep
                            for (const id of newOnes) {
                                const parts = id.split('|');
                                const sym = parts[0], t = parts[1], typ = parts[2];
                                showToast(`<span class="font-semibold">Signal</span> — ${sym} <span class="text-slate-400">(${typ})</span><br><span class="text-slate-400 text-xs">${t}</span>`);
                            }
                            playBeep();
                        }
                        // Update stored ids
                        try { localStorage.setItem('signalIds', JSON.stringify(ids)); } catch(e){}
                    </script>
                    <script>
                        // Risk-based sizing: qty = (equity * risk) / (price * stop_loss_pct)
                        function getSelectedSymbol(){
                            const sel = document.querySelector('select[name="symbol"]');
                            return sel ? sel.value : 'BTC/USDT';
                        }
                        function getLivePrice(sym){
                            try { const lp = {{ status['live_prices'] | tojson }}; return lp[sym]; } catch(e){ return null; }
                        }
                        const riskBtn = document.getElementById('riskSizeBtn');
                        if (riskBtn) {
                            riskBtn.addEventListener('click', ()=>{
                                const sym = getSelectedSymbol();
                                const price = parseFloat(getLivePrice(sym));
                                const equity = parseFloat({{ status['config']['equity_proxy'] }});
                                const risk = parseFloat({{ status['config']['risk'] }});
                                const sl = parseFloat({{ status['config']['stop_loss_pct'] }});
                                if (!isFinite(price) || price <= 0) return;
                                const qty = (equity * risk) / Math.max(price * Math.max(sl, 1e-6), 1e-6);
                                const inp = document.getElementById('qtyInput');
                                if (inp) inp.value = (Math.max(qty, 0)).toFixed(6);
                            });
                        }
                    </script>
        </body>
        </html>
    ''', status=status)

if __name__ == '__main__':
    # Local/dev run: start bot and Flask dev server
    threading.Thread(target=bot_loop, daemon=True).start()
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)), debug=True)
else:
    # Production (WSGI): Flask 3.x removed before_first_request; start once at import time.
    try:
        if not globals().get('_bot_started', False):
            threading.Thread(target=bot_loop, daemon=True).start()
            globals()['_bot_started'] = True
    except Exception:
        pass
