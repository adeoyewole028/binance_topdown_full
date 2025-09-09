
from flask import Flask, render_template_string, request, redirect, url_for, flash, jsonify, make_response
import threading
import time
import os
import json
import ccxt
import pandas as pd
from utils import ema, rsi, is_higher_highs_lows, detect_bullish_engulfing, detect_hammer, log_trade_csv, atr
from datetime import datetime

# --- Strategy helpers (mirrors topdown_daemon) ---
def daily_trend_check(df_daily: pd.DataFrame):
    df = df_daily.copy()
    fast = int(status['config'].get('ema_fast', 21))
    slow = int(status['config'].get('ema_slow', 50))
    df['ema_fast'] = ema(df['close'], fast)
    df['ema_slow'] = ema(df['close'], slow)
    above_ma = bool((df['close'].iloc[-1] > df['ema_fast'].iloc[-1]) and (df['close'].iloc[-1] > df['ema_slow'].iloc[-1]))
    hhll = bool(is_higher_highs_lows(df['close']))
    # Include legacy key names for existing template references
    return {'bullish': bool(above_ma and hhll), 'above_fast': above_ma, 'above21': above_ma, 'hhll': hhll}

def bias_check_4h(df_4h: pd.DataFrame):
    df = df_4h.copy()
    fast = int(status['config'].get('ema_fast', 21))
    df['ema_fast'] = ema(df['close'], fast)
    df['rsi'] = rsi(df['close'], 14)
    last = df.iloc[-1]
    near_ma = bool(abs(last['close'] - df['ema_fast'].iloc[-1]) / max(float(last['close']), 1e-12) <= status['config']['near_ema_pct'])
    momentum_ok = bool(last['rsi'] > status['config']['rsi_threshold'])
    return {'near_ma': near_ma, 'momentum_ok': momentum_ok, 'rsi': float(last['rsi'])}

def bias_check_1h(df_1h: pd.DataFrame):
    df = df_1h.copy()
    fast = int(status['config'].get('ema_fast', 21))
    df['ema_fast'] = ema(df['close'], fast)
    df['rsi'] = rsi(df['close'], 14)
    last = df.iloc[-1]
    near_ma = bool(abs(last['close'] - df['ema_fast'].iloc[-1]) / max(float(last['close']), 1e-12) <= status['config']['near_ema_pct'])
    momentum_ok = bool(last['rsi'] > status['config']['rsi_threshold'])
    return {'near_ma': near_ma, 'momentum_ok': momentum_ok, 'rsi': float(last['rsi'])}

def execution_check_15m(df_15m: pd.DataFrame):
    # Candlestick entries
    if detect_bullish_engulfing(df_15m):
        return {'signal': True, 'type': 'bullish_engulfing', 'direction': 'up'}
    if detect_hammer(df_15m):
        return {'signal': True, 'type': 'hammer', 'direction': 'up'}

    # Indicator-based entry
    df = df_15m.copy()
    fast = int(status['config'].get('ema_fast', 21))
    df['ema_fast'] = ema(df['close'], fast)
    df['rsi'] = rsi(df['close'], 14)
    if len(df) >= 2:
        last, prev = df.iloc[-1], df.iloc[-2]
        price_above = bool(last['close'] > last['ema_fast'])
        rsi_cross_up = bool((prev['rsi'] <= status['config']['rsi_exec_threshold']) and (last['rsi'] > status['config']['rsi_exec_threshold']))
        rsi_cross_down = bool((prev['rsi'] >= status['config']['rsi_exec_threshold']) and (last['rsi'] < status['config']['rsi_exec_threshold']))
        if price_above and rsi_cross_up:
            return {'signal': True, 'type': f'ema{fast}_rsi_cross_up', 'direction': 'up'}
        if (last['close'] < last['ema_fast']) and rsi_cross_down:
            return {'signal': True, 'type': f'ema{fast}_rsi_cross_down', 'direction': 'down'}
    return {'signal': False}

def fetch_ohlcv_df(exchange, symbol: str, timeframe: str, limit: int = 200) -> pd.DataFrame:
    # Basic in-memory cache (per process) to avoid repeated identical pulls inside loop
    cache = status.setdefault('_ohlcv_cache', {})
    key = (symbol, timeframe, limit)
    now_ts = time.time()
    rec = cache.get(key)
    ttl = 30  # seconds
    if rec and (now_ts - rec['ts'] < ttl):
        return rec['df'].copy()
    delay = 0.5
    last_err = None
    for attempt in range(3):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            cache[key] = {'df': df, 'ts': now_ts}
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
    'equity_proxy': float(os.getenv('EQUITY_PROXY', '100')),
    # trading market type: 'spot' or 'future' (Binance USD-M futures)
    'market_type': os.getenv('MARKET_TYPE', 'spot'),
    # enable bearish entries (shorts in paper; requires futures for live shorts)
    'enable_bearish': os.getenv('ENABLE_BEARISH', 'false').lower() in ('1','true','yes','y'),
    # Dynamic strategy parameters (Phase 1 additions)
    'ema_fast': int(os.getenv('EMA_FAST', '21')),
    'ema_slow': int(os.getenv('EMA_SLOW', '50')),
    'atr_period': int(os.getenv('ATR_PERIOD', '14')),
    'atr_stop_mult': float(os.getenv('ATR_STOP_MULT', '1.5')),
    'atr_tp_mult': float(os.getenv('ATR_TP_MULT', '2.5')),
    'min_stop_pct': float(os.getenv('MIN_STOP_PCT', '0.004')),  # 0.4% floor
    # Advanced position management
    'partial_tp_r_multiple': float(os.getenv('PARTIAL_TP_R_MULTIPLE', '1.0')),  # take partial at 1R
    'partial_tp_fraction': float(os.getenv('PARTIAL_TP_FRACTION', '0.5')),      # fraction to close
    'breakeven_after_partial': os.getenv('BREAKEVEN_AFTER_PARTIAL', 'true').lower() in ('1','true','yes','y'),
    'trail_activate_r_multiple': float(os.getenv('TRAIL_ACTIVATE_R_MULTIPLE', '1.5')),
    'trail_atr_mult': float(os.getenv('TRAIL_ATR_MULT', '1.0')),
    'daily_loss_limit_pct': float(os.getenv('DAILY_LOSS_LIMIT_PCT', '0.05')),  # 5% of equity_proxy
    'enable_daily_loss_guard': os.getenv('ENABLE_DAILY_LOSS_GUARD', 'true').lower() in ('1','true','yes','y'),
    # Volatility regime filter: minimum ATR/price ratio on 15m to consider entries
    'min_atr_rel': float(os.getenv('MIN_ATR_REL', '0.001')),
}

# Human-readable descriptions for each config parameter (for legend display)
CONFIG_PARAM_DESCRIPTIONS = {
    'risk': 'Fraction of equity_proxy risked per trade (not fully enforced yet).',
    'max_positions': 'Maximum simultaneous open positions.',
    'cooldown': 'Seconds to wait after a filled trade before re-entering same symbol.',
    'paper_trading': 'Simulate locally instead of sending live orders.',
    'near_ema_pct': 'Max % distance from fast EMA (1h) to count as near for gating.',
    'rsi_threshold': 'Minimum 1h RSI for momentum gating.',
    'rsi_exec_threshold': '15m RSI cross level required to trigger execution signal.',
    'rsi_overbought': 'RSI level considered overbought for exit conditions.',
    'take_profit_pct': 'Static TP percent (fallback if ATR TP absent).',
    'stop_loss_pct': 'Static SL percent (fallback if ATR SL absent).',
    'trailing_pct': 'Legacy static trailing stop percent (superseded by ATR trailing).',
    'universe_size': 'Number of symbols maintained in active universe.',
    'base_quote': 'Quote currency filter (e.g. USDT).',
    'use_dynamic_universe': 'Rebuild universe each loop using top volume symbols.',
    'min_quote_vol': 'Minimum 24h quote volume for inclusion in dynamic universe.',
    'equity_proxy': 'Synthetic account size used for sizing & daily loss guard.',
    'market_type': 'Market type: spot or future (USD-M).',
    'enable_bearish': 'Enable short (bearish) entries.',
    'ema_fast': 'Fast EMA period.',
    'ema_slow': 'Slow EMA period.',
    'atr_period': 'ATR period for volatility / stop sizing.',
    'atr_stop_mult': 'ATR multiple applied to derive initial stop distance.',
    'atr_tp_mult': 'Relative ATR multiple for take profit vs stop.',
    'min_stop_pct': 'Floor for stop percent if ATR-implied stop is too small.',
    'partial_tp_r_multiple': 'R multiple at which partial profit executes.',
    'partial_tp_fraction': 'Fraction of position closed at partial target.',
    'breakeven_after_partial': 'Move stop to entry after partial.',
    'trail_activate_r_multiple': 'R threshold to activate ATR trailing.',
    'trail_atr_mult': 'ATR multiple behind price for trailing stop.',
    'daily_loss_limit_pct': 'Daily realized loss limit (pct of equity_proxy) triggering pause.',
    'enable_daily_loss_guard': 'Enable daily loss guard mechanism.',
    'min_atr_rel': 'Minimum ATR/price ratio (15m) required to allow new entries.',
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
    'last_loop_ts': time.time(),
    'equity_history': [],
    'peak_equity': 0.0,
    # Daily loss guard tracking
    'current_day': '',
    'daily_realized_pnl': 0.0,
    'daily_guard_active': False,
    'bot_pause_reason': ''
}
status['atr_info'] = {}

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
        'equity_proxy': float,
        'market_type': str,
        'enable_bearish': bool,
    'ema_fast': int,
    'ema_slow': int,
    'atr_period': int,
    'atr_stop_mult': float,
    'atr_tp_mult': float,
    'min_stop_pct': float,
    'partial_tp_r_multiple': float,
    'partial_tp_fraction': float,
    'breakeven_after_partial': bool,
    'trail_activate_r_multiple': float,
    'trail_atr_mult': float,
    'daily_loss_limit_pct': float,
    'enable_daily_loss_guard': bool,
    'min_atr_rel': float,
    }
    out = {}
    for k, caster in type_map.items():
        if k not in incoming:
            continue
        v = incoming[k]
        try:
            if caster is bool:
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
                            rec = {
                                'symbol': p['symbol'],
                                'entry': float(p['entry']),
                                'qty': float(p['qty']),
                                'pnl': float(p.get('pnl', 0)),
                                'side': p.get('side', 'long'),
                                'sl_price': float(p.get('sl_price')) if p.get('sl_price') is not None else None,
                                'tp_price': float(p.get('tp_price')) if p.get('tp_price') is not None else None,
                                'atr_stop_pct': float(p.get('atr_stop_pct')) if p.get('atr_stop_pct') is not None else None,
                                'atr_tp_pct': float(p.get('atr_tp_pct')) if p.get('atr_tp_pct') is not None else None,
                                # Advanced management fields (may be absent in legacy saves)
                                'initial_risk_per_unit': float(p.get('initial_risk_per_unit')) if p.get('initial_risk_per_unit') is not None else None,
                                'target1_price': float(p.get('target1_price')) if p.get('target1_price') is not None else None,
                                'partial_filled': bool(p.get('partial_filled', False)),
                                'trailing_active': bool(p.get('trailing_active', False)),
                                'atr_at_entry': float(p.get('atr_at_entry')) if p.get('atr_at_entry') is not None else None,
                                'r_multiple': float(p.get('r_multiple')) if p.get('r_multiple') not in (None, '') else None,
                            }
                            out.append(rec)
                        except Exception:
                            continue
                    return out
    except Exception as e:
        status['logs'].append(f"Positions load error: {e}")
    return []

def save_positions(positions: list, path: str = POSITIONS_PATH):
    try:
        # Persist only required keys (defensive against non-serializable additions)
        serializable = []
        keep = {'symbol','entry','qty','pnl','side','sl_price','tp_price','atr_stop_pct','atr_tp_pct','initial_risk_per_unit','target1_price','partial_filled','trailing_active','atr_at_entry','r_multiple'}
        for p in positions:
            try:
                serializable.append({k: p.get(k) for k in keep})
            except Exception:
                continue
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, indent=2)
        return True
    except Exception as e:
        status['logs'].append(f"Positions save error: {e}")
        return False

# Initialize positions from disk
_pos = load_positions()
if _pos:
    status['open_positions'] = _pos

# --- Exit strategy helpers ---
def compute_exit_reasons(symbol: str, entry: float, qty: float, exchange, pos: dict | None = None) -> dict:
    reasons = []
    try:
        price = exchange.fetch_ticker(symbol)['last']
    except Exception:
        price = status['live_prices'].get(symbol)
    if not isinstance(price, (int, float)) or price is None:
        return {'should_exit': False, 'reasons': [], 'price': None}

    side = (pos or {}).get('side', 'long')
    sl_price = (pos or {}).get('sl_price')
    tp_price = (pos or {}).get('tp_price')
    # Fallback to static percentages if dynamic not present
    if sl_price is None or tp_price is None:
        if side == 'short':
            tp_price = entry * (1 - status['config']['take_profit_pct']) if tp_price is None else tp_price
            sl_price = entry * (1 + status['config']['stop_loss_pct']) if sl_price is None else sl_price
        else:
            tp_price = entry * (1 + status['config']['take_profit_pct']) if tp_price is None else tp_price
            sl_price = entry * (1 - status['config']['stop_loss_pct']) if sl_price is None else sl_price

    if side == 'short':
        if price <= tp_price:
            reasons.append('take_profit')
        if price >= sl_price:
            reasons.append('stop_loss')
    else:
        if price >= tp_price:
            reasons.append('take_profit')
        if price <= sl_price:
            reasons.append('stop_loss')

    # Indicator-based exits using configurable EMAs
    try:
        fast_p = int(status['config'].get('ema_fast', 21))
        slow_p = int(status['config'].get('ema_slow', 50))
    except Exception:
        fast_p, slow_p = 21, 50

    # 15m checks
    try:
        df15 = fetch_ohlcv_df(exchange, symbol, '15m', 120)
        df15['ema_fast'] = ema(df15['close'], fast_p)
        df15['ema_slow'] = ema(df15['close'], slow_p)
        df15['rsi'] = rsi(df15['close'], 14)
        if len(df15) >= 2:
            last, prev = df15.iloc[-1], df15.iloc[-2]
            if (prev['close'] >= prev['ema_fast']) and (last['close'] < last['ema_fast']):
                reasons.append(f'ema{fast_p}_cross_down_15m')
            if (prev['close'] <= prev['ema_fast']) and (last['close'] > last['ema_fast']):
                reasons.append(f'ema{fast_p}_cross_up_15m')
            # Fast/slow cross
            if prev['ema_fast'] >= prev['ema_slow'] and last['ema_fast'] < last['ema_slow']:
                reasons.append(f'ema{fast_p}_{slow_p}_bear_cross_15m')
            if prev['ema_fast'] <= prev['ema_slow'] and last['ema_fast'] > last['ema_slow']:
                reasons.append(f'ema{fast_p}_{slow_p}_bull_cross_15m')
            if (prev['rsi'] >= status['config']['rsi_overbought']) and (last['rsi'] < status['config']['rsi_overbought']):
                reasons.append('rsi_overbought_cross_down_15m')
    except Exception as e:
        status['logs'].append(f"Exit check 15m error {symbol}: {e}")

    # 1h momentum weakening & EMA fast break
    try:
        df1h = fetch_ohlcv_df(exchange, symbol, '1h', 120)
        df1h['ema_fast'] = ema(df1h['close'], fast_p)
        df1h['ema_slow'] = ema(df1h['close'], slow_p)
        df1h['rsi'] = rsi(df1h['close'], 14)
        if len(df1h) >= 2:
            l1, p1 = df1h.iloc[-1], df1h.iloc[-2]
            if (p1['close'] >= p1['ema_fast']) and (l1['close'] < l1['ema_fast']) and (l1['rsi'] < p1['rsi']):
                reasons.append(f'ema{fast_p}_break_1h_rsi_down')
            # Fast below slow with RSI weakening
            if (p1['ema_fast'] >= p1['ema_slow']) and (l1['ema_fast'] < l1['ema_slow']) and (l1['rsi'] < p1['rsi']):
                reasons.append(f'ema{fast_p}_{slow_p}_bear_cross_rsi_down_1h')
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
                if pos.get('side', 'long') == 'short':
                    # buy reduces short
                    cover_qty = min(qty, pos['qty'])
                    pnl = (pos['entry'] - price) * cover_qty
                    pos['qty'] -= cover_qty
                    if pos['qty'] <= 1e-12:
                        self.status['open_positions'] = [p for p in self.status['open_positions'] if p['symbol'] != symbol]
                    trade = {'time': ts, 'symbol': symbol, 'side': 'buy', 'qty': cover_qty, 'price': price, 'status': 'filled', 'pnl': pnl}
                else:
                    new_qty = pos['qty'] + qty
                    # Weighted average entry
                    pos['entry'] = (pos['entry'] * pos['qty'] + price * qty) / max(new_qty, 1e-12)
                    pos['qty'] = new_qty
                    trade = {'time': ts, 'symbol': symbol, 'side': 'buy', 'qty': qty, 'price': price, 'status': 'filled', 'pnl': 0}
            else:
                # Determine dynamic sl/tp and ATR from latest signal if present
                sl_price = None; tp_price = None; atr_stop_pct=None; atr_tp_pct=None; atr_at_entry=None
                partial_target_price = None; initial_risk_per_unit = None
                try:
                    for sig in self.status.get('signals', []):
                        if sig.get('symbol') == symbol:
                            atr_stop_pct = sig.get('atr_stop_pct')
                            atr_tp_pct = sig.get('atr_tp_pct')
                            atr_at_entry = sig.get('atr')
                            if atr_stop_pct and price > 0:
                                sl_price = price * (1 - atr_stop_pct)
                            if atr_tp_pct and price > 0:
                                tp_price = price * (1 + atr_tp_pct)
                            break
                except Exception:
                    pass
                # Compute initial risk per unit (R) using chosen stop
                if sl_price and price:
                    initial_risk_per_unit = price - sl_price
                # Compute partial target using configured R multiple if possible
                try:
                    r_mult = float(self.status['config'].get('partial_tp_r_multiple', 1.0))
                    if initial_risk_per_unit and r_mult > 0:
                        partial_target_price = price + initial_risk_per_unit * r_mult
                except Exception:
                    partial_target_price = None
                self.status['open_positions'].append({
                    'symbol': symbol,
                    'entry': price,
                    'qty': qty,
                    'pnl': 0,
                    'side': 'long',
                    'sl_price': sl_price,
                    'tp_price': tp_price,
                    'atr_stop_pct': atr_stop_pct,
                    'atr_tp_pct': atr_tp_pct,
                    'initial_risk_per_unit': initial_risk_per_unit,
                    'target1_price': partial_target_price,
                    'partial_filled': False,
                    'trailing_active': False,
                    'atr_at_entry': atr_at_entry,
                })
                trade = {'time': ts, 'symbol': symbol, 'side': 'buy', 'qty': qty, 'price': price, 'status': 'filled', 'pnl': 0}
        else:  # sell
            if pos:
                if pos.get('side', 'long') == 'long':
                    sell_qty = min(qty, pos['qty'])
                    pnl = (price - pos['entry']) * sell_qty
                    pos['qty'] -= sell_qty
                    if pos['qty'] <= 1e-12:
                        # close position fully
                        self.status['open_positions'] = [p for p in self.status['open_positions'] if p['symbol'] != symbol]
                    trade = {'time': ts, 'symbol': symbol, 'side': 'sell', 'qty': sell_qty, 'price': price, 'status': 'filled', 'pnl': pnl}
                else:
                    # add to short
                    new_qty = pos['qty'] + qty
                    pos['entry'] = (pos['entry'] * pos['qty'] + price * qty) / max(new_qty, 1e-12)
                    pos['qty'] = new_qty
                    trade = {'time': ts, 'symbol': symbol, 'side': 'sell', 'qty': qty, 'price': price, 'status': 'filled', 'pnl': 0}
            else:
                # open short in paper mode with dynamic sl/tp + advanced fields
                sl_price = None; tp_price = None; atr_stop_pct=None; atr_tp_pct=None; atr_at_entry=None
                partial_target_price = None; initial_risk_per_unit = None
                try:
                    for sig in self.status.get('signals', []):
                        if sig.get('symbol') == symbol:
                            atr_stop_pct = sig.get('atr_stop_pct')
                            atr_tp_pct = sig.get('atr_tp_pct')
                            atr_at_entry = sig.get('atr')
                            if atr_stop_pct and price > 0:
                                sl_price = price * (1 + atr_stop_pct)  # stop above entry for short
                            if atr_tp_pct and price > 0:
                                tp_price = price * (1 - atr_tp_pct)    # target below entry
                            break
                except Exception:
                    pass
                if sl_price and price:
                    initial_risk_per_unit = sl_price - price
                try:
                    r_mult = float(self.status['config'].get('partial_tp_r_multiple', 1.0))
                    if initial_risk_per_unit and r_mult > 0:
                        partial_target_price = price - initial_risk_per_unit * r_mult
                except Exception:
                    partial_target_price = None
                self.status['open_positions'].append({
                    'symbol': symbol,
                    'entry': price,
                    'qty': qty,
                    'pnl': 0,
                    'side': 'short',
                    'sl_price': sl_price,
                    'tp_price': tp_price,
                    'atr_stop_pct': atr_stop_pct,
                    'atr_tp_pct': atr_tp_pct,
                    'initial_risk_per_unit': initial_risk_per_unit,
                    'target1_price': partial_target_price,
                    'partial_filled': False,
                    'trailing_active': False,
                    'atr_at_entry': atr_at_entry,
                })
                trade = {'time': ts, 'symbol': symbol, 'side': 'sell', 'qty': qty, 'price': price, 'status': 'filled', 'pnl': 0}

        # Record trade in memory and CSV
        self.status.setdefault('recent_trades', []).insert(0, trade)
        self.status['recent_trades'] = self.status['recent_trades'][:20]
        try:
            # Insert dynamic sl/tp if opening new position
            slp = ''
            tpp = ''
            if pos is None:  # new position attempt
                # fetch last inserted (if any)
                try:
                    lastp = None
                    for p0 in self.status['open_positions']:
                        if p0['symbol'] == symbol:
                            lastp = p0
                    if lastp:
                        slp = lastp.get('sl_price') or ''
                        tpp = lastp.get('tp_price') or ''
                except Exception:
                    pass
            log_trade_csv(self.trade_log_path, [trade['time'], trade['symbol'], trade['side'], trade['qty'], price, tpp, slp, 'local', 'paper'])
        except Exception:
            pass
        # Apply per-symbol cooldown after a filled order
        try:
            cd = int(max(0, self.status['config'].get('cooldown', 0)))
            if cd > 0:
                current = float(self.status.get('cooldowns', {}).get(symbol, 0))
                self.status.setdefault('cooldowns', {})
                self.status['cooldowns'][symbol] = max(current, float(cd))
        except Exception:
            pass
        # Persist positions
        try:
            save_positions(self.status['open_positions'])
        except Exception:
            pass
        return trade

    def close_position(self, symbol: str, qty: float):
        # side-aware close: if short, buy to close; if long, sell to close
        pos = self._find_position(symbol)
        close_side = 'sell'
        if pos and pos.get('side', 'long') == 'short':
            close_side = 'buy'
        return self.create_order(symbol, close_side, qty)

exchange = None
local_broker = LocalBroker(status)
def ensure_exchange(force: bool = False):
    """Create or recreate the global exchange instance with expected options.
    Honours status['config']['market_type'] (spot|future).
    """
    global exchange
    if exchange is None or force:
        status['logs'].append('Initializing exchange...')
        print('Initializing exchange...')
        default_type = 'future' if str(status['config'].get('market_type', 'spot')).lower() == 'future' else 'spot'
        exchange = ccxt.binance({
            'apiKey': os.getenv('API_KEY'),
            'secret': os.getenv('API_SECRET'),
            'enableRateLimit': True,
            'timeout': int(os.getenv('CCXT_TIMEOUT_MS', '10000')),
            'options': {'defaultType': default_type, 'adjustForTimeDifference': True}
        })
        try:
            exchange.timeout = int(os.getenv('CCXT_TIMEOUT_MS', '10000'))
        except Exception:
            pass
        try:
            exchange.load_markets()
        except Exception:
            pass

def recompute_universe_and_prices():
    """Rebuild the trading universe based on current config and refresh live prices immediately.
    Returns the list of symbols in the refreshed universe.
    """
    ensure_exchange()
    # Mark immediate refresh time
    status['last_scan'] = time.strftime('%Y-%m-%d %H:%M:%S')
    mtype = str(status['config'].get('market_type', 'spot')).lower()
    try:
        mkts = exchange.markets if getattr(exchange, 'markets', None) else exchange.load_markets()
    except Exception:
        mkts = {}
    # Build sensible defaults based on available markets to avoid bad symbol forms
    default_pool_spot = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT']
    default_pool_fut = []
    if mkts:
        wanted_bases = ['BTC','ETH','BNB','SOL']
        for base in wanted_bases:
            # find a linear USDT contract symbol for each base
            for sym, m in mkts.items():
                try:
                    if m.get('base') == base and m.get('quote') == status['config']['base_quote'] and m.get('contract', False) and m.get('linear', False):
                        default_pool_fut.append(sym)
                        break
                except Exception:
                    continue
    if not default_pool_fut:
        # fallback to unified guesses; ccxt will error if invalid, but we will fallback per-symbol later
        default_pool_fut = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'BNB/USDT:USDT', 'SOL/USDT:USDT']
    symbols = status.get('universe') or (default_pool_fut if mtype == 'future' else default_pool_spot)
    try:
        if status['config']['use_dynamic_universe']:
            status['logs'].append('Refreshing dynamic universe...')
            tickers_all = exchange.fetch_tickers()
            candidates = []
            for sym, data in tickers_all.items():
                m = mkts.get(sym) or {}
                # Filter by desired market type and quote
                if m.get('quote') != status['config']['base_quote']:
                    continue
                if mtype == 'future':
                    if not (m.get('contract', False) and m.get('linear', False)):
                        continue
                else:
                    if not m.get('spot', True):
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
                symbols = (default_pool_fut if mtype == 'future' else default_pool_spot)
            status['logs'].append(f"Universe size now {len(symbols)}")
        else:
            # Static mode: just respect size over existing/default pool
            pool = symbols or (default_pool_fut if mtype == 'future' else ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT'])
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
                    # fall back to per-symbol fetch below
                    tickers = None
                    break
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

            # Cooldowns: decrement based on elapsed time and preserve across loops
            try:
                now_ts = time.time()
                prev_ts = float(status.get('last_loop_ts') or now_ts)
                dt = max(0.0, now_ts - prev_ts)
                new_cds = {}
                for s in symbols:
                    rem = float(status.get('cooldowns', {}).get(s, 0))
                    rem = max(0.0, rem - dt)
                    new_cds[s] = int(round(rem))
                status['cooldowns'] = new_cds
                status['last_loop_ts'] = now_ts
            except Exception as e:
                status['logs'].append(f"Cooldown update error: {e}")

            # Update P&L for local open positions (long/short aware)
            for pos in status['open_positions']:
                sym = pos['symbol']
                price = status['live_prices'].get(sym)
                try:
                    if isinstance(price, (int, float)):
                        if pos.get('side', 'long') == 'short':
                            pos['pnl'] = (float(pos['entry']) - float(price)) * float(pos['qty'])
                        else:
                            pos['pnl'] = (float(price) - float(pos['entry'])) * float(pos['qty'])
                        # R multiple tracking
                        risk_unit = pos.get('initial_risk_per_unit')
                        if risk_unit and risk_unit > 0:
                            if pos.get('side','long') == 'short':
                                pos['r_multiple'] = (float(pos['entry']) - float(price)) / risk_unit
                            else:
                                pos['r_multiple'] = (float(price) - float(pos['entry'])) / risk_unit
                        else:
                            pos['r_multiple'] = None
                except Exception:
                    pos['pnl'] = 0

            # --- Partial profit execution (Phase 2) ---
            try:
                partial_fraction = float(status['config'].get('partial_tp_fraction', 0.5))
                r_mult_cfg = float(status['config'].get('partial_tp_r_multiple', 1.0))
                breakeven_flag = bool(status['config'].get('breakeven_after_partial', True))
                if partial_fraction > 0 and r_mult_cfg > 0:
                    # Iterate over a snapshot to allow modifications via order execution
                    for pos in list(status['open_positions']):
                        if pos.get('partial_filled'):
                            continue
                        t1 = pos.get('target1_price')
                        if not t1:
                            continue
                        sym = pos['symbol']
                        price = status['live_prices'].get(sym)
                        if not isinstance(price, (int, float)):
                            continue
                        side = pos.get('side', 'long')
                        hit = False
                        if side == 'long' and price >= t1:
                            hit = True
                        elif side == 'short' and price <= t1:
                            hit = True
                        if not hit:
                            continue
                        # Determine close quantity
                        close_frac = max(0.0, min(1.0, partial_fraction))
                        qty_close = float(pos['qty']) * close_frac
                        if qty_close <= 1e-12:
                            continue
                        close_side = 'sell' if side == 'long' else 'buy'
                        # Execute partial/full close
                        local_broker.create_order(sym, close_side, qty_close)
                        # If still open (partial), mark flags & adjust stop to breakeven if configured
                        remaining = None
                        for p2 in status['open_positions']:
                            if p2['symbol'] == sym:
                                remaining = p2
                                break
                        if remaining:
                            remaining['partial_filled'] = True
                            if breakeven_flag:
                                # Move stop to entry (breakeven) only if improves protection
                                try:
                                    if side == 'long':
                                        if (remaining.get('sl_price') or 0) < remaining['entry']:
                                            remaining['sl_price'] = remaining['entry']
                                    else:  # short
                                        if (remaining.get('sl_price') or 1e12) > remaining['entry']:
                                            remaining['sl_price'] = remaining['entry']
                                except Exception:
                                    pass
                            status['logs'].append(f"Partial filled {sym} at ~R{r_mult_cfg} qty {qty_close:.6f}")
                        else:
                            status['logs'].append(f"Full exit via partial target {sym} (fraction {close_frac:.2f})")
                    # Persist after potential modifications
                    try:
                        save_positions(status['open_positions'])
                    except Exception:
                        pass
            except Exception as e:
                status['logs'].append(f"Partial exec error: {e}")

            # --- Trailing stop logic (Phase 2) ---
            try:
                trail_activate_r = float(status['config'].get('trail_activate_r_multiple', 1.5))
                trail_atr_mult = float(status['config'].get('trail_atr_mult', 1.0))
                if trail_activate_r > 0 and trail_atr_mult >= 0:
                    for pos in status['open_positions']:
                        try:
                            sym = pos['symbol']
                            price = status['live_prices'].get(sym)
                            if not isinstance(price, (int, float)):
                                continue
                            side = pos.get('side', 'long')
                            entry = float(pos['entry'])
                            risk_unit = pos.get('initial_risk_per_unit')
                            if not risk_unit or risk_unit <= 0:
                                continue
                            # Determine current R multiple achieved
                            if side == 'long':
                                r_now = (price - entry) / risk_unit
                            else:
                                r_now = (entry - price) / risk_unit
                            # Activate trailing after threshold and partial taken (optional; allow even if no partial)
                            if r_now >= trail_activate_r and not pos.get('trailing_active'):
                                pos['trailing_active'] = True
                                status['logs'].append(f"Trailing activated {sym} at R={r_now:.2f}")
                            if not pos.get('trailing_active'):
                                continue
                            # Need an ATR for dynamic trail; fall back to atr_at_entry if fresh ATR unavailable
                            atr_val = None
                            # Recompute a small ATR on 15m if possible for fresher trailing
                            try:
                                df_15m_tmp = fetch_ohlcv_df(exchange, sym, '15m', 60)
                                ap = int(status['config'].get('atr_period', 14))
                                a_tmp = atr(df_15m_tmp.copy(), ap)
                                if len(a_tmp.dropna()):
                                    atr_val = float(a_tmp.iloc[-1])
                            except Exception:
                                atr_val = None
                            if atr_val is None:
                                atr_val = pos.get('atr_at_entry')
                            if not atr_val or atr_val <= 0:
                                continue
                            # Proposed new stop using ATR multiple
                            if side == 'long':
                                new_sl = price - atr_val * trail_atr_mult
                                # Only ratchet upward (never loosen)
                                if new_sl > (pos.get('sl_price') or 0):
                                    pos['sl_price'] = new_sl
                            else:
                                new_sl = price + atr_val * trail_atr_mult
                                # Only ratchet downward (never loosen)
                                cur_sl = pos.get('sl_price') or 1e12
                                if new_sl < cur_sl:
                                    pos['sl_price'] = new_sl
                        except Exception:
                            continue
                    try:
                        save_positions(status['open_positions'])
                    except Exception:
                        pass
            except Exception as e:
                status['logs'].append(f"Trailing logic error: {e}")
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
                            if pos.get('side', 'long') == 'short':
                                unreal += (float(pos['entry']) - float(price)) * float(pos['qty'])
                            else:
                                unreal += (float(price) - float(pos['entry'])) * float(pos['qty'])
                        except Exception:
                            pass
                total_profit = realized + unreal

                # equity tracking (relative)
                equity = total_profit
                try:
                    # Append tuple (ts, equity) for new entries
                    status['equity_history'].append((time.time(), equity))
                except Exception:
                    try:
                        status['equity_history'].append(equity)
                    except Exception:
                        pass
                if len(status['equity_history']) > 1500:
                    status['equity_history'] = status['equity_history'][-1500:]
                if equity > status['peak_equity']:
                    status['peak_equity'] = equity
                peak = status['peak_equity'] or 0.0
                drawdown = ((peak - equity) / peak * 100.0) if peak > 0 else 0.0

                status['performance']['win_rate'] = win_rate
                status['performance']['total_profit'] = round(total_profit, 4)
                status['performance']['drawdown'] = round(drawdown, 2)
            except Exception as e:
                status['logs'].append(f"Perf calc error: {e}")

            # --- Daily loss guard update ---
            try:
                day_str = datetime.utcnow().strftime('%Y-%m-%d')
                if status.get('current_day') != day_str:
                    # New day: reset counters and guard
                    status['current_day'] = day_str
                    status['daily_realized_pnl'] = 0.0
                    status['daily_guard_active'] = False
                    if status.get('bot_status') != 'Running':
                        status['bot_status'] = 'Running'
                        status['bot_pause_reason'] = ''
                # Recompute realized from trade history for robustness
                realized_today = 0.0
                for t in status.get('recent_trades', []):
                    try:
                        # parse trade time date portion
                        tday = t.get('time', '')[:10]
                        if tday == day_str and isinstance(t.get('pnl'), (int, float)):
                            realized_today += float(t.get('pnl'))
                    except Exception:
                        continue
                status['daily_realized_pnl'] = realized_today
                if bool(status['config'].get('enable_daily_loss_guard')):
                    limit_pct = float(status['config'].get('daily_loss_limit_pct', 0.05))
                    equity_proxy = float(status['config'].get('equity_proxy', 0))
                    loss_limit_abs = equity_proxy * limit_pct
                    if realized_today <= -abs(loss_limit_abs) and not status.get('daily_guard_active'):
                        status['daily_guard_active'] = True
                        status['bot_status'] = 'Paused'
                        status['bot_pause_reason'] = f"Daily loss limit hit (<= -{loss_limit_abs:.2f})"
                        status['logs'].append('Daily loss guard activated.')
                else:
                    # Ensure guard not stuck if disabled mid-day
                    if status.get('daily_guard_active'):
                        status['daily_guard_active'] = False
                        status['bot_status'] = 'Running'
                        status['bot_pause_reason'] = ''
            except Exception as e:
                status['logs'].append(f"Daily guard error: {e}")

            # Multi-timeframe signal gating: 1d + 4h trend alignment, 1h bias, 15m execution
            detected_signals = []
            if status.get('daily_guard_active'):
                status['signals'] = []
                status['logs'].append('Signals suppressed (daily loss guard active)')
            else:
                for symbol in symbols:
                    try:
                        df_1d = fetch_ohlcv_df(exchange, symbol, '1d', 200)
                        df_4h = fetch_ohlcv_df(exchange, symbol, '4h', 200)
                        df_1h = fetch_ohlcv_df(exchange, symbol, '1h', 200)
                        df_15m = fetch_ohlcv_df(exchange, symbol, '15m', 120)
                        # ATR (15m) for dynamic stop sizing
                        try:
                            ap = int(status['config'].get('atr_period', 14))
                            stop_mult = float(status['config'].get('atr_stop_mult', 1.5))
                            tp_mult = float(status['config'].get('atr_tp_mult', 2.5))
                            min_stop_pct = float(status['config'].get('min_stop_pct', 0.004))
                            df_calc = df_15m.copy()
                            a = atr(df_calc, ap)
                            cur_atr = float(a.iloc[-1]) if len(a.dropna()) else None
                            if cur_atr and cur_atr > 0 and df_calc['close'].iloc[-1] > 0:
                                atr_stop_pct = max(cur_atr * stop_mult / df_calc['close'].iloc[-1], min_stop_pct)
                                atr_tp_pct = atr_stop_pct * (tp_mult / max(stop_mult, 1e-9))
                            else:
                                atr_stop_pct = None
                                atr_tp_pct = None
                        except Exception as e:
                            cur_atr = None
                            atr_stop_pct = None
                            atr_tp_pct = None
                            status['logs'].append(f"ATR err {symbol}: {e}")
                        trend1d = daily_trend_check(df_1d)
                        trend4h = daily_trend_check(df_4h)
                        bias1h = bias_check_1h(df_1h)
                        exec15 = execution_check_15m(df_15m)
                        # Bearish proxy using 4h EMAs
                        df4 = df_4h.copy()
                        df4['ema21'] = ema(df4['close'], 21)
                        df4['ema50'] = ema(df4['close'], 50)
                        last4 = df4.iloc[-1]
                        bearish_trend4h = bool((last4['close'] < last4['ema21']) and (last4['close'] < last4['ema50']))
                        momentum_down = bool(bias1h['rsi'] < status['config']['rsi_threshold'])
                        daily_bearish_align = (not trend1d['bullish']) and (not trend4h['bullish']) and bearish_trend4h

                        # Long: daily + 4h both bullish
                        atr_ok = True
                        try:
                            if cur_atr and df_15m['close'].iloc[-1] > 0:
                                atr_ok = (cur_atr / df_15m['close'].iloc[-1]) >= float(status['config'].get('min_atr_rel', 0.0))
                        except Exception:
                            atr_ok = True
                        if atr_ok and trend1d['bullish'] and trend4h['bullish'] and bias1h['near_ma'] and bias1h['momentum_ok'] and exec15['signal'] and exec15.get('direction') == 'up':
                            detected_signals.append({
                                'symbol': symbol,
                                'type': exec15.get('type', 'entry'),
                                'side': 'buy',
                                'time': now,
                                'trend1d': trend1d,
                                'trend4h': trend4h,
                                'bias1h': bias1h,
                                'atr': cur_atr,
                                'atr_stop_pct': atr_stop_pct,
                                'atr_tp_pct': atr_tp_pct,
                            })
                        # Short: require bearish alignment if enabled
                        elif atr_ok and status['config'].get('enable_bearish') and daily_bearish_align and bias1h['near_ma'] and momentum_down and exec15['signal'] and exec15.get('direction') == 'down':
                            detected_signals.append({
                                'symbol': symbol,
                                'type': exec15.get('type', 'entry'),
                                'side': 'sell',
                                'time': now,
                                'trend1d': {'bullish': False, 'above21': False, 'hhll': False},
                                'trend4h': {'bullish': False, 'above21': False, 'hhll': False},
                                'bias1h': bias1h,
                                'atr': cur_atr,
                                'atr_stop_pct': atr_stop_pct,
                                'atr_tp_pct': atr_tp_pct,
                            })
                        else:
                            status['logs'].append(
                                f"NoSignal {symbol}: 1dTrend={trend1d['bullish']} 4hTrend={trend4h['bullish']} nearEMA1h={bias1h['near_ma']} RSI1h={bias1h['rsi']:.2f} exec={exec15['signal']}"
                            )
                    except Exception as e:
                        status['logs'].append(f"Signal error for {symbol}: {e}")
            status['signals'] = detected_signals
            # Surface just the symbol list for easy UI checks
            try:
                status['signal_symbols'] = [d.get('symbol') for d in detected_signals]
                status['signal_sides'] = {d.get('symbol'): d.get('side', 'buy') for d in detected_signals}
            except Exception:
                status['signal_symbols'] = []

            # Build exit signals for open positions
            exit_sigs = []
            for pos in status['open_positions']:
                sym = pos['symbol']
                entry = float(pos['entry'])
                qty = float(pos['qty'])
                res = compute_exit_reasons(sym, entry, qty, exchange, pos)
                if res['should_exit']:
                    if pos.get('side', 'long') == 'short':
                        pnl = (entry - res['price']) * qty
                    else:
                        pnl = (res['price'] - entry) * qty
                    exit_sigs.append({'symbol': sym, 'reasons': res['reasons'], 'price': res['price'], 'pnl': pnl})
            status['exit_signals'] = exit_sigs

            # (cooldowns updated above)
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
        # basic guards
        cd = int(status.get('cooldowns', {}).get(symbol, 0))
        if cd > 0:
            raise Exception(f"{symbol} on cooldown ({cd}s)")
        if (not status['config']['paper_trading']) and status['config'].get('market_type','spot')=='spot' and str(side).lower()=='sell':
            raise Exception('Live short only supported on Futures. Switch market_type to future or enable paper_trading.')
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
        # determine side to close
        pos = None
        for p in status['open_positions']:
            if p['symbol'] == symbol:
                pos = p
                break
        close_side = 'sell'
        if pos and pos.get('side', 'long') == 'short':
            close_side = 'buy'
        if status['config']['paper_trading']:
            order = local_broker.create_order(symbol, close_side, q)
        else:
            order = exchange.create_order(symbol, 'market', close_side, q)
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

@app.route('/api/ohlcv', methods=['GET'])
def api_ohlcv():
    """Return OHLCV for a symbol and timeframe for charts.
    Query: symbol=BTC/USDT&timeframe=15m&limit=200
    """
    sym = request.args.get('symbol')
    tf = request.args.get('timeframe', '15m')
    try:
        limit = int(request.args.get('limit', '200'))
    except Exception:
        limit = 200
    if not sym:
        return make_response(jsonify({'error': 'symbol required'}), 400)
    try:
        ensure_exchange()
        df = fetch_ohlcv_df(exchange, sym, tf, limit)
        # compact payload
        out = []
        for _, r in df.iterrows():
            try:
                out.append({
                    'time': int(pd.Timestamp(r['timestamp']).timestamp()),
                    'open': float(r['open']),
                    'high': float(r['high']),
                    'low': float(r['low']),
                    'close': float(r['close']),
                    'volume': float(r['volume']),
                })
            except Exception:
                continue
        resp = make_response(jsonify({'symbol': sym, 'timeframe': tf, 'data': out}), 200)
        resp.headers['Access-Control-Allow-Origin'] = '*'
        return resp
    except Exception as e:
        return make_response(jsonify({'error': str(e)}), 500)

@app.route('/update-config', methods=['POST'])
def update_config():
    # Build dict from form fields, then coerce types
    form = request.form.to_dict(flat=True)
    # Only accept known keys, but handle booleans even if unchecked (missing => False)
    updates = {k: v for k, v in form.items() if k in status['config']}
    for bk in ('paper_trading', 'use_dynamic_universe', 'enable_bearish'):
        if bk in status['config']:
            updates[bk] = '1' if form.get(bk) is not None else '0'
    coerced = coerce_config_types(updates)
    # Merge into status
    # Detect if market_type changed to re-init exchange
    prev_mtype = str(status['config'].get('market_type', 'spot')).lower()
    status['config'].update(coerced)
    new_mtype = str(status['config'].get('market_type', 'spot')).lower()
    ok = save_persisted_config(status['config'])
    if ok:
        flash('Configuration saved.', 'success')
    else:
        flash('Failed to save configuration; using in-memory values.', 'error')
    # Apply config immediately to universe and prices so UI reflects changes without waiting
    try:
        # Re-init exchange if market type flipped
        if new_mtype != prev_mtype:
            ensure_exchange(force=True)
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
            {% if not request.cookies.get('refreshPaused') %}
            <meta http-equiv="refresh" content="15" />
            {% endif %}
            <title>Binance Topdown Bot Dashboard</title>
            <script src="https://cdn.tailwindcss.com"></script>
        </head>
        <body class="h-full text-slate-100">
            <header class="bg-gradient-to-r from-slate-900 to-slate-800 border-b border-slate-800">
                <div class="max-w-7xl mx-auto px-4 py-4 flex items-center justify-between">
                    <h1 class="text-xl font-semibold">Binance Topdown Bot</h1>
                    <div class="flex items-center gap-2">
                        <div id="candleClock" class="px-2 py-1 rounded border border-slate-700 text-xs text-slate-200">15m closes in --:--</div>
                        <button id="soundToggle" class="px-2 py-1 rounded border border-slate-700 text-xs text-slate-200 hover:bg-slate-800">Sound: Off</button>
                        <span class="inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium {{ 'bg-emerald-600 text-slate-900' if status['bot_status']=='Running' else 'bg-rose-600 text-white' }}">
                            {{ status['bot_status'] }}
                        </span>
                        {% if status.get('daily_guard_active') %}
                        <span class="inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium bg-amber-500 text-slate-900" title="Daily loss guard engaged">DL Guard</span>
                        {% endif %}
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
                        <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-4 gap-3 mt-3">
                            {% for symbol in uni[:status['config']['universe_size']] %}
                            <div class="rounded-lg border border-slate-800 bg-slate-900/60 p-3 cursor-pointer hover:bg-slate-900/80" onclick="openChartModal('{{ symbol }}')">
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
                                {% set sigsyms = status.get('signal_symbols', []) %}
                                {% if symbol in sigsyms and (status['cooldowns'].get(symbol, 0) | int) == 0 %}
                                    {% set sside = (status.get('signal_sides', {}).get(symbol) or 'buy') %}
                                    <div class="mt-2">
                                        {% if sside == 'sell' %}
                                            <span class="inline-flex items-center rounded-full bg-rose-600 text-white px-2 py-0.5 text-xs font-semibold">TAKE TRADE (sell)</span>
                                        {% else %}
                                            <span class="inline-flex items-center rounded-full bg-emerald-600 text-slate-900 px-2 py-0.5 text-xs font-semibold">TAKE TRADE (buy)</span>
                                        {% endif %}
                                    </div>
                                {% endif %}
                            </div>
                            {% endfor %}
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
                                        <th class="text-left py-2">1d Trend</th>
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
                                            {% set t1d = sig.get('trend1d') %}
                                            {% if t1d %}
                                            <span class="inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium {{ 'bg-emerald-600 text-slate-900' if t1d['bullish'] else 'bg-slate-700 text-slate-200' }}">bullish</span>
                                            <span class="inline-flex items-center rounded-full bg-slate-700 text-slate-200 px-2 py-0.5 text-xs font-medium">{{ status['config']['ema_fast'] }}/{{ 'Y' if t1d.get('above_fast') or t1d.get('above21') else 'N' }}</span>
                                            <span class="inline-flex items-center rounded-full bg-slate-700 text-slate-200 px-2 py-0.5 text-xs font-medium">HHLL/{{ 'Y' if t1d['hhll'] else 'N' }}</span>
                                            {% else %}
                                            <span class="text-xs text-slate-500">-</span>
                                            {% endif %}
                                        </td>
                                        <td class="py-2 space-x-1">
                                            <span class="inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium {{ 'bg-emerald-600 text-slate-900' if sig['trend4h']['bullish'] else 'bg-slate-700 text-slate-200' }}">bullish</span>
                                            <span class="inline-flex items-center rounded-full bg-slate-700 text-slate-200 px-2 py-0.5 text-xs font-medium">{{ status['config']['ema_fast'] }}/{{ 'Y' if sig['trend4h'].get('above_fast') or sig['trend4h'].get('above21') else 'N' }}</span>
                                            <span class="inline-flex items-center rounded-full bg-slate-700 text-slate-200 px-2 py-0.5 text-xs font-medium">HHLL/{{ 'Y' if sig['trend4h']['hhll'] else 'N' }}</span>
                                        </td>
                                        <td class="py-2 space-x-1">
                                            <span class="inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium {{ 'bg-emerald-600 text-slate-900' if sig['bias1h']['near_ma'] else 'bg-slate-700 text-slate-200' }}">near EMA{{ status['config']['ema_fast'] }}</span>
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
                                        <th class="text-left py-2">SL</th>
                                        <th class="text-left py-2">TP</th>
                                        <th class="text-left py-2">R</th>
                                        <th class="text-left py-2">Flags</th>
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
                                        <td class="py-2 text-xs">{% if pos.get('sl_price') %}{{ pos['sl_price']|round(6) }}{% else %}-{% endif %}</td>
                                        <td class="py-2 text-xs">{% if pos.get('tp_price') %}{{ pos['tp_price']|round(6) }}{% else %}-{% endif %}</td>
                                        <td class="py-2 text-xs">{% if pos.get('r_multiple') is not none %}{{ pos['r_multiple']|round(2) }}{% else %}-{% endif %}</td>
                                        <td class="py-2 text-xs space-x-1">
                                            {% if pos.get('partial_filled') %}<span class="inline-block px-1 rounded bg-indigo-600 text-[10px]">PART</span>{% endif %}
                                            {% if pos.get('trailing_active') %}<span class="inline-block px-1 rounded bg-amber-500 text-[10px] text-slate-900">TRAIL</span>{% endif %}
                                        </td>
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

                <section class="grid grid-cols-1 gap-4">
                    <div class="bg-slate-900/50 rounded-lg border border-slate-800 p-4">
                        <h2 class="text-sm font-semibold text-slate-300">Equity Curve</h2>
                        <div id="equityChart" class="w-full mt-3" style="height:220px;"></div>
                    </div>
                </section>

               

                <section class="grid grid-cols-1 xl:grid-cols-3 gap-4">
                    <div class="xl:col-span-2 bg-slate-900/50 rounded-lg border border-slate-800 p-4">
                        <h2 class="text-sm font-semibold text-slate-300">Place Trade</h2>
                        {% set eq = status['config']['equity_proxy'] %}
                        {% set risk = status['config']['risk'] %}
                        {% set sl = status['config']['stop_loss_pct'] %}
                        {% set cds = status['cooldowns'] %}
                        {% set sigs = status['signals'] %}
                        {% set usd_risk = (eq * risk) %}
                        {% if sigs %}
                        <div class="mt-2">
                            <h3 class="text-xs font-semibold text-slate-400">Ready to take</h3>
                            <div class="mt-2 grid grid-cols-1 sm:grid-cols-2 gap-2">
                                {% for sig in sigs %}
                                    {% set sym = sig['symbol'] %}
                                    {% set cd = cds.get(sym, 0) %}
                                    {% if cd == 0 %}
                                    {% set p = status['live_prices'].get(sym) %}
                                    <div class="rounded border border-slate-800 bg-slate-900/70 p-3">
                                        <div class="flex items-center justify-between">
                                            <div class="text-sm font-medium">{{ sym }}</div>
                                            <div class="text-xs text-slate-400">{{ sig['type'] }}{% if sig.get('side')=='sell' %} • sell{% endif %}</div>
                                        </div>
                                        <div class="mt-1 text-xs text-slate-400">Price:
                                            {% if p is string or p is none %}
                                                <span class="italic">N/A</span>
                                            {% else %}
                                                {{ p|round(6) }}
                                            {% endif %}
                                        </div>
                                        {% if p is number and p > 0 %}
                                            {% set atr_stop = sig.get('atr_stop_pct') %}
                                            {% if atr_stop %}
                                                {% set eff_stop = atr_stop if atr_stop > sl else sl %}
                                            {% else %}
                                                {% set eff_stop = sl %}
                                            {% endif %}
                                            {% set den = p * (eff_stop if eff_stop > 0 else 0.000001) %}
                                            {% set qty = (eq * risk) / den %}
                                            <div class="mt-1 text-xs text-slate-400">Risk: ${{ usd_risk|round(2) }} • Stop%: {{ (eff_stop*100)|round(2) }} • Qty: <span class="text-slate-200">{{ (qty if qty>0 else 0)|round(6) }}</span></div>
                                            {% if sig.get('atr') %}
                                            <div class="mt-1 text-xs text-slate-500">ATR: {{ sig['atr']|round(6) }} (stop {{ (sig.get('atr_stop_pct',0)*100)|round(2) }}%, tp {{ (sig.get('atr_tp_pct',0)*100)|round(2) }}%)</div>
                                            {% endif %}
                                            <form method="post" action="/trade" class="mt-2">
                                                <input type="hidden" name="symbol" value="{{ sym }}">
                                                <input type="hidden" name="side" value="{{ sig.get('side','buy') }}">
                                                <input type="hidden" name="qty" value="{{ (qty if qty>0 else 0)|round(6) }}">
                                                <button type="submit" class="px-2 py-1 rounded bg-emerald-600 hover:bg-emerald-500 text-slate-900 text-xs font-medium">Trade {{ sym }}</button>
                                            </form>
                                        {% else %}
                                            <div class="mt-1 text-xs text-amber-400">Waiting for price to size order…</div>
                                        {% endif %}
                                    </div>
                                    {% endif %}
                                {% endfor %}
                            </div>
                        </div>
                        {% endif %}
                                    <form method="post" action="/trade" class="mt-4 grid grid-cols-1 sm:grid-cols-4 gap-2 items-end">
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
                        <div class="mt-3 space-y-1 text-xs">
                            <div class="flex items-center justify-between">
                                <span class="text-slate-400">Daily Realized PnL</span>
                                <span class="{{ 'text-emerald-400' if status.get('daily_realized_pnl',0)>0 else ('text-rose-400' if status.get('daily_realized_pnl',0)<0 else 'text-slate-300') }}">{{ status.get('daily_realized_pnl',0)|round(2) }}</span>
                            </div>
                            {% if status.get('daily_guard_active') %}
                            <div class="text-amber-400">Guard Active: {{ status.get('bot_pause_reason','') }}</div>
                            {% endif %}
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
                                <label>market_type
                                    <select name="market_type" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1">
                                        <option value="spot" {% if status['config']['market_type']=='spot' %}selected{% endif %}>spot</option>
                                        <option value="future" {% if status['config']['market_type']=='future' %}selected{% endif %}>future</option>
                                    </select>
                                </label>
                                <label class="flex items-center gap-2">enable_bearish
                                    <input name="enable_bearish" type="checkbox" value="1" {% if status['config']['enable_bearish'] %}checked{% endif %} class="mt-1 rounded border-slate-700" />
                                </label>
                                <label>partial_tp_r_multiple
                                    <input name="partial_tp_r_multiple" type="number" step="0.1" min="0" value="{{ status['config']['partial_tp_r_multiple'] }}" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1" />
                                </label>
                                <label>partial_tp_fraction
                                    <input name="partial_tp_fraction" type="number" step="0.05" min="0" max="1" value="{{ status['config']['partial_tp_fraction'] }}" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1" />
                                </label>
                                <label class="flex items-center gap-2">breakeven_after_partial
                                    <input name="breakeven_after_partial" type="checkbox" value="1" {% if status['config']['breakeven_after_partial'] %}checked{% endif %} class="mt-1 rounded border-slate-700" />
                                </label>
                                <label>trail_activate_r_multiple
                                    <input name="trail_activate_r_multiple" type="number" step="0.1" min="0" value="{{ status['config']['trail_activate_r_multiple'] }}" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1" />
                                </label>
                                <label>trail_atr_mult
                                    <input name="trail_atr_mult" type="number" step="0.1" min="0" value="{{ status['config']['trail_atr_mult'] }}" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1" />
                                </label>
                                <label>daily_loss_limit_pct
                                    <input name="daily_loss_limit_pct" type="number" step="0.001" min="0" value="{{ status['config']['daily_loss_limit_pct'] }}" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1" />
                                </label>
                                <label class="flex items-center gap-2">enable_daily_loss_guard
                                    <input name="enable_daily_loss_guard" type="checkbox" value="1" {% if status['config']['enable_daily_loss_guard'] %}checked{% endif %} class="mt-1 rounded border-slate-700" />
                                </label>
                                <label>ema_fast
                                    <input name="ema_fast" type="number" step="1" min="1" value="{{ status['config']['ema_fast'] }}" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1" />
                                </label>
                                <label>ema_slow
                                    <input name="ema_slow" type="number" step="1" min="2" value="{{ status['config']['ema_slow'] }}" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1" />
                                </label>
                                <label>atr_period
                                    <input name="atr_period" type="number" step="1" min="1" value="{{ status['config']['atr_period'] }}" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1" />
                                </label>
                                <label>atr_stop_mult
                                    <input name="atr_stop_mult" type="number" step="0.1" min="0" value="{{ status['config']['atr_stop_mult'] }}" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1" />
                                </label>
                                <label>atr_tp_mult
                                    <input name="atr_tp_mult" type="number" step="0.1" min="0" value="{{ status['config']['atr_tp_mult'] }}" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1" />
                                </label>
                                <label>min_stop_pct
                                    <input name="min_stop_pct" type="number" step="0.0001" min="0" value="{{ status['config']['min_stop_pct'] }}" class="mt-1 w-full rounded bg-slate-800 border border-slate-700 px-2 py-1" />
                                </label>

                                <div class="sm:col-span-2 flex items-center justify-end gap-2">
                                    <button type="submit" class="px-3 py-2 rounded bg-emerald-600 hover:bg-emerald-500 text-slate-900 font-medium">Save</button>
                                </div>
                            </form>
                            {% if (not status['config']['paper_trading']) and status['config']['market_type']=='spot' and status['config']['enable_bearish'] %}
                                <div class="mt-2 text-xs text-amber-400">Warning: Live short selling requires Futures. Switch market_type to future or keep paper_trading on.</div>
                            {% endif %}
                        </div>
                    </div>
                </section>

                            <section>
                                <div class="bg-slate-900/50 rounded-lg border border-slate-800 p-4">
                        <footer class="max-w-7xl mx-auto p-4">
                            <div class="bg-slate-900/60 border border-slate-800 rounded-lg">
                                <button id="cfgLegendToggle" class="w-full flex items-center justify-between px-4 py-2 text-sm font-semibold">
                                    <span>Configuration Legend</span>
                                    <span id="cfgLegendState" class="text-xs text-slate-400">toggle</span>
                                </button>
                                <div id="cfgLegend" class="hidden max-h-80 overflow-y-auto">
                                    <table class="w-full text-xs">
                                        <thead class="bg-slate-800/60 sticky top-0">
                                            <tr>
                                                <th class="px-3 py-2 text-left font-medium text-slate-300">Key</th>
                                                <th class="px-3 py-2 text-left font-medium text-slate-300">Value</th>
                                                <th class="px-3 py-2 text-left font-medium text-slate-300">Description</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {% for k, v in status['config'].items() %}
                                            <tr class="border-b border-slate-800/50 hover:bg-slate-800/40">
                                                <td class="px-3 py-1 font-mono text-slate-200">{{ k }}</td>
                                                <td class="px-3 py-1 text-slate-300">{{ v }}</td>
                                                <td class="px-3 py-1 text-slate-400">{{ config_desc.get(k, '—') }}</td>
                                            </tr>
                                            {% endfor %}
                                            <!-- Derived metric definitions -->
                                            <tr class="border-b border-slate-800/50 bg-slate-800/30">
                                                <td class="px-3 py-1 font-mono text-amber-300">ATR</td>
                                                <td class="px-3 py-1 text-slate-300">derived</td>
                                                <td class="px-3 py-1 text-slate-400">Average True Range: moving average of True Range (max(high-low, |high-prevClose|, |low-prevClose|)); measures recent volatility for dynamic stop (atr_stop_mult), target (atr_tp_mult) sizing, and min volatility filter (min_atr_rel).</td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </footer>
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
            <!-- Chart Modal -->
            <div id="chartModal" class="hidden fixed inset-0 z-50 bg-black/70">
                <div class="absolute inset-0 flex items-center justify-center p-4">
                    <div class="w-full max-w-5xl bg-slate-900 rounded-lg border border-slate-800 shadow-xl">
                        <div class="flex items-center justify-between p-3 border-b border-slate-800">
                            <div class="text-sm font-semibold" id="chartTitle">Chart</div>
                            <div class="flex items-center gap-2">
                                <select id="tfSelect" class="rounded bg-slate-800 border border-slate-700 px-2 py-1 text-xs">
                                    <option value="15m">15m</option>
                                    <option value="1h">1h</option>
                                    <option value="4h">4h</option>
                                    <option value="1d">1d</option>
                                </select>
                                <button class="px-2 py-1 rounded border border-slate-700 text-slate-200 hover:bg-slate-800 text-xs" onclick="closeChartModal()">Close</button>
                            </div>
                        </div>
                        <div class="p-3">
                            <div id="chartContainerPrice" class="w-full" style="height: 420px;"></div>
                            <div id="chartContainerRSI" class="w-full mt-2" style="height: 160px;"></div>
                        </div>
                    </div>
                </div>
            </div>
                    <script src="https://unpkg.com/lightweight-charts@4.2.2/dist/lightweight-charts.standalone.production.js"></script>
                    <script>
                        // Legend persistence
                        (function(){
                            try {
                                const btn = document.getElementById('cfgLegendToggle');
                                const panel = document.getElementById('cfgLegend');
                                const stateSpan = document.getElementById('cfgLegendState');
                                if (!btn || !panel) return;
                                const KEY = 'legendOpen';
                                function apply(open){
                                    if (open){ panel.classList.remove('hidden'); stateSpan && (stateSpan.textContent='(open)'); }
                                    else { panel.classList.add('hidden'); stateSpan && (stateSpan.textContent='(closed)'); }
                                }
                                let open = false;
                                try { open = localStorage.getItem(KEY)==='1'; } catch(e){}
                                apply(open);
                                btn.addEventListener('click', ()=>{ open=!open; apply(open); try { localStorage.setItem(KEY, open?'1':'0'); } catch(e){} });
                            } catch(e){ console.error('legend persistence error', e); }
                        })();
                        // Equity curve rendering
                        (function(){
                            try {
                                const rawHist = {{ status['equity_history'] | tojson }};
                                // Normalize possible legacy (float) vs new (ts, equity)
                                const norm = [];
                                for (const item of rawHist){
                                    if (Array.isArray(item) && item.length===2){
                                        norm.push({ time: Math.floor(item[0]), value: item[1] });
                                    } else if (typeof item === 'number'){
                                        // fabricate time sequentially
                                        const base = (norm.length? norm[norm.length-1].time : Math.floor(Date.now()/1000)) + 60;
                                        norm.push({ time: base, value: item });
                                    }
                                }
                                const el = document.getElementById('equityChart');
                                if (el && norm.length){
                                    const ec = LightweightCharts.createChart(el,{ layout:{ background:{type:'solid',color:'#0f172a'}, textColor:'#cbd5e1'}, grid:{ vertLines:{color:'#1f2937'}, horzLines:{color:'#1f2937'} } });
                                    const line = ec.addLineSeries({ color:'#10b981', lineWidth:2 });
                                    line.setData(norm);
                                }
                            } catch(e) { console.error('equity chart err', e); }
                        })();
                        // Chart modal logic
                        // Auto-refresh control (pause while modal open)
                        function pauseAutoRefresh(){
                            const head = document.querySelector('head');
                            const meta = document.querySelector('meta[http-equiv="refresh"]');
                            if (meta){
                                try { localStorage.setItem('refreshContent', meta.getAttribute('content') || '15'); } catch(e){}
                                meta.remove();
                            }
                            try { localStorage.setItem('refreshPaused', '1'); } catch(e){}
                            // Also set cookie so next render omits meta tag entirely
                            try {
                                const d = new Date(Date.now() + 7*24*60*60*1000);
                                document.cookie = `refreshPaused=1; expires=${d.toUTCString()}; path=/`;
                            } catch(e){}
                        }
                        function resumeAutoRefresh(){
                            const head = document.querySelector('head');
                            const exists = document.querySelector('meta[http-equiv="refresh"]');
                            if (!exists && head){
                                let content = '15';
                                try { content = localStorage.getItem('refreshContent') || '15'; } catch(e){}
                                const m = document.createElement('meta');
                                m.setAttribute('http-equiv','refresh');
                                m.setAttribute('content', content);
                                head.appendChild(m);
                            }
                            try { localStorage.removeItem('refreshPaused'); } catch(e){}
                            // Clear cookie so future renders include meta again
                            try { document.cookie = 'refreshPaused=; Max-Age=0; path=/'; } catch(e){}
                        }
                        let chartPrice = null;
                        let chartRSI = null;
                        let candleSeries = null;
                        let volumeSeries = null;
                        let emaFastSeries = null;
                        let emaSlowSeries = null;
                        // Higher timeframe overlay EMA series
                        let emaFastSeries1h = null, emaSlowSeries1h = null;
                        let emaFastSeries4h = null, emaSlowSeries4h = null;
                        let emaFastSeries1d = null, emaSlowSeries1d = null;
                        let rsiSeries = null;
                        // Dynamic RSI threshold lines
                        let rsiThreshSeries = null; // rsi_threshold (e.g. 30)
                        let rsiExecSeries = null;   // rsi_exec_threshold (e.g. 45)
                        let rsiOverSeries = null;   // rsi_overbought (e.g. 70)
                        let currentSymbol = null;
                        let currentTf = '15m';
                        let restoredRangeOnce = false;
                        let userAdjusted = false;
                        let isProgrammaticRangeSet = false;
                        let saveRangeTimer = null;
                        let lastDataKey = null;
                        let inFlight = 0;
                        let pendingFetch = false;
                        function timeRangesEqual(a,b){
                            if (!a || !b) return false;
                            return a.from === b.from && a.to === b.to;
                        }
                        function getVisibleTimeRange(){
                            try { return chartPrice?.timeScale().getVisibleRange() || null; } catch(e){ return null; }
                        }
                        function setVisibleTimeRange(r){
                            if (!r) return;
                            try {
                                const cur = chartPrice.timeScale().getVisibleRange();
                                if (!timeRangesEqual(cur, r)){
                                    chartPrice.timeScale().setVisibleRange(r);
                                    chartRSI.timeScale().setVisibleRange(r);
                                }
                            } catch(e){}
                        }
                        function saveTimeRange(){
                            const r = getVisibleTimeRange();
                            if (r && r.from && r.to){
                                try { localStorage.setItem(`chartTimeRange:${currentSymbol}:${currentTf}`, JSON.stringify(r)); } catch(e){}
                            }
                        }
                        function rangesClose(a,b,eps=0.5){
                            if (!a || !b) return false;
                            return Math.abs((a.from||0) - (b.from||0)) < eps && Math.abs((a.to||0) - (b.to||0)) < eps;
                        }
                        function saveVisibleRange(){
                            if (!chartPrice) return;
                            try{
                                const rng = chartPrice.timeScale().getVisibleLogicalRange();
                                if (rng && isFinite(rng.from) && isFinite(rng.to)){
                                    const key = `chartRange:${currentSymbol}:${currentTf}`;
                                    localStorage.setItem(key, JSON.stringify(rng));
                                }
                            }catch(e){}
                        }
                        function openChartModal(symbol){
                            currentSymbol = symbol;
                            const modal = document.getElementById('chartModal');
                            const title = document.getElementById('chartTitle');
                            const tfSel = document.getElementById('tfSelect');
                            if (title) title.textContent = `${symbol} — Candles`;
                            if (modal) modal.classList.remove('hidden');
                            if (tfSel) tfSel.value = currentTf;
                            // pause auto refresh and persist state
                            pauseAutoRefresh();
                            try { localStorage.setItem('chartState', JSON.stringify({ open: true, symbol: currentSymbol, tf: currentTf })); } catch(e){}
                            // create chart if needed
                            restoredRangeOnce = false; userAdjusted = false;
                            const containerPrice = document.getElementById('chartContainerPrice');
                            const containerRSI = document.getElementById('chartContainerRSI');
                            if (containerPrice){
                                while (containerPrice.firstChild) containerPrice.removeChild(containerPrice.firstChild);
                                chartPrice = LightweightCharts.createChart(containerPrice, {
                                    layout: { background: { type: 'solid', color: '#0f172a' }, textColor: '#cbd5e1' },
                                    grid: { vertLines: { color: '#1f2937' }, horzLines: { color: '#1f2937' } },
                                    crosshair: { mode: 1 },
                                });
                                candleSeries = chartPrice.addCandlestickSeries({ upColor: '#10b981', downColor: '#ef4444', borderDownColor: '#ef4444', borderUpColor: '#10b981', wickDownColor: '#ef4444', wickUpColor: '#10b981' });
                                volumeSeries = chartPrice.addHistogramSeries({ priceFormat: { type: 'volume' }, priceScaleId: '' });
                                chartPrice.priceScale('')?.applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });
                                emaFastSeries = chartPrice.addLineSeries({ color: '#22d3ee', lineWidth: 1, title: 'EMA ' + {{ status['config']['ema_fast'] }} });
                                emaSlowSeries = chartPrice.addLineSeries({ color: '#f59e0b', lineWidth: 1, title: 'EMA ' + {{ status['config']['ema_slow'] }} });
                                // Overlay series (dashed)
                                emaFastSeries1h = chartPrice.addLineSeries({ color: '#0891b2', lineWidth: 1, lineStyle: 1 });
                                emaSlowSeries1h = chartPrice.addLineSeries({ color: '#0ea5e9', lineWidth: 1, lineStyle: 1 });
                                emaFastSeries4h = chartPrice.addLineSeries({ color: '#6366f1', lineWidth: 1, lineStyle: 1 });
                                emaSlowSeries4h = chartPrice.addLineSeries({ color: '#4f46e5', lineWidth: 1, lineStyle: 1 });
                                emaFastSeries1d = chartPrice.addLineSeries({ color: '#e2e8f0', lineWidth: 1, lineStyle: 1 });
                                emaSlowSeries1d = chartPrice.addLineSeries({ color: '#94a3b8', lineWidth: 1, lineStyle: 1 });
                                // watch for user zoom/pan and persist range
                chartPrice.timeScale().subscribeVisibleLogicalRangeChange(() => {
                                    if (!isProgrammaticRangeSet) {
                                        userAdjusted = true;
                                        if (saveRangeTimer) clearTimeout(saveRangeTimer);
                                        saveRangeTimer = setTimeout(saveVisibleRange, 300);
                    saveTimeRange();
                                    }
                                });
                            }
                            if (containerRSI){
                                while (containerRSI.firstChild) containerRSI.removeChild(containerRSI.firstChild);
                                chartRSI = LightweightCharts.createChart(containerRSI, {
                                    layout: { background: { type: 'solid', color: '#0f172a' }, textColor: '#cbd5e1' },
                                    grid: { vertLines: { color: '#1f2937' }, horzLines: { color: '#1f2937' } },
                                    crosshair: { mode: 1 },
                                    rightPriceScale: { visible: true },
                                });
                                rsiSeries = chartRSI.addLineSeries({ color: '#94a3b8', lineWidth: 1 });
                                rsiThreshSeries = chartRSI.addLineSeries({ color: '#475569', lineWidth: 1, lineStyle: 2 });
                                rsiExecSeries = chartRSI.addLineSeries({ color: '#0ea5e9', lineWidth: 1, lineStyle: 2 });
                                rsiOverSeries = chartRSI.addLineSeries({ color: '#f59e0b', lineWidth: 1, lineStyle: 2 });
                chartRSI.timeScale().subscribeVisibleLogicalRangeChange(() => {
                                    if (!isProgrammaticRangeSet) {
                                        userAdjusted = true;
                                        if (saveRangeTimer) clearTimeout(saveRangeTimer);
                                        saveRangeTimer = setTimeout(saveVisibleRange, 300);
                    saveTimeRange();
                                    }
                                });
                            }
                            fetchAndRender();
                        }
                        function closeChartModal(){
                            const modal = document.getElementById('chartModal');
                            if (modal) modal.classList.add('hidden');
                            // resume auto refresh and clear state
                            resumeAutoRefresh();
                            try { localStorage.removeItem('chartState'); } catch(e){}
                        }
                        // --- Indicators (client-side) ---
                        function computeEMA(values, period){
                            const k = 2 / (period + 1);
                            const ema = [];
                            let prev = null;
                            for (let i = 0; i < values.length; i++){
                                const v = values[i];
                                if (v == null || !isFinite(v)) { ema.push(null); continue; }
                                if (prev == null){ prev = v; ema.push(v); }
                                else { prev = v * k + prev * (1 - k); ema.push(prev); }
                            }
                            return ema;
                        }
                        function computeRSI(closes, period=14){
                            const rsi = new Array(closes.length).fill(null);
                            let gains = 0, losses = 0;
                            for (let i=1; i<=period; i++){
                                const ch = closes[i] - closes[i-1];
                                if (ch>=0) gains += ch; else losses -= ch;
                            }
                            let avgGain = gains/period;
                            let avgLoss = losses/period;
                            function calcRS(avgG, avgL){
                                if (avgL === 0) return 100;
                                const rs = avgG / avgL;
                                return 100 - (100 / (1 + rs));
                            }
                            rsi[period] = calcRS(avgGain, avgLoss);
                            for (let i=period+1; i<closes.length; i++){
                                const ch = closes[i] - closes[i-1];
                                const gain = Math.max(ch, 0);
                                const loss = Math.max(-ch, 0);
                                avgGain = (avgGain * (period - 1) + gain) / period;
                                avgLoss = (avgLoss * (period - 1) + loss) / period;
                                rsi[i] = calcRS(avgGain, avgLoss);
                            }
                            return rsi;
                        }
                        const CFG = { rsi_threshold: {{ status['config']['rsi_threshold'] }}, rsi_exec_threshold: {{ status['config']['rsi_exec_threshold'] }}, rsi_overbought: {{ status['config']['rsi_overbought'] }} };
                        async function fetchHigherTf(symbol){
                            if (currentTf !== '15m') return null; // only overlay on execution timeframe
                            try {
                                const qs = [
                                    fetch(`/api/ohlcv?symbol=${encodeURIComponent(symbol)}&timeframe=1h&limit=300`).then(r=>r.json()),
                                    fetch(`/api/ohlcv?symbol=${encodeURIComponent(symbol)}&timeframe=4h&limit=300`).then(r=>r.json()),
                                    fetch(`/api/ohlcv?symbol=${encodeURIComponent(symbol)}&timeframe=1d&limit=400`).then(r=>r.json()),
                                ];
                                const [d1h,d4h,d1d] = await Promise.all(qs);
                                return { d1h, d4h, d1d };
                            } catch(e){ console.error('higherTf fetch err', e); return null; }
                        }
                        function expandEMA(baseTimes, htData, period=21, period2=50){
                            // htData: array of {time, open, high, low, close}
                            if (!htData || !htData.length) return { ema21: [], ema50: [] };
                            const closes = htData.map(c=>c.close);
                            const e21 = computeEMA(closes, period);
                            const e50 = computeEMA(closes, period2);
                            // forward fill across base times
                            let idx = 0;
                            const out21 = [], out50 = [];
                            for (const t of baseTimes){
                                while (idx+1 < htData.length && htData[idx+1].time <= t) idx++;
                                const v21 = e21[idx];
                                const v50 = e50[idx];
                                out21.push(v21!=null?{ time: t, value: +v21.toFixed(8)}:null);
                                out50.push(v50!=null?{ time: t, value: +v50.toFixed(8)}:null);
                            }
                            return { ema21: out21.filter(Boolean), ema50: out50.filter(Boolean) };
                        }
                        async function fetchAndRender(){
                            if (!currentSymbol) return;
                            if (inFlight > 0){ pendingFetch = true; return; }
                            inFlight++;
                            // capture current logical range to restore after update if user adjusted
                            let rangeBefore = null;
                            try { if (chartPrice) rangeBefore = chartPrice.timeScale().getVisibleLogicalRange(); } catch(e){}
                            let timeRangeBefore = getVisibleTimeRange();
                            const url = `/api/ohlcv?symbol=${encodeURIComponent(currentSymbol)}&timeframe=${encodeURIComponent(currentTf)}&limit=500`;
                            try{
                                const res = await fetch(url);
                                const js = await res.json();
                                if (!js || !js.data) return;
                                const candles = js.data.map(r => ({ time: r.time, open: r.open, high: r.high, low: r.low, close: r.close }));
                                const dataKey = `${currentSymbol}|${currentTf}|${candles.length}|${candles.length?candles[candles.length-1].time:0}`;
                                const sameData = (dataKey === lastDataKey);
                                const vols = js.data.map(r => ({ time: r.time, value: r.volume, color: (r.close>=r.open?'#10b981':'#ef4444') }));
                                if (!sameData){
                                    if (candleSeries) candleSeries.setData(candles);
                                    if (volumeSeries) volumeSeries.setData(vols);
                                }
                                // indicators
                                const closes = js.data.map(r => r.close);
                                const fastP = {{ status['config']['ema_fast'] }};
                                const slowP = {{ status['config']['ema_slow'] }};
                                const emaFast = computeEMA(closes, fastP);
                                const emaSlow = computeEMA(closes, slowP);
                                const emaFastData = js.data.map((r,i) => (emaFast[i]!=null?{ time: r.time, value: +emaFast[i].toFixed(8)}:null)).filter(Boolean);
                                const emaSlowData = js.data.map((r,i) => (emaSlow[i]!=null?{ time: r.time, value: +emaSlow[i].toFixed(8)}:null)).filter(Boolean);
                                if (!sameData){
                                    if (emaFastSeries) emaFastSeries.setData(emaFastData);
                                    if (emaSlowSeries) emaSlowSeries.setData(emaSlowData);
                                }
                                // RSI pane
                                const rsi = computeRSI(closes, 14);
                                const rsiData = js.data.map((r,i) => (rsi[i]!=null?{ time: r.time, value: +rsi[i].toFixed(2)}:null)).filter(Boolean);
                                if (!sameData){ if (rsiSeries) rsiSeries.setData(rsiData); }
                                // Threshold lines (sparse approach: full span across times)
                                if (!sameData){
                                    const baseLine = candles.map(c => ({ time: c.time }));
                                    if (rsiThreshSeries) rsiThreshSeries.setData(baseLine.map(o => ({ time: o.time, value: CFG.rsi_threshold })));
                                    if (rsiExecSeries) rsiExecSeries.setData(baseLine.map(o => ({ time: o.time, value: CFG.rsi_exec_threshold })));
                                    if (rsiOverSeries) rsiOverSeries.setData(baseLine.map(o => ({ time: o.time, value: CFG.rsi_overbought })));
                                }
                                // Higher timeframe overlays only on 15m
                                if (!sameData && currentTf === '15m'){
                                    const higher = await fetchHigherTf(currentSymbol);
                                    if (higher){
                                        const bTimes = candles.map(c=>c.time);
                                        const mapHT = (d) => (d && d.data)? d.data.map(r=>({ time: r.time, open:r.open, high:r.high, low:r.low, close:r.close})):[];
                                        const d1h = mapHT(higher.d1h);
                                        const d4h = mapHT(higher.d4h);
                                        const d1d = mapHT(higher.d1d);
                                        // Use configured fast/slow periods for higher TF expansion too
                                        const h1 = expandEMA(bTimes, d1h, fastP, slowP);
                                        const h4 = expandEMA(bTimes, d4h, fastP, slowP);
                                        const hD = expandEMA(bTimes, d1d, fastP, slowP);
                                        if (emaFastSeries1h) emaFastSeries1h.setData(h1.ema21);
                                        if (emaSlowSeries1h) emaSlowSeries1h.setData(h1.ema50);
                                        if (emaFastSeries4h) emaFastSeries4h.setData(h4.ema21);
                                        if (emaSlowSeries4h) emaSlowSeries4h.setData(h4.ema50);
                                        if (emaFastSeries1d) emaFastSeries1d.setData(hD.ema21);
                                        if (emaSlowSeries1d) emaSlowSeries1d.setData(hD.ema50);
                                    }
                                } else if (!sameData && currentTf !== '15m') {
                                    [emaFastSeries1h, emaSlowSeries1h, emaFastSeries4h, emaSlowSeries4h, emaFastSeries1d, emaSlowSeries1d].forEach(s => { if (s) s.setData([]); });
                                }
                                // --- Markers: EMA crosses + patterns ---
                                const markers = [];
                                for (let i=1; i<emaFast.length; i++){
                                    const fPrev = emaFast[i-1], sPrev = emaSlow[i-1];
                                    const fCur = emaFast[i], sCur = emaSlow[i];
                                    if ([fPrev,sPrev,fCur,sCur].some(v => v==null)) continue;
                                    if (fPrev < sPrev && fCur > sCur){
                                        markers.push({ time: candles[i].time, position: 'belowBar', color: '#10b981', shape: 'arrowUp', text: `EMA${fastP}>${slowP}` });
                                    }
                                    if (fPrev > sPrev && fCur < sCur){
                                        markers.push({ time: candles[i].time, position: 'aboveBar', color: '#ef4444', shape: 'arrowDown', text: `EMA${fastP}<${slowP}` });
                                    }
                                }
                                // Bullish/Bearish engulfing + hammer/shooting-star markers
                                function isBullishEngulf(prev, cur){
                                    if (!(prev && cur)) return false;
                                    const prevBear = prev.close < prev.open;
                                    const curBull = cur.close > cur.open;
                                    if (!prevBear || !curBull) return false;
                                    const curBodyLow = Math.min(cur.open, cur.close);
                                    const curBodyHigh = Math.max(cur.open, cur.close);
                                    const prevBodyLow = Math.min(prev.open, prev.close);
                                    const prevBodyHigh = Math.max(prev.open, prev.close);
                                    return curBodyLow <= prevBodyHigh && curBodyHigh >= prevBodyLow;
                                }
                                function isBearishEngulf(prev, cur){
                                    if (!(prev && cur)) return false;
                                    const prevBull = prev.close > prev.open;
                                    const curBear = cur.close < cur.open;
                                    if (!prevBull || !curBear) return false;
                                    const curBodyLow = Math.min(cur.open, cur.close);
                                    const curBodyHigh = Math.max(cur.open, cur.close);
                                    const prevBodyLow = Math.min(prev.open, prev.close);
                                    const prevBodyHigh = Math.max(prev.open, prev.close);
                                    return curBodyLow <= prevBodyHigh && curBodyHigh >= prevBodyLow;
                                }
                                function isHammer(c){
                                    if (!c) return false;
                                    const body = Math.abs(c.close - c.open);
                                    const range = c.high - c.low;
                                    if (range <= 0) return false;
                                    const lower = Math.min(c.open, c.close) - c.low;
                                    const upper = c.high - Math.max(c.open, c.close);
                                    return lower >= 2*body && upper <= body && body/range < 0.4;
                                }
                                function isShootingStar(c){
                                    if (!c) return false;
                                    const body = Math.abs(c.close - c.open);
                                    const range = c.high - c.low;
                                    if (range <= 0) return false;
                                    const lower = Math.min(c.open, c.close) - c.low;
                                    const upper = c.high - Math.max(c.open, c.close);
                                    return upper >= 2*body && lower <= body && body/range < 0.4;
                                }
                                for (let i=1; i<candles.length; i++){
                                    const prev = candles[i-1], cur = candles[i];
                                    if (isBullishEngulf(prev, cur)){
                                        markers.push({ time: cur.time, position: 'belowBar', color: '#10b981', shape: 'circle', text: 'BE' });
                                    }
                                    if (isBearishEngulf(prev, cur)){
                                        markers.push({ time: cur.time, position: 'aboveBar', color: '#ef4444', shape: 'circle', text: 'BEAR' });
                                    }
                                    if (isHammer(cur)){
                                        markers.push({ time: cur.time, position: 'belowBar', color: '#22d3ee', shape: 'circle', text: 'H' });
                                    }
                                    if (isShootingStar(cur)){
                                        markers.push({ time: cur.time, position: 'aboveBar', color: '#f97316', shape: 'circle', text: 'SS' });
                                    }
                                }
                                if (!sameData){
                                    if (candleSeries && markers.length){ candleSeries.setMarkers(markers); } else if (candleSeries) { candleSeries.setMarkers([]); }
                                }
                                // Never auto-fit on update; restore previous or saved view once per open
                                if (chartPrice && chartRSI){
                                    const applyRange = () => {
                                        isProgrammaticRangeSet = true;
                                        try {
                                            // Prefer time-based range for stability
                                            const curTime = getVisibleTimeRange();
                                            if (userAdjusted && timeRangeBefore && !timeRangesEqual(curTime, timeRangeBefore)){
                                                setVisibleTimeRange(timeRangeBefore);
                                            } else if (!restoredRangeOnce) {
                                                let savedTime = null;
                                                try { savedTime = JSON.parse(localStorage.getItem(`chartTimeRange:${currentSymbol}:${currentTf}`) || 'null'); } catch(e){}
                                                if (savedTime && savedTime.from && savedTime.to && !timeRangesEqual(curTime, savedTime)){
                                                    setVisibleTimeRange(savedTime);
                                                } else if (!savedTime) {
                                                    const n = candles.length;
                                                    if (n > 0){
                                                        const last = candles[n-1]?.time;
                                                        const from = candles[Math.max(0, n-150)]?.time || candles[0]?.time;
                                                        const defTime = { from, to: last };
                                                        setVisibleTimeRange(defTime);
                                                    }
                                                }
                                                restoredRangeOnce = true;
                                            }
                                        } finally {
                                            isProgrammaticRangeSet = false;
                                        }
                                    };
                                    // Defer to next frame to avoid thrash
                                    requestAnimationFrame(applyRange);
                                }
                                lastDataKey = dataKey;
                            } catch(e) {
                                console.error('chart fetch error', e);
                            } finally {
                                inFlight = Math.max(0, inFlight-1);
                                if (pendingFetch){ pendingFetch = false; fetchAndRender(); }
                            }
                        }
                        const tfSel = document.getElementById('tfSelect');
                        if (tfSel){
                            tfSel.addEventListener('change', () => { currentTf = tfSel.value; restoredRangeOnce = false; userAdjusted = false; lastDataKey = null; pendingFetch = false; inFlight = 0; fetchAndRender(); });
                        }
                        window.openChartModal = openChartModal;
                        window.closeChartModal = closeChartModal;
                        // Restore chart after reload if needed
                        (function(){
                            try {
                                const st = JSON.parse(localStorage.getItem('chartState') || 'null');
                                if (st && st.open && st.symbol){
                                    currentTf = st.tf || '15m';
                                    pauseAutoRefresh();
                                    // open after a tiny delay to ensure DOM is ready
                                    setTimeout(()=> openChartModal(st.symbol), 0);
                                }
                            } catch(e){}
                        })();
                    </script>
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
                    <script>
                        // 15m candle-close countdown and cue
                        (function(){
                            const clockEl = document.getElementById('candleClock');
                            const header = document.querySelector('header');
                            const PERIOD_MS = 15 * 60 * 1000;

                            function nextBoundaryMs(now){
                                return Math.ceil(now / PERIOD_MS) * PERIOD_MS;
                            }

                            function fmt(ms){
                                const s = Math.max(0, Math.floor(ms / 1000));
                                const m = Math.floor(s / 60);
                                const ss = String(s % 60).padStart(2, '0');
                                return `${m}:${ss}`;
                            }

                            function flashHeader(){
                                if (!header) return;
                                header.classList.add('ring-2','ring-emerald-500','ring-offset-0');
                                setTimeout(()=> header.classList.remove('ring-2','ring-emerald-500','ring-offset-0'), 1200);
                            }

                            function getLastFired(){
                                try { return parseInt(localStorage.getItem('last15mBoundaryFired') || '0', 10); } catch { return 0; }
                            }
                            function setLastFired(v){
                                try { localStorage.setItem('last15mBoundaryFired', String(v)); } catch {}
                            }

                            // Initialize to current boundary to avoid double-fire on first paint
                            setLastFired(Math.floor(Date.now() / PERIOD_MS) * PERIOD_MS);

                            function tick(){
                                const now = Date.now();
                                const next = nextBoundaryMs(now);
                                const remain = next - now;
                                if (clockEl) clockEl.textContent = `15m closes in ${fmt(remain)}`;

                                if (remain <= 1000){
                                    const thisBoundary = next;
                                    if (getLastFired() !== thisBoundary){
                                        setLastFired(thisBoundary);
                                        flashHeader();
                                        if (typeof showToast === 'function') {
                                            showToast('<span class="font-semibold">15m candle closed</span> — new bar started');
                                        }
                                        if (typeof playBeep === 'function') {
                                            playBeep();
                                        }
                                    }
                                }
                            }

                            tick();
                            setInterval(tick, 500);
                        })();
                    </script>
        </body>
        </html>
    ''', status=status, config_desc=CONFIG_PARAM_DESCRIPTIONS)

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
