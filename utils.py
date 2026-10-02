import os
import time
import logging
import pandas as pd

logger = logging.getLogger(__name__)

# Global cache for OHLCV data
_ohlcv_cache = {}

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0.0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
    rs = gain / (loss.replace(0, 1e-12))
    return 100 - (100 / (1 + rs))

def is_higher_highs_lows(close: pd.Series) -> bool:
    def get_high(x):
        return x[1] if x[1] == max(x) else float('nan')
    def get_low(x):
        return x[1] if x[1] == min(x) else float('nan')
    highs = close.rolling(window=3, center=True).apply(get_high, raw=True).dropna()
    lows  = close.rolling(window=3, center=True).apply(get_low, raw=True).dropna()
    try:
        hh = highs.values[-3:]
        ll = lows.values[-3:]
        return len(hh)==3 and len(ll)==3 and hh[0]<hh[1]<hh[2] and ll[0]<ll[1]<ll[2]
    except Exception:
        return False

def detect_bullish_engulfing(df):
    if len(df) < 2: return False
    a = df.iloc[-2]; b = df.iloc[-1]
    prev_bear = a['close'] < a['open']
    curr_bull = b['close'] > b['open']
    if not (prev_bear and curr_bull): return False
    prev_low, prev_high = min(a['open'],a['close']), max(a['open'],a['close'])
    curr_low, curr_high = min(b['open'],b['close']), max(b['open'],b['close'])
    return curr_low <= prev_low and curr_high >= prev_high

def detect_hammer(df):
    if len(df) < 1: return False
    c = df.iloc[-1]
    body = abs(c['close'] - c['open'])
    lower_wick = min(c['open'], c['close']) - c['low']
    upper_wick = c['high'] - max(c['open'], c['close'])
    if body == 0: return False
    return (lower_wick >= 2 * body) and (upper_wick <= 0.5 * body)

def detect_bearish_engulfing(df):
    if len(df) < 2: return False
    a = df.iloc[-2]; b = df.iloc[-1]
    prev_bull = a['close'] > a['open']
    curr_bear = b['close'] < b['open']
    if not (prev_bull and curr_bear): return False
    prev_low, prev_high = min(a['open'],a['close']), max(a['open'],a['close'])
    curr_low, curr_high = min(b['open'],b['close']), max(b['open'],b['close'])
    return curr_low <= prev_low and curr_high >= prev_high

def detect_shooting_star(df):
    if len(df) < 1: return False
    c = df.iloc[-1]
    body = abs(c['close'] - c['open'])
    upper_wick = c['high'] - max(c['open'], c['close'])
    lower_wick = min(c['open'], c['close']) - c['low']
    if body == 0: return False
    return (upper_wick >= 2 * body) and (lower_wick <= 0.5 * body)

def ensure_logs(path='logs'):
    os.makedirs(path, exist_ok=True)

def log_trade_csv(path, row):
    ensure_logs(os.path.dirname(path) or '.')
    write_header = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        import csv
        w = csv.writer(f)
        if write_header:
            w.writerow(['time','symbol','side','qty','entry','tp','sl','reason','mode'])
        w.writerow(row)

def atr(df, period: int = 14):
    """Compute Average True Range. df must have columns: high, low, close."""
    if df is None or len(df) == 0:
        import pandas as pd
        return pd.Series(dtype=float)
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)
    tr = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = tr.combine(tr2, max).combine(tr3, max)
    return true_range.rolling(window=period).mean()

def fetch_ohlcv_cached(exchange, symbol: str, timeframe: str, limit: int = 200, ttl: int = 30) -> pd.DataFrame:
    """Fetch OHLCV data with in-memory caching to avoid repeated pulls.

    Args:
        exchange: ccxt exchange instance
        symbol: Trading pair symbol
        timeframe: Timeframe string (e.g., '15m')
        limit: Number of candles to fetch
        ttl: Cache time-to-live in seconds

    Returns:
        DataFrame with OHLCV data
    """
    global _ohlcv_cache
    key = (symbol, timeframe, limit)
    now_ts = time.time()
    rec = _ohlcv_cache.get(key)
    if rec and (now_ts - rec['ts'] < ttl):
        logger.debug(f"Cache hit for {key}")
        return rec['df'].copy()
    delay = 0.5
    last_err = None
    for attempt in range(3):
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            _ohlcv_cache[key] = {'df': df, 'ts': now_ts}
            logger.info(f"Fetched {len(df)} rows for {symbol} {timeframe}")
            return df
        except Exception as e:
            last_err = e
            logger.warning(f"Fetch attempt {attempt+1} failed for {symbol} {timeframe}: {e}")
            if attempt < 2:
                time.sleep(delay)
                delay *= 2
    logger.error(f"All fetch attempts failed for {symbol} {timeframe}: {last_err}")
    raise last_err

def detect_volatility_regime(df: pd.DataFrame, atr_period: int = 14, threshold: float = 1.0) -> str:
    """Detect volatility regime based on recent ATR relative to historical average.

    Args:
        df: DataFrame with OHLCV
        atr_period: Period for ATR calculation
        threshold: Multiplier for regime classification

    Returns:
        'high_vol' if current ATR > threshold * historical mean, else 'low_vol'
    """
    if len(df) < atr_period * 2:
        return 'neutral'
    atr_series = atr(df, atr_period)
    if len(atr_series.dropna()) < atr_period:
        return 'neutral'
    current_atr = atr_series.iloc[-1]
    historical_mean = atr_series.iloc[:-1].mean()  # Exclude current
    if pd.isna(current_atr) or pd.isna(historical_mean) or historical_mean == 0:
        return 'neutral'
    if current_atr > threshold * historical_mean:
        return 'high_vol'
    else:
        return 'low_vol'
