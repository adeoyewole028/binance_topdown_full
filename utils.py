import os
import pandas as pd

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
