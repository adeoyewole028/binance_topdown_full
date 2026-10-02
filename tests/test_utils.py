import pytest
import pandas as pd
import numpy as np
from utils import ema, rsi, atr, is_higher_highs_lows, detect_bullish_engulfing, detect_hammer, detect_bearish_engulfing, detect_shooting_star


class TestEMA:
    def test_ema_basic(self):
        series = pd.Series([100, 101, 102, 103])
        result = ema(series, 2)
        # EMA2: 100, (100+101)/2=100.5, (100.5*1 + 102*1)/2=101.25, (101.25*1 + 103*1)/2=102.125
        expected = pd.Series([100.0, 100.5, 101.25, 102.125])
        pd.testing.assert_series_equal(result, expected, atol=1e-6)

    def test_ema_single_value(self):
        series = pd.Series([100])
        result = ema(series, 2)
        expected = pd.Series([100.0])
        pd.testing.assert_series_equal(result, expected)

    def test_ema_empty_series(self):
        series = pd.Series([], dtype=float)
        result = ema(series, 2)
        assert len(result) == 0


class TestRSI:
    def test_rsi_basic(self):
        # Simple uptrend
        series = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114])
        result = rsi(series, 14)
        # RSI should be calculated for last 14 values
        assert len(result) == 15
        assert result.iloc[-1] > 50  # Uptrend, RSI > 50
        assert not np.isnan(result.iloc[-1])

    def test_rsi_constant(self):
        series = pd.Series([100] * 20)
        result = rsi(series, 14)
        # Constant price, RSI should be 50 or NaN
        assert result.iloc[-1] == 50.0 or np.isnan(result.iloc[-1])

    def test_rsi_short_series(self):
        series = pd.Series([100, 101])
        result = rsi(series, 14)
        # Not enough data, should be NaN
        assert np.isnan(result.iloc[-1])


class TestATR:
    def test_atr_basic(self):
        # Create OHLC data
        data = {
            'high': [105, 106, 107, 108, 109],
            'low': [95, 96, 97, 98, 99],
            'close': [100, 101, 102, 103, 104]
        }
        df = pd.DataFrame(data)
        result = atr(df, 2)
        assert len(result) == 5
        assert result.iloc[-1] > 0
        # First ATR is (high-low), then average
        expected_first = (105 - 95)  # 10
        expected_second = ((10 + (106 - 96)) / 2)  # 10
        assert abs(result.iloc[1] - expected_second) < 1e-6

    def test_atr_empty_df(self):
        df = pd.DataFrame()
        result = atr(df, 14)
        assert len(result) == 0

    def test_atr_none_df(self):
        result = atr(None, 14)
        assert len(result) == 0


class TestHigherHighsLows:
    def test_higher_highs_lows_true(self):
        # Ascending highs and lows
        series = pd.Series([100, 101, 102, 103, 104, 105])
        result = is_higher_highs_lows(series)
        assert result is True

    def test_higher_highs_lows_false(self):
        # Not ascending
        series = pd.Series([100, 101, 100, 103, 104, 105])
        result = is_higher_highs_lows(series)
        assert result is False

    def test_higher_highs_lows_short(self):
        series = pd.Series([100, 101])
        result = is_higher_highs_lows(series)
        assert result is False


class TestCandlestickPatterns:
    def test_detect_bullish_engulfing_true(self):
        # Previous bearish, current bullish engulfing
        data = {
            'open': [105, 100],
            'high': [110, 115],
            'low': [95, 90],
            'close': [100, 110]
        }
        df = pd.DataFrame(data)
        result = detect_bullish_engulfing(df)
        assert result is True

    def test_detect_bullish_engulfing_false(self):
        # Not engulfing
        data = {
            'open': [105, 106],
            'high': [110, 111],
            'low': [95, 96],
            'close': [100, 107]
        }
        df = pd.DataFrame(data)
        result = detect_bullish_engulfing(df)
        assert result is False

    def test_detect_hammer_true(self):
        # Hammer pattern
        data = {
            'open': [100],
            'high': [105],
            'low': [95],
            'close': [101]
        }
        df = pd.DataFrame(data)
        result = detect_hammer(df)
        assert result is True

    def test_detect_hammer_false(self):
        # Not hammer
        data = {
            'open': [100],
            'high': [105],
            'low': [99],
            'close': [101]
        }
        df = pd.DataFrame(data)
        result = detect_hammer(df)
        assert result is False

    def test_detect_bearish_engulfing_true(self):
        # Previous bullish, current bearish engulfing
        data = {
            'open': [100, 110],
            'high': [115, 120],
            'low': [90, 85],
            'close': [110, 100]
        }
        df = pd.DataFrame(data)
        result = detect_bearish_engulfing(df)
        assert result is True

    def test_detect_shooting_star_true(self):
        # Shooting star
        data = {
            'open': [100],
            'high': [110],
            'low': [99],
            'close': [101]
        }
        df = pd.DataFrame(data)
        result = detect_shooting_star(df)
        assert result is True

    def test_short_df(self):
        df = pd.DataFrame()
        assert detect_bullish_engulfing(df) is False
        assert detect_hammer(df) is False
        assert detect_bearish_engulfing(df) is False
        assert detect_shooting_star(df) is False