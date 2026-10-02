"""Literal characterization at the approved existing evaluator boundaries."""
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from tests.legacy_sandbox import LegacySandbox

FIXTURES: dict[str, Any] = json.loads(
    (Path(__file__).parent / 'fixtures/legacy_strategy_v1.json').read_text(encoding='utf-8'),
)


def test_worked_ema_sequence() -> None:
    fixture = FIXTURES['ema']
    with LegacySandbox() as sandbox:
        evaluator = sandbox.load_module('utils')
        result = evaluator.ema(pd.Series(fixture['closes']), fixture['span'])
        assert result.tolist() == fixture['expected']


def test_worked_rolling_rsi_sequence() -> None:
    fixture = FIXTURES['rsi']
    with LegacySandbox() as sandbox:
        evaluator = sandbox.load_module('utils')
        result = evaluator.rsi(pd.Series(fixture['closes']), fixture['period'])
        for actual, expected in zip(result, fixture['expected'], strict=True):
            if expected is None:
                assert pd.isna(actual)
            else:
                assert actual == pytest.approx(expected)


@pytest.mark.parametrize('fixture', FIXTURES['pivots'], ids=lambda case: case['name'])
def test_close_based_pivots(fixture: dict[str, Any]) -> None:
    with LegacySandbox() as sandbox:
        evaluator = sandbox.load_module('utils')
        assert bool(evaluator.is_higher_highs_lows(pd.Series(fixture['closes']))) == fixture['expected']


@pytest.mark.parametrize('fixture', FIXTURES['candlesticks'], ids=lambda case: case['name'])
def test_worked_candlestick_patterns(fixture: dict[str, Any]) -> None:
    bars = pd.DataFrame(fixture['rows'], columns=['open', 'high', 'low', 'close'])
    with LegacySandbox() as sandbox:
        evaluator = sandbox.load_module('utils')
        assert bool(evaluator.detect_bullish_engulfing(bars)) == fixture['engulfing']
        assert bool(evaluator.detect_hammer(bars)) == fixture['hammer']


@pytest.mark.parametrize('profile', ['daemon_daily', 'dashboard_intraday'])
def test_existing_profile_trend_and_bias(profile: str) -> None:
    fixture = FIXTURES['profiles']
    with LegacySandbox() as sandbox:
        daemon = sandbox.load_module('topdown_daemon')
        evaluator = daemon if profile == 'daemon_daily' else sandbox.module
        trend_minutes, bias_minutes = (1440, 240) if profile == 'daemon_daily' else (240, 60)
        trend = evaluator.daily_trend_check(sandbox.candles(fixture['trend_closes'], trend_minutes))
        bias_bars = sandbox.candles(fixture['bias_closes'], bias_minutes)
        bias = evaluator.bias_check_4h(bias_bars) if profile == 'daemon_daily' else evaluator.bias_check_1h(bias_bars)
        assert trend == fixture['trend_expected']
        assert bias == fixture['bias_expected']


def test_profiles_preserve_documented_indicator_entry_difference() -> None:
    fixture = FIXTURES['profiles']
    with LegacySandbox() as sandbox:
        daemon = sandbox.load_module('topdown_daemon')
        bars = sandbox.candles(fixture['cross_closes'])
        assert daemon.execution_check_15m(bars) == fixture['daemon_cross_expected']
        assert sandbox.module.execution_check_15m(bars) == fixture['dashboard_cross_expected']


@pytest.mark.parametrize('module', ['dashboard', 'topdown_daemon'])
def test_both_profiles_keep_engulfing_entry(module: str) -> None:
    fixture = FIXTURES['candlesticks'][0]
    bars = pd.DataFrame(fixture['rows'], columns=['open', 'high', 'low', 'close'])
    with LegacySandbox() as sandbox:
        evaluator = sandbox.module if module == 'dashboard' else sandbox.load_module('topdown_daemon')
        assert evaluator.execution_check_15m(bars) == FIXTURES['profiles']['engulfing_expected']
