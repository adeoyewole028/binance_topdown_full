"""Explicit bounded UTC historical coverage; incomplete data is not a result."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from typing import Any, Protocol

from strategy_profiles import PROFILES, DURATIONS, completed_frame
from request_budget import ProviderFailure


class HistoricalSource(Protocol):
    def candles_range(self, symbol: str, timeframe: str, start: int, end: int,
                      limit: int) -> list[list[float]]: ...


def requirements(profile: str, start: int, end: int) -> dict[str, tuple[int,int]]:
    if profile not in PROFILES or type(start) is not int or type(end) is not int or not 0 <= start < end:
        raise ValueError('Explicit profile and increasing UTC millisecond range required')
    datetime(1970,1,1,tzinfo=timezone.utc)+timedelta(milliseconds=end)
    selected=PROFILES[profile]
    return {timeframe:(start//(DURATIONS[timeframe]*1000)*(DURATIONS[timeframe]*1000)-count*DURATIONS[timeframe]*1000,
                       end//(DURATIONS[timeframe]*1000)*(DURATIONS[timeframe]*1000))
            for timeframe,count in ((selected.trend,selected.trend_window),(selected.bias,selected.bias_window),('15m',selected.execution_window))}


def validate_history(bars: dict[str,Any], *, symbol: str, venue: str, profile: str,
                     start: int, end: int, as_of: int) -> dict[str,Any]:
    import re
    if venue not in {'testnet','mainnet'} or not re.fullmatch(r'[A-Z0-9]{2,12}/[A-Z0-9]{2,12}',symbol):
        raise ValueError('Explicit venue and canonical Spot symbol required')
    if type(as_of) is not int or as_of < 0:
        raise ValueError('Explicit UTC millisecond capture time required')
    datetime(1970,1,1,tzinfo=timezone.utc)+timedelta(milliseconds=as_of)
    coverage: dict[str,Any]={}
    retained: dict[str,Any]={}
    for timeframe,(lower,upper) in requirements(profile,start,end).items():
        duration=DURATIONS[timeframe]*1000
        expected=(upper-lower)//duration
        try:
            rows=[]
            for row in bars.get(timeframe,[]):
                if not row or isinstance(row[0],bool) or not math.isfinite(float(row[0])):
                    raise ValueError('invalid_timestamp')
                stamp=float(row[0])
                if not lower<=stamp<upper or stamp+duration>as_of:
                    continue
                if len(row)!=6 or any(isinstance(v,bool) or not math.isfinite(float(v)) for v in row):
                    raise ValueError('invalid_bar')
                stamp,opening,high,low,closing,volume=map(float,row)
                if stamp % duration or min(opening,high,low,closing)<=0 or volume<0 or not low<=min(opening,closing)<=max(opening,closing)<=high:
                    raise ValueError('invalid_ohlc')
                rows.append([stamp,opening,high,low,closing,volume])
            timestamps=[r[0] for r in rows]
            if timestamps != sorted(set(timestamps)):
                raise ValueError('unordered_or_duplicate_history')
            missing=expected-len(timestamps)
            reason='missing_history' if missing else None
            if not missing:
                completed_frame(rows,timeframe,end/1000,expected,expected)
            retained[timeframe]=rows
            coverage[timeframe]={'requested_start':lower,'requested_end':upper,'expected':expected,
                                 'received':len(rows),'missing':missing,'reason':reason}
        except (ValueError,TypeError,IndexError,OverflowError):
            retained[timeframe]=[]
            coverage[timeframe]={'requested_start':lower,'requested_end':upper,'expected':expected,
                                 'received':0,'missing':expected,'reason':'invalid_history'}
    complete=all(c['reason'] is None for c in coverage.values())
    return {'state':'complete' if complete else 'incomplete','symbol':symbol,'venue':venue,
            'profile':profile,'profile_version':PROFILES[profile].version,'start':start,'end':end,'as_of':as_of,
            'coverage':coverage,'bars':retained,
            'data_sha256':hashlib.sha256(json.dumps(retained,separators=(',',':'),allow_nan=False).encode()).hexdigest()}


def fetch_history(source: HistoricalSource, *, symbol: str, venue: str, profile: str,
                  start: int, end: int, as_of: int) -> dict[str,Any]:
    # Validate the entire request before any external call.
    validate_history({},symbol=symbol,venue=venue,profile=profile,start=start,end=end,as_of=as_of)
    bars: dict[str,Any]={}
    failures: dict[str,str]={}
    for timeframe,(lower,upper) in requirements(profile,start,end).items():
        duration=DURATIONS[timeframe]*1000
        upper=min(upper,as_of//duration*duration)
        cursor=lower
        collected: list[list[float]]=[]
        try:
            while cursor<upper:
                page=source.candles_range(symbol,timeframe,cursor,upper,1000)
                if not page:
                    break
                if len(page)>1000 or any(len(r)!=6 for r in page):
                    raise ValueError('invalid_page')
                stamps=[r[0] for r in page]
                if stamps != sorted(set(stamps)):
                    raise ValueError('unordered_page')
                progress=[r for r in page if cursor<=r[0]<upper]
                if not progress:
                    break
                collected.extend(progress)
                cursor=int(progress[-1][0])+duration
        except (ProviderFailure,TimeoutError,ConnectionError,ValueError,TypeError,IndexError,OverflowError) as error:
            failures[timeframe]=error.kind if isinstance(error,ProviderFailure) else 'transient' if isinstance(error,(TimeoutError,ConnectionError)) else 'invalid_history'
        bars[timeframe]=collected
    result=validate_history(bars,symbol=symbol,venue=venue,profile=profile,start=start,end=end,as_of=as_of)
    for timeframe,reason in failures.items():
        result['coverage'][timeframe]['reason']=reason
    if failures:result['state']='incomplete'
    return result
