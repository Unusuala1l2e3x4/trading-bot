
from numba import jit
import numpy as np

from datetime import datetime, timedelta, time, date
from zoneinfo import ZoneInfo

from dataclasses import dataclass, field
from typing import Optional, Callable

    
@jit(nopython=True)
def np_median(arr):
    return np.median(arr)

@jit(nopython=True)
def np_mean(arr):
    return np.mean(arr)


# @dataclass
# class TouchDetectionParameters:
#     symbol: str
#     start_date: str
#     end_date: str
#     atr_period: int = 15
#     level1_period: int = 15
#     multiplier: float = 2.0
#     min_touches: int = 3
#     bid_buffer_pct: float = 0.005
#     start_time: Optional[str] = None
#     end_time: Optional[str] = None
#     use_median: bool = False
#     rolling_avg_decay_rate: float = 0.85
#     touch_area_width_agg: Callable = np.median
#     export_bars_path: Optional[str] = None

@dataclass
class BaseTouchDetectionParameters:
    symbol: str
    atr_period: int = 15
    level1_period: int = 15
    multiplier: float = 1.4
    min_touches: int = 3
    bid_buffer_pct: float = 0.005
    start_time: Optional[time] = None
    end_time: Optional[time] = None
    use_median: bool = False
    rolling_avg_decay_rate: float = 0.85
    # touch_area_width_agg: Callable = np.median
    touch_area_width_agg: Callable = np_median

@dataclass
# class BacktestTouchDetectionParameters(BaseTouchDetectionParameters):
class BacktestTouchDetectionParameters():
    symbol: str
    start_date: datetime # | str
    end_date: datetime # | str
    atr_period: int = 15
    level1_period: int = 15
    multiplier: float = 1.4
    min_touches: int = 3
    bid_buffer_pct: float = 0.005
    start_time: Optional[time] = None
    end_time: Optional[time] = None
    use_median: bool = False
    rolling_avg_decay_rate: float = 0.85
    # touch_area_width_agg: Callable = np.median
    touch_area_width_agg: Callable = np_median
    export_bars_path: Optional[str] = None
    export_quotes_path: Optional[str] = None

@dataclass
class LiveTouchDetectionParameters(BaseTouchDetectionParameters):
    pass  # This class doesn't need any additional parameters
