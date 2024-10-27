
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


@dataclass
class LiveTouchDetectionParameters:
    symbol: str
    atr_period: int = 15
    level1_period: int = 15
    multiplier: float = 1.4
    min_touches: int = 3
    bid_buffer_pct: float = 0.005
    start_time: Optional[time] = None
    end_time: Optional[time] = None
    use_median: bool = False
    # touch_area_width_agg: Callable = np.median
    touch_area_width_agg: Callable = np_median

    ema_span: float = 12
    price_ema_span: float = 26
    
@dataclass
class BacktestTouchDetectionParameters(LiveTouchDetectionParameters):
    start_date: datetime = field(default_factory=lambda: datetime.combine(datetime.now(), time.min)-timedelta(weeks=1)), # default to previous week
    end_date: datetime = field(default_factory=lambda: datetime.combine(datetime.now(), time.min)),
    export_bars_path: Optional[str] = None
    export_quotes_path: Optional[str] = None
