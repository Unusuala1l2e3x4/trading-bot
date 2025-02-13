
from numba import jit
import numpy as np

from datetime import datetime, timedelta, time, date
from zoneinfo import ZoneInfo
ny_tz = ZoneInfo("America/New_York")

from dataclasses import dataclass, field
from typing import Optional, Callable

    
@jit(nopython=True)
def np_median(arr):
    return np.median(arr)

@jit(nopython=True)
def np_mean(arr):
    return np.mean(arr)



@jit(nopython=True)
def calculate_touch_area_bounds(atr_values, level, is_long, touch_area_width_agg, multiplier):
    touch_area_width = touch_area_width_agg(atr_values) * multiplier
    # print(is_long)
    
    # touch_area_low = level - (1 * touch_area_width / 3) if is_long else level - (2 * touch_area_width / 3)
    # touch_area_high = level + (2 * touch_area_width / 3) if is_long else level + (1 * touch_area_width / 3)

    # touch_area_low = level - (1 * touch_area_width / 2) if is_long else level - (1 * touch_area_width / 2)
    # touch_area_high = level + (1 * touch_area_width / 2) if is_long else level + (1 * touch_area_width / 2)
    
    if is_long:
        touch_area_low = level - (2 * touch_area_width / 3)
        touch_area_high = level + (1 * touch_area_width / 3)
    else:
        touch_area_low = level - (1 * touch_area_width / 3)
        touch_area_high = level + (2 * touch_area_width / 3)
    return touch_area_width, touch_area_low, touch_area_high



@jit(nopython=True)
def calculate_touch_area_bounds_mirrored(atr_values, level, is_long, touch_area_width_agg, multiplier):
    touch_area_width = touch_area_width_agg(atr_values) * multiplier
    # print(is_long)
    
    # touch_area_low = level - (1 * touch_area_width / 3) if is_long else level - (2 * touch_area_width / 3)
    # touch_area_high = level + (2 * touch_area_width / 3) if is_long else level + (1 * touch_area_width / 3)

    # touch_area_low = level - (1 * touch_area_width / 2) if is_long else level - (1 * touch_area_width / 2)
    # touch_area_high = level + (1 * touch_area_width / 2) if is_long else level + (1 * touch_area_width / 2)
    
    if not is_long:
        touch_area_low = level - (2 * touch_area_width / 3)
        touch_area_high = level + (1 * touch_area_width / 3)
    else:
        touch_area_low = level - (1 * touch_area_width / 3)
        touch_area_high = level + (2 * touch_area_width / 3)
    return touch_area_width, touch_area_low, touch_area_high



@jit(nopython=True)
def calculate_touch_area_bounds_mean_reversion(atr_values, level, is_long, touch_area_width_agg, multiplier):
    touch_area_width = touch_area_width_agg(atr_values) * multiplier
    # print(is_long)
    
    # touch_area_low = level - (1 * touch_area_width / 3) if is_long else level - (2 * touch_area_width / 3)
    # touch_area_high = level + (2 * touch_area_width / 3) if is_long else level + (1 * touch_area_width / 3)

    # touch_area_low = level - (1 * touch_area_width / 2) if is_long else level - (1 * touch_area_width / 2)
    # touch_area_high = level + (1 * touch_area_width / 2) if is_long else level + (1 * touch_area_width / 2)
    
    if not is_long:
        touch_area_high = level + (4 * touch_area_width / 3)
        touch_area_low = level + (1 * touch_area_width / 3)
    else:
        touch_area_high = level - (1 * touch_area_width / 3)
        touch_area_low = level - (4 * touch_area_width / 3)
    return touch_area_width, touch_area_low, touch_area_high



@jit(nopython=True)
def get_latest_value(values: np.ndarray) -> float:
    """
    Simple aggregator that returns the last value from an array.
    Since ATR/MTR are already calculated as EMA/rolling values,
    this captures current volatility while maintaining smoothing.
    
    Args:
        values: Array of ATR/MTR values at touch points
        
    Returns:
        float: The most recent ATR/MTR value
    """
    if len(values) == 0:
        return 0.0
    return values[-1]

@dataclass
class LiveTouchDetectionParameters:
    symbol: str
    client_type: str # {'stock','crypto'}
    atr_period: int = 15
    level1_period: int = 15
    multiplier: float = 1.725
    min_touches: int = 3
    # start_time: Optional[time] = '09:30'
    # end_time: Optional[time] = '16:00'
    start_time: Optional[time] = '00:00'
    end_time: Optional[time] = '23:59'
    use_median: bool = True # True is better
    touch_area_width_agg: Callable = get_latest_value
    # touch_area_width_agg: Callable = np.median # doesnt work inside jitted functions
    # touch_area_width_agg: Callable = np_median
    calculate_bounds: Callable = calculate_touch_area_bounds
    # calculate_bounds: Callable = calculate_touch_area_bounds_mean_reversion
    
    ema_span: float = 12
    price_ema_span: float = 26
    
    exit_ema_span: float = 26
    # exit_ema_span: float = 13
    
    # New parameters for volume processing
    volume_window: int = 15  # Window for spike detection
    volume_std_threshold: float = 3.0  # Number of standard deviations for spike detection

@dataclass
class BacktestTouchDetectionParameters(LiveTouchDetectionParameters):
    start_date: datetime = field(default_factory=lambda: datetime.combine(datetime.now(tz=ny_tz), time.min)-timedelta(weeks=1)), # default to previous week
    end_date: datetime = field(default_factory=lambda: datetime.combine(datetime.now(tz=ny_tz), time.min)),
    export_bars_path: Optional[str] = None
    export_quotes_path: Optional[str] = None
