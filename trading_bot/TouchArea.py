from datetime import datetime, date
from dataclasses import dataclass, field
from bisect import bisect_right, bisect_left
from collections import defaultdict
from itertools import takewhile
from numba import jit
import numpy as np
from typing import List, Dict, Optional, Tuple, Callable, Set
import copy

from TypedBarData import TypedBarData

import logging
def setup_logger(log_level=logging.INFO):
    logger = logging.getLogger('TouchArea')
    logger.setLevel(log_level)

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Add a new handler
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

logger = setup_logger(logging.INFO)

def log(message, level=logging.INFO):
    logger.log(level, message, exc_info=level >= logging.ERROR)


@dataclass
class EntryExit:
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float

@dataclass
class TouchArea:
    date: date
    id: int # if id is smaller, the first price in the level is earlier in the day
    level: float
    lmin: float
    lmax: float
    upper_bound: float
    lower_bound: float
    initial_touches: List[datetime] # should be sorted
    touches: List[datetime] # should be sorted
    is_long: bool
    min_touches: int
    valid_atr: np.ndarray
    touch_area_width_agg: Callable
    calculate_bounds: Callable
    multiplier: float
    is_side_switched: bool = False
    bar_at_switch: TypedBarData = None
    min_touches_time: datetime = field(init=False)
    last_bounds_update_time: datetime = datetime.min
    entries_exits: List[EntryExit] = field(default_factory=list)
    
        
    def __post_init__(self):
        assert self.min_touches > 1, f'{self.min_touches} > 1'
        # assert self.lower_bound < self.level < self.upper_bound, f'{self.lower_bound} < {self.level} < {self.upper_bound}'
        assert self.lmin < self.level < self.lmax, f'{self.lmin} < {self.level} < {self.lmax}'
        assert len(self.valid_atr) == len(self.touches), f'{len(self.valid_atr)} == {len(self.touches)}'
        assert len(self.initial_touches) == self.min_touches, f'{len(self.initial_touches)} >= {self.min_touches}'
        self.min_touches_time = self.initial_touches[self.min_touches - 1] # if len(self.initial_touches) >= self.min_touches else None

    # Ensure objects are compared based on date and id
    def __eq__(self, other):
        if isinstance(other, TouchArea):
            return self.id == other.id and self.date == other.date
        return False

    # Ensure that objects have a unique hash based on date and id
    def __hash__(self):
        return hash((self.id, self.date))
    
    def add_touch(self, touch_time: datetime):
        self.touches.append(touch_time)

    
    def record_entry_exit(self, entry_time: datetime, entry_price: float, exit_time: datetime, exit_price: float):
        self.entries_exits.append(EntryExit(entry_time, entry_price, exit_time, exit_price))

    @property
    def is_active(self) -> bool:
        return len(self.initial_touches) >= self.min_touches

    @property
    def get_min_touch_time(self) -> datetime:
        # return self.initial_touches[self.min_touches-1] if self.initial_touches is None else None
        return self.min_touches_time
    
    @property
    def get_latest_touch(self, current_time: datetime) -> Optional[datetime]:
        if not self.touches:
            return None  # No touches, return None
        # Find the position to insert current_time in the sorted list
        index = bisect_right(self.touches, current_time) - 1
        # If index is negative, it means all touches are after current_time
        if index < 0:
            return None
        # Return the latest touch that is <= current_time
        return self.touches[index]
    
    @property
    def get_buy_price(self) -> float:
        return self.upper_bound if self.is_long else self.lower_bound
    
    @property
    def calculate_profit(self) -> float:
        total_profit = 0
        for _, entry_price, _, exit_price in self.entries_exits:
            if self.is_long:
                total_profit += exit_price - entry_price
            else:
                total_profit += entry_price - exit_price
        return total_profit
    
    @property
    def get_range(self) -> float:
        return self.upper_bound - self.lower_bound
    
    @property
    def get_level_range(self) -> float:
        return self.lmax - self.lmin
    
    def current_touches(self, current_timestamp: datetime):
        return [touch for touch in self.touches if touch <= current_timestamp]
    
    def switch_side(self, bar_at_switch=None): # , current_timestamp: datetime
        if not self.is_side_switched:
            self.is_long = not self.is_long
            self.is_side_switched = True
            self.bar_at_switch = bar_at_switch
        
    def reset_side(self): # , current_timestamp: datetime
        if self.is_side_switched:
            self.is_long = not self.is_long
            self.is_side_switched = False
            self.bar_at_switch = None
        
    def update_bounds(self, current_timestamp: datetime, monotonic_duration: Optional[int] = 0):
        if self.last_bounds_update_time == current_timestamp:
            return
        
        self.last_bounds_update_time = current_timestamp
        current_atr = self.valid_atr[:len(self.current_touches(current_timestamp))]
        
        _, new_lower_bound, new_upper_bound = self.calculate_bounds(
            current_atr, 
            self.level, 
            self.is_long, 
            self.touch_area_width_agg, 
            self.multiplier
        )
        if self.lower_bound != new_lower_bound or self.upper_bound != new_upper_bound:
            self.lower_bound = new_lower_bound
            self.upper_bound = new_upper_bound

    @staticmethod
    def print_areas_list(arr):
        k = lambda x: (x.lmin, x.level, x.lmax, x.id)
        ret = [f'Printing list of {len(arr)} areas...']
        for area in sorted(arr, key=k):
            ret.append(f"{area}")
        
        log('\n\t'.join(ret))
        
            
    # @property
    def __str__(self):
        return (f"{'Long ' if self.is_long else 'Short'} TouchArea {self.id}: Level={self.level:.4f} ({self.lmin:.4f},{self.lmax:.4f}), "
                f"Bounds={self.lower_bound:.4f}-{self.upper_bound:.4f}= {self.get_range:.4f}, initial touches [{" ".join([a.time().strftime("%H:%M") for a in self.initial_touches])}]"
                # f"Num Touches={len(self.touches)}"
                # f"Profit={self.calculate_profit:.4f}"
                )

    def is_within_range(self, low: float, high: float) -> bool:
        """
        Check if this area's bounds lie completely within the given range.
        """
        if low > high:
            high, low = low, high
        return low <= self.lower_bound <= high and low <= self.upper_bound <= high # Better??? or amplifies gain or loss
        # return low <= self.level <= high
        # return low <= self.lower_bound and self.upper_bound <= high


    def get_time_since_latest_touch(self, current_time: datetime, touches: List[datetime] = None) -> float:
        """Return minutes since most recent touch before current_time"""
        if touches is None:
            touches = self.current_touches(current_time)
        if not touches:
            return float('inf')
        return (current_time - touches[-1]).total_seconds() / 60

    def get_time_since_min_touch(self, current_time: datetime) -> float: #
        """Return minutes since min_touches was reached"""
        if not self.min_touches_time or current_time < self.min_touches_time:
            return float('inf')
        return (current_time - self.min_touches_time).total_seconds() / 60

    def get_touch_formation_time(self) -> float: #
        """Return minutes between first and min touch that made area active"""
        if len(self.initial_touches) < self.min_touches:
            return float('inf')
        return (self.initial_touches[-1] - self.initial_touches[0]).total_seconds() / 60

    def get_touch_density(self, current_time: datetime, touches: List[datetime] = None) -> float:
        """Return touches per minute up to current_time"""
        if touches is None:
            touches = self.current_touches(current_time)
        if not touches:
            return 0.0
        duration = (current_time - touches[0]).total_seconds() / 60
        return len(touches) / duration if duration > 0 else 0.0

    def get_touch_regularity(self, current_time: datetime, touches: List[datetime] = None) -> float:
        """
        Return coefficient of variation of time between touches.
        Lower values indicate more regular spacing.
        """
        if touches is None:
            touches = self.current_touches(current_time)
        if len(touches) < 2:
            return float('inf')
            
        intervals = np.array([(touches[i+1] - touches[i]).total_seconds() / 60 
                            for i in range(len(touches)-1)])
        return np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else float('inf')

    def get_atr_trend(self, current_time: datetime, touches: List[datetime] = None) -> float:
        """Return slope of ATR values over time (in minutes)"""
        if touches is None:
            touches = self.current_touches(current_time)
        if len(touches) < 2:
            return 0.0
        
        atr_values = self.valid_atr[:len(touches)]
        time_intervals = np.array([(touch - touches[0]).total_seconds() / 60 
                                 for touch in touches])
        return np.polyfit(time_intervals, atr_values, 1)[0]

    def get_metrics_dict(self, current_time: datetime, prefix: str = '') -> dict:
        """
        Get all metrics for the area at current_time.
        
        Args:
            current_time: Current timestamp
            prefix: Optional prefix for metric names (e.g. 'exit_')
            
        Returns:
            dict: Dictionary of metric names and their values
        """
        touches = self.current_touches(current_time)
        
        return {
            f'{prefix}time_since_latest_touch': self.get_time_since_latest_touch(current_time, touches),
            f'{prefix}time_since_min_touch': self.get_time_since_min_touch(current_time), #
            f'{prefix}touch_formation_time': self.get_touch_formation_time(), #
            f'{prefix}touch_density': round(self.get_touch_density(current_time, touches), 6),
            f'{prefix}touch_regularity': round(self.get_touch_regularity(current_time, touches), 6),
            f'{prefix}atr_trend': round(self.get_atr_trend(current_time, touches), 6),
            f'{prefix}num_touches': len(self.current_touches(current_time)), #
            # f'{prefix}area_width': self.get_range, #
        }
        
        
@dataclass
class TouchAreaCollection:
    touch_areas: List[TouchArea]
    min_touches: int
    areas_by_date: Dict[datetime, List[TouchArea]] = field(default_factory=lambda: defaultdict(list))
    active_date: datetime = None
    active_date_areas: List[TouchArea] = field(default_factory=list)
    terminated_date_areas: List[TouchArea] = field(default_factory=list)
    traded_date_areas: List[TouchArea] = field(default_factory=list)
    # open_position_areas: Set[TouchArea] = field(default_factory=set)
            
    def __post_init__(self):
        for area in self.touch_areas:
            if area.min_touches_time is not None:
                self.areas_by_date[area.date].append(area)
        
        for date in self.areas_by_date:
            self.areas_by_date[date].sort(key=self.area_sort_key)
            
    def get_all_areas(self, date: date):
        return self.areas_by_date[date]
    
    # Areas with earlier min_touches_time's have PRIORITY.
    # Secondarily, areas that have existed for longer have priority (earlier first initial touch)
    # Lastly by id, just in case
    def area_sort_key(self, area: TouchArea):
        # return (area.min_touches_time, area.id)
        return (area.min_touches_time, area.initial_touches[0], area.id)

    def get_area(self, area: TouchArea):
        index = bisect_left(self.active_date_areas, self.area_sort_key(area),
                            key=self.area_sort_key)
        if index < len(self.active_date_areas) and self.active_date_areas[index].id == area.id:
            return self.active_date_areas[index]
        return None
    
    def switch_side(self, area: TouchArea): # , current_timestamp: datetime
        index = bisect_left(self.active_date_areas, self.area_sort_key(area),
                            key=self.area_sort_key)
        if index < len(self.active_date_areas) and self.active_date_areas[index].id == area.id:
            self.active_date_areas[index].switch_side() # current_timestamp
            return True
        return False
    
    
    # def add_open_position_area(self, area: TouchArea):
    #     index = bisect_left(self.active_date_areas, self.area_sort_key(area),
    #                         key=self.area_sort_key)
    #     if index < len(self.active_date_areas) and self.active_date_areas[index].id == area.id:
    #         assert self.active_date_areas[index] not in self.open_position_areas
    #         self.open_position_areas.add(self.active_date_areas[index])
    #         return True
    #     return False
        
    # def del_open_position_area(self, area: TouchArea):
    #     index = bisect_left(self.active_date_areas, self.area_sort_key(area),
    #                         key=self.area_sort_key)
    #     if index < len(self.active_date_areas) and self.active_date_areas[index].id == area.id:
    #         self.open_position_areas.remove(self.active_date_areas[index])
    #         return True
    #     return False
    
    def reset_active_areas(self, current_time: datetime):
        try:
            current_date = current_time.date()
            if current_date not in self.areas_by_date:
                # log(f"1 {current_date} {current_time}")
                self.active_date = None
                self.active_date_areas = list()
                self.terminated_date_areas = list()
                self.traded_date_areas = list()
                return list()
            
            if self.active_date is None or self.active_date != current_date: # change the date and get areas in that date
                # log(f"2 {current_date} {current_time}")
                self.active_date = current_date
                # self.active_date_areas = self.areas_by_date[self.active_date]
                self.active_date_areas = copy.deepcopy(self.areas_by_date[self.active_date]) # NOTE: gets copy so that original isnt modified
                self.terminated_date_areas = list()
                self.traded_date_areas = list()
            
            # return list(takewhile(lambda area: area.min_touches_time <= current_time, self.active_date_areas)) # to enable area terminations without deleting the data
        except Exception as e:
            log(f"{type(e).__qualname__} in reset_active_areas at {current_time}: {e}", logging.ERROR)
            raise e
        
    def terminate_area(self, area: TouchArea):
        try:
            
            assert self.active_date is not None, (self.active_date, area.date)
            assert area.date == self.active_date, (area.date, self.active_date)
            
            # Find the exact position of the area to remove
            index = bisect_left(self.active_date_areas, self.area_sort_key(area),
                                key=self.area_sort_key)
            
            if index < len(self.active_date_areas) and self.active_date_areas[index].id == area.id:
                self.terminated_date_areas.append(self.active_date_areas[index])
                del self.active_date_areas[index]

        except Exception as e:
            log(f"{type(e).__qualname__} in terminate_area: {e}", logging.ERROR)
            raise e

    def add_traded_area(self, area: TouchArea):
        """Add area to the list of areas used for trading."""
        if area not in self.traded_date_areas:
            self.traded_date_areas.append(area)
            
    def remove_areas_in_range(self, low: float, high: float, current_time: datetime, other_areas_to_remove: List[TouchArea] = None,
                              is_long: bool=False, filter_side: bool=False) -> Set[int]:
        """
        Remove areas whose bounds lie completely within the given price range,
        plus any additional areas specified. Only removes areas that were already active
        by current_time (min_touches_time < current_time).
        
        Args:
            low (float): Lower bound of range to check
            high (float): Upper bound of range to check
            current_time: Current timestamp in backtest
            other_areas_to_remove: Optional list of additional areas to remove
            
        Returns:
            List[int]: IDs of all removed areas
        """
        if not self.active_date_areas:
            return []
        
        if low > high:
            high, low = low, high
            
        other_ids_to_remove = {area.id for area in other_areas_to_remove} if other_areas_to_remove else set()
        
        remaining = []
        removed_ids = set()
        
        for area in self.active_date_areas:
            # Only consider removing areas that were active before current_time
            if area.min_touches_time < current_time and ((filter_side and is_long == area.is_long) or not filter_side):
            # if area.min_touches_time <= current_time:
                area.update_bounds(current_time) # added, need to test
                if area.is_within_range(low, high) or area.id in other_ids_to_remove:
                    removed_ids.add(area.id)
                    self.terminated_date_areas.append(area)
                else:
                    remaining.append(area)
            else:
                # Keep areas that haven't become active yet
                remaining.append(area)
                
        if removed_ids:
            self.active_date_areas = remaining
            
        return removed_ids