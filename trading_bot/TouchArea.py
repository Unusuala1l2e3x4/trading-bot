from datetime import datetime, date
from dataclasses import dataclass, field
from bisect import bisect_right, bisect_left
from collections import defaultdict
from itertools import takewhile
from numba import jit
import numpy as np
from typing import List, Dict, Optional, Tuple, Callable, Set

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
    logger.log(level, message)



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
    bid_buffer_pct: float
    valid_atr: np.ndarray
    touch_area_width_agg: Callable
    multiplier: float
    calculate_bounds: Callable
    min_touches_time: datetime = field(init=False)
    entries_exits: List[tuple] = field(default_factory=list)
        
    def __post_init__(self):
        assert self.min_touches > 1, f'{self.min_touches} > 1'
        assert self.lower_bound < self.level < self.upper_bound, f'{self.lower_bound} < {self.level} < {self.upper_bound}'
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
        self.entries_exits.append((entry_time, entry_price, exit_time, exit_price))

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
        # return self.upper_bound * (1 + self.bid_buffer_pct / 100) if self.is_long else self.lower_bound * (1 - self.bid_buffer_pct / 100)
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
    
    # def terminate(self, touch_area_collection):
    #     # print(self.id,self.entries_exits[0][2],len([touch for touch in self.touches if touch <= self.entries_exits[0][2]]),self.get_range,'\n')
    #     touch_area_collection.terminate_area(self)
    
    def current_touches(self, current_timestamp: datetime):
        return [touch for touch in self.touches if touch <= current_timestamp]
        
    def update_bounds(self, current_timestamp: datetime, monotonic_duration: Optional[int] = 0):        
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
            # print(self.id,current_timestamp,len(self.current_touches(current_timestamp)),self.get_range)

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

@dataclass
class TouchAreaCollection:
    touch_areas: List[TouchArea]
    min_touches: int
    areas_by_date: Dict[datetime, List[TouchArea]] = field(default_factory=lambda: defaultdict(list))
    active_date: datetime = None
    active_date_areas: List[TouchArea] = field(default_factory=list)
    terminated_date_areas: List[TouchArea] = field(default_factory=list)
    open_position_areas: Set[TouchArea] = field(default_factory=set)
            
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

    def add_open_position_area(self, area: TouchArea):
        index = bisect_left(self.active_date_areas, self.area_sort_key(area),
                            key=self.area_sort_key)
        if index < len(self.active_date_areas) and self.active_date_areas[index].id == area.id:
            assert self.active_date_areas[index] not in self.open_position_areas
            self.open_position_areas.add(self.active_date_areas[index])
            return True
        return False
        
    def del_open_position_area(self, area: TouchArea):
        index = bisect_left(self.active_date_areas, self.area_sort_key(area),
                            key=self.area_sort_key)
        if index < len(self.active_date_areas) and self.active_date_areas[index].id == area.id:
            self.open_position_areas.remove(self.active_date_areas[index])
            return True
        return False
    
    def get_active_areas(self, current_time: datetime):
        try:
            current_date = current_time.date()
            if current_date not in self.areas_by_date:
                # log(f"1 {current_date} {current_time}")
                self.active_date = None
                self.active_date_areas = list()
                self.terminated_date_areas = list()
                return list()
            
            if self.active_date is None or self.active_date != current_date: # change the date and get areas in that date
                # log(f"2 {current_date} {current_time}")
                self.active_date = current_date
                self.active_date_areas = list(self.areas_by_date[self.active_date]) # get copy so that original isnt modified
                self.terminated_date_areas = list()
            
            # return list(takewhile(lambda area: area.min_touches_time <= current_time, self.active_date_areas)) # to enable area terminations without deleting the data
        except Exception as e:
            log(f"{type(e).__qualname__} in get_active_areas at {current_time}: {e}", logging.ERROR)
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
