from datetime import datetime
from dataclasses import dataclass, field
from bisect import bisect_right, bisect_left
from collections import defaultdict
from itertools import takewhile
from numba import jit
import numpy as np
from typing import List, Dict, Optional, Tuple, Callable

@dataclass
class TouchArea:
    date: datetime.date
    id: int
    level: float
    upper_bound: float
    lower_bound: float
    initial_touches: List[datetime]
    touches: List[datetime]
    is_long: bool
    min_touches: int
    bid_buffer_pct: float
    valid_atr: np.ndarray
    touch_area_width_agg: Callable
    multiplier: float
    calculate_bounds: Callable
    min_touches_time: datetime = field(init=False)
    entries_exits: List[tuple] = field(default_factory=list)
    # fresh: bool = field(init=False)
        
    def __post_init__(self):
        assert self.min_touches > 1
        assert self.lower_bound < self.level < self.upper_bound
        assert len(self.valid_atr) == len(self.touches)
        self.min_touches_time = self.initial_touches[self.min_touches - 1] if len(self.initial_touches) >= self.min_touches else None
        # self.fresh = True

    def __eq__(self, other):
        if isinstance(other, TouchArea):
            return self.id == other.id and self.date == other.date
        return NotImplemented
    
    
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
    def get_last_touch(self) -> datetime:
        return self.touches[-1] if self.touches else None
    
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
    
    def terminate(self, touch_area_collection):
        # print(self.id,'end range  ',self.get_range)
        touch_area_collection.terminate_area(self)
        
    def update_bounds(self, current_timestamp: datetime):
        # if self.fresh:
        #     print(self.id,'start range',self.get_range)
        #     self.fresh = False
        current_touches = [touch for touch in self.touches if touch <= current_timestamp]
        current_atr = self.valid_atr[:len(current_touches)]
        
        _, new_lower_bound, new_upper_bound = self.calculate_bounds(
            current_atr, 
            self.level, 
            self.is_long, 
            self.touch_area_width_agg, 
            self.multiplier
        )
        
        self.lower_bound = new_lower_bound
        self.upper_bound = new_upper_bound
        
    @property
    def __str__(self):
        return (f"{'Long ' if self.is_long else 'Short'} TouchArea {self.id}: Level={self.level:.4f}, "
                f"Bounds={self.lower_bound:.4f}-{self.upper_bound:.4f}, {self.touches[0].time()}-{self.touches[-1].time()}, "
                f"Num Touches={len(self.touches)}, "
                f"Profit={self.calculate_profit:.4f}")

@dataclass
class TouchAreaCollection:
    def __init__(self, touch_areas, min_touches):
        self.areas_by_date = defaultdict(list)
        
        for area in touch_areas:
            if len(area.touches) >= min_touches:  #
                area_date = area.min_touches_time.date()
                self.areas_by_date[area_date].append(area)
        
        for date in self.areas_by_date:
            # Sort areas for each date by their min_touches_time, then by id
            self.areas_by_date[date].sort(key=lambda x: (x.min_touches_time, x.id))

    def get_active_areas(self, current_time: datetime):
        current_date = current_time.date()
        if current_date not in self.areas_by_date:
            return []
        
        return list(takewhile(lambda area: area.min_touches_time <= current_time, 
                              self.areas_by_date[current_date]))

    def terminate_area(self, area: TouchArea):
        area_date = area.min_touches_time.date()
        if area_date in self.areas_by_date:
            areas = self.areas_by_date[area_date]
            
            # Find the exact position of the area to remove
            index = bisect_left(areas, (area.min_touches_time, area.id), 
                                key=lambda x: (x.min_touches_time, x.id))
            
            if index < len(areas) and areas[index].id == area.id:
                del areas[index]
            
            if not areas:
                del self.areas_by_date[area_date]
                    
                
                
