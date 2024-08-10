from datetime import datetime
from dataclasses import dataclass, field
from bisect import bisect_right, bisect_left
from collections import defaultdict
from itertools import takewhile

@dataclass
class TouchArea:
    def __init__(self, date, id, level, upper_bound, lower_bound, touches, is_long, min_touches, bid_buffer_pct):
        assert lower_bound < level < upper_bound
        self.date = date
        self.id = id
        self.level = level
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.touches = touches
        self.is_long = is_long
        self.min_touches = min_touches
        self.bid_buffer_pct = bid_buffer_pct # currently unused
        self.entries_exits = []
        self.min_touches_time = touches[min_touches - 1] if len(touches) >= min_touches else None # New attribute
        
    def __eq__(self, other):
        if isinstance(other, TouchArea):
            return self.id == other.id
        return NotImplemented
    
    
    def add_touch(self, touch_time: datetime):
        self.touches.append(touch_time)

    
    def record_entry_exit(self, entry_time: datetime, entry_price: float, exit_time: datetime, exit_price: float):
        self.entries_exits.append((entry_time, entry_price, exit_time, exit_price))

    @property
    def is_active(self) -> bool:
        return len(self.touches) >= self.min_touches

    @property
    def get_min_touch_time(self) -> datetime:
        return self.touches[self.min_touches-1] if self.touches else None
    
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
        for entry_time, entry_price, exit_time, exit_price in self.entries_exits:
            if self.is_long:
                total_profit += exit_price - entry_price
            else:
                total_profit += entry_price - exit_price
        return total_profit
    
    @property
    def get_range(self) -> float:
        return self.upper_bound - self.lower_bound
    
    def terminate(self, touch_area_collection):
        touch_area_collection.terminate_area(self)
    
    @property
    def __str__(self):
        return (f"{'Long ' if self.is_long else 'Short'} TouchArea {self.id}: Level={self.level:.4f}, "
                f"Bounds={self.lower_bound:.4f}-{self.upper_bound:.4f}, {self.touches[0].time()}-{self.touches[-1].time()}, "
                f"Num Touches={len(self.touches)}, "
                # f"Buy Price={self.get_buy_price}, "
                f"Profit={self.calculate_profit:.4f}")

@dataclass
class TouchAreaCollection:
    def __init__(self, touch_areas, min_touches):
        self.areas_by_date = defaultdict(list)
        
        for area in touch_areas:
            if len(area.touches) >= min_touches:
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
                    
                
                
