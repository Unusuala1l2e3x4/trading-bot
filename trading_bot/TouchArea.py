from datetime import datetime
from dataclasses import dataclass, field
from bisect import bisect_right
from collections import defaultdict

@dataclass
class TouchArea:
    def __init__(self, id, level, upper_bound, lower_bound, touches, is_long, min_touches, bid_buffer_pct):
        assert lower_bound < level < upper_bound
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
        self.date_sorted_times = defaultdict(list)
        
        for area in touch_areas:
            if len(area.touches) >= min_touches:
                area_date = area.min_touches_time.date()
                self.areas_by_date[area_date].append(area)
                if area.min_touches_time not in self.date_sorted_times[area_date]:
                    self.date_sorted_times[area_date].append(area.min_touches_time)
        
        for date in self.date_sorted_times:
            self.date_sorted_times[date].sort()

    def get_active_areas(self, current_time:datetime):
        current_date = current_time.date()
        if current_date not in self.date_sorted_times:
            return []
        
        index = bisect_right(self.date_sorted_times[current_date], current_time) # should be the last index if in real time
        active_times = self.date_sorted_times[current_date][:index]

        return [area for time in active_times for area in self.areas_by_date[current_date] if area.min_touches_time == time]

    def terminate_area(self, area:TouchArea):
        area_date = area.min_touches_time.date()
        if area_date in self.areas_by_date:
            self.areas_by_date[area_date] = [a for a in self.areas_by_date[area_date] if a.id != area.id]
            self.date_sorted_times[area_date] = [t for t in self.date_sorted_times[area_date] if t != area.min_touches_time]
            if not self.areas_by_date[area_date]:
                del self.areas_by_date[area_date]
                del self.date_sorted_times[area_date]
                
                
                
