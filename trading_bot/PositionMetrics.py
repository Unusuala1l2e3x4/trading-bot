from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional
import numpy as np

@dataclass
class PositionSnapshot:
    """Records position metrics at a specific minute."""
    timestamp: datetime
    is_long: bool
    
    # Price data
    close: float
    high: float
    low: float
    
    # Position sizing at this minute
    shares: int
    max_shares: int
    max_target_shares_limit: int
    
    # Area metrics  
    area_width: float
    area_buy_price: float  # Original trigger price from area
    avg_entry_price: float  # Current average entry price
    
    # P&L metrics at this minute
    running_pl: float  # Includes realized, unrealized, and fees
    running_pl_pct: float  # As percentage of initial balance
    realized_pl: float  # Store components for analysis
    unrealized_pl: float
    total_fees: float

    @property
    def is_profitable(self) -> bool:
        """Check if position is profitable at this snapshot."""
        return self.running_pl > 0

    @property
    def price_above_buy_price(self) -> bool:
        """Check if price is above entry relative to position direction."""
        if self.is_long:
            return self.close > self.area_buy_price  # Still use area_buy_price here
        return self.close < self.area_buy_price  # Still use area_buy_price here

class PositionMetrics:
    """Manages position metrics over time."""
    def __init__(self, is_long: bool, initial_balance: float):
        self.snapshots: List[PositionSnapshot] = []
        self.initial_balance = initial_balance
        self.is_long = is_long
        
        # Running extremes
        self.area_width_min: float = float('inf')
        self.area_width_max: float = float('-inf')
        
        # Price difference tracking
        self.best_price_diff_body: float = float('-inf')
        self.best_price_diff_wick: float = float('-inf')
        self.worst_price_diff_body: float = float('inf')
        self.worst_price_diff_wick: float = float('inf')
        self.net_price_diff_body: float = 0.0  # Will be calculated in first snapshot
        
        # P&L tracking
        self.max_pl_body: float = float('-inf')
        self.max_pl_wick: float = float('-inf')
        self.max_pl_minute_body: Optional[int] = None
        self.max_pl_minute_wick: Optional[int] = None
            
    def add_snapshot(self, snapshot: PositionSnapshot):
        """Add a new snapshot and update running metrics."""
        self.snapshots.append(snapshot)
        minute = len(self.snapshots)
        
        # Update area width extremes
        self.area_width_min = min(self.area_width_min, snapshot.area_width)
        self.area_width_max = max(self.area_width_max, snapshot.area_width)
        
        # Calculate price differences based on position direction
        if self.is_long:
            # For longs: positive diff when price increases
            price_diff_body = snapshot.close - snapshot.avg_entry_price
            price_diff_wick_high = snapshot.high - snapshot.avg_entry_price  # Best for longs
            price_diff_wick_low = snapshot.low - snapshot.avg_entry_price   # Worst for longs
        else:
            # For shorts: positive diff when price decreases
            price_diff_body = snapshot.avg_entry_price - snapshot.close
            price_diff_wick_high = snapshot.avg_entry_price - snapshot.low  # Best for shorts
            price_diff_wick_low = snapshot.avg_entry_price - snapshot.high  # Worst for shorts
                
        # Skip first bar for extremes tracking
        if len(self.snapshots) > 1:
            # Update best/worst differences
            self.best_price_diff_body = max(self.best_price_diff_body, price_diff_body)
            self.best_price_diff_wick = max(self.best_price_diff_wick, price_diff_wick_high)
            self.worst_price_diff_body = min(self.worst_price_diff_body, price_diff_body)
            self.worst_price_diff_wick = min(self.worst_price_diff_wick, price_diff_wick_low)
        
        # Calculate net price diff (compare to first snapshot's price)
        if len(self.snapshots) == 1:  # First snapshot
            self.net_price_diff_body = 0.0
        else:
            first_close = self.snapshots[0].close
            if self.is_long:
                self.net_price_diff_body = snapshot.close - first_close
            else:
                self.net_price_diff_body = first_close - snapshot.close
                
        # Update P&L peaks
        if snapshot.running_pl > self.max_pl_body:
            self.max_pl_body = snapshot.running_pl
            self.max_pl_minute_body = minute
                
        # Calculate theoretical P&L at wick prices
        wick_price = snapshot.high if self.is_long else snapshot.low
        if self.is_long:
            wick_pl = snapshot.realized_pl + (wick_price - snapshot.avg_entry_price) * snapshot.shares - snapshot.total_fees
        else:
            wick_pl = snapshot.realized_pl + (snapshot.avg_entry_price - wick_price) * snapshot.shares - snapshot.total_fees
            
        if wick_pl > self.max_pl_wick:
            self.max_pl_wick = wick_pl
            self.max_pl_minute_wick = minute
                
            
    @property
    def minutes_above_buy_price(self) -> int:
        """Count minutes price was above entry level."""
        return sum(1 for s in self.snapshots if s.price_above_buy_price)
        
    @property
    def minutes_profitable(self) -> int:
        """Count minutes position had positive P&L."""
        return sum(1 for s in self.snapshots if s.is_profitable)

    @property
    def holding_time(self) -> int:
        """Total number of minutes position was held."""
        return len(self.snapshots)
        
    def get_metrics_dict(self) -> dict:
        """Return all metrics as a dictionary for easy export."""
        return {
            'holding_time': self.holding_time,
            
            'max_pl_body': self.max_pl_body,
            'max_pl_wick': self.max_pl_wick,
            'max_pl_minute_body': self.max_pl_minute_body,
            'max_pl_minute_wick': self.max_pl_minute_wick,
            
            'minutes_above_buy_price': self.minutes_above_buy_price,
            'minutes_profitable': self.minutes_profitable,
            'minutes_profitable_pct': (self.minutes_profitable / self.holding_time * 100) if self.holding_time > 0 else 0,
            
            
            'net_price_diff_body': self.net_price_diff_body,
            'best_price_diff_body': self.best_price_diff_body,
            'best_price_diff_wick': self.best_price_diff_wick,
            'worst_price_diff_body': self.worst_price_diff_body,
            'worst_price_diff_wick': self.worst_price_diff_wick,
            
            'area_width_min': self.area_width_min,
            'area_width_max': self.area_width_max,
            
        }