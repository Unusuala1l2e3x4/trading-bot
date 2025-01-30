from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
import numpy as np
from trading_bot.TypedBarData import TypedBarData


@dataclass
class PositionSnapshot:
    """Records position metrics at a specific minute."""
    timestamp: datetime
    is_long: bool
    
    bar: TypedBarData

    # Position sizing at this minute
    shares: int
    prev_shares: int
    max_shares: int
    max_target_shares_limit: int
    
    # Area metrics  
    area_width: float
    area_buy_price: float  # Original trigger price from area
    avg_entry_price: float  # Current average entry price
    
    # P&L metrics at this minute
    running_pl: float  # Includes realized, unrealized, and fees
    cost_basis_sold_accum: float  # Denominator for ROI calculations
    running_plpc: float  # based on cost_basis_sold_accum
    realized_pl: float
    unrealized_pl: float
    total_fees: float
    
    has_entered: bool
    has_exited: bool

    @property
    def is_profitable(self) -> bool:
        """Check if position is pospl at this snapshot."""
        return self.running_pl > 0 and self.prev_shares > 0

    @property
    def body_above_buy_price(self) -> bool:
        """Check if close price is above entry relative to position direction."""
        if self.is_long:
            return self.bar.close > self.area_buy_price  # Still use area_buy_price here
        return self.bar.close < self.area_buy_price  # Still use area_buy_price here
    
    @property
    def wick_above_buy_price(self) -> bool:
        """Check if wick price is above entry relative to position direction."""
        if self.is_long:
            return self.bar.high > self.area_buy_price  # Still use area_buy_price here
        return self.bar.low < self.area_buy_price  # Still use area_buy_price here
    

class PositionMetrics:
    """Manages position metrics over time."""
    def __init__(self, is_long: bool, prior_relevant_bars: List[TypedBarData] = []):
        self.snapshots: List[PositionSnapshot] = []
        self.is_long = is_long
        
        self.prior_relevant_bars: List[TypedBarData] = prior_relevant_bars
        
        # Running extremes
        self.area_width_min: float = float('inf')
        self.area_width_max: float = float('-inf')
        
        # Best/worst price differences and their timing
        self.best_price_diff_body: float = float('-inf')
        self.best_price_diff_wick: float = float('-inf')
        self.worst_price_diff_body: float = float('inf')
        self.worst_price_diff_wick: float = float('inf')
        
        self.best_price_diff_body_time: Optional[int] = None
        self.best_price_diff_wick_time: Optional[int] = None
        self.worst_price_diff_body_time: Optional[int] = None
        self.worst_price_diff_wick_time: Optional[int] = None
        
        self.net_price_diff_body: float = 0.0  # Will be calculated in first snapshot
        
        # P&L extremes and their timing
        self.max_pl_body: float = float('-inf')
        self.max_pl_wick: float = float('-inf')
        self.min_pl_body: float = float('inf')
        self.min_pl_wick: float = float('inf')

        self.max_plpc_body: float = float('-inf')
        self.max_plpc_wick: float = float('-inf')
        self.min_plpc_body: float = float('inf')
        self.min_plpc_wick: float = float('inf')
        
        self.max_pl_body_time: Optional[int] = None
        self.max_pl_wick_time: Optional[int] = None
        self.min_pl_body_time: Optional[int] = None
        self.min_pl_wick_time: Optional[int] = None
        
        self.entry_snapshot_index: Optional[int] = None  # Index of first entered snapshot
        self.exit_snapshot_index: Optional[int] = None
        self.first_pospl_time: Optional[int] = None
        self.first_negpl_time: Optional[int] = None
        self.last_pospl_time: Optional[int] = None
        self.last_negpl_time: Optional[int] = None
        
        self.first_pospl_to_negpl_time: Optional[int] = None
        self.last_pospl_to_negpl_time: Optional[int] = None
        
        # Position sizing peaks
        self.max_shares_reached: int = 0
        self.max_shares_reached_pct: float = 0.0
        self.max_shares_reached_time: Optional[int] = None
        
        
        # VWAP tracking from entry
        self.cumulative_volume_since_entry: float = 0.0
        self.cumulative_vwap_volume_since_entry: float = 0.0
        self.position_vwap: Optional[float] = None
        self.best_vwap_dist: float = float('-inf')
        self.worst_vwap_dist: float = float('inf')
        self.best_vwap_dist_time: Optional[int] = None
        self.worst_vwap_dist_time: Optional[int] = None
        self.vwap_dist_list: List[float] = []
        
        

    def finalize_metrics(self):
        """
        Sets any stored numeric attributes to 0 if they are nan or inf.
        """
        for attr_name, value in vars(self).items():
            # Skip non-numeric attributes
            if not isinstance(value, (int, float)):
                continue
            # Check if value is nan or inf
            if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
                setattr(self, attr_name, 0.0)
            
    @property
    def reference_area_width(self) -> float:
        """Get R unit based on position state."""
        if not self.snapshots:
            return float('inf')
        if self.entry_snapshot_index is not None:
            # Use width at entry for entered positions
            return self.snapshots[self.entry_snapshot_index].area_width
        else:
            # # For unentered positions, use width at best price
            # best_idx = self._get_best_price_index()
            # return self.snapshots[best_idx].area_width if best_idx is not None else float('inf')
        
            # For unentered positions, use first width
            return self.snapshots[0].area_width
        
        # return self.snapshots[0].area_width
            
    @property 
    def reference_price(self) -> float:
        """Get reference price based on position state."""
        if not self.snapshots:
            return float('inf')
        # if self.entry_snapshot_index is not None:
        #     # Use area_buy_price at entry for entered positions
        #     return self.snapshots[self.entry_snapshot_index].area_buy_price
        # else:
        #     # # For unentered positions, use area_buy_price at best price
        #     # best_idx = self._get_best_price_index()
        #     # return self.snapshots[best_idx].area_buy_price if best_idx is not None else 0.0
            
        #     # For unentered positions, use first area_buy_price
        #     return self.snapshots[0].area_buy_price
        
        # return self.snapshots[0].area_buy_price
        
        if self.entry_snapshot_index is not None:
            # Use area_buy_price at entry for entered positions
            return self.snapshots[self.entry_snapshot_index].bar.close
        else:
            # # For unentered positions, use area_buy_price at best price
            # best_idx = self._get_best_price_index()
            # return self.snapshots[best_idx].area_buy_price if best_idx is not None else 0.0
            
            # For unentered positions, use first area_buy_price
            return self.snapshots[0].bar.close

    @property 
    def reference_price_norm(self) -> float:
        return self.reference_price/100

        
    def add_snapshot(self, snapshot: PositionSnapshot, best_wick_pl: float, worst_wick_pl: float, best_wick_plpc: float, worst_wick_plpc: float):
        """Add a new snapshot and update running metrics.

        Calculate theoretical exit at best price.
        
        Args:
            snapshot: Current minute's position data  
            wick_pl: Expected P&L if exited at best price (including prev P&L)
            wick_plpc: Expected P&L% if exited at best price (including prev P&L)
        """
        self.snapshots.append(snapshot)
        minute = self.num_snapshots - 1
            
        # Track first entry
        if snapshot.has_entered and self.entry_snapshot_index is None:
            self.entry_snapshot_index = self.num_snapshots - 1
            
        if snapshot.has_exited and self.exit_snapshot_index is None:
            self.exit_snapshot_index = self.num_snapshots - 1
            
        # Track first pospl minute
        if self.entry_snapshot_index is not None and (self.exit_snapshot_index is None or self.exit_snapshot_index == minute):
            if self.exit_snapshot_index == minute:
                assert snapshot.prev_shares > 0
            held_time = minute - self.entry_snapshot_index
            if snapshot.is_profitable:
                assert held_time > 0
                if self.first_pospl_time is None:
                    self.first_pospl_time = held_time
                self.last_pospl_time = held_time
            elif held_time > 0:  # Only record negative after entry
                if self.first_negpl_time is None:
                    self.first_negpl_time = held_time
                self.last_negpl_time = held_time
                    
            self.cumulative_volume_since_entry += snapshot.bar.volume
            self.cumulative_vwap_volume_since_entry += snapshot.bar.vwap * snapshot.bar.volume
            self.position_vwap = (self.cumulative_vwap_volume_since_entry / 
                                self.cumulative_volume_since_entry if self.cumulative_volume_since_entry > 0 else None)
            
            if self.position_vwap is not None:
                # Calculate price differences vs VWAP similar to other metrics
                if self.is_long:
                    vwap_dist = snapshot.bar.close - self.position_vwap
                else:
                    vwap_dist = self.position_vwap - snapshot.bar.close
                    
                self.vwap_dist_list.append(vwap_dist)
                    
                if vwap_dist > self.best_vwap_dist:
                    self.best_vwap_dist = vwap_dist
                    self.best_vwap_dist_time = minute
                if vwap_dist < self.worst_vwap_dist:
                    self.worst_vwap_dist = vwap_dist
                    self.worst_vwap_dist_time = minute
                
                
        # Update share peaks (do this for all snapshots)
        if snapshot.shares > self.max_shares_reached:
            self.max_shares_reached = snapshot.shares
            self.max_shares_reached_pct = round(100*snapshot.shares / snapshot.max_shares, 2)
            self.max_shares_reached_time = minute

        
        # Update area width extremes - do this for all snapshots
        self.area_width_min = min(self.area_width_min, snapshot.area_width)
        self.area_width_max = max(self.area_width_max, snapshot.area_width)
        
        # Skip remaining calculations for first bar
        if self.num_snapshots == 1:
            return
        
        # Get reference price for this snapshot's updates
        ref_price = self.reference_price
            
        # Calculate price differences based on position direction
        if self.is_long:
            price_diff_body = snapshot.bar.close - ref_price
            price_diff_wick_high = snapshot.bar.high - ref_price
            price_diff_wick_low = snapshot.bar.low - ref_price
        else:
            price_diff_body = ref_price - snapshot.bar.close
            price_diff_wick_high = ref_price - snapshot.bar.low  # Best for shorts 
            price_diff_wick_low = ref_price - snapshot.bar.high  # Worst for shorts
                
        # Update best/worst price differences with timing
        if price_diff_body > self.best_price_diff_body:
            self.best_price_diff_body = price_diff_body
            self.best_price_diff_body_time = minute
        if price_diff_body < self.worst_price_diff_body:
            self.worst_price_diff_body = price_diff_body
            self.worst_price_diff_body_time = minute
            
        if price_diff_wick_high > self.best_price_diff_wick:
            self.best_price_diff_wick = price_diff_wick_high
            self.best_price_diff_wick_time = minute
        if price_diff_wick_low < self.worst_price_diff_wick:
            self.worst_price_diff_wick = price_diff_wick_low
            self.worst_price_diff_wick_time = minute
        
        # Update net price diff
        self.net_price_diff_body = price_diff_body
                    
        # Only update P&L peaks if we had trades
        if snapshot.avg_entry_price is not None:
            # Body metrics based on actual closing price
            if snapshot.running_pl > self.max_pl_body:
                self.max_pl_body = snapshot.running_pl
                self.max_pl_body_time = minute
            if snapshot.running_pl < self.min_pl_body:
                self.min_pl_body = snapshot.running_pl
                self.min_pl_body_time = minute
                
            # Percentage metrics
            if snapshot.running_plpc > self.max_plpc_body:
                self.max_plpc_body = snapshot.running_plpc
                self.max_plpc_body_time = minute
            if snapshot.running_plpc < self.min_plpc_body:
                self.min_plpc_body = snapshot.running_plpc
                self.min_plpc_body_time = minute
                
            # Wick metrics using both best and worst cases
            if best_wick_pl > self.max_pl_wick:
                self.max_pl_wick = best_wick_pl
                self.max_pl_wick_time = minute
            if worst_wick_pl < self.min_pl_wick:
                self.min_pl_wick = worst_wick_pl
                self.min_pl_wick_time = minute
                
            if best_wick_plpc > self.max_plpc_wick:
                self.max_plpc_wick = best_wick_plpc
                self.max_plpc_wick_time = minute
            if worst_wick_plpc < self.min_plpc_wick:
                self.min_plpc_wick = worst_wick_plpc
                self.min_plpc_wick_time = minute

            
            
            # print(f"{self.num_snapshots} {self.holding_time} {snapshot.avg_entry_price}")
            
            # assert self.min_pl_wick <= self.min_pl_body <= self.max_pl_body <= self.max_pl_wick, \
            #     f"{self.min_pl_wick} <= {self.min_pl_body} <= {self.max_pl_body} <= {self.max_pl_wick}"
            # assert self.min_plpc_wick <= self.min_plpc_body <= self.max_plpc_body <= self.max_plpc_wick, \
            #     f"{self.min_plpc_wick} <= {self.min_plpc_body} <= {self.max_plpc_body} <= {self.max_plpc_wick}"
            # NOTE: dont assert; treat wick metrics as approximations
                    
    # @property
    # def has_entered_time(self) -> datetime:
    #     return next((a.timestamp for a in self.snapshots if a.has_entered), datetime.min)
    
    @property
    def body_above_buy_price_time(self) -> int:
        """Count minutes price was above entry level."""
        return sum(1 for s in self.snapshots if s.body_above_buy_price)
    
    @property
    def wick_above_buy_price_time(self) -> int:
        """Count minutes price was above entry level."""
        return sum(1 for s in self.snapshots if s.wick_above_buy_price)
    
    @property
    def body_above_buy_price_time_pct(self) -> int:
        """Count minutes price was above entry level."""
        return (self.body_above_buy_price_time / self.num_snapshots * 100) if self.holding_time > 0 else 0
    
    @property
    def wick_above_buy_price_time_pct(self) -> int:
        """Count minutes price was above entry level."""
        return (self.wick_above_buy_price_time / self.num_snapshots * 100) if self.holding_time > 0 else 0
    
    @property
    def wick_above_buy_price_time_pct(self) -> int:
        """Count minutes price was above entry level."""
        return (self.wick_above_buy_price_time / self.num_snapshots * 100) if self.holding_time > 0 else 0
    
    @property
    def profitable_time(self) -> int:
        """Count minutes position had positive P&L."""
        return sum(1 for s in self.snapshots if s.is_profitable)
    
    @property
    def profitable_time_pct(self) -> int:
        """Count minutes position had positive P&L."""
        return (self.profitable_time / self.holding_time * 100) if self.holding_time > 0 else 0

    @property
    def holding_time(self) -> int:
        """Total number of minutes position was held."""
        return sum(1 for a in self.snapshots if a.shares > 0)
    
    @property
    def num_snapshots(self) -> int:
        """Total number of minutes position was held."""
        return len(self.snapshots)
    
    @property
    def avg_entry_price_diff(self) -> float:
        avg_entry_price_first = next((a.avg_entry_price for a in self.snapshots if a.avg_entry_price is not None), None)
        if avg_entry_price_first:
            avg_entry_price_last = next((a.avg_entry_price for a in reversed(self.snapshots) if a.avg_entry_price is not None), None)
            return avg_entry_price_last - avg_entry_price_first if self.is_long else avg_entry_price_first - avg_entry_price_last
        return 0.0
    
    @property
    def avg_area_width(self) -> float:
        return np.mean([a.area_width for a in self.snapshots])
    
    @property
    def avg_central_value_dist(self) -> float:
        ret = np.mean([a.bar.central_value_dist for a in self.snapshots])
        return ret if self.is_long else -ret
        
    @property
    def avg_prior_central_value_dist(self) -> float:
        ret = np.mean([a.central_value_dist for a in self.prior_relevant_bars])
        return ret if self.is_long else -ret
        
    @property
    def avg_vwap_dist(self) -> float:
        if not self.vwap_dist_list:
            return 0.0
        return np.mean(self.vwap_dist_list)
    
    
    def normalize_by_r(self, value: float) -> float:
        """Normalize value by R-unit (area_width)."""
        r_unit = self.reference_area_width
        return value / r_unit if r_unit else 0.0
        
    def normalize_by_price(self, value: float) -> float:
        """Normalize value as percentage of reference price."""
        ref_price = self.reference_price
        return (value / ref_price * 100) if ref_price else 0.0
    
    
    def get_metrics_dict(self) -> dict:
        """Return all metrics as a dictionary for easy export."""
        r_unit = self.reference_area_width
                
        metrics = {
            # Time-based metrics
            # 'total_time': self.num_snapshots,
            'holding_time': self.holding_time,
            'profitable_time': self.profitable_time,
            'profitable_time_pct': self.profitable_time_pct,
            
            'first_negpl_time': self.first_negpl_time,
            'first_pospl_time': self.first_pospl_time,
            'last_negpl_time': self.last_negpl_time,
            'last_pospl_time': self.last_pospl_time,
            
            # Position sizing metrics
            # 'max_shares_reached': self.max_shares_reached,
            'max_shares_reached_pct': self.max_shares_reached_pct,
            'max_shares_reached_time': self.max_shares_reached_time,
            
            'body_above_buy_price_time': self.body_above_buy_price_time,
            'wick_above_buy_price_time': self.wick_above_buy_price_time,
            # 'body_above_buy_price_time_pct': self.body_above_buy_price_time_pct,
            # 'wick_above_buy_price_time_pct': self.wick_above_buy_price_time_pct,
            
            
        
            # P&L metrics
            'max_pl_body': self.max_pl_body,
            'min_pl_body': self.min_pl_body,
            'max_pl_wick': self.max_pl_wick,
            'min_pl_wick': self.min_pl_wick,
            
            'max_plpc_body': self.max_plpc_body,
            'min_plpc_body': self.min_plpc_body,
            'max_plpc_wick': self.max_plpc_wick,
            'min_plpc_wick': self.min_plpc_wick,
            
            # P&L timing metrics
            'max_pl_body_time': self.max_pl_body_time,
            'min_pl_body_time': self.min_pl_body_time,
            'max_pl_wick_time': self.max_pl_wick_time,
            'min_pl_wick_time': self.min_pl_wick_time,
            

            
                
            # # Raw VWAP metrics
            # 'best_vwap_dist': self.best_vwap_dist,
            # 'worst_vwap_dist': self.worst_vwap_dist,
            # Normalized VWAP metrics
            'avg_vwap_dist_P': self.normalize_by_price(self.avg_vwap_dist),
            'best_vwap_dist_P': self.normalize_by_price(self.best_vwap_dist),
            'worst_vwap_dist_P': self.normalize_by_price(self.worst_vwap_dist),
            
            'best_vwap_dist_time': self.best_vwap_dist_time,
            'worst_vwap_dist_time': self.worst_vwap_dist_time,
            

                
                
            # # Raw price differences
            # 'avg_entry_price_diff': self.avg_entry_price_diff,
            # 'avg_central_value_dist': self.avg_central_value_dist,
            # 'net_price_diff_body': self.net_price_diff_body,
            # 'best_price_diff_body': self.best_price_diff_body,
            # 'worst_price_diff_body': self.worst_price_diff_body,
            # 'best_price_diff_wick': self.best_price_diff_wick,
            # 'worst_price_diff_wick': self.worst_price_diff_wick,
            
            
            # Normalized price differences (R-multiples)
            'avg_entry_price_diff_P': self.normalize_by_price(self.avg_entry_price_diff),
            'avg_central_value_dist_P': self.normalize_by_price(self.avg_central_value_dist),
            'avg_prior_central_value_dist_P': self.normalize_by_price(self.avg_prior_central_value_dist),
            
            'net_price_diff_body_P': self.normalize_by_price(self.net_price_diff_body),
            'best_price_diff_body_P': self.normalize_by_price(self.best_price_diff_body),
            'worst_price_diff_body_P': self.normalize_by_price(self.worst_price_diff_body),
            'best_price_diff_wick_P': self.normalize_by_price(self.best_price_diff_wick),
            'worst_price_diff_wick_P': self.normalize_by_price(self.worst_price_diff_wick),
            
            # Price movement timing
            'best_price_diff_body_time': self.best_price_diff_body_time,
            'worst_price_diff_body_time': self.worst_price_diff_body_time,
            'best_price_diff_wick_time': self.best_price_diff_wick_time,
            'worst_price_diff_wick_time': self.worst_price_diff_wick_time,
            
            # Area characteristics
            # 'area_width_min': self.area_width_min,
            # 'area_width_max': self.area_width_max,
            'ref_price': self.reference_price,
            'ref_area_width': self.reference_area_width,
            'avg_area_width': self.avg_area_width

        }
        
        
        return metrics
        