from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Union, Tuple, Dict
import numpy as np
from trading_bot.TypedBarData import TypedBarData, AnchoredVWAPMetrics
from bisect import bisect_right





@dataclass
class BreakoutCharacteristics:
    type: str  # 'Aggressive', 'Passive', 'Indecisive', 'Lethargic'
    confidence: float  # 0.0-1.0
    vwap_metrics: Dict[str, float]  # Key metrics that led to classification
    volume_metrics: Dict[str, float]  # Volume analysis metrics



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
    is_entry_snapshot: bool = False
    is_exit_snapshot: bool = False
    
    position_vwap_dist: Optional[float] = np.nan
    position_vwap: Optional[float] = np.nan
    position_vwap_std: Optional[float] = np.nan
    
    vwap_std_close: Optional[float] = np.nan
    vwap_std_high: Optional[float] = np.nan
    vwap_std_low: Optional[float] = np.nan


    @property
    def is_before_or_at_exit(self) -> bool:
        return not self.has_exited or self.is_exit_snapshot

    @property
    def is_profitable(self) -> bool:
        """Check if position is pospl at this snapshot."""
        return self.running_pl > 0 and self.is_before_or_at_exit

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
    
    def running_plpc_with_accum(self, cost_basis_sold_accum) -> float:
        if cost_basis_sold_accum <= 0:
            return 0.0
        return (self.running_pl / cost_basis_sold_accum) * 100

class PositionMetrics:
    """Manages position metrics over time."""
    def __init__(self, is_long: bool, norm_strategy: str = 'price', prior_relevant_bars: List[TypedBarData] = []):
        self.snapshots: List[PositionSnapshot] = []
        self.is_long = is_long
        self.norm_strategy = norm_strategy # 'r','price', or neither for no normalization
        
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
        self.best_vwap_std_close: float = float('-inf')
        self.worst_vwap_std_close: float = float('inf')
        self.best_vwap_std_close_time: Optional[int] = None
        self.worst_vwap_std_close_time: Optional[int] = None


    @property
    def snapshot_at_entry(self) -> PositionSnapshot | None:
        if self.entry_snapshot_index is not None:
            return self.snapshots[self.entry_snapshot_index]
    @property
    def snapshot_at_exit(self) -> PositionSnapshot | None:
        if self.exit_snapshot_index is not None:
            return self.snapshots[self.exit_snapshot_index]
        

    def finalize_metrics(self):
        """
        Sets any stored numeric attributes to 0 if they are nan or inf.
        """
        for attr_name, value in vars(self).items():
            # Skip non-numeric attributes
            if not isinstance(value, (int, float)):
                continue
            # Check if value is nan or inf
            if isinstance(value, float) and not np.isfinite(value):
                setattr(self, attr_name, np.nan)
            
    @property
    def reference_area_width(self) -> float:
        """Get R unit based on position state."""
        if not self.snapshots:
            return float('inf')
        # return self.snapshots[0].area_width
        if self.entry_snapshot_index is not None:
            # Use width at entry for entered positions
            return self.snapshots[self.entry_snapshot_index].area_width
        else:
            # # For unentered positions, use width at best price
            # best_idx = self._get_best_price_index()
            # return self.snapshots[best_idx].area_width if best_idx is not None else float('inf')
        
            # For unentered positions, use first width
            return self.snapshots[0].area_width
        

            
    @property 
    def reference_price(self) -> float:
        """Get reference price based on position state."""
        if not self.snapshots:
            return float('inf')
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
    
    
        
    @property
    def avg_reference_area_width(self) -> float:
        """Get average area width between entry and exit."""
        if not self.snapshots:
            return float('inf')
            
        if self.entry_snapshot_index is None:
            # Not entered - use all snapshots
            snapshots = self.snapshots
        else:
            # Get snapshots from entry to exit (inclusive)
            end_idx = self.exit_snapshot_index + 1 if self.exit_snapshot_index is not None else None
            snapshots = self.snapshots[self.entry_snapshot_index:end_idx]
            
        return np.mean([s.area_width for s in snapshots])

    @property 
    def avg_reference_price(self) -> float:
        """Get average close price between entry and exit."""
        if not self.snapshots:
            return float('inf')
            
        if self.entry_snapshot_index is None:
            # Not entered - use all snapshots
            snapshots = self.snapshots
        else:
            # Get snapshots from entry to exit (inclusive)
            end_idx = self.exit_snapshot_index + 1 if self.exit_snapshot_index is not None else None
            snapshots = self.snapshots[self.entry_snapshot_index:end_idx]
            
        return np.mean([s.bar.close for s in snapshots])

    def normalize_relative_metrics(self, value: float) -> float:
        """
        Normalize metrics that depend on bar-by-bar values.
        Used for central_value_dist and vwap_dist.
        """
        if self.norm_strategy == 'r':
            return value / self.avg_reference_area_width if np.isfinite(value) else np.nan
        elif self.norm_strategy == 'price':
            return (value / self.avg_reference_price * 100) if np.isfinite(value) else np.nan
        return value
        
    

        
    def add_snapshot(self, snapshot: PositionSnapshot, best_wick_pl: float, worst_wick_pl: float, best_wick_plpc: float, worst_wick_plpc: float):
        """Add a new snapshot and update running metrics.

        Calculate theoretical exit at best price.
        
        Args:
            snapshot: Current minute's position data  
            wick_pl: Expected P&L if exited at best price (including prev P&L)
            wick_plpc: Expected P&L% if exited at best price (including prev P&L)
        """
        minute = self.num_snapshots # +1
        # Track first entry
        if snapshot.has_entered and self.entry_snapshot_index is None:
            self.entry_snapshot_index = minute
            snapshot.is_entry_snapshot = True
            assert snapshot.running_pl == 0, snapshot.running_pl
            assert snapshot.running_plpc == 0, snapshot.running_plpc
            
        if snapshot.has_exited and self.exit_snapshot_index is None:
            self.exit_snapshot_index = minute
            snapshot.is_exit_snapshot = True
        
        # NOTE: flags should all be set above    
        if snapshot.has_entered and snapshot.is_before_or_at_exit:
            # Calculate high/low std values BEFORE appending current snapshot
            snapshot.vwap_std_high, snapshot.vwap_std_low = self.calculate_std_value(snapshot.bar.high, snapshot.bar.low)
        
        self.snapshots.append(snapshot)
        assert minute == self.num_snapshots - 1
            
        # Track first pospl minute
        if snapshot.has_entered and snapshot.is_before_or_at_exit:
            if snapshot.is_exit_snapshot:
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
            snapshot.position_vwap = (self.cumulative_vwap_volume_since_entry / 
                                self.cumulative_volume_since_entry if self.cumulative_volume_since_entry > 0 else np.nan)
            
            snapshot.position_vwap_std = self.vwap_std
            
            # Calculate price differences vs VWAP similar to other metrics
            assert np.isfinite(snapshot.position_vwap), snapshot.position_vwap
            
            # Calculate close std AFTER appending (since we want current bar included)
            snapshot.vwap_std_close = self.calculate_std_value(snapshot.bar.close)

            if self.is_long:
                vwap_dist = snapshot.bar.close - snapshot.position_vwap
            else:
                vwap_dist = snapshot.position_vwap - snapshot.bar.close
            snapshot.position_vwap_dist = vwap_dist
            
            # Using the std value (already calculated earlier)
            if snapshot.vwap_std_close > self.best_vwap_std_close:  # rename these fields too
                self.best_vwap_std_close = snapshot.vwap_std_close
                self.best_vwap_std_close_time = minute
            if snapshot.vwap_std_close < self.worst_vwap_std_close:
                self.worst_vwap_std_close = snapshot.vwap_std_close
                self.worst_vwap_std_close_time = minute
                

        # Update share peaks (do this for all snapshots)
        if snapshot.shares > self.max_shares_reached:
            assert snapshot.has_entered
            self.max_shares_reached = snapshot.shares
            self.max_shares_reached_pct = round(100*snapshot.shares / snapshot.max_shares, 2)
            self.max_shares_reached_time = minute
            
        # record area width peaks anytime <= exit
        if snapshot.is_before_or_at_exit:
            # Update area width extremes - do this for all snapshots
            self.area_width_min = min(self.area_width_min, snapshot.area_width)
            self.area_width_max = max(self.area_width_max, snapshot.area_width)
            
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
            
            
                    
        # Only update P&L peaks after entry, not at entry
        if snapshot.has_entered and snapshot.is_before_or_at_exit:
            assert snapshot.avg_entry_price is not None
            
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
                # print(snapshot.timestamp,'> max:', minute, self.max_plpc_body)
                
                
            if snapshot.running_plpc < self.min_plpc_body:
                self.min_plpc_body = snapshot.running_plpc
                self.min_plpc_body_time = minute
                # print(snapshot.timestamp,'< min:', minute, self.min_plpc_body)
                
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
                    

    @property
    def num_snapshots(self) -> int:
        """Total number of minutes position was held."""
        return len(self.snapshots)
    
    @property
    def snapshots_before_or_at_exit(self) -> List[PositionSnapshot]:
        return [a for a in self.snapshots if a.is_before_or_at_exit]
    
    @property
    def num_snapshots_before_or_at_exit(self) -> int:
        """Total number of minutes position was held."""
        return len(self.snapshots_before_or_at_exit)

    @property
    def profitable_time(self) -> int:
        """Count minutes position had positive P&L."""
        return sum(1 for s in self.snapshots if s.is_profitable)
    
    @property
    def holding_time(self) -> int:
        """Total number of minutes position was held."""
        return sum(1 for a in self.snapshots if a.shares > 0)
    
    @property
    def profitable_time_pct(self) -> int:
        """Count minutes position had positive P&L."""
        assert self.profitable_time <= self.holding_time, f'{self.profitable_time} <= {self.holding_time}'
        return (self.profitable_time / self.holding_time * 100) if self.holding_time > 0 else 0
    
    @property
    def body_above_buy_price_time(self) -> int:
        """Count minutes price was above entry level."""
        return sum(1 for s in self.snapshots_before_or_at_exit if s.body_above_buy_price)
    
    @property
    def wick_above_buy_price_time(self) -> int:
        """Count minutes price was above entry level."""
        return sum(1 for s in self.snapshots_before_or_at_exit if s.wick_above_buy_price)
    
    @property
    def body_above_buy_price_time_pct(self) -> int:
        """Count minutes price was above entry level."""
        return (self.body_above_buy_price_time / self.num_snapshots_before_or_at_exit * 100) if self.num_snapshots_before_or_at_exit > 0 else 0
    
    @property
    def wick_above_buy_price_time_pct(self) -> int:
        """Count minutes price was above entry level."""
        return (self.wick_above_buy_price_time / self.num_snapshots_before_or_at_exit * 100) if self.num_snapshots_before_or_at_exit > 0 else 0
    
    @property
    def avg_entry_price_diff(self) -> float:
        avg_entry_price_first = next((a.avg_entry_price for a in self.snapshots_before_or_at_exit if a.avg_entry_price is not None), None)
        if avg_entry_price_first:
            avg_entry_price_last = next((a.avg_entry_price for a in reversed(self.snapshots_before_or_at_exit) if a.avg_entry_price is not None), None)
            return avg_entry_price_last - avg_entry_price_first if self.is_long else avg_entry_price_first - avg_entry_price_last
        return 0.0
    
    @property
    def avg_area_width(self) -> float:
        return np.mean([a.area_width for a in self.snapshots_before_or_at_exit])
    
    @property
    def avg_central_value_dist(self) -> float:
        ret = np.mean([a.bar.central_value_dist for a in self.snapshots_before_or_at_exit])
        return ret if self.is_long else -ret
        
    @property
    def avg_prior_central_value_dist(self) -> float:
        ret = np.mean([a.central_value_dist for a in self.prior_relevant_bars])
        return ret if self.is_long else -ret
    
    
    # avwap-based metrics and functions:
        
    @property
    def avg_vwap_dist(self) -> float:
        return np.mean([a.position_vwap_dist for a in self.snapshots if np.isfinite(a.position_vwap_dist)])
    
    @property
    def avg_vwap_std(self) -> float:
        return np.mean([a.position_vwap_std for a in self.snapshots if np.isfinite(a.position_vwap_std)])

    @property
    def avg_vwap_std_close(self) -> float:
        return np.mean([a.vwap_std_close for a in self.snapshots if np.isfinite(a.vwap_std_close)])
    
    @property
    def vwap_std(self) -> float:
        """Calculate volume-weighted standard deviation of bar VWAPs from position VWAP."""
        if not self.snapshots or len(self.snapshots) == 1 or self.entry_snapshot_index is None:
            return np.nan

        snapshots = self.snapshots[self.entry_snapshot_index:]
        total_volume = sum(s.bar.volume for s in snapshots)
        if total_volume == 0:
            return np.nan
            
        # Calculate volume-weighted squared deviations
        squared_devs = sum(
            s.bar.volume * (s.bar.vwap - s.position_vwap)**2 
            for s in snapshots
        )
        return np.sqrt(squared_devs / total_volume)

    def calculate_std_value(self, *prices: float) -> Union[float, Tuple[float, ...]]:
        """
        Calculate how many standard deviations a given price is from current VWAP.
        
        Args:
            *prices: One or more prices to calculate standard deviations for
            
        Returns:
            Single float if one price passed, tuple of floats if multiple.
            Each value represents number of standard deviations 
            (positive if above VWAP, negative if below)
            np.nan: If std_dev is too small, no snapshots available, or result exceeds threshold
        """
        if not self.snapshots or len(self.snapshots) == 1 or self.entry_snapshot_index is None:
            return tuple(np.nan for _ in prices) if len(prices) > 1 else np.nan
        
        latest = self.snapshots[-1]
        std_dev = latest.position_vwap_std  # Using stored std from snapshot
        
        MIN_STD_DEV = 1e-8  # Minimum meaningful standard deviation - Values beyond 10 standard deviations are extremely rare in normal distributions
        MAX_STD_VALUE = 10.0  # Cap at 10 standard deviations
        
        if not np.isfinite(std_dev) or std_dev < MIN_STD_DEV:
            return tuple(np.nan for _ in prices) if len(prices) > 1 else np.nan
            
        std_values = []
        for price in prices:
            std_value = (price - latest.position_vwap) / std_dev
            # Return np.nan if value is too extreme
            if abs(std_value) > MAX_STD_VALUE:
                std_values.append(np.nan)
            else:
                std_values.append(std_value)
        
        return tuple(std_values) if len(prices) > 1 else std_values[0]

    @property 
    def vwap_upper_band(self) -> float:
        """Upper VWAP band at 2 standard deviations."""
        if not self.snapshots:
            return np.nan
        latest = self.snapshots[-1]
        std_dev = self.vwap_std
        if not np.isfinite(std_dev):
            return np.nan
        return latest.position_vwap + (2 * std_dev)

    @property 
    def vwap_lower_band(self) -> float:
        """Lower VWAP band at 2 standard deviations."""
        if not self.snapshots:
            return np.nan
        latest = self.snapshots[-1]
        std_dev = self.vwap_std
        if not np.isfinite(std_dev):
            return np.nan
        return latest.position_vwap - (2 * std_dev)
    
    
            
    def get_breakout_bars(self) -> Tuple[List[TypedBarData], List[TypedBarData]]:
        """
        Get bars for breakout analysis.
        
        Returns:
            Tuple of (prior_bars, position_bars) where:
            - prior_bars: Exactly 5 bars ending at entry bar (includes the entry bar)
            - position_bars: Up to 5 bars strictly after entry
        """
        
        if not self.snapshots:
            return [], []
            
        # Get all bars in chronological order
        # self.prior_relevant_bars and [s.bar for s in self.snapshots] are consecutive and mutually exclusive
        all_bars = (
            self.prior_relevant_bars + 
            [s.bar for s in self.snapshots]
        )
        
        if self.entry_snapshot_index is None:
            # Not entered yet - calculate indices relative to current state
            # entry_idx points to latest snapshot since we haven't entered
            entry_idx = len(self.prior_relevant_bars) + len(self.snapshots) - 1
            prior_end = entry_idx + 1  # Include the latest bar
            prior_start = max(0, prior_end - 5)
            return all_bars[prior_start:prior_end], []
        
        # Calculate full index of entry bar in all_bars
        # entry_snapshot_index references the bar received before entry
        # Since prior_relevant_bars and snapshots are consecutive:
        # - If entry_snapshot_index is 0, entry bar is first snapshot
        # - If entry_snapshot_index > 0, entry bar is that many bars into snapshots
        entry_idx = len(self.prior_relevant_bars) + self.entry_snapshot_index
        
        # Get exactly 5 bars ending at entry (includes entry bar)
        prior_end = entry_idx + 1  # Add 1 to include entry bar
        prior_start = max(0, prior_end - 5)
        prior_bars = all_bars[prior_start:prior_end]
        
        # Get exactly 5 bars after entry (or all available if < 5)
        # entry_idx + 1 starts at first bar after entry
        position_bars = all_bars[entry_idx + 1:entry_idx + 6]
        
        return prior_bars, position_bars
            
        
    def calculate_breakout_metrics(self) -> Tuple[Optional[AnchoredVWAPMetrics], Optional[AnchoredVWAPMetrics]]:
        prior_bars, position_bars = self.get_breakout_bars()
        return TypedBarData.calculate_breakout_metrics(
            position_bars=position_bars,
            prior_bars=prior_bars,
            is_long=self.is_long
        )
        
        
    def analyze_volume_trend(self, bars: List[TypedBarData]) -> Dict[str, float]:
        """Analyze volume characteristics in bar sequence."""
        if not bars:
            return {
                'volume_trend': 0.0,
                'volume_consistency': 0.0,
                'cumulative_progress': 0.0
            }
            
        # Calculate volume ratios (vs average)
        volume_ratios = [bar.volume / bar.avg_volume for bar in bars]
        
        # Calculate trend (are later bars stronger?)
        volume_trend = np.polyfit(range(len(volume_ratios)), volume_ratios, 1)[0]
        
        # Measure consistency (std dev of ratios)
        volume_consistency = 1.0 - min(1.0, np.std(volume_ratios))
        
        # Calculate how volume progresses
        cumulative_volumes = np.cumsum(volume_ratios)
        expected_progress = np.linspace(0, sum(volume_ratios), len(bars))
        cumulative_progress = np.mean(cumulative_volumes >= expected_progress)
        
        return {
            'volume_trend': volume_trend,
            'volume_consistency': volume_consistency,
            'cumulative_progress': cumulative_progress
        }


        
    # def characterize_breakout(self) -> Optional[BreakoutCharacteristics]:
    #     """Analyze breakout characteristics using VWAP and volume metrics.
        
    #     Inspired by https://www.youtube.com/watch?v=FdMcPKGtFgA&ab_channel=SMBCapital
        
    #     """
    #     prior_bars, position_bars = self.get_breakout_bars()
    #     if not position_bars:
    #         return None
            
    #     # Get VWAP metrics
    #     pre_metrics, post_metrics = self.calculate_breakout_metrics()
    #     if not post_metrics:
    #         return None
            
    #     # Analyze volume trends
    #     vol_metrics = self.analyze_volume_trend(position_bars)
        
    #     # Combine key indicators
    #     price_strength = post_metrics.vwap_std_close  # How far we've moved
    #     price_conviction = post_metrics.vwap_std  # How direct the move is
    #     volume_strength = vol_metrics['volume_trend']
    #     volume_consistency = vol_metrics['volume_consistency']
        
    #     # Classification logic
    #     if (price_strength > 2.0 and  # Strong move
    #         volume_strength > 0.1 and  # Increasing volume
    #         volume_consistency > 0.7):  # Consistent buying/selling
    #         type = 'Aggressive'
    #         confidence = min(1.0, (price_strength/3 + volume_consistency)/2)
            
    #     elif (1.0 <= price_strength <= 2.0 and  # Moderate move
    #           volume_consistency > 0.8 and      # Very consistent
    #           abs(volume_strength) < 0.1):      # Steady volume
    #         type = 'Passive'
    #         confidence = volume_consistency
            
    #     elif (abs(price_strength) < 0.5 or     # Weak move
    #           volume_consistency < 0.3):        # Inconsistent
    #         type = 'Lethargic'
    #         confidence = 1.0 - max(abs(price_strength)/2, volume_consistency)
            
    #     else:
    #         type = 'Indecisive'
    #         confidence = 0.6  # Medium confidence default for mixed signals
            
    #     return BreakoutCharacteristics(
    #         type=type,
    #         confidence=confidence,
    #         vwap_metrics={
    #             'price_strength': price_strength,
    #             'price_conviction': price_conviction,
    #             'vwap_dist': post_metrics.vwap_dist
    #         },
    #         volume_metrics=vol_metrics
    #     )



    
    def normalize_by_r(self, *values: float) -> Union[float, Tuple[float, ...]]:
        """
        Normalize value(s) by R-unit (area_width).
        
        Args:
            *values: One or more float values to normalize
            
        Returns:
            Single float if one value passed, tuple of floats if multiple
        """
        r_unit = self.reference_area_width
        if r_unit == 0.0:
            return tuple(0.0 for _ in values) if len(values) > 1 else 0.0
            
        normalized = tuple(value / r_unit if np.isfinite(value) else np.nan 
                        for value in values)
        return normalized if len(values) > 1 else normalized[0]

    def normalize_by_price(self, *values: float) -> Union[float, Tuple[float, ...]]:
        """
        Normalize value(s) as percentage of reference price.
        
        Args:
            *values: One or more float values to normalize
            
        Returns:
            Single float if one value passed, tuple of floats if multiple
        """
        ref_price = self.reference_price
        if ref_price == 0.0:
            return tuple(0.0 for _ in values) if len(values) > 1 else 0.0
            
        normalized = tuple((value / ref_price * 100) if np.isfinite else np.nan 
                        for value in values)
        return normalized if len(values) > 1 else normalized[0]


    def get_metrics_dict(self) -> dict:
        """Return all metrics as a dictionary for easy export."""
        
        if self.norm_strategy == 'r':
            norm_func = self.normalize_by_r
            norm_relative = self.normalize_relative_metrics
        elif self.norm_strategy == 'price':
            norm_func = self.normalize_by_price
            norm_relative = self.normalize_relative_metrics
        else:
            norm_func = lambda x: x
            norm_relative = lambda x: x
                
        
        
            
        pre_metrics, post_metrics = self.calculate_breakout_metrics()
        
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



            # Add breakout metrics
            'breakout_pre_vwap_dist': norm_func(pre_metrics.vwap_dist), # NOTE: okay to use norm_func since it is only 5 bars
            'breakout_post_vwap_dist': norm_func(post_metrics.vwap_dist), # NOTE: okay to use norm_func since it is only 5 bars
            'breakout_pre_vwap_std': norm_func(pre_metrics.vwap_std),
            'breakout_post_vwap_std': norm_func(post_metrics.vwap_std),
            'breakout_pre_vwap_std_close': pre_metrics.vwap_std_close,
            'breakout_post_vwap_std_close': post_metrics.vwap_std_close,


            # # Raw VWAP metrics
            # 'best_vwap_std_close': self.best_vwap_std_close,
            # 'worst_vwap_std_close': self.worst_vwap_std_close,
            # Normalized VWAP metrics
            'avg_vwap_dist': norm_relative(self.avg_vwap_dist), # NOTE: normalized by average
            'avg_vwap_std': norm_func(self.avg_vwap_std),
            'avg_vwap_std_close': self.avg_vwap_std_close,
            'best_vwap_std_close': self.best_vwap_std_close,
            'worst_vwap_std_close': self.worst_vwap_std_close,
                
            'best_vwap_std_close_time': self.best_vwap_std_close_time,
            'worst_vwap_std_close_time': self.worst_vwap_std_close_time,
            

                
                
            # # Raw price differences
            # 'avg_entry_price_diff': self.avg_entry_price_diff,
            # 'avg_central_value_dist': self.avg_central_value_dist,
            # 'net_price_diff_body': self.net_price_diff_body,
            # 'best_price_diff_body': self.best_price_diff_body,
            # 'worst_price_diff_body': self.worst_price_diff_body,
            # 'best_price_diff_wick': self.best_price_diff_wick,
            # 'worst_price_diff_wick': self.worst_price_diff_wick,
            
            
            # Normalized price differences (R-multiples)
            'avg_entry_price_diff': norm_func(self.avg_entry_price_diff),
            'avg_central_value_dist': norm_relative(self.avg_central_value_dist), # NOTE: normalized by average
            'avg_prior_central_value_dist': norm_relative(self.avg_prior_central_value_dist), # NOTE: normalized by average
            
            'net_price_diff_body': norm_func(self.net_price_diff_body),
            'best_price_diff_body': norm_func(self.best_price_diff_body),
            'worst_price_diff_body': norm_func(self.worst_price_diff_body),
            'best_price_diff_wick': norm_func(self.best_price_diff_wick),
            'worst_price_diff_wick': norm_func(self.worst_price_diff_wick),
            
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
        return {k: round(v, 10) if isinstance(v, float) else v for k, v in metrics.items()}
        # return metrics
        