import pandas as pd
import numpy as np
from numba import jit

from typing import List, Tuple, Optional, Dict
import numpy as np
from datetime import datetime

import logging
import traceback

from dataclasses import dataclass, field
from typing import Optional, Callable

from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
    
    
@jit(nopython=True)
def floor_to_two_decimals(x):
    """Floor value x down to the nearest cent."""
    return np.floor(x * 100) / 100.0

@jit(nopython=True)
def ceil_to_two_decimals(x):
    """Ceil value x up to the nearest cent."""
    return np.ceil(x * 100) / 100.0




@dataclass
class VolumeProfile:
    """
    Maintains and updates volume profiles without lookahead bias.
    Each bar gets its own volume profile based only on data up to that bar.
    """
    bin_width: float = 0.01
    body_weight: float = 0.7
    wick_weight: float = 0.3
    min_peak_width: int = 3  # Minimum width of HVN peaks
    min_peak_prominence_pct: float = 0.1  # Minimum relative prominence for HVN peaks
    
    # For temporal weighting (span in minutes)
    ema_span: float = 90  # ~90 minutes looks back far enough while emphasizing recent activity
    
    # Internal state
    bin_edges: Optional[np.ndarray] = None
    bin_centers: Optional[np.ndarray] = None
    profile: Optional[np.ndarray] = None
    smoothed_profile: Optional[np.ndarray] = None
    
    def __post_init__(self):
        assert self.body_weight + self.wick_weight == 1.0
        assert self.ema_span > 1  # Allow inf
        if self.ema_span is None:
            self.ema_span = np.inf
        self.alpha = 2.0 / (self.ema_span + 1)  # EMA decay factor - naturally becomes 0 with inf

    def reset_for_day(self, day_low: float, day_high: float):
        """Reset bins and profile for a new trading day."""
        self.bin_edges = np.arange(
            floor_to_two_decimals(day_low) - 0.005,
            ceil_to_two_decimals(day_high) + self.bin_width + 0.005,
            self.bin_width
        )
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0
        self.profile = np.zeros(len(self.bin_centers), dtype=np.float64)
        
            
    # from trading_bot.TypedBarData import TypedBarData
    # def update_profile(self, bar: TypedBarData):
    def update_profile(self, bar) -> None:
        """Update volume profile with new bar data."""
        # Initial case - create first set of bins
        if self.bin_edges is None:
            self.reset_for_day(bar.low, bar.high)
            
        else:
            # Apply temporal decay to all bins
            self.apply_temporal_decay(self.profile, self.alpha)
            
            # Check if we need to expand bins
            need_lower = bar.low < self.bin_edges[0]
            need_upper = bar.high > self.bin_edges[-1]
            
            if need_lower or need_upper:
                # Calculate new bins needed on each end
                new_lower = []
                if need_lower:
                    first_edge = self.bin_edges[0]
                    while first_edge >= bar.low:
                        first_edge -= self.bin_width
                        new_lower.append(first_edge)
                    new_lower = new_lower[::-1]  # Reverse to maintain order
                    
                new_upper = []
                if need_upper:
                    last_edge = self.bin_edges[-1]
                    while last_edge <= bar.high:
                        last_edge += self.bin_width
                        new_upper.append(last_edge)
                
                # Create expanded arrays
                new_edges = np.concatenate([
                    new_lower,
                    self.bin_edges,
                    new_upper
                ])
                
                # Calculate new centers
                new_centers = (new_edges[:-1] + new_edges[1:]) / 2.0
                
                # Create expanded profile with zeros in new bins
                new_profile = np.zeros(len(new_centers))
                
                # Copy existing profile data to correct position
                if len(new_lower) > 0:
                    new_profile[len(new_lower):len(new_lower)+len(self.profile)] = self.profile
                else:
                    new_profile[:len(self.profile)] = self.profile
                
                # Update instance variables
                self.bin_edges = new_edges
                self.bin_centers = new_centers
                self.profile = new_profile
                
        # Normalize bar volume based on rolling window
        if bar.volume > 0:
            # Distribute volumes using normalized decay factor
            body_low = min(bar.open, bar.close)
            body_high = max(bar.open, bar.close)
            
            # Distribute body volume
            self.distribute_volume_to_profile(
                self.profile,
                self.bin_edges,
                body_low,
                body_high,
                bar.volume * self.body_weight
            )
            
            # Distribute upper wick volume if exists
            if bar.high > body_high:
                self.distribute_volume_to_profile(
                    self.profile,
                    self.bin_edges,
                    body_high,
                    bar.high,
                    bar.volume * self.wick_weight * 0.5
                )
            
            # Distribute lower wick volume if exists
            if bar.low < body_low:
                self.distribute_volume_to_profile(
                    self.profile,
                    self.bin_edges,
                    bar.low,
                    body_low,
                    bar.volume * self.wick_weight * 0.5
                )


    def calculate_min_peak_width(self, atr: float) -> int:
        """
        Calculate dynamic minimum peak width for 1-minute bars.
        
        Since we're using 1-minute bars:
        - price movement is more granular 
        - HVNs can form in tighter ranges
        - ATR represents very recent volatility
        - bin_width is 0.01 (1 cent)
        """
        if self.bin_edges is None:
            return 3
            
        # Convert ATR to number of bins (1 bin = 1 cent)
        atr_bins = atr / self.bin_width  # e.g., ATR of 0.05 -> 5 bins
        
        # Use body-weighted portion of ATR
        core_movement_bins = atr_bins * self.body_weight
        
        # For 1-min bars, target ~10% of core movement
        # Example: if ATR = 0.05 (5 cents):
        # - core_movement_bins = 5 * 0.7 ≈ 3.5 bins
        # - width = max(3, 3.5 * 0.1) ≈ 3 bins
        # width = max(self.min_peak_width, int(core_movement_bins * 0.1))
        
        
        # Target width around 1/3 of core movement
        # This represents a significant consolidation within normal trading range
        # width = max(self.min_peak_width, int(core_movement_bins / 3))
        
        # width = max(self.min_peak_width, int(core_movement_bins / 5))
        width = max(self.min_peak_width, int(core_movement_bins / 5))
        
        # return min(width, 7)  # Cap at 7 to avoid too-wide peaks
        return width  # No upper limit needed for 1-min granularity


    def find_hvns(self, atr: float) -> Tuple[np.ndarray, np.ndarray]:
        """Find High Volume Nodes using peak detection."""
        from scipy.signal import find_peaks
        from scipy.ndimage import gaussian_filter1d
        
        if self.profile is None or len(self.profile) == 0:
            return np.array([]), np.array([])

        min_width = self.calculate_min_peak_width(atr)
        
        # Normalize volume profile
        total_vol = np.sum(self.profile)
        if total_vol == 0:
            return np.array([]), np.array([])
            
        norm_profile = self.profile / total_vol
        
        # Apply Gaussian smoothing
        sigma = max(1.0, min_width / 4)  # Scale smoothing with peak width
        self.smoothed_profile = gaussian_filter1d(norm_profile, sigma=sigma)
        
        # print(np.sum(norm_profile), np.sum(self.smoothed_profile))
        
        # Calculate local threshold using rolling mean
        window = min_width * 2 + 1  # Odd window centered on potential peak
        rolling_mean = np.convolve(self.smoothed_profile, np.ones(window)/window, mode='same')
        
        # Each point must be X% higher than its local neighborhood average
        min_prominence = rolling_mean * self.min_peak_prominence_pct
        
        # Detect peaks with more relaxed width constraint
        peaks, properties = find_peaks(
            self.smoothed_profile,
            prominence=min_prominence,
            width=min_width,
            rel_height=1  # Consider half-height for width calculation
        )
        
        if len(peaks) == 0:
            return np.array([]), np.array([])
        
        # Return peaks and their prominences 
        # (prominences still relative to local baseline)
        return self.bin_centers[peaks], properties['prominences']
        # return self.bin_centers[peaks], properties['prominences'] * total_vol  # Denormalize prominences

    def get_hvn_metrics(self, current_price: float, atr: float) -> Tuple[float, float, float]:
        """
        Calculate HVN-specific metrics without ATR-based distance weighting.
        """
        hvn_prices, prominences = self.find_hvns(atr)
        if len(hvn_prices) == 0:
            return 0.0, 0.0, 0.0

        price_diffs = hvn_prices - current_price
        probs = prominences / np.sum(prominences)
        
        # Same metrics as calculate_moments but only for HVNs
        hvn_balance = np.sum(probs * np.sign(price_diffs))
        hvn_concentration = np.sum(probs * np.exp(-np.abs(price_diffs)))
        
        return hvn_balance, hvn_concentration, np.mean(prominences)

                    
    @staticmethod
    @jit(nopython=True)
    def calculate_moments_relative_to_price(
        bin_centers: np.ndarray,
        volumes: np.ndarray,
        current_price: float
    ) -> Tuple[float, float, float]:
        """
        Calculate volume distribution metrics similar to HVN detection but without peaks.
        
        Returns:
            Tuple of:
            - balance: Avg signed distance of volume (positive -> above price)
            - concentration: How clustered volume is near price
            - kurtosis: Peakedness of distribution
        """
        if len(volumes) == 0 or np.sum(volumes) == 0:
            return 0.0, 0.0, 0.0
            
        # Calculate relative distances from current price
        price_diffs = bin_centers - current_price
        
        # Normalize volumes to get probabilities
        total_volume = np.sum(volumes)
        probs = volumes / total_volume
        
        # Weighted balance - similar to HVN balance but for all volume
        balance = np.sum(probs * np.sign(price_diffs))
        
        # Concentration - how clustered volume is near price
        concentration = np.sum(probs * np.exp(-np.abs(price_diffs)))
        
        stds = np.std(price_diffs)
        
        # Kurtosis - relative to uniform distribution
        norm_diffs = price_diffs / stds if stds > 0 else np.zeros(len(price_diffs), dtype=np.float64)
        kurtosis = np.sum((norm_diffs ** 4) * probs) - 3
        
        return balance, concentration, kurtosis


        
    @staticmethod
    @jit(nopython=True)
    def apply_temporal_decay(profile: np.ndarray, decay_factor: float) -> None:
        """Apply EMA decay to entire profile in-place."""
        if profile is not None:
            profile *= (1 - decay_factor)

    @staticmethod
    @jit(nopython=True)
    def distribute_volume_to_profile(
        volume_profile: np.ndarray,
        bin_edges: np.ndarray,
        start_price: float,
        end_price: float,
        volume_to_distribute: float
    ) -> None:
        """Distribute volume uniformly across price range."""
        if start_price >= end_price:
            return
            
        start_idx = np.searchsorted(bin_edges, start_price) - 1
        end_idx = np.searchsorted(bin_edges, end_price) - 1
        
        if start_idx == end_idx:
            if 0 <= start_idx < len(volume_profile):
                # Add decayed amount
                volume_profile[start_idx] += volume_to_distribute
            return
            
        num_bins = end_idx - start_idx + 1
        if num_bins > 0:
            volume_per_bin = volume_to_distribute / num_bins
            for idx in range(start_idx, end_idx + 1):
                if 0 <= idx < len(volume_profile):
                    volume_profile[idx] += volume_per_bin
                    



    def plot_profile(self, current_price: float, current_timestamp: datetime, atr: float = None) -> None:
        """
        Plot volume profile distribution and detected HVNs.
        
        Args:
            current_price: Current price for reference line
            atr: ATR for HVN detection
        """
        return
        import matplotlib.pyplot as plt
        
        if self.profile is None or len(self.profile) == 0:
            print("No data to plot")
            return
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot volume profile as bars
        plt.bar(self.bin_centers, self.profile, width=self.bin_width,
                alpha=0.3, color='blue', label='VP')
        
        # Find and plot HVNs
        if atr:
            hvn_prices, prominences = self.find_hvns(atr)
            if len(hvn_prices) > 0:
                # Scale prominences for visibility
                scale_factor = max(self.profile) / max(self.smoothed_profile)
            
                plt.plot(self.bin_centers, self.smoothed_profile*scale_factor,
                        color='green', label='Smoothed VP')
            
                plt.scatter(hvn_prices, prominences * scale_factor, color='red', s=100,
                        label='HVNs', zorder=3)
            
        # Plot current price line
        plt.axvline(x=current_price, color='green', linestyle='--', 
                    label='Current Price', alpha=0.6)
        
        # Formatting
        plt.title(f'Volume Profile with HVNs at {current_timestamp}')
        plt.xlabel('Price')
        plt.ylabel('Volume')
        plt.grid(True, alpha=0.2)
        plt.legend()
        
        # Ensure some padding around the data
        plt.margins(x=0.02)
        
        plt.show()