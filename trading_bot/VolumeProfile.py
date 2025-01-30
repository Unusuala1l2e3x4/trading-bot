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

import matplotlib.pyplot as plt
    
@jit(nopython=True)
def floor_to_two_decimals(x):
    """Floor value x down to the nearest cent."""
    return np.floor(x * 100) / 100.0

@jit(nopython=True)
def ceil_to_two_decimals(x):
    """Ceil value x up to the nearest cent."""
    return np.ceil(x * 100) / 100.0


@jit(nopython=True)
def estimate_base_distribution_parameters(
    bar_high: float,
    bar_low: float,
    bar_vwap: float,
    atr: float,
    bin_width: float
) -> Tuple[float, float, float]:
    """
    Calculate parameters for base distribution that fits within bar range.
    The distribution will have its mean (not mode) at VWAP and be contained
    between high and low.
    
    Args:
        bar_high: High price of the bar
        bar_low: Low price of the bar
        bar_vwap: Volume-weighted average price
        atr: Average True Range
        
    Returns:
        Tuple of:
        - location (mode of the distribution)
        - scale (width parameter, carefully calibrated)
        - alpha (skewness parameter)
    """
    bar_range = bar_high - bar_low
    if bar_range == 0:
        # bar_range = max(bin_width, atr * 0.1)
        bar_range = bin_width
        bar_high = bar_vwap + bar_range/2
        bar_low = bar_vwap - bar_range/2
        return bar_vwap, bar_range/8, 0.0  # No skew needed for zero-range bars
    
    # Calculate relative position of VWAP in range [-1, 1]
    relative_pos = 2 * ((bar_vwap - bar_low) / bar_range) - 1
    
    # Calculate alpha (skewness)
    alpha = relative_pos * 2
    
    # For a skewed normal, we need to account for how skewness affects spread
    # The larger the skew, the more we need to reduce scale to stay within bounds
    skew_factor = 1.0 + abs(alpha) * 0.5  # Increase scale reduction with skewness
    
    # Calculate scale that will contain most of distribution within bar_range
    # For normal distribution, ±3σ contains 99.7% of the distribution
    # We'll be more conservative and use a smaller multiple to account for skewness
    scale = (bar_range / (3 * skew_factor))  # Divide by 6 for ±3σ, then reduce further based on skew
    
    # Calculate location parameter (mode) that will give us mean at VWAP
    sqrt_2_pi = np.sqrt(2/np.pi)
    location = bar_vwap - alpha * scale * sqrt_2_pi
    
    return location, scale, alpha

@jit(nopython=True)
def calculate_volume_split(
    base_distribution: np.ndarray,
    bin_edges: np.ndarray,
    bar_open: float,
    bar_close: float,
    is_bullish: bool
) -> Tuple[float, float]:
    """
    Calculate buy/sell volume split using distribution integrals.
    
    Returns:
        Tuple of (buy_fraction, sell_fraction)
    """
    # Find indices for open and close prices
    open_idx = np.searchsorted(bin_edges, bar_open) - 1
    close_idx = np.searchsorted(bin_edges, bar_close) - 1
    
    # Calculate area in body
    body_start = min(open_idx, close_idx)
    body_end = max(open_idx, close_idx)
    body_area = np.sum(base_distribution[body_start:body_end+1])
    
    # Calculate total area
    total_area = np.sum(base_distribution)
    wick_area = total_area - body_area
    
    total_area_adjusted = total_area + wick_area
    if total_area_adjusted == 0:
        return 0.5, 0.5

    smaller_fraction = wick_area / total_area_adjusted 
    larger_fraction = total_area / total_area_adjusted 
    
    if is_bullish:
        return larger_fraction, smaller_fraction
    else:
        return smaller_fraction, larger_fraction


@jit(nopython=True)
def create_skewed_distribution(
    profile: np.ndarray,
    bin_edges: np.ndarray,
    mean: float,
    std: float,
    skew: float,
    volume: float,
    bar_low: float,
    bar_high: float
) -> None:
    """
    Create skewed normal distribution centered at mean.
    Truncated to stay within bar's range.
    Updates profile in-place.
    """
    # Pre-compute constants
    a = 1 / (std * np.sqrt(2 * np.pi))
    b = -1 / (2 * std * std)
    
    # Calculate probabilities with skew
    probs = np.zeros(len(profile))
    for i in range(len(profile)):
        x = (bin_edges[i] + bin_edges[i+1]) / 2
        
        # Only calculate probability if within bar's range
        if bar_low <= x <= bar_high:
            z = (x - mean) / std
            
            # Basic normal distribution
            normal = a * np.exp(b * (x - mean) * (x - mean))
            
            # Apply skew using error function
            # skew_factor = 1 + skew * z
            skew_factor = np.exp(skew * z)  # Exponential
            probs[i] = normal * skew_factor
    
    # Ensure non-negative probabilities
    probs = np.maximum(probs, 0)
    
    # Normalize and apply volume
    total_prob = np.sum(probs)
    if total_prob > 0:
        probs = probs / total_prob
        profile += probs * volume



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
    
    # Separate profiles for buy/sell volume
    buy_profile: Optional[np.ndarray] = None
    sell_profile: Optional[np.ndarray] = None
    
    # Smoothed versions (used for HVN detection)
    smoothed_buy_profile: Optional[np.ndarray] = None  
    smoothed_sell_profile: Optional[np.ndarray] = None
    
    def __post_init__(self):
        assert self.body_weight + self.wick_weight == 1.0
        assert self.ema_span > 1  # Allow inf
        if self.ema_span is None:
            self.ema_span = np.inf
        self.alpha = 2.0 / (self.ema_span + 1)  # EMA decay factor - naturally becomes 0 with inf

    def reset_for_day(self, day_low: float, day_high: float):
        """Reset bins and both buy/sell profiles for a new trading day."""
        self.bin_edges = np.arange(
            floor_to_two_decimals(day_low) - 0.005,
            ceil_to_two_decimals(day_high) + self.bin_width + 0.005,
            self.bin_width
        )
        self.bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0
        
        # Initialize both profiles
        self.buy_profile = np.zeros(len(self.bin_centers), dtype=np.float64)
        self.sell_profile = np.zeros(len(self.bin_centers), dtype=np.float64)
        self.smoothed_buy_profile = None
        self.smoothed_sell_profile = None
        
            
    # from trading_bot.TypedBarData import TypedBarData
    # def update_profile(self, bar: TypedBarData):
    def update_profile(self, bar) -> None:
        """Update buy/sell profiles with new bar data."""
        # Initial case
        if self.bin_edges is None:
            self.reset_for_day(bar.low, bar.high)
        else:
            # Apply temporal decay to both profiles
            self.apply_temporal_decay(self.buy_profile, self.alpha)
            self.apply_temporal_decay(self.sell_profile, self.alpha)
            
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
                
                # When expanding, need to expand both profiles
                # Create expanded profiles with zeros in new bins
                new_buy_profile = np.zeros(len(new_centers))
                new_sell_profile = np.zeros(len(new_centers))
                
                # Copy existing profile data to correct position
                if len(new_lower) > 0:
                    new_buy_profile[len(new_lower):len(new_lower)+len(self.buy_profile)] = self.buy_profile
                    new_sell_profile[len(new_lower):len(new_lower)+len(self.sell_profile)] = self.sell_profile
                else:
                    new_buy_profile[:len(self.buy_profile)] = self.buy_profile
                    new_sell_profile[:len(self.sell_profile)] = self.sell_profile
                
                # Update instance variables
                self.bin_edges = new_edges
                self.bin_centers = new_centers
                self.buy_profile = new_buy_profile
                self.sell_profile = new_sell_profile
                
        # Normalize bar volume based on rolling window
        if bar.volume > 0:
            self.distribute_volume_to_profile(
                self.buy_profile,
                self.sell_profile,
                self.bin_edges,
                bar.open,
                bar.close,
                bar.high,
                bar.low,
                bar.vwap,
                bar.volume,
                bar.ATR,
                self.bin_width
            )



    @staticmethod
    @jit(nopython=True)
    def distribute_volume_to_profile(
        buy_profile: np.ndarray,
        sell_profile: np.ndarray,
        bin_edges: np.ndarray,
        bar_open: float,
        bar_close: float,
        bar_high: float,
        bar_low: float,
        bar_vwap: float,
        volume: float,
        atr: float,
        bin_width: float
    ) -> None:
        """
        Distribute volume across buy and sell profiles using bounded gaussian distributions.
        1. Creates base distribution
        2. Calculates exact volume split
        3. Applies to buy/sell profiles
        """
        # Get base distribution parameters
        mean, std, skew = estimate_base_distribution_parameters(
            bar_high, bar_low, bar_vwap, atr, bin_width
        )
        
        # Create temporary array for base distribution
        base_dist = np.zeros(len(buy_profile))
        create_skewed_distribution(base_dist, bin_edges, mean, std, skew, 1.0, bar_low, bar_high)  # Unit volume
        
        # Calculate volume split
        is_bullish = bar_close >= bar_open
        buy_frac, sell_frac = calculate_volume_split(
            base_dist, bin_edges, bar_open, bar_close, is_bullish
        )
        
        # Apply to buy/sell profiles
        create_skewed_distribution(buy_profile, bin_edges, mean, std, skew, volume * buy_frac, bar_low, bar_high)
        create_skewed_distribution(sell_profile, bin_edges, mean, std, skew, volume * sell_frac, bar_low, bar_high)



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


    def _find_profile_hvns(self, profile: np.ndarray, other_profile: np.ndarray, atr: float) -> Tuple[np.ndarray, np.ndarray]:
        """Helper to find HVNs in a single profile."""
        if profile is None or len(profile) == 0:
            return np.array([]), np.array([])

        min_width = self.calculate_min_peak_width(atr)
        
        # Normalize volume profile
        total_vol = np.sum(profile) + np.sum(other_profile)
        if total_vol == 0:
            return np.array([]), np.array([])
            
        norm_profile = profile / total_vol
        
        # Apply Gaussian smoothing
        sigma = max(1.0, min_width / 4)  # Scale smoothing with peak width
        smoothed = gaussian_filter1d(norm_profile, sigma=sigma)
        
        # print(np.sum(norm_profile), np.sum(self.smoothed_profile))
        
        # Calculate local threshold using rolling mean
        window = min_width * 2 + 1  # Odd window centered on potential peak
        rolling_mean = np.convolve(smoothed, np.ones(window)/window, mode='same')
        
        # Each point must be X% higher than its local neighborhood average
        min_prominence = rolling_mean * self.min_peak_prominence_pct
        
        # Detect peaks with more relaxed width constraint
        peaks, properties = find_peaks(
            smoothed,
            prominence=min_prominence,
            width=min_width,
            rel_height=1  # Consider half-height for width calculation
        )
        
        if len(peaks) == 0:
            return np.array([]), np.array([])
        
        # Return peaks and their prominences 
        # (prominences still relative to local baseline)
        return self.bin_centers[peaks], properties['prominences'], smoothed
        # return self.bin_centers[peaks], properties['prominences'] * total_vol  # Denormalize prominences



    def find_hvns(self, atr: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Find HVNs for both buy and sell profiles.
        
        Returns:
            Tuple containing:
            - buy_hvn_prices, buy_prominences
            - sell_hvn_prices, sell_prominences
        """
        # Find HVNs in buy profile
        buy_hvn_prices, buy_prominences, self.smoothed_buy_profile = self._find_profile_hvns(
            self.buy_profile, self.sell_profile, atr
        )
        
        # Find HVNs in sell profile  
        sell_hvn_prices, sell_prominences, self.smoothed_sell_profile = self._find_profile_hvns(
            self.sell_profile, self.buy_profile, atr
        )
        
        return buy_hvn_prices, buy_prominences, sell_hvn_prices, sell_prominences
    
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_profile_hvn_metrics(
        hvn_prices: np.ndarray,
        prominences: np.ndarray,
        current_price: float
    ) -> Tuple[float, float, float]:
        """
        Calculate HVN-specific metrics for a single profile.
        
        Args:
            hvn_prices: Prices where HVNs were detected
            prominences: Prominence values for each HVN
            current_price: Current price for reference
        
        Returns:
            Tuple of:
            - balance: If HVNs are mostly above/below price 
            - concentration: How clustered HVNs are near price
            - avg_prominence: Average strength of detected HVNs
        """
        if len(hvn_prices) == 0:
            return 0.0, 0.0, 0.0

        # Calculate price differences from current
        price_diffs = hvn_prices - current_price
        
        # Normalize prominences to get probabilities
        probs = prominences / np.sum(prominences)
        
        # Calculate metrics
        balance = np.sum(probs * np.sign(price_diffs))
        concentration = np.sum(probs * np.exp(-np.abs(price_diffs)))
        avg_prominence = np.mean(prominences)
        
        return balance, concentration, avg_prominence


    def get_hvn_metrics(self, current_price: float, atr: float) -> Tuple[float, float, float, float, float, float]:
        """Calculate HVN metrics for both buy and sell profiles.
        
        Returns:
            Tuple containing:
            - buy_balance, buy_concentration, buy_avg_prominence
            - sell_balance, sell_concentration, sell_avg_prominence
        """
        # Get HVNs for both profiles
        buy_prices, buy_proms, sell_prices, sell_proms = self.find_hvns(atr)
        
        # Calculate metrics for buy HVNs
        buy_balance, buy_concentration, buy_avg_prom = self._calculate_profile_hvn_metrics(
            buy_prices, buy_proms, current_price
        )
        
        # Calculate metrics for sell HVNs
        sell_balance, sell_concentration, sell_avg_prom = self._calculate_profile_hvn_metrics(
            sell_prices, sell_proms, current_price
        )
        
        return (
            (buy_balance, buy_concentration, buy_avg_prom),
            (sell_balance, sell_concentration, sell_avg_prom)
        )



                    
    @staticmethod
    @jit(nopython=True)
    def calculate_moments_relative_to_price(
        bin_centers: np.ndarray,
        volumes: np.ndarray,
        other_volumes: np.ndarray,
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
        total_volume = np.sum(volumes) + np.sum(other_volumes)
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


    def calculate_profile_moments(
        self, 
        current_price: float
    ) -> Tuple[
        Tuple[float, float, float],  # Buy profile moments
        Tuple[float, float, float]   # Sell profile moments
    ]:
        """
        Calculate moments for both buy and sell profiles relative to current price.
        
        Args:
            current_price: Current price for calculating relative distances
            
        Returns:
            Tuple containing:
            - (buy_balance, buy_concentration, buy_kurtosis)
            - (sell_balance, sell_concentration, sell_kurtosis)
        """
        if self.buy_profile is None or self.sell_profile is None:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

        # Calculate moments for buy profile
        buy_balance, buy_concentration, buy_kurtosis = self.calculate_moments_relative_to_price(
            self.bin_centers,
            self.buy_profile,
            self.sell_profile,
            current_price
        )
        
        # Calculate moments for sell profile
        sell_balance, sell_concentration, sell_kurtosis = self.calculate_moments_relative_to_price(
            self.bin_centers,
            self.sell_profile,
            self.buy_profile,
            current_price
        )
        
        return (
            (buy_balance, buy_concentration, buy_kurtosis),
            (sell_balance, sell_concentration, sell_kurtosis)
        )
        
    @staticmethod
    @jit(nopython=True)
    def apply_temporal_decay(profile: np.ndarray, decay_factor: float) -> None:
        """Apply EMA decay to entire profile in-place."""
        if profile is not None:
            profile *= (1 - decay_factor)


    def plot_profile(self, current_price: float, current_vwap: float, current_timestamp: datetime, atr: float = None) -> None:
        """
        Plot buy and sell volume profiles overlayed with transparency.
        
        Args:
            current_price: Current price for reference line
            current_timestamp: Current timestamp for title
            atr: ATR for HVN detection
        """
        # return
        
        if self.buy_profile is None or self.sell_profile is None or len(self.buy_profile) == 0:
            print("No data to plot")
            return
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Plot buy profile in green with transparency
        plt.bar(self.bin_centers, self.buy_profile, width=self.bin_width,
                alpha=0.4, color='green', label='Buy Volume')
                
        # Plot sell profile in red with transparency
        plt.bar(self.bin_centers, self.sell_profile, width=self.bin_width,
                alpha=0.4, color='red', label='Sell Volume')
        
        # Find and plot HVNs for both profiles
        if atr:
            # Buy profile HVNs
            buy_hvn_prices, buy_prominences, sell_hvn_prices, sell_prominences = self.find_hvns(atr)
            
            # Calculate scaling factors
            buy_scale = max(self.buy_profile) / max(self.smoothed_buy_profile) if len(self.smoothed_buy_profile) > 0 else 1
            sell_scale = max(self.sell_profile) / max(self.smoothed_sell_profile) if len(self.smoothed_sell_profile) > 0 else 1
            
            # Plot smoothed profiles
            if len(self.smoothed_buy_profile) > 0:
                plt.plot(self.bin_centers, self.smoothed_buy_profile * buy_scale,
                        color='darkgreen', linestyle='--', alpha=0.5, label='Smoothed Buy Volume')
                
            if len(self.smoothed_sell_profile) > 0:
                plt.plot(self.bin_centers, self.smoothed_sell_profile * sell_scale,
                        color='darkred', linestyle='--', alpha=0.5, label='Smoothed Sell Volume')
            
            # Plot HVN points
            if len(buy_hvn_prices) > 0:
                plt.scatter(buy_hvn_prices, buy_prominences * buy_scale,
                        color='darkgreen', s=100, marker='^', alpha=0.7, label='Buy HVNs', zorder=3)
                
            if len(sell_hvn_prices) > 0:
                plt.scatter(sell_hvn_prices, sell_prominences * sell_scale,
                        color='darkred', s=100, marker='v', alpha=0.7, label='Sell HVNs', zorder=3)
        
        # Plot current price line
        plt.axvline(x=current_price, color='blue', linestyle='--', 
                    label='Current Price', alpha=0.6)
        
        plt.axvline(x=current_vwap, color='gray', linestyle='--', 
                    label='Current Price', alpha=0.6)
        
        # Formatting
        plt.title(f'Buy/Sell Volume Profiles with HVNs at {current_timestamp}')
        plt.xlabel('Price')
        plt.ylabel('Volume')
        plt.grid(True, alpha=0.2)
        plt.legend()
        
        # Ensure some padding around the data
        plt.margins(x=0.02)
        
        plt.show()