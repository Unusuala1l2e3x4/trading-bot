from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Set
from numba import jit, njit
import numpy as np
from datetime import datetime, time
import pandas as pd

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from tqdm import tqdm

from trading_bot.MultiSymbolDataRetrieval import calculate_twap_micro_data
from trading_bot.TypedBarData import TypedBarData

ny_tz = ZoneInfo("America/New_York")

POSITION_OPENED = True
NO_POSITION_OPENED = False

from alpaca.trading import TradingClient

import os, toml
from dotenv import load_dotenv

load_dotenv(override=True)
livepaper = os.getenv('LIVEPAPER')
config = toml.load('../config.toml')

# Replace with your Alpaca API credentials
API_KEY = config[livepaper]['key']
API_SECRET = config[livepaper]['secret']

trading_client = TradingClient(API_KEY, API_SECRET)


import logging
def setup_logger(log_level=logging.INFO):
    logger = logging.getLogger('TradingStrategyParameters')
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


# symbol,group,slippage_factor,atr_sensitivity,notes
# NVDA,High Volume Meme,0.02,15.0,Highest volume + tight spread despite being meme stock
# TSLA,High Volume Meme,0.03,12.0,High volume but wider spreads + high volatility means higher base slippage
# AAPL,Blue Chip,0.02,10.0,Very liquid with tight spreads despite lower volume than memes
# AMZN,Blue Chip,0.025,10.0,Similar to AAPL but slightly wider spreads historically
# MARA,Low Price High Vol,0.01,20.0,Lower $ slippage due to price but very sensitive to volatility

@dataclass
class SlippageEstimationParameters:
    # slippage_factor: Optional[float] = 0.001
    # slippage_factor: Optional[float] = 0.02 # Dollar cost per share, which you can calibrate based on historical data.
    # beta:  Optional[float] = 0.95 # An exponent typically less than 1 (commonly between 0.5 and 0.8), representing the non-linearity of the impact.]
    
    
    # Average/Representative
    slippage_factor: Optional[float] = 0.005
    atr_sensitivity: Optional[float] = 12
    
    # # Conservative/Overestimate
    # slippage_factor: Optional[float] = 0.008  # 0.8 cents per share base slippage
    # atr_sensitivity: Optional[float] = 18.0   # higher sensitivity to catch volatile periods
        
    

@jit(nopython=True)
def normalize_quote_count(count_sum: float, duration_seconds: float) -> float:
    """Normalize quote count to per-second rate."""
    return count_sum / duration_seconds if duration_seconds > 0 else 0.0

@dataclass
class OrderSizingParameters:
    """Order sizing specifically for initial position entries"""
    # Base size calculation
    max_volume_percentage: float = 1
    
    # Minimum thresholds for trading
    min_trade_count: int = 100 # Minimum rolling average trades per minute
    min_quote_count: int = 10  # Minimum quotes in recent micro data 
    
    # All betas = 0.5 for balanced sensitivity
    spread_beta: float = 0.5 # Sensitivity to spread changes
    stability_beta: float = 0.1 # Sensitivity to quote bouncing
    persistence_beta: float = 0.8 # Sensitivity to persistence changes.
    
    # Allow slight increase in good conditions
    # But more defensive in poor conditions
    min_spread_scaling: float = 0.5 # less important that max_spread_ratio
    max_spread_scaling: float = 2.0
    min_stability_scaling: float = 0.5
    max_stability_scaling: float = 2
    min_persistence_scaling: float = 0.8
    max_persistence_scaling: float = 1.2

    # Cap spread ratio to prevent extreme scaling
    max_spread_ratio: float = 5 # Cap on spread ratio. higher -> reduce size more (potentially)
    
    current_timestamp: pd.Timestamp = field(init=False)
    
    # def __post_init__(self):
    #     print(f"Spread scaling limits with max_spread_ratio {self.max_spread_ratio} and spread_beta {self.spread_beta}: "
    #         f"{(1/self.max_spread_ratio) ** self.spread_beta :.3f} - {self.max_spread_ratio ** self.spread_beta :.3f}, "
    #         f"{self.min_spread_scaling :.3f} - {self.max_spread_scaling :.3f} set min-max\n")
        
    # @jit(nopython=True)
    def is_trading_allowed(
        self,
        total_equity: float,
        avg_trade_count: float,
        avg_volume: float,
        # micro_data: pd.DataFrame,
        # micro_interval_start: datetime,
        # micro_interval_end: datetime,
        multiplier: float = 1.0
    ) -> bool:
        """
        Determine if trading is allowed based on multiple criteria.
        
        Args:
            total_equity: Account equity
            avg_trade_count: Rolling average trades per minute from bar data
            avg_volume: Rolling average volume per minute from bar data
            micro_data: Recent high-frequency quote data
            micro_duration_seconds_start: Start of micro data interval
            micro_duration_seconds_end: End of micro data interval
            multiplier: Optional adjustment to minimum thresholds
        """
        # assert micro_interval_start < micro_interval_end
        
        # if total_equity < 25000:  # PDT threshold             # NOTE: comment out when necessary:
        #     return False
        
        min_trade_count = max(1, np.round(self.min_trade_count * multiplier).astype(int)) # >= 1
        if avg_trade_count < min_trade_count or avg_volume < min_trade_count:
            return False
        
        # # NOTE: no quotes data used for live trading to prioritize speed
        # # Thus: removing the functionality below
        # quote_count = normalize_quote_count( 
        #     micro_data['count'].sum(),
        #     (micro_interval_end - micro_interval_start).total_seconds()
        # )
        # min_quote_count = max(1, np.round(self.min_quote_count * multiplier).astype(int)) # >= 1
        # if quote_count < min_quote_count:
        #     return False
        
        # TODO: there needs to be at least one quote
            
        return True
    
    # @jit(nopython=True)
    def calculate_max_trade_size(self, avg_volume: float) -> int:
        return np.round(avg_volume * self.max_volume_percentage / 100).astype(int)
    
    # @jit(nopython=True)
    def calculate_spread_scaling(
        self,
        current_spread: float,
        rolling_spread: float
    ) -> float:
        """
        Calculate scaling based on spread conditions.
        Returns 1.0 when current spread equals rolling spread.
        """
        if rolling_spread <= 0:
            log(f"{self.current_timestamp}: rolling spread = {rolling_spread}",level=logging.WARNING)
            
        if current_spread < 0:
            log(f"{self.current_timestamp}: current spread = {current_spread}",level=logging.WARNING)
            
        if current_spread == 0 and rolling_spread == 0:
            spread_ratio = 1.0
            
        elif rolling_spread < 0 and current_spread < 0:
            spread_ratio = np.clip(
                rolling_spread / current_spread,
                1/self.max_spread_ratio,
                self.max_spread_ratio
            )
            
        elif current_spread > 0 and rolling_spread <= 0:
            spread_ratio = self.max_spread_ratio
        
        elif current_spread <= 0 and rolling_spread > 0:
            spread_ratio = 1/self.max_spread_ratio
        
        else:
            # Calculate bounded spread ratio relative to 1.0 (equal spreads)
            # When spreads are equal, spread_ratio will be 1.0
            spread_ratio = np.clip(
                current_spread / rolling_spread,
                1/self.max_spread_ratio,
                self.max_spread_ratio
            )

        # When spread_ratio = 1.0, scaling will be 1.0 regardless of beta
        scaling = (1/spread_ratio) ** self.spread_beta
        ret = np.clip(
            scaling,
            self.min_spread_scaling,
            self.max_spread_scaling
        )
        
        # log(f"{self.current_timestamp}: current_spread {current_spread:.6f}, rolling_spread {rolling_spread:.6f} -> {ret:.6f}", level=logging.INFO)
        
        return ret, spread_ratio


    def calculate_size_persistence(
        self,
        micro_data: pd.DataFrame,
        is_buy: bool
    ) -> float:
        """
        Calculate how persistent quote sizes are.
        Large sizes that stay in the book suggest real interest.
        Returns 1.0 in perfectly stable conditions.
        """
        if is_buy:
            # For buys, check ask persistence (we'll hit the ask)
            added = micro_data['ask_size_intra_pos_sum'].sum()  # Already positive
            removed = -micro_data['ask_size_intra_neg_sum'].sum()  # Make positive
        else:
            # For sells, check bid persistence (we'll hit the bid)
            added = micro_data['bid_size_intra_pos_sum'].sum()  # Already positive
            removed = -micro_data['bid_size_intra_neg_sum'].sum()  # Make positive

        assert added >= 0, added
        assert removed >= 0, removed
        total = added + removed
        if total == 0:
            return 1.0  # Perfect stability
            
        # Normalize to 1.0 in perfectly stable conditions (added = removed)
        persistence = 2 * (added / total)  # Now equals 1.0 when added = removed
        
        # Apply sensitivity
        scaling = persistence ** self.persistence_beta
        
        return np.clip(
            scaling, 
            self.min_persistence_scaling,
            self.max_persistence_scaling
        )
        
            
    def calculate_quote_stability(
        self,
        micro_data: pd.DataFrame,
        macro_data: pd.DataFrame,
        is_buy: bool,
        micro_interval_start: datetime,
        micro_interval_end: datetime
    ) -> float:
        """
        Calculate quote stability normalized to macro data.
        Returns 1.0 for average stability conditions.
        """
        def get_changes_per_second(data, is_micro=False):
            if is_buy:
                # For buys, we care about ask side (we're going to hit the ask)
                price_changes = data['ask_price_intra_pos_sum'].sum() - data['ask_price_intra_neg_sum'].sum()
                size_changes = data['ask_size_intra_pos_sum'].sum() - data['ask_size_intra_neg_sum'].sum()
            else:
                # For sells, we care about bid side (we're going to hit the bid)
                price_changes = data['bid_price_intra_pos_sum'].sum() - data['bid_price_intra_neg_sum'].sum()
                size_changes = data['bid_size_intra_pos_sum'].sum() - data['bid_size_intra_neg_sum'].sum()
            
            assert price_changes >= 0, price_changes
            assert size_changes >= 0, size_changes
            if is_micro:
                actual_interval = (micro_interval_end - micro_interval_start).total_seconds()
                # Normalize each component separately
                return (price_changes / actual_interval) * (size_changes / actual_interval)
            else:
                # Macro data is over 60 seconds - normalize each component
                return (price_changes / 60.0) * (size_changes / 60.0)
            
        micro_changes = get_changes_per_second(micro_data, is_micro=True)
        macro_changes = get_changes_per_second(macro_data)
        
        if macro_changes == 0:
            # log(f"{self.current_timestamp}: no macro data detected",level=logging.WARNING)
            return 1.0  # No changes = perfectly stable
            
        # Normalize relative to macro activity (1.0 means typical stability)
        relative_stability = macro_changes / micro_changes if micro_changes > 0 else 2.0
        
        # print(macro_changes, micro_changes, relative_stability)
        
        # Apply sensitivity
        scaling = relative_stability ** self.stability_beta
        
        return np.clip(
            scaling,
            self.min_stability_scaling,
            self.max_stability_scaling
        )
            
        
    def adjust_max_trade_size(
        self,
        current_timestamp: datetime,
        base_size: int,
        micro_data: pd.DataFrame,
        macro_data: pd.DataFrame,
        is_buy: bool,
        micro_interval_start: datetime,
        micro_interval_end: datetime
    ) -> int:
        """
        Adjust the maximum trade size based on market conditions.
        
        Args:
            current_timestamp: Current time of the trade decision.
            base_size: Base trade size to adjust.
            micro_data: Recent high-frequency quote data.
            macro_data: Aggregated quote data over 1 minute, grouped in 1-second intervals.
            is_buy: Boolean indicating if the trade is a buy order.
            micro_interval_start: Start of the micro data interval.
            micro_interval_end: End of the micro data interval.
        """

        self.current_timestamp = current_timestamp
        
        assert micro_interval_start < micro_interval_end
        
        # Calculate TWAP spread for micro data
        current_spread = calculate_twap_micro_data(micro_data, micro_interval_start, micro_interval_end)
        
        # Macro spread mean (stability from aggregation)
        rolling_spread = macro_data['spread_twap'].mean() 
            
        # Calculate scalings
        spread_scaling, spread_ratio = self.calculate_spread_scaling(current_spread, rolling_spread)
        stability_scaling = self.calculate_quote_stability(micro_data, macro_data, is_buy, micro_interval_start, micro_interval_end)
        persistence_scaling = self.calculate_size_persistence(micro_data, is_buy)
        
        # Combine scalings (spread has highest priority)
        
        # TODO: adjust this:
        # final_scaling = spread_scaling * (
        #     0.65 * stability_scaling + 
        #     0.35 * persistence_scaling
        # )
        
        # final_scaling = spread_scaling * (
        #     0.5 * stability_scaling + 
        #     0.5 * persistence_scaling
        # )
        
        # final_scaling = spread_scaling * stability_scaling
        # final_scaling = spread_scaling * persistence_scaling
        
        # final_scaling = spread_scaling * (
        #     0.5 * stability_scaling + 
        #     0.5 * persistence_scaling
        # )
        
        # final_scaling = spread_scaling * stability_scaling * persistence_scaling
        # final_scaling = spread_scaling * stability_scaling 
        # final_scaling = spread_scaling
        
        final_scaling = (spread_scaling + stability_scaling + persistence_scaling) / 3
        # final_scaling = spread_scaling
                         
        # final_scaling = 1 # TEST
        
        # ret = np.round(base_size * final_scaling).astype(int) # closest int
        ret = np.round(base_size).astype(int) # TEST
            
        # log(f"{self.current_timestamp}: scaling calc {pressure_scaling} * {spread_scaling} = {final_scaling} ({base_size} -> {ret})", level=logging.INFO)
        
        return ret, spread_ratio, spread_scaling, stability_scaling, persistence_scaling, final_scaling


@dataclass
class StrategyParameters:
    initial_investment: float = 10_000
    max_investment: float = float("inf") # highest allowed notional trade size 
    do_longs: bool = True
    do_shorts: bool = True
    sim_longs: bool = True
    sim_shorts: bool = True
    use_margin: bool = False
    assume_marginable_and_etb: bool = False
    times_buying_power: float = 1
    min_stop_dist_relative_change_for_partial: Optional[float] = 0
    soft_start_time: Optional[time] = None
    soft_end_time: Optional[time] = None
    
    gradual_entry_range_multiplier: Optional[float] = 1.0
    
    # gradual_entry_range_multiplier: Optional[float] = 0.75 # better for meme stocks but not others
    # gradual_entry_range_multiplier: Optional[float] = 0.5 # bad
    # gradual_entry_range_multiplier: Optional[float] = 1.25 # bad except for MARA
    # gradual_entry_range_multiplier: Optional[float] = 0.9 # better for NVDA and TSLA but not others

    # lunch_stop_entry_time: Optional[time] = time(11, 45)
    # lunch_resume_entry_time: Optional[time] = time(12, 15)
    
    plot_day_results: bool = False
    plot_volume_profiles: bool = False
    allow_reversal_detection: bool = False
    clear_passed_areas: bool = False
    clear_traded_areas: bool = False
    
    rsi_overbought: float = 70
    rsi_oversold: float = 30
    
    mfi_overbought: float = 80
    mfi_oversold: float = 20
    # mfi_overbought: float = 70
    # mfi_oversold: float = 30
    
    # For temporal weighting (span in minutes)
    volume_profile_ema_span: float = 120  # ~90 minutes looks back far enough while emphasizing recent activity    volume_profile_ema: float = 90
    # volume_profile_ema_span: float = np.inf
    # volume_profile_ema_span: float = 240
    

    def __post_init__(self):
        assert self.rsi_overbought > self.rsi_oversold
        assert self.mfi_overbought > self.mfi_oversold
    
    slippage: SlippageEstimationParameters = field(default_factory=SlippageEstimationParameters)
    ordersizing: OrderSizingParameters = field(default_factory=OrderSizingParameters)
    

    
    def __post_init__(self):
        assert 0 < self.times_buying_power <= 4
        assert self.do_longs or self.do_shorts
        assert 0 <= self.min_stop_dist_relative_change_for_partial <= 1
        if self.soft_start_time:
            if not isinstance(self.soft_start_time, time):
                self.soft_start_time = pd.to_datetime(self.soft_start_time, format='%H:%M').time()
            assert self.soft_start_time.second == 0 and self.soft_start_time.microsecond == 0

        if self.soft_end_time:
            if not isinstance(self.soft_end_time, time):
                self.soft_end_time = pd.to_datetime(self.soft_end_time, format='%H:%M').time()
            assert self.soft_end_time.second == 0 and self.soft_end_time.microsecond == 0   

    def copy(self, **changes):
        new_params = deepcopy(asdict(self))
        new_params.update(changes)
        return StrategyParameters(**new_params)
