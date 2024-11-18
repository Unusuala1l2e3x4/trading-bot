from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Set
from numba import jit, njit
import numpy as np
from datetime import datetime, time
import pandas as pd
import math

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from tqdm import tqdm

from MultiSymbolDataRetrieval import calculate_twap_micro_data

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



@dataclass
class SlippageEstimationParameters:
    # slippage_factor: Optional[float] = 0.001
    slippage_factor: Optional[float] = 0.02 # A scaling factor or slippage coefficient, which you can calibrate based on historical data.
    beta:  Optional[float] = 0.95 # An exponent typically less than 1 (commonly between 0.5 and 0.8), representing the non-linearity of the impact.
    
    

@jit(nopython=True)
def normalize_quote_count(count_sum: float, duration_seconds: float) -> float:
    """Normalize quote count to per-second rate."""
    return count_sum / duration_seconds if duration_seconds > 0 else 0.0

@dataclass
class OrderSizingParameters:
    """Order sizing specifically for initial position entries"""
    # Base size calculation
    max_volume_percentage: float = 1
    
    # Minimum thresholds
    min_trade_count: int = 100
    min_quote_count: int = 10  # Minimum quotes in recent micro data 
    
    # Pressure-based scaling parameters
    micro_pressure_beta: float = 0.3        # Sensitivity to pressure imbalance
    macro_pressure_beta: float = 0.6
    min_pressure_scaling: float = 0.4
    max_pressure_scaling: float = 1.3
    
    # Spread-based scaling parameters
    spread_beta: float = 0.5         # Sensitivity to spread changes
    min_spread_scaling: float = 0.5  # Minimum spread-based multiplier
    max_spread_scaling: float = 1.3  # Maximum spread-based multiplier (NOTE: using 1.0 -> only decreases allowed)
    max_spread_ratio: float = 3.0    # Cap on spread ratio. higher -> reduce size more (potentially)

    
    current_timestamp: pd.Timestamp = field(init=False)
    
    # @jit(nopython=True)
    def is_trading_allowed(
        self,
        total_equity: float,
        avg_trade_count: float,
        avg_volume: float,
        micro_data: pd.DataFrame,
        micro_interval_start: datetime,
        micro_interval_end: datetime,
        multiplier: float = 1.0
    ) -> bool:
        """
        Determine if trading is allowed based on multiple criteria.
        
        Args:
            total_equity: Account equity
            avg_trade_count: Average trades per minute from bar data
            avg_volume: Average volume per minute from bar data
            micro_data: Recent high-frequency quote data
            micro_duration_seconds_start: Start of micro data interval
            micro_duration_seconds_end: End of micro data interval
            multiplier: Optional adjustment to minimum thresholds
        """
        assert micro_interval_start < micro_interval_end
        
        # if total_equity < 25000:  # PDT threshold             # NOTE: comment out when necessary:
        #     return False
        
        min_trade_count = max(1, np.floor(self.min_trade_count * multiplier).astype(int)) # >= 1
        if avg_trade_count < min_trade_count or avg_volume < min_trade_count:
            return False
        
        # TODO: filter micro_data
        
        quote_count = normalize_quote_count( 
            micro_data['count'].sum(),
            (micro_interval_end - micro_interval_start).total_seconds()
        )
        min_quote_count = max(1, np.floor(self.min_quote_count * multiplier).astype(int)) # >= 1
        if quote_count < min_quote_count:
            return False
            
        return True
    
    # @jit(nopython=True)
    def calculate_max_trade_size(self, avg_volume: float) -> int:
        return np.floor(avg_volume * self.max_volume_percentage / 100).astype(int)
    
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
            log(f"rolling spread = {rolling_spread}",level=logging.WARNING)
            
        if current_spread < 0:
            log(f"current spread = {current_spread}",level=logging.WARNING)
            
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
        
        return ret
    
    # @jit(nopython=True)
    def calculate_pressure_scaling(
        self,
        buy_pressure: float,
        sell_pressure: float,
        is_buy: bool,
        is_micro: bool
    ) -> float:
        """
        Calculate scaling based on directional pressure.
        Returns 1.0 when:
        - pressures are equal
        - total pressure is zero (no clear directional pressure)
        """    
        assert buy_pressure >= 0, buy_pressure
        assert sell_pressure >= 0, sell_pressure
        total_pressure = buy_pressure + sell_pressure
        if total_pressure <= 0:
            ret = (self.min_pressure_scaling + self.max_pressure_scaling) / 2
            # log(f"{self.current_timestamp}: No market pressure detected (total pressure = {total_pressure}). Applying midpoint ({ret}).", level=logging.WARNING)
            return ret
            
        # Calculate pressure ratio relative to 0.5 (equal pressure)
        if is_buy:
            pressure_ratio = buy_pressure / total_pressure / 0.5  # normalize to 1.0 at equal pressure
        else:
            pressure_ratio = sell_pressure / total_pressure / 0.5
            
        # Scale based on favorable pressure
        # When pressure_ratio = 1.0 (equal pressure), scaling will be 1.0 regardless of beta
        if is_micro:
            scaling = pressure_ratio ** self.micro_pressure_beta
        else:
            scaling = pressure_ratio ** self.macro_pressure_beta
        ret = np.clip(
            scaling,
            self.min_pressure_scaling,
            self.max_pressure_scaling
        )
        
        # log(f"{self.current_timestamp}: buy_pressure {buy_pressure}, sell_pressure {sell_pressure} -> {ret}", level=logging.INFO)
        return ret 
        
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
        Adjust initial entry size based on market conditions and pressure.
        More conservative than regular position adjustments.
        """
        self.current_timestamp = current_timestamp
        
        assert micro_interval_start < micro_interval_end
        
        # Get spread metrics
        
        
        
        # TODO: filter micro_data
        
        # TODO: or take time-weighted EMA of spread_last (macro)
        
        # current_spread = micro_data['spread_last'].iloc[-1]
        current_spread = calculate_twap_micro_data(micro_data, micro_interval_start, micro_interval_end)
        
        # TODO: or take EMA of spread_twap                                                    
        rolling_spread = macro_data['spread_twap'].mean() # make sure its a full minute of data
        
        # Calculate pressure-based scaling
        buy_pressure_macro = ( # TODO: or take EMA of each column
            macro_data['bid_size_intra_pos_sum'].sum() + 
            abs(macro_data['ask_size_intra_neg_sum'].sum())
        )
        sell_pressure_macro = ( # TODO: or take EMA of each column
            macro_data['ask_size_intra_pos_sum'].sum() + 
            abs(macro_data['bid_size_intra_neg_sum'].sum())
        )
        
        pressure_scaling_macro = self.calculate_pressure_scaling(
            buy_pressure_macro,
            sell_pressure_macro,
            is_buy,
            is_micro=False
        )

        buy_pressure_micro = (
            micro_data['bid_size_intra_pos_sum'].sum() + 
            abs(micro_data['ask_size_intra_neg_sum'].sum())
        )
        sell_pressure_micro = (
            micro_data['ask_size_intra_pos_sum'].sum() + 
            abs(micro_data['bid_size_intra_neg_sum'].sum())
        )
        
        pressure_scaling_micro = self.calculate_pressure_scaling(
            buy_pressure_micro,
            sell_pressure_micro,
            is_buy,
            is_micro=True
        )
        
        pressure_scaling = 0.5*pressure_scaling_macro + 0.5*pressure_scaling_micro
        
        
        # Get spread-based scaling
        spread_scaling = self.calculate_spread_scaling(
            current_spread,
            rolling_spread
        )
        
        # Combine scalings (spread has higher priority)
        final_scaling = spread_scaling * pressure_scaling
        ret = np.round(base_size * final_scaling).astype(int) # closest int
        
        # log(f"{self.current_timestamp}: scaling calc {pressure_scaling} * {spread_scaling} = {final_scaling} ({base_size} -> {ret})", level=logging.INFO)
        
        return ret, pressure_scaling, spread_scaling, final_scaling


@dataclass
class StrategyParameters:
    initial_investment: float = 10_000
    max_investment: float = float("inf") # highest allowed notional trade size 
    do_longs: bool = True
    do_shorts: bool = True
    sim_longs: bool = True
    sim_shorts: bool = True
    use_margin: bool = False
    times_buying_power: float = 4
    min_stop_dist_relative_change_for_partial: Optional[float] = 0
    soft_start_time: Optional[time] = None
    soft_end_time: Optional[time] = None

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
