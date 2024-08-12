from dataclasses import dataclass, field
from typing import List, Dict, Optional
from numba import jit
import numpy as np
from datetime import datetime, time
import pandas as pd
from TouchArea import TouchArea, TouchAreaCollection
from TradePosition import TradePosition, export_trades_to_csv, plot_cumulative_pnl_and_price

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from tqdm import tqdm

ny_tz = ZoneInfo("America/New_York")


@dataclass
class StrategyParameters:
    initial_investment: float = 10_000
    max_investment: float = float("inf")
    do_longs: bool = True
    do_shorts: bool = True
    sim_longs: bool = True
    sim_shorts: bool = True
    use_margin: bool = False
    times_buying_power: float = 4
    min_stop_dist_relative_change_for_partial: Optional[float] = 0
    soft_start_time: Optional[time] = None
    soft_end_time: Optional[time] = None
    max_volume_percentage: float = 0.01
    min_trade_count: int = 100
    slippage_factor: Optional[float] = 0.001

class TradingStrategy:
    def __init__(self, touch_detection_areas: dict, params: StrategyParameters):
        self.touch_detection_areas = touch_detection_areas
        self.params = params
        self.initialize_strategy()

    def initialize_strategy(self):
        self.symbol = self.touch_detection_areas['symbol']
        self.long_touch_area = self.touch_detection_areas['long_touch_area']
        self.short_touch_area = self.touch_detection_areas['short_touch_area']
        self.market_hours = self.touch_detection_areas['market_hours']
        self.df = self.touch_detection_areas['bars']
        self.mask = self.touch_detection_areas['mask']
        self.min_touches = self.touch_detection_areas['min_touches']
        self.start_time = self.touch_detection_areas['start_time']
        self.end_time = self.touch_detection_areas['end_time']

        self.balance = self.params.initial_investment
        self.total_account_value = self.params.initial_investment
        self.open_positions = {}
        self.trades = []
        self.trades_executed = 0

        self.initialize_touch_areas()

    def initialize_touch_areas(self):
        all_touch_areas = []
        if self.params.do_longs or self.params.sim_longs:
            all_touch_areas.extend(self.long_touch_area)
        if self.params.do_shorts or self.params.sim_shorts:
            all_touch_areas.extend(self.short_touch_area)
        self.touch_area_collection = TouchAreaCollection(all_touch_areas, self.min_touches)

    def update_total_account_value(self, current_price: float, name: str):
        for position in self.open_positions.values():
            position.update_market_value(current_price)
        
        market_value = sum(position.market_value for position in self.open_positions.values())
        cash_committed = sum(position.cash_committed for position in self.open_positions.values())
        self.total_account_value = self.balance + cash_committed
        
        # Debug printing logic here (similar to your original function)

    def rebalance(self, is_simulated: bool, cash_change: float, current_price: float = None):
        if not is_simulated:
            old_balance = self.balance
            new_balance = self.balance + cash_change
            assert new_balance >= 0, f"Negative balance encountered: {new_balance:.4f} ({old_balance:.4f} {cash_change:.4f})"
            self.balance = new_balance

        if current_price is not None:
            self.update_total_account_value(current_price, 'REBALANCE')
        
        # Assert and debug printing logic here (similar to your original function)

    def exit_action(self, area_id: int, position: TradePosition):
        # Logic for handling position exit (similar to your original function)
        self.trades.append(position)

    def close_all_positions(self, timestamp: datetime, exit_price: float, vwap: float, volume: float, avg_volume: float):
        # Logic for closing all positions (similar to your original function)
        pass

    def calculate_position_details(self, current_price: float, avg_volume: float, avg_trade_count: float, volume: float,
                                   existing_sub_positions=None, target_shares=None):
        # Logic for calculating position details (similar to your original function)
        pass

    def place_stop_market_buy(self, area: TouchArea, timestamp: datetime, data, prev_close: float):
        # Logic for placing a stop market buy order (similar to your original function)
        pass

    def calculate_exit_details(self, shares_to_sell: int, volume: float, avg_volume: float, avg_trade_count: float):
        # Logic for calculating exit details (similar to your original function)
        pass

    def update_positions(self, timestamp: datetime, data):
        # Logic for updating positions (similar to your original function)
        pass

    def run_backtest(self):
        # Main backtesting loop
        for i in range(1, len(self.df)):
            current_time = self.df.index[i]
            data = self.df.iloc[i]
            prev_close = self.df['close'].iloc[i-1]

            self.update_positions(current_time, data)

            if self.can_open_new_position(current_time):
                active_areas = self.touch_area_collection.get_active_areas(current_time)
                for area in active_areas:
                    if self.should_enter_position(area):
                        if self.place_stop_market_buy(area, current_time, data, prev_close):
                            break  # Exit the loop after placing a position

            if self.should_close_all_positions(current_time):
                self.close_all_positions(current_time, data['close'], data['vwap'], data['volume'], data['avg_volume'])

        self.finalize_backtest()

    def can_open_new_position(self, current_time: datetime) -> bool:
        # Check if we can open a new position based on time and existing positions
        pass

    def should_enter_position(self, area: TouchArea) -> bool:
        # Check if we should enter a position for this area
        pass

    def should_close_all_positions(self, current_time: datetime) -> bool:
        # Check if we should close all positions (e.g., end of day)
        pass

    def finalize_backtest(self):
        # Calculate and print final backtest results
        pass

    # Add any additional helper methods here

# Usage
params = StrategyParameters(
    initial_investment=10_000,
    do_longs=True,
    do_shorts=True,
    use_margin=True,
    times_buying_power=4,
    min_stop_dist_relative_change_for_partial=0,
    soft_start_time=None,
    soft_end_time=time(15, 50)
)

strategy = TradingStrategy(touch_detection_areas, params)
results = strategy.run_backtest()