from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from numba import jit
import numpy as np
from datetime import datetime, time
import pandas as pd
import math
from TouchArea import TouchArea, TouchAreaCollection
from TradePosition import TradePosition, export_trades_to_csv, plot_cumulative_pnl_and_price

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from tqdm import tqdm

ny_tz = ZoneInfo("America/New_York")

POSITION_OPENED = True
NO_POSITION_OPENED = False

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.trading import TradingClient
from alpaca.trading.requests import GetCalendarRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment

import os, toml
from dotenv import load_dotenv

load_dotenv(override=True)
livepaper = os.getenv('LIVEPAPER')
config = toml.load('../config.toml')

# Replace with your Alpaca API credentials
API_KEY = config[livepaper]['key']
API_SECRET = config[livepaper]['secret']

trading_client = TradingClient(API_KEY, API_SECRET)

def is_security_shortable_and_etb(symbol: str) -> bool:
    asset = trading_client.get_asset(symbol)
    return asset.shortable and asset.easy_to_borrow

def is_security_marginable(symbol: str) -> bool:
    try:
        asset = trading_client.get_asset(symbol)
        return asset.marginable
    except Exception as e:
        print(f"Error checking marginability for {symbol}: {e}")
        return False


@jit(nopython=True)
def is_trading_allowed(avg_trade_count, min_trade_count, avg_volume) -> bool:
    return avg_trade_count >= min_trade_count and avg_volume >= min_trade_count # at least 1 share per trade

@jit(nopython=True)
def calculate_max_trade_size(avg_volume: float, max_volume_percentage: float) -> int:
    return math.floor(avg_volume * max_volume_percentage)

@dataclass
class TouchDetectionAreas:
    symbol: str
    long_touch_area: List[TouchArea]
    short_touch_area: List[TouchArea]
    market_hours: Dict[datetime.date, Tuple[datetime, datetime]]
    bars: pd.DataFrame
    mask: pd.Series
    min_touches: int
    start_time: Optional[time]
    end_time: Optional[time]

    @classmethod
    def from_dict(cls, data: dict) -> 'TouchDetectionAreas':
        return cls(
            symbol=data['symbol'],
            long_touch_area=data['long_touch_area'],
            short_touch_area=data['short_touch_area'],
            market_hours=data['market_hours'],
            bars=data['bars'],
            mask=data['mask'],
            min_touches=data['min_touches'],
            start_time=data['start_time'],
            end_time=data['end_time']
        )



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
    
    def __post_init__(self):
        assert 0 < self.times_buying_power <= 4
        assert self.do_longs or self.do_shorts
        assert 0 <= self.min_stop_dist_relative_change_for_partial <= 1
        if self.soft_start_time:
            self.soft_start_time = pd.to_datetime(self.soft_start_time, format='%H:%M').time()
        if self.soft_end_time:
            self.soft_end_time = pd.to_datetime(self.soft_end_time, format='%H:%M').time()        

    
class TradingStrategy:
    def __init__(self, touch_detection_areas: TouchDetectionAreas, params: StrategyParameters, export_trades_path: Optional[str]=None):
        self.touch_detection_areas = touch_detection_areas
        self.params = params        
        self.export_trades_path = export_trades_path
        
        self.symbol = self.touch_detection_areas.symbol
        self.long_touch_area = self.touch_detection_areas.long_touch_area
        self.short_touch_area = self.touch_detection_areas.short_touch_area
        self.market_hours = self.touch_detection_areas.market_hours
        self.bars = self.touch_detection_areas.bars
        self.mask = self.touch_detection_areas.mask
        self.min_touches = self.touch_detection_areas.min_touches
        self.start_time = self.touch_detection_areas.start_time
        self.end_time = self.touch_detection_areas.end_time

        self.df = self.bars[self.mask].sort_index(level='timestamp')
        self.timestamps = self.df.index.get_level_values('timestamp')
        
        self.initialize_strategy()

    def initialize_strategy(self, export_trades_path: Optional[str]=None):
        self.balance = self.params.initial_investment
        self.total_account_value = self.params.initial_investment
        self.open_positions = {}
        self.trades = []
        self.trades_executed = 0

        self.is_marginable = is_security_marginable(self.symbol) 
        self.is_etb = is_security_shortable_and_etb(self.symbol)
        
        self.current_id = 0
        self.count_entry_adjust = 0
        self.count_exit_adjust = 0
        self.count_entry_skip = 0
        self.count_exit_skip = 0
        
        if export_trades_path:
            self.export_trades_path = export_trades_path
        
        print(f'{self.symbol} is {'NOT ' if not self.is_marginable else ''}marginable.')
        print(f'{self.symbol} is {'NOT ' if not self.is_etb else ''}shortable and ETB.')
        
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
        positions_to_remove = []

        for area_id, position in list(self.open_positions.items()):
            realized_pnl, cash_released, fees = position.partial_exit(timestamp, exit_price, position.shares, vwap, volume, avg_volume, self.params.slippage_factor)
            self.rebalance(position.is_simulated, cash_released + realized_pnl - fees, exit_price)
            position.close(timestamp, exit_price)
            self.trades_executed += 1
            position.area.record_entry_exit(position.entry_time, position.entry_price, 
                                            timestamp, exit_price)
            position.area.terminate(self.touch_area_collection)
            positions_to_remove.append(area_id)

        temp = {}
        for area_id in positions_to_remove:
            temp[area_id] = self.open_positions[area_id]
            del self.open_positions[area_id]
        for area_id in positions_to_remove:
            self.exit_action(area_id, temp[area_id])
        assert not self.open_positions

    def calculate_position_details(self, current_price: float, times_buying_power: float, avg_volume: float, avg_trade_count: float, volume: float,
                                max_volume_percentage: float, min_trade_count: int,
                                existing_sub_positions: Optional[np.ndarray] = np.array([]), target_shares: Optional[int]=None):
        # Logic for calculating position details (similar to your original function)
        if not is_trading_allowed(avg_trade_count, min_trade_count, avg_volume):
            return 0, 0, 0, 0, 0, 0, 0
        
        # when live, need to call is_security_marginable
        if self.params.use_margin and self.is_marginable:
            initial_margin_requirement = 0.5
            overall_margin_multiplier = min(times_buying_power, 4.0)
        else:
            initial_margin_requirement = 1.0
            overall_margin_multiplier = min(times_buying_power, 1.0)
            
        actual_margin_multiplier = min(overall_margin_multiplier, 1.0/initial_margin_requirement)
        
        available_balance = min(self.balance, self.params.max_investment) * overall_margin_multiplier
        
        current_shares = np.sum(existing_sub_positions) if existing_sub_positions is not None else 0
        if target_shares is not None:
            assert target_shares > current_shares
        max_volume_shares = calculate_max_trade_size(avg_volume, max_volume_percentage)
            
        max_additional_shares = min(
            (target_shares - current_shares) if target_shares is not None else math.floor((available_balance - (current_shares * current_price)) / current_price),
            max_volume_shares - current_shares
        )
        
        # Binary search to find the maximum number of shares we can buy
        low, high = 0, max_additional_shares
        while low <= high:
            mid = (low + high) // 2
            total_shares = current_shares + mid
            invest_amount = mid * current_price
            estimated_entry_cost = TradePosition.estimate_entry_cost(total_shares, overall_margin_multiplier, existing_sub_positions)
            if invest_amount + estimated_entry_cost * overall_margin_multiplier <= available_balance:
                low = mid + 1
            else:
                high = mid - 1
        
        max_additional_shares = high
        max_shares = current_shares + max_additional_shares
        
        # Ensure max_shares is divisible when times_buying_power > 2
        num_subs = TradePosition.calculate_num_sub_positions(overall_margin_multiplier)
        
        if num_subs > 1:
            assert self.is_marginable
            rem = max_shares % num_subs
            if rem != 0:
                max_shares -= rem
                max_additional_shares -= rem

        invest_amount = max_additional_shares * current_price
        actual_cash_used = invest_amount / overall_margin_multiplier
        estimated_entry_cost = TradePosition.estimate_entry_cost(max_shares, overall_margin_multiplier, existing_sub_positions)

        return max_shares, actual_margin_multiplier, overall_margin_multiplier, estimated_entry_cost, actual_cash_used, max_additional_shares, invest_amount


    def place_stop_market_buy(self, area: TouchArea, timestamp: datetime, data, prev_close: float):
        # Logic for placing a stop market buy order (similar to your original function)
        open_price, high_price, low_price, close_price, volume, trade_count, vwap, avg_volume, avg_trade_count = \
            data.open, data.high, data.low, data.close, data.volume, data.trade_count, data.vwap, data.avg_volume, data.avg_trade_count
        
        if not is_trading_allowed(avg_trade_count, self.params.min_trade_count, avg_volume):
            return NO_POSITION_OPENED
        
        if self.open_positions or self.balance <= 0:
            return NO_POSITION_OPENED

        # debug_print(f"Attempting order: {'Long' if area.is_long else 'Short'} at {area.get_buy_price:.4f}")
        # debug_print(f"  Balance: {balance:.4f}, Total Account Value: {total_account_value:.4f}")

        # Check if the stop buy would have executed based on high/low.
        if area.is_long:
            if prev_close > area.get_buy_price:
                # debug_print(f"  Rejected: Previous close ({prev_close:.4f}) above buy price, likey re-entering area ({area.get_buy_price:.4f})")
                return NO_POSITION_OPENED
            if high_price < area.get_buy_price or close_price > high_price:
                # debug_print(f"  Rejected: High price ({high_price:.4f}) didn't reach buy price ({area.get_buy_price:.4f})")
                return NO_POSITION_OPENED
            # if close_price < area.get_buy_price: # biggest decrease in performance
            #     return NO_POSITION_OPENED
        else:  # short
            if prev_close < area.get_buy_price:
                # debug_print(f"  Rejected: Previous close ({prev_close:.4f}) below buy price, likey re-entering area ({area.get_buy_price:.4f})")
                return NO_POSITION_OPENED
            if low_price > area.get_buy_price or close_price < low_price:
                # debug_print(f"  Rejected: Low price ({low_price:.4f}) didn't reach buy price ({area.get_buy_price:.4f})")
                return NO_POSITION_OPENED
            # if close_price > area.get_buy_price: # biggest decrease in performance
            #     return NO_POSITION_OPENED

        # execution_price = area.get_buy_price # Stop buy (placed at time of min_touches) would have executed
        # execution_price = np.mean([area.get_buy_price, close_price]) # balanced approach, may account for slippage
        execution_price = close_price # if not using stop buys
        
        # debug3_print(f"Execution price: {execution_price:.4f}")

        # Calculate position size, etc...
        max_shares, actual_margin_multiplier, overall_margin_multiplier, estimated_entry_cost, actual_cash_used, max_additional_shares, invest_amount = self.calculate_position_details(
            execution_price, self.params.times_buying_power, avg_volume, avg_trade_count, volume,
            self.params.max_volume_percentage, self.params.min_trade_count
        )


        if actual_cash_used + estimated_entry_cost > self.balance:
            return NO_POSITION_OPENED
        
        # Create the position
        position = TradePosition(
            date=timestamp.date(),
            id=self.current_id,
            area=area,
            is_long=area.is_long,
            entry_time=timestamp,
            initial_balance=actual_cash_used,
            initial_shares=max_shares,
            entry_price=execution_price,
            use_margin=self.params.use_margin,
            is_marginable=self.is_marginable, # when live, need to call is_security_marginable
            times_buying_power=overall_margin_multiplier,
            actual_margin_multiplier=actual_margin_multiplier,
            current_stop_price=high_price - area.get_range if area.is_long else low_price + area.get_range,
            max_price=high_price if area.is_long else None,
            min_price=low_price if not area.is_long else None
        )
    
        if (area.is_long and self.params.do_longs) or (not area.is_long and self.params.do_shorts and self.is_etb):  # if conditions not met, simulate position only.
            position.is_simulated = False
        else:
            position.is_simulated = True

        cash_needed, fees = position.initial_entry(vwap, volume, avg_volume, self.params.slippage_factor)

        assert estimated_entry_cost >= fees
        
        self.current_id += 1
        
        # Add to open positions (regardless if real or simulated)
        self.open_positions[area.id] = position
        
        self.rebalance(position.is_simulated, -cash_needed - fees, close_price)
            
        return POSITION_OPENED



    def calculate_exit_details(self, times_buying_power: float, shares_to_sell: int, volume: float, avg_volume: float, avg_trade_count: float, max_volume_percentage: float, min_trade_count: int):
        # Logic for calculating exit details (similar to your original function)
        
        # Calculate the adjustment factor
        adjustment = min(max(shares_to_sell / avg_volume, 0), 1)
        
        # Adjust min_trade_count, with a lower bound of 10% of the original value
        adjusted_min_trade_count = max(min_trade_count * (1 - adjustment), min_trade_count * 0.1)
        
        # Check if trading is allowed
        if not is_trading_allowed(avg_trade_count, adjusted_min_trade_count, avg_volume):
            return 0  # No trading allowed, return 0 shares to sell
        
        # when live, need to call is_security_marginable
        if self.params.use_margin and self.is_marginable:
            overall_margin_multiplier = min(times_buying_power, 4.0)
        else:
            overall_margin_multiplier = min(times_buying_power, 1.0)

        # Adjust max_volume_percentage, with an upper bound of 3 times the original value
        adjusted_max_volume_percentage = min(max_volume_percentage * (1 + adjustment), max_volume_percentage * 3)
        
        max_volume_shares = calculate_max_trade_size(avg_volume, adjusted_max_volume_percentage)

        # Ensure we don't sell more than the available shares or the calculated max_volume_shares
        shares_to_sell = min(shares_to_sell, max_volume_shares)
            
        # Ensure target_shares is divisible
        num_subs = TradePosition.calculate_num_sub_positions(overall_margin_multiplier)
        if num_subs > 1:
            rem = shares_to_sell % num_subs
            if rem != 0:
                shares_to_sell -= rem
                
        return shares_to_sell


    def update_positions(self, timestamp: datetime, data):
        # Logic for updating positions (similar to your original function)
        
        open_price, high_price, low_price, close_price, volume, trade_count, vwap, avg_volume, avg_trade_count = \
            data.open, data.high, data.low, data.close, data.volume, data.trade_count, data.vwap, data.avg_volume, data.avg_trade_count
        
        positions_to_remove = []

        # if using trailing stops, exit_price = None
        def perform_exit(area_id, position, exit_price=None):
            price = position.current_stop_price if exit_price is None else exit_price
            position.close(timestamp, price)
            self.trades_executed += 1
            position.area.record_entry_exit(position.entry_time, position.entry_price, 
                                            timestamp, price)
            position.area.terminate(self.touch_area_collection)
            positions_to_remove.append(area_id)
            

        def calculate_target_shares(position: TradePosition, current_price):
            if position.is_long:
                price_movement = current_price - position.current_stop_price
            else:
                price_movement = position.current_stop_price - current_price
            target_pct = min(max(0, price_movement / position.area.get_range),  1.0)
            target_shares = math.floor(target_pct * position.max_shares)

            num_subs = TradePosition.calculate_num_sub_positions(position.times_buying_power)
            rem = target_shares % num_subs
                
            # Ensure target_shares is divisible
            if num_subs > 1 and rem != 0:
                assert self.is_marginable
                target_shares -= rem
            
            return target_shares

        for area_id, position in self.open_positions.items():
            price_at_action = None
            
            # OHLC logic for trailing stops
            # Initial tests found that just using close_price is more effective
            # Implies we aren't using trailing stop sells
            # UNLESS theres built-in functionality to wait until close
            
            # if not price_at_action:
            #     should_exit = position.update_stop_price(open_price)
            #     target_shares = calculate_target_shares(position, open_price)
            #     if should_exit or target_shares == 0:
            #         perform_exit(area_id, position) # DO NOT pass price into function since order would have executed at current_stop_price.
            #         price_at_action = open_price
            
            # # If not stopped out at open, simulate intra-minute price movement
            # if not price_at_action:
            #     should_exit = position.update_stop_price(high_price)
            #     target_shares = calculate_target_shares(position, high_price)
            #     if not position.is_long and (should_exit or target_shares == 0):
            #         # For short positions, the stop is crossed if high price increases past it
            #         perform_exit(area_id, position) # DO NOT pass price into function since order would have executed at current_stop_price.
            #         price_at_action = high_price
            
            # if not price_at_action:
            #     should_exit = position.update_stop_price(low_price)
            #     target_shares = calculate_target_shares(position, low_price)
            #     if position.is_long and (should_exit or target_shares == 0):
            #         # For long positions, the stop is crossed if low price decreases past it
            #         perform_exit(area_id, position) # DO NOT pass price into function since order would have executed at current_stop_price.
            #         price_at_action = low_price
            
            
            if not price_at_action:
                should_exit = position.update_stop_price(close_price)
                target_shares = calculate_target_shares(position, close_price)
                if should_exit or target_shares == 0:
                    price_at_action = close_price
                
            if price_at_action:
                assert target_shares == 0
            
            if not price_at_action:
                price_at_action = close_price
            
            # Partial exit and entry logic
            assert target_shares <= position.initial_shares

            target_pct = target_shares / position.initial_shares
            current_pct = min( 1.0, position.shares / position.initial_shares)
            assert 0.0 <= target_pct <= 1.0, target_pct
            assert 0.0 <= current_pct <= 1.0, current_pct


            # To prevent over-trading, skip partial buy/sell if difference between target and current shares percentage is less than threshold
            # BUT only if not increasing/decrease to/from 100%
            # Initial tests found that a threshold closer to 0 or 1, not in between, gives better results
            if abs(target_pct - current_pct) < self.params.min_stop_dist_relative_change_for_partial:
                self.update_total_account_value(price_at_action, 'skip')
                continue

            
            if target_shares < position.shares:
                shares_to_adjust = position.shares - target_shares
                
                if shares_to_adjust > 0:

                    shares_to_sell = self.calculate_exit_details(
                        position.times_buying_power,
                        shares_to_adjust,
                        volume,
                        avg_volume,
                        avg_trade_count,
                        self.params.max_volume_percentage,
                        math.floor(self.params.min_trade_count * (shares_to_adjust / position.max_shares))
                    )
                    
                    if shares_to_sell > 0:
                        realized_pnl, cash_released, fees = position.partial_exit(timestamp, price_at_action, shares_to_sell, vwap, volume, avg_volume, self.params.slippage_factor)

                        self.rebalance(position.is_simulated, cash_released + realized_pnl - fees, price_at_action)
                            
                        if position.shares == 0:
                            perform_exit(area_id, position, price_at_action)

                        if shares_to_sell < shares_to_adjust:

                            self.count_exit_adjust += 1
                    else:
                        self.count_exit_skip += 1
                        
            elif target_shares > position.shares:
                shares_to_adjust = target_shares - position.shares
                if shares_to_adjust > 0:

                    existing_sub_positions = np.array([sp.shares for sp in position.sub_positions if sp.shares > 0])
                    max_shares, _, _, estimated_entry_cost, actual_cash_used, max_additional_shares, invest_amount = self.calculate_position_details(
                        price_at_action, position.times_buying_power, avg_volume, avg_trade_count, volume,
                        self.params.max_volume_percentage, math.floor(self.params.min_trade_count * (shares_to_adjust / position.max_shares)), 
                        existing_sub_positions=existing_sub_positions, target_shares=target_shares
                    )
                    
                    shares_to_buy = min(shares_to_adjust, max_additional_shares)
                    
                    if shares_to_buy > 0:
                        cash_needed, fees = position.partial_entry(timestamp, price_at_action, shares_to_buy, vwap, volume, avg_volume, self.params.slippage_factor)
                        self.rebalance(position.is_simulated, -cash_needed - fees, price_at_action)
                        
                        # Update max_shares after successful partial entry
                        position.max_shares = max(position.max_shares, position.shares)
                        assert position.shares == max_shares

                        if shares_to_buy < shares_to_adjust:
                            self.count_entry_adjust += 1
                    else:
                        # Update max_shares when entry is skipped
                        position.max_shares = min(position.max_shares, position.shares)
                        self.count_entry_skip += 1

        temp = {}
        for area_id in positions_to_remove:
            temp[area_id] = self.open_positions[area_id]
            del self.open_positions[area_id]
        for area_id in positions_to_remove:
            self.exit_action(area_id, temp[area_id])
        
        self.update_total_account_value(close_price, 'AFTER removing exited positions')

    def run_backtest(self):
        timestamps = self.df.index.get_level_values('timestamp')
        
        daily_data = None
        current_date, market_open, market_close = None, None, None
        
        for i in tqdm(range(1, len(timestamps))):
            current_time = timestamps[i].tz_convert(ny_tz)
            
            if current_time.date() != current_date:
                self.handle_new_trading_day(current_time, timestamps)
                current_date = current_time.date()
                daily_data, market_open, market_close, day_start_time, day_end_time, day_soft_start_time = self.get_daily_data(current_date)
                daily_index = 1
                assert not self.open_positions
                soft_end_triggered = False
            
            if not market_open or not market_close:
                continue
            
            if self.is_trading_time(current_time, day_soft_start_time, day_end_time, daily_index, daily_data, i):
                if self.params.soft_end_time and not soft_end_triggered:
                    soft_end_triggered = self.check_soft_end_time(current_time, current_date)

                prev_close = daily_data['close'].iloc[daily_index - 1]
                data = daily_data.iloc[daily_index]
                
                self.update_positions(current_time, data)
                
                if not soft_end_triggered:
                    self.process_active_areas(current_time, data, prev_close)
            elif current_time >= day_end_time:
                self.close_all_positions(current_time, self.df['close'].iloc[i], self.df['vwap'].iloc[i], 
                                         self.df['volume'].iloc[i], self.df['avg_volume'].iloc[i])
            elif i >= len(self.df) - 1: # only for testing. not in live environment
                self.close_all_positions(current_time, self.df['close'].iloc[i], self.df['vwap'].iloc[i], 
                                         self.df['volume'].iloc[i], self.df['avg_volume'].iloc[i])
            daily_index += 1
        
        if current_time >= day_end_time:
            assert not self.open_positions

        return self.generate_backtest_results()


    # def can_open_new_position(self, current_time: datetime) -> bool:
    #     # Check if we can open a new position based on time and existing positions
    #     pass

    # def should_enter_position(self, area: TouchArea) -> bool:
    #     # Check if we should enter a position for this area
    #     pass

    # def should_close_all_positions(self, current_time: datetime) -> bool:
    #     # Check if we should close all positions (e.g., end of day)
    #     pass
    
    
    def handle_new_trading_day(self, current_time, timestamps):
        self.current_id = 0
        # Any other new day initialization logic

    def get_daily_data(self, current_date):
        daily_data = self.df[self.df.index.get_level_values('timestamp').date == current_date]
        market_open, market_close = self.market_hours.get(current_date, (None, None))
        if market_open and market_close:
            day_start_time, day_end_time, day_soft_start_time = self.calculate_day_times(current_date, market_open, market_close)
        else:
            day_start_time = day_end_time = day_soft_start_time = None

        return daily_data, market_open, market_close, day_start_time, day_end_time, day_soft_start_time

    def calculate_day_times(self, current_date, market_open, market_close):
        date_obj = pd.Timestamp(current_date).tz_localize(ny_tz)
        
        day_start_time = date_obj.replace(hour=self.start_time.hour, minute=self.start_time.minute) if self.start_time else market_open
        day_end_time = min(date_obj.replace(hour=self.end_time.hour, minute=self.end_time.minute), 
                           market_close - pd.Timedelta(minutes=3)) if self.end_time else market_close - pd.Timedelta(minutes=3)
        
        if self.params.soft_start_time:
            day_soft_start_time = max(market_open, day_start_time, 
                                      date_obj.replace(hour=self.params.soft_start_time.hour, minute=self.params.soft_start_time.minute))
        else:
            day_soft_start_time = max(market_open, day_start_time)
        
        return day_start_time, day_end_time, day_soft_start_time

    def is_trading_time(self, current_time, day_soft_start_time, day_end_time, daily_index, daily_data, i):
        return (day_soft_start_time <= current_time < day_end_time and 
                daily_index < len(daily_data) - 1 and i < len(self.df) - 1) # LAST PART is only for testing. not in live environment

    def check_soft_end_time(self, current_time, current_date):
        if current_time >= pd.Timestamp.combine(current_date, self.params.soft_end_time).tz_localize(ny_tz):
            return True
        return False

    def process_active_areas(self, current_time, data, prev_close):
        active_areas = self.touch_area_collection.get_active_areas(current_time)
        for area in active_areas:
            if self.balance <= 0:
                break
            if self.open_positions:  # ensure only 1 live position at a time
                break
            
            if ((area.is_long and (self.params.do_longs or self.params.sim_longs)) or 
                (not area.is_long and (self.params.do_shorts or self.params.sim_shorts))):
                if self.place_stop_market_buy(area, current_time, data, prev_close):
                    break  # Exit the loop after placing a position

    def generate_backtest_results(self):
        # Calculate and return backtest results
        balance_change = ((self.balance - self.params.initial_investment) / self.params.initial_investment) * 100

        # Buy and hold strategy
        start_price = self.df['close'].iloc[0]
        end_price = self.df['close'].iloc[-1]
        baseline_change = ((end_price - start_price) / start_price) * 100
        
        total_profit_loss = sum(trade.profit_loss for trade in self.trades if not trade.is_simulated)
        
        total_profit = sum(trade.profit_loss for trade in self.trades if not trade.is_simulated and trade.profit_loss > 0)
        total_loss = sum(trade.profit_loss for trade in self.trades if not trade.is_simulated and trade.profit_loss < 0)
        
        total_transaction_costs = sum(trade.total_transaction_costs for trade in self.trades if not trade.is_simulated)
        total_stock_borrow_costs = sum(trade.total_stock_borrow_cost for trade in self.trades if not trade.is_simulated)

        mean_profit_loss = np.mean([trade.profit_loss for trade in self.trades if not trade.is_simulated])
        mean_profit_loss_pct = np.mean([trade.profit_loss_pct for trade in self.trades if not trade.is_simulated])

        win_mean_profit_loss_pct = np.mean([trade.profit_loss_pct for trade in self.trades if not trade.is_simulated and trade.profit_loss > 0])
        lose_mean_profit_loss_pct = np.mean([trade.profit_loss_pct for trade in self.trades if not trade.is_simulated and trade.profit_loss < 0])
        
        long_win_mean_profit_loss_pct = np.mean([trade.profit_loss_pct for trade in self.trades if not trade.is_simulated and trade.is_long and trade.profit_loss > 0])
        long_lose_mean_profit_loss_pct = np.mean([trade.profit_loss_pct for trade in self.trades if not trade.is_simulated and trade.is_long and trade.profit_loss < 0])
        
        short_win_mean_profit_loss_pct = np.mean([trade.profit_loss_pct for trade in self.trades if not trade.is_simulated and not trade.is_long and trade.profit_loss > 0])
        short_lose_mean_profit_loss_pct = np.mean([trade.profit_loss_pct for trade in self.trades if not trade.is_simulated and not trade.is_long and trade.profit_loss < 0])
        
        win_trades = sum(1 for trade in self.trades if not trade.is_simulated and trade.profit_loss > 0)
        lose_trades = sum(1 for trade in self.trades if not trade.is_simulated and trade.profit_loss < 0)
        win_longs = sum(1 for trade in self.trades if not trade.is_simulated and trade.is_long and trade.profit_loss > 0)
        lose_longs = sum(1 for trade in self.trades if not trade.is_simulated and trade.is_long and trade.profit_loss < 0)
        win_shorts = sum(1 for trade in self.trades if not trade.is_simulated and not trade.is_long and trade.profit_loss > 0)
        lose_shorts = sum(1 for trade in self.trades if not trade.is_simulated and not trade.is_long and trade.profit_loss < 0)
        
        avg_sub_pos = np.mean([len(trade.sub_positions) for trade in self.trades if not trade.is_simulated])
        avg_transact = np.mean([len(trade.transactions) for trade in self.trades if not trade.is_simulated])
        
        assert self.trades_executed == len(self.trades)

        # Print statistics
        print(f"END\nStrategy: {'Long' if self.params.do_longs else ''}{'&' if self.params.do_longs and self.params.do_shorts else ''}{'Short' if self.params.do_shorts else ''}")
        print(f'{self.symbol} is {'NOT ' if not self.is_marginable else ''}marginable.')
        print(f'{self.symbol} is {'NOT ' if not self.is_etb else ''}shortable and ETB.')
        print(f"{self.timestamps[0]} -> {self.timestamps[-1]}")

        # debug2_print(df['close'])
        
        print("\nOverall Statistics:")
        print('Initial Investment:', self.params.initial_investment)
        print(f'Final Balance:      {self.balance:.4f}')
        print(f"Balance % change:   {balance_change:.4f}% ***")
        print(f"Baseline % change:  {baseline_change:.4f}%")
        print('Number of Trades Executed:', self.trades_executed)
        print(f"\nTotal Profit/Loss (including fees): ${total_profit_loss:.4f}")
        print(f"  Total Profit: ${total_profit:.4f}")
        print(f"  Total Loss:   ${total_loss:.4f}")
        print(f"Total Transaction Costs: ${total_transaction_costs:.4f}")
        print(f"\nBorrow Fees: ${total_stock_borrow_costs:.4f}")
        print(f"Average Profit/Loss per Trade (including fees): ${mean_profit_loss:.4f}")
        print(f"Average Profit/Loss Percentage (ROE) per Trade (including fees): {mean_profit_loss_pct:.4f}%")
        print(f"  for winning trades: {win_mean_profit_loss_pct:.4f}%")
        print(f"    longs:  {long_win_mean_profit_loss_pct:.4f}%")
        print(f"    shorts: {short_win_mean_profit_loss_pct:.4f}%")
        print(f"  for losing trades:  {lose_mean_profit_loss_pct:.4f}%")
        print(f"    longs:  {long_lose_mean_profit_loss_pct:.4f}%")
        print(f"    shorts: {short_lose_mean_profit_loss_pct:.4f}%")
        print(f"Number of Winning Trades: {win_trades} ({win_longs} long, {win_shorts} short)")
        print(f"Number of Losing Trades:  {lose_trades} ({lose_longs} long, {lose_shorts} short)")
        
        
        print(f"Win Rate: {win_trades / len(self.trades) * 100:.4f}%" if self.trades else "Win Rate: N/A")
        print(f"\nMargin Usage:")
        print(f"Margin Enabled: {'Yes' if self.params.use_margin else 'No'}")
        print(f"Max Buying Power: {self.params.times_buying_power}x")
        print(f"Average Margin Multiplier: {sum(trade.actual_margin_multiplier for trade in self.trades) / len(self.trades):.4f}x")
        print(f"Average Sub Positions per Position: {avg_sub_pos:.4f}")
        print(f"Average Transactions per Position: {avg_transact:.4f}")
        
        # # print(trades)
        if self.export_trades_path:
            export_trades_to_csv(self.trades, self.export_trades_path)

        plot_cumulative_pnl_and_price(self.trades, self.bars, self.params.initial_investment)

        return self.balance, sum(1 for trade in self.trades if trade.is_long), sum(1 for trade in self.trades if not trade.is_long), balance_change, mean_profit_loss_pct, win_mean_profit_loss_pct, lose_mean_profit_loss_pct, \
            win_trades / len(self.trades) * 100,  \
            total_transaction_costs, avg_sub_pos, avg_transact, self.count_entry_adjust, self.count_entry_skip, self.count_exit_adjust, self.count_exit_skip



        
# # Usage
# params = StrategyParameters(
#     initial_investment=10_000,
#     do_longs=True,
#     do_shorts=True,
#     use_margin=True,
#     times_buying_power=4,
#     min_stop_dist_relative_change_for_partial=0,
#     soft_start_time=None,
#     soft_end_time=time(15, 50)
# )

# strategy = TradingStrategy(TouchDetectionAreas.from_dict(touch_detection_areas), params)
# results = strategy.run_backtest()