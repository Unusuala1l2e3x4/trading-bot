from copy import deepcopy
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Set
from numba import jit
import numpy as np
from datetime import datetime, time
import pandas as pd
import math
from TouchDetection import TouchDetectionAreas, plot_touch_detection_areas
from TouchArea import TouchArea, TouchAreaCollection
from TradePosition2 import TradePosition, export_trades_to_csv, plot_cumulative_pl_and_price

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
from alpaca.trading.enums import OrderSide
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment

import logging

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
        print(f"{type(e).__qualname__} while checking marginability for {symbol}: {e}")
        return False


@dataclass
class IntendedOrder:
    action: str
    side: OrderSide
    symbol: str
    qty: int
    price: float
    position: TradePosition
    fees: float

    def __copy__(self):
        return IntendedOrder(
            action=self.action,
            side=self.side,
            symbol=self.symbol,
            qty=self.qty,
            price=self.price,
            position=self.position,  # Passing TradePosition by reference
            fees=self.fees
        )

from TradingStrategyParameters import *


def filter_df_by_timerange(df: pd.DataFrame, start_time: datetime, end_time: datetime) -> pd.DataFrame:
    """
    Filter DataFrame to only include rows between start_time and end_time (inclusive).
    Uses DatetimeIndex's searchsorted for efficient index searching.
    
    Args:
        df: DataFrame with MultiIndex including 'timestamp'
        start_time: Start of time range (inclusive)
        end_time: End of time range (inclusive)
    
    Returns:
        Filtered DataFrame
    """
    timestamps = df.index.get_level_values('timestamp')
    
    # Use DatetimeIndex's native searchsorted method
    left_pos = timestamps.searchsorted(start_time, side='left')
    right_pos = timestamps.searchsorted(end_time, side='right')
    
    # Return filtered DataFrame
    return df.iloc[left_pos:right_pos]


@dataclass
class TradingStrategy:
    touch_detection_areas: TouchDetectionAreas
    params: StrategyParameters
    export_trades_path: Optional[str] = None
    export_graph_path: Optional[str] = None
    is_live_trading: bool = False

    # Fields with default values
    all_bars: pd.DataFrame = field(init=False)
    logger: logging.Logger = field(init=False)
    balance: float = field(init=False)
    open_positions: Set[TradePosition] = field(default_factory=set)
    trades: List[TradePosition] = field(default_factory=list)
    terminated_area_ids: Dict[pd.Timestamp, List[int]] = field(default_factory=dict)
    trades_executed: int = 0
    simultaneous_close_open: int = 0
    is_marginable: bool = field(init=False)
    is_etb: bool = field(init=False)
    next_position_id: int = 0
    count_entry_adjust: int = 0
    count_exit_adjust: int = 0
    count_entry_skip: int = 0
    count_exit_skip: int = 0
    day_accrued_fees: float = 0
    current_date: pd.Timestamp = None
    market_open: pd.Timestamp = field(init=False)
    market_close: pd.Timestamp = field(init=False)
    day_start_time: pd.Timestamp = field(init=False)
    day_end_time: pd.Timestamp = field(init=False)
    now_time: pd.Timestamp = field(init=False)
    current_timestamp: pd.Timestamp = field(init=False)
    lookback_before_latest_quote: timedelta = timedelta(seconds=1)
    latest_quote_time: pd.Timestamp = field(init=False)
    day_soft_start_time: pd.Timestamp = field(init=False)
    daily_bars: pd.DataFrame = field(init=False)
    daily_quotes_raw: pd.DataFrame = field(init=False)
    daily_quotes_agg: pd.DataFrame = field(init=False)
    daily_quotes_raw_indices: pd.Series = field(init=False)
    daily_quotes_agg_indices: pd.Series = field(init=False)
    now_quotes_raw: pd.DataFrame = field(init=False)
    now_quotes_agg: pd.DataFrame = field(init=False)
    now_bid_price: float = field(init=False)
    now_ask_price: float = field(init=False)
    daily_bars_index: int = field(init=False)
    soft_end_triggered: bool = False
    touch_area_collection: TouchAreaCollection = field(init=False)
    
    # record stats
    pressure_scalars: List[float] = field(default_factory=list)
    spread_scalars: List[float] = field(default_factory=list)
    final_scalars: List[float] = field(default_factory=list)
    max_trade_size_adjustments: List[float] = field(default_factory=list)
    max_trade_sizes: List[float] = field(default_factory=list)
    trade_size_reductions: List[float] = field(default_factory=list)
    
    
    def __post_init__(self):
        self.all_bars = self.touch_detection_areas.bars[self.touch_detection_areas.mask].sort_index(level='timestamp')
        # self.logger = self.setup_logger(logging.INFO)
        self.logger = self.setup_logger(logging.WARNING)
        self.initialize_strategy()

    def setup_logger(self, log_level=logging.INFO):
        logger = logging.getLogger('TradingStrategy')
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

    def log(self, message, level=logging.INFO):
        self.logger.log(level, message, exc_info=level >= logging.ERROR)
        
    def initialize_strategy(self):
        self.balance = self.params.initial_investment

        self.is_marginable = is_security_marginable(self.touch_detection_areas.symbol) 
        self.is_etb = is_security_shortable_and_etb(self.touch_detection_areas.symbol)
        
        print(f'{self.touch_detection_areas.symbol} is {'NOT ' if not self.is_marginable else ''}marginable.')
        print(f'{self.touch_detection_areas.symbol} is {'NOT ' if not self.is_etb else ''}shortable and ETB.')
        
        self.initialize_touch_areas()

    def initialize_touch_areas(self):
        all_touch_areas = []
        if self.params.do_longs or self.params.sim_longs:
            all_touch_areas.extend(self.touch_detection_areas.long_touch_area)
        if self.params.do_shorts or self.params.sim_shorts:
            all_touch_areas.extend(self.touch_detection_areas.short_touch_area)
        # print(f'{len(all_touch_areas)} touch areas in TouchAreaCollection ({len(self.touch_detection_areas.long_touch_area)} long, {len(self.touch_detection_areas.short_touch_area)} short)')
        self.touch_area_collection = TouchAreaCollection(all_touch_areas, self.touch_detection_areas.min_touches)

    def update_strategy(self, touch_detection_areas: TouchDetectionAreas):
        self.touch_detection_areas = touch_detection_areas
        self.all_bars = self.touch_detection_areas.bars[self.touch_detection_areas.mask].sort_index(level='timestamp')
        self.initialize_touch_areas()

    @property
    def buying_power(self):
        if self.params.use_margin and self.is_marginable:
            # Use 25% initial margin requirement for marginable securities (intraday)
            initial_margin_requirement = 0.25  # Intraday margin requirement
            # initial_margin_requirement = 0.5   # Uncomment for overnight margin requirement if needed
        else:
            # Use 100% initial margin requirement for non-marginable securities or cash accounts
            initial_margin_requirement = 1.0
        # Calculate max leverage based on initial margin requirement
        max_leverage = 1.0 / initial_margin_requirement
        
        # actual_margin_multiplier = min(self.params.times_buying_power, max_leverage)
        # return self.total_equity * actual_margin_multiplier - self.total_market_value
    
        return self.total_equity * max_leverage - self.total_market_value  # Assuming 4x margin
    

    
    
    def get_account_summary(self):
        return {
            "Cash Balance": self.balance,
            "Total Market Value": self.total_market_value,
            "Margin Used": self.margin_used,
            "Total Equity": self.total_equity,
            "Buying Power": self.buying_power
        }

        
    def rebalance(self, is_simulated: bool, cash_change: float, current_price: float = None):
        if not is_simulated:
            old_balance = self.balance
            new_balance = self.balance + cash_change
            assert new_balance >= 0, f"Negative balance encountered: {new_balance:.4f} ({old_balance:.4f} {cash_change:.4f})"
            self.balance = new_balance

        # Assert and debug printing logic here (similar to your original function)
        
    @property
    def total_market_value(self):
        return sum(abs(position.market_value) for position in self.open_positions)

    @property
    def margin_used(self):
        return sum(
            abs(position.market_value) * (1 - 1/position.times_buying_power)
            for position in self.open_positions
            if position.use_margin
        )

    @property
    def total_cash_committed(self):
        return self.total_market_value - self.margin_used
    
    @property
    def total_equity(self):
        return self.balance + self.total_market_value - self.margin_used


    def update_market_values(self, current_price: float):
        for position in self.open_positions:
            position.update_market_value(current_price)
        

    def exit_action(self, position: TradePosition):
        # Logic for handling position exit (similar to your original function)
        self.trades.append(position) # append position
        self.touch_area_collection.del_open_position_area(position.area)
        # del self.open_positions[position]
        self.open_positions.remove(position)
        
    def close_all_positions(self, current_timestamp: datetime, price_at_action: float, vwap: float, volume: float, avg_volume: float) -> Tuple[List[IntendedOrder], Set[TradePosition]]:
        # Logic for closing all positions (similar to your original function)
        orders = []
        positions_to_remove = set()

        for position in self.open_positions:
            price_at_action = self.get_price_at_action(position.is_long, False)
            remaining_shares = position.shares
            realized_pl, cash_released, fees_expected, qty_intended = position.partial_exit(current_timestamp, price_at_action, position.shares, vwap, volume, avg_volume, self.params.slippage.slippage_factor)
            self.log(f"    cash_released {cash_released:.4f}, realized_pl {realized_pl:.4f}, fees {fees_expected:.4f}",level=logging.INFO)
            assert qty_intended == remaining_shares, (qty_intended, remaining_shares)
            
            self.rebalance(position.is_simulated, (cash_released / position.times_buying_power) + realized_pl, price_at_action)
            if not position.is_simulated:
                self.day_accrued_fees += fees_expected
                
            position.close(current_timestamp, price_at_action)
            self.trades_executed += 1
            position.area.record_entry_exit(position.entry_time, position.entry_price, 
                                            current_timestamp, price_at_action)
            # position.area.terminate(self.touch_area_collection)
            self.touch_area_collection.terminate_area(position.area)
            
            orders.append(IntendedOrder(
                action = 'close',
                side = OrderSide.SELL if position.is_long else OrderSide.BUY,
                symbol = self.touch_detection_areas.symbol,
                qty = qty_intended,
                price = price_at_action,
                position = position,
                fees = fees_expected
            ))

            positions_to_remove.add(position)

        for position in positions_to_remove:
            self.exit_action(position)
            
        assert not self.open_positions, self.open_positions
        return orders, positions_to_remove # set([position.area.id for position in positions_to_remove])

    def calculate_position_details(self, is_long: bool, current_price: float, times_buying_power: float, 
                                avg_volume: float, avg_trade_count: float, volume: float,
                                trade_count_mult: Optional[float] = 1.0, existing_shares: Optional[int] = 0, 
                                target_shares: Optional[int] = None):
        # # Logic for calculating position details
        
        # TODO: make sure now_quotes_raw is filtered to before cutoff
        if not self.params.ordersizing.is_trading_allowed(self.total_equity, avg_trade_count, avg_volume, self.now_quotes_raw, 
                                                          self.latest_quote_time - self.lookback_before_latest_quote, self.now_time, trade_count_mult):
            return 0, 0, 0, 0, 0, 0, 0, 0

        
        # NOTE: micro_quote_count may need to be normalized to 1 second average count, or filtered to last second only
        
        
        # when live, need to call is_security_marginable
        # Determine initial margin requirement and calculate max leverage
        if self.params.use_margin and self.is_marginable:
            # Use 25% initial margin requirement for marginable securities (intraday)
            initial_margin_requirement = 0.25  # Intraday margin requirement
            # initial_margin_requirement = 0.5   # Uncomment for overnight margin requirement if needed
        else:
            # Use 100% initial margin requirement for non-marginable securities or cash accounts
            initial_margin_requirement = 1.0

        # Calculate max leverage based on initial margin requirement
        max_leverage = 1.0 / initial_margin_requirement
        # Apply the times_buying_power constraint
        actual_margin_multiplier = min(times_buying_power, max_leverage)
        
        current_shares = existing_shares
        max_trade_size_by_volume = self.params.ordersizing.calculate_max_trade_size(avg_volume)
        
        max_trade_size, pressure_scaling, spread_scaling, final_scaling = self.params.ordersizing.adjust_max_trade_size(self.current_timestamp, max_trade_size_by_volume, self.now_quotes_raw, self.now_quotes_agg, is_long, 
                                                                        self.latest_quote_time - self.lookback_before_latest_quote, self.now_time)
                                                                        # self.now_time - self.lookback_before_latest_quote, self.now_time)
        
        self.pressure_scalars.append(pressure_scaling)
        self.spread_scalars.append(spread_scaling)
        self.final_scalars.append(final_scaling)
        self.max_trade_sizes.append(max_trade_size)
        
        # Adjust available balance based on current position
        available_balance = min(self.balance, self.params.max_investment) * actual_margin_multiplier

        # Calculate max additional shares based on available balance
        max_additional_shares_by_balance = math.floor(available_balance / current_price)
        
        if target_shares is not None:
            # Ensure target_shares is greater than current_shares for entries
            assert target_shares > current_shares, (target_shares, current_shares)
            shares_change_unadjusted = min(target_shares - current_shares, max_additional_shares_by_balance)
        else:
            shares_change_unadjusted = max_additional_shares_by_balance
        
        # shares_change = min(shares_change_unadjusted, max_trade_size - current_shares)
        shares_change = min(shares_change_unadjusted, max_trade_size)
        shares_change = max(0, shares_change)
        
        if max_trade_size != max_trade_size_by_volume:
            self.max_trade_size_adjustments.append(max_trade_size_by_volume - max_trade_size)
        
        if shares_change != shares_change_unadjusted:
            assert shares_change_unadjusted > shares_change
            self.trade_size_reductions.append(shares_change_unadjusted - shares_change)

        total_shares = current_shares + shares_change
        invest_amount = shares_change * current_price
        actual_cash_used = invest_amount / actual_margin_multiplier
        estimated_entry_cost = 0  # Set to 0 as we're no longer considering entry costs

        # Calculate minimum price movement
        next_lower_shares = shares_change - 1
            
        if shares_change > 1 and next_lower_shares > 0:
            price_for_next_lower = available_balance / next_lower_shares
            if is_long:
                min_price_movement = price_for_next_lower - current_price
            else:  # Short position
                min_price_movement = current_price - price_for_next_lower
        elif shares_change > 1 and next_lower_shares == 0:
            # When we can only buy one share, any price increase (for long) or decrease (for short) will make it unaffordable
            min_price_movement = available_balance - current_price if is_long else current_price - available_balance
        else:
            min_price_movement = 0  # No trade is possible at current price

        # Ensure min_price_movement is positive
        min_price_movement = abs(min_price_movement)

        return total_shares, actual_margin_multiplier, initial_margin_requirement, estimated_entry_cost, actual_cash_used, shares_change, invest_amount, min_price_movement


    def calculate_exit_details(self, is_long: bool, times_buying_power: float, shares_to_exit: int, volume: float, avg_volume: float, avg_trade_count: float, trade_count_mult: Optional[float] = 1.0):
        assert shares_to_exit > 0

        # TODO: make sure now_quotes_raw is filtered to before cutoff
        if not self.params.ordersizing.is_trading_allowed(self.total_equity, avg_trade_count, avg_volume, self.now_quotes_raw, 
                                                          self.latest_quote_time - self.lookback_before_latest_quote, self.now_time, trade_count_mult):
            return 0
        
        # when live, need to call is_security_marginable
        # Determine initial margin requirement and calculate max leverage
        if self.params.use_margin and self.is_marginable:
            # Use 25% initial margin requirement for marginable securities (intraday)
            initial_margin_requirement = 0.25  # Intraday margin requirement
            # initial_margin_requirement = 0.5   # Uncomment for overnight margin requirement if needed
        else:
            # Use 100% initial margin requirement for non-marginable securities or cash accounts
            initial_margin_requirement = 1.0

        # Calculate max leverage based on initial margin requirement
        max_leverage = 1.0 / initial_margin_requirement
        # Apply the times_buying_power constraint
        actual_margin_multiplier = min(times_buying_power, max_leverage)
        
        # Calculate max shares that can be exited based on volume
        max_trade_size_by_volume = self.params.ordersizing.calculate_max_trade_size(avg_volume)
        
        max_trade_size, pressure_scaling, spread_scaling, final_scaling = self.params.ordersizing.adjust_max_trade_size(self.current_timestamp, max_trade_size_by_volume, self.now_quotes_raw, self.now_quotes_agg, not is_long, 
                                                                        self.latest_quote_time - self.lookback_before_latest_quote, self.now_time)
                                                                        # self.now_time - self.lookback_before_latest_quote, self.now_time)
        
        self.pressure_scalars.append(pressure_scaling)
        self.spread_scalars.append(spread_scaling)
        self.final_scalars.append(final_scaling)
        self.max_trade_sizes.append(max_trade_size)
                                                                        
        shares_change = min(shares_to_exit, max_trade_size)
        shares_change = max(0, shares_change)
        
        if max_trade_size != max_trade_size_by_volume:
            self.max_trade_size_adjustments.append(max_trade_size_by_volume - max_trade_size)
            
        if shares_change != shares_to_exit:
            assert shares_to_exit > shares_change
            self.trade_size_reductions.append(shares_to_exit - shares_change)
            
        return shares_change
    
    
    def create_new_position(self, area: TouchArea, current_timestamp: datetime, now_bar, prev_close: float, pending_orders: List[IntendedOrder]) -> List[IntendedOrder]:
        # Logic for placing a stop market buy order (similar to your original function)
        # ['open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap', 'central_value', 'is_res', 'shares_per_trade', 
        # 'avg_volume', 'avg_trade_count', 'log_return', 'volatility', 'time', 'date']
        open_price, high_price, low_price, close_price, volume, trade_count, vwap, avg_volume, avg_trade_count = \
            now_bar.open, now_bar.high, now_bar.low, now_bar.close, now_bar.volume, now_bar.trade_count, now_bar.vwap, now_bar.avg_volume, now_bar.avg_trade_count
        
        if self.open_positions or self.balance <= 0:
            return NO_POSITION_OPENED

        # debug_print(f"Attempting order: {'Long' if area.is_long else 'Short'} at {area.get_buy_price:.4f}")
        # debug_print(f"  Balance: {balance:.4f}, Total Account Value: {total_equity:.4f}")
        
        area.update_bounds(current_timestamp)

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

        # price_at_action = area.get_buy_price # Stop buy (placed at time of min_touches) would have executed
        # price_at_action = np.mean([area.get_buy_price, close_price]) # balanced approach, may account for slippage
        # price_at_action = close_price # if not using stop buys
        # debug3_print(f"Execution price: {price_at_action:.4f}")
        price_at_action = self.get_price_at_action(area.is_long, True)
        

        # Calculate position size, etc...
        total_shares, actual_margin_multiplier, initial_margin_requirement, estimated_entry_cost, actual_cash_used, shares_change, invest_amount, min_price_movement \
            = self.calculate_position_details(
                area.is_long, price_at_action, self.params.times_buying_power, avg_volume, avg_trade_count, volume
            )
        assert total_shares == shares_change
        
        if shares_change == 0:
            return NO_POSITION_OPENED
        
        if actual_cash_used + estimated_entry_cost > self.balance:
            return NO_POSITION_OPENED
        
        # Create the position
        position = TradePosition(
            date=current_timestamp.date(),
            id=self.next_position_id,
            area=area,
            is_long=area.is_long,
            entry_time=current_timestamp,
            initial_balance=actual_cash_used,
            initial_shares=total_shares,
            entry_price=price_at_action,
            bar_price_at_entry=close_price,
            use_margin=self.params.use_margin,
            is_marginable=self.is_marginable, # when live, need to call is_security_marginable
            times_buying_power=actual_margin_multiplier,
            
            # # Use price_at_action for initial peak-stop
            # current_stop_price=price_at_action - area.get_range if area.is_long else price_at_action + area.get_range,
            # max_price=price_at_action if area.is_long else None,
            # min_price=price_at_action if not area.is_long else None
            
            # Use H-L prices for initial peak-stop (seems to have better results - certain trades exit earlier)
            current_stop_price=high_price - area.get_range if area.is_long else low_price + area.get_range,
            max_price=high_price if area.is_long else None,
            min_price=low_price if not area.is_long else None,
            
            # TODO: consider using high prices and low prices determined by macro quotes data, thus affecting initial current_stop_price, max_price, and min_price.

        )
        
        # print(list(now_bar.index))
        # print(list(now_bar.values))
        # print(now_bar.iat[0], now_bar.iat[1])
    
        if (area.is_long and self.params.do_longs) or (not area.is_long and self.params.do_shorts and self.is_etb):  # if conditions not met, simulate position only.
            position.is_simulated = False
        else:
            position.is_simulated = True

        cash_needed, fees_expected, qty_intended = position.initial_entry(vwap, volume, avg_volume, self.params.slippage.slippage_factor)
        self.log(f"    cash_needed {cash_needed:.4f}, fees {fees_expected:.4f}\t\tarea.get_buy_price={area.get_buy_price:.4f}",level=logging.INFO)
        assert cash_needed == invest_amount, (cash_needed, invest_amount)
        assert cash_needed == actual_cash_used * actual_margin_multiplier, (cash_needed, actual_cash_used * actual_margin_multiplier)
        assert actual_margin_multiplier == position.times_buying_power, (actual_margin_multiplier, position.times_buying_power)

        assert qty_intended == position.shares, (qty_intended, position.shares)
        
        self.next_position_id += 1
        
        # Add to open positions (regardless if real or simulated)
        self.open_positions.add(position)
        self.touch_area_collection.add_open_position_area(area)
        
        self.rebalance(position.is_simulated, -(cash_needed / position.times_buying_power), close_price)
        if not position.is_simulated:
            self.day_accrued_fees += fees_expected
        
        # return POSITION_OPENED
        return [IntendedOrder(
            action = 'open',
            side = OrderSide.BUY if area.is_long else OrderSide.SELL,
            symbol = self.touch_detection_areas.symbol,
            qty = qty_intended,
            price = position.entry_price, # = price_at_action
            position = position,
            fees = fees_expected
        )]


    def update_positions(self, current_timestamp: datetime, now_bar: pd.Series) -> Tuple[List[IntendedOrder], Set[TradePosition]]:
        # Logic for updating positions (similar to your original function)
        # ['open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap', 'central_value', 'is_res', 'shares_per_trade', 
        # 'avg_volume', 'avg_trade_count', 'log_return', 'volatility', 'time', 'date']
        open_price, high_price, low_price, close_price, volume, trade_count, vwap, avg_volume, avg_trade_count = \
            now_bar.open, now_bar.high, now_bar.low, now_bar.close, now_bar.volume, now_bar.trade_count, now_bar.vwap, now_bar.avg_volume, now_bar.avg_trade_count
        
        positions_to_remove = set()

        # if using trailing stops, exit_price = None
        def perform_exit(position, exit_price=None):
            price = position.current_stop_price if exit_price is None else exit_price
            position.close(current_timestamp, price)
            self.trades_executed += 1
            position.area.record_entry_exit(position.entry_time, position.entry_price, 
                                            current_timestamp, price)
            # position.area.terminate(self.touch_area_collection)
            self.touch_area_collection.terminate_area(position.area)
            positions_to_remove.add(position)
            

        def calculate_target_shares(position: TradePosition, current_price):
            if position.is_long:
                price_movement = current_price - position.current_stop_price
            else:
                price_movement = position.current_stop_price - current_price
            target_pct = min(max(0, price_movement / position.area.get_range),  1.0)
            target_shares = math.floor(target_pct * position.max_shares)
            return target_shares
        
        def get_price_at_action_from_shares_diff(current_shares, target_shares, is_long):
            if target_shares < current_shares:
                return self.get_price_at_action(is_long, False)
            elif target_shares > current_shares:
                return self.get_price_at_action(is_long, True)
            else:
                return None

        orders = []
        
        for position in self.open_positions:
            price_at_action = None
            
            # OHLC logic for trailing stops
            # Initial tests found that just using close_price is more effective
            # Implies we aren't using trailing stop sells
            # UNLESS theres built-in functionality to wait until close
            
            # if not price_at_action:
            #     should_exit = position.update_stop_price(open_price, current_timestamp)
            #     target_shares = calculate_target_shares(position, open_price)
            #     if should_exit or target_shares == 0:
            #         perform_exit(area_id, position) # DO NOT pass price into function since order would have executed at current_stop_price.
            #         price_at_action = open_price
            
            # # If not stopped out at open, simulate intra-minute price movement
            # if not price_at_action:
            #     should_exit = position.update_stop_price(high_price, current_timestamp)
            #     target_shares = calculate_target_shares(position, high_price)
            #     if not position.is_long and (should_exit or target_shares == 0):
            #         # For short positions, the stop is crossed if high price increases past it
            #         perform_exit(area_id, position) # DO NOT pass price into function since order would have executed at current_stop_price.
            #         price_at_action = high_price
            
            # if not price_at_action:
            #     should_exit = position.update_stop_price(low_price, current_timestamp)
            #     target_shares = calculate_target_shares(position, low_price)
            #     if position.is_long and (should_exit or target_shares == 0):
            #         # For long positions, the stop is crossed if low price decreases past it
            #         perform_exit(area_id, position) # DO NOT pass price into function since order would have executed at current_stop_price.
            #         price_at_action = low_price
            
            
            if not price_at_action:
                should_exit, should_exit_2 = position.update_stop_price(close_price, None, current_timestamp) # NOTE: continue using bar price here
                target_shares = calculate_target_shares(position, close_price) # NOTE: continue using bar price here
                
                # print(target_shares, position.max_shares, should_exit)
                if should_exit or target_shares == 0:
                    assert target_shares == 0, target_shares
                    # price_at_action = close_price
                    price_at_action = get_price_at_action_from_shares_diff(position.shares, target_shares, position.is_long)
                    
                    # if using stop market order safeguard, use this:
                    # price_at_action = position.current_stop_price_2 if should_exit_2 else close_price
                    
                    # current_stop_price_2 is the stop market order price
                    # stop market order would have executed before the minute is up, if should_exit_2 is True
                    # worry about this in LiveTrader later, after close price logic is implemented
                    # must use TradingStream that pings frequently.
                    
                
            if price_at_action:
                assert target_shares == 0, target_shares
            
            if not price_at_action:
                # price_at_action = close_price
                price_at_action = get_price_at_action_from_shares_diff(position.shares, target_shares, position.is_long)
            
            # Partial exit and entry logic
            assert target_shares <= position.max_shares, (target_shares, position.max_shares)

            target_pct = target_shares / position.max_shares
            current_pct = min( 1.0, position.shares / position.max_shares)
            assert 0.0 <= target_pct <= 1.0, target_pct
            assert 0.0 <= current_pct <= 1.0, current_pct


            # To prevent over-trading, skip partial buy/sell if difference between target and current shares percentage is less than threshold
            # BUT only if not increasing/decrease to/from 100%
            # Initial tests found that a threshold closer to 0 or 1, not in between, gives better results
            if abs(target_pct - current_pct) < self.params.min_stop_dist_relative_change_for_partial:
                continue
            
            if target_shares < position.shares:
                shares_to_adjust = position.shares - target_shares
                if shares_to_adjust > 0:

                    shares_change = self.calculate_exit_details(
                        position.is_long,
                        position.times_buying_power,
                        shares_to_adjust,
                        volume,
                        avg_volume,
                        avg_trade_count,
                        # trade_count_mult=(shares_to_adjust / position.max_shares)
                    )
                    
                    if shares_change > 0:
                        realized_pl, cash_released, fees_expected, qty_intended = position.partial_exit(current_timestamp, price_at_action, shares_change, vwap, volume, avg_volume, self.params.slippage.slippage_factor)
                        self.log(f"    cash_released {cash_released:.4f}, realized_pl {realized_pl:.4f}, fees {fees_expected:.4f}",level=logging.INFO)
                        assert qty_intended == shares_change, (qty_intended, shares_change)
                        
                        self.rebalance(position.is_simulated, (cash_released / position.times_buying_power) + realized_pl, price_at_action)
                        if not position.is_simulated:
                            self.day_accrued_fees += fees_expected
                        
                        orders.append(IntendedOrder(
                            action = 'close' if position.shares == 0 else 'partial_exit',
                            side = OrderSide.SELL if position.is_long else OrderSide.BUY,
                            symbol = self.touch_detection_areas.symbol,
                            qty = qty_intended,
                            price = price_at_action,
                            position = position,
                            fees = fees_expected
                        ))
                        if position.shares == 0:
                            perform_exit(position, price_at_action)

                        if shares_change < shares_to_adjust:

                            self.count_exit_adjust += 1
                    else:
                        self.count_exit_skip += 1
                        
            elif target_shares > position.shares:
                shares_to_adjust = target_shares - position.shares
                if shares_to_adjust > 0:

                    existing_shares = position.shares
                    total_shares, actual_margin_multiplier, initial_margin_requirement, estimated_entry_cost, actual_cash_used, shares_change, invest_amount, min_price_movement = self.calculate_position_details(
                        position.area.is_long, price_at_action, position.times_buying_power, avg_volume, avg_trade_count, volume,
                        # trade_count_mult=(shares_to_adjust / position.max_shares), 
                        existing_shares=existing_shares, target_shares=target_shares
                    )
                    
                    shares_to_buy = min(shares_to_adjust, shares_change)
                    
                    if shares_to_buy > 0:
                        if shares_to_buy < shares_to_adjust:
                            self.count_entry_adjust += 1
                            
                        if not self.soft_end_triggered:
                            cash_needed, fees_expected, qty_intended = position.partial_entry(current_timestamp, price_at_action, shares_to_buy, vwap, volume, avg_volume, self.params.slippage.slippage_factor)
                            self.log(f"    cash_needed {cash_needed:.4f}, fees {fees_expected:.4f}",level=logging.INFO)
                            assert cash_needed == invest_amount, (cash_needed, invest_amount)
                            assert cash_needed == actual_cash_used * actual_margin_multiplier, (cash_needed, actual_cash_used * actual_margin_multiplier)
                            assert actual_margin_multiplier == position.times_buying_power, (actual_margin_multiplier, position.times_buying_power)
                            assert qty_intended == shares_to_buy, (qty_intended, shares_to_buy)
                            
                            self.rebalance(position.is_simulated, -(cash_needed / position.times_buying_power), price_at_action)
                            if not position.is_simulated:
                                self.day_accrued_fees += fees_expected
                            
                            orders.append(IntendedOrder(
                                action = 'partial_entry',
                                side = OrderSide.BUY if position.is_long else OrderSide.SELL,
                                symbol = self.touch_detection_areas.symbol,
                                qty = qty_intended,
                                price = price_at_action,
                                position = position,
                                fees = fees_expected
                            ))
                            
                            position.increase_max_shares(position.shares) # Entry possible. Increase max_shares.
                            assert position.shares == total_shares, (position.shares, total_shares)
                            
                        else:
                            position.increase_max_shares(position.shares + shares_to_buy) # Entry possible but soft end triggered. Increase max_shares regardless.
                            assert position.shares + shares_to_buy == total_shares, (position.shares, shares_to_buy, total_shares)
      
                    else:
                        self.count_entry_skip += 1
                        position.decrease_max_shares(position.shares) # Entry not possible. Decrease max_shares.
                        
        for position in positions_to_remove:
            self.exit_action(position)

        return orders, positions_to_remove # set([position.area.id for position in positions_to_remove])

    def update_daily_parameters(self, current_date):
        self.market_open, self.market_close = self.touch_detection_areas.market_hours.get(current_date, (None, None))
        if self.market_open and self.market_close:
            self.day_start_time, self.day_end_time, self.day_soft_start_time = self.calculate_day_times(current_date, self.market_open, self.market_close)
        else:
            self.day_start_time = self.day_end_time = self.day_soft_start_time = None

    def update_balance(self, new_balance):
        # if abs(self.balance - new_balance) > 0.01:  # Check if difference is more than 1 cent; not sure if necessary to have this check
        if self.balance != new_balance:
            self.log(f"Updating balance from {self.balance:.2f} to {new_balance:.2f}")
        self.balance = new_balance
        
        
    def get_quote_indices(self, all_data: pd.DataFrame, indices: pd.DatetimeIndex, seconds_offset=0) -> Tuple[pd.DataFrame, np.ndarray]:
        # Extract the timestamps from all_data
        start_date = pd.Timestamp(self.current_date, tz=ny_tz)
        end_date = start_date + pd.Timedelta(days=1)

        # Slice the daily data for the current date
        daily = filter_df_by_timerange(all_data, start_date, end_date) # all_data.iloc[start_pos:end_pos]
        daily_timestamps = daily.index.get_level_values('timestamp')

        # Adjust the indices by the seconds_offset
        adjusted_indices = (indices + pd.Timedelta(seconds=seconds_offset))

        # Use searchsorted to find positions without converting to NumPy
        positions = daily_timestamps.searchsorted(adjusted_indices, side='left')

        # Handle positions that are out of bounds
        positions = positions.clip(0, len(daily) - 1)
        return daily, positions

    def handle_new_trading_day(self, current_timestamp):
        # self.log(f"handle_new_trading_day start", level=logging.WARNING)
        self.current_date = current_timestamp.date()
        self.update_daily_parameters(self.current_date)
        self.next_position_id = 0
        self.soft_end_triggered = False
        self.log(f"Starting balance on {self.current_date}: {self.balance}", level=logging.INFO)

        if self.is_live_trading: # handled in FUNCTION: update_strategy
            self.daily_bars = self.all_bars  # In live trading, all data is "daily data"
            self.daily_bars_index = len(self.daily_bars) - 1  # Current index is always the last one in live trading
            # self.daily_bars_index = len(self.all_bars) - 1
            # daily_bars_minutes = self.all_bars.index.get_level_values('timestamp').tz_convert(ny_tz)
            # daily_bars_minutes = self.daily_bars.index.get_level_values('timestamp').tz_convert(ny_tz)
            # pass
            
            self.daily_quotes_raw = self.touch_detection_areas.quotes_raw
            self.daily_quotes_agg = self.touch_detection_areas.quotes_agg
        else:
            # Use searchsorted to find start and end positions for daily_bars
            start_date = pd.Timestamp(self.current_date, tz=ny_tz)
            end_date = start_date + pd.Timedelta(days=1)

            # Filter the data for the current trading day based on timestamp
            self.daily_bars = filter_df_by_timerange(self.all_bars, start_date, end_date) # self.all_bars.iloc[start_pos:end_pos]
            self.daily_bars_index = 1  # Start from index 1 (exclude opening minute)
            # daily_bars_minutes = self.daily_bars.index.get_level_values('timestamp').tz_convert(ny_tz)
            
            daily_bars_minutes = self.daily_bars.index.get_level_values('timestamp').tz_convert(ny_tz)
            assert len(pd.unique(daily_bars_minutes.date)) == 1, pd.unique(daily_bars_minutes.date)

            # For raw quotes with an offset
            self.daily_quotes_raw, self.daily_quotes_raw_indices = self.get_quote_indices(
                # self.touch_detection_areas.quotes_raw, daily_bars_minutes, seconds_offset = -30
                self.touch_detection_areas.quotes_raw, daily_bars_minutes, seconds_offset = -59 # NOTE: -59 respects raw quote data boundary in each minute
            )
            # For aggregated quotes without an offset
            self.daily_quotes_agg, self.daily_quotes_agg_indices = self.get_quote_indices(
                self.touch_detection_areas.quotes_agg, daily_bars_minutes, seconds_offset = -59
            )
            assert not self.open_positions, self.open_positions # intraday only. should not have any open positions held overnight from previous day

    def get_quotes_raw(self, current_timestamp: datetime) -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
        # NOTE: MAKE SURE self.handle_new_trading_day got the matching raw quotes data 
        assert self.touch_detection_areas.quotes_raw is not None
        assert self.daily_quotes_raw is not None
        assert self.daily_bars_index is not None
            
        if self.is_live_trading:
            # raise NotImplementedError('get_quotes_raw not implemented for live trading')

            self.now_time = datetime.now()
            delay = self.now_time - current_timestamp
            assert timedelta(0) <= delay < timedelta(seconds=1), f"Delay of {delay} after {current_timestamp}. In normal circumstances, delay should be < 1 second."
            self.latest_quote_time = self.daily_quotes_raw.index.get_level_values('timestamp')[-1]
            self.now_bid_price = self.daily_quotes_raw.iloc[-1]['bid_price_last'].item()
            self.now_ask_price = self.daily_quotes_raw.iloc[-1]['ask_price_last'].item()
            return filter_df_by_timerange(self.daily_quotes_raw, self.latest_quote_time - self.lookback_before_latest_quote, self.now_time)
        else:
            if self.daily_bars_index >= len(self.daily_quotes_raw_indices)-1:
                return None
            
            ret = self.daily_quotes_raw.iloc[self.daily_quotes_raw_indices[self.daily_bars_index] : self.daily_quotes_raw_indices[self.daily_bars_index + 1]]
            dqr_timestamps = ret.index.get_level_values('timestamp')
            
            # NOTE: assuming a constant delay when backtesting
            self.now_time = current_timestamp + timedelta(seconds=0.4)
            # self.now_time = current_timestamp
            pos = dqr_timestamps.searchsorted(self.now_time, side='right')  # Use 'right' to include the cutoff
            if pos == 0:
                self.latest_quote_time = None  # Or some default value indicating no valid timestamp
                return ret.iloc[:0]
            
            self.latest_quote_time = dqr_timestamps[pos-1] # -1 since side='right'
            self.now_bid_price = ret.iloc[pos-1]['bid_price_last'].item()
            self.now_ask_price = ret.iloc[pos-1]['ask_price_last'].item()
            return filter_df_by_timerange(ret.iloc[:pos], self.latest_quote_time - self.lookback_before_latest_quote, self.now_time)
        
    def get_quotes_agg(self, current_timestamp: datetime) -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
        # NOTE: MAKE SURE self.handle_new_trading_day got the matching raw quotes data 
        assert self.touch_detection_areas.quotes_agg is not None
        assert self.daily_quotes_agg is not None
        assert self.daily_bars_index is not None
        
        if self.is_live_trading:
            # raise NotImplementedError('get_quotes_agg not implemented for live trading')
            ret = self.daily_quotes_agg
        else:
            # NOTE: excludes any delay after bar timestamp
            if self.daily_bars_index >= len(self.daily_quotes_agg_indices)-1:
                return None
            ret = self.daily_quotes_agg.iloc[self.daily_quotes_agg_indices[self.daily_bars_index] : self.daily_quotes_agg_indices[self.daily_bars_index + 1]]
            
        ret_latest = ret.index.get_level_values('timestamp')[-1]
        bars_latest = self.daily_bars.iloc[self.daily_bars_index].name[1]
        assert ret_latest <= bars_latest, (ret_latest, bars_latest)
        if ret_latest < bars_latest:
            self.log(f"Missing aggregated quotes data in last second before {current_timestamp.time()} bar. Latest at {ret_latest.time()}.",level=logging.INFO)
        return ret
    
    def get_price_at_action(self, is_long, is_entry):
        if (is_long and is_entry) or (not is_long and not is_entry): # buy
            return self.now_ask_price
        else: # sell
            return self.now_bid_price
            
        
    def run_backtest(self):
        assert not self.is_live_trading
        
        timestamps = self.all_bars.index.get_level_values('timestamp')
        
        # for i in tqdm(range(1, len(timestamps)), desc='run_backtest'):
        for i in range(1, len(timestamps)):
            current_timestamp = timestamps[i].tz_convert(ny_tz)
            self.current_timestamp = current_timestamp
            
            if self.current_date is None or current_timestamp.date() != self.current_date:
                self.handle_new_trading_day(current_timestamp)
            
            if not self.market_open or not self.market_close:
                continue
            
            if self.daily_bars.empty or len(self.daily_bars) < 2: 
                continue
            
            now_bar = self.daily_bars.iloc[self.daily_bars_index]

            # NOTE: quotes raw data should be passed into touch_detection_areas
            self.now_quotes_agg = self.get_quotes_agg(current_timestamp)
            self.now_quotes_raw = self.get_quotes_raw(current_timestamp)
            
            # print(self.now_bid_price, self.now_ask_price, now_bar.close)

            # print(current_timestamp)
            # print(now_bar)
            # print(self.now_quotes_raw.index.get_level_values('timestamp'))
            # print(self.now_quotes_agg.index.get_level_values('timestamp'))
            # print()

                
            if current_timestamp < now_bar.name[1]: # now_bar.name[1] is the timestamp in now_bar
                continue
            assert current_timestamp == now_bar.name[1], (current_timestamp, now_bar.name[1])
            prev_close = self.daily_bars.iloc[self.daily_bars_index - 1].close
            
            self.log(f"{current_timestamp.strftime("%H:%M")}, price {now_bar.close:.4f}, H-L {now_bar.high:.4f}-{now_bar.low:.4f}:", level=logging.INFO)

            if self.now_quotes_raw is not None and self.now_quotes_agg is not None and self.is_trading_time(current_timestamp, self.day_soft_start_time, self.day_end_time, self.daily_bars_index, self.daily_bars, i):
                if self.params.soft_end_time and not self.soft_end_triggered:
                    self.soft_end_triggered = self.check_soft_end_time(current_timestamp, self.current_date)
                
                self.touch_area_collection.get_active_areas(current_timestamp)
                update_orders, _ = self.update_positions(current_timestamp, now_bar)
                
                new_position_order = []
                if not self.soft_end_triggered:
                    new_position_order = self.process_active_areas(current_timestamp, now_bar, prev_close, update_orders)

                all_orders = update_orders + new_position_order
            elif self.should_close_all_positions(current_timestamp, self.day_end_time, i):
                self.touch_area_collection.get_active_areas(current_timestamp)
                all_orders, _ = self.close_all_positions(current_timestamp, now_bar.close, now_bar.vwap, now_bar.volume, now_bar.avg_volume)
                
                self.terminated_area_ids[self.current_date] = sorted([x.id for x in self.touch_area_collection.terminated_date_areas])
                self.log(f"    terminated areas on {self.touch_area_collection.active_date}: {self.terminated_area_ids[self.current_date]}", level=logging.INFO)
                
                # plot the used touch areas in the past day
                # plot_touch_detection_areas(self.touch_detection_areas, filter_date=self.current_date, filter_areas=self.terminated_area_ids)

            else:
                all_orders = []
                
            if all_orders:
                self.log(f"    Remaining ${self.balance:.4f}, committed ${self.total_cash_committed:.4f}, total equity ${self.total_equity:.4f} after {len(all_orders)} orders.", level=logging.INFO)
                self.log(f"    market value ${self.total_market_value:.4f}, margin used ${self.margin_used:.4f}, buying power ${self.buying_power:.4f}", level=logging.INFO)
                for a in all_orders:
                    peak = a.position.max_price if a.position.is_long else a.position.min_price
                    self.log(f"       {a.position.id} {a.action} {str(a.side).split('.')[1]} {int(a.qty)} * {a.price}, peak-stop {peak:.4f}-{a.position.current_stop_price:.4f}, {a.position.area}", 
                         level=logging.INFO)
                    
            if self.should_close_all_positions(current_timestamp, self.day_end_time, i) and self.day_accrued_fees != 0:
                # sum up transaction costs from the day and subtract it from balance
                self.rebalance(False, -self.day_accrued_fees)
                self.log(f"    Fees accrued on {self.current_date}: ${self.day_accrued_fees:.4f}", level=logging.INFO)
                self.log(f"    Remaining ${self.balance:.4f}, committed ${self.total_cash_committed:.4f}, total equity ${self.total_equity:.4f}.", level=logging.INFO)
                self.day_accrued_fees = 0
                
            self.daily_bars_index += 1
        
        if current_timestamp >= self.day_end_time:
            assert not self.open_positions, self.open_positions

        # self.log(f"Printing areas for {current_timestamp.date()}", level=logging.INFO)
        # TouchArea.print_areas_list(self.touch_area_collection.active_date_areas) # print if in log level
        # self.log(f"terminated areas on {self.touch_area_collection.active_date}: {self.terminated_area_ids[self.current_date]}", level=logging.INFO)
        # TouchArea.print_areas_list(self.touch_area_collection.terminated_date_areas) # print if in log level
        
        # plot_touch_detection_areas(self.touch_detection_areas, filter_areas=self.terminated_area_ids)
        
        
        
        return self.generate_backtest_results()

    def process_live_data(self, current_timestamp: datetime, new_timer_start: datetime = None) -> Tuple[List[IntendedOrder], Set[TradePosition]]:
        self.current_timestamp = current_timestamp
        
        assert self.is_live_trading
        # NOTE: quotes_raw is updated in LiveTrader
        assert self.touch_detection_areas.bars is not None
        assert self.touch_detection_areas.quotes_raw is not None
        assert self.touch_detection_areas.quotes_agg is not None
        
        try:
            if self.current_date is None or current_timestamp.date() != self.current_date:
                self.handle_new_trading_day(current_timestamp)
            
            if not self.market_open or not self.market_close:
                return [], set()
            
            if self.daily_bars.empty or len(self.daily_bars) < 2: 
                return [], set()
            
            assert self.daily_bars_index == len(self.daily_bars)-1
            now_bar = self.daily_bars.iloc[self.daily_bars_index]
            self.now_quotes_agg = self.get_quotes_agg(current_timestamp)
            self.now_quotes_raw = self.get_quotes_raw(current_timestamp)
            
            # check equality by memory location
            assert self.daily_bars is self.all_bars
            assert self.now_quotes_raw is self.daily_quotes_raw is self.touch_detection_areas.quotes_raw
            assert self.now_quotes_agg is self.daily_quotes_agg is self.touch_detection_areas.quotes_agg
            

            if current_timestamp < now_bar.name[1]: # < end time, just in case misaligned
                return [], set()
            elif current_timestamp > now_bar.name[1]: # > end time
                return [], set()
            assert current_timestamp == now_bar.name[1], (current_timestamp, now_bar.name[1])
            prev_close = self.daily_bars.iloc[self.daily_bars_index - 1].close
            
            positions_to_remove1, positions_to_remove2 = set(), set()

            if self.now_quotes_raw is not None and self.now_quotes_agg is not None and self.is_trading_time(current_timestamp, self.day_soft_start_time, self.day_end_time, None, None, None):
                if self.params.soft_end_time and not self.soft_end_triggered:
                    self.soft_end_triggered = self.check_soft_end_time(current_timestamp, self.current_date)

                self.touch_area_collection.get_active_areas(current_timestamp)
                for position in self.open_positions:
                    position.area = self.touch_area_collection.get_area(position.area) # find with hash/eq
                    assert position.area is not None
                
                update_orders, positions_to_remove1 = self.update_positions(current_timestamp, now_bar)
                
                new_position_order = []
                if not self.soft_end_triggered:
                    new_position_order = self.process_active_areas(current_timestamp, now_bar, prev_close, update_orders)
                
                all_orders = update_orders + new_position_order
            elif self.should_close_all_positions(current_timestamp, self.day_end_time, self.daily_bars_index):
                self.touch_area_collection.get_active_areas(current_timestamp)
                for position in self.open_positions:
                    position.area = self.touch_area_collection.get_area(position.area) # find with hash/eq
                    assert position.area is not None
                        
                all_orders, positions_to_remove2 = self.close_all_positions(current_timestamp, now_bar.close, now_bar.vwap, 
                                                    now_bar.volume, now_bar.avg_volume)
                    
            else:
                all_orders = []
                
            if all_orders:
                self.log(f"    Remaining ${self.balance:.4f}, committed ${self.total_cash_committed:.4f}, total equity ${self.total_equity:.4f} after {len(all_orders)} orders.", level=logging.INFO)
                self.log(f"    market value ${self.total_market_value:.4f}, margin used ${self.margin_used:.4f}, buying power ${self.buying_power:.4f}", level=logging.INFO)
                for a in all_orders:
                    peak = a.position.max_price if a.position.is_long else a.position.min_price
                    self.log(f"       {a.position.id} {a.action} {str(a.side).split('.')[1]} {int(a.qty)} * {a.price}, peak-stop {peak:.4f}-{a.position.current_stop_price:.4f}, {a.position.area}", 
                         level=logging.INFO)
                    
            if self.should_close_all_positions(current_timestamp, self.day_end_time, self.daily_bars_index) and self.day_accrued_fees != 0:
                # sum up transaction costs from the day and subtract it from balance
                self.rebalance(False, -self.day_accrued_fees)
                self.log(f"After ${self.day_accrued_fees:.4f} fees: ${self.balance:.4f}", level=logging.INFO)
                self.log(f"{current_timestamp.strftime("%H:%M")}: Remaining ${self.balance:.4f}, committed ${self.total_cash_committed:.4f}, total equity ${self.total_equity:.4f}.", level=logging.INFO)
                self.day_accrued_fees = 0
            
            # assert self.daily_bars_index == len(self.daily_bars) - 1
            # self.daily_bars_index = len(self.daily_bars) - 1  # Update daily_bars_index for live trading

            return all_orders, positions_to_remove1 | positions_to_remove2
            # if using stop market order safeguard, need to also modify existing stop market order (in LiveTrader)
            # remember to Limit consecutive stop order modifications to ~80 minutes (stop changing when close price has been monotonic in favorable direction for 80 or more minutes)
            

        except Exception as e:
            self.log(f"{type(e).__qualname__} in process_live_data at {current_timestamp}: {e}", logging.ERROR)
            raise Exception( e.args )
            

    def should_close_all_positions(self, current_timestamp: datetime, day_end_time: datetime, df_index: int) -> bool:
        if self.is_live_trading:
            return current_timestamp >= day_end_time
        else:
            return current_timestamp >= day_end_time \
                or df_index >= len(self.all_bars) - 1


    def calculate_day_times(self, current_date, market_open, market_close):
        date_obj = pd.Timestamp(current_date).tz_localize(ny_tz)
        
        day_start_time = date_obj.replace(hour=self.touch_detection_areas.start_time.hour, minute=self.touch_detection_areas.start_time.minute) if self.touch_detection_areas.start_time else market_open
        day_end_time = min(date_obj.replace(hour=self.touch_detection_areas.end_time.hour, minute=self.touch_detection_areas.end_time.minute), 
                           market_close - pd.Timedelta(minutes=3)) if self.touch_detection_areas.end_time else market_close - pd.Timedelta(minutes=3)
        
        if self.params.soft_start_time:
            day_soft_start_time = max(market_open, day_start_time, 
                                      date_obj.replace(hour=self.params.soft_start_time.hour, minute=self.params.soft_start_time.minute))
        else:
            day_soft_start_time = max(market_open, day_start_time)
        
        return day_start_time, day_end_time, day_soft_start_time

    def is_trading_time(self, current_timestamp: datetime, day_soft_start_time: datetime, day_end_time: datetime, daily_bars_index, daily_bars, i):
        if self.is_live_trading:
            return day_soft_start_time <= current_timestamp < day_end_time
        else:
            return day_soft_start_time <= current_timestamp < day_end_time \
                and daily_bars_index < len(daily_bars) - 1 \
                and i < len(self.all_bars) - 1

    def check_soft_end_time(self, current_timestamp, current_date):
        if self.params.soft_end_time:
            soft_end_time = pd.Timestamp.combine(current_date, self.params.soft_end_time).tz_localize(ny_tz)
            return current_timestamp >= soft_end_time
        return False

    def process_active_areas(self, current_timestamp: datetime, now_bar, prev_close, pending_orders: List[IntendedOrder]) -> List[IntendedOrder]:
        
        # # do not close then open at same time
        # if pending_orders:
        #     return []
        
        assert len(pending_orders) <= 1, len(pending_orders)
        pending_long_close = any([a.action == 'close' and a.position.is_long for a in pending_orders])
        pending_short_close = any([a.action == 'close' and not a.position.is_long for a in pending_orders])
        
        for area in self.touch_area_collection.active_date_areas:
            if area.min_touches_time is None or area.min_touches_time > current_timestamp:
                continue
            
            if self.balance <= 0:
                break
            if self.open_positions:  # ensure only 1 live position at a time
                break
            if ((area.is_long and (self.params.do_longs or self.params.sim_longs)) or 
                (not area.is_long and (self.params.do_shorts or self.params.sim_shorts))):
                
                
                # do not close long then open short (or close short then open long) at same time
                # reduces slippage
                # removes need to submit 2 separate orders
                if (area.is_long and pending_short_close) or (not area.is_long and pending_long_close):
                    continue
                
                # # do not close long then open long (or close short then open short) at same time
                # if area.is_long and pending_long_close:
                #     continue
                # if not area.is_long and pending_short_close:
                #     continue
                
                    
                new_position_order = self.create_new_position(area, current_timestamp, now_bar, prev_close, pending_orders)
                if new_position_order:
                    if pending_long_close or pending_short_close:
                        self.simultaneous_close_open += 1
                    return new_position_order
        return []

    def generate_backtest_results(self, trades: Optional[list[TradePosition]] = None):
        if trades is None:
            trades = self.trades
        trades = [a for a in trades if not a.is_simulated]
        
        # Calculate and return backtest results
        balance_change = ((self.balance - self.params.initial_investment) / self.params.initial_investment) * 100

        # Buy and hold strategy
        start_price = self.all_bars.iloc[0].close
        end_price = self.all_bars.iloc[-1].close
        baseline_change = ((end_price - start_price) / start_price) * 100
        
        total_pl = sum(trade.pl for trade in trades)
        
        total_profit = sum(trade.pl for trade in trades if trade.pl > 0)
        total_loss = sum(trade.pl for trade in trades if trade.pl < 0)
        
        total_transaction_costs = sum(trade.total_transaction_costs for trade in trades)
        total_stock_borrow_costs = sum(trade.total_stock_borrow_cost for trade in trades)

        mean_pl = np.mean([trade.pl for trade in trades])
        mean_plpc = np.mean([trade.plpc for trade in trades])

        win_mean_plpc = np.mean([trade.plpc for trade in trades if trade.pl > 0])
        lose_mean_plpc = np.mean([trade.plpc for trade in trades if trade.pl < 0])
        
        win_trades = sum(1 for trade in trades if trade.pl > 0)
        lose_trades = sum(1 for trade in trades if trade.pl < 0)
        win_longs = sum(1 for trade in trades if trade.is_long and trade.pl > 0)
        lose_longs = sum(1 for trade in trades if trade.is_long and trade.pl < 0)
        win_shorts = sum(1 for trade in trades if not trade.is_long and trade.pl > 0)
        lose_shorts = sum(1 for trade in trades if not trade.is_long and trade.pl < 0)
        avg_transact = np.mean([len(trade.transactions) for trade in trades])
        
        assert self.trades_executed == len(self.trades)

        # Print statistics
        print(f"END\nStrategy: {'Long' if self.params.do_longs else ''}{'&' if self.params.do_longs and self.params.do_shorts else ''}{'Short' if self.params.do_shorts else ''}")
        print(f'{self.touch_detection_areas.symbol} is {'NOT ' if not self.is_marginable else ''}marginable.')
        print(f'{self.touch_detection_areas.symbol} is {'NOT ' if not self.is_etb else ''}shortable and ETB.')
        timestamps = self.all_bars.index.get_level_values('timestamp')
        print(f"{timestamps[0]} -> {timestamps[-1]}")

        # debug2_print(all_bars['close'])
        
        print("\nOverall Statistics:")
        print('Initial Investment:', self.params.initial_investment)
        print(f'Final Balance:      {self.balance:.4f}')
        print(f"Balance % change:   {balance_change:.4f}% ***")
        print(f"Baseline % change:  {baseline_change:.4f}%")
        print('Number of Trades Executed:', self.trades_executed)
        print(f"Simultaneous close and open count: {self.simultaneous_close_open}")
        print(f"\nTotal Profit/Loss (after fees): ${total_pl:.4f}")
        print(f"  Total Profit: ${total_profit:.4f}")
        print(f"  Total Loss:   ${total_loss:.4f}")
        print(f"Total Transaction Costs: ${total_transaction_costs:.4f}")
        print(f"  Borrow Fees: ${total_stock_borrow_costs:.4f}")
        
        print(f"\nAverage Profit/Loss per Trade (after fees): ${mean_pl:.4f}")

        # Create Series for different trade categories
        trade_categories = {
            'All': [trade.plpc for trade in trades],
            # 'Long': [trade.plpc for trade in trades if trade.is_long],
            # 'Short': [trade.plpc for trade in trades if not trade.is_long],
            'Win': [trade.plpc for trade in trades if trade.pl > 0],
            'Lose': [trade.plpc for trade in trades if trade.pl <= 0],
            'Lwin': [trade.plpc for trade in trades if trade.is_long and trade.pl > 0],
            'Swin': [trade.plpc for trade in trades if not trade.is_long and trade.pl > 0],
            'Llose': [trade.plpc for trade in trades if trade.is_long and trade.pl <= 0],
            'Slose': [trade.plpc for trade in trades if not trade.is_long and trade.pl <= 0]
        }

        describe_results = pd.DataFrame({category: pd.Series(data).describe() for category, data in trade_categories.items()})
        describe_results = describe_results.transpose()
        describe_results.index.name = 'Trade Category'
        describe_results.columns.name = 'Statistic'
        describe_results = describe_results.round(4)
        describe_results['count'] = describe_results['count'].astype(int)

        # Print the full statistics table
        print("\nDetailed Trade Statistics:")
        print(describe_results)

        # Extract key statistics
        key_stats = {}
        for category in trade_categories.keys():
            if category.endswith('win'):
                key_stats[f'{category}Avg'] = describe_results.loc[category, 'mean']
                # key_stats[f'{category}Med'] = describe_results.loc[category, '50%']
                key_stats[f'{category}Max'] = describe_results.loc[category, 'max']
            elif category.endswith('lose'):
                key_stats[f'{category}Avg'] = describe_results.loc[category, 'mean']
                # key_stats[f'{category}Med'] = describe_results.loc[category, '50%']
                key_stats[f'{category}Min'] = describe_results.loc[category, 'min']
            else:
                key_stats[f'{category}Avg'] = describe_results.loc[category, 'mean']
                # key_stats[f'{category}Med'] = describe_results.loc[category, '50%']
                # key_stats[f'{category}Std'] = describe_results.loc[category, 'std']
        
        print(f"Number of Winning Trades: {win_trades} ({win_longs} long, {win_shorts} short)")
        print(f"Number of Losing Trades:  {lose_trades} ({lose_longs} long, {lose_shorts} short)")
        
        
        print(f"Win Rate: {win_trades / len(self.trades) * 100:.4f}%" if self.trades else "Win Rate: N/A")
        print(f"\nMargin Usage:")
        print(f"Margin Enabled: {'Yes' if self.params.use_margin else 'No'}")
        print(f"Max Buying Power: {self.params.times_buying_power}x")
        # print(f"Average Margin Multiplier: {sum(trade.actual_margin_multiplier for trade in trades) / len(self.trades):.4f}x")
        print(f"Average Transactions per Position: {avg_transact:.4f}")
        
        print('\nAdjusted scaling statistics:')
        combined_describe = pd.concat([
            pd.Series(self.pressure_scalars).describe(),
            pd.Series(self.spread_scalars).describe(),
            pd.Series(self.final_scalars).describe(),
            pd.Series(self.max_trade_sizes).describe(),
            pd.Series(self.max_trade_size_adjustments).describe(),
            pd.Series(self.trade_size_reductions).describe()
        ], axis=1).round(3)
        combined_describe.columns = ['Pressure', 'Spread', 'Final', 'MaxSize','MaxSizeDiff','SizeDiff']
        print(combined_describe)
            
        # # print(trades)
        if self.export_trades_path:
            export_trades_to_csv(self.trades, self.export_trades_path)

        plot_cumulative_pl_and_price(self.trades, self.touch_detection_areas.bars, self.params.initial_investment, filename=self.export_graph_path)

        # return self.balance, sum(1 for trade in trades if trade.is_long), sum(1 for trade in trades if not trade.is_long), balance_change, mean_plpc, win_mean_plpc, lose_mean_plpc, \
        #     win_trades / len(self.trades) * 100,  \
        #     total_transaction_costs, avg_transact, self.count_entry_adjust, self.count_entry_skip, self.count_exit_adjust, self.count_exit_skip
        return self.balance, sum(1 for trade in trades if trade.is_long), sum(1 for trade in trades if not trade.is_long), balance_change, mean_plpc, win_mean_plpc, lose_mean_plpc, \
            win_trades / len(self.trades) * 100, total_transaction_costs, \
                               avg_transact, self.count_entry_adjust, self.count_entry_skip, self.count_exit_adjust, self.count_exit_skip, key_stats


        
# # # Usage
# params = StrategyParameters(
#     initial_investment=10_000,
#     do_longs=True,
#     do_shorts=True,
#     sim_longs=True,
#     sim_shorts=True,
    
#     use_margin=True,
    
#     times_buying_power=4,
    
#     soft_start_time = None, 
#     soft_end_time = '15:50'
# )

# strategy = TradingStrategy(touch_detection_areas, params, export_trades_path='trades_output.csv')
# results = strategy.run_backtest()
