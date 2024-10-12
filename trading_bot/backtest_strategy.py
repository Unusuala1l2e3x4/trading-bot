from typing import List, Dict, Optional, Tuple, Set
from numba import jit
import numpy as np
from datetime import datetime, time
import pandas as pd
import math
from TouchArea import TouchArea, TouchAreaCollection
from TradePosition import TradePosition, export_trades_to_csv, plot_cumulative_pl_and_price

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from tqdm import tqdm

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

logger = setup_logger(logging.INFO)

def log(message, level=logging.INFO):
    logger.log(level, message)



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


@jit(nopython=True)
def is_trading_allowed(total_account_value, avg_trade_count, min_trade_count, avg_volume) -> bool:
    # if total_account_value < 25000: # pdt_threshold
    #     return False
    return avg_trade_count >= min_trade_count and avg_volume >= min_trade_count # at least 1 share per trade

@jit(nopython=True)
def calculate_max_trade_size(avg_volume: float, max_volume_percentage: float) -> int:
    return math.floor(avg_volume * max_volume_percentage)


def backtest_strategy(touch_detection_areas, initial_investment: float=10_000, max_investment: float=float("inf"),  do_longs=True, do_shorts=True, \
    sim_longs=True, sim_shorts=True, use_margin=False, times_buying_power: float=4, min_stop_dist_relative_change_for_partial:Optional[int]=0, \
    soft_start_time:Optional[str]=None, soft_end_time:Optional[str]=None, export_trades_path:Optional[str]=None, \
    max_volume_percentage: float = 0.01, min_trade_count: int = 100, slippage_factor:Optional[float]=0.001):
    """
    Backtests a trading strategy based on touch detection areas, simulating trades and tracking performance.

    Parameters:
    touch_detection_areas (dict): Dictionary containing touch detection areas and other market data.
    initial_investment (float): The initial capital to start the backtest with. Default is 10000.
    do_longs (bool): Whether to execute long trades. Default is True.
    do_shorts (bool): Whether to execute short trades. Default is True.
    use_margin (bool): Whether to use margin for trading. Default is False.
    times_buying_power (float): The multiplier for buying power, up to the maximum of 4x allowed by Alpaca. Default is 4.
    min_stop_dist_relative_change_for_partial (Optional[int]): Minimum relative change in stop distance to trigger a partial trade. Default is 0.
    soft_start_time (Optional[str]): The earliest time to start trading each day. Format: 'HH:MM'. Default is None.
    soft_end_time (Optional[str]): Time after which no new positions are opened, but existing positions can be closed. Format: 'HH:MM'. Default is None.
    export_trades_path (Optional[str]): File path to export trade data. Default is None.

    Returns:
    tuple: A tuple containing various performance metrics and statistics from the backtest,
           including final balance, number of trades, win rate, and transaction costs.

    This function simulates trading based on the provided touch detection areas, applying the specified
    strategy parameters. It handles position entry and exit, manages risk, and calculates various
    performance metrics. The function supports both long and short trades, margin trading up to 4x as 
    per Alpaca's limits, and allows for partial position sizing based on price movements relative to 
    the touch areas. After the soft_end_time, the strategy only allows for closing existing positions.
    """
    symbol = touch_detection_areas['symbol']
    long_touch_area = touch_detection_areas['long_touch_area']
    short_touch_area = touch_detection_areas['short_touch_area']
    market_hours = touch_detection_areas['market_hours']
    df = touch_detection_areas['bars']
    mask = touch_detection_areas['mask']
    # bid_buffer_pct = touch_detection_areas['bid_buffer_pct']
    min_touches = touch_detection_areas['min_touches']
    start_time = touch_detection_areas['start_time']
    end_time = touch_detection_areas['end_time']
    # use_median = touch_detection_areas['use_median']

    # convert floats to float
    assert 0 < times_buying_power <= 4
    
    if soft_start_time:
        soft_start_time = pd.to_datetime(soft_start_time, format='%H:%M').time()
    if soft_end_time:
        soft_end_time = pd.to_datetime(soft_end_time, format='%H:%M').time()
        
    debug = False
    debug2 = False
    debug3 = False
    def debug_print(*args, **kwargs):
        if debug:
            print(*args, **kwargs)
    def debug2_print(*args, **kwargs):
        if debug2:
            print(*args, **kwargs)
    def debug3_print(*args, **kwargs):
        if debug3:
            print(*args, **kwargs)
            
    assert do_longs or do_shorts
    assert 0 <= min_stop_dist_relative_change_for_partial <= 1
    
    POSITION_OPENED = True
    NO_POSITION_OPENED = False
    
    all_touch_areas = []
    if do_longs or sim_longs:
        all_touch_areas.extend(long_touch_area)
    if do_shorts or sim_shorts:
        all_touch_areas.extend(short_touch_area)
    touch_area_collection = TouchAreaCollection(all_touch_areas, min_touches)

    df = df[mask]
    df = df.sort_index(level='timestamp')
    timestamps = df.index.get_level_values('timestamp')


    def update_total_account_value(current_price, name):
        nonlocal total_account_value, balance
        for position in open_positions.values():
            position.update_market_value(current_price)
        
        market_value = sum(position.market_value for position in open_positions.values())
        cash_committed = sum(position.cash_committed for position in open_positions.values())
        total_account_value = balance + cash_committed
        
        if sum(position.cash_committed for position in open_positions.values()) > 0:
            debug2_print(f"  {name} - update_total_account_value(current_price={current_price}):")
            debug2_print(f"    balance: {balance:.6f}")
            debug2_print(f"    market_value: {market_value:.6f}")
            debug2_print(f"    cash_committed: {cash_committed:.6f}")
            debug2_print(f"    total_account_value: {total_account_value:.6f}")
        
            for area_id, position in open_positions.items():
                debug2_print(f"      Position {position.id} in {'res' if position.area.is_long else 'sup'} area {area_id} : Shares: {position.shares}, Market Value: {position.market_value:.6f}, Realized PnL: {position.get_realized_pnl:.2f}, Unrealized PnL: {position.get_unrealized_pnl:.2f}, Cash Committed: {position.cash_committed:.6f}")
                debug2_print([f" Sub-position {i}: Shares: {sp.shares}, Entry Price: {sp.entry_price:.4f}, Exit Price: {sp.exit_price if sp.exit_time else np.nan:.4f}" for i, sp in enumerate(position.sub_positions)])
                # for i, sp in enumerate(position.sub_positions):
                #     if sp.exit_time is None:
                #         debug2_print(f"        Sub-position {i}: Shares: {sp.shares}, Entry Price: {sp.entry_price:.4f}")
                #     else:
                #         debug2_print(f"        Sub-position {i}: Shares: {sp.shares}, Entry Price: {sp.entry_price:.4f}, Exit Price: {sp.exit_price:.4f}")

                                    
    def rebalance(is_simulated: bool, cash_change: float, current_price: float = None):
        nonlocal balance, total_account_value
        
        if not is_simulated:
            old_balance = balance
            new_balance = balance + cash_change
            
            assert new_balance >= 0, f"Negative balance encountered: {new_balance:.4f} ({old_balance:.4f} {cash_change:.4f})"
            balance = new_balance

        if current_price is not None:
            update_total_account_value(current_price, 'REBALANCE')
        
        if not is_simulated:
            s = sum(pos.cash_committed for pos in open_positions.values())
            assert abs(total_account_value - (balance + s)) < 1e-8, \
                f"Total account value mismatch: {total_account_value:.2f} != {balance + s:.2f} ({balance:.2f} + {s:.2f})"

            debug2_print(f"Rebalance: Old balance: {old_balance:.4f}, Change: {cash_change:.4f}, New balance: {balance:.4f}, Total Account Value: {total_account_value:.4f}")
            
        
            
    def exit_action(area_id, position):
        nonlocal trades
        
        debug2_print(f"{'res' if position.area.is_long else 'sup'} area {area_id}:\t{position.id} {position.exit_time} - Exit {'Long ' if position.is_long else 'Short'} at {position.exit_price:.4f}")
        
        # Calculate and print additional statistics
        debug2_print(f"  Trade Summary:")
        debug2_print(f"    Entry Price: {position.entry_price:.4f}")
        debug2_print(f"    Exit Price: {position.exit_price:.4f}")
        debug2_print(f"    Initial Shares: {position.initial_shares}")
        debug2_print(f"    Total P/L: {position.profit_loss:.4f}")
        debug2_print(f"    ROE (P/L %): {position.profit_loss_pct:.2f}%")
        debug2_print(f"    Holding Time: {position.holding_time}")
        debug2_print(f"    Number of Partial Entries: {position.partial_entry_count}")
        debug2_print(f"    Number of Partial Exits: {position.partial_exit_count}")
        debug2_print(f"    Total Transaction Costs: {position.total_transaction_costs:.4f}")
        
        # Calculate and print transaction statistics
        entry_transactions = [t for t in position.transactions if t.is_entry]
        exit_transactions = [t for t in position.transactions if not t.is_entry]
                        
        debug2_print("TRANSACTIONS entry:",f"Total cost: {sum(t.transaction_cost for t in entry_transactions):.4f} | ", ", ".join([f"({t.value:.2f}, {t.transaction_cost:.4f})" for t in entry_transactions]))
        debug2_print("TRANSACTIONS exit: ",f"Total cost: {sum(t.transaction_cost for t in exit_transactions):.4f} | ", ", ".join([f"({t.value:.2f}, {t.transaction_cost:.4f})" for t in exit_transactions]))
                
        avg_entry_price = sum(t.price * t.shares for t in entry_transactions) / sum(t.shares for t in entry_transactions)
        avg_exit_price = sum(t.price * t.shares for t in exit_transactions) / sum(t.shares for t in exit_transactions)
        
        debug2_print(f"  Transaction Statistics:")
        debug2_print(f"    Total Transactions: {len(position.transactions)}")
        debug2_print(f"    Entry Transactions: {len(entry_transactions)}")
        debug2_print(f"    Exit Transactions: {len(exit_transactions)}")
        debug2_print(f"    Average Entry Price: {avg_entry_price:.4f}")
        debug2_print(f"    Average Exit Price: {avg_exit_price:.4f}")
        
        # Print any other relevant information
        if position.is_long:
            price_movement = position.exit_price - position.entry_price
        else:
            price_movement = position.entry_price - position.exit_price
        debug2_print(f"  Price Movement: {price_movement:.4f} ({(price_movement / position.entry_price) * 100:.2f}%)")
        
        trades.append(position)
                
                
    def close_all_positions(timestamp, exit_price, vwap, volume, avg_volume, slippage_factor):
        nonlocal trades_executed
        areas_to_remove = []
        
        debug2_print('CLOSING ALL POSITIONS...')

        for area_id, position in list(open_positions.items()):
            # position.update_market_value(exit_price)
            realized_pnl, cash_released, fees = position.partial_exit(timestamp, exit_price, position.shares, vwap, volume, avg_volume, slippage_factor)
            debug2_print(f"  Partial exit complete - Realized PnL: {realized_pnl:.2f}, Cash released: {cash_released:.2f}")
            rebalance(position.is_simulated, cash_released + realized_pnl - fees, exit_price)
            debug2_print(f"  Partial exit: Sold {position.shares} shares at {exit_price:.4f}, Realized PnL: {realized_pnl:.2f}, Cash released: {cash_released:.4f}")

            position.close(timestamp, exit_price)
            trades_executed += 1
            position.area.record_entry_exit(position.entry_time, position.entry_price, 
                                            timestamp, exit_price)
            position.area.terminate(touch_area_collection)
            areas_to_remove.append(area_id)

        temp = {}
        for area_id in areas_to_remove:
            temp[area_id] = open_positions[area_id]
            del open_positions[area_id]
        for area_id in areas_to_remove:
            exit_action(area_id, temp[area_id])

        assert not open_positions
        open_positions.clear()


    def calculate_position_details(current_price: float, times_buying_power: float, avg_volume: float, avg_trade_count: float, volume: float,
                                max_volume_percentage: float, min_trade_count: int,
                                existing_sub_positions: Optional[np.ndarray] = np.array([]), target_shares: Optional[int]=None):
        nonlocal balance, use_margin, is_marginable, max_investment
        
        if not is_trading_allowed(avg_trade_count, min_trade_count, avg_volume):
            return 0, 0, 0, 0, 0, 0, 0
        
        if use_margin and is_marginable:
            initial_margin_requirement = 0.5
            overall_margin_multiplier = min(times_buying_power, 4.0)
        else:
            initial_margin_requirement = 1.0
            overall_margin_multiplier = min(times_buying_power, 1.0)
            
        actual_margin_multiplier = min(overall_margin_multiplier, 1.0/initial_margin_requirement)
        
        available_balance = min(balance, max_investment) * overall_margin_multiplier
        
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
            assert is_marginable
            rem = max_shares % num_subs
            if rem != 0:
                max_shares -= rem
                max_additional_shares -= rem
                debug3_print(f"Adjusted max_shares to {max_shares} and max_additional_shares to {max_additional_shares} to ensure even distribution among {num_subs} sub-positions")
        
        invest_amount = max_additional_shares * current_price
        actual_cash_used = invest_amount / overall_margin_multiplier
        estimated_entry_cost = TradePosition.estimate_entry_cost(max_shares, overall_margin_multiplier, existing_sub_positions)

        return max_shares, actual_margin_multiplier, overall_margin_multiplier, estimated_entry_cost, actual_cash_used, max_additional_shares, invest_amount

    
    def create_new_position(area: TouchArea, timestamp: datetime, data, prev_close: float):
        nonlocal balance, current_id, total_account_value, open_positions, trades_executed
        open_price, high_price, low_price, close_price, volume, trade_count, vwap, avg_volume, avg_trade_count = \
            data.open, data.high, data.low, data.close, data.volume, data.trade_count, data.vwap, data.avg_volume, data.avg_trade_count
        
        if not is_trading_allowed(avg_trade_count, min_trade_count, avg_volume):
            return NO_POSITION_OPENED
        
        if open_positions or balance <= 0:
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
        
        debug3_print(f"\n{timestamp}\texecution_price: {execution_price:.4f}\t{'long' if area.is_long else 'short'} {current_id}")
        debug3_print(f'Balance: {balance:.4f}')
        
        # debug3_print(f"Execution price: {execution_price:.4f}")

        # Calculate position size, etc...
        max_shares, actual_margin_multiplier, overall_margin_multiplier, estimated_entry_cost, actual_cash_used, max_additional_shares, invest_amount = calculate_position_details(
            execution_price, times_buying_power, avg_volume, avg_trade_count, volume,
            max_volume_percentage, min_trade_count
        )

        debug3_print(f"  Calculated position details: max_shares={max_shares}, actual_margin_multiplier={actual_margin_multiplier:.4f}, overall_margin_multiplier={overall_margin_multiplier:.4f}")
        debug3_print(f"  Estimated entry cost: {estimated_entry_cost:.4f}, Actual cash used: {actual_cash_used:.4f}")
    
        if actual_cash_used + estimated_entry_cost > balance:
            debug3_print(f"  Order rejected: Insufficient balance ({actual_cash_used:.4f}+{estimated_entry_cost:.4f}={actual_cash_used + estimated_entry_cost:4f} > {balance:.4f})")
            return NO_POSITION_OPENED
        
        debug3_print(f"  Invest amount: {invest_amount:.4f}")
        
        debug_print(f"    Shares: {max_shares}, Invest Amount: {invest_amount:.4f}")
        debug_print(f"    Margin Multiplier: {actual_margin_multiplier:.2f}")
        debug_print(f"    execution_price: {execution_price:.2f}")

        # Create the position
        position = TradePosition(
            date=timestamp.date(),
            id=current_id,
            area=area,
            is_long=area.is_long,
            entry_time=timestamp,
            initial_balance=actual_cash_used,
            initial_shares=max_shares,
            entry_price=execution_price,
            use_margin=use_margin,
            is_marginable=is_marginable, # when live, need to call is_security_marginable
            times_buying_power=overall_margin_multiplier,
            actual_margin_multiplier=actual_margin_multiplier,
            current_stop_price=high_price - area.get_range if area.is_long else low_price + area.get_range,
            max_price=high_price if area.is_long else None,
            min_price=low_price if not area.is_long else None
        )
    
        if (area.is_long and do_longs) or (not area.is_long and do_shorts and is_etb):  # if conditions not met, simulate position only.
            position.is_simulated = False
        else:
            position.is_simulated = True

        debug3_print(f'Balance {balance:.4f}, invest_amount {invest_amount:.4f}, actual_cash_used {actual_cash_used:.4f}')
        cash_needed, fees = position.initial_entry(vwap, volume, avg_volume, slippage_factor)
        
        debug3_print(f'INITIAL entry fees estimated {estimated_entry_cost:.4f}, actual {fees:.4f}')
        debug3_print(f'  cash needed {cash_needed:.4f}')
        
        assert estimated_entry_cost >= fees
        
        current_id += 1
        
        # Add to open positions (regardless if real or simulated)
        open_positions[area.id] = position
        
        debug2_print(f"{'res' if area.is_long else 'sup'} area {area.id}: {position.id} {timestamp} - Enter {'Long ' if area.is_long else 'Short'} at {execution_price:.4f}. "
              f"Shares: {max_shares}, Amount: ${invest_amount:.4f} (Margin: {actual_margin_multiplier:.2f}x, Overall: {overall_margin_multiplier:.2f}x, Sub-positions: {len(position.sub_positions)})")
        rebalance(position.is_simulated, -cash_needed - fees, close_price)
            
        return POSITION_OPENED

    def calculate_exit_details(times_buying_power: float, shares_to_sell: int, volume: float, avg_volume: float, avg_trade_count: float, max_volume_percentage: float, min_trade_count: int):
        # Calculate the adjustment factor
        adjustment = min(max(shares_to_sell / avg_volume, 0), 1)
        
        # Adjust min_trade_count, with a lower bound of 10% of the original value
        adjusted_min_trade_count = max(min_trade_count * (1 - adjustment), min_trade_count * 0.1)
        
        # Check if trading is allowed
        if not is_trading_allowed(avg_trade_count, adjusted_min_trade_count, avg_volume):
            return 0  # No trading allowed, return 0 shares to sell
        
        # Adjust max_volume_percentage, with an upper bound of 3 times the original value
        adjusted_max_volume_percentage = min(max_volume_percentage * (1 + adjustment), max_volume_percentage * 3)
        
        max_volume_shares = calculate_max_trade_size(avg_volume, adjusted_max_volume_percentage)

        # Ensure we don't sell more than the available shares or the calculated max_volume_shares
        shares_to_sell = min(shares_to_sell, max_volume_shares)
        
        # if max_volume_shares == shares_to_sell:
        #     when_at_max_volume_percentage.append(timestamp)
            
        # Ensure target_shares is divisible
        num_subs = TradePosition.calculate_num_sub_positions(times_buying_power)
        if num_subs > 1:
            rem = shares_to_sell % num_subs
            if rem != 0:
                shares_to_sell -= rem
                debug3_print(f"Adjusted shares_to_sell to {shares_to_sell} to ensure even distribution among {num_subs} sub-positions")
        return shares_to_sell
            
    def update_positions(timestamp:datetime, data):
        nonlocal trades_executed, count_entry_adjust, count_entry_skip, count_exit_adjust, count_exit_skip
        open_price, high_price, low_price, close_price, volume, trade_count, vwap, avg_volume, avg_trade_count = \
            data.open, data.high, data.low, data.close, data.volume, data.trade_count, data.vwap, data.avg_volume, data.avg_trade_count
        
        areas_to_remove = []
        # debug_print(f"\nDEBUG: Updating positions at {timestamp}, Close price: {close_price:.4f}")

        # if using trailing stops, exit_price = None
        def perform_exit(area_id, position, exit_price=None):
            nonlocal trades_executed
            price = position.current_stop_price if exit_price is None else exit_price
            position.close(timestamp, price)
            trades_executed += 1
            position.area.record_entry_exit(position.entry_time, position.entry_price, 
                                            timestamp, price)
            position.area.terminate(touch_area_collection)
            areas_to_remove.append(area_id)
            

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
                assert is_marginable
                target_shares -= rem
            
            return target_shares

        for area_id, position in open_positions.items():
            # position = TradePosition(position)
            debug2_print(f"\nDEBUG: Processing position {position.id} for area {area_id}")
            debug2_print(f"  Current position - Shares: {position.shares}, Cash committed: {position.cash_committed:.2f}")
            # debug_print(f"Updating position {position.id} at {timestamp}")
            # debug_print(f"  Current stop price: {position.current_stop_price:.4f}")
            
            price_at_action = None
            
            # OHLC logic for trailing stops
            # Initial tests found that just using close_price is more effective
            # Implies we aren't using trailing stop sells
            # UNLESS theres built-in functionality to wait until close
            
            # if not price_at_action:
            #     should_exit = position.update_stop_price(open_price)
            #     target_shares = calculate_target_shares(position, open_price)
            #     if should_exit or target_shares == 0:
            #         debug2_print(f"  Trailing Stop - Exiting at open: {open_price:.4f} {'<=' if position.is_long else '>='} {position.current_stop_price:.4f}")
            #         perform_exit(area_id, position) # DO NOT pass price into function since order would have executed at current_stop_price.
            #         price_at_action = open_price
            
            # # If not stopped out at open, simulate intra-minute price movement
            # if not price_at_action:
            #     should_exit = position.update_stop_price(high_price)
            #     target_shares = calculate_target_shares(position, high_price)
            #     if not position.is_long and (should_exit or target_shares == 0):
            #         # For short positions, the stop is crossed if high price increases past it
            #         debug2_print(f"  Trailing Stop - Exiting at high: {high_price:.4f} {'<=' if position.is_long else '>='} {position.current_stop_price:.4f}")
            #         perform_exit(area_id, position) # DO NOT pass price into function since order would have executed at current_stop_price.
            #         price_at_action = high_price
            
            # if not price_at_action:
            #     should_exit = position.update_stop_price(low_price)
            #     target_shares = calculate_target_shares(position, low_price)
            #     if position.is_long and (should_exit or target_shares == 0):
            #         # For long positions, the stop is crossed if low price decreases past it
            #         debug2_print(f"  Trailing Stop - Exiting at low: {low_price:.4f} {'<=' if position.is_long else '>='} {position.current_stop_price:.4f}")
            #         perform_exit(area_id, position) # DO NOT pass price into function since order would have executed at current_stop_price.
            #         price_at_action = low_price
            
            
            if not price_at_action:
                should_exit = position.update_stop_price(close_price)
                target_shares = calculate_target_shares(position, close_price)
                if should_exit or target_shares == 0:
                    debug2_print(f"  Trailing Stop - Exiting at close: {close_price:.4f} {'<=' if position.is_long else '>='} {position.current_stop_price:.4f}")
                    price_at_action = close_price
                
            if price_at_action:
                assert target_shares == 0
            
            if not price_at_action:
                price_at_action = close_price
            
            # Partial exit and entry logic
            # target_shares = calculate_target_shares(position, price_at_action)
            debug2_print(f"  Target shares: {target_shares}, Current shares: {position.shares}")
            assert target_shares <= position.initial_shares

            target_pct = target_shares / position.initial_shares
            current_pct = min( 1.0, position.shares / position.initial_shares)
            assert 0.0 <= target_pct <= 1.0, target_pct
            assert 0.0 <= current_pct <= 1.0, current_pct


            # To prevent over-trading, skip partial buy/sell if difference between target and current shares percentage is less than threshold
            # BUT only if not increasing/decrease to/from 100%
            # Initial tests found that a threshold closer to 0 or 1, not in between, gives better results
            if abs(target_pct - current_pct) < min_stop_dist_relative_change_for_partial:
                debug2_print(f"    SKIP - Current -> Target percentage: {current_pct*100:.2f}% ({position.shares}) -> {target_pct*100:.2f}% ({target_shares})")
                update_total_account_value(price_at_action, 'skip')
                continue

            
            if target_shares < position.shares:
                
                debug3_print(f"\n{timestamp}\tprice_at_action: {price_at_action:.4f}\t{'long' if position.is_long else 'short'} {position.id}")
                debug3_print(f'Balance: {balance:.4f}')
                shares_to_adjust = position.shares - target_shares
                # debug3_print(f"Initiating partial exit - Shares to sell: {shares_to_adjust}")
                
                if shares_to_adjust > 0:
                    debug3_print(f"    Current -> Target percentage: {current_pct*100:.2f}% ({position.shares}) -> {target_pct*100:.2f}% ({target_shares})")

                    shares_to_sell = calculate_exit_details(
                        position.times_buying_power,
                        shares_to_adjust,
                        volume,
                        avg_volume,
                        avg_trade_count,
                        max_volume_percentage,
                        math.floor(min_trade_count * (shares_to_adjust / position.max_shares))
                    )
                    
                    if shares_to_sell > 0:
                        realized_pnl, cash_released, fees = position.partial_exit(timestamp, price_at_action, shares_to_sell, vwap, volume, avg_volume, slippage_factor)
                        # debug3_print(f"  Partial exit complete - Realized PnL: {realized_pnl:.2f}, Cash released: {cash_released:.2f}")
                        
                        rebalance(position.is_simulated, cash_released + realized_pnl - fees, price_at_action)
                            
                        if position.shares == 0:
                            perform_exit(area_id, position, price_at_action)

                        debug3_print(f"  Partial exit: Sold {shares_to_adjust} shares at {price_at_action:.4f}, Realized PnL: {realized_pnl:.2f}, Cash released: {cash_released:.4f}")
                        
                        if shares_to_sell < shares_to_adjust:
                            debug3_print(f'PARTIAL exit ADJUSTED.')
                            debug3_print(f'  shares_to_adjust {shares_to_adjust}, shares_to_sell {shares_to_sell}')
                            count_exit_adjust += 1
                    else:
                        debug3_print(f'PARTIAL exit SKIPPED.')
                        debug3_print(f'  shares_to_adjust {shares_to_adjust}, shares_to_sell {shares_to_sell}')
                        count_exit_skip += 1
                        
            elif target_shares > position.shares:
                
                debug3_print(f"\n{timestamp}\tprice_at_action: {price_at_action:.4f}\t{'long' if position.is_long else 'short'} {position.id}")
                debug3_print(f'Balance: {balance:.4f}')
                shares_to_adjust = target_shares - position.shares
                if shares_to_adjust > 0:
                    debug3_print(f"    Current -> Target percentage: {current_pct*100:.2f}% ({position.shares}) -> {target_pct*100:.2f}% ({target_shares})")
                    
                    existing_sub_positions = np.array([sp.shares for sp in position.sub_positions if sp.shares > 0])
                    max_shares, _, _, estimated_entry_cost, actual_cash_used, max_additional_shares, invest_amount = calculate_position_details(
                        price_at_action, position.times_buying_power, avg_volume, avg_trade_count, volume,
                        max_volume_percentage, math.floor(min_trade_count * (shares_to_adjust / position.max_shares)), 
                        existing_sub_positions=existing_sub_positions, target_shares=target_shares
                    )
                    
                    shares_to_buy = min(shares_to_adjust, max_additional_shares)
                    
                    if shares_to_buy > 0:
                        cash_needed, fees = position.partial_entry(timestamp, price_at_action, shares_to_buy, vwap, volume, avg_volume, slippage_factor)
                        debug3_print(f'PARTIAL entry fees estimated {estimated_entry_cost:.4f}, actual {fees:.4f}')
                        debug3_print(f'  cash needed {cash_needed:.4f}')
                        
                        # debug3_print(f"  Partial entry complete - Shares bought: {shares_to_buy}, Cash used: {cash_needed:.2f}")
                        rebalance(position.is_simulated, -cash_needed - fees, price_at_action)
                        
                        # Update max_shares after successful partial entry
                        position.max_shares = max(position.max_shares, position.shares)
                        assert position.shares == max_shares

                        if shares_to_buy < shares_to_adjust:
                            debug3_print(f'PARTIAL entry ADJUSTED.')
                            debug3_print(f'  shares_to_adjust {shares_to_adjust}, shares_to_buy {shares_to_buy}, max_additional_shares {max_additional_shares}.')
                            count_entry_adjust += 1
                    else:
                        debug3_print(f'PARTIAL entry SKIPPED.')
                        debug3_print(f'  shares_to_adjust {shares_to_adjust}, shares_to_buy {shares_to_buy}, max_additional_shares {max_additional_shares}.')
                        
                        # Update max_shares when entry is skipped
                        position.max_shares = min(position.max_shares, position.shares)
                        count_entry_skip += 1

        temp = {}
        for area_id in areas_to_remove:
            temp[area_id] = open_positions[area_id]
            del open_positions[area_id]
        for area_id in areas_to_remove:
            exit_action(area_id, temp[area_id])
        
        update_total_account_value(close_price, 'AFTER removing exited positions')
        if areas_to_remove:
            debug_print(f"  Updated Total Account Value: {total_account_value:.4f}")

    debug_print(f"Strategy: {'Long' if do_longs else ''}{'&' if do_longs and do_shorts else ''}{'Short' if do_shorts else ''}")
    debug2_print(f"{timestamps[0]} -> {timestamps[-1]}")
    debug_print(f"Initial Investment: {initial_investment}, Times Buying Power: {times_buying_power}")
    debug2_print('Number of touch areas:', len(all_touch_areas))
    
    # in live setting, call these every time before placing order. for backtesting, cannot due to API constraints.
    is_marginable = is_security_marginable(symbol) 
    is_etb = is_security_shortable_and_etb(symbol)

    print(f'{symbol} is {'NOT ' if not is_marginable else ''}marginable.')
    print(f'{symbol} is {'NOT ' if not is_etb else ''}shortable and ETB.')
    
    balance = initial_investment
    total_account_value = initial_investment
    when_above_max_investment = []
    when_at_max_volume_percentage = []

    trades = []  # List to store all trades
    trades_executed = 0
    open_positions = {}
    
    count_entry_adjust, count_exit_adjust, count_entry_skip, count_exit_skip = 0, 0, 0, 0
    
    daily_data = None
    current_date, market_open, market_close = None, None, None
    for i in tqdm(range(1, len(timestamps))):
        
        current_time = timestamps[i].tz_convert(ny_tz)
        debug2_print('TIMESTAMP',current_time)
        # print(current_time, len(open_positions))
        if total_account_value > max_investment:
            when_above_max_investment.append(current_time)
        
        if current_time.date() != current_date:
            current_id = 0
            debug_print(f"\nNew trading day: {current_time.date()}")
            # New day, reset daily data
            current_date = current_time.date()
            daily_data = df[timestamps.date == current_date]
            
            market_open, market_close = market_hours.get(current_date, (None, None))
            if market_open and market_close:
                date_obj = pd.Timestamp(current_date).tz_localize(ny_tz)
                if start_time:
                    day_start_time = date_obj.replace(hour=start_time.hour, minute=start_time.minute)
                else:
                    day_start_time = market_open
                if end_time:
                    day_end_time = min(date_obj.replace(hour=end_time.hour, minute=end_time.minute), market_close - pd.Timedelta(minutes=3))
                else:
                    day_end_time = market_close - pd.Timedelta(minutes=3)

                if soft_start_time:
                    day_soft_start_time = max(market_open, day_start_time, date_obj.replace(hour=soft_start_time.hour, minute=soft_start_time.minute))
                else:
                    day_soft_start_time = max(market_open, day_start_time)
                    
            daily_index = 1 # start at 2nd position
            # debug_print('MARKET CLOSE AT',market_close)
            assert not open_positions
            
            soft_end_triggered = False
            
        
        if not market_open or not market_close:
            continue
        
        if day_soft_start_time <= current_time < day_end_time and daily_index < len(daily_data)-1 and i < len(df)-1:  # LAST PART is only for testing. not in live environment
            if soft_end_time and not soft_end_triggered:
                if current_time >= pd.Timestamp.combine(current_date, soft_end_time).tz_localize(ny_tz):
                    soft_end_triggered = True
                    debug_print(f"Soft end time reached: {current_time.strftime("%H:%M")}")

            # debug_print(f"\n{current_time.strftime("%H:%M")} - Market Open")
            
            prev_close = daily_data['close'].iloc[daily_index - 1]
            data = daily_data.iloc[daily_index]
            
            update_positions(current_time, data)
            
            if not soft_end_triggered:
                active_areas = touch_area_collection.get_active_areas(current_time)
                for area in active_areas:
                    if balance <= 0:
                        break
                    if open_positions: # ensure only 1 live position at a time
                        break
                    
                    if (area.is_long and (do_longs or sim_longs)) or (not area.is_long and (do_shorts or sim_shorts)):
                        if create_new_position(area, current_time, data, prev_close):
                            break  # Exit the loop after placing a position
        elif current_time >= day_end_time:
            debug_print(f"\n{current_time.strftime("%H:%M")} - Market Close")
            close_all_positions(current_time, df['close'].iloc[i], df['vwap'].iloc[i], df['volume'].iloc[i], df['avg_volume'].iloc[i], slippage_factor)
        elif i >= len(df)-1:
            debug_print(f"\n{current_time.strftime("%H:%M")} - Reached last timestamp")
            close_all_positions(current_time, df['close'].iloc[i], df['vwap'].iloc[i], df['volume'].iloc[i], df['avg_volume'].iloc[i], slippage_factor) # only for testing. not in live environment
            
        daily_index += 1

    if current_time >= day_end_time:
        assert not open_positions

    balance_change = ((balance - initial_investment) / initial_investment) * 100

    # Buy and hold strategy
    start_price = df['close'].iloc[0]
    end_price = df['close'].iloc[-1]
    baseline_change = ((end_price - start_price) / start_price) * 100
    
    total_profit_loss = sum(trade.profit_loss for trade in trades if not trade.is_simulated)
    
    total_profit = sum(trade.profit_loss for trade in trades if not trade.is_simulated and trade.profit_loss > 0)
    total_loss = sum(trade.profit_loss for trade in trades if not trade.is_simulated and trade.profit_loss < 0)
    
    total_transaction_costs = sum(trade.total_transaction_costs for trade in trades if not trade.is_simulated)
    # total_entry_transaction_costs = sum(trade.entry_transaction_costs for trade in trades if not trade.is_simulated)
    # total_exit_transaction_costs = sum(trade.exit_transaction_costs for trade in trades if not trade.is_simulated)
    # total_short_transaction_costs = sum(trade.total_transaction_costs for trade in trades if not trade.is_simulated and not trade.is_long)
    # total_long_transaction_costs = sum(trade.total_transaction_costs for trade in trades if not trade.is_simulated and trade.is_long)
    
    total_stock_borrow_costs = sum(trade.total_stock_borrow_cost for trade in trades if not trade.is_simulated)

    mean_profit_loss = np.mean([trade.profit_loss for trade in trades if not trade.is_simulated])
    mean_profit_loss_pct = np.mean([trade.profit_loss_pct for trade in trades if not trade.is_simulated])

    win_mean_profit_loss_pct = np.mean([trade.profit_loss_pct for trade in trades if not trade.is_simulated and trade.profit_loss > 0])
    lose_mean_profit_loss_pct = np.mean([trade.profit_loss_pct for trade in trades if not trade.is_simulated and trade.profit_loss < 0])
    
    long_win_mean_profit_loss_pct = np.mean([trade.profit_loss_pct for trade in trades if not trade.is_simulated and trade.is_long and trade.profit_loss > 0])
    long_lose_mean_profit_loss_pct = np.mean([trade.profit_loss_pct for trade in trades if not trade.is_simulated and trade.is_long and trade.profit_loss < 0])
    
    short_win_mean_profit_loss_pct = np.mean([trade.profit_loss_pct for trade in trades if not trade.is_simulated and not trade.is_long and trade.profit_loss > 0])
    short_lose_mean_profit_loss_pct = np.mean([trade.profit_loss_pct for trade in trades if not trade.is_simulated and not trade.is_long and trade.profit_loss < 0])
    
    win_trades = sum(1 for trade in trades if not trade.is_simulated and trade.profit_loss > 0)
    lose_trades = sum(1 for trade in trades if not trade.is_simulated and trade.profit_loss < 0)
    win_longs = sum(1 for trade in trades if not trade.is_simulated and trade.is_long and trade.profit_loss > 0)
    lose_longs = sum(1 for trade in trades if not trade.is_simulated and trade.is_long and trade.profit_loss < 0)
    win_shorts = sum(1 for trade in trades if not trade.is_simulated and not trade.is_long and trade.profit_loss > 0)
    lose_shorts = sum(1 for trade in trades if not trade.is_simulated and not trade.is_long and trade.profit_loss < 0)
    
    avg_sub_pos = np.mean([len(trade.sub_positions) for trade in trades if not trade.is_simulated])
    avg_transact = np.mean([len(trade.transactions) for trade in trades if not trade.is_simulated])
    
    # avg_partial_entry_count = np.mean([trade.partial_entry_count for trade in trades if not trade.is_simulated])
    # avg_partial_exit_count = np.mean([trade.partial_exit_count for trade in trades if not trade.is_simulated])
    
    # long_count = sum(1 for trade in trades if not trade.is_simulated and trade.is_long)
    # short_count = sum(1 for trade in trades if not trade.is_simulated and not trade.is_long)
    
    assert trades_executed == len(trades)
    
    debug_print("\nBacktest Complete")
    debug_print(f"Final Balance: {balance:.2f}")
    debug_print(f"Total Trades Executed: {trades_executed}")
    debug_print(f"Win Rate: {win_trades / len(trades) * 100:.2f}%")

    # Print statistics
    print(f"END\nStrategy: {'Long' if do_longs else ''}{'&' if do_longs and do_shorts else ''}{'Short' if do_shorts else ''}")
    print(f'{symbol} is {'NOT ' if not is_marginable else ''}marginable.')
    print(f'{symbol} is {'NOT ' if not is_etb else ''}shortable and ETB.')
    print(f"{timestamps[0]} -> {timestamps[-1]}")

    # debug2_print(df['close'])
    
    print("\nOverall Statistics:")
    print('Initial Investment:', initial_investment)
    print(f'Final Balance:      {balance:.4f}')
    print(f"Balance % change:   {balance_change:.4f}% ***")
    print(f"Baseline % change:  {baseline_change:.4f}%")
    print('Number of Trades Executed:', trades_executed)
    print(f"\nTotal Profit/Loss (including fees): ${total_profit_loss:.4f}")
    print(f"  Total Profit: ${total_profit:.4f}")
    print(f"  Total Loss:   ${total_loss:.4f}")
    print(f"Total Transaction Costs: ${total_transaction_costs:.4f}")
    # print(f"  Long:  ${total_long_transaction_costs:.4f}")
    # print(f"  Short: ${total_short_transaction_costs:.4f}")
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
    
    
    print(f"Win Rate: {win_trades / len(trades) * 100:.4f}%" if trades else "Win Rate: N/A")
    print(f"\nMargin Usage:")
    print(f"Margin Enabled: {'Yes' if use_margin else 'No'}")
    print(f"Max Buying Power: {times_buying_power}x")
    print(f"Average Margin Multiplier: {sum(trade.actual_margin_multiplier for trade in trades) / len(trades):.4f}x")
    print(f"Average Sub Positions per Position: {avg_sub_pos:.4f}")
    print(f"Average Transactions per Position: {avg_transact:.4f}")
    
    # print(trades)
    if export_trades_path:
        export_trades_to_csv(trades, export_trades_path)

    plot_cumulative_pl_and_price(trades, touch_detection_areas['bars'], initial_investment, when_above_max_investment)

    return balance, sum(1 for trade in trades if trade.is_long), sum(1 for trade in trades if not trade.is_long), balance_change, mean_profit_loss_pct, win_mean_profit_loss_pct, lose_mean_profit_loss_pct, \
        win_trades / len(trades) * 100,  \
        total_transaction_costs, avg_sub_pos, avg_transact, count_entry_adjust, count_entry_skip, count_exit_adjust, count_exit_skip


