import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta, date
from zoneinfo import ZoneInfo
from alpaca.data.live.stock import StockDataStream
from alpaca.data.requests import StockBarsRequest
from alpaca.data.enums import DataFeed
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetCalendarRequest
from alpaca.trading.enums import OrderSide, OrderStatus, TimeInForce, OrderType
from alpaca.trading.models import Order
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.enums import Adjustment
from types import SimpleNamespace
from typing import List, Tuple, Optional, Dict

from alpaca.data.models import Bar

from TradePosition import TradePosition
from TouchArea import TouchArea
from TradingStrategy import StrategyParameters, TouchDetectionAreas, TradingStrategy, is_security_shortable_and_etb, is_security_marginable
from TouchDetection import calculate_touch_detection_area, plot_touch_detection_areas, LiveTouchDetectionParameters, np_mean, np_median


import logging

import os, toml
from dotenv import load_dotenv

import sys, io

STANDARD_DATETIME_STR = '%Y-%m-%d %H:%M:%S'
ROUNDING_DECIMAL_PLACES = 10  # Choose an appropriate number of decimal places

load_dotenv(override=True)
livepaper = os.getenv('LIVEPAPER')
config = toml.load('../config.toml')

# Replace with your Alpaca API credentials
API_KEY = config[livepaper]['key']
API_SECRET = config[livepaper]['secret']


debug = False
def debug_print(*args, **kwargs):
    if debug:
        print(*args, **kwargs)


class LiveTrader:
    def __init__(self, api_key, secret_key, symbol, initial_balance, touch_detection_params: LiveTouchDetectionParameters, strategy_params: StrategyParameters, simulation_mode=False):
        self.trading_client = TradingClient(api_key, secret_key, paper=True)
        self.data_stream = StockDataStream(api_key, secret_key, feed=DataFeed.IEX, websocket_params={"ping_interval": 1,"ping_timeout": 180,"max_queue": 1024})
        self.historical_client = StockHistoricalDataClient(api_key, secret_key)
        self.touch_detection_params = touch_detection_params
        self.strategy_params = strategy_params
        self.trading_strategy = None # not initialized until execute_trading_logic
        self.symbol = symbol
        self.balance = initial_balance # self.strategy_params.initial_investment
        self.data = pd.DataFrame()
        self.is_ready = False
        self.gap_filled = False
        self.streamed_data = pd.DataFrame()
        self.last_historical_timestamp = None
        self.first_streamed_timestamp = None
        self.open_positions = {}
        self.areas_to_remove = set()
        self.ny_tz = ZoneInfo("America/New_York")
        self.logger = self.setup_logger(logging.INFO)
        self.simulation_mode = simulation_mode
        
    def setup_logger(self, log_level=logging.INFO):
        logger = logging.getLogger('LiveTrader')
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
        self.logger.log(level, message)
            
    async def reset_daily_data(self):
        self.log("Resetting daily data...")
        current_day_start = self.get_current_trading_day_start()
        
        # Reset data and streamed_data
        if not self.data.empty:
            # self.data = self.data[self.data.index.get_level_values('timestamp') >= current_day_start]
            self.data = pd.DataFrame()
        self.streamed_data = pd.DataFrame()
        
        # Reset other daily variables
        self.is_ready = False
        self.gap_filled = False
        self.last_historical_timestamp = None
        self.first_streamed_timestamp = None
        self.areas_to_remove = set()
        
        # Re-initialize historical data for the new day
        await self.initialize_data()

        # Initialize or update TradingStrategy
        if self.trading_strategy is None:
            self.trading_strategy = TradingStrategy(self.calculate_touch_detection_area(), self.strategy_params, is_live_trading=True)
        else:
            self.trading_strategy.touch_detection_areas = self.calculate_touch_detection_area(self.trading_strategy.market_hours)

        # Update daily parameters in TradingStrategy
        current_date = datetime.now(self.ny_tz).date()
        self.trading_strategy.update_daily_parameters(current_date)
        
        self.log("Daily data reset complete.")
    
    def is_market_open(self, check_time: Optional[datetime] = None):
        if check_time is None:
            check_time = datetime.now(self.ny_tz)
        else:
            check_time = check_time.astimezone(self.ny_tz)
        
        return (check_time.weekday() < 5 and 
                time(4, 0) <= check_time.time() <= time(20, 0))

    def get_current_trading_day_start(self):
        now = datetime.now(self.ny_tz)
        current_date = now.date()
        if now.time() < time(4, 0):
            current_date -= timedelta(days=1)
        return datetime.combine(current_date, time(4, 0)).replace(tzinfo=self.ny_tz)

    async def wait_for_market_open(self):
        while not self.is_market_open():
            self.log("Market is closed. Waiting for market to open.")
            await asyncio.sleep(60)

    def get_historical_bars(self, start, end):
        request_params = StockBarsRequest(
            symbol_or_symbols=self.symbol,
            timeframe=TimeFrame.Minute,
            start=start.astimezone(ZoneInfo("UTC")),
            end=end.astimezone(ZoneInfo("UTC")),
            adjustment=Adjustment.ALL,
        )
        df = self.historical_client.get_stock_bars(request_params).df
        if df.empty:
            return pd.DataFrame()
        df.index = df.index.set_levels(
            df.index.get_level_values('timestamp').tz_convert(self.ny_tz),
            level='timestamp'
        )
        df.sort_index(inplace=True)
        return df

    def get_lagged_time(self):
        return datetime.now(self.ny_tz) - timedelta(minutes=15, seconds=30)

    async def initialize_data(self):
        try:
            end = self.get_lagged_time()
            start = self.get_current_trading_day_start()
            
            self.log(f"Fetching historical data from {start} to {end}")
            
            self.data = self.get_historical_bars(start, end)
            
            if self.data.empty:
                self.log("No historical data available. Waiting for data.", logging.WARNING)
                return
            
            self.last_historical_timestamp = self.data.index.get_level_values('timestamp')[-1]
            time_diff = (end - self.last_historical_timestamp).total_seconds() / 60

            if time_diff >= 16:
                self.log(f"Historical data is too old. Latest data point: {self.last_historical_timestamp}", logging.WARNING)
                return

            self.log(f"Initialized historical data: {len(self.data)} bars")
            self.log(f"Data range: {self.data.index.get_level_values('timestamp')[0]} to {self.last_historical_timestamp}")

        except Exception as e:
            self.log(f"Error initializing data: {e}", logging.ERROR)

    async def update_historical_data(self):
        try:
            end = self.get_lagged_time()
            start = self.last_historical_timestamp + timedelta(minutes=1)
            debug_print('---')
            if end > start:
                new_data = self.get_historical_bars(start, end)
                debug_print('---')
                if not new_data.empty and new_data.index.get_level_values('timestamp').max() > self.last_historical_timestamp:
                    debug_print('---')
                    self.data = pd.concat([self.data, new_data])
                    debug_print('update_historical_data')
                    debug_print('before remove dups',len(self.data))
                    self.data = self.data.loc[~self.data.index.duplicated(keep='last')].sort_index() # needed if update_historical_data is called in less than 1 minute intervals
                    debug_print('after remove dups ',len(self.data))
                    self.last_historical_timestamp = self.data.index.get_level_values('timestamp')[-1]
                    self.log(f"Updated historical data. New range: {self.data.index.get_level_values('timestamp')[0]} to {self.last_historical_timestamp}")
        
        except Exception as e:
            self.log(f"Error updating historical data: {e}", logging.ERROR)
                
    async def simulate_bar(self):
        if self.simulation_mode and not self.data.empty:
            latest_data = self.data.iloc[-1]
            bar = SimpleNamespace(
                symbol=latest_data.name[0],
                timestamp=latest_data.name[1],  # Assuming multi-index with (symbol, timestamp)
                open=latest_data['open'],
                high=latest_data['high'],
                low=latest_data['low'],
                close=latest_data['close'],
                volume=latest_data['volume'],
                trade_count=latest_data['trade_count'],
                vwap=latest_data['vwap'],
                is_simulate_bar=True
            )
            await self.on_bar(bar, check_time=latest_data.name[1])
        
    async def on_bar(self, bar:Bar, check_time: Optional[datetime] = None):
        debug_print(bar)
        try:
            if not self.is_market_open(check_time):
                self.log("Received bar outside market hours. Ignoring.")
                return

            if hasattr(bar, 'is_simulate_bar'):
                is_simulate_bar = bar.is_simulate_bar
            else:
                is_simulate_bar = False
                
            bar_time = pd.to_datetime(bar.timestamp).tz_convert(self.ny_tz)
            now = datetime.now(self.ny_tz)
            time_diff = (now - bar_time).total_seconds() / 60

            if time_diff >= 16 and not is_simulate_bar:
                self.log(f"Received outdated bar data. Bar time: {bar_time}, Current time: {now}", logging.WARNING)
                return
            
            if self.first_streamed_timestamp is None:
                self.first_streamed_timestamp = bar_time
                
            bar_data = {
                'symbol': self.symbol,
                'timestamp': bar_time,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume,
                'trade_count': bar.trade_count,
                'vwap': bar.vwap
            }
            new_row = pd.DataFrame([bar_data])
            new_row.set_index(['symbol','timestamp'], inplace=True)
            debug_print(new_row)
            
            if not is_simulate_bar:
                self.streamed_data = pd.concat([self.streamed_data, new_row])
                debug_print('on_bar streamed_data')
                debug_print('before remove dups',len(self.streamed_data))
                self.streamed_data = self.streamed_data.loc[~self.streamed_data.index.duplicated(keep='last')].sort_index()
                debug_print('after remove dups ',len(self.streamed_data))
                self.log(f"Added streamed bar. {len(self.streamed_data)} streamed bars total.")

            # Check if we've filled the gap
            if not self.is_ready:
                await self.check_gap_filled()
                
            if self.is_ready:
                if not self.simulation_mode:
                    self.data = pd.concat([self.data, new_row])
                    debug_print('on_bar data')
                    debug_print('before remove dups',len(self.data))
                    self.data = self.data.loc[~self.data.index.duplicated(keep='last')].sort_index()
                    debug_print('after remove dups ',len(self.data))
                    debug_print('final data:\n',self.data)
                    # debug_print('final streamed_data:\n',self.streamed_data)
                await self.execute_trading_logic()
                
        except Exception as e:
            self.log(f"Error in on_bar: {e}", logging.ERROR)

    async def check_gap_filled(self):
        time_diff = (self.first_streamed_timestamp - self.last_historical_timestamp).total_seconds() / 60
        if time_diff <= 1 or self.simulation_mode:
            self.is_ready = True
            debug_print('check_gap_filled: self.is_ready = True',time_diff)
            if not self.simulation_mode: # do not use streamed_data if in simulation mode
                self.data = pd.concat([self.data, self.streamed_data])
                debug_print('before remove dups',len(self.data))
                self.data = self.data.loc[~self.data.index.duplicated(keep='last')].sort_index()
                debug_print('after remove dups ',len(self.data))
            self.log("Gap filled. Data is now ready for trading.")
            self.log(f"Continuous data range: {self.data.index.get_level_values('timestamp')[0]} to {self.data.index.get_level_values('timestamp')[-1]}")
        else:
            self.log(f"Gap not properly filled. Time difference: {time_diff} minutes", logging.WARNING)


    async def execute_trading_logic(self):
        try:
            if not self.is_ready:
                self.log("Data not ready for trading")
                return

            # Update TradingStrategy balance
            current_time = self.data.index.get_level_values('timestamp')[-1]
            
            # Calculate touch detection areas every minute
            self.trading_strategy.update_strategy(self.calculate_touch_detection_area(self.trading_strategy.market_hours, current_time))

            
            # Update daily parameters if it's a new day (already handled in process_live_data)
            # if current_time.date() != getattr(self.trading_strategy, 'current_date', None):
                # self.trading_strategy.update_daily_parameters(current_time.date())
                # self.trading_strategy.handle_new_trading_day(current_time)

            # print(self.trading_strategy.touch_detection_areas.symbol)
            self.log(len(self.trading_strategy.touch_detection_areas.long_touch_area), len(self.trading_strategy.touch_detection_areas.short_touch_area))

            # if self.trading_strategy.df is not None and not self.trading_strategy.df.empty:
            #     self.log(f'after mask:\n{self.trading_strategy.df}')

            orders, areas_to_remove = self.trading_strategy.process_live_data(current_time)
            
            self.areas_to_remove = self.areas_to_remove | areas_to_remove

            if orders:
                # self.log(f"{current_time.strftime("%H:%M")}: {len(orders)} ORDERS CREATED")  
                
                # if not self.simulation_mode:
                #     # Place all orders concurrently
                #     await asyncio.gather(*[self.place_order(order) for order in orders])
                # else:
                #     # In simulation mode, just log the orders
                #     # for order in orders:
                #     #     self.log({k:order[k] for k in order if k != 'position'})

                #     self.log(f"{[f"{a['position'].id} {a['position'].is_long} {a['action']} {str(a['order_side']).split('.')[1]} {int(a['qty'])} * {a['price']}, width {a['position'].area.get_range:.4f}" for a in orders]} {self.trading_strategy.balance:.4f}")
                #     # self.balance not updated yet
                    
                    
                # if orders[0]['action'] == 'open':
                    # self.log(self.trading_strategy.df)
                # plot_touch_detection_areas(self.trading_strategy.touch_detection_areas) # for testing
                pass
            
        except Exception as e:
            self.log(f"Error in execute_trading_logic: {e}", logging.ERROR)

    def calculate_touch_detection_area(self, market_hours=None, current_timestamp=None):
        return calculate_touch_detection_area(self.touch_detection_params, self.data, market_hours, current_timestamp, self.areas_to_remove)

    async def place_order(self, order):
        assert isinstance(order['qty'], int)
        assert isinstance(order['order_side'], OrderSide)
        try:
            order_request = MarketOrderRequest(
                symbol=order['symbol'],
                qty=order['qty'],
                side=order['order_side'],
                time_in_force=TimeInForce.DAY
            )
            # placed_order = self.trading_client.submit_order(order_request)
            self.log(f"Placed {order['order_side']} order for {order['qty']} shares of {order['symbol']} - {order['action']}")
            
            # await self.process_placed_order(placed_order, order)
            
            # return placed_order
        except Exception as e:
            self.log(f"Error placing order: {e}", logging.ERROR)

    async def process_placed_order(self, placed_order: Order, original_order: dict):
        # Log order details
        self.log(f"Order ID: {placed_order.id}")
        self.log(f"Order Status: {placed_order.status}")
        self.log(f"Filled Qty: {placed_order.filled_qty}")
        self.log(f"Filled Avg Price: {placed_order.filled_avg_price}")

        # Update internal state based on order status
        if placed_order.status == OrderStatus.FILLED:
            await self.update_position_after_fill(placed_order, original_order)
        elif placed_order.status == OrderStatus.PARTIALLY_FILLED:
            await self.handle_partial_fill(placed_order, original_order)
        elif placed_order.status in [OrderStatus.REJECTED, OrderStatus.CANCELED]:
            await self.handle_failed_order(placed_order, original_order)
        else:
            # For other statuses like NEW, ACCEPTED, etc.
            self.log(f"Order {placed_order.id} is in {placed_order.status} status. Waiting for fill.")

    async def update_position_after_fill(self, placed_order: Order, original_order: dict):
        self.log(f"Order {placed_order.id} filled completely - {placed_order.filled_qty} out of {placed_order.qty}")
        
        position = original_order['position']
        action = original_order['action']
        
        if action == 'open':
            # Update the position with actual filled quantity and price
            position.shares = int(placed_order.filled_qty)
            position.entry_price = float(placed_order.filled_avg_price)
        elif action in ['partial_entry', 'partial_exit', 'close']:
            # Update the position for partial fills
            if original_order['order_side'] == OrderSide.BUY:
                position.shares += int(placed_order.filled_qty)
            else:
                position.shares -= int(placed_order.filled_qty)
        # elif action == 'close':
        #     # Remove the position from open positions -> already done in TradingStrategy
        #     del self.trading_strategy.open_positions[position.area.id]

        # Recalculate account balance
        self.recalculate_balance(placed_order, original_order)

    async def handle_partial_fill(self, placed_order: Order, original_order: dict):
        self.log(f"Order {placed_order.id} partially filled. Filled {placed_order.filled_qty} out of {placed_order.qty}")
        # Update position with partially filled amount
        await self.update_position_after_fill(placed_order, original_order)
        # You might want to create a new order for the remaining quantity
        remaining_qty = float(placed_order.qty) - float(placed_order.filled_qty)
        if remaining_qty > 0:
            new_order = original_order.copy()
            new_order['qty'] = remaining_qty
            await self.place_order(new_order)

    async def handle_failed_order(self, placed_order: Order, original_order: dict):
        self.log(f"Order {placed_order.id} failed with status {placed_order.status}")
        # Implement logic to handle failed orders (e.g., retry, adjust strategy, etc.)

    def recalculate_balance(self, placed_order: Order, original_order: dict):
        # Recalculate balance based on the filled order
        filled_value = float(placed_order.filled_qty) * float(placed_order.filled_avg_price)
        if original_order['order_side'] == OrderSide.BUY:
            self.balance -= filled_value
        else:
            self.balance += filled_value
        self.log(f"Updated balance: {self.balance}")
        
        
    # async def place_order(self, side, qty, order_type=OrderType.MARKET, limit_price=None):
    #     try:
    #         if not self.is_market_open():
    #             self.log("Market is closed. Cannot place order.")
    #             return

    #         order_data = MarketOrderRequest(
    #             symbol=self.symbol,
    #             qty=qty, # number of shares. use int to prevent fractional orders.
    #             side=side, # OrderSide.BUY, OrderSide.SELL
    #             time_in_force=TimeInForce.DAY
    #         )
            
    #         # if order_type == OrderType.LIMIT and limit_price is not None:
    #         #     order_data.type = "limit"
    #         #     order_data.limit_price = limit_price

    #         if self.validate_order(order_data):
    #             order = self.trading_client.submit_order(order_data)
    #             self.log(f"Placed {side} order for {qty} shares of {self.symbol}")
    #             return order
    #         else:
    #             self.log("Order validation failed", logging.WARNING)
    #             return None

    #     except Exception as e:
    #         self.log(f"Error placing order: {e}", logging.ERROR)

    def validate_order(self, order_data):
        # Implement order validation logic
        # Check for sufficient balance, position limits, etc.
        return True

    async def manage_positions(self):
        # Implement position management logic
        # Check for open positions, update stop losses, take profits, etc.
        pass

    async def close_positions(self):
        # Implement logic to close all positions at end of day
        pass
    
    def is_receiving_data(self):
        if not self.streamed_data.empty:
            last_data_time = self.streamed_data.index.get_level_values('timestamp').max()
            time_since_last_data = (datetime.now(self.ny_tz) - last_data_time).total_seconds()
            return time_since_last_data < 120  # Consider it active if data received in last 2 minutes
        return False
                
    async def update_historical_periodically(self):
        while not self.is_ready or self.simulation_mode:
            await asyncio.sleep(60)  # Wait for 1 minute
            if not self.is_market_open():
                break
            if not self.is_ready or self.simulation_mode:
                await self.update_historical_data()
            if self.simulation_mode: # don't wait 15 minutes for gap to fill
                await self.simulate_bar()
            # await asyncio.sleep(60)  # Wait for 1 minute
                            
    async def run(self):
        try:
            while True:
                await self.wait_for_market_open()
                await self.reset_daily_data()

                self.data_stream.subscribe_bars(self.on_bar, self.symbol)


                update_task = asyncio.create_task(self.update_historical_periodically())
                stream_task = asyncio.create_task(self.data_stream._run_forever())
                
                while self.is_market_open():
                    await asyncio.sleep(1)
                
                self.log("Market closed. Stopping data stream.")
                await self.close()
                self.data_stream.unsubscribe_bars(self.symbol)
                
                update_task.cancel()
                stream_task.cancel()
                # try:
                #     await update_task
                #     await stream_task
                # except asyncio.CancelledError:
                #     pass
                
                self.log("Waiting for next trading day...")
                await asyncio.sleep(2)  # Wait a minute before checking market open status again

        except Exception as e:
            self.log(f"Error in run: {e}", logging.ERROR)

    async def close(self):
        if hasattr(self, 'data_stream'):
            await self.data_stream.stop_ws()
            
    async def run_day_sim(self, current_date: date, sleep_interval: float = 0.1):
        self.simulation_mode = True
        assert self.simulation_mode == True
        
        # Get the day's data
        # start = datetime.combine(current_date, time(4, 0)).replace(tzinfo=self.ny_tz)
        # start = datetime.combine(current_date, time(9, 0)).replace(tzinfo=self.ny_tz)
        
        # end = datetime.combine(current_date + timedelta(days=1), time(4, 0)).replace(tzinfo=self.ny_tz)
        # end = datetime.combine(current_date, time(20, 0)).replace(tzinfo=self.ny_tz)
        
        start = datetime.combine(current_date, time(0, 0)).replace(tzinfo=self.ny_tz)
        end = datetime.combine(current_date + timedelta(days=1), time(0, 0)).replace(tzinfo=self.ny_tz)
        
        day_data = self.get_historical_bars(start, end)
        
        if day_data.empty:
            self.log(f"No data available for {current_date}")
            return
        
        # Initialize data with the first minute
        self.data = day_data.iloc[:2]
        
        # reset data logic
        self.trading_strategy = TradingStrategy(self.calculate_touch_detection_area(), self.strategy_params, is_live_trading=True)
        self.trading_strategy.update_daily_parameters(current_date) # assuming only simulating one day

        text_trap = io.StringIO()
        
        # Simulate each minute of the day
        for timestamp in day_data.index.get_level_values('timestamp')[2:]:
            # self.log(timestamp)
            # Update self.data with all rows up to and including the current timestamp
            self.data = day_data.loc[day_data.index.get_level_values('timestamp') <= timestamp]
            self.last_historical_timestamp = self.data.index.get_level_values('timestamp')[-1]
            
            await self.simulate_bar()
            
            # if len(self.trading_strategy.touch_detection_areas.long_touch_area) + len(self.trading_strategy.touch_detection_areas.short_touch_area) == 0:
            if timestamp.time() < time(11,3):
                sys.stdout = text_trap
                sleep_interval = 0
            else:
                sys.stdout = sys.__stdout__
                sleep_interval = 0.3
            await asyncio.sleep(sleep_interval)
        
        self.log(f"Completed simulation for {current_date}")
        
            
import tracemalloc
tracemalloc.start()

async def main():
    symbol = "AMZN"
    initial_balance = 10000
    simulation_mode = True  # Set this to True for simulation, False for live trading

    touch_detection_params = LiveTouchDetectionParameters(
        symbol=symbol,
        atr_period=15,
        level1_period=15,
        multiplier=1.4,
        min_touches=3,
        # bid_buffer_pct=0.005,
        start_time=None,
        end_time='15:55',
        # end_time='11:20',
        use_median=True,
        touch_area_width_agg=np_median,
        rolling_avg_decay_rate=0.85
    )

    strategy_params = StrategyParameters(
        initial_investment=10_000,
        do_longs=True,
        do_shorts=True,
        sim_longs=True,
        sim_shorts=True,
        
        use_margin=True,
        
        times_buying_power=1,
        
        soft_start_time = None, 
        soft_end_time = '15:50',
    )

    trader = LiveTrader(API_KEY, API_SECRET, symbol, initial_balance, touch_detection_params, strategy_params, simulation_mode)

    try:
        # await trader.run()
        
        
        date_to_simulate = date(2024, 8, 20)
        await trader.run_day_sim(date_to_simulate)
        
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Stopping trader.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        await trader.close()
        print("Trader stopped.")

if __name__ == "__main__":
    asyncio.run(main())