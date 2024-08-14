import asyncio
import pandas as pd
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo
from alpaca.data.live.stock import StockDataStream
from alpaca.data.requests import StockBarsRequest
from alpaca.data.enums import DataFeed
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetCalendarRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.enums import Adjustment

import logging


import os, toml
from dotenv import load_dotenv

STANDARD_DATETIME_STR = '%Y-%m-%d %H:%M:%S'
ROUNDING_DECIMAL_PLACES = 10  # Choose an appropriate number of decimal places

load_dotenv(override=True)
livepaper = os.getenv('LIVEPAPER')
config = toml.load('../config.toml')

# Replace with your Alpaca API credentials
API_KEY = config[livepaper]['key']
API_SECRET = config[livepaper]['secret']



debug = True
def debug_print(*args, **kwargs):
    if debug:
        print(*args, **kwargs)


class LiveTrader:
    def __init__(self, api_key, secret_key, symbol, initial_balance):
        self.trading_client = TradingClient(api_key, secret_key, paper=True)
        self.data_stream = StockDataStream(api_key, secret_key, feed=DataFeed.IEX, websocket_params={"ping_interval": 1,"ping_timeout": 180,"max_queue": 1024})
        self.historical_client = StockHistoricalDataClient(api_key, secret_key)
        self.symbol = symbol
        self.balance = initial_balance
        self.data = pd.DataFrame()
        self.is_ready = False
        self.gap_filled = False
        self.streamed_data = pd.DataFrame()
        self.last_historical_timestamp = None
        self.first_streamed_timestamp = None
        self.open_positions = {}
        self.ny_tz = ZoneInfo("America/New_York")
        self.logger = self.setup_logger()

    def setup_logger(self):
        logger = logging.getLogger('LiveTrader')
        logger.setLevel(logging.DEBUG)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def log(self, message, level=logging.INFO):
        self.logger.log(level, message)

    def is_market_open(self, check_time=None):
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
        df.index = df.index.set_levels(
            df.index.get_level_values('timestamp').tz_convert(self.ny_tz),
            level='timestamp'
        )
        df.sort_index(inplace=True)
        return df


    async def initialize_data(self):
        try:
            end = datetime.now(self.ny_tz) - timedelta(minutes=15, seconds=1)
            start = self.get_current_trading_day_start()
            
            self.log(f"Fetching historical data from {start} to {end}")
            
            self.data = self.get_historical_bars(start, end)
            
            if self.data.empty:
                self.log("No historical data available. Waiting for data.", logging.WARNING)
                return
            
            self.last_historical_timestamp = self.data.index.get_level_values('timestamp').max()
            time_diff = (end - self.last_historical_timestamp).total_seconds() / 60

            if time_diff >= 16:
                self.log(f"Historical data is too old. Latest data point: {self.last_historical_timestamp}", logging.WARNING)
                return

            self.log(f"Initialized historical data: {len(self.data)} bars")
            self.log(f"Data range: {self.data.index.get_level_values('timestamp').min()} to {self.last_historical_timestamp}")

        except Exception as e:
            self.log(f"Error initializing data: {e}", logging.ERROR)

    async def update_historical_data(self):
        try:
            end = datetime.now(self.ny_tz) - timedelta(minutes=15, seconds=1)
            start = self.last_historical_timestamp + timedelta(minutes=1)
            
            new_data = self.get_historical_bars(start, end)
            
            if not new_data.empty:
                self.data = pd.concat([self.data, new_data])
                debug_print('update_historical_data')
                debug_print('before remove dups',len(self.data))
                self.data = self.data.loc[~self.data.index.duplicated(keep='last')].sort_index() # probably not necessary, but test to make sure
                debug_print('after remove dups ',len(self.data))
                self.last_historical_timestamp = self.data.index.get_level_values('timestamp').max()
                self.log(f"Updated historical data. New range: {self.data.index.get_level_values('timestamp').min()} to {self.last_historical_timestamp}")

        except Exception as e:
            self.log(f"Error updating historical data: {e}", logging.ERROR)

    async def on_bar(self, bar):
        
        try:
            if not self.is_market_open():
                self.log("Received bar outside market hours. Ignoring.")
                return

            bar_time = pd.to_datetime(bar.timestamp).tz_convert(self.ny_tz)
            now = datetime.now(self.ny_tz)
            time_diff = (now - bar_time).total_seconds() / 60

            if time_diff >= 16:
                self.log(f"Received outdated bar data. Bar time: {bar_time}, Current time: {now}", logging.WARNING)
                return

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
            self.streamed_data = pd.concat([self.streamed_data, new_row])
            debug_print('on_bar streamed_data')
            debug_print('before remove dups',len(self.streamed_data))
            self.streamed_data = self.streamed_data.loc[~self.streamed_data.index.duplicated(keep='last')].sort_index()
            debug_print('after remove dups ',len(self.streamed_data))
            self.log(f"Added streamed bar. {len(self.streamed_data)} streamed bars total.")

            if self.first_streamed_timestamp is None:
                self.first_streamed_timestamp = bar_time
            
            # Check if we've filled the gap
            if not self.is_ready:
                await self.check_gap_filled()
            else:
                self.data = pd.concat([self.data, new_row])
                debug_print('on_bar data')
                debug_print('before remove dups',len(self.streamed_data))
                self.data = self.data.loc[~self.data.index.duplicated(keep='last')].sort_index()
                debug_print('after remove dups ',len(self.streamed_data))
                
                debug_print('data final:',self.data)
                
                await self.execute_trading_logic()

        except Exception as e:
            self.log(f"Error in on_bar: {e}", logging.ERROR)

    async def check_gap_filled(self):
        # if len(self.streamed_data) >= 15:  # We have at least 15 minutes of streamed data
        time_diff = (self.first_streamed_timestamp - self.last_historical_timestamp).total_seconds() / 60
        # if abs(time_diff - 1) <= 0.1 or time_diff <= 0:  # Allow for a small tolerance due to potential timing issues
        if time_diff <= 1:
            self.is_ready = True
            
            # debug_print('data:')
            # debug_print(self.data)
            # debug_print('streamed_data:')
            # debug_print(self.streamed_data)
            
            self.data = pd.concat([self.data, self.streamed_data])
            debug_print('check_gap_filled')
            debug_print('before remove dups',len(self.data))
            self.data = self.data.loc[~self.data.index.duplicated(keep='last')].sort_index()
            debug_print('after remove dups ',len(self.data))
            self.log("Gap filled. Data is now ready for trading.")
            self.log(f"Continuous data range: {self.data.index.get_level_values('timestamp').min()} to {self.data.index.get_level_values('timestamp').max()}")
        else:
            self.log(f"Gap not properly filled. Time difference: {time_diff} minutes", logging.WARNING)


    async def execute_trading_logic(self):
        try:
            if not self.is_ready:
                self.log("Data not ready for trading")
                return

            self.log("Data READY for trading. Creating touch detection areas...")
            # Implement your trading logic here
            touch_areas = self.calculate_touch_detection_area()
            if touch_areas:
                await self.process_touch_areas(touch_areas)
        except Exception as e:
            self.log(f"Error in execute_trading_logic: {e}", logging.ERROR)

    def calculate_touch_detection_area(self):
        # Implement your touch detection logic here
        # This should be adapted from your backtesting code
        pass

    async def process_touch_areas(self, touch_areas):
        # Implement your strategy logic here
        # This should be adapted from your backtesting code
        # Use self.place_order() to execute trades
        pass

    async def place_order(self, side, qty, order_type=OrderType.MARKET, limit_price=None):
        try:
            if not self.is_market_open():
                self.log("Market is closed. Cannot place order.")
                return

            order_data = MarketOrderRequest(
                symbol=self.symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY
            )
            
            # if order_type == OrderType.LIMIT and limit_price is not None:
            #     order_data.type = "limit"
            #     order_data.limit_price = limit_price

            if self.validate_order(order_data):
                order = self.trading_client.submit_order(order_data)
                self.log(f"Placed {side} order for {qty} shares of {self.symbol}")
                return order
            else:
                self.log("Order validation failed", logging.WARNING)
                return None

        except Exception as e:
            self.log(f"Error placing order: {e}", logging.ERROR)

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
    
    async def run(self):
        try:
            while True:
                await self.wait_for_market_open()
                if self.data.empty:
                    await self.initialize_data()

                self.data_stream.subscribe_bars(self.on_bar, self.symbol)
                
                async def update_historical_periodically():
                    while not self.is_ready:
                        await asyncio.sleep(59)  # Wait for 1 minute
                        await self.update_historical_data()

                update_task = asyncio.create_task(update_historical_periodically())
                await self.data_stream._run_forever()
                update_task.cancel()  # Cancel the update task when the stream stops
                
                await asyncio.sleep(1)
        except Exception as e:
            self.log(f"Error in run: {e}", logging.ERROR)

    async def close(self):
        if hasattr(self, 'data_stream'):
            await self.data_stream.stop_ws()
            
            
import tracemalloc
tracemalloc.start()


async def main():
    symbol = "AAPL"
    initial_balance = 10000
    trader = LiveTrader(API_KEY, API_SECRET, symbol, initial_balance)
    
    run_task = asyncio.create_task(trader.run())
    
    try:
        start_time = asyncio.get_event_loop().time()
        while True:
            if trader.is_receiving_data():
                print("Receiving data successfully.")
                break
            
            if asyncio.get_event_loop().time() - start_time > 240:
                raise TimeoutError("No data received within 240 seconds.")
            
            await asyncio.sleep(10)  # Check every 10 seconds
        
        # If we've reached here, we're receiving data. Let the trader run indefinitely.
        print("Trader is now running indefinitely.")
        await run_task
        
    except TimeoutError as e:
        print(f"{e} This is normal outside of market hours for IEX feed.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        run_task.cancel()
        try:
            await run_task
        except asyncio.CancelledError:
            pass
        await trader.close()
        print("Trader stopped.")

if __name__ == "__main__":
    asyncio.run(main())