import asyncio
import pandas as pd
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo
from alpaca.data.live.stock import StockDataStream
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetCalendarRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.enums import Adjustment

from time import sleep
from TradePosition import TradePosition, SubPosition, export_trades_to_csv
from TouchArea import TouchArea, TouchAreaCollection

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


import logging
logging.getLogger('alpaca').setLevel(logging.DEBUG)


debug = True

def debug_print(*args, **kwargs):
    if debug:
        print(*args, **kwargs)

class LiveTrader:
    def __init__(self, api_key, secret_key, symbol, initial_balance):
        self.trading_client = TradingClient(api_key, secret_key, paper=True)
        self.data_stream = StockDataStream(api_key, secret_key)
        self.historical_client = StockHistoricalDataClient(api_key, secret_key)
        self.symbol = symbol
        self.balance = initial_balance
        self.data = None
        self.is_ready = False
        self.current_position = None
        self.ny_tz = ZoneInfo("America/New_York")
        
    
    def is_market_open(self, check_time=None):
        if check_time is None:
            check_time = datetime.now(self.ny_tz)
        else:
            # Ensure check_time is in Eastern Time
            check_time = check_time.astimezone(self.ny_tz)
        
        return (check_time.weekday() < 5 and 
                time(4, 0) <= check_time.time() <= time(20, 0))
        
        
    def get_current_trading_day_start(self):
        now = datetime.now(self.ny_tz)
        current_date = now.date()
        if now.time() < time(4, 0):  # If it's before 4:00 AM ET, use the previous day
            current_date -= timedelta(days=1)
        return datetime.combine(current_date, time(4, 0)).replace(tzinfo=self.ny_tz)
    
    
    # IEX (str): Investor's exchange data feed
    # SIP (str): Securities Information Processor feed
    # OTC (str): Over the counter feed
    
    def get_historical_bars(self, start:datetime, end:datetime):
        request_params = StockBarsRequest(
            symbol_or_symbols=self.symbol,
            timeframe=TimeFrame.Minute,
            start=start.astimezone(ZoneInfo("UTC")),
            end=end.astimezone(ZoneInfo("UTC")),
            adjustment=Adjustment.ALL,
            feed='iex' # iex, sip, otc
        )
        df = self.historical_client.get_stock_bars(request_params).df
        df.index = df.index.set_levels(
            pd.Series(df.index.get_level_values('timestamp').to_list()).dt.tz_convert(self.ny_tz),
            level='timestamp'
        )
        df.sort_index(inplace=True)
        return df
    
    async def initialize_data(self):
        try:
            if not self.is_market_open():
                debug_print("Market is closed. Waiting for market to open.")
                return

            # Debug: Uncomment these lines to use a specific time range for testing
            debug_start = datetime(2024, 7, 24, 9, 30).replace(tzinfo=None)#.replace(tzinfo=self.ny_tz)
            debug_end = datetime(2024, 7, 24, 16, 0).replace(tzinfo=None)#.replace(tzinfo=self.ny_tz)
            
            # - timedelta(minutes=15,seconds=1) needed for SIP but not IEX
            
            end = datetime.now(self.ny_tz) - timedelta(minutes=15,seconds=1)
            start = self.get_current_trading_day_start()

            # # Debug: Uncomment these lines to use the debug time range
            # start, end = debug_start, debug_end
            # print('initialize_data - DEBUG MODE DATES:',start, end)
            
            debug_print(f"Fetching historical data from {start} to {end}")
            
            self.data = self.get_historical_bars(start, end)
            
            if self.data.empty:
                debug_print("No historical data available. Waiting for data.")
                return
            
            self.data.index = self.data.index.set_levels(pd.Series(self.data.index.get_level_values('timestamp').to_list()).map(lambda x: x.tz_convert(self.ny_tz)), level='timestamp')
            self.data.sort_index(inplace=True)
            
            print(self.data.index.get_level_values('timestamp'))
            
            latest_data_time = pd.to_datetime(self.data.index.get_level_values('timestamp').max())#.tz_convert(self.ny_tz)
            time_diff = (end - latest_data_time).total_seconds() / 60
            
            print(latest_data_time, end, time_diff)

            if time_diff > 16:  # Allow for a small buffer beyond 15 minutes
                debug_print(f"Historical data is too old. Latest data point: {latest_data_time}")
                return

            debug_print(f"Initialized historical data: {len(self.data)} bars")
            debug_print(f"Data range: {self.data.index.get_level_values('timestamp').min()} to {latest_data_time}")
            self.is_ready = True

        except Exception as e:
            debug_print(f"Error initializing data: {e}")
            

    async def on_bar(self, bar):
        debug_print('on_bar')
        
        debug_print(f"Received bar: {bar}")

        try:
            if not self.is_market_open():
                debug_print("Received bar outside market hours. Ignoring.")
                return

            bar_time = pd.to_datetime(bar.t).tz_convert(self.ny_tz)
            now = datetime.now(self.ny_tz)
            time_diff = (now - bar_time).total_seconds() / 60

            if time_diff > 1:  # If the bar is more than 1 minute old
                debug_print(f"Received outdated bar data. Bar time: {bar_time}, Current time: {now}")
                return

            bar_data = {
                'timestamp': bar_time,
                'open': bar.o,
                'high': bar.h,
                'low': bar.l,
                'close': bar.c,
                'volume': bar.v,
            }
            new_row = pd.DataFrame([bar_data])
            new_row.set_index('timestamp', inplace=True)
            self.data = pd.concat([self.data, new_row])
            debug_print(f"Received bar: {bar_data}")

            await self.execute_trading_logic()

        except Exception as e:
            debug_print(f"Error in on_bar: {e}")

    async def execute_trading_logic(self):
        debug_print('execute_trading_logic')
        try:
            if not self.is_ready:
                debug_print("Data not ready for trading")
                return

            # Implement your trading logic here
            touch_areas = self.calculate_touch_detection_area()
            if touch_areas:
                await self.backtest_strategy(touch_areas)
        except Exception as e:
            debug_print(f"Error in execute_trading_logic: {e}")

    def calculate_touch_detection_area(self):
        # Implement your touch detection logic here
        # This should be adapted from your backtesting code
        pass

    async def backtest_strategy(self, touch_areas):
        # Implement your strategy logic here
        # This should be adapted from your backtesting code
        # Use self.place_order() to execute trades
        pass

    async def place_order(self, side, qty):
        try:
            if not self.is_market_open():
                debug_print("Market is closed. Cannot place order.")
                return

            order_data = MarketOrderRequest(
                symbol=self.symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY
            )
            order = self.trading_client.submit_order(order_data)
            debug_print(f"Placed {side} order for {qty} shares of {self.symbol}")
            return order
        except Exception as e:
            debug_print(f"Error placing order: {e}")

    async def run(self):
        try:
            while True:
                if self.is_market_open(): # set return to True when testing outsite horus
                    if not self.is_ready:
                        print('self.initialize_data()...')
                        await self.initialize_data()
                        print('...done')

                    print('self.is_ready',self.is_ready)
                    if self.is_ready:
                        print('self.data_stream.subscribe_bars...')
                        self.data_stream.subscribe_bars(self.on_bar, self.symbol)
                        print('...done')
                        
                        # print('self.data_stream.run()...')
                        # await self.data_stream.run()
                        
                        # print('BaseStream_run_custom(self.data_stream)...')
                        # await BaseStream_run_custom(self.data_stream)
                        
                        print('self.data_stream._run_forever()...')
                        await self.data_stream._run_forever()
                        
                        print('...done')
                else:
                    debug_print("Market is closed. Waiting for market to open.")
                    await asyncio.sleep(60)  # Check every minute
                    
                sleep(2)
        except Exception as e:
            debug_print(f"Error in run: {e}")
            
    async def close(self):
        if hasattr(self, 'data_stream'):
            await self.data_stream.stop_ws()
            # await self.data_stream.close()
           
        
# async def BaseStream_run_custom(data_stream) -> None:
#     """Starts up the websocket connection's event loop"""
#     try:
#         print('BaseStream_run_custom - data_stream._run_forever()...')
#         await data_stream._run_forever()
#         print('...done')
#     except KeyboardInterrupt:
#         print("keyboard interrupt, bye")
#         pass
#     finally:
#         data_stream.stop()
            
            
            
import tracemalloc
tracemalloc.start()


async def main():
    symbol = "AAPL"
    initial_balance = 10000
    trader = LiveTrader(API_KEY, API_SECRET, symbol, initial_balance)
    
    timeout = 120
    
    try:
        # Wait for up to 2 minutes (120 seconds) to receive data
        await asyncio.wait_for(trader.run(), timeout=timeout)
    except asyncio.TimeoutError:
        print(f"No data received within {timeout} seconds timeout. This may be normal outside of market hours.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        await trader.close()
        print("Trader stopped.")

if __name__ == "__main__":
    asyncio.run(main())