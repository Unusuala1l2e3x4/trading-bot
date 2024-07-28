import asyncio
import pandas as pd
from datetime import datetime, timedelta
from alpaca.data.live.stock import StockDataStream
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient


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




class LiveTrader:
    def __init__(self, api_key, secret_key, symbol, initial_balance):
        self.trading_client = TradingClient(api_key, secret_key, paper=True)
        self.data_stream = StockDataStream(api_key, secret_key)
        self.historical_client = StockHistoricalDataClient(api_key, secret_key)
        self.symbol = symbol
        self.balance = initial_balance
        self.data = pd.DataFrame()
        self.is_ready = False
        self.current_position = None

    async def initialize_data(self):
        end = datetime.now()
        start = end - timedelta(minutes=30)  # Fetch 30 minutes of historical data
        self.data = self.get_historical_bars(start, end)

    def get_historical_bars(self, start, end):
        request_params = StockBarsRequest(
            symbol_or_symbols=self.symbol,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
            adjustment=Adjustment.ALL
        )
        bars = self.historical_client.get_stock_bars(request_params).df
        return bars

    async def on_bar(self, bar):
        bar_data = {
            'timestamp': bar.timestamp,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume
        }
        new_row = pd.DataFrame([bar_data])
        self.data = pd.concat([self.data, new_row], ignore_index=True)

        if not self.is_ready:
            self.check_data_readiness()
        else:
            await self.execute_trading_logic()

    def check_data_readiness(self):
        if len(self.data) > 0:
            latest_timestamp = self.data['timestamp'].max()
            if (datetime.now() - latest_timestamp).total_seconds() / 60 <= 1:  # 1-minute threshold
                self.is_ready = True
                print("Data is ready for trading")

    async def execute_trading_logic(self):
        # Implement your trading logic here
        touch_areas = self.calculate_touch_detection_area()
        if touch_areas:
            await self.backtest_strategy(touch_areas)

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
        order_data = MarketOrderRequest(
            symbol=self.symbol,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.DAY
        )
        order = self.trading_client.submit_order(order_data)
        print(f"Placed {side} order for {qty} shares of {self.symbol}")
        return order

    async def run(self):
        await self.initialize_data()
        self.data_stream.subscribe_bars(self.on_bar, self.symbol)
        await self.data_stream.run()

# Usage
symbol = "AAPL"
initial_balance = 10000

trader = LiveTrader(API_KEY, API_SECRET, symbol, initial_balance)
asyncio.run(trader.run())



