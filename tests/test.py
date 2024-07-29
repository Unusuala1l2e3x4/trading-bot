import asyncio
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.enums import Adjustment
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

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

    async def initialize_data(self):
        try:
            debug_start = datetime(2024, 7, 24, 9, 30).replace(tzinfo=None)#.replace(tzinfo=self.ny_tz)
            debug_end = datetime(2024, 7, 24, 16, 0).replace(tzinfo=None)#.replace(tzinfo=self.ny_tz)
            
            
            end = datetime.now(self.ny_tz)
            start = end.replace(hour=4, minute=0, second=0, microsecond=0)
            if end.time() < datetime.time(4, 0):
                start -= timedelta(days=1)

            debug_print(f"Fetching historical data from {start} to {end}")
            
            self.data = self.get_historical_bars(start, end)
            
            if self.data.empty:
                debug_print("No historical data available. Waiting for data.")
                return

            latest_data_time = self.data.index.get_level_values('timestamp').max().tz_convert(self.ny_tz)
            debug_print(f"Initialized historical data: {len(self.data)} bars")
            debug_print(f"Data range: {self.data.index.get_level_values('timestamp').min()} to {latest_data_time}")
            self.is_ready = True

        except Exception as e:
            debug_print(f"Error initializing data: {e}")

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
        try:
            debug_print(f"Received bar: {bar}")
            # Process the bar data here
            # Update your strategy, make trading decisions, etc.
        except Exception as e:
            debug_print(f"Error in on_bar: {e}")

    async def run(self):
        try:
            await self.initialize_data()
            
            async def _process_bar(bar):
                await self.on_bar(bar)

            self.data_stream.subscribe_bars(_process_bar, self.symbol)
            await self.data_stream.run()
        except Exception as e:
            debug_print(f"Error in run: {e}")
