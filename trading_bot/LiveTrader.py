import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta, date
from zoneinfo import ZoneInfo
from alpaca.data.live.stock import StockDataStream
from alpaca.data.requests import StockBarsRequest, StockQuotesRequest, StockLatestQuoteRequest
from alpaca.data.enums import DataFeed
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.stream import TradingStream
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, StopOrderRequest, LimitOrderRequest, StopLimitOrderRequest, TrailingStopOrderRequest, GetCalendarRequest
from alpaca.trading.enums import OrderSide, OrderStatus, TimeInForce, OrderType, TradeEvent
from alpaca.trading.models import Order, TradeUpdate
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.enums import Adjustment
from types import SimpleNamespace
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Set
from collections import defaultdict

from alpaca.data.models import Bar, Quote

from trading_bot.TradePosition import TradePosition
# from trading_bot.TouchArea import TouchArea
from trading_bot.TradingStrategy import StrategyParameters, TouchDetectionAreas, TradingStrategy, is_security_shortable_and_etb, is_security_marginable, IntendedOrder
from trading_bot.TouchDetection import calculate_touch_detection_area, plot_touch_detection_areas, LiveTouchDetectionParameters, TouchDetectionState
from trading_bot.TouchDetectionParameters import np_mean, np_median
from trading_bot.MultiSymbolDataRetrieval import retrieve_bar_data, retrieve_quote_data, fill_missing_data, get_data_with_retry, clean_quotes_data, get_stock_latest_quote_with_retry
from SlippageData import SlippageData
from OrderTracker import OrderTracker, OrderState, OrderFill

from uuid import UUID

from tqdm import tqdm
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


debug = True
def debug_print(*args, **kwargs):
    if debug:
        print(*args, **kwargs)


def setup_logger(log_level=logging.INFO):
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

logger = setup_logger(logging.INFO)

def log(message, level=logging.INFO):
    logger.log(level, message, exc_info=level >= logging.ERROR) # show trackback if level is ERROR
    


def fill_latest_missing_data(df, latest_timestamp):
    # Get the last timestamp from the dataframe
    last_timestamp = df.index.get_level_values('timestamp')[-1]

    # Ensure the latest timestamp is greater than the last timestamp
    if latest_timestamp > last_timestamp:
        # Get the symbol for the entire dataframe (assumes only one unique symbol)
        symbol = df.index.get_level_values('symbol')[0]
        
        # Create a date range from the last timestamp + 1 minute up to the latest timestamp
        missing_timestamps = pd.date_range(
            start=last_timestamp + pd.Timedelta(minutes=1),
            end=latest_timestamp,
            freq='min',
            tz=last_timestamp.tz
        )
        
        # Get the last row of the DataFrame to replicate values
        val = df['close'].iloc[-1]

        # Iterate over missing timestamps and set values directly (to ensure in place)
        for timestamp in missing_timestamps:
            df.loc[(symbol, timestamp), :] = {
                'open': val,
                'high': val,
                'low': val,
                'close': val,
                'volume': 0,
                'trade_count': 0
            }


def combine_two_orders_same_symbol(orders_list: List[IntendedOrder]) -> List[IntendedOrder]:
    """
    Combine two orders for the same symbol into a single net order when possible.
    Only combines close->open order pairs in the same direction (long->long or short->short).
    
    Args:
        orders_list: List of 1-2 IntendedOrder objects to potentially combine
        
    Returns:
        List containing either the original orders or a single combined net order
        
    Raises:
        ValueError: If more than 2 orders provided
        AssertionError: If orders don't meet combining requirements
    """
    # Validate input length
    assert len(orders_list) <= 2
    if not orders_list:
        return orders_list
    
    # Return single orders unchanged
    if len(orders_list) == 1:
        assert orders_list[0].qty > 0, f"Order quantity must be positive: {orders_list[0].qty}"
        return orders_list

    if len(orders_list) == 2:
        first_order, second_order = orders_list
        
        # Validate basic order properties
        assert first_order.qty > 0 and second_order.qty > 0, \
            f"Order quantities must be positive: {first_order.qty}, {second_order.qty}"
        assert first_order.symbol == second_order.symbol, \
            f"Orders must be for same symbol: {first_order.symbol} != {second_order.symbol}"
        
        # Validate order sequence
        assert first_order.action == 'close' and second_order.action == 'open', \
            f"When 2 orders are present, the first must be a close and the second must be an open. ({first_order.action}, {second_order.action})"

        # Validate order sides match position directions
        assert (first_order.side == OrderSide.SELL) == first_order.position.is_long, \
            f"First order side {first_order.side} doesn't match position direction {first_order.position.is_long}"
        assert (second_order.side == OrderSide.BUY) == second_order.position.is_long, \
            f"Second order side {second_order.side} doesn't match position direction {second_order.position.is_long}"

        # Case 1: Same order side (closing one direction and opening the opposite)
        # NOTE: this should not happen given changes to TradingStrategy.process_active_areas function
        if first_order.side == second_order.side:
            assert first_order.position.is_long != second_order.position.is_long, \
                f"Orders with same side must be switching directions: {first_order.position.is_long} != {second_order.position.is_long}"
            # These orders must be executed sequentially (Alpaca prevents switching between long and short in a single order)
            log(f"Received long-to-short or short-to-long order pair. Alpaca prevents switching between long and short in a single order.",
                level=logging.WARNING)
            return orders_list

        # Case 2: Different order side (closing and reopening in the same direction)
        else:
            assert first_order.position.is_long == second_order.position.is_long, \
                f"Orders with different sides must maintain same direction: {first_order.position.is_long} == {second_order.position.is_long}"
            
            # Calculate net quantity change needed
            qty_difference = first_order.qty - second_order.qty
            
            # No order needed if quantities exactly match
            if qty_difference == 0:
                return []
            
            # Create combined order
            combined_order = IntendedOrder(
                action='net_partial_exit' if qty_difference > 0 else 'net_partial_entry',
                side=first_order.side if qty_difference > 0 else second_order.side,
                symbol=second_order.symbol,
                qty=abs(qty_difference),
                price=second_order.price,  # Use second order price as it's more recent
                position=second_order.position,  # Use new position for tracking
                fees=first_order.fees + second_order.fees # NOTE: overestimating fees for now
            )
            
            return [combined_order]

    # If there are more than 2 orders, raise an error
    raise ValueError("More than 2 orders in a minute is not supported")

@dataclass
class LiveTrader:
    api_key: str
    secret_key: str
    symbol: str
    touch_detection_params: LiveTouchDetectionParameters
    strategy_params: StrategyParameters
    simulation_mode: bool = False
    
    slippage_data_path: str = None
    slippage_data_list: List[SlippageData] = field(default_factory=list)
    
    trading_client: TradingClient = field(init=False)
    data_stream: StockDataStream = field(init=False)
    trading_stream: TradingStream = field(init=False)
    historical_client: StockHistoricalDataClient = field(init=False)
    trading_strategy: Optional[TradingStrategy] = None
    bars: pd.DataFrame = field(default_factory=pd.DataFrame)
    bars_adjusted: pd.DataFrame = field(default_factory=pd.DataFrame)
    prev_quotes_raw: pd.DataFrame = field(default_factory=pd.DataFrame)
    trade_updates: List[TradeUpdate] = field(default_factory=list)
    is_ready: bool = False
    gap_filled: bool = False
    streamed_bars: pd.DataFrame = field(default_factory=pd.DataFrame)
    last_hist_bar_dt: Optional[pd.Timestamp] = None
    first_streamed_timestamp: Optional[pd.Timestamp] = None
    timer_start: datetime = field(init=False)
    open_positions: Dict = field(default_factory=dict)
    area_ids_to_remove: defaultdict = field(default_factory=lambda: defaultdict(set))
    area_ids_to_side_switch: defaultdict = field(default_factory=lambda: defaultdict(set))
    ny_tz: ZoneInfo = field(default_factory=lambda: ZoneInfo("America/New_York"))
    trades: List[TradePosition] = field(default_factory=list)
    
    touch_detection_state: TouchDetectionState = None
    
    
    order_tracker: OrderTracker = None
    reconciliation_lock: asyncio.Lock = None

    def __post_init__(self):
        self.trading_client = TradingClient(self.api_key, self.secret_key, paper= livepaper == 'paper')
        self.data_stream = StockDataStream(self.api_key, self.secret_key, feed=DataFeed.IEX, 
                                           websocket_params={"ping_interval": 2, "ping_timeout": 180, "max_queue": 1024})
        self.trading_stream = TradingStream(self.api_key, self.secret_key, paper= livepaper == 'paper',
                                           websocket_params={"ping_interval": 2, "ping_timeout": 180, "max_queue": 1024})
        self.historical_client = StockHistoricalDataClient(self.api_key, self.secret_key)
        
        self.order_tracker = OrderTracker()
        self.reconciliation_lock = asyncio.Lock()  # Prevent concurrent reconciliation
            
    async def reset_daily_data(self):
        log("Resetting daily data...")
        current_day_start = self.get_current_trading_day_start()
        
        # Reset data and streamed_bars
        if not self.bars.empty:
            # self.bars = self.bars[self.bars.index.get_level_values('timestamp') >= current_day_start]
            self.bars = pd.DataFrame()
        self.streamed_bars = pd.DataFrame()
        
        # Reset other daily variables
        self.is_ready = False
        self.gap_filled = False
        self.last_hist_bar_dt = None
        self.first_streamed_timestamp = None
        
        # Re-initialize historical data for the new day
        await self.initialize_data()

        # Initialize or update TradingStrategy
        if self.trading_strategy is None:
            self.trading_strategy = TradingStrategy(self.calculate_touch_detection_area(), self.strategy_params, is_live_trading=True)
        else:
            self.trading_strategy.touch_detection_areas = self.calculate_touch_detection_area(self.trading_strategy.touch_detection_areas.market_hours)

        # Update daily parameters in TradingStrategy
        current_date = datetime.now(self.ny_tz).date()
        self.trading_strategy.update_daily_parameters(current_date)
        
        self.touch_detection_state = None
        
        log("Daily data reset complete.")
    
    def is_market_open(self, check_time: Optional[datetime] = None):
        if check_time is None:
            check_time = datetime.now(self.ny_tz)
        else:
            check_time = check_time.astimezone(self.ny_tz)
        
        return (check_time.weekday() < 5 and 
                time(4, 0) <= check_time.time() < time(20, 0))

    def get_current_trading_day_start(self):
        now = datetime.now(self.ny_tz)
        current_date = now.date()
        # if now.time() < time(4, 0):
        #     current_date -= timedelta(days=1)
        return datetime.combine(current_date, time(4, 0)).replace(tzinfo=self.ny_tz)

    async def wait_for_market_open(self):
        while not self.is_market_open():
            log("Market is closed. Waiting for market to open.")
            await asyncio.sleep(60)

    def get_historical_bars(self, start, end, adjustment = Adjustment.RAW):
        request_params = StockBarsRequest(
            symbol_or_symbols=self.symbol,
            timeframe=TimeFrame.Minute,
            start=start.astimezone(ZoneInfo("UTC")),
            end=end.astimezone(ZoneInfo("UTC")),
            adjustment=adjustment,
            feed='sip'
        )
        try:
            df = get_data_with_retry(self.historical_client, self.historical_client.get_stock_bars, request_params)
        except Exception as e:
            log(f"Error requesting bars for {self.symbol}: {str(e)}", level=logging.ERROR)
            return pd.DataFrame()
        if not df.empty:
            df.index = df.index.set_levels(
                df.index.get_level_values('timestamp').tz_convert(self.ny_tz) + timedelta(minutes=1),
                level='timestamp'
            )
        df.sort_index(inplace=True)
        return fill_missing_data(df) # only fills betweem min and max time already in df
    
    def get_historical_quotes(self, start, end):
        request_params = StockQuotesRequest(
            symbol_or_symbols=self.symbol,
            start=start.astimezone(ZoneInfo("UTC")),
            end=end.astimezone(ZoneInfo("UTC")),
            feed='sip' # default for market data subscription?
            # feed='iex'
        )
        try:
            df = get_data_with_retry(self.historical_client, self.historical_client.get_stock_quotes, request_params)
        except Exception as e:
            log(f"Error requesting quotes for {self.symbol}: {str(e)}", level=logging.ERROR)
            return pd.DataFrame()
        df, _ = clean_quotes_data(df, False, start, end, calculate_durations=False)
        return df
    
    def get_latest_quote(self):
        request_params = StockLatestQuoteRequest(
            symbol_or_symbols=self.symbol,
            feed='sip' # default for market data subscription?
            # feed='iex'
        )
        try:
            ret = get_stock_latest_quote_with_retry(self.historical_client, request_params)
        except Exception as e:
            log(f"Error requesting latest quote for {self.symbol}: {str(e)}", level=logging.ERROR)
            return None
        return ret

    def get_lagged_time(self):
        # return datetime.now(self.ny_tz) - timedelta(minutes=15, seconds=30)
        now = datetime.now(self.ny_tz).replace(second=0, microsecond=0)
        # return now - timedelta(minutes=16)
        return now # allowed with market data subscription
    
    async def initialize_data(self):
        try:
            end = self.get_lagged_time()
            start = self.get_current_trading_day_start()
            
            log(f"Fetching historical data from {start} to {end}")
            
            self.bars = self.get_historical_bars(start, end)
            
            if self.bars.empty:
                log("No historical data available. Waiting for data.", logging.WARNING)
                return
            
            self.last_hist_bar_dt = self.bars.index.get_level_values('timestamp')[-1]
            minutes_diff = (end - self.last_hist_bar_dt).total_seconds() / 60

            if minutes_diff >= 16:
                log(f"Historical data is too old. Latest data point: {self.last_hist_bar_dt}", logging.WARNING)
                # return

            log(f"Initialized historical data: {len(self.bars)} bars")
            log(f"Data range: {self.bars.index.get_level_values('timestamp')[0]} to {self.last_hist_bar_dt}")

        except Exception as e:
            log(f"{type(e).__qualname__} initializing data: {e}", logging.ERROR)
            raise e

    async def update_historical_data(self):
        try:
            end = self.get_lagged_time()
            start = self.last_hist_bar_dt + timedelta(minutes=1)
            debug_print('---')
            if end > start:
                new_data = self.get_historical_bars(start, end)
                if not new_data.empty:
                    assert new_data.index.get_level_values('timestamp').min() > self.last_hist_bar_dt, (new_data.index.get_level_values('timestamp').min(), self.last_hist_bar_dt)

                    debug_print('---')
                    self.bars = pd.concat([self.bars, new_data])
                    debug_print('update_historical_data')
                    debug_print('before remove dups',len(self.bars))
                    self.bars = self.bars.loc[~self.bars.index.duplicated(keep='last')].sort_index() # needed if update_historical_data is called in less than 1 minute intervals
                    debug_print('after remove dups ',len(self.bars))
                    self.last_hist_bar_dt = self.bars.index.get_level_values('timestamp')[-1]
                    log(f"Updated historical data. New range: {self.bars.index.get_level_values('timestamp')[0]} to {self.last_hist_bar_dt}")
                
                assert self.bars.index.get_level_values('timestamp')[-1] == self.last_hist_bar_dt
                
                if self.last_hist_bar_dt < end:
                    log(f"calling fill_latest_missing_data for {self.last_hist_bar_dt} -> {end}", level=logging.WARNING)
                    # print(self.bars)
                    fill_latest_missing_data(self.bars, end)
                    # print(self.bars)
                    self.last_hist_bar_dt = self.bars.index.get_level_values('timestamp')[-1]
        
        except Exception as e:
            log(f"{type(e).__qualname__} updating historical data: {e}", logging.ERROR)
            raise e
                
    async def simulate_bar(self):
        try:
            if self.simulation_mode and not self.bars.empty:
                latest_data = self.bars.iloc[-1]
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
        except Exception as e:
            log(f"{type(e).__qualname__} in simulate_bar: {e}", logging.ERROR)
            raise e
        
    async def on_bar(self, bar:Bar, check_time: Optional[datetime] = None):
        self.timer_start = datetime.now()
        # debug_print('on_bar') # ,bar
        # log("on_bar")
        try:
            if not self.is_market_open(check_time):
                log("Received bar outside market hours. Ignoring.")
                return

            if hasattr(bar, 'is_simulate_bar'):
                is_simulate_bar = bar.is_simulate_bar
                assert self.simulation_mode
            else:
                is_simulate_bar = False
                assert not self.simulation_mode
                
            bar_time = pd.to_datetime(bar.timestamp, utc=True).tz_convert(self.ny_tz)


            # NOTE: there can be updated bars sent 30 seconds after the initial bar to account for any late-reported trades.
            # These updated bars will have the same timestamp as the original bar but with updated data.
            # https://forum.alpaca.markets/t/why-is-a-small-subset-of-1m-ohlcv-bars-delayed-by-30s-from-sip-websockets-data-connection/6207
            
            # OR, instead of ignoring updated repeat-timestamp bars, accept them as normal bars. 
            # May allow performance to be closer to backtesting but  slightly increases transaction costs.
            
            # VERDICT: ignore these bars. very little benefit.
            
            # TODO: double check we are comparing with right data point (or just compare with both)
            if not self.simulation_mode and (bar_time == self.streamed_bars.index.get_level_values('timestamp')[-1] or \
                                             bar_time == self.bars.index.get_level_values('timestamp')[-1]):
                log(f"Skipping repeated bar {bar}")
                return
            
            now = datetime.now(self.ny_tz)
            minutes_diff = (now - bar_time).total_seconds() / 60
            
            # if not is_simulate_bar:
            #     assert 0.5 < minutes_diff < 1.5, minutes_diff # NOTE: bar timestamp is at least 1 minute before current time.

            if minutes_diff >= 16 and not is_simulate_bar:
                log(f"Received outdated bar data. Bar time: {bar_time}, Current time: {now}", logging.WARNING)
                return
            
            if self.first_streamed_timestamp is None:
                self.first_streamed_timestamp = bar_time
                
            assert self.symbol == bar.symbol
            bar_data = {
                'symbol': bar.symbol,
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
            # debug_print('new_row',new_row)
            
            if not is_simulate_bar:
                self.streamed_bars = pd.concat([self.streamed_bars, new_row])
                debug_print('on_bar streamed_bars')
                debug_print('before remove dups',len(self.streamed_bars))
                self.streamed_bars = self.streamed_bars.loc[~self.streamed_bars.index.duplicated(keep='last')].sort_index()
                debug_print('after remove dups ',len(self.streamed_bars))
                log(f"Added streamed bar. {len(self.streamed_bars)} streamed bars total.")

            # Check if we've filled the gap
            if not self.is_ready:
                await self.check_gap_filled()
                
            if self.is_ready:
                if not self.simulation_mode:
                    self.bars = pd.concat([self.bars, new_row])
                    debug_print('on_bar data')
                    debug_print('before remove dups',len(self.bars))
                    self.bars = self.bars.loc[~self.bars.index.duplicated(keep='last')].sort_index()
                    debug_print('after remove dups ',len(self.bars))
                    debug_print('final data:\n',self.bars)
                    # debug_print('final streamed_bars:\n',self.streamed_bars)
                await self.execute_trading_logic()
                
        except Exception as e:
            log(f"{type(e).__qualname__} in on_bar: {e}", logging.ERROR)
            raise e

    # async def on_quote(self, quote:Quote):
    #     try:
    #         self.trading_strategy.latest_live_bid_price = quote.bid_price
    #         self.trading_strategy.latest_live_ask_price = quote.ask_price

    #     except Exception as e:
    #         log(f"{type(e).__qualname__} in on_bar: {e}", logging.ERROR)
    #         raise e
        
        
    async def check_gap_filled(self):
        minutes_diff = (self.first_streamed_timestamp - self.last_hist_bar_dt).total_seconds() / 60
        if minutes_diff <= 1 or self.simulation_mode:
            self.is_ready = True
            debug_print('check_gap_filled: self.is_ready = True',minutes_diff)
            if not self.simulation_mode: # do not use streamed_bars if in simulation mode
                self.bars = pd.concat([self.bars, self.streamed_bars])
                debug_print('before remove dups',len(self.bars))
                self.bars = self.bars.loc[~self.bars.index.duplicated(keep='last')].sort_index()
                debug_print('after remove dups ',len(self.bars))
            log("Gap filled. Data is now ready for trading.")
            log(f"Continuous data range: {self.bars.index.get_level_values('timestamp')[0]} to {self.bars.index.get_level_values('timestamp')[-1]}")
        else:
            log(f"Gap not properly filled. Time difference: {minutes_diff} minutes", logging.WARNING)


    async def execute_trading_logic(self):
        log('execute_trading_logic',level=logging.DEBUG)
        current_time = None
        try:
            if not self.is_ready:
                log("Data not ready for trading")
                return

            # Update TradingStrategy balance
            current_time = self.bars.index.get_level_values('timestamp')[-1]
            current_date = current_time.date()
            # log(f'execute_trading_logic {current_time}')
            
            if self.trading_strategy.touch_detection_areas.quotes_raw is not None and not self.trading_strategy.touch_detection_areas.quotes_raw.empty:
                self.prev_quotes_raw = self.trading_strategy.touch_detection_areas.quotes_raw.iloc[[-1]].copy(deep=True) # store copy of last row
        
            # Calculate touch detection areas every minute
            # TODO: CALL THIS IN SEPARATE THREAD. It does not retrieve quotes data in the live params scenario.
            self.trading_strategy.update_strategy(self.calculate_touch_detection_area(self.trading_strategy.touch_detection_areas.market_hours))

            log('strategy updated',level=logging.DEBUG)

            # get quotes data
            if self.simulation_mode:
                # NOTE: follow similar quotes data processing as retrieve_quote_data
                self.trading_strategy.touch_detection_areas.quotes_raw = self.get_historical_quotes(start = current_time - timedelta(seconds=2), 
                                                             end = current_time + timedelta(seconds=1))
                # ensure there is at least 1 datapoint before the minute mark
                if self.trading_strategy.touch_detection_areas.quotes_raw.empty or \
                    self.trading_strategy.touch_detection_areas.quotes_raw.index.get_level_values('timestamp')[0] > current_time:
                    temp = self.get_historical_quotes(start = current_time - timedelta(seconds=59), 
                                                             end = current_time - timedelta(seconds=2))
                    if not temp.empty:
                        self.trading_strategy.touch_detection_areas.quotes_raw = pd.concat([temp, self.trading_strategy.touch_detection_areas.quotes_raw])
                    else:
                        self.trading_strategy.touch_detection_areas.quotes_raw = pd.concat([self.prev_quotes_raw, self.trading_strategy.touch_detection_areas.quotes_raw])
                    
                self.trading_strategy.touch_detection_areas.quotes_agg = pd.DataFrame() # placeholder

            else:
                # NOTE: need real time data!
                self.trading_strategy.touch_detection_areas.quotes_raw = pd.DataFrame() # placeholder
                self.trading_strategy.touch_detection_areas.quotes_agg = pd.DataFrame() # placeholder
                
                # TODO: get latest quote BUT do it AFTER order submitted (for recordkeeping), and in SEPERATE THREAD
                self.trading_strategy.latest_quote = self.get_latest_quote() # ~ 0.02 - 0.03 seconds

            log('quotes data retrieved',level=logging.DEBUG)
            
            orders_list0, positions_to_remove = self.trading_strategy.process_live_data(current_time, self.timer_start, self.area_ids_to_side_switch[current_date])
            orders_list = combine_two_orders_same_symbol(orders_list0)
            
            if orders_list0:
                # log(f"{current_time.strftime("%H:%M")}: {len(orders_list0)} ORDERS CREATED")  
                
                if len(orders_list) != len(orders_list0):
                    
                    if len(orders_list) == 0:
                        a = orders_list0[-1]
                        log(f"            Zero net qty change. No order created.",level=logging.INFO)
                    
                    for a in orders_list:
                        # log({k:order[k] for k in order if k != 'position'})
                        peak = a.position.max_close if a.position.is_long else a.position.min_close
                        log(f"            {a.position.id} {a.action} {str(a.side).split('.')[1]} {int(a.qty)} * {a.price}, peak-stop {peak:.4f}-{a.position.current_stop_price:.4f}, {a.position.area}", 
                            level=logging.INFO)
                        
                        
                if not self.simulation_mode:
                    # Place all orders concurrently
                    
                    # TODO: PLACE ORDER
                    pass
                    
                    # orders_list SHOULD only have 1 order, but there MIGHT be multi-order situations if I choose to implement it later
                    # not sure how to do this. 2 options:
                    for order in orders_list:
                        await self.place_order(order) # NOTE: preferred, MUST be sequential (assuming for same symbol)
                        
                        
                    # OR:
                    # await asyncio.gather(*[self.place_order(order) for order in orders_list])
                else:
                    # In simulation mode, just log the orders (already done in TradingStrategy)

                    pass
                
                # plot_touch_detection_areas(self.trading_strategy.touch_detection_areas) # for testing
            
            
            # do after to not slow things down
            if positions_to_remove:
                log(f'positions_to_remove: {[a.id for a in positions_to_remove]}', logging.DEBUG)
                self.trades.extend(positions_to_remove)
            # if positions_to_remove:
            #     self.area_ids_to_remove[current_date] = self.area_ids_to_remove[current_date] | {position.area.id for position in positions_to_remove}
            if positions_to_remove:
                self.area_ids_to_remove[current_date] = self.area_ids_to_remove[current_date] | set.union(*(position.cleared_area_ids for position in positions_to_remove))
            self.area_ids_to_side_switch[current_date] = self.area_ids_to_side_switch[current_date] | {a.position.area.id for a in orders_list0 if a.position.area.is_side_switched}
            
            log('trading logic complete',level=logging.DEBUG)
            
            
        except Exception as e:
            log(f"{type(e).__qualname__} in execute_trading_logic at {current_time}: {e}", logging.ERROR)
            raise e

    def calculate_touch_detection_area(self, market_hours=None):
        areas, self.touch_detection_state = calculate_touch_detection_area(self.touch_detection_params, self.bars, market_hours, self.area_ids_to_remove, 
                                                                           previous_state=self.touch_detection_state)
        return areas
    
    async def place_order(self, intended_order: IntendedOrder):
        """Place a new order and track it"""
        order_request = MarketOrderRequest(
            symbol=intended_order.symbol,
            qty=intended_order.qty,
            side=intended_order.side,
            time_in_force=TimeInForce.DAY
        )
        
        try:
            assert self.trading_strategy.latest_quote is not None
            pre_submit_timestamp = datetime.now(self.ny_tz)
            new_order = self.trading_client.submit_order(order_request)
            
            # Track order with quote info for slippage analysis
            self.order_tracker.add_order(
                new_order, 
                intended_order,
                pre_submit_timestamp=pre_submit_timestamp,
                quote_timestamp=self.trading_strategy.latest_quote_time,
                quote_bid_price=self.trading_strategy.now_bid_price,
                quote_ask_price=self.trading_strategy.now_ask_price
            )
            
            log(f"Placed {intended_order.side} order for {intended_order.qty} shares "
                f"of {intended_order.symbol} - {intended_order.action}")
            
            return new_order
            
        except Exception as e:
            log(f"{type(e).__qualname__} placing order: {e}", logging.ERROR)
            raise e

    async def is_order_filled(self, order: Order, intended_qty):
        pass
        # TODO
        # look at data coming in from on_msg
        # if order.filled_qty == intended_qty: # NOTE: complete fill condition
            # TODO


    async def on_msg(self, data: TradeUpdate):
        """Handle trade updates from websocket"""
        try:
            order = data.order
            event = data.event
            timestamp = data.timestamp
            
            log(f"Update for order {order.id}. Event: {event}. Status: {order.status}")
            
            # Process the update in order tracker
            await self.order_tracker.process_trade_update(event, order, timestamp)
            
            # If fills occurred, attempt reconciliation
            if event in (TradeEvent.FILL, TradeEvent.PARTIAL_FILL):
                await self.reconcile_fills()
                
        except Exception as e:
            log(f"{type(e).__qualname__} processing trade update: {e}", logging.ERROR)
            raise e


    async def reconcile_fills(self) -> None:
        """Reconcile any pending fills with trading strategy"""
        async with self.reconciliation_lock:  # Ensure only one reconciliation at a time
            try:
                # Get current account state
                account = self.trading_client.get_account()
                
                # Get fills ready for reconciliation
                fills_to_reconcile = await self.order_tracker.get_fills_to_reconcile()
                if not fills_to_reconcile:
                    return
                    
                reconciled_orders = set()
                
                for order_id, fill in fills_to_reconcile:
                    order_state = self.order_tracker.get_order_state(order_id)
                    intended = fill.intended_order
                    position = intended.position
                    
                    if not position.is_simulated:
                        # Calculate fill impact
                        fill_value = fill.qty * fill.avg_price
                        
                        if fill.event == TradeEvent.FILL:
                            # Complete fill - update balance and record slippage data
                            self.trading_strategy.rebalance(False, fill_value)
                            
                            # Create SlippageData entry
                            slippage_data = SlippageData.from_order(
                                order=Order(
                                    id=order_id,
                                    client_order_id=order_state.client_order_id,
                                    filled_qty=fill.qty,
                                    filled_avg_price=fill.avg_price,
                                    created_at=order_state.pre_submit_timestamp,
                                    submitted_at=order_state.pre_submit_timestamp,  # Best estimate
                                    updated_at=fill.transaction_time,
                                    filled_at=order_state.filled_at
                                ),
                                intended_qty=intended.qty,
                                is_entry=(intended.action in ('open', 'partial_entry', 'net_partial_entry')),
                                is_long=position.is_long,
                                quote_bid_price=order_state.quote_bid_price,
                                quote_ask_price=order_state.quote_ask_price,
                                quote_timestamp=order_state.quote_timestamp,
                                pre_submit_timestamp=order_state.pre_submit_timestamp,
                                atr=position.bar_at_entry.ATR,
                                avg_volume=position.bar_at_entry.avg_volume,
                                minute_volume=position.bar_at_entry.volume
                            )
                            
                            # Record slippage data
                            self.slippage_data_list.append(slippage_data)
                            if self.slippage_data_path:
                                slippage_data.append_to_csv(self.slippage_data_path)
                            
                        elif fill.event == TradeEvent.PARTIAL_FILL:
                            # Partial fill - update based on filled portion
                            self.trading_strategy.rebalance(False, fill_value)
                    
                    reconciled_orders.add(order_id)
                
                # Update strategy balance from account
                self.trading_strategy.update_balance_from_account(account)
                
                # Mark fills as reconciled
                await self.order_tracker.mark_fills_reconciled(reconciled_orders)
                
            except Exception as e:
                log(f"{type(e).__qualName__} reconciling fills: {e}", logging.ERROR)
                raise e


    # NOTE: outdated. prefer to use on_msg
    async def process_trade_update(self, trade_update: TradeUpdate, intended_order: IntendedOrder):
        # Log NEW order details
        
        # see class Order
        # TODO: this function should handle Order objects from TradeUpdate, not NEW orders
        
        placed_order = trade_update.order
        position_qty = trade_update.position_qty
        log(f"Order ID: {placed_order.id}") # NEED ID FOR CANCELLATION
        log(f"Order Status: {placed_order.status}")
        log(f"Filled Qty: {placed_order.filled_qty}")
        log(f"Filled Avg Price: {placed_order.filled_avg_price}")

        # Update internal state based on order status
        if placed_order.status == OrderStatus.FILLED:
            await self.update_position_after_fill(placed_order, position_qty, intended_order)
        elif placed_order.status == OrderStatus.PARTIALLY_FILLED:
            await self.handle_partial_fill(placed_order, position_qty, intended_order)
        elif placed_order.status in [OrderStatus.REJECTED, OrderStatus.CANCELED]:
            await self.handle_failed_order(placed_order, position_qty, intended_order)
        else:
            # For other statuses like NEW, ACCEPTED, etc.
            log(f"Order {placed_order.id} is in {placed_order.status} status. Waiting for fill.")

    # NOTE: outdated
    async def update_position_after_fill(self, placed_order: Order, position_qty: float, intended_order: IntendedOrder):
        log(f"Order {placed_order.id} filled completely - {placed_order.filled_qty} out of {placed_order.qty}")
        
        position = intended_order.position
        action = intended_order.action
        
        if action == 'open':
            assert placed_order.filled_qty == position_qty
            assert int(position_qty) == position_qty
            
            # Update the position with actual filled quantity and price
            position.shares = int(position_qty)
            position.entry_price = float(placed_order.filled_avg_price)
        elif action in ['net_partial_entry', 'net_partial_exit', 'partial_entry', 'partial_exit', 'close']:
            # Update the position for partial fills
            
            if (position.is_long and intended_order.side == OrderSide.BUY) or (not position.is_long and intended_order.side == OrderSide.SELL): # entry
                # position.shares += int(placed_order.filled_qty)
                # position.shares -= (intended_order.qty - int(placed_order.filled_qty)) # position.shares is already changed to be the INTENDED FINAL shares
                position.shares = int(position_qty)
            else: # exit
                # position.shares -= int(placed_order.filled_qty) 
                # position.shares += (intended_order.qty - int(placed_order.filled_qty)) # position.shares is already changed to be the INTENDED FINAL shares
                position.shares = int(position_qty)
                
        # Recalculate account balance
        self.recalculate_balance(placed_order, intended_order)

    # NOTE: outdated
    # NOTE: probably only needed for limit orders
    async def handle_partial_fill(self, placed_order: Order, position_qty: float, intended_order: IntendedOrder):
        log(f"Order {placed_order.id} partially filled. Filled {placed_order.filled_qty} out of {placed_order.qty}")
        # Update position with partially filled amount
        await self.update_position_after_fill(placed_order, position_qty, intended_order)
        
        # TODO: modify max_close/min_close of the corresponding position; be sure to replace/update the position in self.trading_strategy.open_positions
        
        # You might want to create a new order for the remaining quantity
        remaining_qty = float(placed_order.qty) - float(placed_order.filled_qty)
        if remaining_qty > 0:
            new_order = intended_order.copy()
            new_order.qty = remaining_qty
            await self.place_order(new_order)





    # NOTE: for market orders, order might be rejected if insufficient funds, given the latency between retrieving latest quote and order submission
    async def handle_failed_order(self, placed_order: Order, position_qty: float, intended_order: IntendedOrder):
        log(f"Order {placed_order.id} failed with status {placed_order.status}")
        # TODO: Implement logic to handle failed orders (e.g., retry, adjust strategy, etc.)
        
        
    # NOTE: outdated. we already have trading_strategy.update_balance_from_account
    # TODO: needs to call self.trading_strategy.rebalance
    def recalculate_balance(self, placed_order: Order, intended_order: IntendedOrder):
        # Recalculate balance based on the filled order
        filled_value = float(placed_order.filled_qty) * float(placed_order.filled_avg_price)
        if intended_order.side == OrderSide.BUY:
            self.trading_strategy.balance -= filled_value
        else:
            self.trading_strategy.balance += filled_value
        log(f"Updated balance: {self.trading_strategy.balance}")
    
    def is_receiving_data(self):
        if not self.streamed_bars.empty:
            last_data_time = self.streamed_bars.index.get_level_values('timestamp').max()  # NOTE: bar timestamp is at least 1 minute before current time.
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
                # await self.wait_for_market_open() # comment out for testing after hours
                await self.reset_daily_data()

                self.data_stream.subscribe_bars(self.on_bar, self.symbol)
                log(f'Subscribed to bars for {self.symbol}')
        
                # self.data_stream.subscribe_quotes(self.on_quote, self.symbol) # not necessary. already have get_latest_quote
                
                # self.data_stream.subscribe_trades(self.on_trade, self.symbol) # not necessary
                
                self.trading_stream.subscribe_trade_updates(self.on_msg)
                log(f'Subscribed to trade updates for current account')
                
                update_task = asyncio.create_task(self.update_historical_periodically())
                data_stream_task = asyncio.create_task(self.data_stream._run_forever())
                trading_stream_task = asyncio.create_task(self.trading_stream._run_forever())
                
                while self.is_market_open():
                    await asyncio.sleep(1)
                
                log("Market closed. Stopping data stream.")
                await self.close()
                self.data_stream.unsubscribe_bars(self.symbol)
                
                update_task.cancel()
                data_stream_task.cancel()
                trading_stream_task.cancel()
                # try:
                #     await update_task
                #     await data_stream_task
                # except asyncio.CancelledError:
                #     pass
                
                log("Waiting for next trading day...")
                await asyncio.sleep(2)  # Wait a minute before checking market open status again

        except Exception as e:
            log(f"{type(e).__qualname__} in run: {e}", logging.ERROR)
            raise e

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
        # day_data_adjusted = self.get_historical_bars(start, end, adjustment=Adjustment.ALL)
        
        if day_data.empty:
            log(f"No data available for {current_date}")
            return
        
        # Initialize data with the first minute
        # self.bars = day_data.iloc[:2]
        # self.bars_adjusted = day_data_adjusted.iloc[:2]
        
        # reset data logic


        text_trap = io.StringIO()
        
        # Simulate each minute of the day
        # for timestamp in tqdm([a for a in day_data.index.get_level_values('timestamp') if a.time() >= time(10, 37)]):
        # for timestamp in [a for a in day_data.index.get_level_values('timestamp') if a.time() >= time(10, 37)]:
        for timestamp in tqdm([a for a in day_data.index.get_level_values('timestamp') if a.time() >= time(9, 25)]):
        # for timestamp in [a for a in day_data.index.get_level_values('timestamp') if a.time() >= time(9, 25)]:
        # for timestamp in tqdm(day_data.index.get_level_values('timestamp')[2:]):
            # log(timestamp)
            # Update self.bars with all rows up to and including the current timestamp
            self.bars = day_data.loc[day_data.index.get_level_values('timestamp') <= timestamp]
            self.last_hist_bar_dt = self.bars.index.get_level_values('timestamp')[-1]
            
            
            if self.trading_strategy is None:
                self.trading_strategy = TradingStrategy(self.calculate_touch_detection_area(), self.strategy_params, log_level=logging.INFO, is_live_trading=True)
                self.trading_strategy.update_daily_parameters(current_date) # assuming only simulating one day
                    
            
            await self.simulate_bar()
            
            # # if len(self.trading_strategy.touch_detection_areas.long_touch_area) + len(self.trading_strategy.touch_detection_areas.short_touch_area) == 0:
            # if timestamp.time() < time(11,3):
            #     sys.stdout = text_trap
            #     sleep_interval = 0
            # else:
            #     sys.stdout = sys.__stdout__
            #     sleep_interval = 0.3
            # await asyncio.sleep(sleep_interval)
            
            if timestamp.time() >= time(16,0):
                break
            
        # log(f"Printing areas for {current_date}", level=logging.WARNING)
        # TouchArea.print_areas_list(self.trading_strategy.touch_area_collection.active_date_areas) # print if in log level
        log(f"{len(self.area_ids_to_remove[current_date])} terminated areas on {current_date}: {sorted(self.area_ids_to_remove[current_date])}", level=logging.WARNING)
        
        log(f"Completed simulation for {current_date}")
        
        # return self.trading_strategy.generate_backtest_results(self.trades)
        
            
import tracemalloc
tracemalloc.start()

async def main():
    # symbol = "AMZN"
    # symbol = "AAPL"
    # symbol = "SOL"
    # symbol = "ETH"
    # symbol = "TSLA"
    symbol = "NVDA"
    
    simulation_mode = True  # Set this to True for simulation, False for live trading

    touch_detection_params = LiveTouchDetectionParameters(
        symbol=symbol,
        # atr_period=15,
        # level1_period=15,
        # multiplier=1.4,
        # min_touches=3,
        start_time=None,
        end_time='15:55',
        # end_time='11:20',
        # use_median=True,
        # touch_area_width_agg=np_median,

        # ema_span=12,
        # price_ema_span=26
        
    )
    
    
    
    initial_balance = 30_000
    
    strategy_params = StrategyParameters(
        initial_investment=initial_balance,
        max_investment=initial_balance,

        do_longs=True,
        do_shorts=True,
        sim_longs=True,
        sim_shorts=True,
        
        use_margin=True,
        
        assume_marginable_and_etb=True,
        
        times_buying_power=1,
        
        soft_start_time = None, 
        soft_end_time = '15:30',
        
        # plot_day_results=True,
        
        # allow_reversal_detection=True, # False (no switching) seems better for more stocks. If True, clear_passed_areas=True might improve performance.
        
        clear_passed_areas=True,

        # ### OUTDATED ### False is better for meme stocks, True better for mid and losing stocks (reduces losses). ### OUTDATED ###
        # True is better for meme stocks, False better for mid and losing stocks (reduces losses).
        
        clear_traded_areas=True,
        # True is better/safer for meme/mid?
        
        
        
        # min_stop_dist_relative_change_for_partial=1,
        
        
        # volume_profile_ema_span=np.inf,
        # volume_profile_ema_span=240,
        # volume_profile_ema_span=390,
        gradual_entry_range_multiplier = 0.9
        
    )

    
    strategy_params.ordersizing.max_volume_percentage = 0.2 # %. default is 1 %
    # strategy_params.slippage.slippage_factor = 0
    
    trader = LiveTrader(API_KEY, API_SECRET, symbol, touch_detection_params, strategy_params, simulation_mode, slippage_data_path='slippage_data_output.csv')

    try:
        
        if not simulation_mode:
            await trader.run()
        
        else:
            # date_to_simulate = date(2024, 8, 20)
            date_to_simulate = date(2024, 9, 4)
            results = await trader.run_day_sim(date_to_simulate)
            print(results)
        
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Stopping trader.")
    except Exception as e:
        print(f"{type(e).__qualname__}: {e}")
    finally:
        await trader.close()
        print("Trader stopped.")

if __name__ == "__main__":
    asyncio.run(main())