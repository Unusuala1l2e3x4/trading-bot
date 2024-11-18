import pandas as pd
import numpy as np
from numba import jit
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockQuotesRequest
from alpaca.trading import TradingClient
from alpaca.trading.requests import GetCalendarRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment

from typing import List, Tuple, Optional, Dict

import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
import toml
import os
import zipfile

from datetime import datetime, timedelta, time, date
from zoneinfo import ZoneInfo
from tqdm import tqdm

from TouchArea import TouchArea
from MultiSymbolDataRetrieval import retrieve_bar_data, retrieve_quote_data
import TouchDetectionParameters
from TouchDetectionParameters import BacktestTouchDetectionParameters, LiveTouchDetectionParameters

import logging
import traceback

import time as t2


from dataclasses import dataclass, field
from typing import Optional, Callable


from dotenv import load_dotenv

ny_tz = ZoneInfo("America/New_York")
STANDARD_DATETIME_STR = '%Y-%m-%d %H:%M:%S'
ROUNDING_DECIMAL_PLACES = 10  # Choose an appropriate number of decimal places

load_dotenv(override=True)
livepaper = os.getenv('LIVEPAPER')
config = toml.load('../config.toml')

# Replace with your Alpaca API credentials
API_KEY = config[livepaper]['key']
API_SECRET = config[livepaper]['secret']

pd.options.mode.copy_on_write = True # https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#why-does-assignment-fail-when-using-chained-indexing

trading_client = TradingClient(API_KEY, API_SECRET)


def setup_logger(log_level=logging.INFO):
    logger = logging.getLogger('TouchDetection')
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

logger = setup_logger(logging.WARNING)

def log(message, level=logging.INFO):
    logger.log(level, message, exc_info=level >= logging.ERROR)



def get_market_hours(dates: List[date]) -> Dict[date, Tuple[datetime, datetime]]:
    """
    Parameters:
    dates (list of str): List of dates in 'YYYY-MM-DD' format.

    Returns:
    
    dict: Dictionary with dates as keys and (market_open, market_close) tuples as values.
    """
    start_date = min(dates)
    end_date = max(dates)
    
    calendar_request = GetCalendarRequest(start=start_date, end=end_date)
    calendar = trading_client.get_calendar(calendar_request)
    
    # market_hours = {str(day.date): (day.open.replace(tzinfo=ny_tz), day.close.replace(tzinfo=ny_tz)) for day in calendar}
    market_hours = {day.date: (day.open.astimezone(ny_tz), day.close.astimezone(ny_tz)) for day in calendar}
    return market_hours



def calculate_dynamic_central_value(df:pd.DataFrame, ema_short=9, ema_long=20):
    """
    Calculate VWAP and EMAs for the dataframe.
    
    :param df: DataFrame with 'close', 'high', 'low', and 'volume' columns
    :param ema_short: Period for the short EMA (default 9)
    :param ema_long: Period for the long EMA (default 20)
    :return: DataFrame with additional columns for VWAP and EMAs
    """
    
    assert 'vwap' in df.columns
    
    # Calculate EMAs
    # df[f'EMA{ema_short}'] = df['close'].ewm(span=ema_short, adjust=False).mean()
    # df[f'EMA{ema_long}'] = df['close'].ewm(span=ema_long, adjust=False).mean()
    
    # Calculate a combined central value
    # df['central_value'] = (df['vwap'] + df[f'EMA{ema_short}'] + df[f'EMA{ema_long}']) / 3
    # df['central_value'] = df['vwap']
    # df['central_value'] = df[f'EMA{ema_short}']
    # df['central_value'] = df[f'EMA{ema_long}']
    
    # 1
    # df['central_value'] = df['close'].ewm(span=26, adjust=True).mean()
    
    # 2
    # df['central_value'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # 3
    # halflife = '26min'  # 26 minutes
    # df['central_value'] = df['close'].ewm(
    #     halflife=halflife,
    #     times=df.index.get_level_values('timestamp'),
    #     adjust=True
    # ).mean()

    # #4
    # span = 26 # span of 26 = 9.006468342000588min
    # alpha = 2 / (span + 1)
    # halflife = np.log(2) / np.log(1 / (1 - alpha))
    # halflife_str = f"{halflife}min"

    # df['central_value'] = df['close'].ewm(
    #     halflife=halflife_str,
    #     times=df.index.get_level_values('timestamp'),
    #     adjust=True
    # ).mean()
    
    #45
    span = 26 # span of 26 = 9.006468342000588min
    # alpha = 2 / (span + 1)
    # halflife = np.log(2) / np.log(1 / (1 - alpha))
    # halflife_str = f"{halflife}min"

    df['central_value'] = df['close'].ewm(
        # halflife=halflife_str,
        span=span,
        # times=df.index.get_level_values('timestamp'),
        adjust=False # adjust=False -> EMA
    ).mean()

def calculate_ema_with_cutoff(df: pd.DataFrame, field: str, span: int, window: int = None, adjust=False):
    """
    A generic EMA calculation function with an optional cutoff.
    
    Parameters:
    df (pd.DataFrame): Input dataframe with the necessary fields.
    field (str): The column on which to apply the EMA.
    span (int): Span for the EMA calculation.
    window (int, optional): The number of periods to use as a cutoff. If None, no cutoff is applied.
    adjust (bool): Adjust parameter for the .ewm function (default False for recursive calculation).
    
    Returns:
    pd.Series: EMA values for the specified field.
    """
    if window:
        # Apply cutoff to only consider the last `window` rows
        return df[field].rolling(window=window, min_periods=1).apply(
            lambda x: x.ewm(span=span, adjust=adjust).mean().iloc[-1]
        )
    else:
        # If no cutoff is provided, apply .ewm without restrictions
        return df[field].ewm(span=span, adjust=adjust).mean()
    

@dataclass
class Level:
    id: int
    lmin: float
    lmax: float
    level: float
    is_res: bool
    touches: List[datetime]


@jit(nopython=True)
def np_searchsorted(a,b): # only used in calculate_touch_area function
    return np.searchsorted(a,b)


@jit(nopython=True)
def calculate_touch_area_bounds(atr_values, level, is_long, touch_area_width_agg, multiplier):
    touch_area_width = touch_area_width_agg(atr_values) * multiplier
    
    # touch_area_low = level - (1 * touch_area_width / 3) if is_long else level - (2 * touch_area_width / 3)
    # touch_area_high = level + (2 * touch_area_width / 3) if is_long else level + (1 * touch_area_width / 3)

    # touch_area_low = level - (1 * touch_area_width / 2) if is_long else level - (1 * touch_area_width / 2)
    # touch_area_high = level + (1 * touch_area_width / 2) if is_long else level + (1 * touch_area_width / 2)
    
    if is_long:
        touch_area_low = level - (2 * touch_area_width / 3)
        touch_area_high = level + (1 * touch_area_width / 3)
    else:
        touch_area_low = level - (1 * touch_area_width / 3)
        touch_area_high = level + (2 * touch_area_width / 3)
    return touch_area_width, touch_area_low, touch_area_high

@jit(nopython=True)
def process_touches(touches, prices, atrs, level, lmin, lmax, is_long, min_touches, touch_area_width_agg, multiplier):
    consecutive_touches = np.full(min_touches, -1, dtype=np.int64)
    count, width = 0, 0
    prev_price = None
    
    for i in range(len(prices)):
        price = prices[i]
        is_touch = (prev_price is not None and 
                    ((prev_price < level <= price) or (prev_price > level >= price)) or 
                    (price == level))
        
        if lmin <= price <= lmax:
            if is_touch:
                # Update bounds after each touch
                width, touch_area_low, touch_area_high = calculate_touch_area_bounds(atrs[:i+1], level, is_long, touch_area_width_agg, multiplier)
                if width > 0:
                    consecutive_touches[count] = touches[i]
                    count += 1
                
                    if count == min_touches:
                        return consecutive_touches[consecutive_touches != -1], touch_area_low, touch_area_high
                
        elif width > 0:
            assert touch_area_high is not None and touch_area_low is not None
            buy_price = touch_area_high if is_long else touch_area_low
            if (is_long and price > buy_price) or (not is_long and price < buy_price):
                consecutive_touches[:] = -1
                count = 0
        
        prev_price = price
    return np.empty(0, dtype=np.int64), touch_area_low, touch_area_high

def calculate_touch_area(levels_by_date: Dict[datetime, List[Level]], is_long, df: pd.DataFrame, symbol, market_hours: Dict[date, Tuple[datetime, datetime]], min_touches, \
    bid_buffer_pct, use_median, touch_area_width_agg: Callable, multiplier, start_time: datetime, end_time: datetime, current_timestamp: datetime=None):
    touch_areas = []
    widths = []

    # for date, levels in tqdm(levels_by_date.items(), desc='calculate_touch_area'):
    for date, levels in levels_by_date.items():
        market_open, market_close = market_hours.get(date, (None, None))
        if market_open and market_close:
            date_obj = pd.Timestamp(date).tz_localize(ny_tz)
            if start_time:
                day_start_time = date_obj.replace(hour=start_time.hour, minute=start_time.minute)
            else:
                day_start_time = market_open
            if end_time:
                day_end_time = min(date_obj.replace(hour=end_time.hour, minute=end_time.minute), market_close - pd.Timedelta(minutes=3))
            else:
                day_end_time = market_close - pd.Timedelta(minutes=3)
                
        else:
            # print('Hours not available. Skipping',date)
            continue
        
        day_data = df[df.index.get_level_values('timestamp').date == date]
        day_timestamps = day_data.index.get_level_values('timestamp')
        day_timestamps_np = np.array(day_timestamps.astype(np.int64))
        day_prices = day_data['close'].values
        day_atr = day_data['MTR' if use_median else 'ATR'].values
        
        for level in levels:
            assert len(level.touches) >= min_touches
            # if len(level.touches) < min_touches:
            #     continue
            
            touch_timestamps_np = np.array([t.value for t in level.touches], dtype=np.int64)
            touch_indices = np_searchsorted(day_timestamps_np, touch_timestamps_np)
            
            valid_mask = (day_timestamps[touch_indices] >= day_start_time) & (day_timestamps[touch_indices] < day_end_time)
            valid_touch_indices = touch_indices[valid_mask]
            
            if valid_touch_indices.size == 0:
                continue
            
            valid_prices = day_prices[valid_touch_indices]
            valid_atr = day_atr[valid_touch_indices]

            consecutive_touch_indices, touch_area_low, touch_area_high = process_touches(
                valid_touch_indices, 
                valid_prices,
                valid_atr,
                level.level, 
                level.lmin,
                level.lmax, 
                is_long, 
                min_touches,
                touch_area_width_agg,
                multiplier
            )
            
            if len(consecutive_touch_indices) == min_touches:
                consecutive_touches = day_timestamps[consecutive_touch_indices]
                touch_area = TouchArea(
                    date=date,
                    id=level.id, # unique since each level has unique INITIAL touch time, and at most 1 TouchArea is created per level
                    level=level.level, # use just the value
                    lmin=level.lmin,
                    lmax=level.lmax,
                    upper_bound=touch_area_high,
                    lower_bound=touch_area_low,
                    initial_touches=consecutive_touches,
                    touches=day_timestamps[valid_touch_indices],
                    is_long=is_long,
                    min_touches=min_touches,
                    bid_buffer_pct=bid_buffer_pct,
                    valid_atr=valid_atr,
                    touch_area_width_agg=touch_area_width_agg,
                    multiplier=multiplier,
                    calculate_bounds=calculate_touch_area_bounds
                )
                # log(f"CALC   area {touch_area.id} ({touch_area.min_touches_time.time()}): get_range {touch_area.get_range:.4f}")
                # if current_timestamp is not None:
                #     touch_area.update_bounds(current_timestamp)  # unnecessary?
                #     log(f"updated to {touch_area.get_range:.4f}")
                touch_areas.append(touch_area)

    return touch_areas, widths


@dataclass
class TouchDetectionAreas:
    symbol: str
    long_touch_area: List[TouchArea]
    short_touch_area: List[TouchArea]
    market_hours: Dict[date, Tuple[datetime, datetime]]
    bars: pd.DataFrame
    bars_adjusted: pd.DataFrame
    quotes_raw: pd.DataFrame
    quotes_agg: pd.DataFrame
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
            bars_adjusted=data['bars_adjusted'],
            quotes_raw=data['quotes_raw'],
            quotes_agg=data['quotes_agg'],
            mask=data['mask'],
            min_touches=data['min_touches'],
            start_time=data['start_time'],
            end_time=data['end_time']
        )



def calculate_touch_detection_area(params: BacktestTouchDetectionParameters | LiveTouchDetectionParameters, live_bars: Optional[pd.DataFrame] = None, 
                                   market_hours: Optional[Dict[date, Tuple[datetime, datetime]]] = None,
                                   current_timestamp: Optional[datetime] = None, area_ids_to_remove: Optional[set] = {}) -> TouchDetectionAreas:
    def log_live(message, level=logging.INFO):
        if isinstance(params, LiveTouchDetectionParameters):
            logger.log(level, message, exc_info=level >= logging.ERROR)
    
    """
    Calculates touch detection areas for a given stock symbol based on historical price data and volatility.

    Parameters:
    params (TouchDetectionParameters): An instance of TouchDetectionParameters containing all necessary parameters.

    Returns:
    dict: A dictionary containing:
        - 'symbol': The analyzed stock symbol
        - 'long_touch_area': List of TouchArea objects for long positions
        - 'short_touch_area': List of TouchArea objects for short positions
        - 'market_hours': Dictionary of market hours for each trading day
        - 'bars': DataFrame of price data
        - 'mask': Boolean mask for filtering data
        - 'bid_buffer_pct': The bid buffer percentage used
        - 'min_touches': The minimum number of touches used
        - 'start_time': The start time used for analysis
        - 'end_time': The end time used for analysis
        - 'use_median': Whether median or mean was used for touch area width calculation (True -> use rolling MTR, False -> using rolling+tapered ATR)

    This function analyzes historical price data to identify significant price levels (support and resistance)
    based on the frequency of price touches. It considers market volatility using ATR and allows for 
    customization of the analysis parameters. The resulting touch areas can be used for trading strategies
    or further market analysis.
    """
    if isinstance(params, TouchDetectionParameters.BacktestTouchDetectionParameters):
        assert params.end_date > params.start_date

        # Alpaca API setup
        client = StockHistoricalDataClient(api_key=API_KEY, secret_key=API_SECRET)

        # get bars data (2 dataframes)
        df_adjusted, df = retrieve_bar_data(client, params)
        
        # get quotes data (2 dataframes)
        minute_intervals = df.index.get_level_values('timestamp')
        minute_intervals = minute_intervals[(minute_intervals.time >= time(9, 30)) & (minute_intervals.time < time(16, 0))]
        minute_intervals_dict = {params.symbol: minute_intervals}
        
        quotes_data = retrieve_quote_data(client, [params.symbol], minute_intervals_dict, params)
        
        if isinstance(quotes_data[params.symbol]['raw'], pd.DataFrame) and isinstance(quotes_data[params.symbol]['agg'], pd.DataFrame):
            raw_df = quotes_data[params.symbol]['raw']
            aggregated_df = quotes_data[params.symbol]['agg']
        else:
            raise ValueError(f"Quote data not found for symbol {params.symbol}")

    elif isinstance(params, TouchDetectionParameters.LiveTouchDetectionParameters):
        if live_bars is None:
            raise ValueError("Live bars data must be provided for live trading parameters")
        df = live_bars
        df_adjusted = None
        raw_df = None
        aggregated_df = None
    else:
        # print(type(params))
        # print(isinstance(params, BacktestTouchDetectionParameters))
        # print(isinstance(params, TouchDetectionParameters.BacktestTouchDetectionParameters))
        raise ValueError("Invalid parameter type")
    
    try:
        log_live('Data retrieved')
        
        timestamps = df.index.get_level_values('timestamp')
        # print(timestamps)
        # print(df.columns)
        # print(df.dtypes)
        # print(df)
        
        # calculate_dynamic_central_value(df, ema_short=9, ema_long=30) # default
        # calculate_dynamic_central_value(df)
        
        df['central_value'] = (df['vwap'] + calculate_ema_with_cutoff(df, 'close', span=params.price_ema_span) * 2) / 3
        df['is_res'] = df['close'] >= df['central_value']

        # Calculate True Range (TR)
        df['H-L'] = df['high'] - df['low']
        df['H-PC'] = np.abs(df['high'] - df['close'].shift(1))
        df['L-PC'] = np.abs(df['low'] - df['close'].shift(1))
        df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        
        
        df['ATR'] = calculate_ema_with_cutoff(df, 'TR', span=params.ema_span) # , window=params.atr_period
        df['MTR'] = df['TR'].rolling(window=params.atr_period).apply(lambda x: np.median(x), raw=True)
        
        log_live('ATR and MTR calculated')

        # Mean: This is more sensitive to outliers and can be useful if you want your strategy to react more quickly to sudden changes in volume or trade count.
        # Median: This is more robust to outliers and can provide a more stable measure of the typical volume or trade count, which might be preferable if you want 
        # your strategy to be less affected by occasional spikes in activity.
        
        # Calculate rolling average volume and trade count
        df['shares_per_trade'] = df['volume'] / df['trade_count']
        df['avg_volume'] = calculate_ema_with_cutoff(df, 'volume', span=params.ema_span) # , window=params.level1_period
        df['avg_trade_count'] = calculate_ema_with_cutoff(df, 'trade_count', span=params.ema_span) # , window=params.level1_period
        
        # df['avg_shares_per_trade'] = df['shares_per_trade'].rolling(window=params.level1_period).mean()
        # df['avg_shares_per_trade'] = calculate_ema_with_cutoff(df, 'shares_per_trade', span=params.ema_span, window=params.level1_period) 
        
        log_live('rolling averages calculated')
        
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_return'].rolling(window=15).std().fillna(0)
        
        # Group data by date
        grouped = df.groupby(timestamps.date)
        
        all_support_levels = defaultdict(list)
        all_resistance_levels = defaultdict(list)

        # def is_res_area(index, day_df):
        #     return day_df.loc[index, 'close'] > day_df.loc[index, 'central_value']
        
        # high_low_diffs_list = []
        
        # for date, day_df in tqdm(grouped, desc='calculate_touch_detection_area'):
        for date, day_df in grouped:
            day_timestamps = day_df.index.get_level_values('timestamp') # need to limit
            
            potential_levels = defaultdict(lambda: Level(0, 0, 0, False, []))
            
            high_low_diffs = [] # only consider the diffs in the current day
            
            for i in range(len(day_df)):
                row = day_df.iloc[i]
                if row['volume'] <= 0 or row['trade_count'] <= 0:
                    continue
                
                high, low, close = row['high'], row['low'], row['close']
                timestamp = day_timestamps[i]
                is_res = row['is_res']
                
                high_low_diffs.append(high - low)
                
                w = np.median(high_low_diffs) / 2
                lmin, lmax = close - w, close + w
                
                # Check if this point falls within any existing levels
                for level in potential_levels.values():
                    if level.lmin <= close <= level.lmax and level.is_res == is_res:
                        level.touches.append(timestamp)

                if w != 0 and i not in potential_levels:
                    # Add this point to its own level (match by lmin, lmax in case the same lmin, lmax is already used)
                    potential_levels[i] = Level(i, lmin, lmax, close, is_res, [timestamp]) # using i as ID since levels have unique INITIAL timestamp
            
            if date in area_ids_to_remove:
                for i in area_ids_to_remove[date]:
                    del potential_levels[i] # area id == level id
                    
            # a = pd.DataFrame(pd.Series(high_low_diffs).describe()).T
            # a['date'] = date
            # high_low_diffs_list.append(a)
            # print(a)

            # Classify levels as support or resistance
            # Filter for strong levels
            for level in potential_levels.values():
                if len(level.touches) < params.min_touches: # not strong level
                    continue
                if level.is_res:
                    all_resistance_levels[date].append(level)
                else:
                    all_support_levels[date].append(level)

        log_live('Levels created')
        unique_dates = list(pd.unique(timestamps.date))
        if market_hours is None:
            market_hours = get_market_hours(unique_dates)
            log_live('market hours retrieved')
        
        if params.end_time:
            end_time = pd.to_datetime(params.end_time, format='%H:%M').time()
        else:
            end_time = None
        if params.start_time:
            start_time = pd.to_datetime(params.start_time, format='%H:%M').time()
        else:
            start_time = None
        
        # print(start_time, end_time)

        # might not need to mask out before market_open
        final_mask = pd.Series(False, index=df.index)
        for date in unique_dates:
            market_open, market_close = market_hours.get(date, (None, None))
            date_obj = pd.Timestamp(date).tz_localize(ny_tz)
            if market_open and market_close:
                if start_time:
                    day_start_time = date_obj.replace(hour=start_time.hour, minute=start_time.minute)
                else:
                    day_start_time = market_open
                if end_time:
                    day_end_time = min(date_obj.replace(hour=end_time.hour, minute=end_time.minute), market_close - timedelta(minutes=3))
                else:
                    day_end_time = market_close - timedelta(minutes=3)
            else:
                day_start_time, day_end_time = date_obj, date_obj

            mask = (timestamps >= day_start_time) & (timestamps <= day_end_time)
            final_mask |= mask

        log_live('Mask created')

        
        long_touch_area, long_widths = calculate_touch_area(
            all_resistance_levels, True, df, params.symbol, market_hours, params.min_touches, 
            params.bid_buffer_pct, params.use_median, params.touch_area_width_agg, params.multiplier, start_time, end_time, current_timestamp
        )
        log_live(f'{len(long_touch_area)} Long touch areas calculated')
        short_touch_area, short_widths = calculate_touch_area(
            all_support_levels, False, df, params.symbol, market_hours, params.min_touches, 
            params.bid_buffer_pct, params.use_median, params.touch_area_width_agg, params.multiplier, start_time, end_time, current_timestamp
        )
        log_live(f'{len(short_touch_area)} Short touch areas calculated')
            
        # widths = long_widths + short_widths

        df = df.drop(columns=['H-L','H-PC','L-PC','TR','ATR','MTR'])
        
        ret = {
            'symbol': df.index.get_level_values('symbol')[0] if isinstance(params, LiveTouchDetectionParameters) else params.symbol,
            'long_touch_area': long_touch_area,
            'short_touch_area': short_touch_area,
            'market_hours': market_hours,
            'bars': df,
            'bars_adjusted': df_adjusted,
            'quotes_raw': raw_df,
            'quotes_agg': aggregated_df,
            'mask': final_mask,
            # 'bid_buffer_pct': params.bid_buffer_pct,
            'min_touches': params.min_touches,
            'start_time': start_time,
            'end_time': end_time,
            # 'use_median': params.use_median
        }
        return TouchDetectionAreas.from_dict(ret) # , high_low_diffs_list
        
    except Exception as e:
        log(f"{type(e).__qualname__} in calculate_touch_detection_area: {e}", level=logging.ERROR)
        raise e


def plot_touch_detection_areas(touch_detection_areas: TouchDetectionAreas, zoom_start_date=None, zoom_end_date=None, save_path=None, filter_date=None, filter_areas=None):
    """
    Visualizes touch detection areas and price data on a chart.

    Parameters:
    touch_detection_areas (dict): Dictionary containing touch areas and market data.
    zoom_start_date (str): Start date for the zoomed view. Format: 'YYYY-MM-DD HH:MM:SS'
    zoom_end_date (str): End date for the zoomed view. Format: 'YYYY-MM-DD HH:MM:SS'
    save_path (str, optional): Path to save the plot as an image file. Default is None.

    This function creates a plot showing the price movement, central value, and touch detection areas
    for both long and short positions. It highlights important points and areas on the chart,
    focusing on the specified date range. The resulting plot can be displayed and optionally saved.
    """
    symbol = touch_detection_areas.symbol
    long_touch_area = touch_detection_areas.long_touch_area
    short_touch_area = touch_detection_areas.short_touch_area
    market_hours = touch_detection_areas.market_hours
    df = touch_detection_areas.bars
    df_adjusted = touch_detection_areas.bars_adjusted
    mask = touch_detection_areas.mask
    # bid_buffer_pct = touch_detection_areas.bid_buffer_pct
    min_touches = touch_detection_areas.min_touches
    start_time = touch_detection_areas.start_time
    end_time = touch_detection_areas.end_time
    # use_median = touch_detection_areas.use_median
    
    
    plt.figure(figsize=(14, 7))
    if not filter_date:
        plt.plot(df.index.get_level_values('timestamp'), df['central_value'], label='central_value', color='yellow')
        plt.plot(df.index.get_level_values('timestamp'), df['close'], label='Close Price', color='blue')
    else:
        df = df.loc[df.index.get_level_values('timestamp').date == filter_date]
        mask = mask.loc[mask.index.get_level_values('timestamp').date == filter_date]
        plt.plot(df.index.get_level_values('timestamp'), df['central_value'], label='central_value', color='yellow')
        plt.plot(df.index.get_level_values('timestamp'), df['close'], label='Close Price', color='blue')

    df = df[mask]
    timestamps = df.index.get_level_values('timestamp')
    

    # Prepare data structures for combined plotting
    scatter_data = defaultdict(lambda: defaultdict(list))
    fill_between_data = defaultdict(list)
    line_data = defaultdict(list)

    def find_area_end_idx(start_idx, area:TouchArea, day_start_time, day_end_time):
        assert day_start_time.tzinfo == day_end_time.tzinfo, f"{day_start_time.tzinfo} == {day_end_time.tzinfo}"
        entry_price = area.get_buy_price
        
        for i in range(start_idx + 1, len(df)):
            current_time = timestamps[i].tz_convert(ny_tz)
            if current_time >= day_end_time:
                return i
            # current_price = df['close'].iloc[i]
            current_price = df.iloc[i, df.columns.get_loc('close')]
            
            if current_time >= area.get_min_touch_time:
                if area.is_long and current_price >= entry_price:
                    return i
                elif not area.is_long and current_price <= entry_price:
                    return i
        
        return len(df) - 1  # If no end condition is met, return the last index
    
    
    def process_area(area: TouchArea):
        if not area.is_active:
            return
        
        mark_pos = area.get_buy_price
        mark_shape = "v" if area.is_long else '^'
        color = 'red' if area.is_long else 'green'

        
        current_date = None
        for i, touch_time in enumerate(area.touches):
            touch_time = touch_time.tz_convert(ny_tz)
            if touch_time in timestamps:
                start_idx = timestamps.get_loc(touch_time)
                
                if timestamps[start_idx].date() != current_date:
                    current_date = timestamps[start_idx].date()
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
                    else:
                        # print('Hours not available. Skipping',date)
                        continue
                    
                    day_data = df[df.index.get_level_values('timestamp').date == current_date]
                    day_timestamps = day_data.index.get_level_values('timestamp')
                    if day_end_time >= day_timestamps[-1]: # only needed for graphing
                        day_end_time = day_timestamps[-1] + pd.Timedelta(minutes=1)

                if area.entries_exits:
                    # exit_time = area.entries_exits[-1].exit_time
                    # end_idx = df.index.get_loc(df[timestamps == exit_time].index[0])
                    entry_time = area.entries_exits[-1].entry_time
                    end_idx = df.index.get_loc(df[timestamps == entry_time].index[0])
                else:
                    end_idx = find_area_end_idx(start_idx, area, day_start_time, day_end_time)
                x1 = [timestamps[start_idx].tz_convert(ny_tz), area.get_min_touch_time]
                x2 = [area.get_min_touch_time, timestamps[end_idx].tz_convert(ny_tz)]
                
                if timestamps[end_idx].tz_convert(ny_tz) >= day_end_time:
                    continue
                        
                if filter_areas and filter_areas[current_date] and area.id not in filter_areas[current_date]:
                    continue
                        
                scatter_color = 'gray' if i != min_touches - 1 else 'red' if end_idx == start_idx else 'blue'
                scatter_data[scatter_color][mark_shape].append((touch_time, mark_pos))
                
                if i == 0:  # first touch
                    fill_between_data[color].append((x1 + x2, [area.lower_bound] * 4, [area.upper_bound] * 4))
                    line_data['blue_alpha'].append((x1, [area.level] * 2))
                    line_data['blue'].append((x2, [area.level] * 2))

    # for area in tqdm(long_touch_area + short_touch_area, desc='plotting areas'):
    for area in long_touch_area + short_touch_area:
        if not filter_date or area.date == filter_date:
            process_area(area)

    # Plot combined data
    for color, shape_data in scatter_data.items():
        for shape, points in shape_data.items():
            if points:
                x, y = zip(*points)
                plt.scatter(x, y, color=color, s=12, marker=shape)

    for color, data in fill_between_data.items():
        for x, lower, upper in data:
            plt.fill_between(x[:2], lower[:2], upper[:2], color=color, alpha=0.1)
            plt.fill_between(x[2:], lower[2:], upper[2:], color=color, alpha=0.25)

    for color, data in line_data.items():
        for x, y in data:
            if color == 'blue_alpha':
                plt.plot(x, y, color='blue', linestyle='-', alpha=0.20)
            else:
                plt.plot(x, y, color='blue', linestyle='-')

    plt.title(f'{symbol} Price Chart with Touch Detection Areas')
    plt.xlabel('Date')
    plt.ylabel('Price')
    # plt.legend(['Close Price', 'Long Touch Area', 'Short Touch Area'])
    plt.legend().remove()
    plt.grid(True)

    if zoom_start_date:
        zstart = pd.to_datetime(zoom_start_date)
        zstart = zstart.tz_localize(ny_tz) if zstart.tz is None else zstart.tz_convert(ny_tz)
    else:
        zstart = timestamps[0]
    
    if zoom_end_date:
        zend = pd.to_datetime(zoom_end_date)
        zend = zend.tz_localize(ny_tz) if zend.tz is None else zend.tz_convert(ny_tz)
    else:
        zend = timestamps[-1]
        
    # print(zstart, zend)

    plt.xlim(max(zstart, timestamps[0]), min(zend, timestamps[-1]))

    ymin, ymax = 0, -1
    for i in range(len(timestamps)):
        if timestamps[i] >= zstart:
            # print(timestamps[i])
            ymin = i-1
            break
    for i in range(len(timestamps)):
        if timestamps[i] >= zend:
            # print(timestamps[i])
            ymax = i
            break
    ys = df['close'].iloc[max(ymin, 0):min(ymax, len(df))]
    plt.ylim(min(ys),max(ys))
    
    if save_path:
        plt.savefig(save_path)

    plt.show()