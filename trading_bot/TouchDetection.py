import pandas as pd
import numpy as np
from numba import jit
from alpaca.data import StockHistoricalDataClient, CryptoHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockQuotesRequest
from alpaca.trading import TradingClient
from alpaca.trading.requests import GetCalendarRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment

from typing import List, Tuple, Optional, Dict, Union

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import defaultdict
import numpy as np
import toml
import os
import zipfile

from datetime import datetime, timedelta, time, date
from zoneinfo import ZoneInfo
from tqdm import tqdm

import trading_bot.TouchDetectionParameters
from trading_bot.TradePosition import TradePosition
from trading_bot.TouchArea import TouchArea
from trading_bot.MultiSymbolDataRetrieval import retrieve_bar_data, retrieve_quote_data
from trading_bot.TouchDetectionParameters import BacktestTouchDetectionParameters, LiveTouchDetectionParameters

import trading_bot

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
accountname = os.getenv('ACCOUNTNAME')
config = toml.load('../config.toml')

# Replace with your Alpaca API credentials
API_KEY = config[accountname]['key']
API_SECRET = config[accountname]['secret']

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
    


def calculate_macd(dfs: List[pd.DataFrame], 
                  fast=12, slow=26, signal=9, hist_ema_span=3, rsi_period=14, rsi_ema_span=3, 
                  mfi_period=14, mfi_ema_span=3):
    """
    Calculate MACD, Signal Line, Smoothed MACD Histogram, Histogram ROC, Smoothed RSI, and MFI.

    Parameters:
    - close: pd.Series, the close prices
    - high: pd.Series, the high prices
    - low: pd.Series, the low prices
    - volume: pd.Series, the volume data
    - fast: int, the period for the fast EMA (default 12)
    - slow: int, the period for the slow EMA (default 26)
    - signal: int, the period for the signal line EMA (default 9)
    - rsi_period: int, period for RSI calculation (default 14)
    - mfi_period: int, period for MFI calculation (default 14)
    - hist_ema_span: int, span for smoothing the MACD histogram (default 3)
    - rsi_ema_span: int, span for smoothing the RSI (default 3)
    - mfi_ema_span: int, span for smoothing the MFI (default 3)

    Returns:
    - pd.DataFrame: A DataFrame with columns ['MACD', 'MACD_signal', 'MACD_hist', 'MACD_hist_roc', 'RSI', 'MFI'].
    """
        
    def calculate_mfi(tp: pd.Series, mf: pd.Series, period: int, smoothing: int):
        # Calculate positive and negative money flow
        delta = tp.diff()
        
        # Initialize as float series instead of int
        positive_flow = pd.Series(0.0, index=tp.index)
        negative_flow = pd.Series(0.0, index=tp.index)
        
        # Use loc for assignment to avoid dtype warning
        positive_flow.loc[delta > 0] = mf[delta > 0]
        negative_flow.loc[delta < 0] = mf[delta < 0]
        
        # Calculate money flow ratio
        positive_mf = positive_flow.rolling(window=period).sum()
        negative_mf = negative_flow.rolling(window=period).sum()
        
        # Avoid division by zero
        money_ratio = np.where(negative_mf != 0, 
                            positive_mf / negative_mf,
                            np.inf)
        
        # Calculate MFI
        mfi = 100 - (100 / (1 + money_ratio))
        
        # Apply smoothing
        smoothed_mfi = pd.Series(mfi).ewm(span=smoothing, adjust=False).mean()
        
        return smoothed_mfi.fillna(50)  # Fill NaN values with 50
    
    def calculate_rsi(series, period, smoothing):
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)  # Fill NaN RSI values with 50
        # Apply smoothing
        smoothed_rsi = rsi.ewm(span=smoothing, adjust=False).mean()
        return smoothed_rsi
    
    results = []
    for df in dfs:
        if df is None:
            results.append(None)
            continue
        
        close, high, low, volume = df['close'], df['high'], df['low'], df['volume']

        # Calculate typical price
        typical_price = (high + low + close) / 3
        
        # Calculate raw money flow
        raw_money_flow = typical_price * volume

        # MACD calculations (unchanged)
        fast_ema = close.ewm(span=fast, adjust=False).mean()
        slow_ema = close.ewm(span=slow, adjust=False).mean()
        macd_line = fast_ema - slow_ema
        macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
        raw_hist = macd_line - macd_signal
        macd_hist = raw_hist.ewm(span=hist_ema_span, adjust=False).mean()
        macd_hist_roc = macd_hist.diff().fillna(0)

        rsi = calculate_rsi(close, rsi_period, rsi_ema_span)
        mfi = calculate_mfi(typical_price, raw_money_flow, mfi_period, mfi_ema_span)
        
        # Calculate ROC for both RSI and MFI
        rsi_roc = rsi.diff().fillna(0)
        mfi_roc = mfi.diff().fillna(0)
        
        mfi.index = rsi.index.copy()
        mfi_roc.index = rsi.index.copy()

        # Combine into a DataFrame
        result = pd.DataFrame({
            'MACD': macd_line.fillna(0),
            'MACD_signal': macd_signal.fillna(0),
            'MACD_hist': macd_hist.fillna(0),
            'MACD_hist_roc': macd_hist_roc,
            'RSI': rsi.fillna(50),
            'RSI_roc': rsi_roc,
            'MFI': mfi.fillna(50),
            'MFI_roc': mfi_roc
        })
        results.append(result)
    return results

@dataclass
class Level:
    id: int
    lmin: float
    lmax: float
    level: float
    is_res: bool
    touches: List[datetime]
    
@jit(nopython=True)
def np_searchsorted(a:np.ndarray,b:np.ndarray): # only used in calculate_touch_area function
    return np.searchsorted(a,b)


@jit(nopython=True)
def process_touches(touch_indices, prices, atrs, level, lmin, lmax, is_long, min_touches, touch_area_width_agg, calculate_bounds, multiplier):
    # Initialize array to store indices of consecutive touches
    consecutive_touch_indices = np.full(min_touches, -1, dtype=np.int64)
    count, width = 0, 0
    prev_price = None
    
    for i in range(len(prices)):
        price = prices[i]
        
        # A touch occurs when price crosses the level (from either direction) or equals it
        is_touch = (prev_price is not None and 
                    ((prev_price < level <= price) or (prev_price > level >= price)) or 
                    (price == level))
        
        # Check if price is within the level's min-max range
        if lmin <= price <= lmax:
            if is_touch:
                # Update bounds using ATR values up to this point
                width, touch_area_low, touch_area_high = calculate_bounds(atrs[:i+1], level, is_long, touch_area_width_agg, multiplier)
                
                if width > 0:
                    # Record this touch
                    consecutive_touch_indices[count] = touch_indices[i]
                    count += 1
                    
                    # If we have enough touches, return
                    if count == min_touches:
                        return consecutive_touch_indices[consecutive_touch_indices != -1], touch_area_low, touch_area_high
                
        # If price moves beyond bounds in wrong direction, reset the count
        elif width > 0:
            assert touch_area_high is not None and touch_area_low is not None
            buy_price = touch_area_high if is_long else touch_area_low
            if (is_long and price > buy_price) or (not is_long and price < buy_price):
                consecutive_touch_indices[:] = -1
                count = 0
        
        prev_price = price
    return np.empty(0, dtype=np.int64), touch_area_low, touch_area_high

def calculate_touch_area(levels_by_date: Dict[datetime, List[Level]], is_long, df: pd.DataFrame, symbol, market_hours: Dict[date, Tuple[datetime, datetime]], min_touches, \
    use_median, touch_area_width_agg: Callable, calculate_bounds: Callable, multiplier, trading_times=None):
    touch_areas = []
    widths = []

    # for date, levels in tqdm(levels_by_date.items(), desc='calculate_touch_area'):
    for date, levels in levels_by_date.items():
        day_start_time, day_end_time = trading_times[date]
        
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
            touch_indices = np.searchsorted(day_timestamps_np, touch_timestamps_np)
            
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
                calculate_bounds,
                multiplier
            )
            
            if len(consecutive_touch_indices) == min_touches:
                consecutive_touches = day_timestamps[consecutive_touch_indices]
                
                assert consecutive_touches[0] >= day_start_time
                
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
                    valid_atr=valid_atr,
                    touch_area_width_agg=touch_area_width_agg,
                    calculate_bounds=calculate_bounds,
                    multiplier=multiplier,
                )

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



@dataclass
class TouchDetectionState:
    """Maintains state between calls to calculate_touch_detection_area."""
    current_date: date
    potential_levels: Dict[int, Level] = field(default_factory=lambda: defaultdict(lambda: Level(0, 0, 0, 0, False, [])))
    high_low_diffs: List[float] = field(default_factory=list)
    last_processed_index: int = 0


def calculate_vwap_std_metrics(df: pd.DataFrame, vwap_col: str = 'VWAP') -> pd.DataFrame:
    """
    Calculate VWAP standard deviation metrics using pandas operations.
    
    Args:
        df: DataFrame with 'volume' and vwap_col columns
        vwap_col: Name of the VWAP column to calculate std against
        
    Returns:
        DataFrame with added columns:
        - VWAP_std: Running std dev of price from VWAP
        - VWAP_std_close: Close price std dev from VWAP
    """
    # Group by date for daily calculations
    grouped = df.groupby(df.index.get_level_values('timestamp').date)
    
    # Calculate running squared deviations weighted by volume
    df['squared_dev'] = df['volume'] * (df['vwap'] - df[vwap_col])**2
    
    # Calculate running sum of squared deviations and volumes
    running_sq_dev = grouped['squared_dev'].cumsum()
    running_volume = grouped['volume'].cumsum()
    
    # Calculate std where volume > 0
    df['VWAP_std'] = np.sqrt(
        np.where(running_volume > 0, 
                running_sq_dev / running_volume,
                np.nan)
    )
    
    # Calculate standardized close price deviation
    df['VWAP_std_close'] = np.where(
        df['VWAP_std'] > 0,
        (df['close'] - df[vwap_col]) / df['VWAP_std'],
        np.nan
    )
    
    # Clean up intermediate column
    df.drop(columns=['squared_dev'], inplace=True)
    
    # return df

    
    
def calculate_touch_detection_area(params: BacktestTouchDetectionParameters | LiveTouchDetectionParameters, 
                                   live_bars: Optional[pd.DataFrame] = None, 
                                   market_hours: Optional[Dict[date, Tuple[datetime, datetime]]] = None,
                                   area_ids_to_remove: Optional[set] = {},
                                   previous_state: Optional[TouchDetectionState] = None) -> Union[TouchDetectionAreas, Tuple[TouchDetectionAreas, TouchDetectionState]]:
    """
    Calculates touch detection areas for a given stock symbol based on historical price data and volatility.
    
    Inspired by https://medium.com/@paullenosky/i-have-created-an-indicator-that-actually-makes-money-unlike-any-other-indicator-i-have-ever-seen-fd7b36aba975
    
    This function analyzes historical price data to identify significant price levels (support and resistance)
    based on the frequency of price touches. It considers market volatility using ATR and allows for 
    customization of the analysis parameters. The resulting touch areas can be used for trading strategies
    or further market analysis.
    """
    
    assert params.client_type in {'stock','crypto'}, params.client_type
    
    def log_live(message, level=logging.INFO):
        if params.__class__.__name__ == 'LiveTouchDetectionParameters':
            logger.log(level, message, exc_info=level >= logging.ERROR)
    def log_backtest(message, level=logging.INFO):
        if params.__class__.__name__ == 'BacktestTouchDetectionParameters':
            logger.log(level, message, exc_info=level >= logging.ERROR)
    
    is_live = params.__class__.__name__ == 'LiveTouchDetectionParameters'

    if not is_live:
        if not params.__class__.__name__ == 'BacktestTouchDetectionParameters':
            raise ValueError("Invalid parameter type")
        assert params.end_date > params.start_date

        # Alpaca API setup
        if params.client_type == 'stock':
            client = StockHistoricalDataClient(api_key=API_KEY, secret_key=API_SECRET)
        else:
            assert params.client_type == 'crypto'
            client = CryptoHistoricalDataClient(api_key=API_KEY, secret_key=API_SECRET)

        # get bars data (2 dataframes)
        df_adjusted, df = retrieve_bar_data(client, params)
        
        # get quotes data (2 dataframes)
        minute_intervals = df.index.get_level_values('timestamp')
        # minute_intervals = minute_intervals[(minute_intervals.time >= time(9, 30)) & (minute_intervals.time < time(16, 0))]
        # print(minute_intervals)
        minute_intervals_dict = {params.symbol: minute_intervals}
        
        quotes_data = retrieve_quote_data(client, [params.symbol], minute_intervals_dict, params)
        
        if isinstance(quotes_data[params.symbol]['raw'], pd.DataFrame) and isinstance(quotes_data[params.symbol]['agg'], pd.DataFrame):
            quotes_raw_df = quotes_data[params.symbol]['raw']
            quotes_agg_df = quotes_data[params.symbol]['agg']
        else:
            raise ValueError(f"Quote data not found for symbol {params.symbol}")

    else:
        if live_bars is None:
            raise ValueError("Live bars data must be provided for live trading parameters")
        df = live_bars
        df_adjusted = None
        quotes_raw_df = None
        quotes_agg_df = None
    
    try:
        log_live('Data retrieved')
        
        timestamps = df.index.get_level_values('timestamp')
        # print(timestamps)
        # print(df.columns)
        # print(df.dtypes)
        # print(df)
        
        # calculate_dynamic_central_value(df, ema_short=9, ema_long=30) # default
        # calculate_dynamic_central_value(df)
        
        
        # df['MACD'], df['MACD_signal'], df['MACD_Hist'] = ta.macd(df['close'], fast=5, slow=13, signal=4)
        
        # 'MACD': macd_line,
        # 'MACD_signal': signal_line,
        # 'MACD_hist': macd_histogram,
        # 'MACD_hist_roc': histogram_roc
        # 'RSI': rsi,

        # macd_df = calculate_macd(df['close'], fast=5, slow=13, signal=4, hist_ema_span=5, rsi_period=13, rsi_ema_span=3)
        # macd_df = calculate_macd(df['close'], fast=5, slow=13, signal=4, hist_ema_span=5, rsi_period=8, rsi_ema_span=3)
        # macd_df = calculate_macd(df['close'], fast=5, slow=13, signal=4, hist_ema_span=5, rsi_period=5, rsi_ema_span=3)
        # macd_df = calculate_macd(df['close'], df['high'], df['low'], df['volume'], fast=5, slow=13, signal=4, hist_ema_span=5, rsi_period=5, rsi_ema_span=5, mfi_period=5, mfi_span=5) # 5 and 5 for RSI seems best
        # macd_df = calculate_macd(df['close'], df['high'], df['low'], df['volume'], fast=5, slow=13, signal=4, hist_ema_span=5, rsi_period=5, rsi_ema_span=5, mfi_period=5, mfi_span=5)
        # macd_df = calculate_macd(df['close'], fast=5, slow=13, signal=4, hist_ema_span=5, rsi_period=5, rsi_ema_span=4)
        # macd_df = calculate_macd(df['close'], fast=5, slow=13, signal=4, hist_ema_span=5, rsi_period=5, rsi_ema_span=6)
                
        macd_dfs = calculate_macd(
            [df,df_adjusted],
            fast=5, slow=13, signal=4,
            hist_ema_span=5,
            # rsi_period=5, 
            rsi_period=9,        # Increased from 5
            rsi_ema_span=3,
            # rsi_ema_span=3,      # Reduced from 5
            # mfi_period=5,
            mfi_period=9,
            # mfi_period=14,       # Increased from 5
            mfi_ema_span=3
            # mfi_ema_span=4          # Slightly reduced from 5
        )
        
        # print(macd_df)
        
        df = pd.concat([df, macd_dfs[0]],axis=1)
        df_adjusted = pd.concat([df_adjusted, macd_dfs[1]],axis=1)
        # print(df)

        # df.drop(columns=['vwap'],errors='ignore',inplace=True) # seems incorrect. calculate vwap manually:
        
        def calculate_vwap(df: pd.DataFrame):

            # Get typical price
            # df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
            # Calculate price * volume 
            # df['pv'] = df['typical_price'] * df['volume']
            # Group by date to reset calculations daily
            grouped = df.groupby(df.index.get_level_values('timestamp').date)
            # Calculate running sums within each day
            # df['VWAP'] = grouped.apply(lambda x: x['pv'].cumsum() / x['volume'].cumsum()).T.values
            df['VWAP'] = grouped.apply(lambda x: (x['vwap'] * x['volume']).cumsum() / x['volume'].cumsum()).T.values
            
            calculate_vwap_std_metrics(df)
            
            df.drop(columns=['typical_price', 'pv'],errors='ignore',inplace=True)
            
            
        calculate_vwap(df)
        calculate_vwap(df_adjusted)
        
        # vwap is bar vwap
        # VWAP is day vwap
        
        # central_value_var = 'close'
        central_value_var = 'vwap'
        
        
        # df['central_value'] = (df['vwap'] + calculate_ema_with_cutoff(df, 'close', span=params.price_ema_span) * 2) / 3
        df['central_value'] = calculate_ema_with_cutoff(df, central_value_var, span=params.price_ema_span)
        # df['central_value'] = df['vwap']

        df_adjusted['central_value'] = calculate_ema_with_cutoff(df_adjusted, central_value_var, span=params.price_ema_span)
        
        
        exit_ema_var = 'close'
        # exit_ema_var = 'vwap'
        
        df['exit_ema'] = calculate_ema_with_cutoff(df, exit_ema_var, span=params.exit_ema_span)
        df_adjusted['exit_ema'] = calculate_ema_with_cutoff(df_adjusted, exit_ema_var, span=params.exit_ema_span)
        
        
        df['is_res'] = df['close'] >= df['central_value'] # if is_res, trade long
        # df['is_res'] = df['close'] < df['central_value'] # mean reversion strategy

        # Calculate True Range (TR)
        df['H_L'] = df['high'] - df['low']
        df['H_PC'] = np.abs(df['high'] - df['close'].shift(1))
        df['L_PC'] = np.abs(df['low'] - df['close'].shift(1))
        df['TR'] = df[['H_L', 'H_PC', 'L_PC']].max(axis=1)
        
        
        df['ATR'] = calculate_ema_with_cutoff(df, 'TR', span=params.ema_span) # , window=params.atr_period
        df['MTR'] = df['TR'].rolling(window=params.atr_period).apply(lambda x: np.median(x), raw=True)
        
        log_live('ATR and MTR calculated')

        # Mean: This is more sensitive to outliers and can be useful if you want your strategy to react more quickly to sudden changes in volume or trade count.
        # Median: This is more robust to outliers and can provide a more stable measure of the typical volume or trade count, which might be preferable if you want 
        # your strategy to be less affected by occasional spikes in activity.
        
        # Calculate rolling average volume and trade count
        # df['shares_per_trade'] = df['volume'] / df['trade_count']
        df['avg_volume'] = calculate_ema_with_cutoff(df, 'volume', span=params.ema_span) # , window=params.level1_period
        df['avg_trade_count'] = calculate_ema_with_cutoff(df, 'trade_count', span=params.ema_span) # , window=params.level1_period
        
        # df['avg_shares_per_trade'] = df['shares_per_trade'].rolling(window=params.level1_period).mean()
        # df['avg_shares_per_trade'] = calculate_ema_with_cutoff(df, 'shares_per_trade', span=params.ema_span, window=params.level1_period) 
        
        log_live('rolling averages calculated')
        
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['volatility'] = df['log_return'].rolling(window=15).std().fillna(0)
            
        # Narrow Range calculations
        df['rolling_range_min_4'] = df['H_L'].rolling(window=4, min_periods=1).min()
        df['rolling_range_min_7'] = df['H_L'].rolling(window=7, min_periods=1).min()
        
        # Add ATR-based calculations
        df['rolling_ATR'] = df['ATR'].rolling(window=15, min_periods=1).mean()
        
                
        # Calculate distance to central value/EMA - positive when price above
        # df['central_value_dist'] = (df[central_value_var] - df['central_value']) / df[central_value_var] # Normalize by price level
        df['central_value_dist'] = (df['close'] - df['central_value']) # / df['MTR' if params.use_median else 'ATR']# Normalize by price level
        # df['central_value_dist'] = (df[central_value_var] - df['central_value'])
        
        
        df['exit_ema_dist'] = (df[exit_ema_var] - df['exit_ema'])
        
        df['VWAP_dist'] = df['close'] - df['VWAP']

        # Calculate basic ADX components
        df['plus_dm'] = df['high'] - df['high'].shift(1) 
        df['minus_dm'] = df['low'].shift(1) - df['low']
        df['plus_dm'] = np.where((df['plus_dm'] > df['minus_dm']) & (df['plus_dm'] > 0), df['plus_dm'], 0)
        df['minus_dm'] = np.where((df['minus_dm'] > df['plus_dm']) & (df['minus_dm'] > 0), df['minus_dm'], 0)

        # Smooth DMs and TR using EMA
        df['smoothed_plus_dm'] = calculate_ema_with_cutoff(df, 'plus_dm', span=params.ema_span)
        df['smoothed_minus_dm'] = calculate_ema_with_cutoff(df, 'minus_dm', span=params.ema_span)
        df['smoothed_tr'] = calculate_ema_with_cutoff(df, 'TR', span=params.ema_span)

        # Calculate DIs and ADX
        df['plus_di'] = (df['smoothed_plus_dm'] / df['smoothed_tr']) * 100
        df['minus_di'] = (df['smoothed_minus_dm'] / df['smoothed_tr']) * 100

        # DX is absolute difference between the DIs divided by their sum
        df['dx'] = abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']) * 100

        # ADX is smoothed DX
        df['ADX'] = calculate_ema_with_cutoff(df, 'dx', span=params.ema_span)

        # Directional trend strength (positive values indicate bullish, negative bearish)
        df['trend_strength'] = df['plus_di'] - df['minus_di']

        # Clean up intermediate columns
        df.drop(columns=[
            'plus_dm', 'minus_dm', 'smoothed_plus_dm', 'smoothed_minus_dm',
            'smoothed_tr', 'plus_di', 'minus_di', 'dx',
            'H_PC','L_PC','TR', 'typical_price', 'pv'
        ],errors='ignore',inplace=True)

        log_live('bar metrics calculated',level=logging.DEBUG)
        
        
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
        trading_times = {}
        
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
            # elif start_time:
            #     day_start_time, day_end_time = date_obj, date_obj.replace(hour=start_time.hour, minute=start_time.minute)
            else:
                day_start_time, day_end_time = date_obj, date_obj
            
            mask = (timestamps >= day_start_time) & (timestamps <= day_end_time)
            final_mask |= mask
            trading_times[date] = (day_start_time, day_end_time)

        log_live('Mask created',level=logging.DEBUG)
        
        
        # Initialize state tracking
        current_state = None
    
        # Create filtered df for level detection
        df_filtered = df[final_mask]
        atr_grouped = df.groupby(timestamps.date)
        
        # Use df_filtered for level detection
        grouped = df_filtered.groupby(df_filtered.index.get_level_values('timestamp').date)
        
        all_support_levels = defaultdict(list)
        all_resistance_levels = defaultdict(list)

        # def is_res_area(index, day_df):
        #     return day_df.loc[index, 'close'] > day_df.loc[index, 'central_value']
        
        # high_low_diffs_list = []
        
        # for date, day_df in tqdm(grouped, desc='calculate_touch_detection_area'):
        for date, day_df in grouped:
            day_timestamps = day_df.index.get_level_values('timestamp') # need to limit
            
            # For live trading, reuse or create state
            if is_live:
                if previous_state is None or previous_state.current_date != date:
                    current_state = TouchDetectionState(current_date=date)
                else:
                    current_state = previous_state
                    
                potential_levels = current_state.potential_levels
                high_low_diffs = current_state.high_low_diffs
                start_idx = current_state.last_processed_index
            else:
                potential_levels = defaultdict(lambda: Level(0, 0, 0, 0, False, []))
                high_low_diffs = []
                start_idx = 0
            # high_close_diffs = []
            # low_close_diffs = []
                    
            # Get all valid ATR values up to day_start_time
            day_start_time, day_end_time = trading_times[date]
            if start_idx == 0 and date in atr_grouped.groups:
                day_atr_df = df.loc[atr_grouped.groups[date]]
                pre_market_df = day_atr_df[day_atr_df.index.get_level_values('timestamp') < day_start_time]
                # print(pre_market_df[['volume','trade_count']].describe())
                # print(len(pre_market_df['ATR']), np.median(pre_market_df['ATR']))
                historical_atrs = pre_market_df.loc[
                    (pre_market_df['volume'] > 0) & 
                    (pre_market_df['trade_count'] > 0)
                , 'ATR']
                # print(len(historical_atrs), np.median(historical_atrs))
                high_low_diffs.extend(historical_atrs.tolist())
                    
            # print(day_df[['volume','trade_count']].describe())
            # print(pd.Series(high_low_diffs).describe())
            # n = len(high_low_diffs)
            # print(n)
            
            for i in range(start_idx, len(day_df)):
            # for i in range(len(day_df)):
            # for i in tqdm(range(len(day_df))):
                row = day_df.iloc[i]
                if row['volume'] <= 0 or row['trade_count'] <= 0:
                    continue
                
                # high, low, close = row['high'], row['low'], row['close']
                # high, low, close = row['high'], row['low'], row['close']
                h_l, atr, close = row['H_L'], row['ATR'], row['close']
                low, high, vwap = row['low'], row['high'], row['vwap'] # TODO: utilize these values
                timestamp = day_timestamps[i]
                is_res = row['is_res']
                
                # high_low_diffs.append(high - low)
                # high_low_diffs.append(h_l)
                high_low_diffs.append(atr)
                # high_close_diffs.append(high - close)
                # low_close_diffs.append(close - low)
                
                # w = np.median(high_low_diffs) / 2
                w = np.median(high_low_diffs) / 3
                # w = (np.median(high_low_diffs) / 3) * params.multiplier
                # w = atr
                # w = np.median(high_low_diffs[-15:]) / 2
                # lmin, lmax = close - w, close + w
                
                
                if is_res:
                    lmin, lmax = high - (2*w), high + w
                else:
                    lmin, lmax = low - w, low + (2*w)

                # w_high = np.median(high_close_diffs)
                # w_low = np.median(low_close_diffs)
                # lmin, lmax = close - w_low, close + w_high
                
                # Check if this point falls within any existing levels
                for level in potential_levels.values():
                    # if level.is_res == is_res and level.lmin <= close <= level.lmax:
                    #     level.touches.append(timestamp)
                    if level.is_res == is_res:
                        if (is_res and level.lmin <= high <= level.lmax) or (not is_res and level.lmin <= low <= level.lmax):
                            level.touches.append(timestamp)

                if w != 0 and i not in potential_levels:
                # if (w_high != 0 or w_low != 0) and i not in potential_levels:
                    # Add this point to its own level (match by lmin, lmax in case the same lmin, lmax is already used)
                    # print(w)
                    # potential_levels[i] = Level(i, lmin, lmax, close, is_res, [timestamp]) # using i as ID since levels have unique INITIAL timestamp
                    if is_res:
                        potential_levels[i] = Level(i, lmin, lmax, high, is_res, [timestamp])
                    else:
                        potential_levels[i] = Level(i, lmin, lmax, low, is_res, [timestamp])

                # lmin_res, lmax_res = high - (2*w), high + w
                # lmin_sup, lmax_sup = low - w, low + (2*w)

                # # Check if this point falls within any existing levels
                # for level in potential_levels.values():
                #     if level.is_res and level.lmin <= high <= level.lmax:
                #         level.touches.append(timestamp)
                #     if not level.is_res and level.lmin <= low <= level.lmax:
                #         level.touches.append(timestamp)
                
                # i_res = i*2
                # i_sup = i_res+1

                # if w != 0:
                #     if i_res not in potential_levels:
                #         potential_levels[i_res] = Level(i_res, lmin_res, lmax_res, high, True, [timestamp])
                #     if i_sup not in potential_levels:
                #         potential_levels[i_sup] = Level(i_sup, lmin_sup, lmax_sup, low, False, [timestamp])
                
            
            # print(pd.Series(high_low_diffs[n:]).describe())
            
            if date in area_ids_to_remove:
                for i in area_ids_to_remove[date]:
                    if i in potential_levels:
                        del potential_levels[i] # area id == level id
                        
            # Update state if live trading
            if is_live:
                current_state.last_processed_index = len(day_df)
                # current_state.high_low_diffs = high_low_diffs # already aliased
                # current_state.potential_levels = potential_levels # already aliased
            
            # a = pd.DataFrame(pd.Series(high_low_diffs).describe()).T
            # a['date'] = date
            # high_low_diffs_list.append(a)
            # print(a)

            # Classify levels as support or resistance
            # Filter for strong levels
            for level in potential_levels.values():
                if len(level.touches) < params.min_touches:
                    continue
                if level.is_res:
                    all_resistance_levels[date].append(level)
                else:
                    all_support_levels[date].append(level)

        log_live('Levels created',level=logging.DEBUG)

        
        long_touch_area, long_widths = calculate_touch_area(
            all_resistance_levels, True, df_filtered, params.symbol, market_hours, params.min_touches,
            params.use_median, params.touch_area_width_agg, params.calculate_bounds, params.multiplier,
            trading_times=trading_times
        )
        log_live(f'{len(long_touch_area)} Long touch areas calculated',level=logging.DEBUG)
        short_touch_area, short_widths = calculate_touch_area(
            all_support_levels, False, df_filtered, params.symbol, market_hours, params.min_touches,
            params.use_median, params.touch_area_width_agg, params.calculate_bounds, params.multiplier,
            trading_times=trading_times
        )
        log_live(f'{len(short_touch_area)} Short touch areas calculated',level=logging.DEBUG)
            
        # widths = long_widths + short_widths

        # df = df.drop(columns=['H_PC','L_PC','TR']) # 'H_L',  ,'ATR','MTR'
        
        ret = {
            'symbol': df.index.get_level_values('symbol')[0] if isinstance(params, LiveTouchDetectionParameters) else params.symbol,
            'long_touch_area': long_touch_area,
            'short_touch_area': short_touch_area,
            'market_hours': market_hours,
            'bars': df,
            'bars_adjusted': df_adjusted,
            'quotes_raw': quotes_raw_df, # None for LiveTrader since not doing limit pricing
            'quotes_agg': quotes_agg_df, # None for LiveTrader since not doing limit pricing
            'mask': final_mask,
            'min_touches': params.min_touches,
            'start_time': start_time,
            'end_time': end_time,
        }
        areas = TouchDetectionAreas.from_dict(ret) # , high_low_diffs_list
        
        # Return additional state for live trading
        if is_live:
            return areas, current_state
        return areas
        
    except Exception as e:
        log(f"{type(e).__qualname__} in calculate_touch_detection_area: {e}", level=logging.ERROR)
        raise e


def plot_touch_detection_areas(touch_detection_areas: TouchDetectionAreas, zoom_start_date=None, zoom_end_date=None, save_path=None, filter_date=None, filter_areas=None, 
                               trades: List[TradePosition] = None, rsi_overbought = 70, rsi_oversold = 30, mfi_overbought = 80, mfi_oversold = 20):
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
    # df = touch_detection_areas.bars
    df = touch_detection_areas.bars_adjusted
    # df_adjusted = touch_detection_areas.bars_adjusted
    mask = touch_detection_areas.mask
    # min_touches = touch_detection_areas.min_touches
    start_time = touch_detection_areas.start_time
    end_time = touch_detection_areas.end_time
    # use_median = touch_detection_areas.use_median


    timestamps = df.index.get_level_values('timestamp')
    # Calculate bar width based on time intervals

    
    # plt.figure(figsize=(14, 7))
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15), height_ratios=[4, 1, 1], sharex=True)
    plt.subplots_adjust(hspace=0.05)
    


    if filter_date:
        df = df.loc[df.index.get_level_values('timestamp').date == filter_date]
        mask = mask.loc[mask.index.get_level_values('timestamp').date == filter_date]
        timestamps = df.index.get_level_values('timestamp')
        
    
    grouped = df.groupby(df.index.get_level_values('timestamp').date)
    if len(grouped) == 1:
        # Convert timestamps once to matplotlib format
        dates = mdates.date2num(timestamps.to_pydatetime())
        time_diff = dates[1] - dates[0]  # Use converted dates for width
        bar_width = time_diff  # No need to convert to days, already in right format
        
        # Create a secondary y-axis for volume
        ax1_volume = ax1.twinx()
        max_volume = df['volume'].max()

        # Plot volume bars at the bottom 15% of the chart
        up_mask = df['close'] >= df['open']
        down_mask = ~up_mask

        # Plot all up volume bars at once
        ax1_volume.bar(dates[up_mask], 
                    df['volume'][up_mask] * 0.15 / max_volume,
                    width=bar_width,
                    color='lightgreen',
                    edgecolor='green',
                    alpha=0.75,
                    linewidth=0.4)

        # Plot all down volume bars at once
        ax1_volume.bar(dates[down_mask],
                    df['volume'][down_mask] * 0.15 / max_volume,
                    width=bar_width,
                    color='lightcoral',
                    edgecolor='red',
                    alpha=0.75,
                    linewidth=0.4)

        # Make the volume axis invisible but keep the bars
        ax1_volume.set_ylim(0, 1)  # Full height of chart
        ax1_volume.set_ylabel('')
        ax1_volume.set_yticklabels([])
        
        # Plot all price bars at once
        up_idx = np.where(up_mask)[0]
        down_idx = np.where(down_mask)[0]
    
        
        # Plot up bars
        if len(up_idx) > 0:
            body_bottom = df['open'].iloc[up_idx]
            body_height = df['close'].iloc[up_idx] - df['open'].iloc[up_idx]
            ax1.bar(dates[up_idx], body_height, bottom=body_bottom,
                    width=bar_width, color='green', edgecolor='green', linewidth=0)
            
            # Plot up wicks
            for idx in up_idx:
                x = dates[idx]
                ax1.vlines(x, df['low'].iloc[idx], df['high'].iloc[idx],
                        color='green', linewidth=0.5)

        # Plot down bars
        if len(down_idx) > 0:
            body_bottom = df['close'].iloc[down_idx]
            body_height = df['open'].iloc[down_idx] - df['close'].iloc[down_idx]
            ax1.bar(dates[down_idx], body_height, bottom=body_bottom,
                    width=bar_width, color='red', edgecolor='red', linewidth=0)
            
            # Plot down wicks
            for idx in down_idx:
                x = dates[idx]
                ax1.vlines(x, df['low'].iloc[idx], df['high'].iloc[idx],
                        color='red', linewidth=0.5)


    # Main price plot (ax1)
    ax1.plot(df.index.get_level_values('timestamp'), df['central_value'], label='Central Value', color='yellow', linewidth=1)
    ax1.plot(df.index.get_level_values('timestamp'), df['VWAP'], label='VWAP', color='purple', linewidth=1)
    if len(grouped) != 1:
        ax1.plot(df.index.get_level_values('timestamp'), df['close'], label='Close Price', color='blue', linewidth=1)
            
    # MACD plot with ROC (ax2)
    ax2_twin = ax2.twinx()


    # Get the values for both metrics
    macd_hist = df['MACD_hist'].values
    macd_roc = df['MACD_hist_roc'].values
    
    # Create line plots for both metrics
    macd_line = ax2.plot(timestamps, macd_hist, label='MACD Hist', color='purple', linewidth=1)
    roc_line = ax2_twin.plot(timestamps, macd_roc, label='MACD Hist RoC', color='orange', linewidth=1)
    
    # Add zero lines
    ax2.axhline(y=0, color='black', linestyle=':', alpha=0.3)
    ax2_twin.axhline(y=0, color='black', linestyle=':', alpha=0.3)
    
    # Set labels and colors
    ax2.set_ylabel('MACD Histogram', color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2_twin.set_ylabel('MACD Rate of Change', color='orange')
    ax2_twin.tick_params(axis='y', labelcolor='orange')
    
    # Align zero lines for both y-axes
    left_ylim = ax2.get_ylim()
    right_ylim = ax2_twin.get_ylim()
    left_range = left_ylim[1] - left_ylim[0]
    right_range = right_ylim[1] - right_ylim[0]
    
    left_zero_pos = -left_ylim[0] / left_range
    right_zero_pos = -right_ylim[0] / right_range
    
    if left_zero_pos < right_zero_pos:
        # Adjust right axis
        new_bottom = -right_ylim[1] * left_zero_pos / (1 - left_zero_pos)
        ax2_twin.set_ylim(new_bottom, right_ylim[1])
    else:
        # Adjust left axis
        new_bottom = -left_ylim[1] * right_zero_pos / (1 - right_zero_pos)
        ax2.set_ylim(new_bottom, left_ylim[1])
    
    # Add combined legend
    lines = macd_line + roc_line
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, loc='upper left')
    ax2.grid(True)
    
    # RSI and MFI plot (ax3)
    ax3_twin = ax3.twinx()
    
    # Plot RSI on left axis
    ax3.plot(df.index.get_level_values('timestamp'), df['RSI'], label='RSI', color='purple', linewidth=1)
    ax3.axhline(y=rsi_overbought, color='purple', linestyle=':', alpha=0.3, label='RSI Overbought')
    ax3.axhline(y=rsi_oversold, color='purple', linestyle=':', alpha=0.3, label='RSI Oversold')
    ax3.axhline(y=50, color='black', linestyle=':', alpha=0.3)
    ax3.set_ylim(0, 100)
    ax3.set_ylabel('RSI', color='purple')
    ax3.tick_params(axis='y', labelcolor='purple')
    
    # Plot MFI on right axis
    ax3_twin.plot(df.index.get_level_values('timestamp'), df['MFI'], label='MFI', color='orange', linewidth=1)
    ax3_twin.axhline(y=mfi_overbought, color='orange', linestyle=':', alpha=0.3, label='MFI Overbought')
    ax3_twin.axhline(y=mfi_oversold, color='orange', linestyle=':', alpha=0.3, label='MFI Oversold')
    ax3_twin.set_ylim(0, 100)
    ax3_twin.set_ylabel('MFI', color='orange')
    ax3_twin.tick_params(axis='y', labelcolor='orange')
    
    # Combine legends from both axes
    lines_rsi, labels_rsi = ax3.get_legend_handles_labels()
    lines_mfi, labels_mfi = ax3_twin.get_legend_handles_labels()
    # ax3.legend(lines_rsi + lines_mfi, labels_rsi + labels_mfi, loc='upper left')
    ax3.legend().remove()
    
    # Keep grid only on the main axis
    ax3.grid(True)
    
    # Helper function to plot trade markers
    def plot_trade_markers(ax, ax_twin, y_value, long_entries, long_exits, long_iswin, short_entries, short_exits, short_iswin, plot_ids=False, custom_marker=None):
        if ax_twin:
            axes = [ax, ax_twin]
            zorder_value = 100
        else:
            axes = [ax]
            zorder_value = 10
        
        for plot_ax in axes:
            if long_entries:
                times, _, ids = zip(*long_entries)
                colors = ['lime' if iswin else 'orangered' for iswin in long_iswin]
                plot_ax.scatter(times, [y_value] * len(times), c=colors, marker='+' if custom_marker is None else custom_marker,
                            s=30, zorder=zorder_value, alpha=1)
                if plot_ids and plot_ax == ax:  # Only plot IDs once, on the main axis
                    for time, trade_id in zip(times, ids):
                        plot_ax.text(time, y_value, str(trade_id), fontsize=8,
                                horizontalalignment='right', verticalalignment='bottom',
                                color='black', zorder=zorder_value)
            
            if long_exits:
                times, _, _ = zip(*long_exits)
                colors = ['lime' if iswin else 'orangered' for iswin in long_iswin]
                plot_ax.scatter(times, [y_value] * len(times), c=colors, marker='x' if custom_marker is None else custom_marker,
                            s=30, zorder=zorder_value, alpha=1)
            
            if short_entries:
                times, _, ids = zip(*short_entries)
                colors = ['lime' if iswin else 'orangered' for iswin in short_iswin]
                plot_ax.scatter(times, [y_value] * len(times), c=colors, marker='+' if custom_marker is None else custom_marker,
                            s=30, zorder=zorder_value, alpha=1)
                if plot_ids and plot_ax == ax:  # Only plot IDs once, on the main axis
                    for time, trade_id in zip(times, ids):
                        plot_ax.text(time, y_value, str(trade_id), fontsize=8,
                                horizontalalignment='right', verticalalignment='top',
                                color='black', zorder=zorder_value)
            
            if short_exits:
                times, _, _ = zip(*short_exits)
                colors = ['lime' if iswin else 'orangered' for iswin in short_iswin]
                plot_ax.scatter(times, [y_value] * len(times), c=colors, marker='x' if custom_marker is None else custom_marker,
                            s=30, zorder=zorder_value, alpha=1)

    if trades:
        traded_areas = {position.area.id: position.area for position in trades}
        
        # Collect trade points with IDs
        long_entries = [(t.entry_time, df.loc[df.index.get_level_values('timestamp') == t.entry_time, 'close'].iloc[0], t.id) 
                       for t in trades if t.is_long]
        long_exits = [(t.exit_time, df.loc[df.index.get_level_values('timestamp') == t.exit_time, 'close'].iloc[0], t.id) 
                     for t in trades if t.is_long]
        long_iswin = [t.pl > 0 for t in trades if t.is_long]
        
        short_entries = [(t.entry_time, df.loc[df.index.get_level_values('timestamp') == t.entry_time, 'close'].iloc[0], t.id) 
                        for t in trades if not t.is_long]
        short_exits = [(t.exit_time, df.loc[df.index.get_level_values('timestamp') == t.exit_time, 'close'].iloc[0], t.id) 
                      for t in trades if not t.is_long]
        short_iswin = [t.pl > 0 for t in trades if not t.is_long]

        if long_iswin:
            long_colors = ['lime' if iswin else 'orangered' for iswin in long_iswin]
        if short_iswin:
            short_colors = ['lime' if iswin else 'orangered' for iswin in short_iswin]
        # Plot on price chart (ax1)
        if long_entries:
            times, prices, ids = zip(*long_entries)
            ax1.scatter(times, prices, c=long_colors, marker='+', s=30, label='Long Entry', 
                       zorder=5, alpha=1)
            for time, price, trade_id in zip(times, prices, ids):
                ax1.text(time, price, str(trade_id), fontsize=8,
                        horizontalalignment='right', verticalalignment='bottom',
                        color='black', zorder=5)
        
        if long_exits:
            times, prices, _ = zip(*long_exits)
            ax1.scatter(times, prices, c=long_colors, marker='x', s=30, label='Long Exit', 
                       zorder=5, alpha=1)
            
        if short_entries:
            times, prices, ids = zip(*short_entries)
            ax1.scatter(times, prices, c=short_colors, marker='+', s=30, label='Short Entry', 
                       zorder=5, alpha=1)
            for time, price, trade_id in zip(times, prices, ids):
                ax1.text(time, price, str(trade_id), fontsize=8,
                        horizontalalignment='right', verticalalignment='top',
                        color='black', zorder=5)
            
        if short_exits:
            times, prices, _ = zip(*short_exits)
            ax1.scatter(times, prices, c=short_colors, marker='x', s=30, label='Short Exit', 
                       zorder=5, alpha=1)
        
        # Plot on MACD chart (ax2)
        plot_trade_markers(ax2, ax2_twin, 0, long_entries, long_exits, long_iswin, short_entries, short_exits, short_iswin, plot_ids=True) # , custom_marker='.'
        
        # Plot on RSI chart (ax3)
        plot_trade_markers(ax3, ax3_twin, 50, long_entries, long_exits, long_iswin, short_entries, short_exits, short_iswin, plot_ids=True)
        
        
    df = df[mask]
    timestamps = df.index.get_level_values('timestamp')
    

    # Prepare data structures for combined plotting
    # scatter_data = defaultdict(lambda: defaultdict(list))
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
                        
                # scatter_color = 'gray' if i != min_touches - 1 else 'red' if end_idx == start_idx else 'blue'
                # scatter_data[scatter_color][mark_shape].append((touch_time, mark_pos))
                
                if i == 0:  # first touch
                    fill_between_data[color].append((x1 + x2, [area.lower_bound] * 4, [area.upper_bound] * 4))
                    line_data['blue_alpha'].append((x1, [area.level] * 2))
                    line_data['blue'].append((x2, [area.level] * 2))

    # for area in tqdm(long_touch_area + short_touch_area, desc='plotting areas'):
    for area in long_touch_area + short_touch_area:
        if not filter_date or area.date == filter_date:
            if trades and area.id in traded_areas:
                process_area(traded_areas[area.id])
            else:
                process_area(area)

    # # Plot combined data on ax1 instead of plt
    # for color, shape_data in scatter_data.items():
    #     for shape, points in shape_data.items():
    #         if points:
    #             x, y = zip(*points)
    #             ax1.scatter(x, y, color=color, s=12, marker=shape)

    for color, data in fill_between_data.items():
        for x, lower, upper in data:
            ax1.fill_between(x[:2], lower[:2], upper[:2], color=color, alpha=0.1)
            ax1.fill_between(x[2:], lower[2:], upper[2:], color=color, alpha=0.25)

    for color, data in line_data.items():
        for x, y in data:
            if color == 'blue_alpha':
                ax1.plot(x, y, color='blue', linestyle='-', alpha=0.20)
            else:
                ax1.plot(x, y, color='blue', linestyle='-', alpha=0.60)

    ax1.set_title(f'{symbol} Price Chart with Touch Detection Areas')
    ax1.set_ylabel('Price')
    ax1.legend().remove()
    # ax1.legend()
    ax1.grid(True)

    # Set x-axis limits for all subplots
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

    ax1.set_xlim(max(zstart, timestamps[0]), min(zend, timestamps[-1]))

    # Set y-axis limits for price plot
    ymin, ymax = 0, -1
    for i in range(len(timestamps)):
        if timestamps[i] >= zstart:
            ymin = i-1
            break
    for i in range(len(timestamps)):
        if timestamps[i] >= zend:
            ymax = i
            break
    highs = df['high'].iloc[max(ymin, 0):min(ymax, len(df))]
    lows = df['low'].iloc[max(ymin, 0):min(ymax, len(df))]
    
    ys_min, ys_max = min(lows), max(highs)
    ys_diff = ys_max - ys_min
    ax1.set_ylim(ys_min - (ys_diff * 0.12), ys_max + (ys_diff * 0.01))
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()