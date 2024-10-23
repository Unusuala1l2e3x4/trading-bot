import os
import zipfile
import pandas as pd
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockQuotesRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment
from datetime import datetime, time, timedelta
from tqdm import tqdm
import time as t2
from typing import List, Tuple, Optional, Dict, Callable
import math
import numpy as np
from TouchDetectionParameters import BacktestTouchDetectionParameters, LiveTouchDetectionParameters

from requests import Session
import hashlib
from numba import jit, prange

from zoneinfo import ZoneInfo

import toml
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



# SLEEP_TIME = 0.3 # 60/200 = 0.3 second if we're using free account
# SLEEP_TIME = 0.06 # 60/1000 = 0.06 sec if we have Elite Smart Router
SLEEP_TIME = 0.006 # 60/10000 = 0.006 sec if we have Algo Trader Plus


import logging
def setup_logger(log_level=logging.INFO):
    logger = logging.getLogger('MultiSymbolDataRetrieval')
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

def fill_missing_data(df: pd.DataFrame):
    """
    Fill missing data in a multi-index DataFrame of stock market data.

    This function processes a DataFrame with a multi-index of (symbol, timestamp),
    filling in missing minutes between the first and last timestamp of each day
    for each symbol. It forward-fills OHLC (Open, High, Low, Close) values and
    fills volume and trade count with zeros.

    Parameters:
    df (pandas.DataFrame): A multi-index DataFrame with levels (symbol, timestamp)
                           containing stock market data.

    Returns:
    pandas.DataFrame: A DataFrame with missing data filled, maintaining the
                      original multi-index structure and timezone.
    """
    if df.empty:
        return df
    
    # Ensure the index is sorted
    df = df.sort_index()

    # Get the timezone
    tz = df.index.get_level_values('timestamp').tz

    # Group by symbol and date
    grouped = df.groupby([df.index.get_level_values('symbol'),
                          df.index.get_level_values('timestamp').date])

    filled_dfs = []

    for (symbol, date), group in grouped:
        # Get min and max timestamps for the current date and symbol
        min_time = group.index.get_level_values('timestamp').min()
        max_time = group.index.get_level_values('timestamp').max()

        # Create a complete range of timestamps for this date
        full_idx = pd.date_range(start=min_time, end=max_time, freq='min', tz=tz)

        # Reindex the group
        filled_group = group.reindex(pd.MultiIndex.from_product([[symbol], full_idx],
                                                                names=['symbol', 'timestamp']))

        # Forward-fill OHLC values
        filled_group[['open', 'high', 'low', 'close']] = filled_group[['open', 'high', 'low', 'close']].ffill()

        # Fill volume and trade_count with 0
        filled_group['volume'] = filled_group['volume'].fillna(0)
        filled_group['trade_count'] = filled_group['trade_count'].fillna(0)
        
        # VWAP remains NaN where missing (may need to be calculated later if new functionality requires it)

        filled_dfs.append(filled_group)

    # Concatenate all filled groups
    result = pd.concat(filled_dfs)

    return result


def retrieve_bar_data(client: StockHistoricalDataClient, params: BacktestTouchDetectionParameters, symbol: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """_summary_

    Args:
        client (StockHistoricalDataClient): 
        params (BacktestTouchDetectionParameters): 
        symbol (str | None): If None, use params.symbol

    Returns:
        (pd.DataFrame, pd.DataFrame): adjusted bars for graphing, unadjusted bars for backtesting
    """
    if symbol is None:
        symbol = params.symbol
    
    df, df_unadjusted = None, None
    
    if isinstance(params.start_date, str):
        params.start_date = pd.to_datetime(params.start_date).tz_localize(ny_tz)
    if isinstance(params.end_date, str):
        params.end_date = pd.to_datetime(params.end_date).tz_localize(ny_tz)

    if params.export_bars_path:
        directory = os.path.dirname(params.export_bars_path)
        bars_zip_path = os.path.join(directory, f'bars_{symbol}_{params.start_date.strftime("%Y-%m-%d")}_{params.end_date.strftime("%Y-%m-%d")}.zip')
        assert directory == os.path.dirname(bars_zip_path)
        os.makedirs(directory, exist_ok=True)
            
        adjusted_csv_name = os.path.basename(bars_zip_path).replace('.zip', '.csv')
        unadjusted_csv_name = os.path.basename(bars_zip_path).replace('.zip', '_unadjusted.csv')
        
        if os.path.isfile(bars_zip_path):
            with zipfile.ZipFile(bars_zip_path, 'r') as zip_file:
                file_list = zip_file.namelist()
                if adjusted_csv_name in file_list:
                    try:
                        with zip_file.open(adjusted_csv_name) as csv_file:
                            df = pd.read_csv(csv_file)
                            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_convert(ny_tz)
                            df.set_index(['symbol', 'timestamp'], inplace=True)
                            log(f'Retrieved {adjusted_csv_name} from {bars_zip_path}')
                    except Exception as e:
                        log(f"Error retrieving adjusted bars from file for {symbol}: {str(e)}", level=logging.ERROR)
                        df = None
                if unadjusted_csv_name in file_list:
                    try:
                        with zip_file.open(unadjusted_csv_name) as csv_file:
                            df_unadjusted = pd.read_csv(csv_file)
                            df_unadjusted['timestamp'] = pd.to_datetime(df_unadjusted['timestamp']).dt.tz_convert(ny_tz)
                            df_unadjusted.set_index(['symbol', 'timestamp'], inplace=True)
                            log(f'Retrieved {unadjusted_csv_name} from {bars_zip_path}')
                    except Exception as e:
                        log(f"Error retrieving unadjusted bars from file for {symbol}: {str(e)}", level=logging.ERROR)
                        df_unadjusted = None
                    
                    
    def fetch_bars(adjustment):
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=params.start_date.tz_convert('UTC'),
            end=params.end_date.tz_convert('UTC'),
            adjustment=adjustment,
        )
        try:
            df = get_data_with_retry(client, client.get_stock_bars, request_params)
        except Exception as e:
            log(f"Error requesting bars for {symbol}: {str(e)}", level=logging.ERROR)
            return pd.DataFrame()
        df.index = df.index.set_levels(df.index.get_level_values('timestamp').tz_convert(ny_tz), level='timestamp')
        df.sort_index(inplace=True)
        return fill_missing_data(df)

    if df is None:
        df = fetch_bars(Adjustment.ALL)
        
    if df_unadjusted is None:
        df_unadjusted = fetch_bars(Adjustment.RAW)

    # Only create or update the zip file if new data was fetched
    if params.export_bars_path and (df is not None or df_unadjusted is not None):
        mode = 'a' if os.path.isfile(bars_zip_path) else 'w'
        with zipfile.ZipFile(bars_zip_path, mode, compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
            for df_name, csv_name in [('df', adjusted_csv_name), 
                                    ('df_unadjusted', unadjusted_csv_name)]:
                # print('bars save',locals()[df_name].index.get_level_values('timestamp'))
                if locals()[df_name] is not None and csv_name not in zip_file.namelist():
                    with zip_file.open(csv_name, 'w') as csv_file:
                        locals()[df_name].reset_index().to_csv(csv_file, index=False)
                    log(f'Saved {df_name} to {bars_zip_path}')

    return df, df_unadjusted


drop_cols = ['bid_exchange', 'ask_exchange', 'conditions', 'tape']

def clean_quotes_data(df: pd.DataFrame, interval_start: pd.Timestamp, interval_end: pd.Timestamp) -> Tuple[pd.DataFrame, float]:
    """
    Clean quotes data and calculate intra-timestamp changes with improved efficiency.
    """
    # assert only 1 distinct symbol
    assert len(pd.unique(df.index.get_level_values('symbol'))) == 1, list(pd.unique(df.index.get_level_values('symbol')))
    
    if df.empty:
        return df, (interval_end - interval_start).total_seconds()

    # Drop unnecessary columns, sort index
    df = df.drop(columns=drop_cols, errors='ignore').sort_index(level=['symbol','timestamp'])
    
    # Efficient calculation of intra-timestamp changes
    for col in ['bid_price', 'ask_price', 'bid_size', 'ask_size']:
        df[f'{col}_intra_change'] = df.groupby(level='timestamp')[col].diff()
        df[f'{col}_intra_pos'] = df[f'{col}_intra_change'].clip(lower=0)
        df[f'{col}_intra_neg'] = df[f'{col}_intra_change'].clip(upper=0)

    # Optimized aggregation
    agg_dict = {
        'bid_price': ['first', 'max', 'min', 'last'],
        'bid_price_intra_pos': 'sum',
        'bid_price_intra_neg': 'sum',
        'ask_price': ['first', 'max', 'min', 'last'],
        'ask_price_intra_pos': 'sum',
        'ask_price_intra_neg': 'sum',
        'bid_size': ['first', 'max', 'min', 'last'],
        'bid_size_intra_pos': 'sum',
        'bid_size_intra_neg': 'sum',
        'ask_size': ['first', 'max', 'min', 'last'],
        'ask_size_intra_pos': 'sum',
        'ask_size_intra_neg': 'sum'
    }
    
    df = df.groupby(level=['symbol', 'timestamp']).agg(agg_dict)
    df.columns = ['_'.join(col).strip() for col in df.columns.values]

    # Handle timezone and sorting
    timestamps = df.index.get_level_values('timestamp')
    df.index = df.index.set_levels(
        timestamps.tz_localize(ny_tz) if timestamps.tz is None else timestamps.tz_convert(ny_tz),
        level='timestamp'
    )
    # df.sort_index(level=['symbol', 'timestamp'], inplace=True) # already sorted
    
    # print(list(df.columns))
    # print(df.loc[:, [a for a in df.columns if a.endswith('_sum')]])
    
    # sec = pd.to_datetime(df.index.get_level_values('timestamp')).map(lambda x: x.microsecond / 1000)
    # df.insert(loc=0,column='microsec',value=sec)
    # sec = pd.to_datetime(df.index.get_level_values('timestamp')).map(lambda x: x.second + x.microsecond / 1_000_000)
    # df.insert(loc=0,column='sec',value=sec)
    # # print(df)
    
    # Calculate durations
    timestamps = df.index.get_level_values('timestamp').to_numpy()
    duration_arr = np.empty(len(timestamps))
    duration_arr[:-1] = np.diff(timestamps).astype('timedelta64[ns]').astype(np.float64) * 1e-9
    duration_arr[-1] = (interval_end - timestamps[-1]).total_seconds()
    df['duration'] = duration_arr
    
    # Validate no zero durations exist
    assert not np.any(duration_arr[:-1] <= 0), ("Zero or negative durations found after processing\n",df.loc[df['duration'] <= 0])
    
    assert interval_start <= timestamps[0], (interval_start, timestamps[0])
    carryover = (timestamps[0] - interval_start).total_seconds()
    
    # print(carryover)
    # print(df['duration'])
    # df.reset_index(drop=False).to_csv('test.csv',index=False) # csv test
    
    return df, carryover


SEED_DIV = 2**32
def get_seed(symbol: str, minute: datetime) -> int:
    return int(hashlib.sha256(f"{symbol}_{minute}".encode()).hexdigest(), 16) % SEED_DIV

# parallelization doesnt seem to speed it up. using parallel=False for now.
@jit(nopython=True, parallel=False)
def compute_weighted_sizes(group_indices, group_counts, bid_sizes, ask_sizes, durations):
    n = len(group_indices)
    result_bid_sizes = np.empty(n, dtype=np.float64)
    result_ask_sizes = np.empty(n, dtype=np.float64)
    for i in prange(n):
        start = group_indices[i]
        end = start + group_counts[i]
        group_durations = durations[start:end]
        group_bid_sizes = bid_sizes[start:end]
        group_ask_sizes = ask_sizes[start:end]
        
        total_duration = np.sum(group_durations)
        if total_duration == 0:
            result_bid_sizes[i] = group_bid_sizes[0]
            result_ask_sizes[i] = group_ask_sizes[0]
        else:
            result_bid_sizes[i] = np.sum(group_bid_sizes * group_durations) / total_duration
            result_ask_sizes[i] = np.sum(group_ask_sizes * group_durations) / total_duration
    return result_bid_sizes, result_ask_sizes

def apply_grouping_and_weighting(df: pd.DataFrame):
    pairs = df['bid_price_last'].astype(str) + '_' + df['ask_price_last'].astype(str)
    
    # Identify groups
    group_changes = (pairs != pairs.shift()).cumsum().values
    _, group_indices, group_counts = np.unique(group_changes, return_index=True, return_counts=True)
    
    # Prepare arrays
    symbols = df.index.get_level_values('symbol').values
    timestamps = df.index.get_level_values('timestamp').values
    bid_prices = df['bid_price_last'].values
    ask_prices = df['ask_price_last'].values
    bid_sizes = df['bid_size_last'].values
    ask_sizes = df['ask_size_last'].values
    durations = df['duration'].values
    
    # Calculate results
    result_bid_prices = bid_prices[group_indices]
    result_ask_prices = ask_prices[group_indices]
    result_durations = np.add.reduceat(durations, group_indices)
    
    # Compute weighted averages using Numba
    result_bid_sizes, result_ask_sizes = compute_weighted_sizes(
        group_indices, group_counts, bid_sizes, ask_sizes, durations
    )
    
    # Create result DataFrame
    result = pd.DataFrame({
        'bid_price': result_bid_prices,
        'ask_price': result_ask_prices,
        'spread': result_ask_prices - result_bid_prices,
        'bid_size': result_bid_sizes,
        'ask_size': result_ask_sizes,
        'duration': result_durations
    }, index=pd.MultiIndex.from_arrays([
        symbols[group_indices], 
        pd.to_datetime(timestamps[group_indices], utc=True).tz_convert(ny_tz)
    ], names=['symbol', 'timestamp']))

    return result


@jit(nopython=True)
def compute_weighted_sizes_single(prices, sizes, durations) -> np.ndarray:
    n = len(prices)
    result_sizes = np.empty(n, dtype=np.float64)
    current_price = prices[0]
    start_idx = 0
    
    for i in range(1, n + 1):
        if i == n or prices[i] != current_price:
            end_idx = i
            group_durations = durations[start_idx:end_idx]
            group_sizes = sizes[start_idx:end_idx]
            total_duration = np.sum(group_durations)
            if total_duration == 0:
                weighted_size = group_sizes[0]
            else:
                weighted_size = np.sum(group_sizes * group_durations) / total_duration
            
            result_sizes[start_idx:end_idx] = weighted_size
            
            if i < n:
                current_price = prices[i]
                start_idx = i
    return result_sizes

@jit(nopython=True)
def aggregate_data(prices: np.ndarray, sizes: np.ndarray, durations: np.ndarray, unique_timestamps: np.ndarray, indices: np.ndarray):
    n = len(prices)
    m = len(unique_timestamps)
    max_prices = np.empty(m, dtype=prices.dtype)
    min_prices = np.empty(m, dtype=prices.dtype)
    wm_prices = np.empty(m, dtype=prices.dtype)
    max_sizes = np.empty(m, dtype=sizes.dtype)
    sum_sizes = np.empty(m, dtype=sizes.dtype)
    
    for i in range(m):
        start = indices[i]
        end = indices[i+1] if i < m-1 else n
        
        group_prices = prices[start:end]
        group_sizes = sizes[start:end]
        group_durations = durations[start:end]
        
        max_prices[i] = np.max(group_prices)
        min_prices[i] = np.min(group_prices)
        max_sizes[i] = np.max(group_sizes)
        sum_sizes[i] = np.sum(group_sizes)
        
        total_duration = np.sum(group_durations)
        if total_duration == 0:
            wm_prices[i] = group_prices[0]
        else:
            wm_prices[i] = np.sum(group_prices * group_durations) / total_duration
    
    return max_prices, min_prices, wm_prices, max_sizes, sum_sizes

@jit(nopython=True)
def aggregate_spread_data(spreads, durations, unique_timestamps, indices):
    n = len(spreads)
    m = len(unique_timestamps)
    min_spreads = np.empty(m, dtype=spreads.dtype)
    max_spreads = np.empty(m, dtype=spreads.dtype)
    weighted_spreads = np.empty(m, dtype=spreads.dtype)
    for i in range(m):
        start = indices[i]
        end = indices[i+1] if i < m-1 else n
        group_spreads = spreads[start:end]
        group_durations = durations[start:end]
        total_duration = np.sum(group_durations)
        
        min_spreads[i] = np.min(group_spreads)
        max_spreads[i] = np.max(group_spreads)
        
        if total_duration == 0:
            weighted_spreads[i] = group_spreads[0]
        else:
            weighted_spreads[i] = np.sum(group_spreads * group_durations) / total_duration
    return min_spreads, max_spreads, weighted_spreads

def aggregate_quotes_time_based(df: pd.DataFrame, interval_seconds=1):
    timestamps = df.index.get_level_values('timestamp').values.astype(np.int64) // 10**9
    symbols = df.index.get_level_values('symbol').values
    bid_prices = df['bid_price'].values
    ask_prices = df['ask_price'].values
    bid_sizes = df['bid_size'].values
    ask_sizes = df['ask_size'].values
    durations = df['duration'].values
    spreads = df['spread'].values
    
    floored_timestamps = (timestamps // interval_seconds) * interval_seconds
    unique_timestamps, indices = np.unique(floored_timestamps, return_index=True)
    
    # Aggregate spread data
    min_spreads, max_spreads, weighted_spreads = aggregate_spread_data(spreads, durations, unique_timestamps, indices)
    
    # Re-weight bid and ask sizes
    bid_sizes = compute_weighted_sizes_single(bid_prices, bid_sizes, durations)
    ask_sizes = compute_weighted_sizes_single(ask_prices, ask_sizes, durations)

    bid_max, _, bid_wm, bid_size_max, bid_size_sum = aggregate_data(bid_prices, bid_sizes, durations, unique_timestamps, indices)
    _, ask_min, ask_wm, ask_size_max, ask_size_sum = aggregate_data(ask_prices, ask_sizes, durations, unique_timestamps, indices)
    
    # Combine results
    result = pd.DataFrame({
        'max_bid_price': bid_max,
        'min_ask_price': ask_min,
        'wm_bid_price': bid_wm,
        'wm_ask_price': ask_wm,
        'max_bid_size': np.floor(bid_size_max),
        'max_ask_size': np.floor(ask_size_max),
        'total_bid_size': np.floor(bid_size_sum),
        'total_ask_size': np.floor(ask_size_sum),
        'min_spread': min_spreads,
        'max_spread': max_spreads,
        'wm_spread': weighted_spreads
    }, index=pd.MultiIndex.from_arrays([
        symbols[indices], 
        pd.to_datetime(unique_timestamps, unit='s', utc=True).tz_convert(ny_tz)
    ], names=['symbol', 'timestamp']))
    return result

def get_quotes_zip_path(symbol, params: BacktestTouchDetectionParameters):
   directory = os.path.dirname(params.export_quotes_path)
   return os.path.join(directory, f'quotes_{symbol}_{params.start_date.strftime("%Y-%m-%d")}_{params.end_date.strftime("%Y-%m-%d")}.zip')


from concurrent.futures import ThreadPoolExecutor

def fetch_next_minute_data(client: StockHistoricalDataClient, symbols: List[str], minute_start: datetime, minute_end: datetime):
    request_params = StockQuotesRequest(
        symbol_or_symbols=symbols,
        start=minute_start.tz_convert('UTC'),
        end=minute_end.tz_convert('UTC'),
    )
    try:
        return get_data_with_retry(client, client.get_stock_quotes, request_params)
    except Exception as e:
        log(f"Error retrieving quotes for {symbols} at {minute_start}: {str(e)}", level=logging.ERROR)
        return pd.DataFrame()

def retrieve_quote_data(client: StockHistoricalDataClient, symbols: List[str], minute_intervals_dict: Dict[str, pd.Index], params: BacktestTouchDetectionParameters, 
                        first_seconds_sample: int = np.inf, last_seconds_sample: int = np.inf, return_data=True) -> Dict | None:
    quotes_data = {symbol: {'raw': None, 'agg': None} for symbol in symbols}
    
    # Try retrieving saved data first
    ignore_symbols = set()
    for symbol in symbols:
        if params.export_quotes_path:
            quotes_zip_path = get_quotes_zip_path(symbol, params)
            os.makedirs(os.path.dirname(quotes_zip_path), exist_ok=True)
            
            if os.path.isfile(quotes_zip_path):
                try:
                    with zipfile.ZipFile(quotes_zip_path, 'r') as zip_file:
                        with zip_file.open(f'{symbol}_raw_quotes.csv') as csv_file:
                            raw_df = pd.read_csv(csv_file)
                            raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp']).dt.tz_convert(ny_tz)
                            raw_df.set_index(['symbol', 'timestamp'], inplace=True)
                        with zip_file.open(f'{symbol}_aggregated_quotes.csv') as csv_file:
                            aggregated_df = pd.read_csv(csv_file)
                            aggregated_df['timestamp'] = pd.to_datetime(aggregated_df['timestamp']).dt.tz_convert(ny_tz)
                            aggregated_df.set_index(['symbol', 'timestamp'], inplace=True)
                    log(f'Retrieved quotes for {symbol} from {quotes_zip_path}')
                    quotes_data[symbol]['raw'] = raw_df
                    quotes_data[symbol]['agg'] = aggregated_df
                    ignore_symbols.add(symbol)
                except Exception as e:
                    log(f"Error retrieving quotes from file for {symbol}: {str(e)}", level=logging.ERROR)
                    
    symbols = [a for a in symbols if a not in ignore_symbols]
    if len(symbols) == 0:
        return quotes_data
    
    def filter_first_and_last_seconds(symbol_df: pd.DataFrame) -> pd.DataFrame:
        seconds = symbol_df.index.get_level_values('timestamp').second
        first_end_idx = np.searchsorted(seconds, 1, side='left')
        last_start_idx = np.searchsorted(seconds, 58, side='left')
        first_seconds_data = symbol_df.iloc[:first_end_idx]
        last_seconds_data = symbol_df.iloc[last_start_idx:] if last_start_idx < len(symbol_df) else pd.DataFrame()
        
        # Ensure last_seconds_data has at least 1 row
        if last_seconds_data.empty:
            last_row = symbol_df.iloc[[-1]]
            if not first_seconds_data.empty and last_row.index[-1] == first_seconds_data.index[-1]:
                last_seconds_data = pd.DataFrame()  # Make it empty if last row is in first_seconds_data
            else:
                last_seconds_data = last_row  # Otherwise, use the last row
        
        if len(first_seconds_data) > first_seconds_sample or len(last_seconds_data) > last_seconds_sample:
            rs = np.random.RandomState(seeds[symbol])
            if len(first_seconds_data) > first_seconds_sample:
                first_indices = rs.choice(first_seconds_data.index, first_seconds_sample, replace=False)
                first_seconds_data = first_seconds_data.loc[first_indices]
            if len(last_seconds_data) > last_seconds_sample:
                last_indices = rs.choice(last_seconds_data.index, last_seconds_sample, replace=False)
                last_seconds_data = last_seconds_data.loc[last_indices]
        return pd.concat([first_seconds_data, last_seconds_data])

    # Initialize temporary files for each symbol
    temp_files = {}
    for symbol in symbols:
        if params.export_quotes_path:
            directory = os.path.dirname(params.export_quotes_path)
            temp_files[symbol] = {
                'raw': os.path.join(directory, f'temp_{symbol}_raw_quotes.csv'),
                'agg': os.path.join(directory, f'temp_{symbol}_aggregated_quotes.csv')
            }

    def append_quote_segment(temp_path, df: pd.DataFrame, is_first: bool, index_format=True, max_retries=240, sleep_seconds=0.3):
        if temp_path is not None:
            if index_format:
                orig_index = df.index.copy()
                df.index = pd.MultiIndex.from_arrays([df.index.get_level_values('symbol'), 
                                df.index.get_level_values('timestamp').strftime('%Y-%m-%d %H:%M:%S.%f%z').str.replace(r'(\d{2})(\d{2})$', r'\1:\2', regex=True)], 
                                names=['symbol', 'timestamp'])
            attempt = 0
            while True:
                try:
                    # Write data, with headers only on first append
                    df.to_csv(temp_path, mode='w' if is_first else 'a', header=is_first)
                except Exception as e:
                    attempt += 1
                    if attempt == 1:
                        log(f"Attempt {attempt}: Error appending to file for {symbol} at {current_minute}: {str(e)}\nRetrying...", level=logging.ERROR)
                    if attempt >= max_retries:
                        log(f"Attempt {attempt}: Error appending to file for {symbol} at {current_minute}: {str(e)}", level=logging.ERROR)
                        log(f"Max retries ({max_retries}) reached. Giving up.", level=logging.ERROR)
                        break
                        # raise  # Re-raise the last exception
                    t2.sleep(sleep_seconds)
                    continue
                if attempt > 1:
                    log(f"Successful after {attempt} retries", level=logging.DEBUG)
                break
            
            if index_format:
                df.index = orig_index
    
    last_minute_data = {symbol: None for symbol in symbols}
    acc_carryover = {symbol: 0 for symbol in symbols}
    first_append = {symbol: True for symbol in symbols}
    
    # Get iterator for minutes
    minutes = minute_intervals_dict[symbols[0]]
    minutes_iter = iter(minutes)
    
    # Start first request
    try:
        current_minute = next(minutes_iter)
        current_minute_start = current_minute
        current_minute_end = current_minute + timedelta(minutes=1)
    except StopIteration:
        return quotes_data
        
    with ThreadPoolExecutor(max_workers=1) as executor, tqdm(total=len(minutes), desc='Fetching quotes') as pbar:
        # Start first request
        future = executor.submit(fetch_next_minute_data, client, symbols, current_minute_start, current_minute_end)
        
        while True:
            # Get current minute's data
            qdf0 = future.result()
            
            # Start next request if there are more minutes
            try:
                next_minute = next(minutes_iter)
                next_minute_start = next_minute
                next_minute_end = next_minute + timedelta(minutes=1)
                future = executor.submit(fetch_next_minute_data, client, symbols, next_minute_start, next_minute_end)
            except StopIteration:
                future = None
            
            # Process current minute's data
            seeds = {symbol: get_seed(symbol, current_minute) for symbol in symbols}
            
            for symbol in symbols:
                try:
                    if not qdf0.empty and symbol in qdf0.index.get_level_values('symbol'):
                        symbol_df = qdf0.xs(symbol, level='symbol', drop_level=False)
                    else:
                        symbol_df = pd.DataFrame()
                        log(f"No data found for symbol {symbol} at {current_minute}.", level=logging.WARNING)
                except Exception as e:
                    symbol_df = pd.DataFrame()
                    log(f"Error processing data for symbol {symbol} at {current_minute}: {str(e)}", level=logging.ERROR)

                symbol_df, new_carryover = clean_quotes_data(symbol_df, current_minute_start, current_minute_end)

                if not symbol_df.empty:
                    # Process the previous minute's data
                    if last_minute_data[symbol] is not None:
                        last_minute_df = last_minute_data[symbol]
                        if not last_minute_df.empty:
                            last_index = last_minute_df.index[-1]
                            last_minute_df.loc[last_index, 'duration'] += acc_carryover[symbol] + new_carryover

                        # weighted_data = apply_grouping_and_weighting(last_minute_df)
                        # filtered_data = filter_first_and_last_seconds(weighted_data)
                        # filtered_data[['bid_size','ask_size']] = filtered_data[['bid_size','ask_size']].apply(np.floor)
                        # filtered_data = filtered_data.drop(columns=['spread','duration'])
                        filtered_data = filter_first_and_last_seconds(last_minute_df)
                        
                        weighted_data = apply_grouping_and_weighting(last_minute_df)
                        aggregated_data = aggregate_quotes_time_based(weighted_data)
                        
                        if params.export_quotes_path:
                            append_quote_segment(temp_files[symbol]['raw'], filtered_data.drop(columns=['duration']), is_first=first_append[symbol], index_format=True)
                            append_quote_segment(temp_files[symbol]['agg'], aggregated_data, is_first=first_append[symbol], index_format=False)
                            if first_append[symbol]:
                                first_append[symbol] = False
                            
                    acc_carryover[symbol] = 0
                    last_minute_data[symbol] = symbol_df
                    
                else:
                    acc_carryover[symbol] += new_carryover
            # Update progress bar
            pbar.update(1)
            if future is None:  # No more minutes to process
                break
                
            current_minute = next_minute
            current_minute_start = next_minute_start
            current_minute_end = next_minute_end

    # Process the last minute's data
    for symbol in symbols:
        if last_minute_data[symbol] is not None:
            last_minute_df = last_minute_data[symbol]
            if not last_minute_df.empty:
                last_index = last_minute_df.index[-1]
                last_minute_df.loc[last_index, 'duration'] += acc_carryover[symbol]
            
            # weighted_data = apply_grouping_and_weighting(last_minute_df)
            # filtered_data = filter_first_and_last_seconds(weighted_data)
            # filtered_data[['bid_size','ask_size']] = filtered_data[['bid_size','ask_size']].apply(np.floor)
            # filtered_data = filtered_data.drop(columns=['spread','duration'])
            filtered_data = filter_first_and_last_seconds(last_minute_df)
            
            weighted_data = apply_grouping_and_weighting(last_minute_df)
            aggregated_data = aggregate_quotes_time_based(weighted_data)
            
            if params.export_quotes_path:
                append_quote_segment(temp_files[symbol]['raw'], filtered_data.drop(columns=['duration']), is_first=first_append[symbol], index_format=True)
                append_quote_segment(temp_files[symbol]['agg'], aggregated_data, is_first=first_append[symbol], index_format=False)
                if first_append[symbol]:
                    first_append[symbol] = False
                  
    # Create final zip files from temp files and load data for return
    for symbol in symbols:
        if params.export_quotes_path:
            quotes_zip_path = get_quotes_zip_path(symbol, params)
            with zipfile.ZipFile(quotes_zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
                for suffix in ['raw_quotes', 'aggregated_quotes']:
                    temp_path = temp_files[symbol]['raw' if 'raw' in suffix else 'agg']
                    with open(temp_path, 'r') as temp_file:
                        with zip_file.open(f'{symbol}_{suffix}.csv', 'w') as zip_csv:
                            zip_csv.write(temp_file.read().encode())
                    os.remove(temp_path)  # Clean up temp file

            if not return_data:
                continue
            # Read data for return
            try:
                with zipfile.ZipFile(quotes_zip_path, 'r') as zip_file:
                    with zip_file.open(f'{symbol}_raw_quotes.csv') as csv_file:
                        raw_df = pd.read_csv(csv_file)
                        # print(pd.to_datetime(raw_df['timestamp']))
                        raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp']).dt.tz_convert(ny_tz)
                        raw_df.set_index(['symbol', 'timestamp'], inplace=True)
                        # print(raw_df.index.get_level_values('timestamp'))
                    with zip_file.open(f'{symbol}_aggregated_quotes.csv') as csv_file:
                        aggregated_df = pd.read_csv(csv_file)
                        # print(pd.to_datetime(aggregated_df['timestamp']))
                        aggregated_df['timestamp'] = pd.to_datetime(aggregated_df['timestamp']).dt.tz_convert(ny_tz)
                        aggregated_df.set_index(['symbol', 'timestamp'], inplace=True)
                        # print(aggregated_df.index.get_level_values('timestamp'))
                quotes_data[symbol]['raw'] = raw_df
                quotes_data[symbol]['agg'] = aggregated_df
            except Exception as e:
                log(f"Error reading saved quotes for {symbol}: {str(e)}", level=logging.ERROR)
    if return_data:
        return quotes_data
    else:
        return None

def refresh_client(client):
    """
    Refresh the session of a StockHistoricalDataClient instance.
    
    Args:
        client: An instance of StockHistoricalDataClient
    """
    if not isinstance(client, StockHistoricalDataClient):
        raise TypeError("Client must be an instance of StockHistoricalDataClient")

    # Refresh the session
    if hasattr(client, '_session'):
        client._session.close()
        client._session = Session()
        log(f"Session refreshed for {client.__class__.__name__}", level=logging.WARNING)
        
def get_data_with_retry(client: StockHistoricalDataClient, client_func: Callable, request_params: StockBarsRequest, max_retries=10, sleep_seconds=10) -> pd.DataFrame:
    attempt = 0
    while True:
        try:
            res = client_func(request_params)
            return res.df  # Successfully retrieved data, exit the loop
        except Exception as e:
            attempt += 1
            symbols = request_params.symbol_or_symbols
            minute = request_params.start  # Assuming 'start' is the relevant timestamp
            
            log(f"Attempt {attempt}: Error retrieving data for {symbols} at {minute}: {str(e)}", level=logging.ERROR)
            
            if attempt >= max_retries:
                log(f"Attempt {attempt}: Error retrieving data for {symbols} at {minute}: {str(e)}", level=logging.ERROR)
                log(f"Max retries reached. Giving up.", level=logging.ERROR)
                return pd.DataFrame()
                # raise e  # Re-raise the last exception
            log(f"Refreshing client session and retrying in {sleep_seconds} seconds...", level=logging.ERROR)
            refresh_client(client)
            t2.sleep(sleep_seconds)
            
            

def retrieve_multi_symbol_data(params: BacktestTouchDetectionParameters, symbols: List[str], first_seconds_sample: int = np.inf, 
                               last_seconds_sample: int = np.inf, return_data=True) -> Dict | None:
    assert params.end_date > params.start_date
    
    if isinstance(params.start_date, str):
        params.start_date = pd.to_datetime(params.start_date).tz_localize(ny_tz)
    if isinstance(params.end_date, str):
        params.end_date = pd.to_datetime(params.end_date).tz_localize(ny_tz)

    client = StockHistoricalDataClient(api_key=API_KEY, secret_key=API_SECRET)

    # Retrieve bar data and build minute_intervals_dict
    minute_intervals_dict = {}
    for symbol in tqdm(symbols, desc="Retrieving bar data"):
        # start_time = t2.time()
        df_adjusted, df = retrieve_bar_data(client,params,symbol)
        # print(df)
        # print(adjustment_factors)

        
        minute_intervals = df.index.get_level_values('timestamp')
        minute_intervals = minute_intervals[(minute_intervals.time >= time(9, 30)) & (minute_intervals.time < time(16, 0))]
        minute_intervals_dict[symbol] = minute_intervals
        
        # elapsed_time = t2.time() - start_time
        # remaining_sleep_time = max(0, SLEEP_TIME - elapsed_time)
        # t2.sleep(remaining_sleep_time)
        # t2.sleep(SLEEP_TIME)
        
    assert len(set([a.size for a in minute_intervals_dict.values()])) <= 1 # make sure len(minute_intervals) are the same

    # Retrieve quote data
    quotes_data = retrieve_quote_data(client, symbols, minute_intervals_dict, params, first_seconds_sample, last_seconds_sample, return_data)

    log("Data retrieval complete for all symbols.")
    return quotes_data


if __name__=="__main__":    
    start_date = "2024-07-01 00:00:00"
    end_date =   "2024-08-01 00:00:00"

    # start_date = "2024-08-19 00:00:00"
    # end_date =   "2024-08-20 00:00:00"
    start_date = "2024-08-20 00:00:00"
    end_date =   "2024-08-21 00:00:00"
    
    # start_date = "2024-08-01 00:00:00"
    # end_date =   "2024-09-01 00:00:00"

    start_date = "2024-09-01 00:00:00"
    end_date =   "2024-10-01 00:00:00"
    
    start_date = "2022-01-01 00:00:00"
    end_date =   "2022-02-01 00:00:00"
    
    start_date = "2022-01-12 00:00:00"
    end_date =   "2022-01-13 00:00:00"
    
    # start_date = "2024-09-03 00:00:00"
    # end_date =   "2024-09-04 00:00:00"

    # Usage example (most params are just placeholders for this module):
    params = BacktestTouchDetectionParameters(
        symbol='',
        start_date=start_date,
        end_date=end_date,
        atr_period=15,
        level1_period=15,
        multiplier=1.4,
        min_touches=3,
        start_time=None,
        end_time='15:55',
        use_median=True,
        touch_area_width_agg=None,
        rolling_avg_decay_rate=0.85,
        export_bars_path=f'bars/',
        export_quotes_path=f'quotes/'
    )


    # symbols = ['AAPL', 'GOOGL', 'NVDA']
    symbols = ['AAPL'] 
    # symbols = ['GOOGL']
    # symbols = ['NVDA'] 
    # symbols = ['AAPL','GOOGL'] 
    # NOTE: seems faster overall to do one at a time: for one day, 1 took ~1 minute, 3 took ~5 minutes
    
    # quotes_data = retrieve_multi_symbol_data(params, symbols, first_seconds_sample=100, last_seconds_sample=200)
    quotes_data = retrieve_multi_symbol_data(params, symbols)


    # symbols = ['AAPL'] 
    # quotes_data = retrieve_multi_symbol_data(params, symbols)
    
    
    # start_date = "2024-09-01 00:00:00"
    # end_date =   "2024-10-01 00:00:00"
    
    # params.start_date = start_date
    # params.end_date = end_date
    
    # retrieve_multi_symbol_data(params, symbols, first_seconds_sample=100, last_seconds_sample=200)
    
    
    # print(quotes_data[symbols[0]]['raw'].index)
    # print(quotes_data[symbols[0]]['agg'].index)