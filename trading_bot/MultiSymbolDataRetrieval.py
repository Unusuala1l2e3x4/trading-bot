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
from TouchDetectionParameters import BacktestTouchDetectionParameters

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


# Global tqdm object
progress_bar = None

def get_segment_sizes(dfs: List[pd.DataFrame]) -> List[int]:
    """
    Calculate the size of each DataFrame in the list.
    
    :param dfs: List of DataFrames
    :return: List of sizes (number of rows) for each DataFrame
    """
    return [df.shape[0] for df in dfs]

def divide_segments(dfs: List[pd.DataFrame], sizes: List[int], num_groups: int) -> List[List[int]]:
    """
    Divide the segments into groups for concatenation.
    
    :param dfs: List of DataFrames
    :param sizes: List of sizes for each DataFrame
    :param num_groups: Number of groups to divide into
    :return: List of lists containing indices for each group
    """
    total_size = sum(sizes)
    target_size = total_size / num_groups if num_groups > 0 else 0
    
    groups = [[] for _ in range(num_groups)]
    group_sizes = [0] * num_groups
    current_group = 0
    
    for i, size in enumerate(sizes):
        if current_group < num_groups - 1 and group_sizes[current_group] + size > target_size:
            current_group += 1
        groups[current_group].append(i)
        group_sizes[current_group] += size
    
    return [group for group in groups if group]

def custom_concat(dfs: List[pd.DataFrame], group_size: int = 10, total_rows: int = None) -> pd.DataFrame:
    global progress_bar
    
    # Initialize progress bar if it's the first call
    if progress_bar is None and total_rows is not None:
        progress_bar = tqdm(total=total_rows, desc="Concatenating")
    
    # Handle edge cases
    if len(dfs) == 0:
        return pd.DataFrame()
    elif len(dfs) == 1:
        if progress_bar is not None:
            progress_bar.update(dfs[0].shape[0])
        return dfs[0]
    elif len(dfs) <= group_size:
        result = pd.concat(dfs, axis=0)
        if progress_bar is not None:
            progress_bar.update(result.shape[0])
        return result
    
    sizes = get_segment_sizes(dfs)
    num_groups = math.ceil(len(dfs) / group_size)
    groups = divide_segments(dfs, sizes, num_groups)
    
    concatenated_groups = []
    for group in groups:
        group_dfs = [dfs[i] for i in group]
        if group_dfs:
            result = pd.concat(group_dfs, axis=0)
            concatenated_groups.append(result)
            if progress_bar is not None:
                progress_bar.update(result.shape[0])
    
    if not concatenated_groups:
        return pd.DataFrame()
    elif len(concatenated_groups) == 1:
        return concatenated_groups[0]
    else:
        return custom_concat(concatenated_groups, group_size)

def concat_with_progress(dfs: List[pd.DataFrame], group_size: int = 10) -> pd.DataFrame:
    global progress_bar
    total_rows = sum(df.shape[0] for df in dfs)
    try:
        return custom_concat(dfs, group_size, total_rows)
    finally:
        if progress_bar is not None:
            progress_bar.close()
        progress_bar = None


def fill_missing_data(df):
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


# Define aggregation functions
clean_quotes_data_agg_funcs = {
    'bid_price': 'max',  # Keep the highest bid price
    'ask_price': 'min',  # Keep the lowest ask price
    'bid_size': 'max',   # Keep the largest bid size
    'ask_size': 'max',   # Keep the largest ask size
    # 'bid_exchange': 'first',  # Arbitrarily keep the first exchange
    # 'ask_exchange': 'first',  # Same for ask exchange
    # 'conditions': 'first',    # Arbitrarily keep the first condition
    # 'tape': 'first'           # Same for tape
}

def clean_quotes_data(df: pd.DataFrame, interval_start: pd.Timestamp, interval_end: pd.Timestamp):
    """
    Clean and aggregate quote data, removing duplicates and calculating weighted average sizes.

    Args:
    - df (pd.DataFrame): A pandas DataFrame with a MultiIndex containing 'symbol' and 'timestamp'.
      The DataFrame should have columns: 'bid_price', 'ask_price', 'bid_size', 'ask_size',
      'bid_exchange', 'ask_exchange', 'conditions', and 'tape'.
    - interval_end (pd.Timestamp): The end time of the interval, used for the last duration calculation.

    Returns:
    - pd.DataFrame: A cleaned DataFrame with weighted average sizes for consecutive price pairs.
    """
    if df.empty:
        return df, (interval_end - interval_start).total_seconds()

    # Step 1: Remove duplicate timestamps
    df = df.groupby(level=['symbol', 'timestamp']).agg(clean_quotes_data_agg_funcs)
    df.index = df.index.set_levels(
        df.index.get_level_values('timestamp').tz_convert(ny_tz),
        level='timestamp'
    )
    df.sort_index(level='timestamp',inplace=True) # MUST BE SORTED for steps needing the returned dataframe
    
    # Calculate the duration until the next quote
    df['duration'] = df.index.get_level_values('timestamp').to_series().diff().shift(-1).values
    
    if df.empty:
        return df, (interval_end - interval_start).total_seconds()
    
    # Set the duration for the last quote to the time until interval_end
    df.loc[df.index[-1], 'duration'] = interval_end - df.index[-1][1]
    carryover = (df.index[0][1] - interval_start).total_seconds() # for adding back to previously processed interval
    df['duration'] = df['duration'].dt.total_seconds()
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
    df = df.reset_index(drop=False)
    df['price_pair'] = df['bid_price'].astype(str) + '_' + df['ask_price'].astype(str)
    
    # Identify groups
    group_changes = (df['price_pair'] != df['price_pair'].shift()).cumsum().values
    _, group_indices, group_counts = np.unique(group_changes, return_index=True, return_counts=True)
    
    # Prepare arrays
    symbols = df['symbol'].values
    bid_prices = df['bid_price'].values
    ask_prices = df['ask_price'].values
    bid_sizes = df['bid_size'].values
    ask_sizes = df['ask_size'].values
    timestamps = df['timestamp'].values
    durations = df['duration'].values
    
    # Calculate results
    result_symbols = symbols[group_indices]
    result_bid_prices = bid_prices[group_indices]
    result_ask_prices = ask_prices[group_indices]
    result_timestamps = timestamps[group_indices]
    result_durations = np.add.reduceat(durations, group_indices)
    
    # Compute weighted averages using Numba
    result_bid_sizes, result_ask_sizes = compute_weighted_sizes(
        group_indices, group_counts, bid_sizes, ask_sizes, durations
    )
    
    # Create result DataFrame
    result = pd.DataFrame({
        'symbol': result_symbols,
        'timestamp': pd.to_datetime(result_timestamps, utc=True).tz_convert(ny_tz),
        'bid_price': result_bid_prices,
        'ask_price': result_ask_prices,
        'spread': result_ask_prices - result_bid_prices,
        'bid_size': result_bid_sizes,
        'ask_size': result_ask_sizes,
        'duration': result_durations
    })
    return result.set_index(['symbol', 'timestamp'])


@jit(nopython=True)
def compute_weighted_sizes_single(prices, sizes, durations):
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
def aggregate_data(prices, sizes, durations, unique_timestamps, indices):
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
    df = df.reset_index(drop=False)
    timestamps = df['timestamp'].values.astype(np.int64) // 10**9
    
    symbols = df['symbol'].values
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
        'symbol': symbols[indices],
        'timestamp': pd.to_datetime(unique_timestamps, unit='s', utc=True).tz_convert(ny_tz),
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
    })
    return result.set_index(['symbol', 'timestamp'])


def retrieve_quote_data(client: StockHistoricalDataClient, symbols: List[str], minute_intervals_dict: Dict[str, pd.Index], params: BacktestTouchDetectionParameters, 
                        first_seconds_sample: int = np.inf, last_seconds_sample: int = np.inf, group_size: int = 10):
    quotes_data = {symbol: {'raw': [], 'agg': [], 'last_minute': None, 'acc_carryover': 0} for symbol in symbols}

    # retrieve data if it exists
    ignore_symbols = set()
    for symbol in symbols:
        if params.export_quotes_path:
            directory = os.path.dirname(params.export_quotes_path)
            quotes_zip_path = os.path.join(directory, f'quotes_{symbol}_{params.start_date.strftime("%Y-%m-%d")}_{params.end_date.strftime("%Y-%m-%d")}.zip')
            os.makedirs(directory, exist_ok=True)

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
                    del quotes_data[symbol]['last_minute']
                    del quotes_data[symbol]['acc_carryover']
                except Exception as e:
                    log(f"Error retrieving quotes from file for {symbol}: {str(e)}", level=logging.ERROR)
                    # pass
                    
    symbols = [a for a in symbols if a not in ignore_symbols]
    if len(symbols) == 0:
        return quotes_data
    
    def filter_first_and_last_seconds(symbol_df: pd.DataFrame) -> pd.DataFrame:
        seconds = symbol_df.index.get_level_values('timestamp').second
        first_end_idx = np.searchsorted(seconds, 1, side='left')
        last_start_idx = np.searchsorted(seconds, 58, side='left')
        first_seconds_data = symbol_df.iloc[:first_end_idx]
        last_seconds_data = symbol_df.iloc[last_start_idx:]
        
        if len(first_seconds_data) > first_seconds_sample or len(last_seconds_data) > last_seconds_sample:
            rs = np.random.RandomState(seeds[symbol])
            print(rs)
            if len(first_seconds_data) > first_seconds_sample:
                first_indices = rs.choice(first_seconds_data.index, first_seconds_sample, replace=False)
                first_seconds_data = first_seconds_data.loc[first_indices]
            if len(last_seconds_data) > last_seconds_sample:
                last_indices = rs.choice(last_seconds_data.index, last_seconds_sample, replace=False)
                last_seconds_data = last_seconds_data.loc[last_indices]
        return pd.concat([first_seconds_data, last_seconds_data])

    drop_cols = ['bid_exchange', 'ask_exchange', 'conditions', 'tape']
    
    for minute in tqdm(minute_intervals_dict[symbols[0]], desc='Fetching quotes'):
        # log(f'---{minute}---')
        
        minute_start = minute
        minute_end = minute + timedelta(minutes=1)
        
        seeds = {symbol: get_seed(symbol, minute) for symbol in symbols}

        request_params = StockQuotesRequest(
            symbol_or_symbols=symbols,
            start=minute_start.tz_convert('UTC'),
            end=minute_end.tz_convert('UTC'),
        )
        
        try:
            qdf0 = get_data_with_retry(client, client.get_stock_quotes, request_params)
        except Exception as e:
            log(f"Error retrieving quotes for {symbols} at {minute}: {str(e)}", level=logging.ERROR)
            
        # log(f'---data fetched---')

        for symbol in symbols:
            try:
                if not qdf0.empty and symbol in qdf0.index.get_level_values('symbol'):
                    symbol_df = qdf0.xs(symbol, level='symbol', drop_level=False).drop(columns=drop_cols)
                else:
                    symbol_df = pd.DataFrame()
                    log(f"No data found for symbol {symbol} at {minute}.", level=logging.WARNING)
            except Exception as e:
                symbol_df = pd.DataFrame()
                log(f"Error processing data for symbol {symbol} at {minute}: {str(e)}", level=logging.ERROR)

            # symbol_df.drop(columns=drop_cols, inplace=True)
            symbol_df, new_carryover = clean_quotes_data(symbol_df, minute_start, minute_end)

            # log(f'---{symbol} cleaned---')
            
            if not symbol_df.empty:
                # Process the previous minute's data
                if quotes_data[symbol]['last_minute'] is not None:
                    last_minute_df = quotes_data[symbol]['last_minute']
                    if not last_minute_df.empty:
                        last_index = last_minute_df.index[-1]
                        last_minute_df.loc[last_index, 'duration'] += quotes_data[symbol]['acc_carryover'] + new_carryover

                    # Apply grouping and weighting to the previous minute's data
                    weighted_data = apply_grouping_and_weighting(last_minute_df)
                    # log(f'---{symbol} grouped + weighted---')

                    # Extract raw data for the previous minute
                    filtered_data = filter_first_and_last_seconds(weighted_data)
                    filtered_data[['bid_size','ask_size']] = filtered_data[['bid_size','ask_size']].apply(np.floor)
                    # print(filtered_data)
                    quotes_data[symbol]['raw'].append(filtered_data.drop(columns=['spread','duration']))
                    
                    # log(f'---{symbol} raw---')
                    
                    aggregated_data = aggregate_quotes_time_based(weighted_data)
                    # print(aggregated_data)
                    quotes_data[symbol]['agg'].append(aggregated_data)
                    
                    # log(f'---{symbol} agg---')
                
                # Reset accumulated carryover as it's been applied
                quotes_data[symbol]['acc_carryover'] = 0

                # Store the current minute's data for processing in the next iteration
                quotes_data[symbol]['last_minute'] = symbol_df
                
            else:
                # Accumulate carryover when there's no activity
                quotes_data[symbol]['acc_carryover'] += new_carryover

    # Process the last minute's data for each symbol
    for symbol in symbols:
        if quotes_data[symbol]['last_minute'] is not None:
            last_minute_df = quotes_data[symbol]['last_minute']
            if not last_minute_df.empty:
                last_index = last_minute_df.index[-1]
                last_minute_df.loc[last_index, 'duration'] += quotes_data[symbol]['acc_carryover']
            
            # Apply grouping and weighting to the last minute's data
            weighted_data = apply_grouping_and_weighting(last_minute_df)
            
            # Extract raw data for the last minute
            filtered_data = filter_first_and_last_seconds(weighted_data)
            filtered_data[['bid_size','ask_size']] = filtered_data[['bid_size','ask_size']].apply(np.floor)
            quotes_data[symbol]['raw'].append(filtered_data.drop(columns=['spread','duration']))
            
            aggregated_data = aggregate_quotes_time_based(weighted_data)
            quotes_data[symbol]['agg'].append(aggregated_data)
            
            del quotes_data[symbol]['last_minute']
            del quotes_data[symbol]['acc_carryover']

    # Concatenate DataFrames for each symbol
    for symbol in symbols:
        try:
            if quotes_data[symbol]['raw'] and quotes_data[symbol]['agg']:
                log(f'{symbol} concat raw...')
                quotes_data[symbol]['raw'] = concat_with_progress(quotes_data[symbol]['raw'], group_size)
                log(f'{symbol} concat agg...')
                quotes_data[symbol]['agg'] = concat_with_progress(quotes_data[symbol]['agg'], group_size)
                log(f'...done (Raw: {quotes_data[symbol]['raw'].shape[0]} rows, Aggregated: {quotes_data[symbol]['agg'].shape[0]} rows)')
                quotes_data[symbol]['raw'].sort_index(inplace=True)
                quotes_data[symbol]['agg'].sort_index(inplace=True)
                save_quote_data(symbol, quotes_data[symbol]['raw'], quotes_data[symbol]['agg'], params)
            else:
                log(f'{symbol} data not found')
        except Exception as e:
            log(f"{type(e).__qualname__}: {e}", level=logging.ERROR)
            raise e
        
    return quotes_data

def save_quote_data(symbol, raw_df: pd.DataFrame, aggregated_df: pd.DataFrame, params: BacktestTouchDetectionParameters):
    directory = os.path.dirname(params.export_quotes_path)
    
    if isinstance(params.start_date, str):
        params.start_date = pd.to_datetime(params.start_date).tz_localize(ny_tz)
    if isinstance(params.end_date, str):
        params.end_date = pd.to_datetime(params.end_date).tz_localize(ny_tz)
    
    quotes_zip_path = os.path.join(directory, f'quotes_{symbol}_{params.start_date.strftime("%Y-%m-%d")}_{params.end_date.strftime("%Y-%m-%d")}.zip')
    os.makedirs(os.path.dirname(quotes_zip_path), exist_ok=True)
    
    with zipfile.ZipFile(quotes_zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
        with zip_file.open(f'{symbol}_raw_quotes.csv', 'w') as csv_file:
            orig_index = raw_df.index.copy()
            raw_df.index = pd.MultiIndex.from_arrays([raw_df.index.get_level_values('symbol'), 
                            raw_df.index.get_level_values('timestamp').strftime('%Y-%m-%d %H:%M:%S.%f%z').str.replace(r'(\d{2})(\d{2})$', r'\1:\2', regex=True)], 
                            names=['symbol', 'timestamp'])
            raw_df.reset_index().to_csv(csv_file, index=False)
            raw_df.index = orig_index
        with zip_file.open(f'{symbol}_aggregated_quotes.csv', 'w') as csv_file: # doesnt have formatting problem caused by microseconds
            aggregated_df.reset_index().to_csv(csv_file, index=False)
    log(f'Saved quotes to {quotes_zip_path}')

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
                log(f"Max retries ({max_retries}) reached. Giving up.", level=logging.ERROR)
                raise  # Re-raise the last exception
            
            log(f"Refreshing client session and retrying in {sleep_seconds} seconds...", level=logging.WARNING)
            refresh_client(client)
            t2.sleep(sleep_seconds)  # Sleep before retrying
            
            

def retrieve_multi_symbol_data(params: BacktestTouchDetectionParameters, symbols: List[str], first_seconds_sample: int = np.inf, 
                               last_seconds_sample: int = np.inf, group_size: int = 10):
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
    quotes_data = retrieve_quote_data(client, symbols, minute_intervals_dict, params, first_seconds_sample, last_seconds_sample, group_size)

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
    
    start_date = "2022-01-01 00:00:00" # test for GOOGL
    end_date =   "2022-02-01 00:00:00"
    
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
    # symbols = ['AAPL','MSFT'] 
    symbols = ['GOOGL']
    # symbols = ['NVDA'] 
    # symbols = ['AAPL'] 
    # NOTE: seems faster overall to do one at a time: for one day, 1 took ~1 minute, 3 took ~5 minutes
    
    # quotes_data = retrieve_multi_symbol_data(params, symbols, first_seconds_sample=100, last_seconds_sample=200)
    quotes_data = retrieve_multi_symbol_data(params, symbols)


    symbols = ['AAPL'] 
    quotes_data = retrieve_multi_symbol_data(params, symbols)
    
    
    # start_date = "2024-09-01 00:00:00"
    # end_date =   "2024-10-01 00:00:00"
    
    # params.start_date = start_date
    # params.end_date = end_date
    
    # retrieve_multi_symbol_data(params, symbols, first_seconds_sample=100, last_seconds_sample=200)
    
    
    # print(quotes_data[symbols[0]]['raw'].index)
    # print(quotes_data[symbols[0]]['agg'].index)