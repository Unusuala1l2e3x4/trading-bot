import os
import zipfile
import pandas as pd
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockQuotesRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment
from alpaca.data.models import Bar, Quote
from datetime import datetime, time, timedelta
from tqdm import tqdm
import time as t2
from typing import List, Tuple, Optional, Dict, Callable
import math
import copy
import numpy as np
from trading_bot.TouchDetectionParameters import BacktestTouchDetectionParameters, LiveTouchDetectionParameters

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
    logger.log(level, message, exc_info=level >= logging.ERROR)

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
                            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert(ny_tz)
                            df.set_index(['symbol', 'timestamp'], inplace=True)
                            log(f'Retrieved {adjusted_csv_name} from {bars_zip_path}')
                    except Exception as e:
                        log(f"Error retrieving adjusted bars from file for {symbol}: {str(e)}", level=logging.ERROR)
                        df = None
                if unadjusted_csv_name in file_list:
                    try:
                        with zip_file.open(unadjusted_csv_name) as csv_file:
                            df_unadjusted = pd.read_csv(csv_file)
                            df_unadjusted['timestamp'] = pd.to_datetime(df_unadjusted['timestamp'], utc=True).dt.tz_convert(ny_tz)
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
            feed='sip'
        )
        try:
            df = get_data_with_retry(client, client.get_stock_bars, request_params)
        except Exception as e:
            log(f"Error requesting bars for {symbol}: {str(e)}", level=logging.ERROR)
            return pd.DataFrame()
        if not df.empty:
            df.index = df.index.set_levels(df.index.get_level_values('timestamp').tz_convert(ny_tz) + timedelta(minutes=1), level='timestamp')
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


def count_changes(series: pd.Series) -> int:
    """Count the number of actual changes in values, excluding consecutive repeats."""
    return (series != series.shift()).sum()
count_changes.__name__ = 'changes'

drop_cols = ['bid_exchange', 'ask_exchange', 'conditions', 'tape']
intra_cols = ['bid_price', 'ask_price', 'bid_size', 'ask_size', 'spread']
base_cols = ['bid_price', 'ask_price', 'bid_size', 'ask_size']

# Create separate aggregation dicts for different use cases
aggregation_dict = {
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
    'ask_size_intra_neg': 'sum',
    'spread': ['first', 'max', 'min', 'last', 'mean']
}

def prepare_quote_data(df: pd.DataFrame, grouping_cols: list) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Common preprocessing steps for quote data.
    """
    df = df.drop(columns=drop_cols, errors='ignore').sort_index(level=['symbol', 'timestamp'])
    df['spread'] = (df['ask_price'] - df['bid_price']).round(4)
    
    # Calculate all diffs at once
    diff_df = df.groupby(level=grouping_cols)[intra_cols].diff()
    
    # Add intra columns efficiently
    for col in intra_cols:
        changes = diff_df[col].round(4)
        df[f'{col}_intra_pos'] = changes.clip(lower=0)
        df[f'{col}_intra_neg'] = changes.clip(upper=0)

    group_sizes = df.groupby(level=grouping_cols).size()
    return df, group_sizes

def add_cross_timestamp_changes(df: pd.DataFrame, df_agg: pd.DataFrame, carryover_exists: bool) -> None:
    """
    Add cross-timestamp changes to intra columns for price and size data.
    Modifies df_agg in-place.
    Assumes single symbol data, but retains symbol level in index.
    Calculates changes between consecutive timestamps using the 'last' value
    from the previous timestamp and the 'first' value from the current timestamp.
    When carryover_exists, uses the carryover quote to calculate changes for the first row.
    
    Args:
        df: Raw quotes DataFrame, contains carryover quote when carryover_exists is True
        df_agg: Aggregated DataFrame with '_first', '_last', '_intra_pos', '_intra_neg' columns
        carryover_exists: Boolean indicating if first row of df is a carryover quote
    """
    base_cols = ['bid_price', 'ask_price', 'bid_size', 'ask_size']
    
    for base_col in base_cols:
        # Calculate regular cross-timestamp changes
        cross_changes = (df_agg[f'{base_col}_first'] - df_agg[f'{base_col}_last'].shift()).fillna(0)
        
        # If carryover exists, calculate change between carryover quote and first aggregated row
        if carryover_exists:
            carryover_value = df[base_col].iloc[0]  # Get value from carryover quote
            first_value = df_agg[f'{base_col}_first'].iloc[0]  # Get first value in aggregated data
            initial_change = first_value - carryover_value
            cross_changes.iloc[0] = initial_change  # Replace first change

        # Add positive and negative changes to respective columns
        df_agg[f'{base_col}_intra_pos_sum'] += cross_changes.clip(lower=0)
        df_agg[f'{base_col}_intra_neg_sum'] += cross_changes.clip(upper=0)


@jit(nopython=True)
def calculate_time_weighted_spread(spreads: np.ndarray, timestamps: np.ndarray, 
                                 interval_start: float, interval_end: float) -> float:
    """
    Calculate time-weighted average spread using numba for optimization.
    All timestamps should be in unix timestamp format (seconds).
    """
    if len(spreads) == 0:
        return 0.0
    
    durations = np.empty(len(spreads))
    durations[0] = timestamps[0] - interval_start
    durations[1:] = timestamps[1:] - timestamps[:-1]
    durations[-1] = interval_end - timestamps[-1]
    
    total_duration = interval_end - interval_start
    weighted_sum = np.sum(spreads * durations)
    return weighted_sum / total_duration

def calculate_twap_per_second_group(group: pd.DataFrame) -> float:
    """
    Calculate time-weighted average spread for a group of quotes.
    For use with pandas apply.
    """
    second_end = group.index.get_level_values('second')[0]
    second_start = second_end - pd.Timedelta(seconds=1)
    
    # Convert to numpy array of timestamps in seconds
    timestamps = group.index.get_level_values('timestamp').to_numpy(dtype='int64') / 1e9
    interval_start_ts = second_start.timestamp()
    interval_end_ts = second_end.timestamp()
    spreads = group['spread'].values
    
    return calculate_time_weighted_spread(
        spreads, 
        timestamps, 
        interval_start_ts, 
        interval_end_ts
    )

def calculate_twap_micro_data(df: pd.DataFrame, interval_start: datetime, interval_end: datetime) -> float:
    """
    Calculate time-weighted average spread for a group of quotes.
    For use with pandas apply.
    """
    # Convert to numpy array of timestamps in seconds
    timestamps = df.index.get_level_values('timestamp').to_numpy(dtype='int64') / 1e9
    interval_start_ts = interval_start.timestamp()
    interval_end_ts = interval_end.timestamp()
    spreads = df['spread_last'].values
    
    return calculate_time_weighted_spread(
        spreads, 
        timestamps, 
        interval_start_ts, 
        interval_end_ts
    )

# NOTE: results in micro dataset
def clean_quotes_data(df: pd.DataFrame, carryover_exists: bool, interval_start: pd.Timestamp, interval_end: pd.Timestamp, calculate_durations: bool = True) -> Tuple[pd.DataFrame, float]:
    """
    Clean quotes data and calculate intra-timestamp changes with improved efficiency.
    """
    if df.empty:
        return df, (interval_end - interval_start).total_seconds()

    # assert len(pd.unique(df.index.get_level_values('symbol'))) == 1, list(pd.unique(df.index.get_level_values('symbol')))
    
    # Prepare data
    grouping_cols = ['symbol', 'timestamp']
    df, group_sizes = prepare_quote_data(df, grouping_cols)
    
    # Aggregate based on repeating timestamps (each group has the same timestamp)
    if carryover_exists:
        df_agg = df.iloc[1:].groupby(level=grouping_cols).agg(aggregation_dict)
    else:
        df_agg = df.groupby(level=grouping_cols).agg(aggregation_dict)
    df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
    df_agg.insert(0, 'count', group_sizes)
    
    add_cross_timestamp_changes(df, df_agg, carryover_exists)

    # Handle timezones
    timestamps = df_agg.index.get_level_values('timestamp')
    df_agg.index = df_agg.index.set_levels(
        timestamps.tz_localize(ny_tz) if timestamps.tz is None else timestamps.tz_convert(ny_tz),
        level='timestamp'
    )

    # Calculate durations
    if calculate_durations:
        timestamps_series = df_agg.index.get_level_values('timestamp').tz_localize(None).to_series().astype('datetime64[us]')
        duration_series = timestamps_series.diff().dt.total_seconds().shift(-1)
        df_agg['duration'] = duration_series.values
        last_duration = (interval_end.tz_localize(None) - timestamps_series.iloc[-1]).total_seconds()
        df_agg.iloc[-1, df_agg.columns.get_loc('duration')] = last_duration
        
        # Validations
        assert not np.any(duration_series.values[:-1] <= 0), \
            ("Zero or negative durations found after processing\n", df_agg.loc[df_agg['duration'] <= 0])
    assert interval_start <= timestamps[0], (interval_start, timestamps[0])
    
    carryover = (timestamps[0] - interval_start).total_seconds()
    return df_agg, carryover

# NOTE: results in macro dataset
def aggregate_by_second(df: pd.DataFrame, carryover_exists: bool, interval_start: pd.Timestamp, interval_end: pd.Timestamp) -> pd.DataFrame:
    """
    Aggregate quote data by second intervals with time-weighted spread calculation.
    """
    if df.empty:
        return df
    
    # assert len(pd.unique(df.index.get_level_values('symbol'))) == 1, list(pd.unique(df.index.get_level_values('symbol')))

    # Add seconds index efficiently
    df = df.assign(
        second=df.index.get_level_values('timestamp').ceil('s')
    ).set_index('second', append=True)
    
    # Prepare data
    grouping_cols = ['symbol', 'second']
    df, group_sizes = prepare_quote_data(df, grouping_cols)
    
    # Aggregate based on repeating timestamps (each group has the same whole-number second value)
    if carryover_exists:
        df_agg = df.iloc[1:].groupby(level=grouping_cols).agg(aggregation_dict)
    else:
        df_agg = df.groupby(level=grouping_cols).agg(aggregation_dict)
    df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
    df_agg.insert(0, 'count', group_sizes)
    
    add_cross_timestamp_changes(df, df_agg, carryover_exists)

    # Calculate TWAP using apply
    df_agg['spread_twap'] = df.groupby(level=grouping_cols).apply(calculate_twap_per_second_group)
    
    # Handle timezone
    timestamps = df_agg.index.get_level_values('second')
    df_agg.index = df_agg.index.set_levels(
        timestamps.tz_localize(ny_tz) if timestamps.tz is None else timestamps.tz_convert(ny_tz),
        level='second'
    )
    
    df_agg.index = df_agg.index.rename('timestamp', level='second')
    return df_agg




SEED_DIV = 2**32
def get_seed(symbol: str, minute: datetime) -> int:
    return int(hashlib.sha256(f"{symbol}_{minute}".encode()).hexdigest(), 16) % SEED_DIV

def get_quotes_zip_path(symbol, params: BacktestTouchDetectionParameters):
   directory = os.path.dirname(params.export_quotes_path)
   return os.path.join(directory, f'quotes_{symbol}_{params.start_date.strftime("%Y-%m-%d")}_{params.end_date.strftime("%Y-%m-%d")}.zip')


from concurrent.futures import ThreadPoolExecutor

def fetch_next_minute_quotes(client: StockHistoricalDataClient, symbols: List[str], minute_start: datetime, minute_end: datetime):
    request_params = StockQuotesRequest(
        symbol_or_symbols=symbols,
        start=minute_start.tz_convert('UTC'),
        end=minute_end.tz_convert('UTC'),
        feed='sip' # default for market data subscription?
        # feed='iex'
    )
    try:
        return get_data_with_retry(client, client.get_stock_quotes, request_params)
    except Exception as e:
        log(f"Error retrieving quotes for {symbols} at {minute_start}: {str(e)}", level=logging.ERROR)
        return pd.DataFrame()

def retrieve_quote_data(client: StockHistoricalDataClient, symbols: List[str], minute_intervals_dict: Dict[str, pd.DatetimeIndex], params: BacktestTouchDetectionParameters, 
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
                            raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'], utc=True).dt.tz_convert(ny_tz)
                            raw_df.set_index(['symbol', 'timestamp'], inplace=True)
                        with zip_file.open(f'{symbol}_aggregated_quotes.csv') as csv_file:
                            aggregated_df = pd.read_csv(csv_file)
                            aggregated_df['timestamp'] = pd.to_datetime(aggregated_df['timestamp'], utc=True).dt.tz_convert(ny_tz)
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
    
    def filter_first_and_last_seconds(df: pd.DataFrame) -> pd.DataFrame:
        seconds = df.index.get_level_values('timestamp').second
        first_end_idx = np.searchsorted(seconds, 1, side='left')
        last_start_idx = np.searchsorted(seconds, 58, side='left')
        first_seconds_data = df.iloc[:first_end_idx]
        last_seconds_data = df.iloc[last_start_idx:] if last_start_idx < len(df) else pd.DataFrame()
        
        # Ensure last_seconds_data has at least 1 row
        if last_seconds_data.empty and not df.empty:
            last_row = df.iloc[[-1]]
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
                    if attempt >= 1:
                        log(f"Successful after {attempt} retries", level=logging.INFO)
                except Exception as e:
                    attempt += 1
                    if attempt == 1:
                        log(f"Attempt {attempt}: Error appending to file for {symbol} at {current_minute}: {str(e)}", level=logging.ERROR)
                    if attempt >= max_retries:
                        log(f"Attempt {attempt}: Error appending to file for {symbol} at {current_minute}: {str(e)}\nMax retries ({max_retries}) reached. Giving up.", level=logging.ERROR)
                        break
                        # raise  # Re-raise the last exception
                    t2.sleep(sleep_seconds)
                    continue
                break
            
            if index_format:
                df.index = orig_index
    
    prev_raw_data = {symbol: None for symbol in symbols}
    
    prev_quote = {symbol: None for symbol in symbols} # use for time-weighted aggregations
    
    acc_carryover = {symbol: 0 for symbol in symbols}
    first_append_raw = {symbol: True for symbol in symbols}
    first_append_agg = {symbol: True for symbol in symbols}
    
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
        future = executor.submit(fetch_next_minute_quotes, client, symbols, current_minute_start, current_minute_end)
        
        while True:
            # Get current minute's data
            # timer_start = datetime.now()
            qdf0 = future.result()
            # timer_end = datetime.now()
            # log(f'wait time {timer_end - timer_start}')
            
            # Start next request if there are more minutes
            try:
                next_minute = next(minutes_iter)
                next_minute_start = next_minute
                next_minute_end = next_minute + timedelta(minutes=1)
                future = executor.submit(fetch_next_minute_quotes, client, symbols, next_minute_start, next_minute_end)
            except StopIteration:
                future = None
            
            # Process current minute's data
            seeds = {symbol: get_seed(symbol, current_minute) for symbol in symbols}
            
            for symbol in symbols:
                try:
                    if not qdf0.empty and symbol in qdf0.index.get_level_values('symbol'):
                        qdf = qdf0.xs(symbol, level='symbol', drop_level=False)
                    else:
                        qdf = pd.DataFrame()
                        log(f"No data found for symbol {symbol} at {current_minute}.", level=logging.WARNING)
                except Exception as e:
                    qdf = pd.DataFrame()
                    log(f"Error processing data for symbol {symbol} at {current_minute}: {str(e)}", level=logging.ERROR)

                carryover_quote = prev_quote[symbol]
                
                # Calculate time-weighted spread
                if carryover_quote is not None:
                    assert carryover_quote.iloc[-1].name[1] <= current_minute_start.tz_convert('UTC')
                    carryover_quote.index = pd.MultiIndex.from_tuples([(carryover_quote.iloc[-1].name[0], current_minute_start.tz_convert('UTC'))], names=carryover_quote.index.names)
                    qdf = pd.concat([carryover_quote, qdf])

                raw_df, new_carryover = clean_quotes_data(qdf, carryover_quote is not None, current_minute_start, current_minute_end)
                agg_df = aggregate_by_second(qdf, carryover_quote is not None, current_minute_start, current_minute_end)
                
                if not qdf.empty:
                    prev_quote[symbol] = qdf.iloc[[-1]]
                # if prev_quote[symbol] is not None: # set index to exact start of interval
                #     prev_quote[symbol].name = (symbol, current_minute_end.tz_convert('UTC'))
                # print(prev_quote[symbol])

                if not agg_df.empty:
                    if params.export_quotes_path:
                        # print(list(agg_df.columns))
                        
                        append_quote_segment(temp_files[symbol]['agg'], agg_df, is_first=first_append_agg[symbol], index_format=False)
                    
                    if first_append_agg[symbol]:
                        first_append_agg[symbol] = False

                if not raw_df.empty:
                    # Process the previous minute's data
                    if prev_raw_data[symbol] is not None:
                        prev_raw_df = prev_raw_data[symbol]
                        if not prev_raw_df.empty:
                            last_index = prev_raw_df.index[-1]
                            prev_raw_df.loc[last_index, 'duration'] += acc_carryover[symbol] + new_carryover

                        filtered_data = filter_first_and_last_seconds(prev_raw_df)
                        
                        if params.export_quotes_path:
                            append_quote_segment(temp_files[symbol]['raw'], filtered_data, is_first=first_append_raw[symbol], index_format=True)
                            # print(list(filtered_data.columns))
                            if first_append_raw[symbol]:
                                first_append_raw[symbol] = False
                            
                    acc_carryover[symbol] = 0
                    prev_raw_data[symbol] = raw_df
                    
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
        if prev_raw_data[symbol] is not None:
            prev_raw_df = prev_raw_data[symbol]
            if not prev_raw_df.empty:
                last_index = prev_raw_df.index[-1]
                prev_raw_df.loc[last_index, 'duration'] += acc_carryover[symbol]

            filtered_data = filter_first_and_last_seconds(prev_raw_df)
            
            if params.export_quotes_path:
                append_quote_segment(temp_files[symbol]['raw'], filtered_data, is_first=first_append_raw[symbol], index_format=True)
                if first_append_raw[symbol]:
                    first_append_raw[symbol] = False
                  
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
                        raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'], utc=True).dt.tz_convert(ny_tz)
                        raw_df.set_index(['symbol', 'timestamp'], inplace=True)
                        # print(raw_df.index.get_level_values('timestamp'))
                    with zip_file.open(f'{symbol}_aggregated_quotes.csv') as csv_file:
                        aggregated_df = pd.read_csv(csv_file)
                        # print(pd.to_datetime(aggregated_df['timestamp']))
                        aggregated_df['timestamp'] = pd.to_datetime(aggregated_df['timestamp'], utc=True).dt.tz_convert(ny_tz)
                        aggregated_df.set_index(['symbol', 'timestamp'], inplace=True)
                        # print(aggregated_df.index.get_level_values('timestamp'))
                quotes_data[symbol]['raw'] = raw_df
                quotes_data[symbol]['agg'] = aggregated_df
            except Exception as e:
                log(f"Error reading saved quotes for {symbol}: {str(e)}", level=logging.ERROR)
    if return_data:
        return quotes_data

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
            
def get_stock_latest_quote_with_retry(client: StockHistoricalDataClient, request_params: StockLatestQuoteRequest, max_retries=10, sleep_seconds=0.01) -> Optional[Quote]:
    """
    Retrieves the latest quote for a stock with retry logic.
    
    Args:
        client: StockHistoricalDataClient instance
        request_params: StockLatestQuoteRequest parameters
        max_retries: Maximum number of retry attempts (default: 10)
        sleep_seconds: Seconds to wait between retries (default: 0.01)
        
    Returns:
        Quote object if successful, None if all retries fail
    """
    attempt = 0
    while True:
        try:
            res = client.get_stock_latest_quote(request_params)
            
            # Handle single symbol vs multiple symbols
            if isinstance(res, dict):
                # For multiple symbols, return the full dictionary
                return res
            else:
                # For single symbol, return the Quote object
                return res

        except Exception as e:
            attempt += 1
            symbols = request_params.symbol_or_symbols
            
            log(f"Attempt {attempt}: Error retrieving latest quote for {symbols}: {str(e)}", level=logging.ERROR)
            
            if attempt >= max_retries:
                log(f"Max retries ({max_retries}) reached. Giving up.", level=logging.ERROR)
                return None
                
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
    for symbol in symbols:
    # for symbol in tqdm(symbols, desc="Retrieving bar data"):
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

    # start_date = "2024-10-01 00:00:00"
    # end_date =   "2024-11-01 00:00:00"

    start_date = "2024-11-01 00:00:00"
    end_date =   "2024-12-01 00:00:00"

    # start_date = "2024-09-04 00:00:00"
    # end_date =   "2024-09-05 00:00:00"
    
    # start_date = "2022-01-01 00:00:00"
    # end_date =   "2022-02-01 00:00:00"
    
    # start_date = "2022-01-12 00:00:00"
    # end_date =   "2022-01-13 00:00:00"
    
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
        
        ema_span=15,
        price_ema_span=26,
        
        export_bars_path=f'bars/',
        export_quotes_path=f'quotes/'
    )


    # symbols = ['AAPL', 'GOOGL', 'NVDA']
    # symbols = ['AAPL'] 
    # symbols = ['GOOGL']
    # symbols = ['NVDA'] 
    # symbols = ['AAPL','GOOGL'] 
    # NOTE: seems faster overall to do one at a time: for one day, 1 took ~1 minute, 3 took ~5 minutes
    
    # quotes_data = retrieve_multi_symbol_data(params, symbols, first_seconds_sample=100, last_seconds_sample=200)
    # quotes_data = retrieve_multi_symbol_data(params, symbols)
    
    
    
    # symbols = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA', 'TSLA']
    symbols = ['MSTR','MARA','INTC','GOOG']
    # symbols = ['AAPL']
    # symbols = ['QFIN']
        
    # params.start_date = "2024-09-04 00:00:00"
    # params.end_date =   "2024-09-05 00:00:00"
    # for symbol in symbols:
    #     retrieve_multi_symbol_data(params, [symbol], return_data=False)
    
    # params.start_date = "2024-11-11 00:00:00"
    # params.end_date =   "2024-11-16 00:00:00"
    # for symbol in symbols:
    #     retrieve_multi_symbol_data(params, [symbol], return_data=False)
        
    # params.start_date = "2024-11-18 00:00:00"
    # params.end_date =   "2024-11-23 00:00:00"
    # for symbol in symbols:
    #     retrieve_multi_symbol_data(params, [symbol], return_data=False)
        
    # params.start_date = "2024-11-11 00:00:00"
    # params.end_date =   "2024-11-23 00:00:00"
    # for symbol in symbols:
    #     retrieve_multi_symbol_data(params, [symbol], return_data=False)

    # params.start_date = "2024-01-01 00:00:00"
    # params.end_date =   "2024-02-01 00:00:00"
    # for symbol in symbols:
    #     retrieve_multi_symbol_data(params, [symbol], return_data=False)
    

    for month in range(11, 0, -1):  # Loop through months
        start_date = f"2024-{month:02d}-01 00:00:00"
        end_date = f"2024-{month + 1:02d}-01 00:00:00"
        print(start_date, end_date)
        
        params.start_date = start_date
        params.end_date = end_date
        
        for symbol in symbols:
            retrieve_multi_symbol_data(params, [symbol], return_data=False)


    # symbols = ['AAPL'] 
    # quotes_data = retrieve_multi_symbol_data(params, symbols)
    
    
    # start_date = "2024-09-01 00:00:00"
    # end_date =   "2024-10-01 00:00:00"
    
    # params.start_date = start_date
    # params.end_date = end_date
    
    # retrieve_multi_symbol_data(params, symbols, first_seconds_sample=100, last_seconds_sample=200)
    
    
    # print(quotes_data[symbols[0]]['raw'].index)
    # print(quotes_data[symbols[0]]['agg'].index)