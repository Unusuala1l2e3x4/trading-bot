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
from typing import List, Tuple, Optional, Dict
import math
import numpy as np
from TouchDetection import BacktestTouchDetectionParameters, fill_missing_data

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



def retrieve_bar_data(client: StockHistoricalDataClient, symbol, params: BacktestTouchDetectionParameters):
    directory = os.path.dirname(params.export_bars_path)
    if isinstance(params.start_date, str):
        params.start_date = pd.to_datetime(params.start_date).tz_localize(ny_tz)
    if isinstance(params.end_date, str):
        params.end_date = pd.to_datetime(params.end_date).tz_convert(ny_tz)
    
    bars_zip_path = os.path.join(directory, f'bars_{symbol}_{params.start_date.strftime("%Y-%m-%d")}_{params.end_date.strftime("%Y-%m-%d")}.zip')
    os.makedirs(os.path.dirname(bars_zip_path), exist_ok=True)

    adjusted_csv_name = os.path.basename(bars_zip_path).replace('.zip', '.csv')
    unadjusted_csv_name = os.path.basename(bars_zip_path).replace('.zip', '_unadjusted.csv')
    split_adjusted_csv_name = os.path.basename(bars_zip_path).replace('.zip', '_split_adjusted.csv')
    dividend_adjusted_csv_name = os.path.basename(bars_zip_path).replace('.zip', '_dividend_adjusted.csv')
    adjustments_csv_name = os.path.basename(bars_zip_path).replace('.zip', '_adjustments.csv')

    df, df_unadjusted, df_split_adjusted, df_dividend_adjusted = None, None, None, None
    if params.use_saved_bars and os.path.isfile(bars_zip_path):
        with zipfile.ZipFile(bars_zip_path, 'r') as zip_file:
            file_list = zip_file.namelist()
            
            for df_name, csv_name in [('df', adjusted_csv_name), 
                                      ('df_unadjusted', unadjusted_csv_name),
                                      ('df_split_adjusted', split_adjusted_csv_name),
                                      ('df_dividend_adjusted', dividend_adjusted_csv_name)
                                      ]:
                if csv_name in file_list:
                    with zip_file.open(csv_name) as csv_file:
                        df = pd.read_csv(csv_file)
                        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_convert(ny_tz)
                        df.set_index(['symbol', 'timestamp'], inplace=True)
                    locals()[df_name] = df
                    log(f'Retrieved {df_name} from {bars_zip_path}')

    def fetch_bars(adjustment):
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Minute,
            start=params.start_date.tz_convert('UTC'),
            end=params.end_date.tz_convert('UTC'),
            adjustment=adjustment,
        )
        df = client.get_stock_bars(request_params).df
        df.index = df.index.set_levels(df.index.get_level_values('timestamp').tz_convert(ny_tz), level='timestamp')
        df.sort_index(inplace=True)
        return fill_missing_data(df)

    def calculate_adjustments(df, df_unadjusted, df_split_adjusted, df_dividend_adjusted):
        # Calculate the difference between adjusted and unadjusted prices
        price_diff = np.round(df['close'] - df_unadjusted['close'], 6)
        
        # Identify index values where the difference changes
        change_indices = price_diff[price_diff.diff() != 0].index
        # print(price_diff[price_diff.diff() != 0])
        
        # Create copies of the relevant parts of the dataframes
        df_all_adj = df.loc[change_indices, 'close'].copy()
        df_unadj = df_unadjusted.loc[change_indices, 'close'].copy()
        df_split_adj = df_split_adjusted.loc[change_indices, 'close'].copy()
        df_div_adj = df_dividend_adjusted.loc[change_indices, 'close'].copy()
        
        adjustments = []
        prev_split_factor = 1.0
        prev_dividend_amount = 0.0
        
        ind = list(reversed(change_indices))
        # Iterate backwards through the change indices
        for i, idx in tqdm(enumerate(ind), desc='calculate_adjustments'):
            diff_orig = price_diff.loc[idx]
            # all_adj_price = df_all_adj.loc[idx]
            unadj_price = df_unadj.loc[idx]
            split_adj_price = df_split_adj.loc[idx]
            div_adj_price = df_div_adj.loc[idx]
            
            # Check for split
            split_factor = split_adj_price / unadj_price
            split_adjustment = None
            if not np.isclose(split_factor, prev_split_factor):
                split_adjustment = {
                    'timestamp': idx[1],  # idx[1] is the timestamp in the MultiIndex
                    'type': 'split',
                    'factor': split_factor / prev_split_factor,  # Relative change in split factor
                    'price_diff': diff_orig
                }
                prev_split_factor = split_factor
            
            # Check for dividend
            dividend_amount = unadj_price - div_adj_price
            dividend_adjustment = None
            if not np.isclose(dividend_amount, prev_dividend_amount):
                dividend_adjustment = {
                    'timestamp': idx[1],  # idx[1] is the timestamp in the MultiIndex
                    'type': 'dividend',
                    'amount': dividend_amount - prev_dividend_amount,  # Relative change in dividend amount
                    'price_diff': diff_orig
                }
                prev_dividend_amount = dividend_amount
            
            # Add adjustments in the correct order (split first, then dividend)
            if split_adjustment:
                adjustments.append(split_adjustment)
            if dividend_adjustment:
                adjustments.append(dividend_adjustment)
            
            # Adjust past close values
            past_mask = change_indices.get_level_values('timestamp') < idx[1]
            if split_adjustment and True in past_mask:
                df_all_adj.loc[past_mask] /= split_factor
                df_split_adj.loc[past_mask] /= split_factor
            if dividend_adjustment and True in past_mask:
                df_all_adj.loc[past_mask] += dividend_amount
                df_div_adj.loc[past_mask] += dividend_amount
                
                
                
            # if split_adjustment and True in past_mask:
            #     df_all_adj.loc[past_mask].iloc[-1] /= split_factor
            #     df_split_adj.loc[past_mask].iloc[-1] /= split_factor
            # if dividend_adjustment and True in past_mask:
            #     df_all_adj.loc[past_mask].iloc[-1] += dividend_amount
            #     df_div_adj.loc[past_mask].iloc[-1] += dividend_amount
                
                
            # if split_adjustment and i < len(ind)-1:
            #     df_all_adj.loc[ind[i+1]] /= split_factor
            #     df_split_adj.loc[ind[i+1]] /= split_factor
            # if dividend_adjustment and i < len(ind)-1:
            #     df_all_adj.loc[ind[i+1]] += dividend_amount
            #     df_div_adj.loc[ind[i+1]] += dividend_amount
                
        # Reverse the adjustments list to have them in chronological order
        adjustments.reverse()
        if len(adjustments) == 0:
            return pd.DataFrame()
        return pd.DataFrame(adjustments).set_index('timestamp')

    if df is None:
        df = fetch_bars(Adjustment.ALL)
        
    if df_unadjusted is None:
        df_unadjusted = fetch_bars(Adjustment.RAW)
    if df_split_adjusted is None:
        df_split_adjusted = fetch_bars(Adjustment.SPLIT)
    if df_dividend_adjusted is None:
        df_dividend_adjusted = fetch_bars(Adjustment.DIVIDEND)
        
    # Calculate adjustment factors
    adjustment_factors = calculate_adjustments(df, df_unadjusted, df_split_adjusted, df_dividend_adjusted)
    
    # TODO: if replacing, need to delete existing file first 
    
    with zipfile.ZipFile(bars_zip_path, 'a', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
        for df_name, csv_name in [('df', adjusted_csv_name), 
                                  ('df_unadjusted', unadjusted_csv_name),
                                  ('df_split_adjusted', split_adjusted_csv_name),
                                  ('df_dividend_adjusted', dividend_adjusted_csv_name)
                                  ]:
            if csv_name not in zip_file.namelist():
                with zip_file.open(csv_name, 'w') as csv_file:
                    locals()[df_name].reset_index().to_csv(csv_file, index=False)
                log(f'Saved {df_name} to {bars_zip_path}')
        
        if adjustments_csv_name not in zip_file.namelist():
            with zip_file.open(adjustments_csv_name, 'w') as csv_file:
                adjustment_factors.to_csv(csv_file, index=True)
            log(f'Saved adjustments to {bars_zip_path}')
            
    return df, adjustment_factors



def aggregate_quotes_time_based(quotes_df: pd.DataFrame, interval_seconds=1):
    quotes_df = quotes_df.reset_index()
    quotes_df = quotes_df.sort_values('timestamp').copy()
    quotes_df['interval_start'] = quotes_df['timestamp'].dt.floor(f'{interval_seconds}s')
    
    grouped = quotes_df.groupby(['symbol', 'interval_start'])
    
    agg_funcs = {
        'bid_price': ['max', 'mean'],
        'ask_price': ['min', 'mean'],
        'bid_size': ['max', 'sum'],
        'ask_size': ['max', 'sum'],
    }
    
    aggregated = grouped.agg(agg_funcs)
    aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns.values]
    
    aggregated.rename(columns={
        'bid_price_max': 'max_bid_price',
        'bid_price_mean': 'mean_bid_price',
        'ask_price_min': 'min_ask_price',
        'ask_price_mean': 'mean_ask_price',
        'bid_size_max': 'max_bid_size',
        'bid_size_sum': 'total_bid_size',
        'ask_size_max': 'max_ask_size',
        'ask_size_sum': 'total_ask_size',
    }, inplace=True)
    
    aggregated_df = aggregated.reset_index()
    aggregated_df.set_index(['symbol', 'interval_start'], inplace=True)
    
    return aggregated_df


# Define aggregation functions
agg_funcs = {
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
    df = df.groupby(level=['symbol', 'timestamp']).agg(agg_funcs)
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
def compute_weighted_averages(group_indices, group_counts, bid_sizes, ask_sizes, durations):
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
            result_bid_sizes[i] = np.floor(np.sum(group_bid_sizes * group_durations) / total_duration)
            result_ask_sizes[i] = np.floor(np.sum(group_ask_sizes * group_durations) / total_duration)
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
    result_bid_sizes, result_ask_sizes = compute_weighted_averages(
        group_indices, group_counts, bid_sizes, ask_sizes, durations
    )
    
    # Create result DataFrame
    result = pd.DataFrame({
        'symbol': result_symbols,
        'timestamp': pd.to_datetime(result_timestamps, utc=True).tz_convert(ny_tz),
        'bid_price': result_bid_prices,
        'ask_price': result_ask_prices,
        'bid_size': result_bid_sizes,
        'ask_size': result_ask_sizes,
        'duration': result_durations
    })
    return result.set_index(['symbol', 'timestamp'])


def retrieve_quote_data(client: StockHistoricalDataClient, symbols: List[str], minute_intervals_dict: Dict[str, pd.Index], params: BacktestTouchDetectionParameters, 
                        first_seconds_sample: int = np.inf, last_seconds_sample: int = np.inf, group_size: int = 10):
    quotes_data = {symbol: {'raw': [], 'agg': [], 'last_minute': None, 'acc_carryover': 0} for symbol in symbols}


    def filter_first_seconds(symbol_df) -> pd.DataFrame:
        seconds = symbol_df.index.get_level_values('timestamp').second
        end_idx = np.searchsorted(seconds, 1, side='left')
        data = symbol_df.iloc[:end_idx]
        if len(data) > first_seconds_sample:
            return data.sample(n=first_seconds_sample, replace=False, random_state=seeds[symbol])
        return data

    def filter_last_seconds(symbol_df) -> pd.DataFrame:
        seconds = symbol_df.index.get_level_values('timestamp').second
        start_idx = np.searchsorted(seconds, 58, side='left')
        data = symbol_df.iloc[start_idx:]
        if len(data) > last_seconds_sample:
            return data.sample(n=last_seconds_sample, replace=False, random_state=seeds[symbol])
        return data

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
        qdf0 = client.get_stock_quotes(request_params).df
        
        # log(f'---data fetched---')

        for symbol in symbols:
            symbol_df = qdf0.xs(symbol, level='symbol', drop_level=False) if not qdf0.empty else pd.DataFrame()
            symbol_df.drop(columns=drop_cols, inplace=True)
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
                    first_seconds = filter_first_seconds(weighted_data)
                    last_seconds = filter_last_seconds(weighted_data)
                    quotes_data[symbol]['raw'].extend([first_seconds.drop('duration', axis=1), last_seconds.drop('duration', axis=1)])
                    
                    # log(f'---{symbol} raw---')
                    
                    # Aggregate the previous minute's data
                    aggregated_data = aggregate_quotes_time_based(weighted_data)
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
            first_seconds = filter_first_seconds(weighted_data)
            last_seconds = filter_last_seconds(weighted_data)
            quotes_data[symbol]['raw'].extend([first_seconds.drop('duration', axis=1), last_seconds.drop('duration', axis=1)])
            
            aggregated_data = aggregate_quotes_time_based(weighted_data)
            quotes_data[symbol]['agg'].append(aggregated_data)

    # Concatenate DataFrames for each symbol
    for symbol in symbols:
        try:
            if quotes_data[symbol]['raw'] and quotes_data[symbol]['agg']:
                log(f'{symbol} concat raw...')
                raw_df = concat_with_progress(quotes_data[symbol]['raw'], group_size)
                log(f'{symbol} concat agg...')
                aggregated_df = concat_with_progress(quotes_data[symbol]['agg'], group_size)
                log(f'...done (Raw: {raw_df.shape[0]} rows, Aggregated: {aggregated_df.shape[0]} rows)')
                raw_df.sort_index(inplace=True)
                aggregated_df.sort_index(inplace=True)
                
                # TODO: adjust quotes data for dividends/splits before saving 
                
                save_quote_data(symbol, raw_df, aggregated_df, params)
                del quotes_data[symbol]
            else:
                log(f'{symbol} data not found')
        except Exception as e:
            log(f"{type(e).__qualname__}: {e}", level=logging.ERROR)
            raise e
        
    # return quotes_data

def save_quote_data(symbol, raw_df: pd.DataFrame, aggregated_df: pd.DataFrame, params: BacktestTouchDetectionParameters):
    directory = os.path.dirname(params.export_quotes_path)
    
    if isinstance(params.start_date, str):
        params.start_date = pd.to_datetime(params.start_date).tz_localize(ny_tz)
    if isinstance(params.end_date, str):
        params.end_date = pd.to_datetime(params.end_date).tz_localize(ny_tz)
    
    quotes_zip_path = os.path.join(directory, f'quotes_{symbol}_{params.start_date.strftime("%Y-%m-%d")}_{params.end_date.strftime("%Y-%m-%d")}.zip')
    os.makedirs(os.path.dirname(quotes_zip_path), exist_ok=True)
    
    with zipfile.ZipFile(quotes_zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
        with zip_file.open(f'{symbol}_raw_quotes.csv', 'w') as csv_file: # should replace existing
            raw_df.reset_index().to_csv(csv_file, index=False)
        with zip_file.open(f'{symbol}_aggregated_quotes.csv', 'w') as csv_file: # should replace existing
            aggregated_df.reset_index().to_csv(csv_file, index=False)
    log(f'Saved quotes to {quotes_zip_path}')

def retrieve_multi_symbol_data(params: BacktestTouchDetectionParameters, symbols: List[str], first_seconds_sample: int = np.inf, last_seconds_sample: int = np.inf, group_size: int = 10):
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
        df, adjustment_factors = retrieve_bar_data(client, symbol, params)
        # df = retrieve_bar_data(client, symbol, params)
        # print(df)
        # print(adjustment_factors)

        
        minute_intervals = df.index.get_level_values('timestamp')
        minute_intervals = minute_intervals[(minute_intervals.time >= time(9, 31)) & (minute_intervals.time <= time(15, 59))]
        minute_intervals_dict[symbol] = minute_intervals
        
        # elapsed_time = t2.time() - start_time
        # remaining_sleep_time = max(0, SLEEP_TIME - elapsed_time)
        # t2.sleep(remaining_sleep_time)
        # t2.sleep(SLEEP_TIME)
        
    assert len(set([a.size for a in minute_intervals_dict.values()])) <= 1 # make sure len(minute_intervals) are the same

    # Retrieve quote data
    retrieve_quote_data(client, symbols, minute_intervals_dict, params, first_seconds_sample, last_seconds_sample, group_size)

    log("Data retrieval complete for all symbols.")


if __name__=="__main__":    
    start_date = "2024-07-01 00:00:00"
    end_date =   "2024-08-01 00:00:00"

    # start_date = "2024-08-19 00:00:00"
    # end_date =   "2024-08-20 00:00:00"
    start_date = "2024-08-20 00:00:00"
    end_date =   "2024-08-21 00:00:00"
    
    # start_date = "2024-08-01 00:00:00"
    # end_date =   "2024-09-01 00:00:00"

    # start_date = "2024-09-01 00:00:00"
    # end_date =   "2024-10-01 00:00:00"
    
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
        use_saved_bars=True, # make False if dont want to save bar data
        rolling_avg_decay_rate=0.85,
        export_bars_path=f'bars/',
        export_quotes_path=f'quotes2/'
    )


    # symbols = ['AAPL', 'GOOGL', 'NVDA']
    symbols = ['AAPL','MSFT'] 
    # symbols = ['NVDA'] 
    # symbols = ['AAPL'] 
    # NOTE: seems faster overall to do one at a time: for one day, 1 took ~1 minute, 3 took ~5 minutes
    
    retrieve_multi_symbol_data(params, symbols, first_seconds_sample=100, last_seconds_sample=200)

    start_date = "2024-09-01 00:00:00"
    end_date =   "2024-10-01 00:00:00"
    
    params.start_date = start_date
    params.end_date = end_date
    
    retrieve_multi_symbol_data(params, symbols, first_seconds_sample=100, last_seconds_sample=200)