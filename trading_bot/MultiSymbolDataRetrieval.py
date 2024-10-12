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
from TouchDetection import BacktestTouchDetectionParameters, clean_quotes_data, fill_missing_data

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

    if df is None:
        df = fetch_bars(Adjustment.ALL)
    if df_unadjusted is None:
        df_unadjusted = fetch_bars(Adjustment.RAW)
    if df_split_adjusted is None:
        df_split_adjusted = fetch_bars(Adjustment.SPLIT)
    if df_dividend_adjusted is None:
        df_dividend_adjusted = fetch_bars(Adjustment.DIVIDEND)

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
        return pd.DataFrame(adjustments).set_index('timestamp')
        
    # Calculate adjustment factors
    adjustment_factors = calculate_adjustments(df, df_unadjusted, df_split_adjusted, df_dividend_adjusted)
    
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
        
        # if adjustments_csv_name not in zip_file.namelist():
        with zip_file.open(adjustments_csv_name, 'w') as csv_file:
            adjustment_factors.to_csv(csv_file, index=True)
        log(f'Saved adjustments to {bars_zip_path}')
            
    return df, adjustment_factors



from numpy.random import MT19937, RandomState, SeedSequence
import hashlib
from functools import lru_cache

@lru_cache(maxsize=None)
def get_seed(symbol, minute):
    return int(hashlib.sha256(f"{symbol}_{minute}".encode()).hexdigest(), 16) % (2**32)

def retrieve_quote_data(client: StockHistoricalDataClient, symbols: List[str], minute_intervals_dict: Dict[str, pd.Index], params: BacktestTouchDetectionParameters, 
                        sample_size: int = None, group_size: int = 10):
    quotes_data = {symbol: [] for symbol in symbols}
    
    after_sec = 1
    before_sec_search = [float(round(a,2)) for a in np.logspace( 0, np.log2(60-after_sec), 5, base=2)]
    log(f'for each minute, searching previous {before_sec_search} seconds to {after_sec} after')
        
    for minute in tqdm(minute_intervals_dict[symbols[0]], desc='Fetching quotes'):
        minute_end = minute + timedelta(seconds=after_sec)
        symbols_to_process = set(symbols)

        for before_sec in before_sec_search:
            if not symbols_to_process:
                break
            # start_time = t2.time()
            minute_start = minute - timedelta(seconds=before_sec)

            request_params = StockQuotesRequest(
                symbol_or_symbols=list(symbols_to_process),
                start=minute_start.tz_convert('UTC'),
                end=minute_end.tz_convert('UTC'),
            )
            qdf0 = client.get_stock_quotes(request_params).df

            if qdf0.empty:
                continue
            for symbol in list(symbols_to_process):
                symbol_df = qdf0.xs(symbol, level='symbol', drop_level=False)
                if not symbol_df.empty:
                    symbol_df = clean_quotes_data(symbol_df)
                    t = symbol_df.index.get_level_values('timestamp')
                    latest_before_earlier = t[t < minute].max()
                    if not pd.isna(latest_before_earlier):
                        symbols_to_process.remove(symbol)
                        df_before = symbol_df.loc[symbol_df.index.get_level_values('timestamp') == latest_before_earlier] # last quote in previous minute
                        df_after = symbol_df.loc[symbol_df.index.get_level_values('timestamp') >= minute] # 
                        assert not df_before.empty
                        if not df_after.empty:
                            if sample_size is not None and len(df_after) > sample_size:
                                seed = get_seed(symbol, minute)
                                rs = RandomState(MT19937(SeedSequence(seed)))
                                df_after = df_after.sample(n=sample_size, replace=False, random_state=rs)
                            quotes_data[symbol].append(pd.concat([df_before, df_after]))
                        else:
                            quotes_data[symbol].append(df_before)

            # elapsed_time = t2.time() - start_time
            # remaining_sleep_time = max(0, 0.3+(0.02*n) - elapsed_time)
            # remaining_sleep_time = max(0.02*n, 0.3 - elapsed_time)
            # t2.sleep(remaining_sleep_time)        # NOTE: may get throttled if len(symbols) > 1
            
            # t2.sleep(SLEEP_TIME)

        if symbols_to_process:
            log(f"{minute.date()} {minute.time()} - No data found for {symbols_to_process}", level=logging.WARNING)
            
    # Concatenate DataFrames for each symbol
    for symbol in symbols:
        try:
            if quotes_data[symbol]:
                log(f'{symbol} concat...')
                # quotes_data[symbol] = pd.concat(quotes_data[symbol])
                # quotes_data[symbol] = custom_concat(quotes_data[symbol])
                
                # test group sizes
                # for i in [11,12,13,14,15,16,17,18]:
                #     log(f'GROUP_SIZE {i}')
                #     concat_with_progress(quotes_data[symbol], i)
                # log(f'DONE TEST')
                
                quotes_data[symbol] = concat_with_progress(quotes_data[symbol], group_size)
                log(f'...done ({quotes_data[symbol].shape[0]} total rows)')
                quotes_data[symbol].sort_index(inplace=True)
                save_quote_data(symbol, quotes_data[symbol], params)
                del quotes_data[symbol]
            else:
                log(f'{symbol} data not found')
        except Exception as e:
            log(f"{type(e).__qualname__}: {e}", level=logging.ERROR)
            raise e
        
    # return quotes_data

def save_quote_data(symbol, qdf: pd.DataFrame, params: BacktestTouchDetectionParameters):
    directory = os.path.dirname(params.export_quotes_path)
    
    if isinstance(params.start_date, str):
        params.start_date = pd.to_datetime(params.start_date).tz_localize(ny_tz)
    if isinstance(params.end_date, str):
        params.end_date = pd.to_datetime(params.end_date).tz_localize(ny_tz)
    
    quotes_zip_path = os.path.join(directory, f'quotes_{symbol}_{params.start_date.strftime("%Y-%m-%d")}_{params.end_date.strftime("%Y-%m-%d")}.zip')
    os.makedirs(os.path.dirname(quotes_zip_path), exist_ok=True)
    
    with zipfile.ZipFile(quotes_zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zip_file:
        with zip_file.open(os.path.basename(quotes_zip_path).replace('.zip', '.csv'), 'w') as csv_file:
            qdf.reset_index().to_csv(csv_file, index=False)
    log(f'Saved quotes to {quotes_zip_path}')

def retrieve_multi_symbol_data(params: BacktestTouchDetectionParameters, symbols: List[str], sample_size: int = None, group_size: int = 10):
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
    retrieve_quote_data(client, symbols, minute_intervals_dict, params, sample_size, group_size)

    log("Data retrieval complete for all symbols.")


if __name__=="__main__":    
    start_date = "2024-07-01 00:00:00"
    end_date =   "2024-08-01 00:00:00"

    # start_date = "2024-08-19 00:00:00"
    # end_date =   "2024-08-20 00:00:00"
    # start_date = "2024-08-20 00:00:00"
    # end_date =   "2024-08-21 00:00:00"
    
    start_date = "2024-08-01 00:00:00"
    end_date =   "2024-10-01 00:00:00"

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
        export_quotes_path=f'quotes/'
    )


    # symbols = ['AAPL', 'GOOGL', 'NVDA']
    symbols = ['AAPL']
    
    retrieve_multi_symbol_data(params, symbols, sample_size=100)
    