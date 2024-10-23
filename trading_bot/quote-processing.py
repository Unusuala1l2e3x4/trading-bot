import pandas as pd
import numpy as np
from typing import Tuple
from zoneinfo import ZoneInfo

ny_tz = ZoneInfo("America/New_York")

def clean_quotes_data(df: pd.DataFrame, interval_start: pd.Timestamp, interval_end: pd.Timestamp) -> Tuple[pd.DataFrame, float]:
    """
    Clean quotes data and calculate intra-timestamp changes.
    """
    if df.empty:
        return df, (interval_end - interval_start).total_seconds()

    # Drop unnecessary columns
    df = df.drop(columns=['bid_exchange', 'ask_exchange', 'conditions', 'tape'], errors='ignore')
    
    # Create sequential index within each timestamp to track intra-timestamp changes
    df = df.reset_index()
    df['seq_num'] = df.groupby('timestamp').cumcount()
    
    # Calculate intra-timestamp changes
    for col in ['bid_price', 'ask_price', 'bid_size', 'ask_size']:
        # Calculate changes within timestamps
        df[f'{col}_intra_change'] = df.groupby('timestamp')[col].diff()
        
        # Sum of positive/negative changes within timestamps
        pos_changes = df[f'{col}_intra_change'].copy()
        neg_changes = df[f'{col}_intra_change'].copy()
        pos_changes[pos_changes <= 0] = 0
        neg_changes[neg_changes >= 0] = 0
        
        df[f'{col}_intra_pos_sum'] = df.groupby('timestamp')[pos_changes].transform('sum')
        df[f'{col}_intra_neg_sum'] = df.groupby('timestamp')[neg_changes].transform('sum')
        
    # Aggregate with the new metrics
    agg_dict = {
        'bid_price': ['first', 'last', 'min', 'max'],
        'ask_price': ['first', 'last', 'min', 'max'],
        'bid_size': ['first', 'last', 'min', 'max'],
        'ask_size': ['first', 'last', 'min', 'max'],
        'bid_price_intra_pos_sum': 'first',
        'bid_price_intra_neg_sum': 'first',
        'ask_price_intra_pos_sum': 'first',
        'ask_price_intra_neg_sum': 'first',
        'bid_size_intra_pos_sum': 'first',
        'bid_size_intra_neg_sum': 'first',
        'ask_size_intra_pos_sum': 'first',
        'ask_size_intra_neg_sum': 'first'
    }
    
    df = df.groupby(['symbol', 'timestamp']).agg(agg_dict)
    df.columns = ['_'.join(col).strip() for col in df.columns.values]
    
    # Handle timezone and sorting
    timestamps = df.index.get_level_values('timestamp')
    df.index = df.index.set_levels(
        timestamps.tz_localize(ny_tz) if timestamps.tz is None else timestamps.tz_convert(ny_tz),
        level='timestamp'
    )
    df.sort_index(level='timestamp', inplace=True)
    
    # Calculate durations
    timestamps = df.index.get_level_values('timestamp').to_series()
    df['duration'] = (timestamps.shift(-1) - timestamps).fillna(interval_end - timestamps.iloc[-1]).dt.total_seconds()
    df = df[df['duration'] > 0]
    
    carryover = (timestamps.iloc[0] - interval_start).total_seconds()
    
    return df, carryover

def calculate_window_features(window_df: pd.DataFrame) -> pd.Series:
    """
    Calculate features from quote data window with enhanced handling of intra-timestamp changes.
    """
    features = {}
    if len(window_df) < 2:
        return pd.Series(features)

    window_df = window_df.reset_index()
    window_df['time_delta'] = (window_df['timestamp'] - window_df['timestamp'].iloc[0]).dt.total_seconds()
    total_duration = window_df['duration'].sum()

    for side in ['bid', 'ask']:
        price_col = f'{side}_price_last'
        size_col = f'{side}_size_last'
        
        # Price analysis
        price_changes = window_df[price_col].diff().fillna(0)
        price_intra_pos = window_df[f'{side}_price_intra_pos_sum_first']
        price_intra_neg = window_df[f'{side}_price_intra_neg_sum_first']
        
        # Total price changes including intra-timestamp
        total_price_increases = price_changes[price_changes > 0].sum() + price_intra_pos.sum()
        total_price_decreases = price_changes[price_changes < 0].sum() + price_intra_neg.sum()
        net_price_change = window_df[price_col].iloc[-1] - window_df[price_col].iloc[0]
        
        # Price volatility ratio (total movement vs net change)
        price_volatility_ratio = (abs(total_price_increases) + abs(total_price_decreases)) / (abs(net_price_change) if net_price_change != 0 else 1)
        
        # Price movement regression
        times = window_df['time_delta']
        prices = window_df[price_col]
        price_slope = np.polyfit(times, prices, 1)[0] if len(window_df) >= 2 else 0
        
        # Price acceleration (second derivative approximation)
        price_velocity = price_changes / window_df['duration']
        price_acceleration = price_velocity.diff().mean() if len(window_df) >= 3 else 0
        
        features.update({
            f'{side}_price_total_increases': total_price_increases,
            f'{side}_price_total_decreases': total_price_decreases,
            f'{side}_price_volatility_ratio': price_volatility_ratio,
            f'{side}_price_slope': price_slope,
            f'{side}_price_acceleration': price_acceleration,
        })
        
        # Size analysis
        size_replenishment = pd.Series(0, index=window_df.index)
        for i in range(1, len(window_df)):
            intra_pos = window_df[f'{side}_size_intra_pos_sum_first'].iloc[i]
            size_change = window_df[size_col].iloc[i] - window_df[size_col].iloc[i-1]
            size_replenishment.iloc[i] = max(0, size_change) + max(0, intra_pos)
        
        # Size replenishment regression
        size_replenishment_rate = size_replenishment / window_df['duration']
        size_slope = np.polyfit(times, size_replenishment_rate, 1)[0] if len(window_df) >= 2 else 0
        
        features.update({
            f'{side}_size_replenishment_slope': size_slope,
            f'{side}_size_replenishment_total': size_replenishment.sum(),
        })
        
        # Price-replenishment correlation
        if len(window_df) >= 2:
            price_pct_change = price_changes / window_df[price_col].shift(1)
            size_pct_change = size_replenishment_rate / size_replenishment_rate.shift(1)
            correlation = np.polyfit(times[1:], (price_pct_change - size_pct_change)[1:], 1)[0]
            features[f'{side}_price_replenishment_divergence_slope'] = correlation
    
    # Spread dynamics
    spreads = window_df['ask_price_last'] - window_df['bid_price_last']
    spread_changes = spreads.diff().fillna(0)
    features.update({
        'spread_volatility_ratio': spread_changes.abs().sum() / (spreads.max() - spreads.min() if spreads.max() != spreads.min() else 1),
        'spread_slope': np.polyfit(times, spreads, 1)[0] if len(window_df) >= 2 else 0,
    })

    return pd.Series(features)