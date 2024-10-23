import pandas as pd
import numpy as np
from typing import Tuple
from zoneinfo import ZoneInfo

ny_tz = ZoneInfo("America/New_York")

def clean_quotes_data(df: pd.DataFrame, interval_start: pd.Timestamp, interval_end: pd.Timestamp) -> Tuple[pd.DataFrame, float]:
    """
    Clean quotes data and calculate intra-timestamp changes with improved efficiency.
    """
    if df.empty:
        return df, (interval_end - interval_start).total_seconds()

    # Drop unnecessary columns
    df = df.drop(columns=['bid_exchange', 'ask_exchange', 'conditions', 'tape'], errors='ignore')
    
    # Sort the dataframe to ensure correct diff calculation
    df = df.sort_values(['timestamp', 'symbol']).reset_index(drop=True)
    
    # Efficient calculation of intra-timestamp changes
    for col in ['bid_price', 'ask_price', 'bid_size', 'ask_size']:
        df[f'{col}_intra_change'] = df.groupby('timestamp')[col].diff()
        df[f'{col}_intra_pos'] = df[f'{col}_intra_change'].clip(lower=0)
        df[f'{col}_intra_neg'] = df[f'{col}_intra_change'].clip(upper=0)

    # Optimized aggregation
    agg_dict = {
        'bid_price': ['first', 'last', 'min', 'max'],
        'ask_price': ['first', 'last', 'min', 'max'],
        'bid_size': ['first', 'last', 'min', 'max'],
        'ask_size': ['first', 'last', 'min', 'max'],
        'bid_price_intra_pos': 'sum',
        'bid_price_intra_neg': 'sum',
        'ask_price_intra_pos': 'sum',
        'ask_price_intra_neg': 'sum',
        'bid_size_intra_pos': 'sum',
        'bid_size_intra_neg': 'sum',
        'ask_size_intra_pos': 'sum',
        'ask_size_intra_neg': 'sum'
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
    timestamps = df.index.get_level_values('timestamp')
    df['duration'] = (timestamps.shift(-1) - timestamps).fillna(pd.Timedelta(seconds=0)).dt.total_seconds()
    df.loc[df.index[-1], 'duration'] = (interval_end - timestamps[-1]).total_seconds()
    df = df[df['duration'] > 0]
    
    carryover = max(0, (timestamps[0] - interval_start).total_seconds())
    
    return df, carryover

def calculate_window_features(window_df: pd.DataFrame) -> pd.Series:
    """
    Calculate features from quote data window with enhanced handling of intra-timestamp changes and improved efficiency.
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
        price_intra_pos = window_df[f'{side}_price_intra_pos_sum_sum']
        price_intra_neg = window_df[f'{side}_price_intra_neg_sum_sum']
        
        total_price_increases = price_changes[price_changes > 0].sum() + price_intra_pos.sum()
        total_price_decreases = price_changes[price_changes < 0].sum() + price_intra_neg.sum()
        net_price_change = window_df[price_col].iloc[-1] - window_df[price_col].iloc[0]
        
        # Avoid division by zero in volatility ratio
        price_volatility_ratio = (abs(total_price_increases) + abs(total_price_decreases)) / max(abs(net_price_change), 1e-8)
        
        # Price regression
        times = window_df['time_delta']
        prices = window_df[price_col]
        price_slope, _ = np.polyfit(times, prices, 1) if len(window_df) >= 2 else (0, 0)
        
        # Price acceleration
        price_velocity = price_changes / window_df['duration'].replace(0, np.nan).fillna(window_df['duration'].mean())
        price_acceleration = price_velocity.diff().mean()
        
        features.update({
            f'{side}_price_total_increases': total_price_increases,
            f'{side}_price_total_decreases': total_price_decreases,
            f'{side}_price_volatility_ratio': price_volatility_ratio,
            f'{side}_price_slope': price_slope,
            f'{side}_price_acceleration': price_acceleration,
        })
        
        # Size analysis (vectorized)
        size_changes = window_df[size_col].diff().fillna(0)
        intra_pos = window_df[f'{side}_size_intra_pos_sum_sum']
        size_replenishment = (size_changes.clip(lower=0) + intra_pos).fillna(0)
        
        size_replenishment_rate = size_replenishment / window_df['duration'].replace(0, np.nan).fillna(window_df['duration'].mean())
        size_slope, _ = np.polyfit(times, size_replenishment_rate, 1) if len(window_df) >= 2 else (0, 0)
        
        features.update({
            f'{side}_size_replenishment_slope': size_slope,
            f'{side}_size_replenishment_total': size_replenishment.sum(),
        })
        
        # Price-replenishment correlation
        if len(window_df) >= 2:
            price_pct_change = price_changes.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
            size_pct_change = size_replenishment_rate.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
            correlation, _ = np.polyfit(times[1:], (price_pct_change - size_pct_change)[1:], 1)
            features[f'{side}_price_replenishment_divergence_slope'] = correlation
    
    # Spread dynamics
    spreads = window_df['ask_price_last'] - window_df['bid_price_last']
    spread_changes = spreads.diff().fillna(0)
    spread_range = spreads.max() - spreads.min()
    features.update({
        'spread_volatility_ratio': spread_changes.abs().sum() / max(spread_range, 1e-8),
        'spread_slope': np.polyfit(times, spreads, 1)[0] if len(window_df) >= 2 else 0,
        'spread_mean': spreads.mean(),
        'spread_std': spreads.std(),
    })

    return pd.Series(features)