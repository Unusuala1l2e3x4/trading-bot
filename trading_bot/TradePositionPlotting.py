from datetime import datetime, date, timedelta, time
from typing import List, Set, Tuple, Optional, Dict


import math
import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from numba import jit
import logging


from trading_bot.TradePosition import TradePosition



from zoneinfo import ZoneInfo
ny_tz = ZoneInfo("America/New_York")


import matplotlib.patches as mpatches

def is_trading_day(date: date):
    return date.weekday() < 5

def prepare_plotting_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[datetime.date], bool]:
    """Common data preparation for plotting."""
    timestamps = df.index.get_level_values('timestamp')
    df['time'] = timestamps.time
    df['date'] = timestamps.date

    # Identify trading days (days with price changes)
    trading_days = df.groupby('date').apply(lambda x: x['close'].nunique() > 1).reset_index()
    trading_days = set(trading_days[trading_days[0]]['date'])

    # Filter df to include only intraday data and trading days
    df_intraday = df[
        (df['time'] >= time(9, 30)) & 
        (df['time'] <= time(16, 0)) &
        (df['date'].isin(trading_days))
    ].copy()

    # Calculate volume data
    date_range = (df_intraday['date'].max() - df_intraday['date'].min()).days
    is_short_period = date_range <= 7
    
    if is_short_period:
        df_intraday['half_hour'] = df_intraday['time'].apply(
            lambda t: t.replace(minute=0 if t.minute < 30 else 30, second=0))
        volume_data = df_intraday.groupby(['date', 'half_hour'])['volume'].sum().reset_index()
        volume_data['datetime'] = volume_data.apply(
            lambda row: pd.Timestamp.combine(row['date'], row['half_hour']), axis=1)
        volume_data = volume_data.set_index('datetime').sort_index()
        volume_data = volume_data[volume_data.index.time != time(16, 0)]
    else:
        volume_data = df_intraday.groupby('date')['volume'].sum()

    # Create continuous index
    unique_dates = sorted(df_intraday['date'].unique())
    continuous_index = []
    cumulative_minutes = 0
    
    for date in unique_dates:
        day_data = df_intraday[df_intraday['date'] == date]
        day_minutes = day_data['time'].apply(lambda t: (t.hour - 9) * 60 + t.minute - 30)
        continuous_index.extend(cumulative_minutes + day_minutes)
        cumulative_minutes += 390

    df_intraday['continuous_index'] = continuous_index
    
    return df_intraday, volume_data, unique_dates, is_short_period

def collect_transaction_points(trades: List['TradePosition'], trading_days: set, use_plpc: bool = False,
                             use_transactions: bool = True) -> Dict[str, list]:
    """Collect data points from transactions."""
    point_times = []
    point_types = []  # 'entry', 'exit', or 'transaction'
    point_trades = []
    cumulative_pl = []
    cumulative_pl_longs = []
    cumulative_pl_shorts = []
    running_pl = 0
    running_pl_longs = 0
    running_pl_shorts = 0
    
    for trade in trades:
        # First add entry point (using previous cumulative values)
        if trade.actual_entry_time and trade.actual_entry_time.date() in trading_days:
            point_times.append(trade.actual_entry_time)
            point_types.append('entry')
            point_trades.append(trade)
            cumulative_pl.append(running_pl)
            cumulative_pl_longs.append(running_pl_longs)
            cumulative_pl_shorts.append(running_pl_shorts)
        
        if use_transactions:
            # Add points for each exit transaction
            for txn in trade.transactions:
                if txn.timestamp.date() in trading_days:
                    point_times.append(txn.timestamp)
                    point_types.append('transaction')
                    point_trades.append(trade)
                    # val = txn.plpc if use_plpc else txn.pl
                    val = txn.plpc_with_accum(trade.cost_basis_sold_accum) if use_plpc else txn.pl
                    running_pl += val
                    if trade.is_long:
                        running_pl_longs += val
                    else:
                        running_pl_shorts += val
                    cumulative_pl.append(running_pl)
                    cumulative_pl_longs.append(running_pl_longs)
                    cumulative_pl_shorts.append(running_pl_shorts)
        else:
            # Original logic for position-level granularity
            if trade.exit_time and trade.exit_time.date() in trading_days:
                point_times.append(trade.exit_time)
                point_types.append('exit')
                point_trades.append(trade)
                val = trade.plpc if use_plpc else trade.pl
                running_pl += val
                if trade.is_long:
                    running_pl_longs += val
                else:
                    running_pl_shorts += val
                cumulative_pl.append(running_pl)
                cumulative_pl_longs.append(running_pl_longs)
                cumulative_pl_shorts.append(running_pl_shorts)
                
        # Add final exit point if using transactions
        if use_transactions and trade.exit_time and trade.exit_time.date() in trading_days:
            point_times.append(trade.exit_time)
            point_types.append('exit')
            point_trades.append(trade)
            cumulative_pl.append(running_pl)
            cumulative_pl_longs.append(running_pl_longs)
            cumulative_pl_shorts.append(running_pl_shorts)
    
    return {
        'times': point_times,
        'types': point_types,
        'trades': point_trades,
        'pl': cumulative_pl,
        'pl_longs': cumulative_pl_longs,
        'pl_shorts': cumulative_pl_shorts
    }

def collect_snapshot_points(trades: List['TradePosition'], trading_days: set, use_plpc: bool = False) -> Dict[str, list]:
    """Collect data points from position metrics snapshots."""
    point_times = []
    point_types = []  # 'entry', 'snapshot', or 'exit'
    point_trades = []
    cumulative_pl = []
    cumulative_pl_longs = []
    cumulative_pl_shorts = []
    running_pl = 0
    running_pl_longs = 0
    running_pl_shorts = 0

    for trade in trades:
        # Get entry pl values to anchor snapshots
        entry_pl = running_pl
        entry_pl_longs = running_pl_longs
        entry_pl_shorts = running_pl_shorts

        # First add entry point (using previous cumulative values)
        if trade.actual_entry_time and trade.actual_entry_time.date() in trading_days:
            point_times.append(trade.actual_entry_time)
            point_types.append('entry')
            point_trades.append(trade)
            cumulative_pl.append(running_pl)
            cumulative_pl_longs.append(running_pl_longs)
            cumulative_pl_shorts.append(running_pl_shorts)

        # Add points for each snapshot 
        for snapshot in trade.position_metrics.snapshots:
            if snapshot.timestamp.date() in trading_days and \
                snapshot.timestamp > trade.actual_entry_time and \
                snapshot.timestamp < trade.exit_time: # use <= to include exit based on close if stop market order was not triggered
                # Anchor snapshot values to cumulative pl at entry
                # val = snapshot.running_plpc if use_plpc else snapshot.running_pl
                val = snapshot.running_plpc_with_accum(trade.cost_basis_sold_accum) if use_plpc else snapshot.running_pl
                point_times.append(snapshot.timestamp)
                point_types.append('snapshot')
                point_trades.append(trade)

                # Update running values 
                if trade.is_long:
                    snapshot_pl_longs = entry_pl_longs + val
                    snapshot_pl_shorts = entry_pl_shorts
                else:
                    snapshot_pl_longs = entry_pl_longs
                    snapshot_pl_shorts = entry_pl_shorts + val
                snapshot_pl = entry_pl + val
                
                cumulative_pl.append(snapshot_pl)
                cumulative_pl_longs.append(snapshot_pl_longs)
                cumulative_pl_shorts.append(snapshot_pl_shorts)

            # Update final cumulative values from last snapshot
            if trade.exit_time and snapshot.timestamp == trade.exit_time:
                val = trade.plpc if use_plpc else trade.pl
                point_times.append(trade.exit_time)
                point_types.append('exit')
                point_trades.append(trade)
                
                if trade.is_long:
                    running_pl_longs = entry_pl_longs + val
                    running_pl_shorts = entry_pl_shorts
                else:
                    running_pl_longs = entry_pl_longs
                    running_pl_shorts = entry_pl_shorts + val
                running_pl = entry_pl + val

                cumulative_pl.append(running_pl)
                cumulative_pl_longs.append(running_pl_longs)
                cumulative_pl_shorts.append(running_pl_shorts)

    return {
        'times': point_times,
        'types': point_types,
        'trades': point_trades,
        'pl': cumulative_pl,
        'pl_longs': cumulative_pl_longs,
        'pl_shorts': cumulative_pl_shorts
    }

def create_plot(df_intraday: pd.DataFrame, volume_data: pd.DataFrame, unique_dates: List[datetime.date], 
                is_short_period: bool, point_data: Dict[str, list], use_plpc: bool = False,
                show_position_markers: bool = True, when_above_max_investment: Optional[List[pd.Timestamp]] = None,
                trading_days: set = None) -> Tuple[plt.Figure, List[plt.Axes]]:
    """Create the plot using prepared data."""
    symbol = df_intraday.index.get_level_values('symbol')[0]
    continuous_points = []
    
    # Convert timestamps to continuous index
    for timestamp in point_data['times']:
        exit_date = timestamp.date()
        if exit_date in unique_dates:
            exit_minute = (timestamp.time().hour - 9) * 60 + (timestamp.time().minute - 30)
            days_passed = unique_dates.index(exit_date)
            continuous_points.append(days_passed * 390 + exit_minute)

    # Create figure and axes
    fig, ax1 = plt.subplots(figsize=(18, 10))
    
    # Plot close price
    ax1.plot(df_intraday['continuous_index'], df_intraday['close'], color='gray', label='Close Price')
    
    # Add points for when above max investment
    if when_above_max_investment and len(when_above_max_investment) > 0:
        above_max_continuous_index = []
        for timestamp in when_above_max_investment:
            if timestamp.date() in trading_days:
                date = timestamp.date()
                minute = (timestamp.time().hour - 9) * 60 + (timestamp.time().minute - 30)
                days_passed = unique_dates.index(date)
                above_max_continuous_index.append(days_passed * 390 + minute)
        min_close = df_intraday['close'].min()
        ax1.plot(above_max_continuous_index, [min_close] * len(above_max_continuous_index), 
                color='red', marker='o', linestyle='None', label='Above Max Investment')

    # Create P/L axis
    ax2 = ax1.twinx()
    
    if len(continuous_points) > 0:
        # Plot main P/L lines
        ax2.plot(continuous_points, point_data['pl'], color='green', label='All P/L', zorder=1)
        ax2.plot(continuous_points, point_data['pl_longs'], color='blue', label='Longs P/L', zorder=1)
        ax2.plot(continuous_points, point_data['pl_shorts'], color='yellow', label='Shorts P/L', zorder=1)
        
        if show_position_markers:
            # Separate indices by position side
            long_entry_indices = [i for i, (t, trade) in enumerate(zip(point_data['types'], point_data['trades'])) 
                                if t == 'entry' and trade.is_long]
            long_exit_indices = [i for i, (t, trade) in enumerate(zip(point_data['types'], point_data['trades'])) 
                            if t == 'exit' and trade.is_long]
            short_entry_indices = [i for i, (t, trade) in enumerate(zip(point_data['types'], point_data['trades'])) 
                                if t == 'entry' and not trade.is_long]
            short_exit_indices = [i for i, (t, trade) in enumerate(zip(point_data['types'], point_data['trades'])) 
                                if t == 'exit' and not trade.is_long]
            
            # Plot markers
            for indices, pl_data, is_entry in [
                (long_entry_indices, point_data['pl_longs'], True),
                (long_exit_indices, point_data['pl_longs'], False),
                (short_entry_indices, point_data['pl_shorts'], True),
                (short_exit_indices, point_data['pl_shorts'], False)
            ]:
                ax2.scatter(
                    [continuous_points[i] for i in indices],
                    [pl_data[i] for i in indices],
                    marker=3,
                    color='lime' if is_entry else 'red',
                    s=40,
                    zorder=2,
                    label='Entry' if is_entry else 'Exit'
                )

    # Create volume axis
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('axes', 1.1))
    
    # Plot volume
    if is_short_period:
        bar_width = 30
        for timestamp, row in volume_data.iterrows():
            date = timestamp.date()
            time = timestamp.time()
            days_passed = unique_dates.index(date)
            minutes = (time.hour - 9) * 60 + time.minute - 30
            x_position = days_passed * 390 + minutes
            ax3.bar(x_position, row['volume'], width=bar_width, alpha=0.3, color='purple', align='edge')
        volume_label = 'Half-hourly Mean Volume'
    else:
        bar_width = 390
        for i, (date, mean_volume) in enumerate(volume_data.items()):
            ax3.bar(i * 390, mean_volume, width=bar_width, alpha=0.3, color='purple', align='edge')
        volume_label = 'Daily Mean Volume'

    # Set labels and format axes
    title_str = 'Cumulative P/L % Change' if use_plpc else 'Cumulative P/L $'
    plt.suptitle(f'{symbol}: {title_str} vs Close Price and {volume_label}')

    # Set labels and format axes
    ax1.set_xlabel('Trading Time (minutes)')
    ax1.set_ylabel('Close Price', color='black')
    ax2.set_ylabel(f'{title_str}', color='black')
    ax3.set_ylabel(volume_label, color='purple')
    
    # Format tick parameters
    ax1.tick_params(axis='y', labelcolor='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax3.tick_params(axis='y', labelcolor='purple')
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

    # Set up legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax3_patch = mpatches.Patch(color='purple', alpha=0.3, label=volume_label)
    ax1.legend(lines1 + lines2 + [ax3_patch], labels1 + labels2 + [volume_label], loc='upper left')

    # Set up x-axis ticks and labels
    all_days = [date.strftime('%Y-%m-%d') for date in unique_dates]
    week_starts = []
    
    for i, date in enumerate(unique_dates):
        if i == 0 or date.weekday() < unique_dates[i-1].weekday():
            week_starts.append(i)

    major_ticks = [i * 390 for i in week_starts]
    all_ticks = list(range(0, len(unique_dates) * 390, 390))

    ax1.set_xticks(major_ticks)
    ax1.set_xticks(all_ticks, minor=True)

    if len(week_starts) < 5:
        ax1.set_xticklabels(all_days, minor=True, rotation=45, ha='right')
        ax1.tick_params(axis='x', which='minor', labelsize=8)
    else:
        ax1.set_xticklabels([], minor=True)

    ax1.set_xticklabels([all_days[i] for i in week_starts], rotation=45, ha='right')

    # Format ticks and grid
    ax1.tick_params(axis='x', which='minor', bottom=True)
    ax1.grid(which='major', axis='x', linestyle='--', alpha=0.75)
    ax1.grid(which='minor', axis='x', linestyle='--', alpha=0.35)
    
    plt.tight_layout()
    return fig, [ax1, ax2, ax3]

def plot_cumulative_pl_and_price(trades: List['TradePosition'], df: pd.DataFrame, initial_investment: float, 
                               when_above_max_investment: Optional[List[pd.Timestamp]]=None, 
                               filename: Optional[str]=None,
                               use_plpc: bool=False,
                               use_transactions: bool=True,
                               show_position_markers: bool=True):
    """Plot cumulative P/L based on trade transactions."""
    
    # Prepare data
    df_intraday, volume_data, unique_dates, is_short_period = prepare_plotting_data(df)
    trading_days = set(unique_dates)
    
    # Collect point data from transactions
    point_data = collect_transaction_points(trades, trading_days, use_plpc, use_transactions)
    
    # Create plot
    fig, axes = create_plot(df_intraday, volume_data, unique_dates, is_short_period, 
                          point_data, use_plpc, show_position_markers, 
                          when_above_max_investment, trading_days)
    if use_transactions:
        plt.title('Transactions level')
    else:
        plt.title('Positions level')
    plt.tight_layout()
    # Save or display
    if filename:
        if use_plpc and 'pl' in filename and 'plpc' not in filename:
            filename = filename.replace('pl','plpc')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300)
        print(f"Graph has been saved as {filename}")
    else:
        plt.show()
        
    plt.close()

def plot_cumulative_pl_and_price_from_snapshots(trades: List['TradePosition'], df: pd.DataFrame, 
                                             initial_investment: float, 
                                             when_above_max_investment: Optional[List[pd.Timestamp]]=None, 
                                             filename: Optional[str]=None,
                                             use_plpc: bool=False,
                                             show_position_markers: bool=True):
    """Plot cumulative P/L based on position metrics snapshots."""
    
    # Prepare data
    df_intraday, volume_data, unique_dates, is_short_period = prepare_plotting_data(df)
    trading_days = set(unique_dates)
    
    # Collect point data from snapshots
    point_data = collect_snapshot_points(trades, trading_days, use_plpc)
    
    # Create plot
    fig, axes = create_plot(df_intraday, volume_data, unique_dates, is_short_period, 
                          point_data, use_plpc, show_position_markers,
                          when_above_max_investment, trading_days)
    plt.title('Snapshots level')
    plt.tight_layout()
    # Save or display
    if filename:
        if use_plpc and 'pl' in filename and 'plpc' not in filename:
            filename = filename.replace('pl','plpc')
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300)
        print(f"Graph has been saved as {filename}")
    else:
        plt.show()
        
    plt.close()
    
    
    
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from scipy import stats

def extract_trade_data(trades: List['TradePosition'], 
                      x_field: str,
                      y_field: str = 'pl',
                      color_field: Optional[str] = None,
                      y_divisor_field: Optional[str] = None,
                      side: Optional[str] = None) -> pd.DataFrame:
    """
    Extract specified fields from trades into a DataFrame.
    
    Args:
        trades: List of TradePosition objects
        x_field: Field name to extract for x-axis
        y_field: Field name to extract for y-axis (default: 'pl')
        side: Optional filter for trade side ('long' or 'short')
        
    Returns:
        DataFrame with extracted fields
    """
    data = []
    
    for trade in trades:
        # Filter by side if specified
        if side == 'long' and not trade.is_long:
            continue
        if side == 'short' and trade.is_long:
            continue
            
        row = {'is_long': trade.is_long}
        
        def extract_field_value(field_name):
            if '.' in field_name:
                obj = trade
                for attr in field_name.split('.'):
                    obj = getattr(obj, attr)
                return obj
            return getattr(trade, field_name)
        
        # Extract values for x, y, and color fields
        x_value = extract_field_value(x_field)
        y_value = extract_field_value(y_field)
        if y_divisor_field:
            y_divisor_value = extract_field_value(y_divisor_field)
            y_value /= y_divisor_value
        if color_field:
            color_value = extract_field_value(color_field)
            row['color'] = color_value
            
        row['x'] = x_value
        row['y'] = y_value
        data.append(row)
        
    return pd.DataFrame(data)

def calculate_correlation_stats(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate correlation statistics between two variables.
    
    Returns:
        Tuple of (correlation coefficient, R-squared, p-value)
    """
    # Remove any NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    
    if len(x) < 2:
        return 0, 0, 1
        
    correlation, p_value = stats.pearsonr(x, y)
    r_squared = correlation ** 2
    
    return correlation, r_squared, p_value

def plot_trade_correlation(trades: List['TradePosition'],
                         x_field: str,
                         y_field: str = 'pl',
                         split_sides: bool = False,
                         figsize: Tuple[int, int] = (8,7),
                         x_label: Optional[str] = None,
                         y_label: Optional[str] = None,
                         title: Optional[str] = None,
                         binwidth_x: Optional[float] = None, 
                         binwidth_y: Optional[float] = None,
                         color_field: Optional[str] = None,
                         cmap: str = 'seismic_r',
                         center_colormap: Optional[float] = 0,
                         is_trinary: bool = False,
                         y_divisor_field: Optional[str] = None) -> None:
    """
    Create correlation plots for trade attributes.
    
    Args:
        trades: List of TradePosition objects
        x_field: Field name for x-axis
        y_field: Field name for y-axis (default: 'pl')
        split_sides: If True, create separate plots for long/short trades
        figsize: Figure size (width, height)
        x_label: Custom x-axis label
        y_label: Custom y-axis label
        title: Custom plot title
        binwidth_x: Custom bin width for x-axis histogram
        binwidth_y: Custom bin width for y-axis histogram
    """
    if split_sides:
        figsize = (figsize[0]*2, figsize[1])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        sides = ['long', 'short']
        axes = [ax1, ax2]
    else:
        fig, ax = plt.subplots(figsize=figsize)
        sides = [None]
        axes = [ax]
    
    
    dfs = []
        
    for side, ax in zip(sides, axes):
        # Extract data
        df = extract_trade_data(trades, x_field, y_field, color_field, y_divisor_field, side)
        if len(df) == 0:
            continue
            
        dfs.append((side, ax, df))
    
    df_all = pd.concat([a[2] for a in dfs])
    
    # Calculate global ranges and bins once for all plots
    x_data_clean = df_all['x'][~(np.isnan(df_all['x']) | np.isinf(df_all['x']))]
    y_data_clean = df_all['y'][~(np.isnan(df_all['y']) | np.isinf(df_all['y']))]
    
    # Handle time data conversion
    is_time_data = isinstance(df_all['x'].iloc[0], (time, pd.Timestamp))
    if is_time_data:
        if isinstance(df_all['x'].iloc[0], pd.Timestamp):
            x_data_minutes = x_data_clean.apply(lambda t: t.hour * 60 + t.minute)
        else:
            x_data_minutes = x_data_clean.apply(lambda t: t.hour * 60 + t.minute)
        start_minute = 570  # 9:30 AM
        x_range = x_data_minutes.max() - start_minute
        if binwidth_x is None:
            binwidth_x = x_range / 20
        num_bins = int(np.ceil(x_range / binwidth_x))
        bins_x = [start_minute + i * binwidth_x for i in range(num_bins + 1)]
        global_xlim = (start_minute, x_data_minutes.max())
    else:
        x_range = x_data_clean.max() - x_data_clean.min()
        if binwidth_x is None:
            binwidth_x = x_range / 20
            
        # Calculate global x bins
        if 0 <= x_data_clean.min() or 0 >= x_data_clean.max():
            num_bins = int(np.ceil(x_range / binwidth_x))
            if x_data_clean.min() >= 0:
                bins_x = [i * binwidth_x for i in range(num_bins + 1)]
            else:
                bins_x = [-i * binwidth_x for i in range(num_bins + 1)][::-1]
        else:
            pos_bins = np.arange(0, x_data_clean.max() + binwidth_x, binwidth_x)
            neg_bins = np.arange(0, x_data_clean.min() - binwidth_x, -binwidth_x)
            bins_x = np.concatenate([neg_bins[:-1][::-1], pos_bins])
        global_xlim = (x_data_clean.min(), x_data_clean.max())
            
    # Calculate global y bins
    y_range = y_data_clean.max() - y_data_clean.min()
    if binwidth_y is None:
        binwidth_y = y_range / 20
        
    if 0 <= y_data_clean.min() or 0 >= y_data_clean.max():
        num_bins = int(np.ceil(y_range / binwidth_y))
        if y_data_clean.min() >= 0:
            bins_y = [i * binwidth_y for i in range(num_bins + 1)]
        else:
            bins_y = [-i * binwidth_y for i in range(num_bins + 1)][::-1]
    else:
        pos_bins = np.arange(0, y_data_clean.max() + binwidth_y, binwidth_y)
        neg_bins = np.arange(0, y_data_clean.min() - binwidth_y, -binwidth_y)
        bins_y = np.concatenate([neg_bins[:-1][::-1], pos_bins])
        
    global_ylim = (y_data_clean.min(), y_data_clean.max())
    
    # Time formatter for x-axis if needed
    if is_time_data:
        def format_time(x, p):
            hours = int(x // 60)
            minutes = int(x % 60)
            return f"{hours:02d}:{minutes:02d}"

    for side, ax, df in dfs:
        if is_time_data:
            # Convert time to minutes for this subplot
            df['x_minutes'] = df['x'].apply(lambda t: t.hour * 60 + t.minute)
            del df['x']
            df.rename(columns={'x_minutes':'x'}, inplace=True)
            ax.xaxis.set_major_formatter(plt.FuncFormatter(format_time))

        x_data = df['x']
        
        # Create scatter plot with color mapping if specified
        if color_field:
            if pd.api.types.is_numeric_dtype(df['color']):
                if is_trinary and center_colormap is not None:
                    # Convert numeric to categorical based on comparison with center
                    def categorize(val):
                        if val > center_colormap:
                            return f'Above {center_colormap}'
                        elif val < center_colormap:
                            return f'Below {center_colormap}'
                        return f'Equal to {center_colormap}'
                    
                    df['color_cat'] = df['color'].map(categorize)
                    # Use categorical palette with meaningful colors
                    sns.scatterplot(data=df, x='x', y='y', hue='color_cat', 
                                  palette={f'Above {center_colormap}': 'green', f'Below {center_colormap}': 'red', f'Equal to {center_colormap}': 'gray'},
                                  ax=ax)
                else:
                    # For numeric fields, use continuous colormap
                    if center_colormap is not None:
                        # Sort by absolute distance from center so extreme values appear on top
                        df = df.copy()
                        df['dist_from_center'] = abs(df['color'] - center_colormap)
                        df = df.sort_values('dist_from_center')
                        
                        # Create diverging colormap centered at specified value
                        vmin = df['color'].min()
                        vmax = df['color'].max()
                        max_abs = max(abs(vmin - center_colormap), abs(vmax - center_colormap))
                        norm = plt.Normalize(center_colormap - max_abs, center_colormap + max_abs)
                        
                        # Choose appropriate colormap for centered data
                        if cmap == 'viridis':  # If default not changed, use better colormap for centered data
                            cmap = 'RdYlBu_r'  # Blue for negative, Red for positive
                    else:
                        # Sort by absolute value so extreme values appear on top
                        df = df.copy()
                        df = df.sort_values('color', key=abs)
                        norm = plt.Normalize(df['color'].min(), df['color'].max())
                        
                    scatter = ax.scatter(df['x'], df['y'], c=df['color'], cmap=cmap, norm=norm, alpha=0.8, edgecolor='gray', linewidth=1)
                    plt.colorbar(scatter, ax=ax, label=color_field)
            else:
                # For categorical fields, use discrete palette
                sns.scatterplot(data=df, x='x', y='y', hue='color', ax=ax)
        else:
            # Default behavior using is_long for color
            sns.scatterplot(data=df, x='x', y='y', 
                          hue='is_long' if side is None else None,
                          palette=['green', 'red'] if side is None else None,
                          ax=ax)
        
        # Add trend line
        x_values = x_data.values.reshape(-1, 1)
        y_values = df['y'].values
        
        if len(x_values) > 1:  # Need at least 2 points for regression
            model = stats.linregress(x_values.flatten(), y_values)
            line_x = np.array([x_values.min(), x_values.max()])
            line_y = model.slope * line_x + model.intercept
            ax.plot(line_x, line_y, color='blue', linestyle='--', alpha=0.5)
        
        # Calculate and display correlation statistics
        corr, r_squared, p_value = calculate_correlation_stats(x_values.flatten(), y_values)
        
        stats_text = f'Correlation: {corr:.3f}\nR²: {r_squared:.3f}\np-value: {p_value:.3f}'
        ax.text(0.05, 0.95, stats_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add reference lines at x=0 and y=0
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, zorder=0)
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5, zorder=0)

        # Create marginal axes
        divider = make_axes_locatable(ax)
        ax_histx = divider.append_axes("top", 1.2, pad=0.3)
        ax_histy = divider.append_axes("right", 1.2, pad=0.3)
        
        # Turn off marginal axes labels
        ax_histx.xaxis.set_tick_params(labelbottom=False)
        ax_histy.yaxis.set_tick_params(labelleft=False)
        
        # Determine appropriate bins for x-axis
        x_data_clean = x_data[~(np.isnan(x_data) | np.isinf(x_data))]
        x_range = x_data_clean.max() - x_data_clean.min()
        if binwidth_x is None:
            binwidth_x = x_range / 20  # default to 20 bins

        # For time data, start bins at 9:30 (570 minutes)
        if 'format_time' in locals():  # Check if we're dealing with time data
            start_minute = 570  # 9:30 AM
            num_bins = int(np.ceil((x_data_clean.max() - start_minute) / binwidth_x))
            bins_x = [start_minute + i * binwidth_x for i in range(num_bins + 1)]
            ax.set_xlim((start_minute, ax.get_xlim()[1]))
        else:
            # For numeric data, ensure zero is at bin edge
            if 0 <= x_data_clean.min() or 0 >= x_data_clean.max():  # All positive or all negative
                num_bins = int(np.ceil(x_range / binwidth_x))
                if x_data_clean.min() >= 0:
                    bins_x = [i * binwidth_x for i in range(num_bins + 1)]
                else:
                    bins_x = [-i * binwidth_x for i in range(num_bins + 1)][::-1]
            else:  # Data crosses zero
                pos_bins = np.arange(0, x_data_clean.max() + binwidth_x, binwidth_x)
                neg_bins = np.arange(0, x_data_clean.min() - binwidth_x, -binwidth_x)
                bins_x = np.concatenate([neg_bins[:-1][::-1], pos_bins])

        # Similar logic for y-axis bins
        y_data_clean = df['y'][~(np.isnan(df['y']) | np.isinf(df['y']))]
        y_range = y_data_clean.max() - y_data_clean.min()
        if binwidth_y is None:
            binwidth_y = y_range / 20  # default to 20 bins
            
        if 0 <= y_data_clean.min() or 0 >= y_data_clean.max():  # All positive or all negative
            num_bins = int(np.ceil(y_range / binwidth_y))
            if y_data_clean.min() >= 0:
                bins_y = [i * binwidth_y for i in range(num_bins + 1)]
            else:
                bins_y = [-i * binwidth_y for i in range(num_bins + 1)][::-1]
        else:  # Data crosses zero
            pos_bins = np.arange(0, y_data_clean.max() + binwidth_y, binwidth_y)
            neg_bins = np.arange(0, y_data_clean.min() - binwidth_y, -binwidth_y)
            bins_y = np.concatenate([neg_bins[:-1][::-1], pos_bins])
            
        # Plot histograms with calculated bins
        sns.histplot(x=x_data, ax=ax_histx, bins=bins_x, color='blue', alpha=0.3)
        if 'format_time' in locals():
            ax_histx.xaxis.set_major_formatter(plt.FuncFormatter(format_time))
            
        sns.histplot(y=df['y'], ax=ax_histy, bins=bins_y, color='blue', alpha=0.3)
        
        # Match axes limits for both plots
        ax.set_xlim(global_xlim)
        ax.set_ylim(global_ylim)
        
        # Labels and title
        ax.set_xlabel(x_label or x_field)
        ax.set_ylabel(y_label or y_field)
        if side:
            ax.set_title(f'{title or ""} ({side.capitalize()} Trades)')
        else:
            ax.set_title(title or f'{x_field} vs {y_field}')
            
        # Add reference lines at x=0 and y=0
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, zorder=0)
        ax.axvline(x=0, color='gray', linestyle='-', alpha=0.5, zorder=0)

        # Create marginal axes
        divider = make_axes_locatable(ax)
        ax_histx = divider.append_axes("top", 1.2, pad=0.3)
        ax_histy = divider.append_axes("right", 1.2, pad=0.3)
        
        # Turn off marginal axes labels
        ax_histx.xaxis.set_tick_params(labelbottom=False)
        ax_histy.yaxis.set_tick_params(labelleft=False)
        
        # Plot histograms with calculated bins
        if is_time_data:
            sns.histplot(x=x_data, ax=ax_histx, bins=bins_x, color='blue', alpha=0.3)
            ax_histx.xaxis.set_major_formatter(plt.FuncFormatter(format_time))
        else:
            sns.histplot(x=x_data, ax=ax_histx, bins=bins_x, color='blue', alpha=0.3)
            
        sns.histplot(y=df['y'], ax=ax_histy, bins=bins_y, color='blue', alpha=0.3)
        
        # Match marginal axes limits
        ax_histx.set_xlim(global_xlim)
        ax_histy.set_ylim(global_ylim)
            
    plt.tight_layout()
    plt.show()