
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, time as time2
from typing import List, Tuple, Optional
from TouchArea import TouchArea
import math
import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from numba import jit

import logging

SEC_FEE_RATE = 0.0000278  # $27.80 per $1,000,000
FINRA_TAF_RATE = 0.000166  # $166 per 1,000,000 shares
FINRA_TAF_MAX = 8.30  # Maximum $8.30 per trade

@dataclass
class Transaction:
    timestamp: datetime
    shares: int
    price: float
    is_entry: bool
    is_long: bool
    transaction_cost: float
    finra_taf: float
    sec_fee: float
    stock_borrow_cost: float
    value: float
    vwap: float
    realized_pl: Optional[float] = None

@dataclass
class TradePosition:
    date: date
    id: int
    area: TouchArea
    is_long: bool
    entry_time: datetime
    initial_balance: float  # Cash used after margin applied
    initial_shares: int
    use_margin: bool
    is_marginable: bool
    entry_price: float
    market_value: float = 0.0
    shares: int = 0
    partial_entry_count: int = 0
    partial_exit_count: int = 0
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    transactions: List[Transaction] = field(default_factory=list)
    current_stop_price: Optional[float] = None
    max_price: Optional[float] = None
    min_price: Optional[float] = None
    last_price: float = field(default=0.0)
    unrealized_pl: float = field(default=0.0)
    realized_pl: float = 0.0
    log_level: Optional[int] = logging.INFO
    stock_borrow_rate: float = 0.03

    def __post_init__(self):
        self.market_value = 0
        self.shares = 0
        self.last_price = self.entry_price
        self.logger = self.setup_logger(logging.WARNING)

    def setup_logger(self, log_level=logging.INFO):
        logger = logging.getLogger('TradePosition')
        logger.setLevel(log_level)
        if logger.hasHandlers():
            logger.handlers.clear()
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def log(self, message, level=logging.INFO):
        self.logger.log(level, message)

    def calculate_slippage(self, price: float, trade_size: int, volume: float, avg_volume: float, slippage_factor: float, is_entry: bool) -> float:
        effective_volume = max((volume + avg_volume) / 2, 1)
        slippage = slippage_factor * (float(trade_size) / effective_volume)
        
        if self.is_long:
            return price * (1 + slippage) if is_entry else price * (1 - slippage)
        else:
            return price * (1 - slippage) if is_entry else price * (1 + slippage)

    def update_market_value(self, current_price: float):
        self.last_price = current_price
        self.market_value = self.shares * current_price if self.is_long else -self.shares * current_price
        self.unrealized_pl = self.market_value - (self.shares * self.entry_price)

    def calculate_transaction_cost(self, shares: int, price: float, is_entry: bool, timestamp: datetime) -> tuple:
        is_sell = (self.is_long and not is_entry) or (not self.is_long and is_entry)
        finra_taf = min(FINRA_TAF_RATE * shares, FINRA_TAF_MAX) if is_sell else 0
        trade_value = price * shares
        sec_fee = SEC_FEE_RATE * trade_value if is_sell else 0
        
        stock_borrow_cost = 0
        if not self.is_long and not is_entry:
            daily_borrow_rate = self.stock_borrow_rate / 360
            holding_time = timestamp - self.entry_time
            days_held = holding_time.total_seconds() / (24 * 60 * 60)
            stock_borrow_cost = shares * price * daily_borrow_rate * days_held

        return finra_taf, sec_fee, stock_borrow_cost

    def add_transaction(self, timestamp: datetime, shares: int, price: float, is_entry: bool, vwap: float):
        finra_taf, sec_fee, stock_borrow_cost = self.calculate_transaction_cost(shares, price, is_entry, timestamp)
        transaction_cost = finra_taf + sec_fee + stock_borrow_cost
        value = -shares * price if is_entry else shares * price
        
        realized_pl = None
        if not is_entry:
            realized_pl = (price - self.entry_price) * shares if self.is_long else (self.entry_price - price) * shares
            self.realized_pl += realized_pl
        
        transaction = Transaction(timestamp, shares, price, is_entry, self.is_long, transaction_cost, 
                                  finra_taf, sec_fee, stock_borrow_cost, value, vwap, realized_pl)
        self.transactions.append(transaction)
        
        self.log(f"Transaction added - {'Entry' if is_entry else 'Exit'}, Shares: {shares}, Price: {price:.4f}, "
                 f"Value: {value:.4f}, Cost: {transaction_cost:.4f}, Realized PnL: {realized_pl if realized_pl is not None else 'N/A'}")

        return transaction_cost

    def partial_entry(self, entry_time: datetime, entry_price: float, shares_to_add: int, vwap: float, volume: float, avg_volume: float, slippage_factor: float):
        self.log(f"Partial entry - Time: {entry_time}, Price: {entry_price:.4f}, Shares to add: {shares_to_add}")

        adjusted_price = self.calculate_slippage(entry_price, shares_to_add, volume, avg_volume, slippage_factor, is_entry=True)
        
        transaction_cost = self.add_transaction(entry_time, shares_to_add, adjusted_price, is_entry=True, vwap=vwap)
        
        self.shares += shares_to_add
        self.update_market_value(adjusted_price)
        self.partial_entry_count += 1

        self.log(f"Partial entry complete - New total shares: {self.shares}, New market value: {self.market_value:.4f}")
        
        return transaction_cost

    def partial_exit(self, exit_time: datetime, exit_price: float, shares_to_remove: int, vwap: float, volume: float, avg_volume: float, slippage_factor: float):
        self.log(f"Partial exit - Time: {exit_time}, Price: {exit_price:.4f}, Shares to remove: {shares_to_remove}")

        adjusted_price = self.calculate_slippage(exit_price, shares_to_remove, volume, avg_volume, slippage_factor, is_entry=False)
        
        transaction_cost = self.add_transaction(exit_time, shares_to_remove, adjusted_price, is_entry=False, vwap=vwap)
        
        self.shares -= shares_to_remove
        self.update_market_value(adjusted_price)
        self.partial_exit_count += 1

        self.log(f"Partial exit complete - New shares: {self.shares}, New market value: {self.market_value:.4f}")

        return transaction_cost

    def update_stop_price(self, current_price: float, current_timestamp: datetime):
        self.area.update_bounds(current_timestamp)
        
        if self.is_long:
            self.max_price = max(self.max_price or self.entry_price, current_price)
            self.current_stop_price = self.max_price - self.area.get_range
        else:
            self.min_price = min(self.min_price or self.entry_price, current_price)
            self.current_stop_price = self.min_price + self.area.get_range
        
        self.update_market_value(current_price)
        self.log(f"Area {self.area.id}: get_range {self.area.get_range:.4f}")
        return self.should_exit(current_price)

    def should_exit(self, current_price: float) -> bool:
        return (self.is_long and current_price <= self.current_stop_price) or \
               (not self.is_long and current_price >= self.current_stop_price)

    def close(self, exit_time: datetime, exit_price: float):
        self.exit_time = exit_time
        self.exit_price = exit_price

    @property
    def total_stock_borrow_cost(self) -> float:
        return sum(t.stock_borrow_cost for t in self.transactions if not t.is_entry) if not self.is_long else 0.0

    @property
    def holding_time(self) -> timedelta:
        return (self.exit_time or datetime.now()) - self.entry_time

    @property
    def entry_transaction_costs(self) -> float:
        return sum(t.transaction_cost for t in self.transactions if t.is_entry)

    @property
    def exit_transaction_costs(self) -> float:
        return sum(t.transaction_cost for t in self.transactions if not t.is_entry)

    @property
    def total_transaction_costs(self) -> float:
        return sum(t.transaction_cost for t in self.transactions)

    @property
    def pl(self) -> float:
        return self.unrealized_pl + self.realized_pl - self.total_transaction_costs

    @property
    def plpc(self) -> float:
        return (self.pl / self.initial_balance) * 100
    
    
    
    


def export_trades_to_csv(trades: List[TradePosition], filename: str):
    """
    Export the trades data to a CSV file using pandas.
    
    Args:
    trades (list): List of TradePosition objects
    filename (str): Name of the CSV file to be created
    """
    data = []
    cumulative_pct_change = 0
    for trade in trades:
        cumulative_pct_change += trade.plpc
        row = {
            'date': trade.date,
            'ID': trade.id,
            'AreaID': trade.area.id,
            'Type': 'Long' if trade.is_long else 'Short',
            'Entry Time': trade.entry_time.time().strftime('%H:%M:%S'),
            'Exit Time': trade.exit_time.time().strftime('%H:%M:%S') if trade.exit_time else None,
            'Holding Time (min)': trade.holding_time.total_seconds() / 60,
            'Entry Price': trade.entry_price,
            'Exit Price': trade.exit_price if trade.exit_price else None,
            'Initial Shares': trade.initial_shares,
            'Realized P/L': trade.get_realized_pl,
            'Unrealized P/L': trade.get_unrealized_pl,
            'Total P/L': trade.pl,
            'ROE (P/L %)': trade.plpc,
            'Cumulative P/L %': cumulative_pct_change,
            'Transaction Costs': trade.total_transaction_costs,
            'Times Buying Power': trade.times_buying_power
        }
        data.append(row)
    df = pd.DataFrame(data)
    if len(os.path.dirname(filename)) > 0:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"Trade summary has been exported to {filename}",level=logging.DEBUG)


def time_to_minutes(t: time2):
    return t.hour * 60 + t.minute - (9 * 60 + 30)

@dataclass
class SimplifiedTradePosition:
    date: date
    id: int
    area_id: int
    is_long: bool
    entry_time: datetime
    exit_time: datetime
    holding_time: timedelta
    entry_price: float
    exit_price: float
    initial_shares: int
    realized_pl: float
    unrealized_pl: float
    pl: float
    plpc: float
    cumulative_pct_change: float
    total_transaction_costs: float
    times_buying_power: float

def csv_to_trade_positions(csv_file_path) -> List[SimplifiedTradePosition]:
    df = pd.read_csv(csv_file_path)
    trade_positions = []
    
    for _, row in df.iterrows():
        trade_date = datetime.strptime(row['date'], '%Y-%m-%d').date()
        
        entry_time = datetime.combine(
            trade_date, 
            datetime.strptime(row['Entry Time'], '%H:%M:%S').time()
        )
        
        exit_time = None
        if pd.notna(row['Exit Time']):
            exit_time = datetime.combine(
                trade_date, 
                datetime.strptime(row['Exit Time'], '%H:%M:%S').time()
            )
            # Handle cases where exit time is on the next day
            if exit_time < entry_time:
                exit_time += timedelta(days=1)
        
        holding_time = timedelta(seconds=row['Holding Time (min)'] * 60)
        
        trade_position = SimplifiedTradePosition(
            date=trade_date,
            id=row['ID'],
            area_id=row['AreaID'],
            is_long=(row['Type'] == 'Long'),
            entry_time=entry_time,
            exit_time=exit_time,
            holding_time=holding_time,
            entry_price=row['Entry Price'],
            exit_price=row['Exit Price'] if pd.notna(row['Exit Price']) else None,
            initial_shares=row['Initial Shares'],
            realized_pl=row['Realized P/L'],
            unrealized_pl=row['Unrealized P/L'],
            pl=row['Total P/L'],
            plpc=row['ROE (P/L %)'],
            cumulative_pct_change=row['Cumulative P/L %'],
            total_transaction_costs=row['Transaction Costs'],
            times_buying_power=row['Times Buying Power']
        )
        trade_positions.append(trade_position)
    
    return trade_positions

# import matplotlib.patches as mpatches
# def plot_cumulative_pl_and_price(trades: List[TradePosition | SimplifiedTradePosition], df: pd.DataFrame, initial_investment: float, when_above_max_investment: Optional[List[pd.Timestamp]]=None, filename: Optional[str]=None):
import matplotlib.patches as mpatches
# import numpy as np

def is_trading_day(date: date):
    return date.weekday() < 5

def plot_cumulative_pl_and_price(trades: List[TradePosition | SimplifiedTradePosition], df: pd.DataFrame, initial_investment: float, when_above_max_investment: Optional[List[pd.Timestamp]]=None, filename: Optional[str]=None):
    """
    Create a graph that plots the cumulative profit/loss (summed percentages) at each corresponding exit time
    overlaid on the close price and volume from the DataFrame, using a triple y-axis.
    Volume is grouped by day for periods longer than a week, and by half-hour for periods of a week or less.
    Empty intervals before and after intraday trading are removed.
    
    Args:
    trades (list): List of TradePosition objects
    df (pd.DataFrame): DataFrame containing the price and volume data
    when_above_max_investment (list): List of timestamps when investment is above max
    filename (str): Name of the image file to be created
    """
    
    symbol = df.index.get_level_values('symbol')[0]
    
    timestamps = df.index.get_level_values('timestamp')
    df['time'] = timestamps.time
    df['date'] = timestamps.date

    # Identify trading days (days with price changes)
    trading_days = df.groupby('date').apply(lambda x: x['close'].nunique() > 1).reset_index()
    trading_days = set(trading_days[trading_days[0]]['date'])

    # Filter df to include only intraday data and trading days
    df_intraday = df[
        (df['time'] >= time2(9, 30)) & 
        (df['time'] <= time2(16, 0)) &
        (df['date'].isin(trading_days))
    ].copy()

    timestamps = df_intraday.index.get_level_values('timestamp')
    
    # Determine if the data spans more than a week
    date_range = (df_intraday['date'].max() - df_intraday['date'].min()).days
    is_short_period = date_range <= 7
    
    if is_short_period:
        # Group by half-hour intervals
        df_intraday['half_hour'] = df_intraday['time'].apply(lambda t: t.replace(minute=0 if t.minute < 30 else 30, second=0))
        volume_data = df_intraday.groupby(['date', 'half_hour'])['volume'].mean().reset_index()
        volume_data['datetime'] = volume_data.apply(lambda row: pd.Timestamp.combine(row['date'], row['half_hour']), axis=1)
        volume_data = volume_data.set_index('datetime').sort_index()
        volume_data = volume_data[volume_data.index.time != time2(16, 0)]
    else:
        # Group by day
        volume_data = df_intraday.groupby('date')['volume'].mean()
    
    # Create a continuous index
    unique_dates = sorted(df_intraday['date'].unique())
    continuous_index = []
    cumulative_minutes = 0
    
    for date in unique_dates:
        day_data = df_intraday[df_intraday['date'] == date]
        day_minutes = day_data['time'].apply(lambda t: (t.hour - 9) * 60 + t.minute - 30)
        continuous_index.extend(cumulative_minutes + day_minutes)
        cumulative_minutes += 390  # 6.5 hours of trading
    
    df_intraday['continuous_index'] = continuous_index
    
    # Prepare data for plotting
    exit_times = []
    cumulative_pl = []
    cumulative_pl_longs = []
    cumulative_pl_shorts = []
    running_pl = 0
    running_pl_longs = 0
    running_pl_shorts = 0
    
    for trade in trades:
        if trade.exit_time and trade.exit_time.date() in trading_days:
            exit_times.append(trade.exit_time)
            running_pl += trade.plpc
            if trade.is_long:
                running_pl_longs += trade.plpc
            else:
                running_pl_shorts += trade.plpc
            cumulative_pl.append(running_pl)
            cumulative_pl_longs.append(running_pl_longs)
            cumulative_pl_shorts.append(running_pl_shorts)

    # Convert exit times to continuous index
    exit_continuous_index = []
    for exit_time in exit_times:
        exit_date = exit_time.date()
        if exit_date in unique_dates:
            exit_minute = (exit_time.time().hour - 9) * 60 + (exit_time.time().minute - 30)
            days_passed = unique_dates.index(exit_date)
            exit_continuous_index.append(days_passed * 390 + exit_minute)

    # Ensure all arrays have the same length
    min_length = min(len(exit_continuous_index), len(cumulative_pl), len(cumulative_pl_longs), len(cumulative_pl_shorts))
    exit_continuous_index = exit_continuous_index[:min_length]
    cumulative_pl = cumulative_pl[:min_length]
    cumulative_pl_longs = cumulative_pl_longs[:min_length]
    cumulative_pl_shorts = cumulative_pl_shorts[:min_length]

    # Create figure and primary y-axis
    fig, ax1 = plt.subplots(figsize=(18, 10))
    
    # Plot close price on primary y-axis
    ax1.plot(df_intraday['continuous_index'], df_intraday['close'], color='gray', label='Close Price')
    
    # Add red dots for the start of each trading day
    day_start_indices = []
    day_start_prices = []
    day_start_pl = []
    day_start_pl_longs = []
    day_start_pl_shorts = []
    
    for date in unique_dates:
        day_data = df_intraday[df_intraday['date'] == date]
        if not day_data.empty:
            start_index = day_data['continuous_index'].iloc[0]
            start_price = day_data['close'].iloc[0]
            day_start_indices.append(start_index)
            day_start_prices.append(start_price)
            
            # Find the closest P/L values for this day start
            closest_index = min(range(len(exit_continuous_index)), 
                                key=lambda i: abs(exit_continuous_index[i] - start_index))
            day_start_pl.append(cumulative_pl[closest_index])
            day_start_pl_longs.append(cumulative_pl_longs[closest_index])
            day_start_pl_shorts.append(cumulative_pl_shorts[closest_index])
    
    ax1.scatter(day_start_indices, day_start_prices, color='red', s=1, zorder=5, label='Day Start')

    ax1.set_xlabel('Trading Time (minutes)')
    ax1.set_ylabel('Close Price', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    
    if when_above_max_investment and len(when_above_max_investment) > 0:
        # Convert when_above_max_investment to continuous index
        above_max_continuous_index = []
        for timestamp in when_above_max_investment:
            if timestamp.date() in trading_days:
                date = timestamp.date()
                minute = (timestamp.time().hour - 9) * 60 + (timestamp.time().minute - 30)
                days_passed = unique_dates.index(date)
                above_max_continuous_index.append(days_passed * 390 + minute)

        # Get the minimum close price for y-value of the points
        min_close = df_intraday['close'].min()

        # Plot points for when above max investment
        ax1.plot(above_max_continuous_index, [min_close] * len(above_max_continuous_index), 
                    color='red', marker='o', linestyle='None', label='Above Max Investment')

    # Create secondary y-axis for cumulative P/L
    ax2 = ax1.twinx()
    
    # Plot cumulative P/L on secondary y-axis
    if len(exit_continuous_index) > 0:
        ax2.plot(exit_continuous_index, cumulative_pl, color='green', label='All P/L')
        ax2.plot(exit_continuous_index, cumulative_pl_longs, color='blue', label='Longs P/L')
        ax2.plot(exit_continuous_index, cumulative_pl_shorts, color='yellow', label='Shorts P/L')
        
        # Add day start markers for P/L lines
        ax2.scatter(day_start_indices, day_start_pl, color='red', s=1, zorder=5)
        ax2.scatter(day_start_indices, day_start_pl_longs, color='red', s=1, zorder=5)
        ax2.scatter(day_start_indices, day_start_pl_shorts, color='red', s=1, zorder=5)
    else:
        print("Warning: No valid exit times found in the trading days.")
    
    ax2.set_ylabel('Cumulative P/L % Change', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    
    # Create tertiary y-axis for volume
    ax3 = ax1.twinx()
    
    # Offset the right spine of ax3 to the left so it's not on top of ax2
    ax3.spines['right'].set_position(('axes', 1.1))
    
    # Plot volume as bars
    if is_short_period:
        bar_width = 30  # Width of half-hour in minutes
        for timestamp, row in volume_data.iterrows():
            date = timestamp.date()
            time = timestamp.time()
            days_passed = unique_dates.index(date)
            minutes = (time.hour - 9) * 60 + time.minute - 30
            x_position = days_passed * 390 + minutes
            ax3.bar(x_position, row['volume'], width=bar_width, alpha=0.3, color='purple', align='edge')
        volume_label = 'Half-hourly Mean Volume'
    else:
        bar_width = 390  # Width of one trading day in minutes
        for i, (date, mean_volume) in enumerate(volume_data.items()):
            ax3.bar(i * 390, mean_volume, width=bar_width, alpha=0.3, color='purple', align='edge')
        volume_label = 'Daily Mean Volume'
    
    ax3.set_ylabel(volume_label, color='purple')
    ax3.tick_params(axis='y', labelcolor='purple')
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    
    # Set title and legend
    plt.title(f'{symbol}: Cumulative P/L % Change vs Close Price and {volume_label}')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax3_patch = mpatches.Patch(color='purple', alpha=0.3, label=volume_label)
    ax1.legend(lines1 + lines2 + [ax3_patch], labels1 + labels2 + [volume_label], loc='upper left')
    
    # Set x-axis ticks to show dates
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

    # Format minor ticks
    ax1.tick_params(axis='x', which='minor', bottom=True)

    # Add gridlines for major ticks (week starts)
    ax1.grid(which='major', axis='x', linestyle='--', alpha=0.7)
        
    # Use a tight layout
    plt.tight_layout()
    
    if filename:
        # Save the figure
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename, dpi=300)
        print(f"Graph has been saved as {filename}",level=logging.DEBUG)
    else:
        plt.show()
        
    plt.close()