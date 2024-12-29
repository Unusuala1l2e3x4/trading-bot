from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, time as time2
from typing import List, Set, Tuple, Optional
from trading_bot.TouchArea import TouchArea
from trading_bot.TypedBarData import TypedBarData # , DefaultTypedBarData
import math
import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from numba import jit
import logging

# https://alpaca.markets/blog/reg-taf-fees/
# check **Alpaca Securities Brokerage Fee Schedule** in [Alpaca Documents Library](https://alpaca.markets/disclosures) for most up-to-date rates
SEC_FEE_RATE = 0.0000278  # $27.80 per $1,000,000
FINRA_TAF_RATE = 0.000166  # $166 per 1,000,000 shares
FINRA_TAF_MAX = 8.30  # Maximum $8.30 per trade

# ELITE_SMART_ROUTER_RATE = 0 # if not using elite smart router

ELITE_SMART_ROUTER_RATE = 0.004 # All-In plan (safest bet)
# ELITE_SMART_ROUTER_RATE = 0.0025
# ELITE_SMART_ROUTER_RATE = 0.002
# ELITE_SMART_ROUTER_RATE = 0.0015
# ELITE_SMART_ROUTER_RATE = 0.001
# ELITE_SMART_ROUTER_RATE = 0.0005


@jit(nopython=True)
def calculate_slippage(is_long: bool, is_entry: bool, price: float, trade_size: int, avg_volume: float, rolling_atr: float,
                       slippage_factor: float, atr_sensitivity: float) -> float:
    # Normalize ATR
    normalized_atr = rolling_atr / price if price > 0 else 0
    
    # ATR effect (dynamic volatility adjustment)
    atr_effect = 1 + atr_sensitivity * normalized_atr
    
    # Trade size impact (square root dampens the effect)
    trade_size_multiplier = (float(trade_size) / avg_volume) ** 0.5
    
    # Base slippage per share adjusted for volatility
    slippage_per_share = slippage_factor * atr_effect
    
    # Total slippage amount including size impact
    total_slippage = slippage_per_share * trade_size * (1 + trade_size_multiplier)
    
    # Convert to price ratio
    slippage_ratio = total_slippage / (price * trade_size)
    
    slippage_price_change = slippage_ratio * price
    
    if is_long:
        if is_entry:
            pass  # Increase price for long entries
        else:
            slippage_ratio *= -1  # Decrease price for long exits
    else:  # short
        if is_entry:
            slippage_ratio *= -1  # Decrease price for short entries
        else:
            pass  # Increase price for short exits

    return price * (1 + slippage_ratio), slippage_price_change


# @jit(nopython=True)
def calculate_price_diff_sums(prices: np.ndarray, is_long: bool) -> Tuple[float, float]:
    """
    Wrapper to use math.fsum() for precise summation of diffs outside of numba constraints.

    Args:
        prices (np.ndarray): A numpy array of sequential floats representing prices.
        is_long (bool): If True, return (positive_sum, negative_sum). If False, flip signs to measure profitability.

    Returns:
        Tuple[float, float]: Sum of positive differences and negative differences (flipped if not long).
    """
    # Get diffs from the JIT-optimized function
    diffs = prices[1:] - prices[:-1]

    # Use math.fsum for precise summation
    positive_sum = math.fsum(diff for diff in diffs if diff > 0)
    negative_sum = math.fsum(diff for diff in diffs if diff < 0)
    net = prices[-1] - prices[0]

    # Adjust for short positions
    if is_long:
        return positive_sum, negative_sum, net
    else:
        return -negative_sum, -positive_sum, -net
    
    
@dataclass
class Transaction:
    timestamp: datetime
    shares: int
    price_unadjusted: float
    price: float
    is_entry: bool # Was it a buy (entry) or sell (exit)
    is_long: bool
    transaction_cost: float # total of next 3 fields
    finra_taf: float
    sec_fee: float  # > 0 for sells (long exits and short entries)
    stock_borrow_cost: float # 0 if not is_long and not is_entry (short exits)
    commission: float
    value: float  # Positive if profit, negative if loss (before transaction costs are applied)
    
    # Record metadata
    bar_latest: TypedBarData
    area_width: float
    shares_remaining: int
    max_shares: int
    
    slippage_price_change: Optional[float] = 0.0
    realized_pl: Optional[float] = 0.0 # None if is_entry is True


@dataclass
class TradePosition:
    symbol: str
    date: date
    id: int
    area: TouchArea
    is_long: bool
    entry_time: datetime
    initial_balance: float
    target_max_shares: int 
    use_margin: bool
    is_marginable: bool
    times_buying_power: float
    entry_price: float
    bar_at_entry: TypedBarData
    
    market_value: float = 0.0
    shares: int = 0 # no fractional trading
    partial_entry_count: int = 0
    partial_exit_count: int = 0
    is_simulated: Optional[bool] = None
    max_shares: Optional[int] = None
    max_shares_reached: Optional[int] = None
    max_max_shares: Optional[int] = None
    min_max_shares: Optional[int] = None
    initial_shares: Optional[int] = None  # actual shares bought at first entry
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    
    avg_entry_price: Optional[float] = None  # Weighted average of partial entries
    borrowed_amount: float = 0.0  # Track total value borrowed
    
    full_entry_price: Optional[float] = None # target price for max shares to hold
    max_target_shares_limit: Optional[int] = None
    has_crossed_full_entry: bool = False
    full_entry_time: Optional[datetime] = None
    gradual_entry_range_multiplier: Optional[float] = 1.0 # Adjust this to control how far price needs to move
    
    transactions: List[Transaction] = field(default_factory=list)
    cleared_area_ids: Set[int] = field(default_factory=set)
    current_stop_price: Optional[float] = None
    current_stop_price_2: Optional[float] = None
    
    max_close: Optional[float] = None
    min_close: Optional[float] = None
    max_high: Optional[float] = None
    min_high: Optional[float] = None
    max_low: Optional[float] = None
    min_low: Optional[float] = None
    
    unrealized_pl: float = field(default=0.0)
    realized_pl: float = 0.0
    slippage_cost: float = 0.0  # Total value lost to slippage
    log_level: Optional[int] = logging.INFO
    # stock_borrow_rate: float = 0.003    # Default to 30 bps (0.3%) annually
    stock_borrow_rate: float = 0.03      # Default to 300 bps (3%) annually
    
    # NOTE: This class assumes intraday trading. No overnight interest is calculated.
    # NOTE: Daytrading buying power cannot increase beyond its start of day value. In other words, closing an overnight position will not add to your daytrading buying power.
    # NOTE: before adding any new features, be sure to review:
    # https://docs.alpaca.markets/docs/margin-and-short-selling
    # https://docs.alpaca.markets/docs/orders-at-alpaca
    # https://docs.alpaca.markets/docs/user-protection
     
    """
    stock_borrow_rate: Annual rate for borrowing the stock (for short positions)
    - Expressed in decimal form (e.g., 0.003 for 30 bps, 0.03 for 300 bps)
    - "bps" means basis points, where 1 bp = 0.01% = 0.0001 in decimal form
    - For ETBs (easy to borrow stocks), this typically ranges from 30 to 300 bps annually
    - 30 bps = 0.30% = 0.003 in decimal form
    - 300 bps = 3.00% = 0.03 in decimal form
    
    Info from website:
    - Borrow fees accrue daily and are billed at the end of each month. Borrow fees can vary significantly depending upon demand to short. 
    - Generally, ETBs cost between 30 and 300bps annually.
    
    - Daily stock borrow fee = Daily ETB stock borrow fee + Daily HTB stock borrow fee
    - Daily ETB stock borrow fee = (settlement date end of day total ETB short $ market value * that stock’s ETB rate) / 360
    - Daily HTB stock borrow fee = Σ((each stock’s HTB short $ market value * that stock’s HTB rate) / 360)
    
    
    See reference: https://docs.alpaca.markets/docs/margin-and-short-selling#stock-borrow-rates
    
    If holding shorts overnight (unimplemented; not applicable to intraday trading):
    - daily_margin_interest_charge = (settlement_date_debit_balance * 0.085) / 360
    
    See reference: https://docs.alpaca.markets/docs/margin-and-short-selling#margin-interest-rate
    """
    
    # Ensure objects are compared based on date and id
    def __eq__(self, other):
        if isinstance(other, TouchArea):
            return self.id == other.id and self.date == other.date
        return False

    # Ensure that objects have a unique hash based on date and id
    def __hash__(self):
        return hash((self.id, self.date))
    

    def __post_init__(self):
        assert self.times_buying_power <= 4
        self.market_value = 0
        self.shares = 0
        self.max_shares, self.max_max_shares, self.min_max_shares = self.target_max_shares, self.target_max_shares, self.target_max_shares
        self.logger = self.setup_logger(self.log_level)
        
        # Calculate full entry price based on area range
        if self.is_long:
            self.full_entry_price = self.area.get_buy_price + (self.area.get_range * self.gradual_entry_range_multiplier) # area bounds already updated in TradingStrategy.create_new_position
        else:
            self.full_entry_price = self.area.get_buy_price - (self.area.get_range * self.gradual_entry_range_multiplier)
            
        # Initial setup only - update_stop_price will handle the rest
        self.has_crossed_full_entry = False
        self.max_target_shares_limit = None
        self.max_shares_reached = 0
        self.initial_shares = None
        
        self.max_close = self.min_close = self.bar_at_entry.close
           
        if self.is_long:
            self.current_stop_price = self.max_close - self.area.get_range
            self.current_stop_price_2 = self.max_close - self.area.get_range * 3
        else:
            self.current_stop_price = self.min_close + self.area.get_range
            self.current_stop_price_2 = self.min_close + self.area.get_range * 3
            
            
        self.update_stop_price(self.bar_at_entry, self.entry_time)
        
    def calculate_target_shares_from_price(self, current_price: float) -> int:
        """Calculate target shares based on how close price is to full entry"""
        if self.max_shares <= 0:
            return 0
            
        if self.is_long:
            price_movement = current_price - self.area.get_buy_price
            full_movement = self.full_entry_price - self.area.get_buy_price
        else:
            price_movement = self.area.get_buy_price - current_price
            full_movement = self.area.get_buy_price - self.full_entry_price
            
        # Note: price_movement could be negative if price moved away
        target_pct = np.clip(price_movement / full_movement, 0, 1.0)
        shares = math.floor(target_pct * self.max_shares)
        
        # Ensure at least 1 share
        return max(1, shares)
    

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
        self.logger.log(level, message, exc_info=level >= logging.ERROR)

    @property
    def is_open(self) -> bool:
        return self.shares > 0

    def calculate_slippage(self, is_entry: bool, price: float, trade_size: int, bar: TypedBarData, slippage_factor: float, atr_sensitivity: float) -> float:
        adjusted_price, slippage_price_change = calculate_slippage(self.is_long, is_entry, price, trade_size, bar.avg_volume, bar.rolling_ATR, slippage_factor, atr_sensitivity)
        # print(f"{self.is_long} {is_entry} {trade_size} {price} -> {adjusted_price} ({slippage_price_change})")
        
        return adjusted_price, slippage_price_change

    def update_market_value(self, current_price: float):
        self.market_value = self.shares * current_price if self.is_long else -self.shares * current_price
        self.unrealized_pl = self.market_value - (self.shares * self.avg_entry_price)

    def calculate_transaction_cost(self, shares: int, price: float, is_entry: bool, timestamp: datetime) -> float:
        is_sell = (self.is_long and not is_entry) or (not self.is_long and is_entry)
        finra_taf = min(FINRA_TAF_RATE * shares, FINRA_TAF_MAX) if is_sell else 0
        trade_value = price * shares
        sec_fee = SEC_FEE_RATE * trade_value if is_sell else 0
        
        commission = ELITE_SMART_ROUTER_RATE * shares
        
        # https://docs.alpaca.markets/docs/margin-and-short-selling#stock-borrow-rates
        
        stock_borrow_cost = 0
        if not self.is_long and not is_entry:  # Stock borrow cost applies only to short position exits
            daily_borrow_rate = self.stock_borrow_rate / 360
            total_cost = 0
            
            # Walk backwards to find relevant entry transactions
            relevant_entries = []
            cumulative_shares = 0
            for transaction in reversed(self.transactions):
                if transaction.is_entry:
                    relevant_shares = min(transaction.shares, shares - cumulative_shares)
                    relevant_entries.append((transaction, relevant_shares))
                    cumulative_shares += relevant_shares
                    assert cumulative_shares <= shares
                    if cumulative_shares == shares:
                        break
            
            # Calculate borrow cost for relevant entries
            for entry_transaction, relevant_shares in relevant_entries:
                holding_time = timestamp - entry_transaction.timestamp
                days_held = float(holding_time.total_seconds()) / float(24 * 60 * 60)
                total_cost += relevant_shares * price * daily_borrow_rate * days_held
            assert cumulative_shares == shares, f"Mismatch in shares calculation: {cumulative_shares} != {shares}"
            stock_borrow_cost = total_cost
        return finra_taf, sec_fee, stock_borrow_cost, commission

    def add_transaction(self, timestamp: datetime, shares: int, price_unadjusted: float, price: float, is_entry: bool, bar: TypedBarData,
                        slippage_price_change: float = 0.0,
                        realized_pl: float = 0.0):
        finra_taf, sec_fee, stock_borrow_cost, commission = self.calculate_transaction_cost(shares, price, is_entry, timestamp)
        transaction_cost = finra_taf + sec_fee + stock_borrow_cost + commission
        
        # value is -shares * price for buys (long entry, short exit)
        # value is shares * price for sells (long exit, short entry)  
        value = -shares * price if ((self.is_long and is_entry) or (not self.is_long and not is_entry)) else shares * price
        
        # For realized P&L: value is already negative for buys and positive for sells 
        # self.realized_pl += value
        self.realized_pl += realized_pl
        self.slippage_cost += slippage_price_change * shares

        transaction = Transaction(timestamp, shares, price_unadjusted, price, is_entry, self.is_long, transaction_cost, finra_taf, sec_fee, stock_borrow_cost, commission, 
                                  value, bar, 
                                  self.area.get_range,
                                  self.shares,
                                  self.max_shares,
                                  slippage_price_change=slippage_price_change,
                                  realized_pl=realized_pl
        )
        self.transactions.append(transaction)
        
        self.log(f"Transaction added - {'Entry' if is_entry else 'Exit'}, Shares: {shares}, Price: {price:.4f}, "
                 f"Value: {value:.4f}, Cost: {transaction_cost:.4f}, Realized PnL: {self.realized_pl}", level=logging.DEBUG)
        
        return transaction_cost

    def increase_max_shares(self, shares):
        self.max_shares = max(self.max_shares, shares)
        self.max_max_shares = max(self.max_max_shares, self.max_shares)
    
    def decrease_max_shares(self, shares):
        self.max_shares = min(self.max_shares, shares)
        self.min_max_shares = min(self.min_max_shares, self.max_shares)
    
    def initial_entry(self, bar: TypedBarData, slippage_factor: float, atr_sensitivity: float):
        """
        Initial entry with gradual sizing based on max_target_shares_limit.
        Uses partial_entry internally.
        """
        # Use the calculated limit instead of initial_shares
        return self.partial_entry(self.entry_time, self.entry_price, self.max_target_shares_limit, bar, slippage_factor, atr_sensitivity)
        # return self.partial_entry(self.entry_time, self.entry_price, self.initial_shares, vwap, volume, avg_volume, slippage_factor)
        
    def partial_entry(self, entry_time: datetime, entry_price: float, shares_to_buy: int, 
                     bar: TypedBarData, slippage_factor: float, atr_sensitivity: float):
        adjusted_price, slippage_price_change = self.calculate_slippage(
            True, entry_price, shares_to_buy, bar, slippage_factor, atr_sensitivity)
        
        # Update weighted average entry price
        if self.shares == 0:
            self.avg_entry_price = adjusted_price
        else:
            old_value = self.shares * self.avg_entry_price
            new_value = shares_to_buy * adjusted_price
            self.avg_entry_price = (old_value + new_value) / (self.shares + shares_to_buy)

        cash_needed = shares_to_buy * adjusted_price
        
        # Track borrowed amount for shorts
        if not self.is_long:
            self.borrowed_amount += cash_needed
        
        self.shares += shares_to_buy
        self.update_market_value(adjusted_price)
        self.partial_entry_count += 1
        self.max_shares_reached = max(self.max_shares_reached, self.shares)
        fees = self.add_transaction(entry_time, shares_to_buy, entry_price, adjusted_price, True, bar, slippage_price_change)
        
        return cash_needed, fees, shares_to_buy

    def partial_exit(self, exit_time: datetime, exit_price: float, shares_to_sell: int, 
                    bar: TypedBarData, slippage_factor: float, atr_sensitivity: float):
        adjusted_price, slippage_price_change = self.calculate_slippage(
            False, exit_price, shares_to_sell, bar, slippage_factor, atr_sensitivity)

        if not self.is_long:
            # Calculate portion of borrowed amount being returned
            portion = shares_to_sell / self.shares
            returned_borrowed = self.borrowed_amount * portion
            self.borrowed_amount -= returned_borrowed
            exit_pl = (self.avg_entry_price - adjusted_price) * shares_to_sell
        else:
            returned_borrowed = 0
            exit_pl = (adjusted_price - self.avg_entry_price) * shares_to_sell

        cash_released = shares_to_sell * adjusted_price
        self.shares -= shares_to_sell
        self.update_market_value(adjusted_price)
        self.partial_exit_count += 1
        
        fees = self.add_transaction(exit_time, shares_to_sell, exit_price, adjusted_price, False, bar, slippage_price_change, realized_pl=exit_pl)
        
        return exit_pl, cash_released, returned_borrowed, fees, shares_to_sell

    # NOTE: updates stop price, as well as min/max close, max high, min low
    def update_stop_price(self, bar: TypedBarData, current_timestamp: datetime):
        # should_exit_price = bar.close
        # should_exit_price_2 = bar.close
        
        if self.is_long:
            should_exit_price = bar.low
            should_exit_price_2 = bar.low
        else:
            should_exit_price = bar.high
            should_exit_price_2 = bar.high
            
        prev_stop_price = self.current_stop_price
        prev_stop_price_2 = self.current_stop_price_2
        should_have_exited = self.should_exit(should_exit_price)
        should_have_exited_2 = self.should_exit_2(should_exit_price_2)

        self.area.update_bounds(current_timestamp)
        self.max_close = max(self.max_close or self.bar_at_entry.close, bar.close)
        self.min_close = min(self.min_close or self.bar_at_entry.close, bar.close)
        
        self.max_high = max(self.max_high or self.bar_at_entry.high, bar.high)
        self.min_high = min(self.min_high or self.bar_at_entry.high, bar.high)
        
        self.max_low = max(self.max_low or self.bar_at_entry.low, bar.low)
        self.min_low = min(self.min_low or self.bar_at_entry.low, bar.low)
           
        if self.is_long:
            self.current_stop_price = self.max_close - self.area.get_range
            self.current_stop_price_2 = self.max_close - self.area.get_range * 3
        else:
            self.current_stop_price = self.min_close + self.area.get_range
            self.current_stop_price_2 = self.min_close + self.area.get_range * 3
        
        # self.update_market_value(quote_price) # NOTE: update_market_value should use quotes data, but not necessary here. quote price isnt determined yet anyways.
        self.log(f"area {self.area.id}: get_range {self.area.get_range:.4f}",level=logging.DEBUG)
            
        # Check if price has crossed full entry threshold
        if not self.has_crossed_full_entry:
            if (self.is_long and bar.close >= self.full_entry_price) or \
            (not self.is_long and bar.close <= self.full_entry_price):
                # Full entry condition met - go to maximum size
                self.has_crossed_full_entry = True
                self.full_entry_time = current_timestamp
                self.max_target_shares_limit = self.max_shares
                if self.initial_shares is None:
                    self.initial_shares = self.max_target_shares_limit
                # self.log(f"100% of target shares ({self.max_shares}) reached at entry",level=logging.INFO)
                self.log(f"100% of target shares ({self.max_shares}) reached {(current_timestamp - self.entry_time).total_seconds()/60 :.2f} min after entry",level=logging.INFO)
            else:
                # Check for close price crossing buy price
                current_limit = self.max_target_shares_limit or 0 # default 0 for comparisons of limits
                
                # NOTE: if doing gradual entry
                new_limit = self.calculate_target_shares_from_price(bar.close)
                
                # # NOTE: if doing immediate entry after close price meets buy price
                # new_limit = self.max_shares
                # self.has_crossed_full_entry = True
                # self.full_entry_time = current_timestamp
                
                # Scale up based on close price (when full entry not reached but close has crossed entry price)
                # NOTE: commenting this out seems to improve performance
                if (self.is_long and bar.close >= self.area.get_buy_price) or \
                (not self.is_long and bar.close <= self.area.get_buy_price):
                    # Close has crossed buy price - calculate size based on movement
                    if new_limit > current_limit:
                        self.max_target_shares_limit = new_limit
                        if self.initial_shares is None:
                            self.initial_shares = self.max_target_shares_limit
                        self.log(f"Close price crossed {(current_timestamp - self.entry_time).total_seconds()/60 :.0f} min after entry: {new_limit} shares ({(new_limit/self.max_shares)*100:.1f}%)",level=logging.INFO)
                else:
                    
                    # Close hasn't crossed but high/low has - start with 1 share
                    if self.max_target_shares_limit is None:
                        self.max_target_shares_limit = 1
                        if self.initial_shares is None:
                            self.initial_shares = self.max_target_shares_limit
                        self.log(f"High/Low price crossed at entry but not Close. Starting with {self.initial_shares} share(s)",level=logging.INFO)
                    elif new_limit > current_limit:
                        # Maintain current limit until close crosses buy price
                        self.max_target_shares_limit = current_limit

        return self.should_exit(bar.close), self.should_exit_2(bar.close), should_have_exited, should_have_exited_2, prev_stop_price, prev_stop_price_2
        # return self.should_exit(should_exit_price), self.should_exit_2(should_exit_price_2), should_have_exited, should_have_exited_2, prev_stop_price, prev_stop_price_2

    def should_exit(self, current_price: float) -> bool:
        return (self.is_long and current_price <= self.current_stop_price) or \
               (not self.is_long and current_price >= self.current_stop_price)

    def should_exit_2(self, current_price: float) -> bool:
        return (self.is_long and current_price <= self.current_stop_price_2) or \
               (not self.is_long and current_price >= self.current_stop_price_2)

    def close(self, exit_time: datetime, exit_price: float):
        self.exit_time = exit_time
        self.exit_price = exit_price

    @property
    def total_commission(self) -> float:
        return sum(t.commission for t in self.transactions)

    @property
    def total_stock_borrow_cost(self) -> float:
        if self.is_long:
            return 0.0
        return sum(t.stock_borrow_cost for t in self.transactions if not t.is_entry)

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
    def get_unrealized_pl(self) -> float:
        return self.unrealized_pl

    @property
    def get_realized_pl(self) -> float:
        return self.realized_pl

    @property
    def pl(self) -> float:
        return self.get_unrealized_pl + self.get_realized_pl - self.total_transaction_costs

    @property
    def plpc(self) -> float:
        return (self.pl / self.initial_balance) * 100
    
    @property
    def price_diff_sum(self) -> Tuple[float, float]:
        assert len(self.transactions) >= 2
        prices = np.array([a.price_unadjusted for a in self.transactions], dtype=np.float64)
        return calculate_price_diff_sums(prices, self.is_long)
    
    
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
        side_string = 'Long' if trade.is_long else 'Short'
        if trade.area.is_side_switched:
            side_string = '*'+side_string

        incs, decs, net = trade.price_diff_sum
        
        area_width_history = [a.area_width for a in trade.transactions]
        at_max_shares_count = sum(1 for a in trade.transactions if a.shares_remaining == a.max_shares)
        
        row = {
            'sym': trade.symbol,
            'date': trade.date,
            'ID': trade.id,
            'AreaID': trade.area.id,
            'Type': side_string,
            'Entry Time': trade.entry_time.time().strftime('%H:%M:%S'),
            'Exit Time': trade.exit_time.time().strftime('%H:%M:%S') if trade.exit_time else None,
            'Holding Time (min)': int(trade.holding_time.total_seconds() / 60),
            'Entry Price': trade.entry_price,
            'Exit Price': trade.exit_price if trade.exit_price else None,
            'Price Increases': incs,
            'Price Decreases': decs,
            'Price Net': net,
            'Initial Qty': trade.initial_shares,
            'Target Qty': trade.target_max_shares,
            'Max Qty Reached (%)': 100*(trade.max_shares_reached / trade.target_max_shares),
            'At Max Shares Count': at_max_shares_count,
            'Min Area Width': min(area_width_history),
            'Max Area Width': max(area_width_history),
            # 'Largest Max Qty': trade.max_max_shares,
            # 'Smallest Max Qty': trade.min_max_shares,
            # 'Realized P/L': f"{trade.get_realized_pl:.6f}",
            # 'Unrealized P/L': f"{trade.get_unrealized_pl:.6f}",
            'Slippage Costs': f"{trade.slippage_cost:.6f}",
            'Total P/L': f"{trade.pl:.6f}",
            'ROE (P/L %)': f"{trade.plpc:.12f}",
            'Cumulative P/L %': f"{cumulative_pct_change:.6f}",
            'Transaction Costs': f"{trade.total_transaction_costs:.6f}",
            'Times Buying Power': trade.times_buying_power,
            
            # bar metrics
            'shares_per_trade': f"{trade.bar_at_entry.shares_per_trade:.6f}",
            'doji_ratio': f"{trade.bar_at_entry.doji_ratio:.6f}",
            'abs_doji_ratio': f"{trade.bar_at_entry.doji_ratio_abs:.6f}",
            'wick_ratio': f"{trade.bar_at_entry.wick_ratio:.6f}",
            'nr4_hl_diff': f"{trade.bar_at_entry.nr4_hl_diff:.6f}",
            'nr7_hl_diff': f"{trade.bar_at_entry.nr7_hl_diff:.6f}",
            'volume_ratio': f"{trade.bar_at_entry.volume_ratio:.6f}",
            'ATR_ratio': f"{trade.bar_at_entry.ATR_ratio:.6f}",
        }
        # get aggregated metrics per area
        row.update(trade.area.get_metrics(trade.entry_time, prefix=''))
        # row.update(trade.area.get_metrics(trade.entry_time, prefix='entry_'))
        # row.update(trade.area.get_metrics(trade.exit_time, prefix='exit_'))
        data.append(row)
        
    df = pd.DataFrame(data)
    bardf = TypedBarData.to_dataframe([trade.bar_at_entry for trade in trades])
    # print(df['Entry Time'])
    # print(bardf['timestamp'].dt.time )
    assert bardf['timestamp'].dt.strftime('%H:%M:%S').equals(df['Entry Time'])
    df = pd.concat([df,bardf.drop(columns=['timestamp','symbol','time','date'],errors='ignore')],axis=1)
    
    if len(os.path.dirname(filename)) > 0:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
    df.to_csv(filename, index=False)
    print(f"Trade summary has been exported to {filename}")
    
    # df.to_csv(filename.replace('.csv', '.tsv'), index=False, sep='\t')   # Replace the existing df.to_csv line
    # print(f"Trade summary has been exported to {filename}")


def time_to_minutes(t: time2):
    return t.hour * 60 + t.minute - (9 * 60 + 30)

@dataclass
class SimplifiedTradePosition:
    symbol: str
    date: date
    id: int
    area_id: int
    is_long: bool
    entry_time: datetime
    exit_time: datetime
    holding_time: timedelta
    entry_price: float
    exit_price: float
    bar_at_entry: TypedBarData
    initial_shares: int
    target_max_shares: int
    max_shares_reached: float
    at_max_shares_count: float
    min_area_width: float
    max_area_width: float
    # max_max_shares: int
    # min_max_shares: int
    # realized_pl: float
    # unrealized_pl: float
    slippage_cost: float
    pl: float
    plpc: float
    cumulative_pct_change: float
    total_transaction_costs: float
    times_buying_power: float

    
    shares_per_trade: float
    doji_ratio: float
    abs_doji_ratio: float
    wick_ratio: float
    nr4_hl_diff: float
    nr7_hl_diff: float
    volume_ratio: float
    ATR_ratio: float
    

def csv_to_trade_positions(csv_file_path) -> List[SimplifiedTradePosition]:
    df = pd.read_csv(csv_file_path)
    trade_positions = []
    
    bar_columns = TypedBarData.get_field_names()
    
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
            symbol=row['sym'],
            date=trade_date,
            id=row['ID'],
            area_id=row['AreaID'],
            is_long=(row['Type'].endswith('Long')),
            entry_time=entry_time,
            exit_time=exit_time,
            holding_time=holding_time,
            entry_price=row['Entry Price'],
            exit_price=row['Exit Price'] if pd.notna(row['Exit Price']) else None,

            bar_at_entry = TypedBarData.from_row(row.loc[bar_columns]),

            initial_shares=row['Initial Shares'],
            target_max_shares=row['Target Qty'],
            max_shares_reached=row['Max Qty Reached (%)'],
            at_max_shares_count=row['At Max Shares Count'],
            min_area_width=row['Min Area Width'],
            max_area_width=row['Max Area Width'],
            # max_max_shares=row['Largest Max Qty'],
            # min_max_shares=row['Smallest Max Qty'],
            # realized_pl=row['Realized P/L'],
            # unrealized_pl=row['Unrealized P/L'],
            slippage_cost=row['Slippage Costs'],
            pl=row['Total P/L'],
            plpc=row['ROE (P/L %)'],
            cumulative_pct_change=row['Cumulative P/L %'],
            total_transaction_costs=row['Transaction Costs'],
            times_buying_power=row['Times Buying Power'],
            
            shares_per_trade=row['shares_per_trade'],
            doji_ratio=row['doji_ratio'],
            abs_doji_ratio=row['abs_doji_ratio'],
            wick_ratio=row['wick_ratio'],
            nr4_hl_diff=row['nr4_hl_diff'],
            nr7_hl_diff=row['nr7_hl_diff'],
            volume_ratio=row['volume_ratio'],
            ATR_ratio=row['ATR_ratio'],
            
        )
        trade_positions.append(trade_position)
    
    return trade_positions

import matplotlib.patches as mpatches

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
        volume_data = df_intraday.groupby(['date', 'half_hour'])['volume'].sum().reset_index()
        volume_data['datetime'] = volume_data.apply(lambda row: pd.Timestamp.combine(row['date'], row['half_hour']), axis=1)
        volume_data = volume_data.set_index('datetime').sort_index()
        volume_data = volume_data[volume_data.index.time != time2(16, 0)]
    else:
        # Group by day
        volume_data = df_intraday.groupby('date')['volume'].sum()
        # TODO: instead of grouping all volume data from each day, filter by the first 15-30 minutes (starting 9:30 AM) of each day's volume
        
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
        print(f"Graph has been saved as {filename}")
    else:
        plt.show()
        
    plt.close()