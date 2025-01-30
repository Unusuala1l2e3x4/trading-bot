from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, time as datetime_time
from typing import List, Set, Tuple, Optional
from trading_bot.TouchArea import TouchArea
from trading_bot.TypedBarData import TypedBarData # , DefaultTypedBarData
from trading_bot.PositionMetrics import PositionMetrics, PositionSnapshot

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
def calculate_slippage(is_long: bool, is_entry: bool, price: float, trade_size: int, avg_volume: float, ATR: float,
                       slippage_factor: float, atr_sensitivity: float) -> float:
    # Normalize ATR
    normalized_atr = ATR / price if price > 0 else 0
    
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
    
    # Record metadata (may or may not be transaction-specific)
    bar_latest: TypedBarData
    area_width: float
    shares_remaining: int
    max_shares: int
    # avg_entry_price
    # current_pl
    
    # Record optional fields (transaction-specific)
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
    bar_at_commit: TypedBarData
    
    prior_relevant_bars: List[TypedBarData]
    
    bar_at_entry: TypedBarData = None
    market_value: float = 0.0
    shares: int = 0 # no fractional trading
    partial_entry_count: int = 0
    partial_exit_count: int = 0
    is_simulated: Optional[bool] = None
    max_shares: Optional[int] = None
    # max_shares_reached: Optional[int] = None
    # max_shares_reached_time: Optional[datetime] = None
    max_max_shares: Optional[int] = None
    min_max_shares: Optional[int] = None
    initial_shares: Optional[int] = None  # actual shares bought at first entry
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    
    avg_entry_price: Optional[float] = None  # Weighted average of partial entries
    borrowed_amount: float = 0.0  # Track total value borrowed
    
    cost_basis_sold_accum: float = 0.0  # Sum of (avg_entry_price * shares_sold) for each exit
    
    full_entry_price: Optional[float] = None # target price for max shares to hold
    max_target_shares_limit: Optional[int] = None
    has_crossed_full_entry: bool = False
    full_entry_time: Optional[datetime] = None
    gradual_entry_range_multiplier: Optional[float] = 1.0 # Adjust this to control how far price needs to move
    
    prev_shares: int = 0  # Track shares before last transaction
    prev_pl: float = 0.0
    prev_avg_entry_price: Optional[float] = None
    prev_cost_basis_sold_accum: float = 0.0
    halfway_price: Optional[float] = None
    
    prev_accum_pl: Optional[float] = 0.0
    prev_accum_plpc: Optional[float] = 0.0
    
    has_entered: bool = False
    has_exited: bool = False
    
    transactions: List[Transaction] = field(default_factory=list)
    cleared_area_ids: Set[int] = field(default_factory=set)
    current_stop_price: Optional[float] = None
    current_stop_price_2: Optional[float] = None
    
    position_metrics: PositionMetrics = None
    
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
        # self.max_shares_reached = 0
        # self.max_shares_reached_time = self.entry_time
        self.initial_shares = None
        
        self.max_close = self.min_close = self.bar_at_commit.close
           
        if self.is_long:
            self.current_stop_price = -np.inf
            self.current_stop_price_2 = -np.inf
        else:
            self.current_stop_price = np.inf
            self.current_stop_price_2 = np.inf
            
        self.update_stop_price(self.bar_at_commit, self.entry_time)
        self.position_metrics = PositionMetrics(self.is_long, prior_relevant_bars=self.prior_relevant_bars)
        
        
        
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
        adjusted_price, slippage_price_change = calculate_slippage(self.is_long, is_entry, price, trade_size, bar.avg_volume, bar.ATR, slippage_factor, atr_sensitivity)
        # print(f"{self.is_long} {is_entry} {trade_size} {price} -> {adjusted_price} ({slippage_price_change})")
        
        return adjusted_price, slippage_price_change

    def update_market_value(self, current_price: float):
        """Update market value and unrealized P&L using current market price."""
        assert self.avg_entry_price is not None
        if self.is_long:
            self.market_value = self.shares * current_price
            self.unrealized_pl = self.market_value - (self.shares * self.avg_entry_price)
        else:
            self.market_value = -self.shares * current_price  # Keep negative for accounting
            self.unrealized_pl = (self.shares * self.avg_entry_price) + self.market_value  # Entry - Current

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
        if self.has_crossed_full_entry:
            self.max_target_shares_limit = self.max_shares
    
    def decrease_max_shares(self, shares):
        self.max_shares = min(self.max_shares, shares)
        self.min_max_shares = min(self.min_max_shares, self.max_shares)
        if self.has_crossed_full_entry:
            self.max_target_shares_limit = self.max_shares
    
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
        
        self.prev_shares = self.shares
        self.prev_pl = self.pl
        self.prev_avg_entry_price = self.avg_entry_price
        # self.prev_cost_basis_sold_accum = self.cost_basis_sold_accum
        if shares_to_buy == 0:
            return 0, 0, 0
        
        adjusted_price, slippage_price_change = self.calculate_slippage(
            True, entry_price, shares_to_buy, bar, slippage_factor, atr_sensitivity)
        
        # Update weighted average entry price
        if self.shares == 0:
            assert not self.has_entered
            assert self.avg_entry_price is None
            self.avg_entry_price = adjusted_price
            self.has_entered = True
            self.bar_at_entry = bar
        else:
            assert self.has_entered
            old_value = self.shares * self.avg_entry_price
            new_value = shares_to_buy * adjusted_price
            self.avg_entry_price = (old_value + new_value) / (self.shares + shares_to_buy)

        cash_needed = shares_to_buy * adjusted_price
        
        # Track borrowed amount for shorts
        if not self.is_long:
            self.borrowed_amount += cash_needed
        
        self.shares += shares_to_buy
        self.update_market_value(bar.close)
        self.partial_entry_count += 1
        # if self.shares > self.max_shares_reached:
        #     self.max_shares_reached = self.shares
        #     self.max_shares_reached_time = entry_time
        fees = self.add_transaction(entry_time, shares_to_buy, entry_price, adjusted_price, True, bar, slippage_price_change)
        
        return cash_needed, fees, shares_to_buy

    def partial_exit(self, exit_time: datetime, exit_price: float, shares_to_sell: int, 
                    bar: TypedBarData, slippage_factor: float, atr_sensitivity: float):
                    
        self.prev_shares = self.shares
        self.prev_pl = self.pl
        # self.prev_avg_entry_price = self.avg_entry_price
        self.prev_cost_basis_sold_accum = self.cost_basis_sold_accum
        if shares_to_sell == 0:
            return 0, 0, 0, 0, 0

        adjusted_price, slippage_price_change = self.calculate_slippage(
            False, exit_price, shares_to_sell, bar, slippage_factor, atr_sensitivity)

        # Calculate cost basis and cash values first
        cost_basis_sold = self.avg_entry_price * shares_to_sell
        self.cost_basis_sold_accum += cost_basis_sold
        cash_released = shares_to_sell * adjusted_price

        # Calculate exit pl using the values we already computed
        if not self.is_long:
            portion = shares_to_sell / self.shares
            returned_borrowed = self.borrowed_amount * portion
            self.borrowed_amount -= returned_borrowed
            exit_pl = cost_basis_sold - cash_released
        else:
            returned_borrowed = 0
            exit_pl = cash_released - cost_basis_sold

        self.shares -= shares_to_sell
        self.update_market_value(bar.close)
        self.partial_exit_count += 1
        
        fees = self.add_transaction(exit_time, shares_to_sell, exit_price, adjusted_price, False, bar, slippage_price_change, realized_pl=exit_pl)
        
        return exit_pl, cash_released, returned_borrowed, fees, shares_to_sell

    def calculate_exit_values(self, exit_time: datetime, exit_price: float, shares_to_sell: int, 
                            bar: TypedBarData, slippage_factor: float, atr_sensitivity: float,
                            use_prev_avg_entry: bool = False) -> Tuple[float, float]:
        """Calculate expected realized P&L and fees for an exit without executing it."""
        
        if shares_to_sell == 0:
            return 0.0, 0.0
        
        # Calculate P&L using appropriate entry price
        entry_price = self.prev_avg_entry_price if use_prev_avg_entry else self.avg_entry_price
        if entry_price is None:
            return 0.0, 0.0
        
        # Calculate slippage-adjusted price 
        adjusted_price, _ = self.calculate_slippage(
            False, exit_price, shares_to_sell, bar, slippage_factor, atr_sensitivity
        )

        if not self.is_long:
            exit_pl = (entry_price - adjusted_price) * shares_to_sell
        else:
            exit_pl = (adjusted_price - entry_price) * shares_to_sell
            
        # Calculate fees
        finra_taf, sec_fee, stock_borrow_cost, commission = self.calculate_transaction_cost(
            shares_to_sell, adjusted_price, False, exit_time
        )
        fees_expected = finra_taf + sec_fee + stock_borrow_cost + commission
        
        return exit_pl, fees_expected


    # NOTE: updates stop price, as well as min/max close, max high, min low
    def update_stop_price(self, bar: TypedBarData, current_timestamp: datetime, 
                          exit_quote_price: float = None, slippage_factor: float = None, atr_sensitivity: float = None):
        # should_exit_price = bar.close
        # should_exit_price_2 = bar.close
        
        if self.is_long:
            should_exit_price = bar.low
            should_exit_price_2 = bar.low
        else:
            should_exit_price = bar.high
            should_exit_price_2 = bar.high
        
        prev_halfway_price = self.halfway_price
        prev_stop_price = self.current_stop_price
        # prev_stop_price = self.avg_entry_price
        prev_stop_price_2 = self.current_stop_price_2
        
        should_have_exited = self.reached_current_stop_price(should_exit_price)
        # should_have_exited = self.reached_avg_entry_price(should_exit_price)
        should_have_exited_2 = self.reached_current_stop_price_2(should_exit_price_2)
        
        should_have_exited_halfway = self.reached_halfway_price(should_exit_price) and \
            self.shares >= self.max_shares * 0.75
        # and \
        #     (
        #         (self.is_long and prev_halfway_price > prev_stop_price) or \
        #         (not self.is_long and prev_halfway_price < prev_stop_price)
        #     )
        
        

        self.area.update_bounds(current_timestamp)
        self.max_close = max(self.max_close or self.bar_at_commit.close, bar.close)
        self.min_close = min(self.min_close or self.bar_at_commit.close, bar.close)
        
        self.max_high = max(self.max_high or self.bar_at_commit.high, bar.high)
        self.min_high = min(self.min_high or self.bar_at_commit.high, bar.high)
        
        self.max_low = max(self.max_low or self.bar_at_commit.low, bar.low)
        self.min_low = min(self.min_low or self.bar_at_commit.low, bar.low)

        if self.is_long:
            self.current_stop_price = self.max_close - self.area.get_range
            # self.current_stop_price = max(self.max_close - self.area.get_range, self.area.get_buy_price)
            # self.current_stop_price = max(self.max_close - self.area.get_range, self.set_halfway_price())
            self.current_stop_price_2 = self.max_close - self.area.get_range * 3
        else:
            self.current_stop_price = self.min_close + self.area.get_range
            # self.current_stop_price = min(self.min_close + self.area.get_range, self.area.get_buy_price)
            # self.current_stop_price = min(self.min_close + self.area.get_range, self.set_halfway_price())
            self.current_stop_price_2 = self.min_close + self.area.get_range * 3
        
        self.set_halfway_price()
        
        # self.update_market_value(exit_quote_price) # NOTE: update_market_value should use quotes data, but not necessary here. quote price isnt determined yet anyways.
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
                        # self.max_target_shares_limit = 1
                        self.max_target_shares_limit = 0
                        if self.initial_shares is None:
                            self.initial_shares = self.max_target_shares_limit
                        self.log(f"High/Low price crossed at entry but not Close. Starting with {self.initial_shares} share(s)",level=logging.INFO)
                    elif new_limit > current_limit:
                        # Maintain current limit until close crosses buy price
                        self.max_target_shares_limit = current_limit

        # return self.reached_current_stop_price(bar.close), self.reached_current_stop_price_2(bar.close), should_have_exited, should_have_exited_2, prev_stop_price, prev_stop_price_2
        return self.reached_exit_ema(bar, exit_quote_price, slippage_factor, atr_sensitivity), self.reached_current_stop_price(bar.close), self.reached_current_stop_price_2(bar.close), \
            should_have_exited_halfway, \
            should_have_exited, should_have_exited_2, prev_halfway_price, prev_stop_price, prev_stop_price_2
        
        # return self.reached_current_stop_price(should_exit_price), self.reached_current_stop_price_2(should_exit_price_2), should_have_exited, should_have_exited_2, prev_stop_price, prev_stop_price_2



    def reached_exit_ema(self, bar: TypedBarData, exit_quote_price: float = None, slippage_factor: float = None, atr_sensitivity: float = None) -> bool:
        if (not self.position_metrics or self.position_metrics.num_snapshots == 0) or (exit_quote_price is None or slippage_factor is None or atr_sensitivity is None):
            return self.reached_current_stop_price(bar.close)
        
        # # Calculate hypothetical cost basis once
        # hypothetical_cost_basis = (self.prev_cost_basis_sold_accum + 
        #                         (self.prev_shares * self.prev_avg_entry_price 
        #                         if self.prev_avg_entry_price else 0))
        
        
        #         # Calculate P/L for each category
        # quote_pl, quote_plpc = self.calculate_exit_pl_values(
        #     bar.timestamp, exit_quote_price, bar, 
        #     slippage_factor, atr_sensitivity, 
        #     hypothetical_cost_basis
        # )

        prev_snapshot = self.position_metrics.snapshots[-1]
        assert prev_snapshot.timestamp == bar.timestamp - timedelta(minutes=1)

        # # if quote_pl > 0:
        if self.is_long:
            # if prev_snapshot.central_value_dist > 0 and bar.central_value_dist <= 0:
            if prev_snapshot.bar.exit_ema_dist > 0 and bar.exit_ema_dist <= 0:
                return True
        else:
            if prev_snapshot.bar.exit_ema_dist < 0 and bar.exit_ema_dist >= 0:
                return True
        
        # if prev_snapshot.central_value_dist != bar.central_value_dist:
        #     if prev_snapshot.central_value_dist >= 0 and bar.central_value_dist <= 0:
        #         return True

        #     if prev_snapshot.central_value_dist <= 0 and bar.central_value_dist >= 0:
        #         return True
        
        return False


    def reached_current_stop_price(self, current_price: float) -> bool:
        return (self.is_long and current_price <= self.current_stop_price) or \
               (not self.is_long and current_price >= self.current_stop_price)

    def reached_current_stop_price_2(self, current_price: float) -> bool:
        return (self.is_long and current_price <= self.current_stop_price_2) or \
               (not self.is_long and current_price >= self.current_stop_price_2)

    def reached_avg_entry_price(self, current_price: float) -> bool:
        return self.avg_entry_price is not None and \
                (
                    (self.is_long and current_price <= self.avg_entry_price) or \
                    (not self.is_long and current_price >= self.avg_entry_price)
                )
                
    def reached_halfway_price(self, current_price: float) -> bool:
        return self.avg_entry_price is not None and \
                (
                    (self.is_long and current_price <= self.halfway_price) or \
                    (not self.is_long and current_price >= self.halfway_price)
                )


    def set_halfway_price(self):
        if self.is_long:
            if self.avg_entry_price is None:
                ret = -np.inf
            else:
                ret = (self.max_close + self.avg_entry_price) / 2
        else:
            if self.avg_entry_price is None:
                ret = np.inf
            else:
                ret = (self.min_close + self.avg_entry_price) / 2
        self.halfway_price = ret
        return ret
            
    def close(self, exit_time: datetime, exit_price: float):
        if not self.has_exited:
            if self.has_entered:
                self.area.record_entry_exit(self.actual_entry_time, self.actual_entry_price, exit_time, exit_price)
            if not self.has_entered:
                assert self.pl == 0
                self.bar_at_entry = self.bar_at_commit # no better value
                self.position_metrics.finalize_metrics()
            self.exit_time = exit_time
            self.exit_price = exit_price
            self.has_exited = True
            

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
    def actual_entry_time(self) -> datetime:
        if self.transactions:
            return self.transactions[0].timestamp
        else:
            return self.entry_time
    
    @property
    def actual_entry_price(self) -> float:
        if self.transactions:
            return self.transactions[0].price_unadjusted
        else:
            return self.entry_price
    
    @property
    def holding_time_minutes(self) -> int:
        return int(self.holding_time.total_seconds() / 60)

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
        # Only include realized P&L and transaction costs
        return self.get_realized_pl - self.total_transaction_costs
        # return self.get_unrealized_pl + self.get_realized_pl - self.total_transaction_costs

    @property 
    def plpc(self) -> float:
        # Only include total sold cost basis
        if self.cost_basis_sold_accum <= 0:
            return 0.0
        return (self.pl / self.cost_basis_sold_accum) * 100
    
    @property 
    def accum_pl(self) -> float:
        return self.pl + self.prev_accum_pl
    
    @property 
    def accum_plpc(self) -> float:
        return self.plpc + self.prev_accum_plpc
        
    @property
    def total_pl(self) -> float:
        return self.get_realized_pl + self.get_unrealized_pl - self.total_transaction_costs

    @property
    def current_cost_basis(self) -> float:
        """Total cost basis including unsold shares"""
        return self.cost_basis_sold_accum + (self.shares * self.avg_entry_price if self.avg_entry_price else 0)

    @property
    def total_plpc(self) -> float:
        basis = self.current_cost_basis
        if basis <= 0:
            return 0.0
        return (self.total_pl / basis) * 100
    
    # @property
    # def net_price_diff(self) -> float:
    #     # net = self.transactions[-1].price_unadjusted - self.transactions[0].price_unadjusted
    #     # return net if self.is_long else -net
    #     return self.position_metrics.net_price_diff_body

    @property
    def side_win_lose_str(self) -> str:
        if not self.has_entered:
            return f"{'Long' if self.is_long else 'Short'} Unentered"
        if self.pl == 0:
            return f"{'Long' if self.is_long else 'Short'} Breakeven"
        return f"{'Long' if self.is_long else 'Short'} {'Win' if self.pl > 0 else 'Lose'}"
    
        
    def calculate_exit_pl_values(self, exit_time: datetime, exit_price: float, 
                            bar: TypedBarData, slippage_factor: float, 
                            atr_sensitivity: float,
                            hypothetical_cost_basis: float) -> Tuple[float, float]:
        """Calculate total P/L and P/L% for exiting all shares at given price."""
        exit_pl, fees = (0.0, 0.0)

        # Determine if we just executed a transaction in this bar
        latest_transaction_time = (self.transactions[-1].timestamp 
                                if self.transactions else None)
        had_transaction_this_bar = (latest_transaction_time == bar.timestamp 
                                if latest_transaction_time else False)

        # Choose appropriate shares and entry price
        shares_to_exit = self.prev_shares if had_transaction_this_bar else self.shares
        use_prev_entry = had_transaction_this_bar

        if shares_to_exit > 0:
            exit_pl, fees = self.calculate_exit_values(
                exit_time, exit_price, shares_to_exit,
                bar, slippage_factor, atr_sensitivity,
                use_prev_avg_entry=use_prev_entry
            )
            exit_pl += self.prev_pl if had_transaction_this_bar else self.pl

        total_pl = exit_pl - fees
        plpc = (total_pl / hypothetical_cost_basis * 100 
                if hypothetical_cost_basis > 0 else 0.0)
                
        return total_pl, plpc


    def record_snapshot(self, bar: TypedBarData, exit_quote_price: float, slippage_factor: float, atr_sensitivity: float, append: bool = True):
        """Record metrics for the current minute.
        
        Should only be called after actions for the current timestamp, if any.
        
        """
        # we want to record even if no trades yet
        self.area.update_bounds(bar.timestamp)
        
        if len(self.position_metrics.snapshots) == 0:
            # For first snapshot or no trades yet
            self.market_value = 0  # Reset/initialize
            self.unrealized_pl = 0
        elif self.avg_entry_price is not None:
            # Only call update_market_value once we have trades
            self.update_market_value(bar.close)
                    

        # Check if we had a transaction this bar
        latest_transaction_time = (self.transactions[-1].timestamp 
                                if self.transactions else None)
        had_transaction_this_bar = (latest_transaction_time == bar.timestamp 
                                if latest_transaction_time else False)

        # Calculate cost basis appropriately
        if had_transaction_this_bar:
            hypothetical_cost_basis = (self.prev_cost_basis_sold_accum + 
                                    (self.prev_shares * self.prev_avg_entry_price 
                                    if self.prev_avg_entry_price else 0))
        else:
            hypothetical_cost_basis = (self.cost_basis_sold_accum + 
                                    (self.shares * self.avg_entry_price 
                                    if self.avg_entry_price else 0))

        # Calculate P/L for each category
        quote_pl, quote_plpc = self.calculate_exit_pl_values(
            bar.timestamp, exit_quote_price, bar, 
            slippage_factor, atr_sensitivity, 
            hypothetical_cost_basis
        )
        
        best_wick_price = bar.high if self.is_long else bar.low
        best_wick_pl, best_wick_plpc = self.calculate_exit_pl_values(
            bar.timestamp, best_wick_price, bar,
            slippage_factor, atr_sensitivity,
            hypothetical_cost_basis
        )
        
        worst_wick_price = bar.low if self.is_long else bar.high
        worst_wick_pl, worst_wick_plpc = self.calculate_exit_pl_values(
            bar.timestamp, worst_wick_price, bar,
            slippage_factor, atr_sensitivity,
            hypothetical_cost_basis
        )
        
        snapshot = PositionSnapshot(
            timestamp=bar.timestamp,
            is_long=self.is_long,
            bar=bar,
            
            shares=self.shares,
            prev_shares=self.prev_shares if had_transaction_this_bar else self.shares,
            max_shares=self.max_shares,
            max_target_shares_limit=self.max_target_shares_limit or 0,
            area_width=self.area.get_range,
            area_buy_price=self.area.get_buy_price,
            avg_entry_price=self.avg_entry_price,
            
            running_pl=quote_pl,  # More accurate running_pl
            cost_basis_sold_accum=self.cost_basis_sold_accum,
            running_plpc=quote_plpc,  # Use hypothetical full exit percentage
            
            realized_pl=self.realized_pl,
            unrealized_pl=self.unrealized_pl,
            total_fees=self.total_transaction_costs,
            has_entered = self.has_entered,
            has_exited = self.has_exited
        )
        
        # assert best_wick_pl >= quote_pl >= worst_wick_pl, f"{best_wick_pl} >= {quote_pl} >= {worst_wick_pl}"
        # assert best_wick_plpc >= quote_plpc >= worst_wick_plpc, f"{best_wick_plpc} >= {quote_plpc} >= {worst_wick_plpc}"
        # NOTE: dont assert; treat wick metrics as approximations
        
        if append:
            self.position_metrics.add_snapshot(snapshot, best_wick_pl, worst_wick_pl, best_wick_plpc, worst_wick_plpc)
        

def export_trades_to_csv(trades: List[TradePosition], filename: str = None):
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

        row = {
            'sym': trade.symbol,
            'date': trade.date,
            'ID': trade.id,
            'AreaID': trade.area.id,
            'Type': side_string,
            # 'Entry Time': trade.entry_time.time().strftime('%H:%M:%S'),
            'Entry Time': trade.actual_entry_time.time().strftime('%H:%M:%S'),
            'Exit Time': trade.exit_time.time().strftime('%H:%M:%S') if trade.exit_time else None,
            # 'Holding Time (min)': trade.holding_time_minutes,
            # 'Entry Price': trade.entry_price,
            'Entry Price': trade.actual_entry_price,
            'Exit Price': trade.exit_price if trade.exit_price else None,
            'Avg Entry Price': trade.avg_entry_price,
            # 'Price Net': trade.net_price_diff,
            
            'Num Transact': len(trade.transactions),
            
            'Initial Qty': trade.initial_shares,
            'Target Qty': trade.target_max_shares,
            # 'Max Qty Reached (%)': round(100*(trade.max_shares_reached / trade.target_max_shares),6),
            # 'Max Qty Time': trade.max_shares_reached_time.time().strftime('%H:%M:%S'),
            'Side Win Lose': trade.side_win_lose_str,
            'Total P/L': round(trade.pl,6),
            'ROE (P/L %)': round(trade.plpc,12),
            # 'Cumulative P/L %': round(cumulative_pct_change,6),
            # 'Slippage Costs': round(trade.slippage_cost,6), # commented out
            # 'Transaction Costs': round(trade.total_transaction_costs,6), # commented out
            # 'Times Buying Power': trade.times_buying_power,
        }

        # Add position metrics directly
        row.update(trade.position_metrics.get_metrics_dict())
        
        # Add bar at entry metrics
        row.update({
            'market_phase_inc': round(trade.bar_at_entry.market_phase_inc,6),
            'market_phase_dec': round(trade.bar_at_entry.market_phase_dec,6),
            # 'shares_per_trade': round(trade.bar_at_entry.shares_per_trade,6),
            'doji_ratio': round(trade.bar_at_entry.doji_ratio,6),
            'mfi_divergence': round(trade.bar_at_entry.mfi_divergence,6),
            'rsi_divergence': round(trade.bar_at_entry.rsi_divergence,6),
            'abs_doji_ratio': round(trade.bar_at_entry.doji_ratio_abs,6),
            'wick_ratio': round(trade.bar_at_entry.wick_ratio,6),
            'nr4_hl_diff': round(trade.bar_at_entry.nr4_hl_diff,6),
            'nr7_hl_diff': round(trade.bar_at_entry.nr7_hl_diff,6),
            'volume_ratio': round(trade.bar_at_entry.volume_ratio,6),
            'ATR_ratio': round(trade.bar_at_entry.ATR_ratio,6),
        })
        # get aggregated metrics per area
        row.update(trade.area.get_metrics_dict(trade.entry_time, prefix=''))
        data.append(row)
        
    df = pd.DataFrame(data)
    bardf = TypedBarData.to_dataframe([trade.bar_at_entry for trade in trades])
    float_cols = bardf.select_dtypes(include=['float']).columns
    bardf[float_cols] = bardf[float_cols].round(10)
    # assert bardf['timestamp'].dt.strftime('%H:%M:%S').equals(df['Entry Time']), f"{bardf['timestamp'].dt.strftime('%H:%M:%S')}\n{df['Entry Time']}"
    df = pd.concat([df,bardf.drop(columns=['timestamp','symbol','time','date'],errors='ignore')],axis=1)
    
    # df.sort_values(by=['Type','ID'], inplace=True) # for quicker subtotalling in excel

    # Fields to flip for shorts
    flip_fields = {
        # Price action
        'doji_ratio',
        'mfi_divergence',
        'rsi_divergence',
        
        # Volume profile
        'buy_vol_balance',
        'buy_hvn_balance',
        'sell_vol_balance',
        'sell_hvn_balance',
        
        # Technical indicators
        'MACD',
        'MACD_signal',
        'MACD_hist',
        'MACD_hist_roc',
        'RSI_roc',
        'MFI_roc',
        
        #
        'trend_strength',
        'central_value_dist'
        
    }
    # Flip values for shorts
    for field in flip_fields:
        if field in df.columns:
            df[field] = df.apply(
                lambda row: -row[field] if not row['Type'].endswith('Long') else row[field], 
                axis=1
            )

    if filename:
        if len(os.path.dirname(filename)) > 0:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        df.to_csv(filename, index=False)
        print(f"Trade summary has been exported to {filename}")
        # df.to_csv(filename.replace('.csv', '.tsv'), index=False, sep='\t')   # Replace the existing df.to_csv line
        # print(f"Trade summary has been exported to {filename}")
    return df


def time_to_minutes(t: datetime_time):
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
    # max_shares_reached: float
    at_max_shares_count: float
    min_area_width: float
    max_area_width: float
    # max_max_shares: int
    # min_max_shares: int
    # realized_pl: float
    # unrealized_pl: float
    side_win_lose_str: str
    pl: float
    plpc: float
    cumulative_pct_change: float
    slippage_cost: float
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
            # max_shares_reached=row['Max Qty Reached (%)'],
            at_max_shares_count=row['At Max Shares Count'],
            min_area_width=row['Min Area Width'],
            max_area_width=row['Max Area Width'],
            # max_max_shares=row['Largest Max Qty'],
            # min_max_shares=row['Smallest Max Qty'],
            # realized_pl=row['Realized P/L'],
            # unrealized_pl=row['Unrealized P/L'],
            side_win_lose_str=row['Side Win Lose'],
            pl=row['Total P/L'],
            plpc=row['ROE (P/L %)'],
            cumulative_pct_change=row['Cumulative P/L %'],
            slippage_cost=row['Slippage Costs'],
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

def plot_cumulative_pl_and_price(trades: List[TradePosition], df: pd.DataFrame, initial_investment: float, 
                                 when_above_max_investment: Optional[List[pd.Timestamp]]=None, filename: Optional[str]=None,
                                 use_plpc=False):
# def plot_cumulative_pl_and_price(trades: List[TradePosition | SimplifiedTradePosition], df: pd.DataFrame, initial_investment: float, when_above_max_investment: Optional[List[pd.Timestamp]]=None, filename: Optional[str]=None):
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
        (df['time'] >= datetime_time(9, 30)) & 
        (df['time'] <= datetime_time(16, 0)) &
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
        volume_data = volume_data[volume_data.index.time != datetime_time(16, 0)]
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
    
    if use_plpc:
        title_str = 'Cumulative P/L % Change'
    else:
        title_str = 'Cumulative P/L $'
        
    
    
    for trade in trades:
        if trade.exit_time and trade.exit_time.date() in trading_days:
            if use_plpc:
                val = trade.plpc
            else:
                val = trade.pl
            
            
            exit_times.append(trade.exit_time)
            running_pl += val
            if trade.is_long:
                running_pl_longs += val
            else:
                running_pl_shorts += val
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
    
    ax2.set_ylabel(f'{title_str}', color='black')
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
    plt.title(f'{symbol}: {title_str} vs Close Price and {volume_label}')
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
    is_time_data = isinstance(df_all['x'].iloc[0], (datetime_time, pd.Timestamp))
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