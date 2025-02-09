from dataclasses import dataclass, field
from datetime import datetime, date, timedelta, time as datetime_time
from typing import List, Set, Tuple, Optional, Dict
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

from zoneinfo import ZoneInfo
ny_tz = ZoneInfo("America/New_York")

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
    cost_basis_sold: Optional[float] = 0.0
    cost_basis_sold_accum: Optional[float] = 0.0

    @property
    def pl(self):
        return self.realized_pl - self.transaction_cost
    
    @property 
    def plpc(self) -> float:
        if self.cost_basis_sold <= 0:
            return 0.0
        return (self.pl / self.cost_basis_sold) * 100

    def plpc_with_accum(self, cost_basis_sold_accum) -> float:
        if cost_basis_sold_accum <= 0:
            return 0.0
        return (self.pl / cost_basis_sold_accum) * 100

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
    is_simulated: Optional[bool] = False
    balance_before_simulation: Optional[float] = None
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
    
    was_profitable: bool = False
    
    prev_accum_pl: Optional[float] = 0.0
    prev_accum_plpc: Optional[float] = 0.0
    
    failed_full_exit_count: int = 0
    
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
    
    exited_from_stop_order: bool = False
    
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
        self.set_full_entry_price()
            
        # Initial setup only - update_stop_price will handle the rest
        self.has_crossed_full_entry = False
        self.max_target_shares_limit = None
        # self.max_shares_reached = 0
        # self.max_shares_reached_time = self.entry_time
        self.initial_shares = None
        
        self.max_close = self.min_close = self.bar_at_commit.close
        
        if self.is_simulated:
            assert self.balance_before_simulation is not None
        else:
            assert self.balance_before_simulation is None
           
        if self.is_long:
            self.current_stop_price = -np.inf
            self.current_stop_price_2 = -np.inf
        else:
            self.current_stop_price = np.inf
            self.current_stop_price_2 = np.inf
            
        self.update_stop_price(self.bar_at_commit, None, self.entry_time)
        self.position_metrics = PositionMetrics(self.is_long, prior_relevant_bars=self.prior_relevant_bars)
    
    
    def set_full_entry_price(self):
        if self.is_long:
            self.full_entry_price = self.area.get_buy_price + (self.area.get_range * self.gradual_entry_range_multiplier) # area bounds already updated in TradingStrategy.create_new_position
        else:
            self.full_entry_price = self.area.get_buy_price - (self.area.get_range * self.gradual_entry_range_multiplier)
        
        
    def calculate_target_shares_from_price(self, current_price: float) -> int:
        """Calculate target shares based on how close price is to full entry"""
        self.set_full_entry_price()
        
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
            self.unrealized_pl = self.market_value - self.cost_basis
        else:
            self.market_value = -self.shares * current_price  # Keep negative for accounting
            self.unrealized_pl = self.cost_basis + self.market_value  # Entry - Current

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
                        realized_pl: float = 0.0,
                        cost_basis_sold: float = 0.0):
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
                                  realized_pl=realized_pl,
                                  cost_basis_sold=cost_basis_sold
        )
        self.transactions.append(transaction)
        self.log(f"{timestamp.strftime("%H:%M")} Transaction added - {'Entry' if is_entry else 'Exit'} {shares} shares at Price: {price:.4f}, AvgEntryPrice: {self.avg_entry_price:4f}, "
                 f"Value: {value:.4f}, Cost: {transaction_cost:.4f}, Realized PnL: {self.realized_pl}", level=logging.DEBUG)
        self.log(f"Now holding {self.shares}/{self.max_shares} with avg_entry_price {self.avg_entry_price:.4f}", level=logging.INFO)
        
        
        return transaction_cost

    def increase_max_shares(self, shares: int, bar: TypedBarData, current_timestamp: datetime):
        if self.max_shares < shares:
            self.max_shares = max(self.max_shares, shares)
            self.max_max_shares = max(self.max_max_shares, self.max_shares)
            self.set_max_target_shares_limit(bar, current_timestamp)

    def decrease_max_shares(self, shares: int, bar: TypedBarData, current_timestamp: datetime):
        if self.max_shares > shares:
            self.max_shares = min(self.max_shares, shares)
            self.min_max_shares = min(self.min_max_shares, self.max_shares)
            self.set_max_target_shares_limit(bar, current_timestamp)

    def set_max_shares(self, shares: int, bar: TypedBarData, current_timestamp: datetime):
        self.increase_max_shares(shares, bar, current_timestamp)
        self.decrease_max_shares(shares, bar, current_timestamp)
    
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
        self.prev_cost_basis_sold_accum = self.cost_basis_sold_accum
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
            old_value = self.cost_basis
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
        
        # self.log(f"Holding {self.shares}/{self.max_shares} {(self.shares/self.max_shares)*100:.1f}%",level=logging.INFO)
        
        return cash_needed, fees, shares_to_buy

    def partial_exit(self, exit_time: datetime, exit_price: float, shares_to_sell: int, 
                    bar: TypedBarData, slippage_factor: float, atr_sensitivity: float):
                    
        self.prev_shares = self.shares
        self.prev_pl = self.pl
        self.prev_avg_entry_price = self.avg_entry_price
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
        
        fees = self.add_transaction(exit_time, shares_to_sell, exit_price, adjusted_price, False, bar, slippage_price_change, realized_pl=exit_pl, cost_basis_sold=cost_basis_sold)

    
        # if self.has_crossed_full_entry:
        #     self.max_target_shares_limit = self.shares
        #     # self.max_target_shares_limit = (self.shares + self.max_shares) / 2
        
        # self.log(f"Holding {self.shares}/{self.max_shares} {(self.shares/self.max_shares)*100:.1f}%",level=logging.INFO)
    
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
    
    def max_shares_ratio_threshold(self, ratio: float = 0.75):
        assert 0 <= ratio <= 1
        return self.shares >= self.max_shares * np.clip(ratio, 0, 1)

    # NOTE: updates stop price, as well as min/max close, max high, min low
    def update_stop_price(self, bar: TypedBarData, prev_bar: TypedBarData, current_timestamp: datetime, exit_quote_price: float = None, 
                          slippage_factor: float = None, atr_sensitivity: float = None):
        # should_exit_price = bar.close
        # should_exit_price_2 = bar.close
        
        if exit_quote_price is None:
            is_profitable, quote_pl, quote_plpc = False, 0.0, 0.0
        else:
            is_profitable, quote_pl, quote_plpc = self.is_profitable(bar, exit_quote_price, slippage_factor, atr_sensitivity)
        
        accum_running_pl = self.prev_accum_pl + quote_pl
        # stop_trading_signal = accum_running_pl < -25 # or accum_running_pl > 200
        # stop_trading_signal = accum_running_pl < -37.5
        # stop_trading_signal = accum_running_pl < -50
        # stop_trading_signal = quote_pl < -25
        # stop_trading_signal = quote_pl < -37.5
        # stop_trading_signal = quote_pl < -100
        stop_trading_signal = False
        
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
        
        had_transaction_last_bar = prev_bar is not None and self.was_latest_transaction_at(prev_bar.timestamp)
        should_have_exited_halfway = False
        # if prev_bar is not None:
        was_price_above_stop = (had_transaction_last_bar and (
                (self.is_long and self.transactions[-1].price > self.halfway_price) or 
                (not self.is_long and self.transactions[-1].price < self.halfway_price))
            ) or \
            (prev_bar is not None and not had_transaction_last_bar and (
                (self.is_long and prev_bar.close > self.halfway_price) or 
                (not self.is_long and prev_bar.close < self.halfway_price))
            )
                
        
        # is_profitable_at_halfway_price = False
        # if slippage_factor and atr_sensitivity:
        #     is_profitable_at_halfway_price, _, _ = self.is_profitable(bar, self.halfway_price, slippage_factor*2, atr_sensitivity*2)
        
        # should_have_exited_halfway = was_price_above_stop and self.reached_halfway_price(should_exit_price) and self.max_shares_ratio_threshold(0.25) and is_profitable_at_halfway_price
            
        # and \
        #     (
        #         (self.is_long and prev_halfway_price > prev_stop_price) or \
        #         (not self.is_long and prev_halfway_price < prev_stop_price)
        #     )
        
        

        # self.area.update_bounds(current_timestamp)
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
        
        self.set_halfway_price(bar.close)
        # self.set_halfway_price(exit_quote_price)

        # should_exit_halfway = False
        # should_exit_halfway = was_price_above_stop and self.reached_halfway_price(bar.close) and self.max_shares_ratio_threshold(0.25) and is_profitable
        # should_exit_halfway = self.reached_halfway_price(bar.close) and self.max_shares_ratio_threshold(0.25) # and is_profitable #and self.max_shares_ratio_threshold(0.75)
        
        # should_exit_halfway = self.reached_exit_ema(bar, exit_quote_price, slippage_factor, atr_sensitivity) # and is_profitable
        # should_exit_halfway = self.max_shares_ratio_threshold(0.25) and is_profitable
        
        # self.update_market_value(exit_quote_price) # NOTE: update_market_value should use quotes data, but not necessary here. quote price isnt determined yet anyways.
        self.log(f"area {self.area.id}: get_range {self.area.get_range:.4f}",level=logging.DEBUG)
        
        # if not self.has_crossed_full_entry:
        self.set_max_target_shares_limit(bar, current_timestamp)

        # return self.reached_current_stop_price(bar.close), self.reached_current_stop_price_2(bar.close), should_have_exited, should_have_exited_2, prev_stop_price, prev_stop_price_2

        self.was_profitable = is_profitable

        reached_stop = self.reached_current_stop_price(bar.close)
                
        # self.reached_exit_ema(bar, exit_quote_price, slippage_factor, atr_sensitivity) or 
        # return stop_trading_signal or self.reached_exit_ema(bar, exit_quote_price, slippage_factor, atr_sensitivity) or reached_stop,
        return self.reached_exit_ema(bar, exit_quote_price, slippage_factor, atr_sensitivity) or reached_stop, reached_stop, self.reached_current_stop_price_2(bar.close), \
            should_have_exited_halfway, \
            should_have_exited, should_have_exited_2, prev_halfway_price, prev_stop_price, prev_stop_price_2
        
        # return self.reached_current_stop_price(should_exit_price), self.reached_current_stop_price_2(should_exit_price_2), should_have_exited, should_have_exited_2, prev_stop_price, prev_stop_price_2


    def set_max_target_shares_limit(self, bar: TypedBarData, current_timestamp: datetime):
        # Check if price has crossed full entry threshold
        # if not self.has_crossed_full_entry:
        if (self.is_long and bar.close >= self.full_entry_price) or \
        (not self.is_long and bar.close <= self.full_entry_price):
            # Full entry condition met - go to maximum size
            self.has_crossed_full_entry = True
            self.full_entry_time = current_timestamp
            self.max_target_shares_limit = self.max_shares
            if self.initial_shares is None:
                self.initial_shares = self.max_target_shares_limit
                # TODO: consider using half of self.max_shares (if initial shares is at max shares)
                
            # self.log(f"100% of target shares ({self.max_shares}) reached at entry",level=logging.INFO)
            self.log(f"100% of target shares ({self.max_shares}) reached {(current_timestamp - self.entry_time).total_seconds()/60 :.2f} min after entry",level=logging.INFO)
        else:
            # Check for close price crossing buy price
            current_limit = self.max_target_shares_limit or 0 # default 0 for comparisons of limits
            
            # NOTE: if doing gradual entry
            new_limit = self.calculate_target_shares_from_price(bar.close)
            
            
            # or, need to call calculate_target_shares_from_price BEFORE max_shares was reduced due to volume changes
            
            
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
                # elif new_limit < current_limit and self.max_shares_ratio_threshold():
                #     self.max_target_shares_limit = new_limit
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
                # elif new_limit < current_limit and self.max_shares_ratio_threshold():
                #     self.max_target_shares_limit = new_limit
        # else:
        #     # self.max_target_shares_limit = min(self.max_target_shares_limit, self.max_shares)
        #     # self.max_target_shares_limit = self.max_shares
        #     self.max_target_shares_limit = self.max_shares
            

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
        
        if not self.has_exited:
            assert prev_snapshot.timestamp == bar.timestamp - timedelta(minutes=1), (prev_snapshot.timestamp, bar.timestamp - timedelta(minutes=1))
        
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
        return self.avg_entry_price is not None and self.halfway_price is not None and \
                (
                    (self.is_long and current_price <= self.halfway_price) or \
                    (not self.is_long and current_price >= self.halfway_price)
                )
    # @property
    # def max_close_to_avg_entry_price(self):
    #     if self.is_long:
    #         return (self.max_close - self.avg_entry_price) / self.avg_entry_price
    #     else:
    #         return (self.avg_entry_price - self.min_close) / self.avg_entry_price

    def set_halfway_price(self, max_price: float, distance_ratio: float = 0.85):
        if self.is_long:
            if self.avg_entry_price is None:
                ret = -np.inf
            else:
                # assert self.max_close >= self.avg_entry_price 
                # ret = (self.max_close + self.avg_entry_price) / 2
                dist = self.max_close - self.avg_entry_price
                # ret = max(self.avg_entry_price + dist*distance_ratio, self.avg_entry_price)
                # ret = max(self.avg_entry_price + dist*distance_ratio, self.avg_entry_price - dist*distance_ratio)
                
                # dist = max(max_price - self.avg_entry_price, 0)
                # if dist == 0:
                #     ret = max_price
                # else:
                ret = self.avg_entry_price + dist*distance_ratio
        else:
            if self.avg_entry_price is None:
                ret = np.inf
            else:
                # assert self.min_close <= self.avg_entry_price
                # ret = (self.min_close + self.avg_entry_price) / 2
                dist = self.avg_entry_price - self.min_close
                # ret = min(self.avg_entry_price - dist*distance_ratio, self.avg_entry_price)
                # ret = min(self.avg_entry_price - dist*distance_ratio, self.avg_entry_price + dist*distance_ratio)
                
                # dist = max(self.avg_entry_price - max_price, 0)
                # if dist == 0:
                #     ret = max_price
                # else:
                ret = self.avg_entry_price - dist*distance_ratio
                
        self.halfway_price = ret
        return ret
            
    def close(self, exit_time: datetime, exit_price: float):
        if not self.has_exited:
            if self.has_entered:
                self.area.record_entry_exit(self.actual_entry_time, self.actual_entry_price, exit_time, exit_price)
            else:
                assert self.pl == 0
                self.bar_at_entry = self.bar_at_commit # no better value
                self.position_metrics.finalize_metrics()
            self.exit_time = exit_time
            self.exit_price = exit_price
            self.has_exited = True
            
            self.max_target_shares_limit = 0
            
        self.log('CLOSING POSITION',level=logging.DEBUG)
            

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
        return (self.exit_time or datetime.now(tz=ny_tz)) - self.actual_entry_time
    
    def holding_time_minutes_at_bar(self, current_timestamp) -> timedelta:
        return int(((self.exit_time or current_timestamp) - self.actual_entry_time).total_seconds() / 60)
    
    @property
    def actual_entry_time(self) -> datetime:
        if self.transactions:
            return self.transactions[0].timestamp
        else:
            return self.entry_time
    
    @property
    def actual_entry_price(self) -> float:
        if self.transactions:
            # return self.transactions[0].price_unadjusted
            return self.transactions[0].price
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
    def cost_basis(self) -> float:
        """Total cost basis including unsold shares"""
        return self.shares * self.avg_entry_price if self.avg_entry_price else 0
    
    @property
    def prev_cost_basis(self) -> float:
        """Total cost basis including unsold shares"""
        return self.prev_shares * self.prev_avg_entry_price if self.prev_avg_entry_price else 0

    @property
    def total_cost_basis(self) -> float:
        """Total cost basis including unsold shares"""
        return self.cost_basis_sold_accum + self.cost_basis
    
    @property
    def prev_total_cost_basis(self) -> float:
        """Total cost basis including unsold shares"""
        return self.prev_cost_basis_sold_accum + self.prev_cost_basis

    @property
    def total_plpc(self) -> float:
        basis = self.total_cost_basis
        if basis <= 0:
            return 0.0
        return (self.total_pl / basis) * 100
    
    @property
    def side_win_lose_str(self) -> str:
        if not self.has_entered:
            return f"{'Long' if self.is_long else 'Short'} Unentered"
        if self.pl == 0:
            return f"{'Long' if self.is_long else 'Short'} Breakeven"
        return f"{'Long' if self.is_long else 'Short'} {'Win' if self.pl > 0 else 'Lose'}"


    def was_latest_transaction_at(self, current_timestamp: datetime):
        latest_transaction_time = self.transactions[-1].timestamp if self.transactions else None
        return latest_transaction_time == current_timestamp if latest_transaction_time else False


    def calculate_exit_pl_values(self, exit_time: datetime, exit_price: float, 
                            bar: TypedBarData, slippage_factor: float, 
                            atr_sensitivity: float, debug=False) -> Tuple[float, float]:
        """
        Calculate total P/L and P/L% for exiting all shares at given price.
        Handles timing of transaction execution and appropriate value selection.
        
        Returns:
            Tuple[float, float]: (total_pl, plpc)
        """
        # Determine appropriate state based on bar timing
        # latest_transaction_time = (self.transactions[-1].timestamp 
        #                         if self.transactions else None)
        # had_transaction_this_bar = (latest_transaction_time == bar.timestamp 
        #                         if latest_transaction_time else False) # and self.exit_time != exit_time
        had_transaction_this_bar = self.was_latest_transaction_at(bar.timestamp)
        

        # Choose values based on transaction timing
        shares_to_exit = self.prev_shares if had_transaction_this_bar else self.shares
        use_prev_entry = had_transaction_this_bar
        base_pl = self.prev_pl if had_transaction_this_bar else self.pl
        
        # Calculate costs
        exit_pl, fees = (0.0, 0.0)
        if shares_to_exit > 0:
            exit_pl, fees = self.calculate_exit_values(
                exit_time, exit_price, shares_to_exit,
                bar, slippage_factor, atr_sensitivity,
                use_prev_avg_entry=use_prev_entry
            )
            exit_pl += base_pl

        # Get appropriate cost basis
        hypothetical_cost_basis = (self.prev_total_cost_basis 
                                if had_transaction_this_bar 
                                else self.total_cost_basis)

        # Calculate final values
        total_pl = exit_pl - fees
        plpc = (total_pl / hypothetical_cost_basis * 100 
                if hypothetical_cost_basis > 0 else 0.0)
                
        if debug:
            if had_transaction_this_bar:
                print(f'{exit_time} HAD transaction ({self.prev_shares}) {self.shares} . '
                    f'({self.prev_total_cost_basis:.2f}) {self.total_cost_basis:.2f}')
            else:
                print(f'{exit_time}  NO transaction  {self.prev_shares} ({self.shares}). '
                    f'{self.prev_total_cost_basis:.2f} ({self.total_cost_basis:.2f})')
            print(f'total_pl: {total_pl:.2f}, plpc: {plpc:.2f}')
        
        return total_pl, plpc, shares_to_exit
        
        
    def is_profitable(self, bar: TypedBarData, exit_quote_price: float, slippage_factor: float, atr_sensitivity: float):
        quote_pl, quote_plpc, shares_to_exit = self.calculate_exit_pl_values(
            bar.timestamp, exit_quote_price, bar,
            slippage_factor, atr_sensitivity
        )
        assert shares_to_exit == self.shares
        return quote_pl > 0, quote_pl, quote_plpc
    

    def record_snapshot(self, bar: TypedBarData, exit_quote_price: float, slippage_factor: float, atr_sensitivity: float, append: bool = True):
        """Record metrics for the current minute.
        
        Should only be called after actions for the current timestamp, if any.
        
        """
        # we want to record even if no trades yet
        # self.area.update_bounds(bar.timestamp)
        
        if len(self.position_metrics.snapshots) == 0:
            # For first snapshot or no trades yet
            self.market_value = 0  # Reset/initialize
            self.unrealized_pl = 0
        elif self.avg_entry_price is not None:
            # Only call update_market_value once we have trades
            self.update_market_value(bar.close)
                    

        # Calculate theoretical exits
        best_wick_price = bar.high if self.is_long else bar.low
        worst_wick_price = bar.low if self.is_long else bar.high
                
        # Calculate P/L for various exit scenarios
        quote_pl, quote_plpc, shares_to_exit = self.calculate_exit_pl_values(
            bar.timestamp, exit_quote_price, bar,
            slippage_factor, atr_sensitivity
        )
        
        best_wick_pl, best_wick_plpc, shares_to_exit = self.calculate_exit_pl_values(
            bar.timestamp, best_wick_price, bar,
            slippage_factor, atr_sensitivity
        )
        
        worst_wick_pl, worst_wick_plpc, shares_to_exit = self.calculate_exit_pl_values(
            bar.timestamp, worst_wick_price, bar,
            slippage_factor, atr_sensitivity
        )
            
        snapshot = PositionSnapshot(
            timestamp=bar.timestamp,
            is_long=self.is_long,
            bar=bar,
            
            shares=self.shares,
            prev_shares=shares_to_exit,
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
        if trade.transactions:
            ret = trade.transactions[0].timestamp
            assert trade.position_metrics.entry_snapshot_index is not None
            assert ret == trade.position_metrics.snapshots[trade.position_metrics.entry_snapshot_index].bar.timestamp, \
            (ret, trade.position_metrics.snapshots[trade.position_metrics.entry_snapshot_index].bar.timestamp, \
                trade.entry_time, trade.actual_entry_time)
            
            first_shares = trade.transactions[0].shares
        else:
            first_shares = 0
            
            
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
            'Commit Time': trade.entry_time.time().strftime('%H:%M:%S'),
            'Entry Time': trade.actual_entry_time.time().strftime('%H:%M:%S'),
            'Exit Time': trade.exit_time.time().strftime('%H:%M:%S') if trade.exit_time else None,
            'exited_from_stop_order': trade.exited_from_stop_order,
            'failed_full_exit_count': trade.failed_full_exit_count,
            # 'Holding Time (min)': trade.holding_time_minutes,
            # 'Entry Price': trade.entry_price,
            'Entry Price': trade.actual_entry_price,
            'Exit Price': trade.exit_price if trade.exit_price else None,
            'Avg Entry Price': trade.avg_entry_price,
            # 'Price Net': trade.net_price_diff,
            
            'Num Transact': len(trade.transactions),
            
            'Last Max Shares': trade.max_shares,            
            # 'Initial Qty': trade.initial_shares,
            'Initial Qty': first_shares,
            'Target Qty': trade.target_max_shares,
            # 'Max Qty Reached (%)': round(100*(trade.max_shares_reached / trade.target_max_shares),6),
            # 'Max Qty Time': trade.max_shares_reached_time.time().strftime('%H:%M:%S'),
            'Side Win Lose': trade.side_win_lose_str,
            'Total P/L': round(trade.pl,6),
            'ROE (P/L %)': round(trade.plpc,12),
            # 'Cumulative P/L %': round(cumulative_pct_change,6),
            'Slippage Costs': round(trade.slippage_cost,6), # commented out
            'Transaction Costs': round(trade.total_transaction_costs,6), # commented out
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
        row.update(trade.area.get_metrics_dict(trade.actual_entry_time, prefix='entry_'))
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
        
        # 
        'entry_position_vwap_dist',
        
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
