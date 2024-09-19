
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time
from typing import List, Tuple, Optional
from TouchArea import TouchArea
import math
import os
import pandas as pd
from datetime import datetime, date
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from numba import jit

import logging

debug = False
def debug_print(*args, **kwargs):
    if debug:
        print(*args, **kwargs)

    
# https://alpaca.markets/blog/reg-taf-fees/
# check **Alpaca Securities Brokerage Fee Schedule** in [Alpaca Documents Library](https://alpaca.markets/disclosures) for most up-to-date rates
# SEC_FEE_RATE = 0.000008  # $8 per $1,000,000
SEC_FEE_RATE = 0.0000278  # $27.80 per $1,000,000
FINRA_TAF_RATE = 0.000166  # $166 per 1,000,000 shares
FINRA_TAF_MAX = 8.30  # Maximum $8.30 per trade

    
# subject to change
@jit(nopython=True)
def calculate_num_sub_positions(times_buying_power: float) -> int:
    if times_buying_power <= 2:
        return 1
    else:
        return 2  # Use 2 sub-positions when times_buying_power > 2
    # If capital is much higher, more sub-positions (and distribution among multiple symbols) may be needed to reduce slippage.

@jit(nopython=True)
def calculate_shares_per_sub(total_shares: int, num_subs: int) -> np.ndarray:
    if num_subs <= 0:
        return np.empty(0, dtype=np.int32)  # Return an empty array instead of raising an exception
    
    # Ensure divisible by num_subs
    adjusted_total_shares = total_shares - (total_shares % num_subs)
    shares_per_sub = adjusted_total_shares // num_subs
    
    # Create the shares distribution array
    shares_distribution = np.full(num_subs, shares_per_sub, dtype=np.int32)
    
    return shares_distribution

@jit(nopython=True)
def estimate_entry_cost(total_shares: int, times_buying_power: float, existing_sub_positions: np.ndarray = np.array([])) -> float:
    num_subs = calculate_num_sub_positions(times_buying_power)
    target_shares = calculate_shares_per_sub(total_shares, num_subs)
    
    total_cost = 0.0
    
    if existing_sub_positions.size > 0:  # Check if the array is not empty
        for i in range(num_subs):
            target = target_shares[i]
            if i < len(existing_sub_positions):
                shares_to_add = max(0, target - existing_sub_positions[i])
                finra_taf = min(FINRA_TAF_RATE * shares_to_add, FINRA_TAF_MAX)
            else:
                shares_to_add = target
                finra_taf = min(FINRA_TAF_RATE * shares_to_add, FINRA_TAF_MAX)
            total_cost += finra_taf
    else:
        for i in range(num_subs):
            target = target_shares[i]
            finra_taf = min(FINRA_TAF_RATE * target, FINRA_TAF_MAX)
            total_cost += finra_taf

    return total_cost


@dataclass
class Transaction:
    timestamp: datetime
    shares: int
    price: float
    is_entry: bool # Was it a buy (entry) or sell (exit)
    transaction_cost: float # total of next 3 fields
    finra_taf: float
    sec_fee: float  # 0 if is_entry is True
    stock_borrow_cost: float # 0 if it is a long.
    value: float  # Positive if profit, negative if loss (before transaction costs are applied)
    vwap: float
    realized_pnl: Optional[float] = None # None if is_entry is True

@dataclass
class SubPosition:
    entry_time: datetime
    entry_price: float
    shares: int
    cash_committed: float
    market_value: float = field(init=False)
    unrealized_pnl: float = field(init=False)
    realized_pnl: float = 0.0
    transactions: List[Transaction] = field(default_factory=list)
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None

    def __post_init__(self):
        self.update_market_value(self.entry_price)

    def update_market_value(self, current_price: float):
        self.market_value = self.shares * current_price
        self.unrealized_pnl = self.market_value - (self.shares * self.entry_price)

    def add_transaction(self, transaction: Transaction):
        self.transactions.append(transaction)

@dataclass
class TradePosition:
    date: date
    id: int
    area: TouchArea
    is_long: bool
    entry_time: datetime
    initial_balance: float
    initial_shares: int
    use_margin: bool
    is_marginable: bool
    times_buying_power: float
    actual_margin_multiplier: float
    entry_price: float
    market_value: float = 0.0
    shares: int = 0
    partial_entry_count: int = 0
    partial_exit_count: int = 0
    max_shares: int = field(init=False)
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    sub_positions: List[SubPosition] = field(default_factory=list)
    transactions: List[Transaction] = field(default_factory=list)
    current_stop_price: Optional[float] = None
    current_stop_price_2: Optional[float] = None
    max_price: Optional[float] = None
    min_price: Optional[float] = None
    last_price: float = field(default=0.0)
    cash_committed: float = field(init=False)
    is_simulated: float = field(init=False)
    unrealized_pnl: float = field(default=0.0)
    realized_pnl: float = 0.0
    log_level: Optional[int] = logging.INFO
    # stock_borrow_rate: float = 0.003    # Default to 30 bps (0.3%) annually
    stock_borrow_rate: float = 0.03      # Default to 300 bps (3%) annually
    
    # Note: This class assumes intraday trading. No overnight interest is calculated.
     
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
    
    def setup_logger(self, log_level=logging.INFO):
        logger = logging.getLogger('TradePosition')
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

    def log(self, message, level=logging.INFO):
        self.logger.log(level, message)
    
    def __post_init__(self):
        # self.market_value = self.initial_shares * self.entry_price
        self.market_value = 0
        self.shares = 0
        self.last_price = self.entry_price
        self.cash_committed = 0
        self.max_shares = self.initial_shares
        assert self.times_buying_power <= 4
        
        self.logger = self.setup_logger(logging.WARNING)
        
        debug_print(f'initial_shares {self.initial_shares}')


    def initial_entry(self, vwap: float, volume: float, avg_volume: float, slippage_factor: float):
        return self.partial_entry(self.entry_time, self.entry_price, self.initial_shares, vwap, volume, avg_volume, slippage_factor)
        
    @property
    def is_open(self) -> bool:
        return any(sp.exit_time is None for sp in self.sub_positions)

    @property
    def total_shares(self) -> int:
        return sum(sp.shares for sp in self.sub_positions if sp.exit_time is None)

    def calculate_slippage(self, price: float, trade_size: int, volume: float, avg_volume: float, slippage_factor: float, is_entry: bool) -> float:
        # Use the average of current volume and average volume, with a minimum to avoid division by zero
        effective_volume = max((volume + avg_volume) / 2, 1)
        
        slippage = slippage_factor * (float(trade_size) / effective_volume)
        
        if self.is_long:
            if is_entry:
                return price * (1 + slippage)  # Increase price for long entries
            else:
                return price * (1 - slippage)  # Decrease price for long exits
        else:  # short
            if is_entry:
                return price * (1 - slippage)  # Decrease price for short entries
            else:
                return price * (1 + slippage)  # Increase price for short exits

    def update_market_value(self, current_price: float):
        self.last_price = current_price
        for sp in self.sub_positions:
            if sp.shares > 0:
                sp.update_market_value(current_price)
        self.unrealized_pnl = sum(sp.unrealized_pnl for sp in self.sub_positions if sp.shares > 0)
        self.market_value = sum(sp.market_value for sp in self.sub_positions if sp.shares > 0)

        t = abs(self.market_value - sum(sp.market_value for sp in self.sub_positions if sp.shares > 0))
        assert t < 1e-8, \
            f"Market value mismatch: {self.market_value} != {sum(sp.market_value for sp in self.sub_positions if sp.shares > 0)}, diff {t}"

    def partial_exit(self, exit_time: datetime, exit_price: float, shares_to_sell: int, vwap: float, volume: float, avg_volume: float, slippage_factor: float):
        num_subs = self.calculate_num_sub_positions(self.times_buying_power)
        new_total_shares = self.shares - shares_to_sell
        assert new_total_shares % num_subs == 0, f"Total shares {new_total_shares} is not evenly divisible by number of sub-positions {num_subs}"

        debug_print(f"DEBUG: partial_exit - Time: {exit_time}, Price: {exit_price:.4f}, Shares to sell: {shares_to_sell}")
        debug_print(f"DEBUG: Current position - Shares: {self.shares}, Cash committed: {self.cash_committed:.4f}")
        
        adjusted_price = self.calculate_slippage(exit_price, shares_to_sell, volume, avg_volume, slippage_factor, is_entry=False)

        cash_released = 0
        realized_pnl = 0
        fees = 0

        active_sub_positions = [sp for sp in self.sub_positions if sp.shares > 0]
        total_shares = sum(sp.shares for sp in active_sub_positions)
        shares_sold_per_position = []

        for sp in active_sub_positions:
            assert sp.shares == int(float(total_shares)/float(num_subs)), (sp.shares, total_shares, num_subs)
            shares_sold = int(shares_to_sell * (float(sp.shares) / float(total_shares)))
            if shares_sold > 0:
                sub_cash_released = (float(shares_sold) / float(sp.shares)) * sp.cash_committed
                sp_realized_pnl = (adjusted_price - sp.entry_price) * shares_sold if self.is_long else (sp.entry_price - adjusted_price) * shares_sold
                
                old_shares = sp.shares
                sp.shares -= shares_sold
                sp.cash_committed -= sub_cash_released
                sp.realized_pnl += sp_realized_pnl
                # sp.update_market_value(exit_price)

                cash_released += sub_cash_released
                realized_pnl += sp_realized_pnl
                
                fees += self.add_transaction(exit_time, shares_sold, adjusted_price, is_entry=False, vwap=vwap, sub_position=sp, sp_realized_pnl=sp_realized_pnl)

                debug_print(f"DEBUG: Selling from sub-position - Entry price: {sp.entry_price:.4f}, Shares sold: {shares_sold}, "
                    f"Realized PnL: {sp_realized_pnl:.4f}, Cash released: {sub_cash_released:.4f}, "
                    f"Old shares: {old_shares}, New shares: {sp.shares}")
                
                shares_sold_per_position.append(shares_sold)

        self.shares -= shares_to_sell
        self.cash_committed -= cash_released
        self.update_market_value(adjusted_price)
        self.partial_exit_count += 1

        t = abs(self.cash_committed - sum(sp.cash_committed for sp in self.sub_positions if sp.shares > 0))
        assert t < 1e-8, \
            f"Cash committed mismatch: {self.cash_committed} != {sum(sp.cash_committed for sp in self.sub_positions if sp.shares > 0)}, diff {t}"
        assert self.shares == sum(sp.shares for sp in self.sub_positions if sp.shares > 0), \
            f"Shares mismatch: {self.shares} != {sum(sp.shares for sp in self.sub_positions if sp.shares > 0)}"

        debug_print(f"DEBUG: Partial exit complete - New shares: {self.shares}, Cash released: {cash_released:.4f}, Realized PnL: {realized_pnl:.4f}")
        debug_print("DEBUG: Remaining sub-positions:")
        for i, sp in enumerate(self.sub_positions):
            if sp.shares > 0:
                debug_print(f"  Sub-position {i}: Shares: {sp.shares}, Entry price: {sp.entry_price:.4f}")

        return realized_pnl, cash_released, fees, shares_sold_per_position

    def partial_entry(self, entry_time: datetime, entry_price: float, shares_to_buy: int, vwap: float, volume: float, avg_volume: float, slippage_factor: float):
        debug_print(f"DEBUG: partial_entry - Time: {entry_time}, Price: {entry_price:.4f}, Shares to buy: {shares_to_buy}")
        debug_print(f"DEBUG: Current position - Shares: {self.shares}, Cash committed: {self.cash_committed:.4f}")

        adjusted_price = self.calculate_slippage(entry_price, shares_to_buy, volume, avg_volume, slippage_factor, is_entry=True)

        new_total_shares = self.shares + shares_to_buy
        new_num_subs = self.calculate_num_sub_positions(self.times_buying_power)
        assert new_total_shares % new_num_subs == 0, f"Total shares {new_total_shares} is not evenly divisible by number of sub-positions {new_num_subs}"

        additional_cash_committed = (shares_to_buy * entry_price) / self.times_buying_power

        active_sub_positions = [sp for sp in self.sub_positions if sp.shares > 0]
        target_shares = self.calculate_shares_per_sub(new_total_shares, new_num_subs)
        shares_bought_per_position = []

        fees = 0
        shares_added = 0

        debug_print(f"DEBUG: Target shares per sub-position: {target_shares}")

        for i, target in enumerate(target_shares):
            if i < len(active_sub_positions):
                # Existing sub-position
                sp = active_sub_positions[i]
                shares_to_add = target - sp.shares
                if shares_to_add > 0:
                    sub_cash_committed = (shares_to_add * entry_price) / self.times_buying_power
                    old_shares = sp.shares
                    sp.shares += shares_to_add
                    sp.cash_committed += sub_cash_committed
                    # sp.update_market_value(entry_price)
                    fees += self.add_transaction(entry_time, shares_to_add, adjusted_price, is_entry=True, vwap=vwap, sub_position=sp)
                    shares_added += shares_to_add
                    debug_print(f"DEBUG: Adding to sub-position {i} - Entry price: {sp.entry_price:.4f}, Shares added: {shares_to_add}, "
                        f"Cash committed: {sub_cash_committed:.4f}, Old shares: {old_shares}, New shares: {sp.shares}")
                    
                    shares_bought_per_position.append(shares_to_add)
            else:
                # New sub-position
                sub_cash_committed = (target * entry_price) / self.times_buying_power
                new_sub = SubPosition(entry_time, adjusted_price, target, sub_cash_committed)
                self.sub_positions.append(new_sub)
                fees += self.add_transaction(entry_time, target, adjusted_price, is_entry=True, vwap=vwap, sub_position=new_sub)
                shares_added += target
                debug_print(f"DEBUG: Created new sub-position {i} - Entry price: {entry_price:.4f}, Shares: {target}, "
                    f"Cash committed: {sub_cash_committed:.4f}")
                
                shares_bought_per_position.append(target)

        self.shares += shares_added
        self.cash_committed += additional_cash_committed
        self.update_market_value(adjusted_price)
        self.partial_entry_count += 1

        t = abs(self.cash_committed - sum(sp.cash_committed for sp in self.sub_positions if sp.shares > 0))
        assert t < 1e-8, \
            f"Cash committed mismatch: {self.cash_committed} != {sum(sp.cash_committed for sp in self.sub_positions if sp.shares > 0)}, diff {t}"
        assert self.shares == sum(sp.shares for sp in self.sub_positions if sp.shares > 0), \
            f"Shares mismatch: {self.shares} != {sum(sp.shares for sp in self.sub_positions if sp.shares > 0)}"

        debug_print(f"DEBUG: Partial entry complete - Shares added: {shares_added}, New total shares: {self.shares}, "
            f"New cash committed: {self.cash_committed:.4f}")
        debug_print("DEBUG: Current sub-positions:")
        for i, sp in enumerate(self.sub_positions):
            if sp.shares > 0:
                debug_print(f"  Sub-position {i}: Shares: {sp.shares}, Entry price: {sp.entry_price:.4f}")
        # debug_print(f"DEBUG: Current Realized PnL: {self.realized_pnl:.4f}, "
        #     f"Total Transaction Costs: {sum(t.transaction_cost for t in self.transactions):.4f}")
        
        return additional_cash_committed, fees, shares_bought_per_position

    @staticmethod
    def calculate_num_sub_positions(times_buying_power: float) -> int:
        return calculate_num_sub_positions(times_buying_power)
    
    @staticmethod
    def calculate_shares_per_sub(total_shares: int, num_subs: int) -> np.ndarray:
        return calculate_shares_per_sub(total_shares, num_subs)
    
    @staticmethod
    def estimate_entry_cost(total_shares: int, times_buying_power: float, existing_sub_positions: Optional[np.ndarray] = np.array([])) -> float:
        return estimate_entry_cost(total_shares, times_buying_power, existing_sub_positions)
    
    def calculate_transaction_cost(self, shares: int, price: float, is_entry: bool, timestamp: datetime, sub_position: SubPosition) -> float:
        finra_taf = min(FINRA_TAF_RATE * shares, FINRA_TAF_MAX)
        sec_fee = 0
        if not is_entry:  # SEC fee only applies to exits
            trade_value = price * shares
            sec_fee = SEC_FEE_RATE * trade_value
        
        stock_borrow_cost = 0
        if not self.is_long and not is_entry:  # Stock borrow cost applies only to short position exits
            daily_borrow_rate = self.stock_borrow_rate / 360
            total_cost = 0
            
            # Walk backwards to find relevant entry transactions
            relevant_entries = []
            cumulative_shares = 0
            for transaction in reversed(sub_position.transactions):
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
                
                # debug_print(f"Entry: {entry_transaction.timestamp}, Total Shares: {entry_transaction.shares}, Relevant Shares: {relevant_shares}, Days Held: {days_held:.2f}")
            
            assert cumulative_shares == shares, f"Mismatch in shares calculation: {cumulative_shares} != {shares}"
            stock_borrow_cost = total_cost

        # debug_print(f"FINRA TAF: {finra_taf:.6f}, SEC Fee: {sec_fee:.6f}, Stock Borrow Cost: {stock_borrow_cost:.6f}")
        # return finra_taf + sec_fee + stock_borrow_cost
        return finra_taf, sec_fee, stock_borrow_cost



    def add_transaction(self, timestamp: datetime, shares: int, price: float, is_entry: bool, vwap: float, sub_position: SubPosition, sp_realized_pnl: Optional[float] = None):
        finra_taf, sec_fee, stock_borrow_cost = self.calculate_transaction_cost(shares, price, is_entry, timestamp, sub_position)
        transaction_cost = finra_taf + sec_fee + stock_borrow_cost
        value = -shares * price if is_entry else shares * price
        
        if is_entry:
            debug_print('add_transaction', shares, transaction_cost)
        
        transaction = Transaction(timestamp, shares, price, is_entry, transaction_cost, finra_taf, sec_fee, stock_borrow_cost, value, vwap, sp_realized_pnl)
        self.transactions.append(transaction)
        sub_position.add_transaction(transaction)
        
        if not is_entry:
            if sp_realized_pnl is None:
                raise ValueError("sp_realized_pnl must be provided for exit transactions")
            self.realized_pnl += sp_realized_pnl
            
        debug_print(f"DEBUG: Transaction added - {'Entry' if is_entry else 'Exit'}, Shares: {shares}, Price: {price:.4f}, "
            f"Value: {value:.4f}, Cost: {transaction_cost:.4f}, Realized PnL: {sp_realized_pnl if sp_realized_pnl is not None else 'N/A'}")

        return transaction_cost

    # customized to specific strategy
    def update_stop_price(self, current_price: float, current_timestamp: datetime):
        # Update the bounds of the TouchArea
        # self.area.update_bounds(current_timestamp)
        
        if self.is_long:
            self.max_price = max(self.max_price or self.entry_price, current_price)
            self.current_stop_price = self.max_price - self.area.get_range
            self.current_stop_price_2 = self.max_price - self.area.get_range*3 # simple logic for now
        else:
            self.min_price = min(self.min_price or self.entry_price, current_price)
            self.current_stop_price = self.min_price + self.area.get_range
            self.current_stop_price_2 = self.min_price + self.area.get_range*3 # simple logic for nowe
        
        self.update_market_value(current_price)

        self.log(f"area {self.area.id}: get_range {self.area.get_range:.4f}")

        return self.should_exit(current_price), self.should_exit_2(current_price)

    def should_exit(self, current_price: float) -> bool:
        return (self.is_long and current_price <= self.current_stop_price) or \
               (not self.is_long and current_price >= self.current_stop_price)

    def should_exit_2(self, current_price: float) -> bool:
        return (self.is_long and current_price <= self.current_stop_price_2) or \
               (not self.is_long and current_price >= self.current_stop_price_2),

    def close(self, exit_time: datetime, exit_price: float):
        self.exit_time = exit_time
        self.exit_price = exit_price
        for sp in self.sub_positions:
            if sp.exit_time is None:
                sp.exit_time = exit_time
                sp.exit_price = exit_price

    @property
    def total_stock_borrow_cost(self) -> float:
        if self.is_long:
            return 0.0
        
        total_cost = 0.0
        for sub_position in self.sub_positions:
            for transaction in sub_position.transactions:
                if not transaction.is_entry:  # We only consider exit transactions
                    total_cost += transaction.stock_borrow_cost
        
        return total_cost


    @property
    def holding_time(self) -> timedelta:
        if not self.sub_positions:
            return timedelta(0)
        start_time = min(sp.entry_time for sp in self.sub_positions)
        end_time = max(sp.exit_time or datetime.now() for sp in self.sub_positions)
        return end_time - start_time

    @property
    def entry_transaction_costs(self) -> float:
        return sum(t.transaction_cost for t in self.transactions if t.is_entry) # / self.times_buying_power

    @property
    def exit_transaction_costs(self) -> float:
        return sum(t.transaction_cost for t in self.transactions if not t.is_entry) # / self.times_buying_power

    @property
    def total_transaction_costs(self) -> float:
        return sum(t.transaction_cost for t in self.transactions) # / self.times_buying_power

    @property
    def get_unrealized_pnl(self) -> float:
        return self.unrealized_pnl # / self.times_buying_power

    @property
    def get_realized_pnl(self) -> float:
        return self.realized_pnl # / self.times_buying_power

    @property
    def profit_loss(self) -> float:
        return self.get_unrealized_pnl + self.get_realized_pnl - self.total_transaction_costs

    @property
    def profit_loss_pct(self) -> float:
        return (self.profit_loss / self.initial_balance) * 100

    @property
    def price_diff(self) -> float:
        if not self.sub_positions or any(sp.exit_time is None for sp in self.sub_positions):
            return 0
        avg_entry_price = sum(sp.entry_price * sp.shares for sp in self.sub_positions) / sum(sp.shares for sp in self.sub_positions)
        avg_exit_price = sum(sp.exit_price * sp.shares for sp in self.sub_positions) / sum(sp.shares for sp in self.sub_positions)
        diff = avg_exit_price - avg_entry_price
        return diff if self.is_long else -diff
                
    @property
    def total_investment(self) -> float:
        return sum(sp.entry_price * sp.shares for sp in self.sub_positions)

    @property
    def margin_used(self) -> float:
        return self.total_investment - self.initial_balance



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
        cumulative_pct_change += trade.profit_loss_pct
        row = {
            'date': trade.date,
            'ID': trade.id,
            'AreaID': trade.area.id,
            'Type': 'Long' if trade.is_long else 'Short',
            'Entry Time': trade.entry_time.time().strftime('%H:%M:%S'),
            'Exit Time': trade.exit_time.time().strftime('%H:%M:%S') if trade.exit_time else None,
            'Holding Time (min)': trade.holding_time.total_seconds() / 60,
            'Entry Price': float(trade.entry_price),
            'Exit Price': float(trade.exit_price) if trade.exit_price else None,
            'Initial Shares': trade.initial_shares,
            'Realized P/L': float(trade.get_realized_pnl),
            'Unrealized P/L': float(trade.get_unrealized_pnl),
            'Total P/L': float(trade.profit_loss),
            'ROE (P/L %)': float(trade.profit_loss_pct),
            'Cumulative P/L %': float(cumulative_pct_change),
            'Transaction Costs': float(trade.total_transaction_costs),
            'Margin Multiplier': float(trade.actual_margin_multiplier),
            'Times Buying Power': float(trade.times_buying_power)
        }
        data.append(row)
    df = pd.DataFrame(data)
    if len(os.path.dirname(filename)) > 0:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    debug_print(f"Trade summary has been exported to {filename}")


def time_to_minutes(t: time):
    return t.hour * 60 + t.minute - (9 * 60 + 30)

def plot_cumulative_pnl_and_price(trades: List[TradePosition], df: pd.DataFrame, initial_investment: float, when_above_max_investment: Optional[List[pd.Timestamp]]=None, filename: Optional[str]=None):
    """
    Create a graph that plots the cumulative profit/loss at each corresponding exit time
    overlaid on the close price from the DataFrame, using a dual y-axis.
    Empty intervals before and after intraday trading are removed.
    
    Args:
    trades (list): List of TradePosition objects
    df (pd.DataFrame): DataFrame containing the price data
    initial_investment (float): Initial investment balance for normalization
    filename (str): Name of the image file to be created
    """
    
    symbol = df.index.get_level_values('symbol')[0]
    
    timestamps = df.index.get_level_values('timestamp')
    # Filter df to include only intraday data
    df['time'] = timestamps.time
    df['date'] = timestamps.date
    df_intraday = df[(df['time'] >= time(9, 30)) & (df['time'] <= time(16, 0))].copy()
    timestamps = df_intraday.index.get_level_values('timestamp')
    
    # Create a continuous index
    unique_dates = sorted(df_intraday['date'].unique())
    continuous_index = []
    cumulative_minutes = 0
    
    for date in unique_dates:
        day_data = df_intraday[df_intraday['date'] == date]
        day_minutes = day_data['time'].apply(time_to_minutes)
        continuous_index.extend(cumulative_minutes + day_minutes)
        cumulative_minutes += 390  # 6.5 hours of trading
    
    df_intraday['continuous_index'] = continuous_index
    
    # Prepare data for plotting
    exit_times = [trades[0].entry_time]
    cumulative_pnl = [0]
    cumulative_pnl_longs = [0]
    cumulative_pnl_shorts = [0]
    running_pnl = 0
    running_pnl_longs = 0
    running_pnl_shorts = 0
    
    for trade in trades:
        if trade.exit_time:
            exit_times.append(trade.exit_time)
            # running_pnl += trade.profit_loss
            # cumulative_pnl.append(100 * running_pnl / initial_investment)
            running_pnl += trade.profit_loss_pct
            if trade.is_long:
                running_pnl_longs += trade.profit_loss_pct
            else:
                running_pnl_shorts += trade.profit_loss_pct
            cumulative_pnl.append(running_pnl)
            cumulative_pnl_longs.append(running_pnl_longs)
            cumulative_pnl_shorts.append(running_pnl_shorts)

    # Create figure and primary y-axis
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot close price on primary y-axis
    ax1.plot(df_intraday['continuous_index'], df_intraday['close'], color='gray', label='Close Price')
    ax1.set_xlabel('Trading Time (minutes)')
    ax1.set_ylabel('Close Price', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    
    if when_above_max_investment and len(when_above_max_investment) > 0:
        # Convert when_above_max_investment to continuous index
        above_max_continuous_index = []
        for timestamp in when_above_max_investment:
            date = timestamp.date()
            minute = (timestamp.time().hour - 9) * 60 + (timestamp.time().minute - 30)
            days_passed = unique_dates.index(date)
            above_max_continuous_index.append(days_passed * 390 + minute)

        # Get the minimum close price for y-value of the points
        min_close = df_intraday['close'].min()

        # Plot points for when above max investment
        ax1.plot(above_max_continuous_index, [min_close] * len(above_max_continuous_index), 
                    color='red', label='Above Max Investment')

    # Create secondary y-axis
    ax2 = ax1.twinx()
    
    # Convert exit times to continuous index
    exit_continuous_index = []
    for exit_time in exit_times:
        exit_date = exit_time.date()
        exit_minute = time_to_minutes(exit_time.time())
        days_passed = unique_dates.index(exit_date)
        exit_continuous_index.append(days_passed * 390 + exit_minute)
    
    # Plot cumulative P/L on secondary y-axis
    ax2.plot(exit_continuous_index, cumulative_pnl, color='green', label='All')
    ax2.plot(exit_continuous_index, cumulative_pnl_longs, color='blue', label='Longs')
    ax2.plot(exit_continuous_index, cumulative_pnl_shorts, color='yellow', label='Shorts')
    # ax2.set_ylabel('Cumulative P/L', color='green')
    ax2.set_ylabel('Cumulative P/L % Change', color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    
    # Set title and legend
    plt.title(f'{symbol}: Cumulative P/L % Change vs Close Price')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
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
    # ax1.tick_params(axis='x', which='major', colors='red')

    # Format minor ticks
    ax1.tick_params(axis='x', which='minor', bottom=True)

    # Add gridlines for major ticks (week starts)
    ax1.grid(which='major', axis='x', linestyle='--', alpha=0.7)
        
    # Use a tight layout
    plt.tight_layout()
    
    if filename:
        # Save the figure
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
        debug_print(f"Graph has been saved as {filename}")
    else:
        plt.show()
        
    plt.close()
    

