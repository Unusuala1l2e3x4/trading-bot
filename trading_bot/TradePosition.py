
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

    
# https://alpaca.markets/blog/reg-taf-fees/
# check **Alpaca Securities Brokerage Fee Schedule** in [Alpaca Documents Library](https://alpaca.markets/disclosures) for most up-to-date rates
SEC_FEE_RATE = 0.0000278  # $27.80 per $1,000,000
FINRA_TAF_RATE = 0.000166  # $166 per 1,000,000 shares
FINRA_TAF_MAX = 8.30  # Maximum $8.30 per trade

    
# subject to change
@jit(nopython=True)
def calculate_num_sub_positions(times_buying_power: float) -> int:
    # For now, always return 1
    return 1
    # In the future, you can modify this to return different numbers based on times_buying_power

@jit(nopython=True)
def calculate_shares_per_sub(total_shares: int, num_subs: int) -> np.ndarray:
    if num_subs <= 0:
        return np.empty(0, dtype=np.int32)  # Return an empty array instead of raising an exception
    
    # Distribute shares evenly among sub-positions
    shares_per_sub = total_shares // num_subs
    assert total_shares % num_subs == 0

    return np.full(num_subs, shares_per_sub, dtype=np.int32)


@jit(nopython=True)
def calculate_slippage(is_long: bool, price: float, trade_size: int, volume: float, avg_volume: float, slippage_factor: float, is_entry: bool) -> float:
    # Use the average of current volume and average volume, with a minimum to avoid division by zero
    effective_volume = max((volume + avg_volume) / 2, 1)
    
    slippage = slippage_factor * (float(trade_size) / effective_volume)
    
    if is_long:
        if is_entry:
            return price * (1 + slippage)  # Increase price for long entries
        else:
            return price * (1 - slippage)  # Decrease price for long exits
    else:  # short
        if is_entry:
            return price * (1 - slippage)  # Decrease price for short entries
        else:
            return price * (1 + slippage)  # Increase price for short exits


# @jit(nopython=True)
# def calculate_slippage(price: float, trade_size: int, avg_volume: float, volatility: float, is_long: bool, is_entry: bool, slippage_factor: float, beta: float = 0.7) -> float:
#     # Use avg_volume directly as effective volume
#     effective_volume = max(avg_volume, 1)
    
#     # Compute the relative trade size
#     relative_size = trade_size / effective_volume
    
#     # Calculate slippage using a non-linear model
#     slippage = slippage_factor * (relative_size ** beta)
    
#     # Adjust slippage for volatility
#     slippage *= (1 + volatility)
    
#     # Adjust the price based on the direction of the trade
#     if is_long:
#         if is_entry:
#             adjusted_price = price * (1 + slippage)
#         else:
#             adjusted_price = price * (1 - slippage)
#     else:
#         if is_entry:
#             adjusted_price = price * (1 - slippage)
#         else:
#             adjusted_price = price * (1 + slippage)
    
#     return adjusted_price




@dataclass
class Transaction:
    timestamp: datetime
    shares: int
    price: float
    is_entry: bool # Was it a buy (entry) or sell (exit)
    is_long: bool
    transaction_cost: float # total of next 3 fields
    finra_taf: float
    sec_fee: float  # > 0 for sells (long exits and short entries)
    stock_borrow_cost: float # 0 if not is_long and not is_entry (short exits)
    value: float  # Positive if profit, negative if loss (before transaction costs are applied)
    vwap: float
    realized_pl: Optional[float] = None # None if is_entry is True
    

@dataclass
class SubPosition:
    is_long: bool
    entry_time: datetime
    entry_price: float
    shares: int
    cash_committed: float
    market_value: float = field(init=False)
    unrealized_pl: float = field(init=False)
    realized_pl: float = 0.0
    transactions: List[Transaction] = field(default_factory=list)
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None

    def __post_init__(self):
        self.update_market_value(self.entry_price)

    def update_market_value(self, current_price: float):
        self.market_value = self.shares * current_price if self.is_long else -self.shares * current_price
        self.unrealized_pl = self.market_value - (self.shares * self.entry_price)

    def add_transaction(self, transaction: Transaction):
        self.transactions.append(transaction)
        
    def calculate_average_entry_price(self) -> float:
        total_cost = 0
        total_shares = 0
        for transaction in self.transactions:
            if transaction.is_entry:
                total_cost += transaction.price * transaction.shares
                total_shares += transaction.shares
        return total_cost / total_shares if total_shares > 0 else 0

    def calculate_realized_pl(self, exit_price: float, shares_sold: int) -> float:
        if self.is_long:
            return (exit_price - self.entry_price) * shares_sold
        else:
            return (self.entry_price - exit_price) * shares_sold
        
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
    unrealized_pl: float = field(default=0.0)
    realized_pl: float = 0.0
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
    
    # Ensure objects are compared based on date and id
    def __eq__(self, other):
        if isinstance(other, TouchArea):
            return self.id == other.id and self.date == other.date
        return False

    # Ensure that objects have a unique hash based on date and id
    def __hash__(self):
        return hash((self.id, self.date))
    
    
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
        self.market_value = 0
        self.shares = 0
        self.last_price = self.entry_price
        self.cash_committed = 0
        self.max_shares = self.initial_shares
        assert self.times_buying_power <= 4
        
        self.logger = self.setup_logger(logging.WARNING)
        
        self.log(f'initial_shares {self.initial_shares}',level=logging.DEBUG)

    @property
    def is_open(self) -> bool:
        return any(sp.exit_time is None for sp in self.sub_positions)

    @property
    def total_shares(self) -> int:
        return sum(sp.shares for sp in self.sub_positions if sp.exit_time is None)

    def calculate_slippage(self, price: float, trade_size: int, volume: float, avg_volume: float, slippage_factor: float, is_entry: bool) -> float:
        return calculate_slippage(self.is_long, price, trade_size, volume, avg_volume, slippage_factor, is_entry)

    # def calculate_slippage(self, price: float, trade_size: int, avg_volume: float, volatility: float, is_entry: bool, slippage_factor: float, beta: float = 0.7) -> float:
    #     return calculate_slippage(price, trade_size, avg_volume, volatility, self.is_long, is_entry, slippage_factor, beta)
    
    def update_market_value(self, current_price: float):
        self.last_price = current_price
        for sp in self.sub_positions:
            if sp.shares > 0:
                sp.update_market_value(current_price)
        self.unrealized_pl = sum(sp.unrealized_pl for sp in self.sub_positions if sp.shares > 0)
        self.market_value = sum(sp.market_value for sp in self.sub_positions if sp.shares > 0)

        t = abs(self.market_value - sum(sp.market_value for sp in self.sub_positions if sp.shares > 0))
        assert t < 1e-8, \
            f"Market value mismatch: {self.market_value} != {sum(sp.market_value for sp in self.sub_positions if sp.shares > 0)}, diff {t}"


    @staticmethod
    def calculate_num_sub_positions(times_buying_power: float) -> int:
        return calculate_num_sub_positions(times_buying_power)
    
    @staticmethod
    def calculate_shares_per_sub(total_shares: int, num_subs: int) -> np.ndarray:
        return calculate_shares_per_sub(total_shares, num_subs)
    
    def calculate_transaction_cost(self, shares: int, price: float, is_entry: bool, timestamp: datetime, sub_position: SubPosition) -> float:
        is_sell = (self.is_long and not is_entry) or (not self.is_long and is_entry)
        finra_taf = min(FINRA_TAF_RATE * shares, FINRA_TAF_MAX) if is_sell else 0
        trade_value = price * shares
        sec_fee = SEC_FEE_RATE * trade_value if is_sell else 0
        
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
            assert cumulative_shares == shares, f"Mismatch in shares calculation: {cumulative_shares} != {shares}"
            stock_borrow_cost = total_cost
        return finra_taf, sec_fee, stock_borrow_cost



    def add_transaction(self, timestamp: datetime, shares: int, price: float, is_entry: bool, vwap: float, sub_position: SubPosition, sp_realized_pl: Optional[float] = None):
        finra_taf, sec_fee, stock_borrow_cost = self.calculate_transaction_cost(shares, price, is_entry, timestamp, sub_position)
        transaction_cost = finra_taf + sec_fee + stock_borrow_cost
        value = -shares * price if is_entry else shares * price
        
        if is_entry:
            self.log(f'add_transaction {shares} {transaction_cost}',level=logging.DEBUG)
        
        transaction = Transaction(timestamp, shares, price, is_entry, self.is_long, transaction_cost, finra_taf, sec_fee, stock_borrow_cost, value, vwap, sp_realized_pl)
        self.transactions.append(transaction)
        sub_position.add_transaction(transaction)
        
        if not is_entry:
            if sp_realized_pl is None:
                raise ValueError("sp_realized_pl must be provided for exit transactions")
            self.realized_pl += sp_realized_pl
            
        self.log(f"Transaction added - {'Entry' if is_entry else 'Exit'}, Shares: {shares}, Price: {price:.4f}, "
            f"Value: {value:.4f}, Cost: {transaction_cost:.4f}, Realized PnL: {sp_realized_pl if sp_realized_pl is not None else 'N/A'}",level=logging.DEBUG)

        return transaction_cost


    def initial_entry(self, vwap: float, volume: float, avg_volume: float, slippage_factor: float):
        return self.partial_entry(self.entry_time, self.entry_price, self.initial_shares, vwap, volume, avg_volume, slippage_factor)
        
    def partial_entry(self, entry_time: datetime, entry_price: float, shares_to_buy: int, vwap: float, volume: float, avg_volume: float, slippage_factor: float):
        self.log(f"partial_entry - Time: {entry_time}, Price: {entry_price:.4f}, Shares to buy: {shares_to_buy}",level=logging.DEBUG)
        self.log(f"Current position - Shares: {self.shares}, Cash committed: {self.cash_committed:.4f}",level=logging.DEBUG)

        adjusted_price = self.calculate_slippage(entry_price, shares_to_buy, volume, avg_volume, slippage_factor, is_entry=True)
        # adjusted_price = self.calculate_slippage(entry_price, shares_to_buy, avg_volume, volatility, True, slippage_factor, beta)

        new_total_shares = self.shares + shares_to_buy
        new_num_subs = self.calculate_num_sub_positions(self.times_buying_power)
        assert new_total_shares % new_num_subs == 0, f"Total shares {new_total_shares} is not evenly divisible by number of sub-positions {new_num_subs}"

        additional_cash_committed = (shares_to_buy * entry_price) / self.times_buying_power

        active_sub_positions = [sp for sp in self.sub_positions if sp.shares > 0]
        target_shares = self.calculate_shares_per_sub(new_total_shares, new_num_subs)
        shares_bought_per_position = []

        fees = 0
        shares_added = 0

        self.log(f"Target shares per sub-position: {target_shares}",level=logging.DEBUG)

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

                    fees += self.add_transaction(entry_time, shares_to_add, adjusted_price, is_entry=True, vwap=vwap, sub_position=sp)
                    shares_added += shares_to_add
                    self.log(f"Adding to sub-position {i} - Entry price: {sp.entry_price:.4f}, Shares added: {shares_to_add}, "
                        f"Cash committed: {sub_cash_committed:.4f}, Old shares: {old_shares}, New shares: {sp.shares}",level=logging.DEBUG)
                    
                    shares_bought_per_position.append(shares_to_add)
            else:
                # New sub-position
                sub_cash_committed = (target * entry_price) / self.times_buying_power
                new_sub = SubPosition(self.is_long, entry_time, adjusted_price, target, sub_cash_committed)
                self.sub_positions.append(new_sub)
                fees += self.add_transaction(entry_time, target, adjusted_price, is_entry=True, vwap=vwap, sub_position=new_sub)
                shares_added += target
                self.log(f"Created new sub-position {i} - Entry price: {entry_price:.4f}, Shares: {target}, "
                    f"Cash committed: {sub_cash_committed:.4f}",level=logging.DEBUG)
                
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

        self.log(f"Partial entry complete - Shares added: {shares_added}, New total shares: {self.shares}, "
            f"New cash committed: {self.cash_committed:.4f}",level=logging.DEBUG)
        self.log("Current sub-positions:",level=logging.DEBUG)
        for i, sp in enumerate(self.sub_positions):
            if sp.shares > 0:
                self.log(f"  Sub-position {i}: Shares: {sp.shares}, Entry price: {sp.entry_price:.4f}",level=logging.DEBUG)
        # self.log(f"Current Realized PnL: {self.realized_pl:.4f}, "
        #     f"Total Transaction Costs: {sum(t.transaction_cost for t in self.transactions):.4f}",level=logging.DEBUG)
        
        return additional_cash_committed, fees, shares_bought_per_position

    def partial_exit(self, exit_time: datetime, exit_price: float, shares_to_sell: int, vwap: float, volume: float, avg_volume: float, slippage_factor: float):
        num_subs = self.calculate_num_sub_positions(self.times_buying_power)
        new_total_shares = self.shares - shares_to_sell
        assert new_total_shares % num_subs == 0, f"Total shares {new_total_shares} is not evenly divisible by number of sub-positions {num_subs}"

        self.log(f"partial_exit - Time: {exit_time}, Price: {exit_price:.4f}, Shares to sell: {shares_to_sell}",level=logging.DEBUG)
        self.log(f"Current position - Shares: {self.shares}, Cash committed: {self.cash_committed:.4f}",level=logging.DEBUG)
        
        adjusted_price = self.calculate_slippage(exit_price, shares_to_sell, volume, avg_volume, slippage_factor, is_entry=False)
        # adjusted_price = self.calculate_slippage(exit_price, shares_to_sell, avg_volume, volatility, False, slippage_factor, beta)

        cash_released = 0
        realized_pl = 0
        fees = 0

        active_sub_positions = [sp for sp in self.sub_positions if sp.shares > 0]
        total_shares = sum(sp.shares for sp in active_sub_positions)
        shares_sold_per_position = []

        for sp in active_sub_positions:
            assert sp.shares == int(float(total_shares)/float(num_subs)), (sp.shares, total_shares, num_subs)
            shares_sold = int(shares_to_sell * (float(sp.shares) / float(total_shares)))
            if shares_sold > 0:
                sub_cash_released = (float(shares_sold) / float(sp.shares)) * sp.cash_committed
                # sp_realized_pl = (adjusted_price - sp.entry_price) * shares_sold if self.is_long else (sp.entry_price - adjusted_price) * shares_sold
                sp_realized_pl = sp.calculate_realized_pl(adjusted_price, shares_sold)
                
                old_shares = sp.shares
                sp.shares -= shares_sold
                sp.cash_committed -= sub_cash_released
                sp.realized_pl += sp_realized_pl

                cash_released += sub_cash_released
                realized_pl += sp_realized_pl
                
                fees += self.add_transaction(exit_time, shares_sold, adjusted_price, is_entry=False, vwap=vwap, sub_position=sp, sp_realized_pl=sp_realized_pl)

                self.log(f"Selling from sub-position - Entry price: {sp.entry_price:.4f}, Shares sold: {shares_sold}, "
                    f"Realized PnL: {sp_realized_pl:.4f}, Cash released: {sub_cash_released:.4f}, "
                    f"Old shares: {old_shares}, New shares: {sp.shares}",level=logging.DEBUG)
                
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

        self.log(f"Partial exit complete - New shares: {self.shares}, Cash released: {cash_released:.4f}, Realized PnL: {realized_pl:.4f}",level=logging.DEBUG)
        self.log("Remaining sub-positions:",level=logging.DEBUG)
        for i, sp in enumerate(self.sub_positions):
            if sp.shares > 0:
                self.log(f"  Sub-position {i}: Shares: {sp.shares}, Entry price: {sp.entry_price:.4f}",level=logging.DEBUG)

        return realized_pl, cash_released, fees, shares_sold_per_position

    # customized to specific strategy
    def update_stop_price(self, current_price: float, current_timestamp: datetime):
        # Update the bounds of the TouchArea
        self.area.update_bounds(current_timestamp)
        
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
                    # assert transaction.stock_borrow_cost != 0
                    total_cost += transaction.stock_borrow_cost
                # else:
                #     assert transaction.stock_borrow_cost == 0
        
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
        # NOTE: currently, initial_balance is passed in as the invest amount divided my times_buying_power, so reflective of usable cash but not margin
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
    print(f"Trade summary has been exported to {filename}")


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
        print(f"Graph has been saved as {filename}")
    else:
        plt.show()
        
    plt.close()