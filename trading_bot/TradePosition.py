from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
from TouchArea import TouchArea
import math

@dataclass
class Transaction:
    timestamp: datetime
    shares: int
    price: float
    is_entry: bool
    transaction_cost: float
    value: float  # This will be the cost (negative) or revenue (positive)
    realized_pnl: Optional[float] = None 
    
@dataclass
class SubPosition:
    entry_time: datetime
    entry_price: float
    shares: int
    cash_committed: float
    market_value: float = field(init=False)
    unrealized_pnl: float = field(init=False)
    realized_pnl: float = 0
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
    # initial_cash_used: float
    entry_price: float
    market_value: float = 0
    shares: int = 0
    partial_entry_count: int = 0
    partial_exit_count: int = 0
    # cumulative_realized_pnl = 0
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    sub_positions: List[SubPosition] = field(default_factory=list)
    transactions: List[Transaction] = field(default_factory=list)
    current_stop_price: Optional[float] = None
    max_price: Optional[float] = None
    min_price: Optional[float] = None
    last_price: float = field(default=0.0)
    cash_committed: float = field(init=False)
    unrealized_pnl: float = field(default=0.0)
    realized_pnl: float = 0
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
    FINRA_TAF_RATE = 0.000119  # per share
    SEC_FEE_RATE = 22.90 / 1_000_000  # per dollar
    
    def __post_init__(self):
        # self.market_value = self.initial_shares * self.entry_price
        self.market_value = 0
        self.shares = 0
        self.last_price = self.entry_price
        self.cash_committed = 0
        assert self.times_buying_power <= 4
        
        print(f'initial_shares {self.initial_shares}')
        
        self.partial_entry(self.entry_time, self.entry_price, self.initial_shares)

        
    @property
    def is_open(self) -> bool:
        return any(sp.exit_time is None for sp in self.sub_positions)

    @property
    def total_shares(self) -> int:
        return sum(sp.shares for sp in self.sub_positions if sp.exit_time is None)

    # def update_last_price(self, price: float):
    #     old_market_value = self.market_value
    #     self.last_price = price
    #     self.market_value = self.shares * price
    #     self.unrealized_pnl = self.market_value - (self.shares * self.entry_price)
    #     return self.market_value - old_market_value
        
                        
    def update_market_value(self, current_price: float):
        self.last_price = current_price
        for sp in self.sub_positions:
            if sp.shares > 0:
                sp.update_market_value(current_price)
        self.unrealized_pnl = sum(sp.unrealized_pnl for sp in self.sub_positions if sp.shares > 0)
        self.market_value = sum(sp.market_value for sp in self.sub_positions if sp.shares > 0)

        assert abs(self.market_value - sum(sp.market_value for sp in self.sub_positions if sp.shares > 0)) < 1e-8, \
            f"Market value mismatch: {self.market_value:.2f} != {sum(sp.market_value for sp in self.sub_positions if sp.shares > 0):.2f}"

    def partial_exit(self, exit_time: datetime, exit_price: float, shares_to_sell: int):
        print(f"DEBUG: Entering partial_exit - Time: {exit_time}, Price: {exit_price}, Shares to sell: {shares_to_sell}")
        print(f"DEBUG: Current position - Shares: {self.shares}, Cash committed: {self.cash_committed:.2f}")
        
        initial_shares = self.shares
        initial_cash_committed = self.cash_committed
        cash_released = 0
        realized_pnl = 0
        remaining_shares_to_sell = shares_to_sell
        
        print(f'BEFORE EXIT cash_committed: ',[f'{sp.cash_committed:.2f} ({sp.shares})' for sp in self.sub_positions if sp.exit_time is None],f' = {sum([sp.cash_committed for sp in self.sub_positions if sp.exit_time is None]):.2f}')
        print(f'BEFORE EXIT market_value:   ',[f'{sp.market_value:.2f} ({sp.shares})' for sp in self.sub_positions if sp.exit_time is None],f' = {sum([sp.market_value for sp in self.sub_positions if sp.exit_time is None]):.2f}')
        # assert abs(self.cash_committed - sum([sp.cash_committed for sp in self.sub_positions if sp.exit_time is None])) < 1e-8, (self.cash_committed, sum([sp.cash_committed for sp in self.sub_positions if sp.exit_time is None]))
        
        for sp in self.sub_positions:
            if sp.shares > 0 and remaining_shares_to_sell > 0:
                shares_sold = min(sp.shares, remaining_shares_to_sell)
                

                # sub_cash_released = (shares_sold / sp.shares) * sp.market_value / self.times_buying_power
                sub_cash_released = (shares_sold / sp.shares) * sp.cash_committed
                sp_realized_pnl = (exit_price - sp.entry_price) * shares_sold if self.is_long else (sp.entry_price - exit_price) * shares_sold
                
                print(f'  ({shares_sold} / {sp.shares}) * {sp.cash_committed:.2f} = {sub_cash_released:.2f}')
                print(f"DEBUG: Selling from sub-position - Entry price: {sp.entry_price:.4f}, Shares sold: {shares_sold}, Realized PnL: {sp_realized_pnl:.2f}, Cash released: {sub_cash_released:.2f}")
                
                old_shares = sp.shares
                sp.shares -= shares_sold
                sp.cash_committed -= sub_cash_released
                sp.realized_pnl += sp_realized_pnl
                sp.update_market_value(exit_price)
                
                cash_released += sub_cash_released
                realized_pnl += sp_realized_pnl
                
                self.add_transaction(exit_time, shares_sold, exit_price, is_entry=False, sub_position=sp, sp_realized_pnl=sp_realized_pnl)
                remaining_shares_to_sell -= shares_sold

                if sp.shares == 0:
                    sp.exit_time = exit_time
                    sp.exit_price = exit_price

                print(f"DEBUG: Selling from sub-position - Entry price: {sp.entry_price:.4f}, Shares sold: {shares_sold}, "
                      f"Realized PnL: {sp_realized_pnl:.2f}, Cash released: {sub_cash_released:.2f}, "
                      f"Old shares: {old_shares}, New shares: {sp.shares}")

        self.shares -= shares_to_sell
        self.cash_committed -= cash_released
        self.update_market_value(exit_price)
        self.partial_exit_count += 1
        
        print(f'AFTER EXIT cash_committed:  ',[f'{sp.cash_committed:.2f} ({sp.shares})' for sp in self.sub_positions if sp.exit_time is None],f' = {sum([sp.cash_committed for sp in self.sub_positions if sp.exit_time is None]):.2f}')
        print(f'AFTER EXIT market_value:    ',[f'{sp.market_value:.2f} ({sp.shares})' for sp in self.sub_positions if sp.exit_time is None],f' = {sum([sp.market_value for sp in self.sub_positions if sp.exit_time is None]):.2f}')
        
        
        assert abs(self.cash_committed - sum([sp.cash_committed for sp in self.sub_positions if sp.exit_time is None])) < 1e-8, (self.cash_committed, sum([sp.cash_committed for sp in self.sub_positions if sp.exit_time is None]))
        
        assert self.shares == initial_shares - shares_to_sell, f"Share mismatch after partial exit: {self.shares} != {initial_shares - shares_to_sell}"
        assert abs(self.cash_committed - (initial_cash_committed - cash_released)) < 1e-8, f"Cash committed mismatch: {self.cash_committed:.2f} != {initial_cash_committed - cash_released:.2f}"

        assert abs(self.cash_committed - sum(sp.cash_committed for sp in self.sub_positions if sp.shares > 0)) < 1e-8, \
            f"Cash committed mismatch: {self.cash_committed:.2f} != {sum(sp.cash_committed for sp in self.sub_positions if sp.shares > 0):.2f}"
        assert self.shares == sum(sp.shares for sp in self.sub_positions if sp.shares > 0), \
            f"Shares mismatch: {self.shares} != {sum(sp.shares for sp in self.sub_positions if sp.shares > 0)}"

        print(f"DEBUG: Partial exit complete - New shares: {self.shares}, Cash released: {cash_released:.2f}, Realized PnL: {realized_pnl:.2f}")
        print(f"DEBUG: Remaining sub-positions:")
        for i, sp in enumerate(self.sub_positions):
            if sp.shares > 0:
                print(f"  Sub-position {i}: Shares: {sp.shares}, Entry price: {sp.entry_price:.4f}")

        return realized_pnl, realized_pnl / self.times_buying_power, cash_released
                        
    # def calculate_num_sub_positions(self, total_shares: int) -> int:
    #     if self.times_buying_power <= 2:
    #         return 1
    #     elif self.initial_shares % 2 == 0:
    #         return 2
    #     else:
    #         return 3
        
    def calculate_num_sub_positions(self, total_shares: int) -> int:
        if self.initial_shares % 2 == 0:
            return 2
        else:
            return 3


    def calculate_shares_per_sub(self, total_shares: int, num_subs: int, current_sub_shares: List[int]) -> List[int]:
        target_shares = current_sub_shares.copy()
        while len(target_shares) < num_subs:
            target_shares.append(0)
        
        shares_to_add = total_shares - sum(current_sub_shares)
        max_shares_per_sub = self.initial_shares // num_subs

        for i in range(num_subs):
            if shares_to_add > 0:
                space_available = max_shares_per_sub - target_shares[i]
                shares_added = min(space_available, shares_to_add)
                target_shares[i] += shares_added
                shares_to_add -= shares_added

        # If there are still shares to add, distribute them to the last sub-position
        if shares_to_add > 0:
            target_shares[-1] += shares_to_add

        return target_shares

    def partial_entry(self, entry_time: datetime, entry_price: float, shares_to_buy: int):
        initial_shares = self.shares
        initial_cash_committed = self.cash_committed
        initial_market_value = self.market_value

        new_total_shares = self.shares + shares_to_buy
        new_num_subs = self.calculate_num_sub_positions(new_total_shares)

        additional_cash_committed = (shares_to_buy * entry_price) / self.times_buying_power
        self.cash_committed += additional_cash_committed

        current_sub_shares = [sp.shares for sp in self.sub_positions if sp.exit_time is None]
        target_shares = self.calculate_shares_per_sub(new_total_shares, new_num_subs, current_sub_shares)

        print(f"Debug - partial_entry:")
        print(f"  New total shares: {new_total_shares}, New num subs: {new_num_subs}")
        print(f"  All sub-positions: {[sp.shares for sp in self.sub_positions]}")
        print(f"  Current active sub-positions: {current_sub_shares}")
        print(f"  Target shares: {target_shares}")

        shares_added = 0
        remaining_shares = shares_to_buy
        
        print(f'BEFORE ENTRY cash_committed: ',[f'{sp.cash_committed:.2f} ({sp.shares})' for sp in self.sub_positions if sp.exit_time is None],f' = {sum([sp.cash_committed for sp in self.sub_positions if sp.exit_time is None]):.2f}')
        print(f'BEFORE ENTRY market_value:   ',[f'{sp.market_value:.2f} ({sp.shares})' for sp in self.sub_positions if sp.exit_time is None],f' = {sum([sp.market_value for sp in self.sub_positions if sp.exit_time is None]):.2f}')
        # assert abs(self.cash_committed - sum([sp.cash_committed for sp in self.sub_positions if sp.exit_time is None])) < 1e-8, (self.cash_committed, sum([sp.cash_committed for sp in self.sub_positions if sp.exit_time is None]))
        
        # Handle both initial entry and partial entries
        target_index = 0
        for sp in self.sub_positions:
            if sp.shares > 0 and remaining_shares > 0:
                shares_to_add = min(target_shares[target_index] - sp.shares, remaining_shares)
                if shares_to_add > 0:
                    sub_cash_committed = (shares_to_add * entry_price) / self.times_buying_power
                    old_shares = sp.shares
                    old_cash_committed = sp.cash_committed
                    sp.shares += shares_to_add
                    sp.cash_committed += sub_cash_committed
                    sp.update_market_value(entry_price)
                    self.add_transaction(entry_time, shares_to_add, entry_price, is_entry=True, sub_position=sp)
                    shares_added += shares_to_add
                    remaining_shares -= shares_to_add
                    print(f"DEBUG: Adding to sub-position - Entry price: {sp.entry_price:.4f}, Shares added: {shares_to_add}, "
                        f"Cash committed: {sub_cash_committed:.2f}, Old shares: {old_shares}, New shares: {sp.shares}, "
                        f"Old cash committed: {old_cash_committed:.2f}, New cash committed: {sp.cash_committed:.2f}")
                target_index += 1

        # Create new sub-positions if necessary
        while remaining_shares > 0 and target_index < new_num_subs:
            new_shares = min(target_shares[target_index], remaining_shares)
            if new_shares > 0:
                sub_cash_committed = (new_shares * entry_price) / self.times_buying_power
                new_sub = SubPosition(entry_time, entry_price, new_shares, sub_cash_committed)
                self.sub_positions.append(new_sub)
                self.add_transaction(entry_time, new_shares, entry_price, is_entry=True, sub_position=new_sub)
                shares_added += new_shares
                remaining_shares -= new_shares
                print(f"DEBUG: Created new sub-position - Entry price: {entry_price:.4f}, Shares: {new_shares}, "
                    f"Cash committed: {sub_cash_committed:.2f}")
            target_index += 1

        self.shares += shares_added
        self.update_market_value(entry_price)
        self.partial_entry_count += 1
        
        print(f'AFTER ENTRY cash_committed:  ',[f'{sp.cash_committed:.2f} ({sp.shares})' for sp in self.sub_positions if sp.exit_time is None],f' = {sum([sp.cash_committed for sp in self.sub_positions if sp.exit_time is None]):.2f}')
        print(f'AFTER ENTRY market_value:    ',[f'{sp.market_value:.2f} ({sp.shares})' for sp in self.sub_positions if sp.exit_time is None],f' = {sum([sp.market_value for sp in self.sub_positions if sp.exit_time is None]):.2f}')
        assert abs(self.cash_committed - sum([sp.cash_committed for sp in self.sub_positions if sp.exit_time is None])) < 1e-8, (self.cash_committed, sum([sp.cash_committed for sp in self.sub_positions if sp.exit_time is None]))
        
        # Assertions
        assert shares_added == shares_to_buy, f"Incorrect number of shares added. Expected: {shares_to_buy}, Added: {shares_added}"
        assert self.shares == initial_shares + shares_to_buy, f"Total shares mismatch. Expected: {initial_shares + shares_to_buy}, Actual: {self.shares}"
        assert abs(self.cash_committed - (initial_cash_committed + additional_cash_committed)) < 1e-8, f"Cash committed mismatch. Expected: {initial_cash_committed + additional_cash_committed}, Actual: {self.cash_committed}"
        assert abs(self.market_value - (initial_market_value + shares_to_buy * entry_price)) < 1e-8, f"Market value mismatch. Expected: {initial_market_value + shares_to_buy * entry_price}, Actual: {self.market_value}"
        assert self.shares == sum(sp.shares for sp in self.sub_positions if sp.exit_time is None), f"Sub-position shares mismatch. Total: {self.shares}, Sum of sub-positions: {sum(sp.shares for sp in self.sub_positions if sp.exit_time is None)}"
        assert abs(self.cash_committed - sum(sp.cash_committed for sp in self.sub_positions if sp.exit_time is None)) < 1e-8, f"Sub-position cash committed mismatch. Total: {self.cash_committed}, Sum of sub-positions: {sum(sp.cash_committed for sp in self.sub_positions if sp.exit_time is None)}"

        assert sum(sp.shares for sp in self.sub_positions) == self.shares, "Sub-position shares don't match total shares"
        assert abs(sum(sp.cash_committed for sp in self.sub_positions) - self.cash_committed) < 1e-8, "Sub-position cash doesn't match total cash committed"
        
        assert abs(self.cash_committed - sum(sp.cash_committed for sp in self.sub_positions if sp.shares > 0)) < 1e-8, \
            f"Cash committed mismatch: {self.cash_committed:.2f} != {sum(sp.cash_committed for sp in self.sub_positions if sp.shares > 0):.2f}"
        assert self.shares == sum(sp.shares for sp in self.sub_positions if sp.shares > 0), \
            f"Shares mismatch: {self.shares} != {sum(sp.shares for sp in self.sub_positions if sp.shares > 0)}"
            
        print(f"Partial entry: Position {self.id} {entry_time.time()} - {'Long' if self.is_long else 'Short'} "
            f"Bought {shares_to_buy} @ {entry_price:.4f} "
            f"(From {new_total_shares - shares_to_buy} to {new_total_shares}). "
            f"Sub-pos: {len([sp for sp in self.sub_positions if sp.exit_time is None])}, "
            f"Cash committed: {additional_cash_committed:.2f}")

        return additional_cash_committed


    def calculate_transaction_cost(self, shares: int, price: float, is_entry: bool, timestamp: datetime, sub_position: SubPosition) -> float:
        finra_taf = max(0.01, self.FINRA_TAF_RATE * shares)
        sec_fee = 0
        if not is_entry:  # SEC fee only applies to exits
            trade_value = price * shares
            sec_fee = self.SEC_FEE_RATE * trade_value
        
        stock_borrow_cost = 0
        if not self.is_long and not is_entry:  # Stock borrow cost applies only to short position exits
            daily_borrow_rate = self.stock_borrow_rate / 360
            # Calculate holding time from transactions
            total_days_held = 0
            
            # Walk backwards to find the earliest share included in current sub_position.shares
            cumulative_shares = 0
            for transaction in reversed(sub_position.transactions):
                if transaction.is_entry:
                    cumulative_shares += transaction.shares
                if cumulative_shares >= sub_position.shares:
                    earliest_relevant_timestamp = transaction.timestamp
                    break
            
            # Walk forwards to calculate the cost for the shares being removed
            shares_to_remove = shares
            for transaction in sub_position.transactions:
                if transaction.is_entry and transaction.timestamp >= earliest_relevant_timestamp:
                    holding_time = timestamp - transaction.timestamp
                    days_held = holding_time.total_seconds() / (24 * 60 * 60)
                    shares_to_calculate = min(shares_to_remove, transaction.shares)
                    total_days_held += shares_to_calculate * days_held
                    shares_to_remove -= shares_to_calculate
                    
                    if shares_to_remove <= 0:
                        break
            
            stock_borrow_cost = shares * price * daily_borrow_rate * (total_days_held / shares)
            
            # print(f'stock_borrow_cost: {stock_borrow_cost:.4f}')
            # print(f'{finra_taf + sec_fee:.4f} + {stock_borrow_cost:.4f} = {finra_taf + sec_fee + stock_borrow_cost:.4f}')
            
        return finra_taf + sec_fee + stock_borrow_cost



    def add_transaction(self, timestamp: datetime, shares: int, price: float, is_entry: bool, sub_position: SubPosition, sp_realized_pnl: Optional[float] = None):
        transaction_cost = self.calculate_transaction_cost(shares, price, is_entry, timestamp, sub_position)
        value = -shares * price if is_entry else shares * price
        
        transaction = Transaction(timestamp, shares, price, is_entry, transaction_cost, value, sp_realized_pnl)
        self.transactions.append(transaction)
        sub_position.add_transaction(transaction)
        
        if not is_entry:
            if sp_realized_pnl is None:
                raise ValueError("sp_realized_pnl must be provided for exit transactions")
            self.realized_pnl += sp_realized_pnl
            
        print(f"DEBUG: Transaction added - {'Entry' if is_entry else 'Exit'}, Shares: {shares}, Price: {price:.4f}, "
            f"Value: {value:.2f}, Cost: {transaction_cost:.4f}, Realized PnL: {sp_realized_pnl if sp_realized_pnl is not None else 'N/A'}")
        print(f"DEBUG: Current Realized PnL: {self.realized_pnl:.2f}, "
            f"Total Transaction Costs: {sum(t.transaction_cost for t in self.transactions):.4f}")
        
        # print(f"    {'Entry' if is_entry else 'Exit'} transaction: {shares} shares at {price:.4f} = {value:.4f}. Fees = {transaction_cost:.4f}")
        # if not is_entry:
        #     print(f"      Realized PnL = {sp_realized_pnl:.4f}")

  
    def update_stop_price(self, current_price: float):
        if self.is_long:
            self.max_price = max(self.max_price or self.entry_price, current_price)
            self.current_stop_price = self.max_price - self.area.get_range
        else:
            self.min_price = min(self.min_price or self.entry_price, current_price)
            self.current_stop_price = self.min_price + self.area.get_range
        
        self.update_market_value(current_price)
        
        return self.should_exit(current_price)

    def should_exit(self, current_price: float) -> bool:
        return (self.is_long and current_price <= self.current_stop_price) or \
               (not self.is_long and current_price >= self.current_stop_price)

    def close(self, exit_time: datetime, exit_price: float):
        self.exit_time = exit_time
        self.exit_price = exit_price
        for sp in self.sub_positions:
            if sp.exit_time is None:
                sp.exit_time = exit_time
                sp.exit_price = exit_price

    @property
    def entry_transaction_costs(self) -> float:
        return sum(t.transaction_cost for t in self.transactions if t.is_entry) / self.times_buying_power

    @property
    def exit_transaction_costs(self) -> float:
        return sum(t.transaction_cost for t in self.transactions if not t.is_entry) / self.times_buying_power

    @property
    def total_transaction_costs(self) -> float:
        return sum(t.transaction_cost for t in self.transactions) / self.times_buying_power

    @property
    def total_stock_borrow_cost(self) -> float:
        daily_borrow_rate = self.stock_borrow_rate / 360
        total_borrow_cost = 0.0

        # Temporary list to keep track of remaining shares for each entry transaction
        remaining_shares_list = []

        # Step 1: Iterate through all transactions and populate the remaining shares list for entry transactions
        for transaction in self.transactions:
            if transaction.is_entry:
                remaining_shares_list.append({
                    'timestamp': transaction.timestamp,
                    'shares': transaction.shares,
                    'price': transaction.price,
                })
            else:
                # Process exit transactions and calculate borrow cost
                shares_to_remove = transaction.shares
                exit_time = transaction.timestamp
                while shares_to_remove > 0 and remaining_shares_list:
                    entry = remaining_shares_list[0]
                    if entry['shares'] <= shares_to_remove:
                        # Entire entry transaction shares are being sold
                        holding_time = exit_time - entry['timestamp']
                        days_held = holding_time.total_seconds() / (24 * 60 * 60)
                        borrow_cost = entry['shares'] * entry['price'] * daily_borrow_rate * days_held
                        total_borrow_cost += borrow_cost

                        shares_to_remove -= entry['shares']
                        remaining_shares_list.pop(0)
                    else:
                        # Partial entry transaction shares are being sold
                        holding_time = exit_time - entry['timestamp']
                        days_held = holding_time.total_seconds() / (24 * 60 * 60)
                        borrow_cost = shares_to_remove * entry['price'] * daily_borrow_rate * days_held
                        total_borrow_cost += borrow_cost

                        entry['shares'] -= shares_to_remove
                        shares_to_remove = 0

        return total_borrow_cost



    @property
    def holding_time(self) -> timedelta:
        if not self.sub_positions:
            return timedelta(0)
        start_time = min(sp.entry_time for sp in self.sub_positions)
        end_time = max(sp.exit_time or datetime.now() for sp in self.sub_positions)
        return end_time - start_time
                    
    @property
    def get_unrealized_pnl(self) -> float:
        return self.unrealized_pnl / self.times_buying_power
                    
    @property
    def get_realized_pnl(self) -> float:
        return self.realized_pnl / self.times_buying_power
            
    @property
    def profit_loss(self) -> float:
        return self.get_unrealized_pnl + self.get_realized_pnl - self.total_transaction_costs
        
    

    @property
    def profit_loss_percentage(self) -> float:
        return (self.profit_loss / self.initial_balance) * 100

    @property
    def return_on_equity(self) -> float:
        equity_used = self.initial_balance # / self.times_buying_power
        return (self.profit_loss / equity_used) * 100

    @property
    def price_diff(self) -> float:
        if not self.sub_positions or any(sp.exit_time is None for sp in self.sub_positions):
            return 0
        avg_entry_price = sum(sp.entry_price * sp.shares for sp in self.sub_positions) / sum(sp.shares for sp in self.sub_positions)
        avg_exit_price = sum(sp.exit_price * sp.shares for sp in self.sub_positions) / sum(sp.shares for sp in self.sub_positions)
        diff = avg_exit_price - avg_entry_price
        return diff if self.is_long else -diff

    # def current_value(self, current_price: float) -> float:
    #     # market_value = self.shares * current_price
    #     if self.is_long:
    #         profit_loss = (current_price - self.entry_price) * self.shares
    #     else:
    #         profit_loss = (self.entry_price - current_price) * self.shares
    #     return self.initial_cash_used + (profit_loss / self.actual_margin_multiplier)
                
    @property
    def total_investment(self) -> float:
        return sum(sp.entry_price * sp.shares for sp in self.sub_positions)

    @property
    def margin_used(self) -> float:
        return self.total_investment - self.initial_balance
            
            



import csv
from datetime import datetime

import pandas as pd

def export_trades_to_csv(trades: List[TradePosition], filename: str):
    """
    Export the trades data to a CSV file using pandas.
    
    Args:
    trades (list): List of TradePosition objects
    filename (str): Name of the CSV file to be created
    """
    data = []
    for trade in trades:
        row = {
            'ID': trade.id,
            'Type': 'Long' if trade.is_long else 'Short',
            'Entry Time': trade.entry_time.strftime('%Y-%m-%d %H:%M:%S'),
            'Exit Time': trade.exit_time.strftime('%Y-%m-%d %H:%M:%S') if trade.exit_time else '',
            'Holding Time': str(trade.holding_time),
            'Entry Price': f"{trade.entry_price:.4f}",
            'Exit Price': f"{trade.exit_price:.4f}" if trade.exit_price else '',
            'Initial Shares': f"{trade.initial_shares:.4f}",
            'Realized P/L': f"{trade.realized_pnl:.4f}",
            'Unrealized P/L': f"{trade.get_unrealized_pnl:.4f}",
            'Total P/L': f"{trade.profit_loss:.4f}",
            'P/L %': f"{trade.profit_loss_percentage:.2f}",
            'ROE %': f"{trade.return_on_equity:.2f}",
            'Margin Multiplier': f"{trade.actual_margin_multiplier:.2f}",
            'Transaction Costs': f"{trade.total_transaction_costs:.4f}"
        }
        data.append(row)
        # print(row)  # Print each row for verification
        # print(trade.transactions)

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Trade summary has been exported to {filename}")

# # In your backtest_strategy function, replace or add after the print statements:
# export_trades_to_csv(trades)