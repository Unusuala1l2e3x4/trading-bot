from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Tuple, Optional
from TouchArea import TouchArea


@dataclass
class Transaction:
    timestamp: datetime
    shares: int
    price: float
    is_entry: bool
    transaction_cost: float
    
@dataclass
class SubPosition:
    entry_time: datetime
    entry_price: float
    shares: int
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None

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
    initial_cash_used: float
    entry_price: float
    current_market_value: float = 0
    current_shares: int = 0
    sub_positions: List[SubPosition] = field(default_factory=list)
    transactions: List[Transaction] = field(default_factory=list)
    current_stop_price: Optional[float] = None
    max_price: Optional[float] = None
    min_price: Optional[float] = None
    last_price: float = field(default=0.0)

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
        self.current_shares = 0  # Ensure it starts at 0
        
    def add_shares(self, shares: int):
        self.current_shares += shares
        
    @property
    def is_open(self) -> bool:
        return any(sp.exit_time is None for sp in self.sub_positions)

    @property
    def total_shares(self) -> int:
        return sum(sp.shares for sp in self.sub_positions if sp.exit_time is None)

    # @property
    # def entry_price(self) -> float:
    #     total_cost = sum(sp.entry_price * sp.shares for sp in self.sub_positions)
    #     return total_cost / self.total_shares if self.total_shares > 0 else 0
    
    def update_last_price(self, price: float):
        self.last_price = price
        
        
    def update_market_value(self, current_price: float):
        self.current_market_value = self.current_shares * current_price

    def cash_value(self) -> float:
        return self.initial_cash_used

    def equity_value(self) -> float:
        return (self.current_market_value - self.initial_cash_used * self.actual_margin_multiplier) / self.actual_margin_multiplier
    
    @property
    def margin_multiplier(self) -> float:
        return self.actual_margin_multiplier

    @property
    def num_sub_positions(self) -> int:
        if self.times_buying_power <= 2:
            return 1
        elif self.initial_shares % 2 == 0:
            return 2
        else:
            return 3
        
    @property
    def shares_per_sub(self) -> List[int]:
        num_subs = self.num_sub_positions
        if num_subs == 1:
            return [self.initial_shares]
        elif num_subs == 2:
            return [self.initial_shares // 2] * 2
        else:  # 3 sub-positions
            base_shares = self.initial_shares // 3
            extra_shares = self.initial_shares % 3
            return [base_shares + (1 if i < extra_shares else 0) for i in range(3)]
        
        
        
    def add_transaction(self, timestamp: datetime, shares: int, price: float, is_entry: bool):
        transaction_cost = self.calculate_transaction_cost(shares, price, is_entry)
        transaction = Transaction(timestamp, shares, price, is_entry, transaction_cost)
        self.transactions.append(transaction)
        
        print(f'add_transaction {timestamp}, {shares}, {price:.4f}, {is_entry}, {transaction_cost:.4f}')
        
    def add_sub_position(self, entry_time: datetime, entry_price: float, shares: int):
        self.sub_positions.append(SubPosition(entry_time, entry_price, shares))
        self.add_shares(shares)
        print(f'buying {shares} shares at {entry_time}, {entry_price:.4f} ({shares*entry_price:.4f})')
        self.add_transaction(entry_time, shares, entry_price, is_entry=True)

    def partial_exit(self, exit_time: datetime, exit_price: float, shares_to_sell: int) -> Tuple[float, float]:
        remaining_shares = shares_to_sell
        realized_pnl = 0
        for sp in self.sub_positions:
            if sp.exit_time is None and remaining_shares > 0:
                shares_sold = min(sp.shares, remaining_shares)
                sp.shares -= shares_sold
                remaining_shares -= shares_sold
                if sp.shares == 0:
                    sp.exit_time = exit_time
                    sp.exit_price = exit_price
                pnl = (exit_price - sp.entry_price) * shares_sold if self.is_long else \
                    (sp.entry_price - exit_price) * shares_sold
                realized_pnl += pnl
        self.current_shares -= shares_to_sell
        self.add_transaction(exit_time, shares_to_sell, exit_price, is_entry=False)
        return realized_pnl / self.actual_margin_multiplier, shares_to_sell * exit_price / self.actual_margin_multiplier
                        
    @staticmethod
    def calculate_shares_per_sub(total_shares: int, num_subs: int) -> List[int]:
        if num_subs == 1:
            return [total_shares]
        elif num_subs == 2:
            return [total_shares // 2] * 2
        else:  # 3 sub-positions
            base_shares = total_shares // 3
            extra_shares = total_shares % 3
            return [base_shares + (1 if i < extra_shares else 0) for i in range(3)]
      
      
    def partial_entry(self, entry_time: datetime, entry_price: float, shares_to_buy: int):
        # Step 1: Determine the current and new number of sub-positions. 
        # Based on the new total shares and margin requirements, we determine how many sub-positions we should have after the partial entry.
        current_num_subs = len([sp for sp in self.sub_positions if sp.shares > 0])
        new_total_shares = self.current_shares + shares_to_buy
        new_num_subs = 3 if new_total_shares % 2 != 0 and self.times_buying_power > 2 else \
                       2 if self.times_buying_power > 2 else 1

        # Step 2: Create new sub-positions if necessary. 
        # If we need more sub-positions than we currently have, we create new ones with zero shares initially.
        if new_num_subs > current_num_subs:
            # Create new sub-position(s)
            for _ in range(new_num_subs - current_num_subs):
                self.add_sub_position(entry_time, entry_price, 0)

        # Step 3: Calculate the target distribution of shares
        # How the shares should be distributed among the sub-positions after the partial entry.
        shares_per_sub = self.calculate_shares_per_sub(new_total_shares, new_num_subs)
        
        # Step 4: Distribute new shares among sub-positions
        remaining_shares = shares_to_buy
        active_sub_positions = [sp for sp in self.sub_positions if sp.shares > 0]
        
        # First, distribute shares to existing active sub-positions
        for sp in active_sub_positions:
            target_shares = shares_per_sub[len(active_sub_positions) - 1]
            shares_to_add = target_shares - sp.shares
            if shares_to_add > 0:
                actual_shares_to_add = min(shares_to_add, remaining_shares)
                sp.shares += actual_shares_to_add
                self.current_shares += actual_shares_to_add
                remaining_shares -= actual_shares_to_add
                if remaining_shares == 0:
                    break
        
        # If there are still shares to distribute
        while remaining_shares > 0 and len(active_sub_positions) < new_num_subs:
            sp = SubPosition(entry_time, entry_price, 0)
            
            target_shares = shares_per_sub[len(active_sub_positions)]
            actual_shares_to_add = min(target_shares, remaining_shares)
            sp.shares = actual_shares_to_add
            self.current_shares += actual_shares_to_add
            remaining_shares -= actual_shares_to_add
            
            self.sub_positions.append(sp)
            active_sub_positions.append(sp)
        
        self.add_transaction(entry_time, shares_to_buy, entry_price, is_entry=True)
            
            
    def update_stop_price(self, current_price: float):
        if self.is_long:
            self.max_price = max(self.max_price or self.entry_price, current_price)
            self.current_stop_price = self.max_price - self.area.get_range
        else:
            self.min_price = min(self.min_price or self.entry_price, current_price)
            self.current_stop_price = self.min_price + self.area.get_range

    def should_exit(self, current_price: float) -> bool:
        return (self.is_long and current_price <= self.current_stop_price) or \
               (not self.is_long and current_price >= self.current_stop_price)

    def close(self, exit_time: datetime, exit_price: float):
        for sp in self.sub_positions:
            if sp.exit_time is None:
                sp.exit_time = exit_time
                sp.exit_price = exit_price




    def calculate_transaction_cost(self, shares: int, price: float, is_entry: bool) -> float:
        finra_taf = max(0.01, self.FINRA_TAF_RATE * shares)
        sec_fee = 0
        if not is_entry or (not self.is_long and is_entry):
            trade_value = price * shares
            sec_fee = self.SEC_FEE_RATE * trade_value
        return finra_taf + sec_fee

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
    def stock_borrow_cost(self) -> float:
        if self.is_long:
            return 0
        total_cost = 0
        for i in range(len(self.transactions) - 1):
            current_transaction = self.transactions[i]
            next_transaction = self.transactions[i + 1]
            holding_days = (next_transaction.timestamp - current_transaction.timestamp).total_seconds() / (24 * 60 * 60)
            daily_borrow_rate = self.stock_borrow_rate / 360
            shares = sum(sp.shares for sp in self.sub_positions if sp.entry_time <= current_transaction.timestamp and (sp.exit_time is None or sp.exit_time > next_transaction.timestamp))
            total_cost += shares * current_transaction.price * daily_borrow_rate * holding_days
        return total_cost


    @property
    def holding_time(self) -> timedelta:
        if not self.sub_positions:
            return timedelta(0)
        start_time = min(sp.entry_time for sp in self.sub_positions)
        end_time = max(sp.exit_time or datetime.now() for sp in self.sub_positions)
        return end_time - start_time

    @property
    def profit_loss(self) -> float:
        realized_pl = sum((sp.exit_price - sp.entry_price) * sp.shares if self.is_long else
                        (sp.entry_price - sp.exit_price) * sp.shares
                        for sp in self.sub_positions if sp.exit_time is not None)
        unrealized_pl = sum((self.last_price - sp.entry_price) * sp.shares if self.is_long else
                            (sp.entry_price - self.last_price) * sp.shares
                            for sp in self.sub_positions if sp.exit_time is None)
        
        # Adjust profit/loss for margin
        total_pl = (realized_pl + unrealized_pl) / self.actual_margin_multiplier
        
        return total_pl - self.total_transaction_costs

    @property
    def profit_loss_percentage(self) -> float:
        return (self.profit_loss / self.initial_balance) * 100

    @property
    def return_on_equity(self) -> float:
        equity_used = self.initial_balance / self.margin_multiplier
        return (self.profit_loss / equity_used) * 100

    @property
    def price_diff(self) -> float:
        if not self.sub_positions or any(sp.exit_time is None for sp in self.sub_positions):
            return 0
        avg_entry_price = sum(sp.entry_price * sp.shares for sp in self.sub_positions) / sum(sp.shares for sp in self.sub_positions)
        avg_exit_price = sum(sp.exit_price * sp.shares for sp in self.sub_positions) / sum(sp.shares for sp in self.sub_positions)
        diff = avg_exit_price - avg_entry_price
        return diff if self.is_long else -diff

    def current_value(self, current_price: float) -> float:
        market_value = self.current_shares * current_price
        if self.is_long:
            profit_loss = (current_price - self.entry_price) * self.current_shares
        else:
            profit_loss = (self.entry_price - current_price) * self.current_shares
        return self.initial_cash_used + (profit_loss / self.actual_margin_multiplier)
                
    @property
    def total_investment(self) -> float:
        return sum(sp.entry_price * sp.shares for sp in self.sub_positions)

    @property
    def margin_used(self) -> float:
        return self.total_investment - self.initial_balance
            
            



import csv
from datetime import datetime

def export_trades_to_csv(trades:List[TradePosition], filename:str):
    """
    Export the trades data to a CSV file.
    
    Args:
    trades (list): List of TradePosition objects
    filename (str): Name of the CSV file to be created
    """
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['ID', 'Type', 'Entry Time', 'Exit Time', 'Holding Time', 'Entry Price', 'Exit Price', 'Shares', 
                      'P/L', 'P/L %', 'ROE %', 'Margin Multiplier', 'Transaction Costs']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for trade in trades:
            writer.writerow({
                'ID': trade.id,
                'Type': 'Long' if trade.is_long else 'Short',
                'Entry Time': trade.entry_time.strftime('%Y-%m-%d %H:%M:%S'),
                'Exit Time': trade.exit_time.strftime('%Y-%m-%d %H:%M:%S'),
                'Holding Time': str(trade.holding_time),
                'Entry Price': f"{trade.entry_price:.4f}",
                'Exit Price': f"{trade.exit_price:.4f}",
                'Shares': f"{trade.shares:.4f}",
                'P/L': f"{trade.profit_loss:.4f}",
                'P/L %': f"{trade.profit_loss_percentage:.2f}",
                'ROE %': f"{trade.return_on_equity:.2f}",
                'Margin Multiplier': f"{trade.margin_multiplier:.2f}",
                'Transaction Costs': f"{trade.total_transaction_costs:.4f}"
            })

    print(f"Trade summary has been exported to {filename}")

# # In your backtest_strategy function, replace or add after the print statements:
# export_trades_to_csv(trades)