from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, List
from TouchArea import TouchArea

@dataclass
class TradePosition:
    id: int
    area: TouchArea
    is_long: bool
    entry_time: datetime
    entry_price: float
    initial_balance: float
    total_shares: int
    use_margin: bool
    is_marginable: bool
    times_buying_power: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    current_stop_price: Optional[float] = None
    max_price: Optional[float] = None
    min_price: Optional[float] = None
    actual_margin_multiplier: float = 1.0
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


    @property
    def is_open(self) -> bool:
        return self.exit_time is None

    def current_value(self, current_price: float) -> float:
        unrealized_pnl = (current_price - self.entry_price) * self.total_shares
        if not self.is_long:
            unrealized_pnl = -unrealized_pnl
        return self.initial_balance + unrealized_pnl
    
    @property
    def margin_multiplier(self) -> float:
        return self.actual_margin_multiplier
    
    @property
    def num_sub_positions(self) -> int:
        if self.times_buying_power <= 2:
            return 1
        elif self.total_shares % 2 == 0:
            return 2
        else:
            return 3
        
    @property
    def shares_per_sub(self) -> List[int]:
        if self.num_sub_positions == 1:
            return [self.total_shares]
        elif self.num_sub_positions == 2:
            return [self.total_shares // 2] * 2
        else:  # 3 sub-positions
            base_shares = self.total_shares // 3
            extra_shares = self.total_shares % 3
            return [base_shares + (1 if i < extra_shares else 0) for i in range(3)]
    
    # @property
    # def total_shares(self) -> float:
    #     return (self.initial_balance * self.margin_multiplier) / self.entry_price
    
    # @property
    # def total_investment(self) -> float:
    #     return self.investment_amount * self.num_sub_positions
    
    # @property
    # def investment_amount(self) -> float:
    #     return self.shares * self.entry_price
    
    @property
    def total_investment(self) -> float:
        return self.total_shares * self.entry_price

    @property
    def margin_used(self) -> float:
        return self.total_investment - self.initial_balance

                
    @property
    def entry_transaction_costs(self) -> float:
        return sum(max(0.01, self.FINRA_TAF_RATE * shares) for shares in self.shares_per_sub)

    @property
    def exit_transaction_costs(self) -> float:
        if not self.exit_price:
            return 0
        finra_taf = sum(max(0.01, self.FINRA_TAF_RATE * shares) for shares in self.shares_per_sub)
        sec_fee = 0
        if self.is_long or (not self.is_long and self.exit_price > self.entry_price):
            trade_value = self.exit_price * self.total_shares
            sec_fee = self.SEC_FEE_RATE * trade_value
        return finra_taf + sec_fee

    @property
    def stock_borrow_cost(self) -> float:
        if self.is_long or not self.exit_time:
            return 0
        holding_days = self.holding_time.total_seconds() / (24 * 60 * 60)
        daily_borrow_rate = self.stock_borrow_rate / 360
        return self.total_shares * self.entry_price * daily_borrow_rate * holding_days

    @property
    def total_transaction_costs(self) -> float:
        return self.entry_transaction_costs + self.exit_transaction_costs + self.stock_borrow_cost

    @property
    def holding_time(self) -> timedelta:
        if not self.exit_time:
            return timedelta(0)
        return self.exit_time - self.entry_time

    @property
    def profit_loss(self) -> float:
        if not self.exit_price:
            return 0
        price_difference = self.exit_price - self.entry_price if self.is_long else self.entry_price - self.exit_price
        gross_profit = price_difference * self.total_shares
        net_profit = gross_profit - self.total_transaction_costs
        # print(f"Debug - Profit/Loss Calculation:")
        # print(f"  Price Diff: {price_difference:.4f}")
        # print(f"  Gross Profit: {gross_profit:.4f}")
        # print(f"  Net Profit: {net_profit:.4f}")
        # print(f"  Total Shares: {self.total_shares:.4f}")
        # print(f"  Transaction Costs: {self.total_transaction_costs:.4f}")
        return net_profit

    @property
    def profit_loss_percentage(self) -> float:
        return (self.profit_loss / self.initial_balance) * 100

    @property
    def return_on_equity(self) -> float:
        equity_used = self.initial_balance / self.margin_multiplier
        return (self.profit_loss / equity_used) * 100

    @property
    def price_diff(self) -> float:
        if not self.exit_price:
            return 0
        diff = self.exit_price - self.entry_price
        if not self.is_long:
            diff = -diff
        return diff

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
        self.exit_time = exit_time
        self.exit_price = exit_price
        # print(f"Debug - Closing position: Entry: {self.entry_price:.4f}, Exit: {self.exit_price:.4f}, Diff: {self.price_diff:.4f}")
            
            



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