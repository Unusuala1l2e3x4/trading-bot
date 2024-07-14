from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class Trade:
    id: int
    is_long: bool
    entry_time: datetime
    entry_price: float
    exit_time: datetime
    exit_price: float
    initial_balance: float
    use_margin: bool
    is_marginable: bool
    times_buying_power: float
    stock_borrow_rate: float = 0.003    # Default to 30 bps (0.3%) annually
    # stock_borrow_rate: float = 0.03      # Default to 300 bps (3%) annually

    FINRA_TAF_RATE = 0.000119  # per share
    SEC_FEE_RATE = 22.90 / 1_000_000  # per dollar
    
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



    @property
    def margin_multiplier(self) -> float:
        if self.use_margin and self.is_marginable:
            return min(self.times_buying_power, 4)  # Cap at 4x as per regulations
        return 1

    @property
    def shares(self) -> float:
        return (self.initial_balance * self.margin_multiplier) / self.entry_price

    @property
    def entry_transaction_costs(self) -> float:
        # FINRA TAF
        finra_taf = max(0.01, self.FINRA_TAF_RATE * self.shares)
        return finra_taf

    @property
    def exit_transaction_costs(self) -> float:
        # FINRA TAF
        finra_taf = max(0.01, self.FINRA_TAF_RATE * self.shares)

        # SEC Fee (only for selling long positions or closing short positions)
        sec_fee = 0
        if self.is_long or (not self.is_long and self.exit_price > self.entry_price):
            trade_value = self.exit_price * self.shares
            sec_fee = self.SEC_FEE_RATE * trade_value

        return finra_taf + sec_fee

    @property
    def stock_borrow_cost(self) -> float:
        if self.is_long:
            return 0
        
        holding_days = self.holding_time.total_seconds() / (24 * 60 * 60)
        daily_borrow_rate = self.stock_borrow_rate / 360
        return self.initial_balance * self.margin_multiplier * daily_borrow_rate * holding_days

    @property
    def total_transaction_costs(self) -> float:
        return self.entry_transaction_costs + self.exit_transaction_costs + self.stock_borrow_cost

    @property
    def holding_time(self) -> timedelta:
        return self.exit_time - self.entry_time

    @property
    def price_diff(self) -> float:
        diff = self.exit_price - self.entry_price if self.is_long else self.entry_price - self.exit_price
        return diff - (self.total_transaction_costs/self.shares)
    
    @property
    def profit_loss(self) -> float:
        price_difference = self.exit_price - self.entry_price if self.is_long else self.entry_price - self.exit_price
        gross_profit = price_difference * self.shares
        return gross_profit - self.total_transaction_costs

    @property
    def profit_loss_percentage(self) -> float:
        return (self.profit_loss / self.initial_balance) * 100

    @property
    def return_on_equity(self) -> float:
        """Calculate return on equity, considering margin if used"""
        equity_used = self.initial_balance / self.margin_multiplier
        return (self.profit_loss / equity_used) * 100