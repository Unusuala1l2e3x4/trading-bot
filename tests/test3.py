# You're absolutely right, and I apologize for missing that earlier. Your analysis is spot on, and your proposed solution is excellent. Let's implement these changes:

# 1. First, let's modify the Transaction class:

# ```python
@dataclass
class Transaction:
    timestamp: datetime
    shares: int
    price: float
    is_entry: bool
    transaction_cost: float
    value: float  # This will be the cost (negative) or revenue (positive)
# ```

# 2. Now, let's modify the TradePosition class to keep track of realized PnL:

# ```python
@dataclass
class TradePosition:
    # ... (other attributes)
    realized_pnl: float = 0
    
    def add_transaction(self, timestamp: datetime, shares: int, price: float, is_entry: bool):
        transaction_cost = self.calculate_transaction_cost(shares, price, is_entry, timestamp)
        value = -shares * price if is_entry else shares * price  # Negative for buys, positive for sells
        transaction = Transaction(timestamp, shares, price, is_entry, transaction_cost, value)
        self.transactions.append(transaction)
        
        self.realized_pnl += value - transaction_cost
        
        print(f'    add_transaction {timestamp}, {shares}, {price:.4f}, {is_entry}, {transaction_cost:.4f}, {value:.4f}')
# ```

# 3. Update the partial_entry and partial_exit functions:

# ```python
    def partial_entry(self, entry_time: datetime, entry_price: float, shares_to_buy: int):
        # ... (existing code)
        
        for i in range(len(self.sub_positions)):
            if self.sub_positions[i].shares > 0:
                shares_to_add = target_shares[i] - self.sub_positions[i].shares
                if shares_to_add > 0:
                    sub_cash_committed = (shares_to_add * entry_price) / self.times_buying_power
                    self.sub_positions[i].shares += shares_to_add
                    self.shares += shares_to_add
                    self.sub_positions[i].cash_committed += sub_cash_committed
                    self.add_transaction(entry_time, shares_to_add, entry_price, is_entry=True)
                    remaining_shares -= shares_to_add

        # ... (rest of the function)

        print(f"Partial entry: {self.id} {entry_time.time()} - {'Long' if self.is_long else 'Short'} "
            f"Bought {shares_to_buy} @ {entry_price:.4f} "
            f"(From {self.shares - shares_to_buy} to {self.shares}). "
            f"Sub-pos: {len([sp for sp in self.sub_positions if sp.shares > 0])}, "
            f"Cash committed: {additional_cash_committed:.2f}, "
            f"Realized PnL: {self.realized_pnl:.2f}")

        return additional_cash_committed

    def partial_exit(self, exit_time: datetime, exit_price: float, shares_to_sell: int):
        # ... (existing code)

        for sp in self.sub_positions:
            if sp.shares > 0 and remaining_shares_to_sell > 0:
                shares_sold = min(sp.shares, remaining_shares_to_sell)
                
                sub_cash_released = (shares_sold / sp.shares) * sp.cash_committed
                sp.shares -= shares_sold
                self.shares -= shares_sold
                sp.cash_committed -= sub_cash_released
                cash_released += sub_cash_released
                self.add_transaction(exit_time, shares_sold, exit_price, is_entry=False)
                remaining_shares_to_sell -= shares_sold

        # ... (rest of the function)

        print(f"Partial exit: {self.id} {exit_time.time()} - {'Long' if self.is_long else 'Short'} "
            f"Sold {shares_to_sell} @ {exit_price:.4f} "
            f"(From {self.shares + shares_to_sell} to {self.shares}). "
            f"Cash released: {cash_released:.2f}, "
            f"Realized PnL: {self.realized_pnl:.2f}")

        return self.realized_pnl, cash_released
# ```

# 4. Update the profit_loss function:

# ```python
    @property
    def profit_loss(self) -> float:
        unrealized_pnl = sum((self.last_price - sp.entry_price) * sp.shares if self.is_long else
                            (sp.entry_price - self.last_price) * sp.shares
                            for sp in self.sub_positions if sp.shares > 0)
        
        total_pl = (self.realized_pnl + unrealized_pnl) / self.times_buying_power
        return total_pl
# ```

# 5. Update the backtest_strategy function:

# ```python
def backtest_strategy(...):
    # ... (existing code)

    winning_trades = sum(1 for trade in trades if trade.realized_pnl > 0)
    losing_trades = sum(1 for trade in trades if trade.realized_pnl < 0)

    # ... (rest of the function)
# ```

# With these changes:

# 1. We're now correctly tracking realized PnL for each transaction.
# 2. The profit_loss function now correctly calculates both realized and unrealized PnL.
# 3. We've removed the need for multiple calls to update_total_account_value, as the PnL is now being tracked accurately within each position.
# 4. The winrate calculation is now based on the realized PnL of closed trades, which should give you a more accurate representation of the strategy's performance.

# These modifications should resolve the issues you've identified and provide a more accurate and consistent way of tracking and calculating profits and losses across your backtesting system.