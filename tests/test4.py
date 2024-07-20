    # You're right to point that out, and I apologize for the oversight. We should indeed keep using the `self.realized_pnl` attribute as it provides a running total of realized profits and losses, which can be useful for various calculations and tracking purposes. Let's modify our approach to incorporate this:

    # 1. Update the `add_transaction` method:

    # ```python
    def add_transaction(self, timestamp: datetime, shares: int, price: float, is_entry: bool):
        transaction_cost = self.calculate_transaction_cost(shares, price, is_entry, timestamp)
        value = -shares * price if is_entry else shares * price  # Negative for buys, positive for sells
        transaction = Transaction(timestamp, shares, price, is_entry, transaction_cost, value)
        self.transactions.append(transaction)
        
        if not is_entry:
            # Update realized_pnl only on exits
            self.realized_pnl += value - transaction_cost
        
        print(f'    add_transaction {timestamp}, {shares}, {price:.4f}, {is_entry}, {transaction_cost:.4f}, {value:.4f}')
    # ```

    # 2. Modify the `profit_loss` property:

    # ```python
    @property
    def profit_loss(self) -> float:
        unrealized_pnl = sum((self.last_price - sp.entry_price) * sp.shares if self.is_long else
                            (sp.entry_price - self.last_price) * sp.shares
                            for sp in self.sub_positions if sp.shares > 0)
        
        total_pnl = self.realized_pnl + unrealized_pnl
        adjusted_pnl = total_pnl / self.times_buying_power
        
        return adjusted_pnl
    # ```

    # 3. Keep the `partial_exit` function as it is in my previous response, but remove the `realized_pnl` calculation since it's now handled in `add_transaction`:

    # ```python
    def partial_exit(self, exit_time: datetime, exit_price: float, shares_to_sell: int):
        cash_released = 0
        remaining_shares_to_sell = shares_to_sell

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

                if sp.shares == 0:
                    sp.exit_time = exit_time
                    sp.exit_price = exit_price

        self.cash_committed -= cash_released
        self.market_value = self.shares * exit_price
        self.partial_exit_count += 1

        print(f"Partial exit: {self.id} {exit_time.time()} - {'Long' if self.is_long else 'Short'} "
            f"Sold {shares_to_sell} @ {exit_price:.4f} "
            f"(From {self.shares + shares_to_sell} to {self.shares}). "
            f"Cash released: {cash_released:.2f}, "
            f"Realized PnL: {self.realized_pnl:.2f}")

        return self.realized_pnl, self.realized_pnl / self.times_buying_power, cash_released
    # ```

    # These changes ensure that:

    # 1. We're using the `self.realized_pnl` attribute to track realized profits and losses.
    # 2. The `add_transaction` method updates `self.realized_pnl` for exit transactions.
    # 3. The `profit_loss` property calculates total P&L using both realized and unrealized P&L, adjusting for leverage.
    # 4. We're not double-counting realized P&L in the `partial_exit` function.

    # This approach should provide more accurate P&L calculations while maintaining the use of the `self.realized_pnl` attribute. The `profit_loss` property now gives a complete picture of both realized and unrealized P&L, adjusted for leverage.