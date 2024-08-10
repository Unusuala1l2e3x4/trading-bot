# You're absolutely right about the slippage calculation and its implementation. Let's modify the code to incorporate your suggestions. We'll adjust the `calculate_slippage` function and modify the `partial_entry` and `partial_exit` methods to handle slippage correctly.

# First, let's update the `calculate_slippage` function:

# ```python
def calculate_slippage(is_long: bool, trade_size: int, volume: Decimal, slippage_factor: Decimal) -> Decimal:
    if is_long:    
        return -slippage_factor * (Decimal(trade_size) / volume)  # decreases price for longs
    else:
        return slippage_factor * (Decimal(trade_size) / volume)  # increases price for shorts
# ```

# Now, let's modify the `partial_entry` and `partial_exit` methods to incorporate slippage:

# ```python
class TradePosition:
    # ... (other methods remain the same)

    def partial_entry(self, entry_time: datetime, entry_price: Decimal, shares_to_buy: int, vwap: Decimal, volume: Decimal, slippage_factor: Decimal):
        if self.times_buying_power > 2 and (self.shares + shares_to_buy) % 2 != 0:
            shares_to_buy -= 1

        slippage = calculate_slippage(self.is_long, shares_to_buy, volume, slippage_factor)
        adjusted_price = entry_price * (1 + slippage)

        new_total_shares = self.shares + shares_to_buy
        new_num_subs = self.calculate_num_sub_positions()

        additional_cash_committed = (shares_to_buy * entry_price) / self.times_buying_power

        active_sub_positions = [sp for sp in self.sub_positions if sp.shares > 0]
        target_shares = self.calculate_shares_per_sub(new_total_shares, new_num_subs)

        fees = 0
        shares_added = 0

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
            else:
                # New sub-position
                sub_cash_committed = (target * entry_price) / self.times_buying_power
                new_sub = SubPosition(entry_time, adjusted_price, target, sub_cash_committed)
                self.sub_positions.append(new_sub)
                fees += self.add_transaction(entry_time, target, adjusted_price, is_entry=True, vwap=vwap, sub_position=new_sub)
                shares_added += target

        self.shares += shares_added
        self.cash_committed += additional_cash_committed
        self.update_market_value(adjusted_price)
        self.partial_entry_count += 1
        return additional_cash_committed, fees

    def partial_exit(self, exit_time: datetime, exit_price: Decimal, shares_to_sell: int, vwap: Decimal, volume: Decimal, slippage_factor: Decimal):
        if self.times_buying_power > 2 and (self.shares - shares_to_sell) % 2 != 0:
            shares_to_sell -= 1

        slippage = calculate_slippage(not self.is_long, shares_to_sell, volume, slippage_factor)
        adjusted_price = exit_price * (1 + slippage)

        cash_released = 0
        realized_pnl = 0
        fees = 0

        active_sub_positions = [sp for sp in self.sub_positions if sp.shares > 0]
        total_shares = sum(sp.shares for sp in active_sub_positions)

        for sp in active_sub_positions:
            assert sp.shares == int(Decimal(total_shares)/Decimal(self.calculate_num_sub_positions())), (sp.shares, total_shares, self.calculate_num_sub_positions())
            shares_sold = int(shares_to_sell * (Decimal(sp.shares) / Decimal(total_shares)))
            if shares_sold > 0:
                sub_cash_released = (Decimal(shares_sold) / Decimal(sp.shares)) * sp.cash_committed
                sp_realized_pnl = (adjusted_price - sp.entry_price) * shares_sold if self.is_long else (sp.entry_price - adjusted_price) * shares_sold
                old_shares = sp.shares
                sp.shares -= shares_sold
                sp.cash_committed -= sub_cash_released
                sp.realized_pnl += sp_realized_pnl
                cash_released += sub_cash_released
                realized_pnl += sp_realized_pnl
                fees += self.add_transaction(exit_time, shares_sold, adjusted_price, is_entry=False, vwap=vwap, sub_position=sp, sp_realized_pnl=sp_realized_pnl)

        self.shares -= shares_to_sell
        self.cash_committed -= cash_released
        self.update_market_value(adjusted_price)
        self.partial_exit_count += 1
        return realized_pnl, cash_released, fees

    def add_transaction(self, timestamp: datetime, shares: int, price: Decimal, is_entry: bool, vwap: float, sub_position: SubPosition, sp_realized_pnl: Optional[Decimal] = None):
        finra_taf, sec_fee, stock_borrow_cost = self.calculate_transaction_cost(shares, price, is_entry, timestamp, sub_position)
        transaction_cost = finra_taf + sec_fee + stock_borrow_cost
        value = -shares * price if is_entry else shares * price
        transaction = Transaction(timestamp, shares, price, is_entry, transaction_cost, finra_taf, sec_fee, stock_borrow_cost, value, vwap, sp_realized_pnl)
        self.transactions.append(transaction)
        sub_position.add_transaction(transaction)
        if not is_entry:
            if sp_realized_pnl is None:
                raise ValueError("sp_realized_pnl must be provided for exit transactions")
            self.realized_pnl += sp_realized_pnl
        return transaction_cost
# ```

# These changes implement slippage in the following way:

# 1. Slippage is calculated based on the trade size, volume, and whether it's a long or short position.
# 2. The adjusted price (including slippage) is used for creating new sub-positions, adding transactions, and updating market value.
# 3. The original price is still used for calculating cash committed, ensuring that the balance remains accurate.
# 4. Realized PnL is calculated using the adjusted price, reflecting the actual execution price after slippage.

# To use these modified methods, you'll need to pass the `volume` and `slippage_factor` parameters when calling `partial_entry` and `partial_exit`. For example:

# ```python
additional_cash_committed, fees = position.partial_entry(entry_time, entry_price, shares_to_buy, vwap, volume, slippage_factor)

realized_pnl, cash_released, fees = position.partial_exit(exit_time, exit_price, shares_to_sell, vwap, volume, slippage_factor)
# ```

# These changes should accurately model unfavorable slippage for both long and short positions while maintaining the correct balance and cash committed values.