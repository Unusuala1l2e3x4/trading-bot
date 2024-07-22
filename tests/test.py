    # You're absolutely correct. We don't need to worry about already-exited sub-positions because, as you pointed out:

    # 1. We adjust all active sub-positions proportionally during partial exits.
    # 2. Sub-positions only go to zero on the final exit.

    # This approach simplifies our logic and ensures that we're always working with active sub-positions. Let's update our code to reflect this understanding:

    # 1. In the `partial_exit` method, we can simplify our loop:

    # ```python
    def partial_exit(self, exit_time: datetime, exit_price: float, shares_to_sell: int):
        print(f"DEBUG: Entering partial_exit - Time: {exit_time}, Price: {exit_price:.4f}, Shares to sell: {shares_to_sell}")
        print(f"DEBUG: Current position - Shares: {self.shares}, Cash committed: {self.cash_committed:.2f}")

        cash_released = 0
        realized_pnl = 0

        active_sub_positions = [sp for sp in self.sub_positions if sp.shares > 0]
        total_shares = sum(sp.shares for sp in active_sub_positions)

        for sp in active_sub_positions:
            shares_sold = int(shares_to_sell * (sp.shares / total_shares))
            if shares_sold > 0:
                sub_cash_released = (shares_sold / sp.shares) * sp.cash_committed
                sp_realized_pnl = (exit_price - sp.entry_price) * shares_sold if self.is_long else (sp.entry_price - exit_price) * shares_sold
                
                old_shares = sp.shares
                sp.shares -= shares_sold
                sp.cash_committed -= sub_cash_released
                sp.realized_pnl += sp_realized_pnl
                sp.update_market_value(exit_price)

                cash_released += sub_cash_released
                realized_pnl += sp_realized_pnl
                
                self.add_transaction(exit_time, shares_sold, exit_price, is_entry=False, sub_position=sp, sp_realized_pnl=sp_realized_pnl)

                print(f"DEBUG: Selling from sub-position - Entry price: {sp.entry_price:.4f}, Shares sold: {shares_sold}, "
                    f"Realized PnL: {sp_realized_pnl:.2f}, Cash released: {sub_cash_released:.2f}, "
                    f"Old shares: {old_shares}, New shares: {sp.shares}")

        self.shares -= shares_to_sell
        self.cash_committed -= cash_released
        self.update_market_value(exit_price)
        self.partial_exit_count += 1

        # ... (rest of the method, including assertions and debug prints)
    # ```

    # 2. In the `partial_entry` method, we can simplify our logic for distributing new shares:

    # ```python
    def partial_entry(self, entry_time: datetime, entry_price: float, shares_to_buy: int):
        print(f"DEBUG: Entering partial_entry - Time: {entry_time}, Price: {entry_price:.4f}, Shares to buy: {shares_to_buy}")
        print(f"DEBUG: Current position - Shares: {self.shares}, Cash committed: {self.cash_committed:.2f}")

        new_total_shares = self.shares + shares_to_buy
        new_num_subs = self.calculate_num_sub_positions(new_total_shares)

        additional_cash_committed = (shares_to_buy * entry_price) / self.times_buying_power
        self.cash_committed += additional_cash_committed

        active_sub_positions = [sp for sp in self.sub_positions if sp.shares > 0]
        current_sub_shares = [sp.shares for sp in active_sub_positions]
        target_shares = self.calculate_shares_per_sub(new_total_shares, new_num_subs, current_sub_shares)

        print(f"DEBUG: Target shares per sub-position: {target_shares}")

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
                    sp.update_market_value(entry_price)
                    self.add_transaction(entry_time, shares_to_add, entry_price, is_entry=True, sub_position=sp)
                    shares_added += shares_to_add
                    print(f"DEBUG: Adding to sub-position {i} - Entry price: {sp.entry_price:.4f}, Shares added: {shares_to_add}, "
                        f"Cash committed: {sub_cash_committed:.2f}, Old shares: {old_shares}, New shares: {sp.shares}")
            else:
                # New sub-position
                sub_cash_committed = (target * entry_price) / self.times_buying_power
                new_sub = SubPosition(entry_time, entry_price, target, sub_cash_committed)
                self.sub_positions.append(new_sub)
                self.add_transaction(entry_time, target, entry_price, is_entry=True, sub_position=new_sub)
                shares_added += target
                print(f"DEBUG: Created new sub-position {i} - Entry price: {entry_price:.4f}, Shares: {target}, "
                    f"Cash committed: {sub_cash_committed:.2f}")

        self.shares += shares_added
        self.update_market_value(entry_price)
        self.partial_entry_count += 1

        # ... (rest of the method, including assertions and debug prints)
    # ```

    # 3. We can also simplify the `calculate_shares_per_sub` method:

    # ```python
    def calculate_shares_per_sub(self, total_shares: int, num_subs: int, current_sub_shares: List[int]) -> List[int]:
        if num_subs == 1:
            return [total_shares]
        elif len(current_sub_shares) == 0:
            # Initial entry
            return [total_shares // 2, total_shares - (total_shares // 2)]
        else:
            # Adjusting existing sub-positions
            total_current_shares = sum(current_sub_shares)
            shares_to_add = total_shares - total_current_shares
            return [s + (shares_to_add * s // total_current_shares) for s in current_sub_shares]
    # ```

    # These changes simplify our logic by focusing only on active sub-positions. We no longer need to worry about already-exited sub-positions, as they are effectively removed from our calculations.

    # This approach should make the code more straightforward and less prone to errors related to inactive sub-positions. It also ensures that we're always working with the correct set of active sub-positions when making adjustments to the position.