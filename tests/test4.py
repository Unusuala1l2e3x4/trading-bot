@dataclass
class TradePosition:
    # ... (existing attributes)
    actual_margin_multiplier: float
    times_buying_power: float

    def calculate_num_sub_positions(self, total_shares: int) -> int:
        if self.times_buying_power <= 2:
            return 1
        elif total_shares == self.initial_shares:
            return 2 if total_shares % 2 == 0 else 3
        else:
            # For partial positions, calculate the minimum number of sub-positions needed
            min_sub_positions = math.ceil(total_shares / (self.initial_shares / 2))
            return min(min_sub_positions, 3)  # Cap at 3 sub-positions

    def partial_entry(self, entry_time: datetime, entry_price: float, shares_to_buy: int):
        new_total_shares = self.current_shares + shares_to_buy
        new_num_subs = self.calculate_num_sub_positions(new_total_shares)
        
        additional_cash_committed = (shares_to_buy * entry_price) / self.actual_margin_multiplier
        self.remaining_cash_committed += additional_cash_committed

        current_sub_shares = [sp.shares for sp in self.sub_positions if sp.shares > 0]
        target_shares = self.calculate_shares_per_sub(new_total_shares, new_num_subs, current_sub_shares)
        
        print(f"Debug - current_sub_shares: {current_sub_shares}, target_shares: {target_shares}, all_sub_shares: {[sp.shares for sp in self.sub_positions]}")

        remaining_shares = shares_to_buy
        for i in range(new_num_subs):
            if i < len(self.sub_positions) and self.sub_positions[i].shares > 0:
                shares_to_add = target_shares[i] - self.sub_positions[i].shares
                if shares_to_add > 0:
                    self.sub_positions[i].shares += shares_to_add
                    self.sub_positions[i].cash_committed += (shares_to_add * entry_price) / self.actual_margin_multiplier
                    self.add_transaction(entry_time, shares_to_add, entry_price, is_entry=True)
                    remaining_shares -= shares_to_add
            else:
                new_shares = min(target_shares[i], remaining_shares)
                new_cash_committed = (new_shares * entry_price) / self.actual_margin_multiplier
                new_sub_position = SubPosition(entry_time, entry_price, new_shares, new_cash_committed)
                if i < len(self.sub_positions):
                    self.sub_positions[i] = new_sub_position
                else:
                    self.sub_positions.append(new_sub_position)
                self.add_transaction(entry_time, new_shares, entry_price, is_entry=True)
                remaining_shares -= new_shares

            if remaining_shares == 0:
                break

        # self.current_shares += shares_to_buy
        self.current_market_value = self.current_shares * entry_price
        
        print(f"Debug - After entry: all_sub_shares: {[sp.shares for sp in self.sub_positions]}")

        return additional_cash_committed

    def partial_exit(self, exit_time: datetime, exit_price: float, shares_to_sell: int):
        if self.is_long:
            realized_pnl = (exit_price - self.entry_price) * shares_to_sell
        else:
            realized_pnl = (self.entry_price - exit_price) * shares_to_sell
        
        cash_released = 0
        remaining_shares_to_sell = shares_to_sell

        for sp in reversed(self.sub_positions):
            if sp.shares > 0 and remaining_shares_to_sell > 0:
                shares_sold = min(sp.shares, remaining_shares_to_sell)
                cash_released_from_sub = (shares_sold / sp.shares) * sp.cash_committed
                sp.shares -= shares_sold
                sp.cash_committed -= cash_released_from_sub
                cash_released += cash_released_from_sub
                remaining_shares_to_sell -= shares_sold

                if sp.shares == 0:
                    sp.exit_time = exit_time
                    sp.exit_price = exit_price

        self.current_shares -= shares_to_sell
        self.current_market_value = self.current_shares * exit_price
        self.remaining_cash_committed -= cash_released
        
        self.add_transaction(exit_time, shares_to_sell, exit_price, is_entry=False)
        
        return realized_pnl / self.actual_margin_multiplier, cash_released