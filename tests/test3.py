class SubPosition:
    entry_time: datetime
    entry_price: float
    shares: int
    cash_committed: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None

class TradePosition:
    #...
    def calculate_num_sub_positions(self, total_shares: int) -> int:
        if self.times_buying_power <= 2:
            return 1
        elif total_shares % 2 == 0:
            return 2
        else:
            return 3
        
    def partial_exit(self, exit_time: datetime, exit_price: float, shares_to_sell: int):
        if self.is_long:
            realized_pnl = (exit_price - self.entry_price) * shares_to_sell
        else:
            realized_pnl = (self.entry_price - exit_price) * shares_to_sell
        cash_released = 0
        remaining_shares_to_sell = shares_to_sell

        for sp in self.sub_positions:
            if sp.shares > 0 and remaining_shares_to_sell > 0:
                shares_sold = min(sp.shares, remaining_shares_to_sell)
                assert sp.shares >= shares_sold
                sub_cash_released = (shares_sold / sp.shares) * sp.cash_committed
                sp.shares -= shares_sold
                self.current_shares -= shares_sold
                sp.cash_committed -= sub_cash_released
                cash_released += sub_cash_released
                self.add_transaction(exit_time, shares_sold, exit_price, is_entry=False)
                remaining_shares_to_sell -= shares_sold
                if sp.shares == 0:
                    sp.exit_time = exit_time
                    sp.exit_price = exit_price
        self.current_market_value = self.current_shares * exit_price
        self.cash_committed -= cash_released
        return realized_pnl / self.actual_margin_multiplier, cash_released

    def calculate_shares_per_sub(self, total_shares: int, num_subs: int, current_sub_shares: List[int]) -> List[int]:
        target_shares = [total_shares // num_subs] * num_subs
        remaining_shares = total_shares % num_subs
        
        # Distribute remaining shares
        for i in range(remaining_shares):
            target_shares[i] += 1
            
        # Adjust target shares based on current sub-position shares
        for i in range(min(num_subs, len(current_sub_shares))):
            print(current_sub_shares[i], target_shares[i])
            if target_shares[i] < current_sub_shares[i]:
                excess = current_sub_shares[i] - target_shares[i]
                target_shares[i] = current_sub_shares[i]
                
                # Redistribute excess to other sub-positions
                for j in range(i + 1, num_subs):
                    if target_shares[j] > excess:
                        target_shares[j] -= excess
                        break
                    else:
                        excess -= target_shares[j]
                        target_shares[j] = 0
        return target_shares


    def partial_entry(self, entry_time: datetime, entry_price: float, shares_to_buy: int):
        new_total_shares = self.current_shares + shares_to_buy
        new_num_subs = self.calculate_num_sub_positions(new_total_shares)
        
        additional_cash_committed = (shares_to_buy * entry_price) / self.times_buying_power
        self.cash_committed += additional_cash_committed

        active_sub_positions = [sp for sp in self.sub_positions if sp.shares > 0]
        current_sub_shares = [sp.shares for sp in active_sub_positions]
        target_shares = self.calculate_shares_per_sub(new_total_shares, new_num_subs, current_sub_shares)

        remaining_shares = shares_to_buy
        new_sub_positions = []
        for i in range(new_num_subs):
            if i < len(active_sub_positions):
                shares_to_add = target_shares[i] - active_sub_positions[i].shares
                if shares_to_add > 0:
                    sub_cash_committed = (shares_to_add * entry_price) / self.actual_margin_multiplier
                    active_sub_positions[i].shares += shares_to_add
                    self.current_shares += shares_to_add
                    active_sub_positions[i].cash_committed += sub_cash_committed
                    self.add_transaction(entry_time, shares_to_add, entry_price, is_entry=True)
                    remaining_shares -= shares_to_add
            else:
                new_shares = min(target_shares[i], remaining_shares)
                self.current_shares += new_shares
                sub_cash_committed = (new_shares * entry_price) / self.actual_margin_multiplier
                new_sub_positions.append(SubPosition(entry_time, entry_price, new_shares, sub_cash_committed))
                self.add_transaction(entry_time, new_shares, entry_price, is_entry=True)
                remaining_shares -= new_shares
            if remaining_shares == 0:
                break
        # Add new sub-positions at the end
        self.sub_positions.extend(new_sub_positions)
        self.current_market_value = self.current_shares * entry_price
        return additional_cash_committed