@dataclass
class TradePosition:
    # ... (keep existing attributes)
    
    def __post_init__(self):
        self.current_market_value = self.initial_shares * self.entry_price
        self.current_shares = self.initial_shares  # Initialize to initial_shares
        self.last_price = self.entry_price
        self.cash_committed = self.initial_cash_used  # Initialize to initial_cash_used

    def partial_entry(self, entry_time: datetime, entry_price: float, shares_to_buy: int):
        additional_cash_committed = (shares_to_buy * entry_price) / self.actual_margin_multiplier
        self.cash_committed += additional_cash_committed
        self.current_shares += shares_to_buy
        self.current_market_value = self.current_shares * entry_price
        
        self.add_sub_position(entry_time, entry_price, shares_to_buy)
        self.add_transaction(entry_time, shares_to_buy, entry_price, is_entry=True)
        
        return additional_cash_committed

    def partial_exit(self, exit_time: datetime, exit_price: float, shares_to_sell: int):
        if self.is_long:
            realized_pnl = (exit_price - self.entry_price) * shares_to_sell
        else:
            realized_pnl = (self.entry_price - exit_price) * shares_to_sell
        
        cash_released = (shares_to_sell / self.current_shares) * self.cash_committed
        self.cash_committed -= cash_released
        
        self.current_shares -= shares_to_sell
        self.current_market_value = self.current_shares * exit_price
        
        self.add_transaction(exit_time, shares_to_sell, exit_price, is_entry=False)
        
        return realized_pnl / self.actual_margin_multiplier, cash_released

    def update_market_value(self, current_price: float):
        old_market_value = self.current_market_value
        self.current_market_value = self.current_shares * current_price
        self.last_price = current_price
        self.unrealized_pnl = (current_price - self.entry_price) * self.current_shares if self.is_long else (self.entry_price - current_price) * self.current_shares
        return self.current_market_value - old_market_value
    
def backtest_strategy(touch_detection_areas, initial_investment=10000, do_longs=True, do_shorts=True, use_margin=False, times_buying_power=4):
    # ... (keep existing setup)

    def update_total_account_value(current_price):
        nonlocal total_account_value, balance
        market_value_change = sum(position.update_market_value(current_price) for position in open_positions.values())
        cash_committed = sum(position.cash_committed for position in open_positions.values())
        total_account_value = balance + sum(position.current_market_value for position in open_positions.values())
        
        print(f"  update_total_account_value:")
        print(f"    balance: {balance:.6f}")
        print(f"    total_account_value: {total_account_value:.6f}")
        print(f"    cash_committed: {cash_committed:.6f}")
        
        for area_id, position in open_positions.items():
            print(f"      Position {area_id}: Shares: {position.current_shares}, Market Value: {position.current_market_value:.6f}, Cash Committed: {position.cash_committed:.6f}")

    def place_stop_market_buy(area: TouchArea, timestamp: datetime, open_price: float, high_price: float, low_price: float, close_price: float, prev_close: float):
        # ... (keep existing logic)
        
        position = TradePosition(
            # ... (keep existing parameters)
            current_shares=total_shares,  # Set initial current_shares
        )
        
        cash_needed = position.partial_entry(timestamp, execution_price, total_shares)
        
        # ... (keep the rest of the function)

    def update_positions(timestamp, open_price, high_price, low_price, close_price):
        # ... (keep existing logic)

        def calculate_target_shares_percentage(position, current_price):
            price_movement = abs(current_price - position.current_stop_price)
            target_percentage = min(price_movement / position.area.get_range, 1)
            return target_percentage

        for area_id, position in open_positions.items():
            # ... (keep existing logic for trailing stops)

            # Partial exit and entry logic
            target_percentage = calculate_target_shares_percentage(position, close_price)
            current_percentage = position.current_shares / position.initial_shares
            
            if target_percentage != current_percentage:
                debug_print(f'current_percentage = {position.current_shares}/{position.initial_shares}')
                debug_print(f"  Current -> Target percentage: {current_percentage*100:.2f}% -> {target_percentage*100:.2f}%")
            
            if target_percentage < current_percentage:
                shares_to_adjust = math.floor(position.current_shares * (current_percentage - target_percentage))
                if shares_to_adjust > 0:
                    realized_pnl, cash_released = position.partial_exit(timestamp, close_price, shares_to_adjust)
                    rebalance(cash_released + realized_pnl, close_price)
                    print(f"Partial exit: Sold {shares_to_adjust} shares at {close_price:.4f}, Realized PnL: {realized_pnl:.4f}, Cash released: {cash_released:.4f}")

            elif target_percentage > current_percentage:
                shares_to_adjust = math.floor(position.initial_shares * (target_percentage - current_percentage))
                if shares_to_adjust > 0:
                    cash_needed = position.partial_entry(timestamp, close_price, shares_to_adjust)
                    if balance >= cash_needed:
                        rebalance(-cash_needed, close_price)
                        print(f"Partial enter: Bought {shares_to_adjust} shares at {close_price:.4f}")

        update_total_account_value(close_price)
        
        # ... (keep the rest of the function)

    # ... (keep the rest of the backtesting loop)