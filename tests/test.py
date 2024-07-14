@dataclass
class TradePosition:
    # ... (existing attributes)
    current_shares: int  # Initialize this to total_shares
    partial_exits: List[Tuple[datetime, float, int]] = field(default_factory=list)
    partial_entries: List[Tuple[datetime, float, int]] = field(default_factory=list)

    # ... (existing methods)

    def partial_exit(self, exit_time: datetime, exit_price: float, shares_to_sell: int):
        if shares_to_sell > self.current_shares:
            shares_to_sell = self.current_shares
        self.partial_exits.append((exit_time, exit_price, shares_to_sell))
        self.current_shares -= shares_to_sell

    def partial_entry(self, entry_time: datetime, entry_price: float, shares_to_buy: int):
        if self.current_shares + shares_to_buy > self.total_shares:
            shares_to_buy = self.total_shares - self.current_shares
        self.partial_entries.append((entry_time, entry_price, shares_to_buy))
        self.current_shares += shares_to_buy

    def distance_to_stop(self, current_price: float) -> float:
        return abs(current_price - self.current_stop_price)
    
    
        
def backtest_strategy(touch_detection_areas, initial_investment=10000, do_longs=True, do_shorts=True, use_margin=False, times_buying_power=4):
    # ...
    def update_positions(timestamp, open_price, close_price, high_price, low_price):
        nonlocal trades_executed
        positions_to_remove = []

        def perform_exit(area_id, position, exit_price):
            nonlocal trades_executed
            position.close(timestamp, exit_price)
            trades_executed += 1
            position.area.record_entry_exit(position.entry_time, position.entry_price, 
                                            timestamp, exit_price)
            position.area.terminate(touch_area_collection)
            positions_to_remove.append(area_id)

        def handle_partial_trades(position, price):
            total_gain = position.max_price - position.entry_price if position.is_long else position.entry_price - position.min_price
            distance_to_stop = position.distance_to_stop(price)

            if distance_to_stop < total_gain * 0.5:  # Start scaling out at 50% retracement
                shares_to_sell = int(position.current_shares * 0.1)  # Sell 10% of current position
                if shares_to_sell > 0:
                    position.partial_exit(timestamp, price, shares_to_sell)
            elif distance_to_stop > total_gain * 0.75 and position.current_shares < position.total_shares:
                shares_to_buy = int((position.total_shares - position.current_shares) * 0.1)  # Buy back 10% of sold shares
                if shares_to_buy > 0:
                    position.partial_entry(timestamp, price, shares_to_buy)

        for area_id, position in open_positions.items():
            old_stop_price = position.current_stop_price
            debug_print(f"{timestamp} - Updating position {position.id} (Old stop price: {old_stop_price:.4f})")

            # Check at open
            position.update_stop_price(open_price)
            handle_partial_trades(position, open_price)
            if position.should_exit(open_price):
                perform_exit(area_id, position, open_price)
                continue

            # Check at high/low for long/short positions
            if position.is_long:
                position.update_stop_price(high_price)
                handle_partial_trades(position, high_price)
                if position.should_exit(high_price):
                    perform_exit(area_id, position, high_price)
                    continue
            else:
                position.update_stop_price(low_price)
                handle_partial_trades(position, low_price)
                if position.should_exit(low_price):
                    perform_exit(area_id, position, low_price)
                    continue

            # Check at close
            position.update_stop_price(close_price)
            handle_partial_trades(position, close_price)
            if position.should_exit(close_price):
                perform_exit(area_id, position, close_price)

        # Remove closed positions and update account
        temp = {}
        for area_id in positions_to_remove:
            temp[area_id] = open_positions[area_id]
            del open_positions[area_id]
        for area_id in positions_to_remove:
            exit_action(area_id, temp[area_id])

        if positions_to_remove:
            debug_print(f"  Updated Total Account Value: {total_account_value:.2f}")