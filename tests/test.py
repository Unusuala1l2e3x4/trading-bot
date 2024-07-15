@dataclass
class TradePosition:
    # ... other attributes ...
    initial_shares: int
    current_shares: int
    sub_positions: List[SubPosition] = field(default_factory=list)

    @property
    def num_sub_positions(self) -> int:
        if self.times_buying_power <= 2:
            return 1
        elif self.initial_shares % 2 == 0:
            return 2
        else:
            return 3
        
    @property
    def shares_per_sub(self) -> List[int]:
        num_subs = self.num_sub_positions
        if num_subs == 1:
            return [self.initial_shares]
        elif num_subs == 2:
            return [self.initial_shares // 2] * 2
        else:  # 3 sub-positions
            base_shares = self.initial_shares // 3
            extra_shares = self.initial_shares % 3
            return [base_shares + (1 if i < extra_shares else 0) for i in range(3)]

    def add_sub_position(self, entry_time: datetime, entry_price: float, shares: int):
        self.sub_positions.append(SubPosition(entry_time, entry_price, shares))
        self.current_shares += shares

    def partial_exit(self, exit_time: datetime, exit_price: float, shares_to_sell: int) -> float:
        remaining_shares = shares_to_sell
        realized_pnl = 0
        for sp in self.sub_positions:
            if sp.exit_time is None and remaining_shares > 0:
                shares_sold = min(sp.shares, remaining_shares)
                sp.shares -= shares_sold
                remaining_shares -= shares_sold
                if sp.shares == 0:
                    sp.exit_time = exit_time
                    sp.exit_price = exit_price
                realized_pnl += (exit_price - sp.entry_price) * shares_sold if self.is_long else \
                                (sp.entry_price - exit_price) * shares_sold
        self.current_shares -= shares_to_sell
        return realized_pnl


    @staticmethod
    def calculate_shares_per_sub(total_shares: int, num_subs: int) -> List[int]:
        if num_subs == 1:
            return [total_shares]
        elif num_subs == 2:
            return [total_shares // 2] * 2
        else:  # 3 sub-positions
            base_shares = total_shares // 3
            extra_shares = total_shares % 3
            return [base_shares + (1 if i < extra_shares else 0) for i in range(3)]
      
      
    def partial_entry(self, entry_time: datetime, entry_price: float, shares_to_buy: int):
        # Determine how to distribute new shares among sub-positions
        current_num_subs = len([sp for sp in self.sub_positions if sp.shares > 0])
        new_total_shares = self.current_shares + shares_to_buy
        new_num_subs = 3 if new_total_shares % 2 != 0 and self.times_buying_power > 2 else \
                       2 if self.times_buying_power > 2 else 1

        if new_num_subs > current_num_subs:
            # Create new sub-position(s)
            for _ in range(new_num_subs - current_num_subs):
                self.add_sub_position(entry_time, entry_price, 0)

        # Distribute shares
        shares_per_sub = self.calculate_shares_per_sub(new_total_shares, new_num_subs)
        remaining_shares = shares_to_buy
        for i, sp in enumerate(self.sub_positions):
            if sp.shares > 0 or i < new_num_subs:
                target_shares = shares_per_sub[i]
                shares_to_add = target_shares - sp.shares
                if shares_to_add > 0:
                    actual_shares_to_add = min(shares_to_add, remaining_shares)
                    sp.shares += actual_shares_to_add
                    remaining_shares -= actual_shares_to_add
                    if remaining_shares == 0:
                        break

        self.current_shares += shares_to_buy

       
       
       
        
def backtest_strategy(touch_detection_areas, initial_investment=10000, do_longs=True, do_shorts=True, use_margin=False, times_buying_power=4):
    # ...
    
    
    def place_stop_market_buy(area: TouchArea, timestamp: datetime, open_price: float, high_price: float, low_price: float, close_price: float, prev_close: float):
        nonlocal balance, current_id, total_account_value, open_positions, trades_executed

        if open_positions or balance <= 0:
            return NO_POSITION_OPENED

        # ... (existing checks for order execution)

        execution_price = area.get_buy_price

        max_position_size, actual_margin_multiplier, overall_margin_multiplier, initial_margin_requirement = calculate_max_position_size()

        total_shares = math.floor((balance * overall_margin_multiplier) / execution_price)
        invest_amount = total_shares * execution_price
        actual_cash_used = invest_amount / overall_margin_multiplier

        if actual_cash_used > balance:
            return NO_POSITION_OPENED

        position = TradePosition(
            id=current_id,
            area=area,
            is_long=area.is_long,
            entry_time=timestamp,
            initial_balance=actual_cash_used,
            use_margin=use_margin,
            is_marginable=is_marginable,
            times_buying_power=times_buying_power,
            actual_margin_multiplier=actual_margin_multiplier,
            current_stop_price=high_price - area.get_range if area.is_long else low_price + area.get_range,
            max_price=high_price if area.is_long else None,
            min_price=low_price if not area.is_long else None,
            initial_shares=total_shares,
            current_shares=total_shares
        )

        shares_per_sub = position.shares_per_sub
        for shares in shares_per_sub:
            position.add_sub_position(timestamp, execution_price, shares)

        current_id += 1
        open_positions[area.id] = position

        print(f"{'res' if area.is_long else 'sup'} area {area.id}: {current_id} {timestamp} - Enter {'Long ' if area.is_long else 'Short'} at {execution_price:.4f}. "
            f"Shares: {total_shares}, Amount: ${invest_amount:.2f} (Margin: {actual_margin_multiplier:.2f}x, Overall: {overall_margin_multiplier:.2f}x, Sub-positions: {position.num_sub_positions})")

        rebalance(add=-actual_cash_used, current_price=close_price)
        return POSITION_OPENED
    
    
    
