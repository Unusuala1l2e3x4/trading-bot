@dataclass
class TradePosition:
    id: int
    area: TouchArea
    is_long: bool
    entry_time: datetime
    initial_balance: float
    initial_shares: int
    use_margin: bool
    is_marginable: bool
    times_buying_power: float
    actual_margin_multiplier: float
    initial_cash_used: float
    entry_price: float
    current_market_value: float = 0
    current_shares: int = 0
    sub_positions: List[SubPosition] = field(default_factory=list)
    transactions: List[Transaction] = field(default_factory=list)
    current_stop_price: Optional[float] = None
    max_price: Optional[float] = None
    min_price: Optional[float] = None
    last_price: float = field(default=0.0)
    cash_committed: float = 0
    unrealized_pnl: float = field(default=0.0)
    # stock_borrow_rate: float = 0.003    # Default to 30 bps (0.3%) annually
    stock_borrow_rate: float = 0.03      # Default to 300 bps (3%) annually
    
    FINRA_TAF_RATE = 0.000119  # per share
    SEC_FEE_RATE = 22.90 / 1_000_000  # per dollar
    
    def __post_init__(self):
        self.current_market_value = self.initial_shares * self.entry_price
        self.current_shares = 0  # Ensure it starts at 0
        self.last_price = self.entry_price
        self.cash_committed = 0  # Initialize to 0, will be set in partial_entry
        
    def add_shares(self, shares: int):
        self.current_shares += shares
        
    @property
    def is_open(self) -> bool:
        return any(sp.exit_time is None for sp in self.sub_positions)

    @property
    def total_shares(self) -> int:
        return sum(sp.shares for sp in self.sub_positions if sp.exit_time is None)

    def update_last_price(self, price: float):
        old_market_value = self.current_market_value
        self.last_price = price
        self.current_market_value = self.current_shares * price
        self.unrealized_pnl = self.current_market_value - (self.current_shares * self.entry_price)
        return self.current_market_value - old_market_value
        
        
    def update_market_value(self, current_price: float):
        old_market_value = self.current_market_value
        self.current_market_value = self.current_shares * current_price
        self.last_price = current_price
        return self.current_market_value - old_market_value
    
    @property
    def cash_value(self):
        return self.initial_cash_used
    @property
    def equity_value(self):
        return (self.current_market_value - self.initial_cash_used * self.actual_margin_multiplier) / self.actual_margin_multiplier
    
    @property
    def margin_multiplier(self) -> float:
        return self.actual_margin_multiplier

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

    def add_transaction(self, timestamp: datetime, shares: int, price: float, is_entry: bool):
        transaction_cost = self.calculate_transaction_cost(shares, price, is_entry)
        transaction = Transaction(timestamp, shares, price, is_entry, transaction_cost)
        self.transactions.append(transaction)
        
        print(f'add_transaction {timestamp}, {shares}, {price:.4f}, {is_entry}, {transaction_cost:.4f}')
        
    def add_sub_position(self, entry_time: datetime, entry_price: float, shares: int):
        self.sub_positions.append(SubPosition(entry_time, entry_price, shares))
        self.add_shares(shares)
        print(f'buying {shares} shares at {entry_time}, {entry_price:.4f} ({shares*entry_price:.4f})')
        self.add_transaction(entry_time, shares, entry_price, is_entry=True)

    def partial_exit(self, exit_time: datetime, exit_price: float, shares_to_sell: int):
        if self.is_long:
            realized_pnl = (exit_price - self.entry_price) * shares_to_sell
        else:
            realized_pnl = (self.entry_price - exit_price) * shares_to_sell
        
        cash_released = (shares_to_sell / self.current_shares) * self.cash_committed
        self.cash_committed -= cash_released
        
        remaining_shares = shares_to_sell
        for sp in self.sub_positions:
            if sp.exit_time is None and remaining_shares > 0:
                shares_sold = min(sp.shares, remaining_shares)
                sp.shares -= shares_sold
                remaining_shares -= shares_sold
                if sp.shares == 0:
                    sp.exit_time = exit_time
                    sp.exit_price = exit_price
        
        self.current_shares -= shares_to_sell
        self.current_market_value = self.current_shares * exit_price
        
        self.add_transaction(exit_time, shares_to_sell, exit_price, is_entry=False)
        
        return realized_pnl / self.actual_margin_multiplier, cash_released        
    
    
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
        additional_cash_committed = (shares_to_buy * entry_price) / self.actual_margin_multiplier
        self.cash_committed += additional_cash_committed
        self.current_shares += shares_to_buy
        self.current_market_value = self.current_shares * entry_price
        
        # Create sub-positions
        shares_per_sub = self.calculate_shares_per_sub(self.current_shares, self.num_sub_positions)
        for shares in shares_per_sub:
            self.add_sub_position(entry_time, entry_price, shares)
        
        self.add_transaction(entry_time, shares_to_buy, entry_price, is_entry=True)
        
        return additional_cash_committed
            
            
    def update_stop_price(self, current_price: float):
        if self.is_long:
            self.max_price = max(self.max_price or self.entry_price, current_price)
            self.current_stop_price = self.max_price - self.area.get_range
        else:
            self.min_price = min(self.min_price or self.entry_price, current_price)
            self.current_stop_price = self.min_price + self.area.get_range

    def should_exit(self, current_price: float) -> bool:
        return (self.is_long and current_price <= self.current_stop_price) or \
               (not self.is_long and current_price >= self.current_stop_price)

    def close(self, exit_time: datetime, exit_price: float):
        for sp in self.sub_positions:
            if sp.exit_time is None:
                sp.exit_time = exit_time
                sp.exit_price = exit_price

    def calculate_transaction_cost(self, shares: int, price: float, is_entry: bool) -> float:
        finra_taf = max(0.01, self.FINRA_TAF_RATE * shares)
        sec_fee = 0
        if not is_entry or (not self.is_long and is_entry):
            trade_value = price * shares
            sec_fee = self.SEC_FEE_RATE * trade_value
        return finra_taf + sec_fee

    @property
    def entry_transaction_costs(self) -> float:
        return sum(t.transaction_cost for t in self.transactions if t.is_entry)

    @property
    def exit_transaction_costs(self) -> float:
        return sum(t.transaction_cost for t in self.transactions if not t.is_entry)

    @property
    def total_transaction_costs(self) -> float:
        return sum(t.transaction_cost for t in self.transactions)

    @property
    def stock_borrow_cost(self) -> float:
        if self.is_long:
            return 0
        total_cost = 0
        for i in range(len(self.transactions) - 1):
            current_transaction = self.transactions[i]
            next_transaction = self.transactions[i + 1]
            holding_days = (next_transaction.timestamp - current_transaction.timestamp).total_seconds() / (24 * 60 * 60)
            daily_borrow_rate = self.stock_borrow_rate / 360
            shares = sum(sp.shares for sp in self.sub_positions if sp.entry_time <= current_transaction.timestamp and (sp.exit_time is None or sp.exit_time > next_transaction.timestamp))
            total_cost += shares * current_transaction.price * daily_borrow_rate * holding_days
        return total_cost


    @property
    def holding_time(self) -> timedelta:
        if not self.sub_positions:
            return timedelta(0)
        start_time = min(sp.entry_time for sp in self.sub_positions)
        end_time = max(sp.exit_time or datetime.now() for sp in self.sub_positions)
        return end_time - start_time

    @property
    def profit_loss(self) -> float:
        realized_pl = sum((sp.exit_price - sp.entry_price) * sp.shares if self.is_long else
                        (sp.entry_price - sp.exit_price) * sp.shares
                        for sp in self.sub_positions if sp.exit_time is not None)
        
        # Use last_price for unrealized P&L
        total_pl = (realized_pl + self.unrealized_pnl) / self.actual_margin_multiplier
        
        return total_pl - self.total_transaction_costs

    @property
    def profit_loss_percentage(self) -> float:
        return (self.profit_loss / self.initial_balance) * 100

    @property
    def return_on_equity(self) -> float:
        equity_used = self.initial_balance / self.margin_multiplier
        return (self.profit_loss / equity_used) * 100

    @property
    def price_diff(self) -> float:
        if not self.sub_positions or any(sp.exit_time is None for sp in self.sub_positions):
            return 0
        avg_entry_price = sum(sp.entry_price * sp.shares for sp in self.sub_positions) / sum(sp.shares for sp in self.sub_positions)
        avg_exit_price = sum(sp.exit_price * sp.shares for sp in self.sub_positions) / sum(sp.shares for sp in self.sub_positions)
        diff = avg_exit_price - avg_entry_price
        return diff if self.is_long else -diff

    def current_value(self, current_price: float) -> float:
        market_value = self.current_shares * current_price
        if self.is_long:
            profit_loss = (current_price - self.entry_price) * self.current_shares
        else:
            profit_loss = (self.entry_price - current_price) * self.current_shares
        return self.initial_cash_used + (profit_loss / self.actual_margin_multiplier)
                
    @property
    def total_investment(self) -> float:
        return sum(sp.entry_price * sp.shares for sp in self.sub_positions)

    @property
    def margin_used(self) -> float:
        return self.total_investment - self.initial_balance
    
    
def backtest_strategy(touch_detection_areas, initial_investment=10000, do_longs=True, do_shorts=True, use_margin=False, times_buying_power=4):
    symbol = touch_detection_areas['symbol']
    long_touch_area = touch_detection_areas['long_touch_area']
    short_touch_area = touch_detection_areas['short_touch_area']
    market_hours = touch_detection_areas['market_hours']
    df = touch_detection_areas['bars']
    mask = touch_detection_areas['mask']
    bid_buffer_pct = touch_detection_areas['bid_buffer_pct']
    min_touches = touch_detection_areas['min_touches']
    sell_time = touch_detection_areas['sell_time']
    use_median = touch_detection_areas['use_median']
    
    debug = True
    def debug_print(message):
        if debug:
            print(message)

    POSITION_OPENED = True
    NO_POSITION_OPENED = False
    
    all_touch_areas = []
    assert do_longs or do_shorts
    if do_longs:
        all_touch_areas.extend(long_touch_area)
    if do_shorts:
        all_touch_areas.extend(short_touch_area)
    touch_area_collection = TouchAreaCollection(all_touch_areas, min_touches)
        
    is_marginable = is_security_marginable(symbol)
    if not use_margin or not is_marginable:
        use_margin = False
        times_buying_power = 1
        
        
    print(f'{symbol} is {'NOT ' if not is_marginable else ''}marginable.')
    # return

    trades = []  # List to store all trades
    
    df = df[mask]
    df = df.sort_index(level='timestamp')
    timestamps = df.index.get_level_values('timestamp')

    balance = initial_investment
    total_account_value = initial_investment
    # buying_power = initial_investment * times_buying_power if use_margin else initial_investment
    
    trades_executed = 0
    open_positions = {}
    current_id = 0
        
    def update_total_account_value(current_price):
        nonlocal total_account_value, balance
        market_value_change = sum(position.update_market_value(current_price) for position in open_positions.values())
        cash_committed = sum(position.cash_committed for position in open_positions.values())
        total_account_value = balance + sum(position.current_market_value for position in open_positions.values()) - cash_committed
        
        print(f"  update_total_account_value:")
        print(f"    balance: {balance:.6f}")
        print(f"    total_account_value: {total_account_value:.6f}")
        
        for area_id, position in open_positions.items():
            print(f"      Position {area_id}: Shares: {position.current_shares}, Market Value: {position.current_market_value:.6f}, Cash Used: {position.cash_value:.6f}")

                        
    def rebalance(cash_change: float, current_price: float = None):
        nonlocal balance, total_account_value
        old_balance = balance
        balance += cash_change

        assert balance >= 0, f"Balance became negative: {balance}"

        if current_price is not None:
            update_total_account_value(current_price)
        print(f"Rebalance: Old balance: {old_balance:.4f}, Change: {(balance - old_balance):.4f}, New balance: {balance:.4f}, Total Account Value: {total_account_value:.4f}")
                        
    def exit_action(area_id, position):
        nonlocal trades, balance, total_account_value
        
        exit_price = position.current_stop_price
        if position.is_long:
            profit_loss = (exit_price - position.entry_price) * position.current_shares
        else:
            profit_loss = (position.entry_price - exit_price) * position.current_shares
        
        cash_to_add = position.initial_cash_used + (profit_loss / position.actual_margin_multiplier)
        
        print(f"  exit_action:")
        print(f"    entry_price: {position.entry_price:.4f}")
        print(f"    exit_price: {exit_price:.4f}")
        print(f"    shares: {position.current_shares}")
        print(f"    initial_cash_used: {position.initial_cash_used:.4f}")
        print(f"    profit_loss: {profit_loss:.4f}")
        print(f"    cash_to_add: {cash_to_add:.4f}")

        rebalance(cash_to_add, current_price=exit_price)

        print(f"{'res' if position.is_long else 'sup'} area {area_id}:\t{position.id} {position.sub_positions[-1].exit_time} - Exit {'Long ' if position.is_long else 'Short'} at {exit_price:.4f} "
            f"({'+'if position.price_diff >= 0 else ''}{position.price_diff:.4f}). "
            f"Net {'+'if profit_loss >= 0 else ''}{profit_loss / position.actual_margin_multiplier:.4f} "
            f"(ROE: {(profit_loss / position.initial_cash_used) * 100:.4f}%, After -{position.total_transaction_costs:.4f} fees)")
        trades.append(position)
        
        
    def close_all_positions(timestamp, exit_price):
        nonlocal trades_executed
        if not open_positions:
            return
        
        positions_to_remove = []

        for area_id, position in list(open_positions.items()):
            position.close(timestamp, exit_price)
            trades_executed += 1
            position.area.record_entry_exit(position.entry_time, position.entry_price, 
                                            timestamp, exit_price)
            position.area.terminate(touch_area_collection)
            positions_to_remove.append(area_id)

        temp = {}
        for area_id in positions_to_remove:
            temp[area_id] = open_positions[area_id]
            del open_positions[area_id]
        for area_id in positions_to_remove:
            exit_action(area_id, temp[area_id])

        assert not open_positions
        open_positions.clear()

    def calculate_max_position_size():
        nonlocal total_account_value
        
        if use_margin and is_marginable:
            initial_margin_requirement = 0.5  # 50% for marginable securities
            overall_margin_multiplier = min(times_buying_power, 4)
            actual_margin_multiplier = min(overall_margin_multiplier, 2.0)
        else:
            initial_margin_requirement = 1.0  # 100% for non-marginable securities
            overall_margin_multiplier = 1.0
            actual_margin_multiplier = 1.0

        available_margin = balance * overall_margin_multiplier
        used_margin = sum(p.initial_balance for p in open_positions.values())
        available_buying_power = available_margin - used_margin
        
        max_position_size = available_buying_power / initial_margin_requirement

        return max_position_size, actual_margin_multiplier, overall_margin_multiplier, initial_margin_requirement

    def place_stop_market_buy(area: TouchArea, timestamp: datetime, open_price: float, high_price: float, low_price: float, close_price: float, prev_close: float):
        nonlocal balance, current_id, total_account_value, open_positions, trades_executed

        if open_positions or balance <= 0:
            return NO_POSITION_OPENED

        debug_print(f"Attempting order: {'Long' if area.is_long else 'Short'} at {area.get_buy_price:.4f}")
        debug_print(f"  Balance: {balance:.4f}, Total Account Value: {total_account_value:.4f}")


        # Check if the stop buy would have executed based on high/low. close price unnecessary (i think)
        if area.is_long:
            if prev_close > area.get_buy_price:
                debug_print(f"  Rejected: Previous close ({prev_close:.4f}) above buy price, likey re-entering area ({area.get_buy_price:.4f})")
                return NO_POSITION_OPENED
            if high_price < area.get_buy_price:
                debug_print(f"  Rejected: High price ({high_price:.4f}) didn't reach buy price ({area.get_buy_price:.4f})")
                return NO_POSITION_OPENED
        else:  # short
            if prev_close < area.get_buy_price:
                debug_print(f"  Rejected: Previous close ({prev_close:.4f}) below buy price, likey re-entering area ({area.get_buy_price:.4f})")
                return NO_POSITION_OPENED
            if low_price > area.get_buy_price:
                debug_print(f"  Rejected: Low price ({low_price:.4f}) didn't reach buy price ({area.get_buy_price:.4f})")
                return NO_POSITION_OPENED

        # Stop buy (placed at time of min_touches) would have executed ############### IMPORTANT
        execution_price = area.get_buy_price


        # Calculate position size, etc...
        max_position_size, actual_margin_multiplier, overall_margin_multiplier, initial_margin_requirement = calculate_max_position_size()
        
        # Calculate total shares and round down to whole number
        total_shares = math.floor((balance * overall_margin_multiplier) / execution_price)

        # Recalculate invest_amount based on rounded total_shares
        invest_amount = total_shares * execution_price
        actual_cash_used = invest_amount / overall_margin_multiplier
        
        if actual_cash_used > balance:
            debug_print(f"  Order rejected: Insufficient balance ({actual_cash_used:.4f} > {balance:.4f})")
            return NO_POSITION_OPENED

        debug_print(f"  Shares: {total_shares}, Invest Amount: {invest_amount:.4f}")
        debug_print(f"  Margin Multiplier: {actual_margin_multiplier:.2f}")
        
        debug_print(f"  execution_price: {execution_price:.2f}")
        
        
        # Create the position
        position = TradePosition(
            id=current_id,
            area=area,
            is_long=area.is_long,
            entry_time=timestamp,
            initial_balance=actual_cash_used,
            initial_cash_used=actual_cash_used,
            current_market_value=invest_amount,
            initial_shares=total_shares,
            current_shares=0,
            entry_price=execution_price,
            use_margin=use_margin,
            is_marginable=is_marginable,
            times_buying_power=times_buying_power,
            actual_margin_multiplier=actual_margin_multiplier,
            current_stop_price=high_price - area.get_range if area.is_long else low_price + area.get_range,
            max_price=high_price if area.is_long else None,
            min_price=low_price if not area.is_long else None,
        )
        
        # Create sub-positions
        cash_needed = position.partial_entry(timestamp, execution_price, total_shares)
        
        current_id += 1
        
        # Add to open positions
        open_positions[area.id] = position
        
        print(f"{'res' if area.is_long else 'sup'} area {area.id}: {current_id} {timestamp} - Enter {'Long ' if area.is_long else 'Short'} at {execution_price:.4f}. "
              f"Shares: {total_shares}, Amount: ${invest_amount:.4f} (Margin: {actual_margin_multiplier:.2f}x, Overall: {overall_margin_multiplier:.2f}x, Sub-positions: {len(position.sub_positions)})")

        rebalance(-cash_needed, close_price)
        return POSITION_OPENED


    def update_positions(timestamp, open_price, high_price, low_price, close_price):
        nonlocal trades_executed
        positions_to_remove = []

        def perform_exit(area_id, position):
            nonlocal trades_executed
            position.close(timestamp, position.current_stop_price)
            trades_executed += 1
            position.area.record_entry_exit(position.entry_time, position.entry_price, 
                                            timestamp, position.current_stop_price)
            position.area.terminate(touch_area_collection)
            positions_to_remove.append(area_id)
            

        def calculate_target_shares_percentage(position, current_price):
            price_movement = abs(current_price - position.current_stop_price)
            target_percentage = min(price_movement / position.area.get_range, 1)
            # debug_print(f'price_movement = abs({current_price}-{position.current_stop_price})')
            # debug_print(f'target_percentage = {price_movement}/{position.area.get_range}')
            
            return target_percentage

        for area_id, position in open_positions.items():
            # old_stop_price = position.current_stop_price
            position.update_last_price(close_price)

            debug_print(f"Updating position {position.id} at {timestamp}")
            debug_print(f"Current shares: {position.current_shares}, Initial shares: {position.initial_shares}")
            debug_print(f"Current stop price: {position.current_stop_price:.4f}")
            
        
            # OHLC logic for trailing stops
            position.update_stop_price(open_price)
            debug_print(f"  Stop price at open: {position.current_stop_price:.4f} ")
            if position.should_exit(open_price): 
                debug_print(f"  Trailing Stop - Exited at open: {open_price:.4f}")
                perform_exit(area_id, position)
                continue
            
            # If not stopped out at open, simulate intra-minute price movement
            if position.is_long:
                # For long positions, the stop moves up if high price increases
                position.update_stop_price(high_price)
                debug_print(f"  Stop price at high: {position.current_stop_price:.4f} ")
                # Check if low price hit the stop
                if position.should_exit(high_price):
                    debug_print(f"  Trailing Stop - Exited at high: {high_price:.4f}")
                    perform_exit(area_id, position)
                    continue
            else:
                # For short positions, the stop moves down if low price decreases
                position.update_stop_price(low_price)
                debug_print(f"  Stop price at low: {position.current_stop_price:.4f} ")
                # Check if high price hit the stop
                if position.should_exit(low_price):
                    debug_print(f"  Trailing Stop - Exited at low: {low_price:.4f}")
                    perform_exit(area_id, position)
                    continue

            position.update_stop_price(close_price)
            debug_print(f"  Stop price at close: {position.current_stop_price:.4f} ")
            if position.should_exit(close_price):
                debug_print(f"  Trailing Stop - Exited at close: {close_price:.4f}")
                perform_exit(area_id, position)
                continue


            # Partial exit and entry logic
            target_percentage = calculate_target_shares_percentage(position, close_price)
            current_percentage = position.current_shares / position.initial_shares
            
            if target_percentage != current_percentage:
                debug_print(f'current_percentage = {position.current_shares}/{position.initial_shares}')
                debug_print(f"  Current -> Target percentage: {current_percentage*100}% -> {target_percentage*100}%")
            
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
        
        temp = {}
        for area_id in positions_to_remove:
            temp[area_id] = open_positions[area_id]
            del open_positions[area_id]
        for area_id in positions_to_remove:
            exit_action(area_id, temp[area_id])
        
        # update_total_account_value(close_price)
        if positions_to_remove:
            debug_print(f"  Updated Total Account Value: {total_account_value:.4f}")

    # print(f"START\nStrategy: {'Long' if do_longs else ''}{'&' if do_longs and do_shorts else ''}{'Short' if do_shorts else ''}.")
    debug_print(f"Strategy: {'Long' if do_longs else ''}{'&' if do_longs and do_shorts else ''}{'Short' if do_shorts else ''}")
    print(f"{timestamps[0]} -> {timestamps[-1]}")
    # print('Initial Investment:', initial_investment)
    debug_print(f"Initial Investment: {initial_investment}, Times Buying Power: {times_buying_power}")
    print('Number of touch areas:', len(all_touch_areas))
    

    daily_data = None
    current_date, market_open, market_close = None, None, None
    for i in tqdm(range(1, len(timestamps))):
        current_time = timestamps[i]
        print(current_time)
        # print(current_time, len(open_positions))
        
        if current_time.date() != current_date:
            debug_print(f"\nNew trading day: {current_time.date()}")
            # New day, reset daily data
            current_date = current_time.date()
            market_open, market_close = market_hours.get(str(current_date), (None, None))
            if market_open and market_close:
                if sell_time:
                    market_close = min(
                        datetime.strptime(str(current_date), '%Y-%m-%d') + timedelta(hours=sell_time.hour, minutes=sell_time.minute),
                        market_close - timedelta(minutes=3)
                    )
            daily_data = df[timestamps.date == current_date]
            daily_index = 1 # start at 2nd position
            assert not open_positions
        
        if not market_open or not market_close:
            continue

        if market_open <= current_time < market_close and i != len(df)-1:
            # debug_print(f"\n{current_time} - Market Open")
            
            prev_close = daily_data['close'].iloc[daily_index - 1]
            open_price = daily_data['open'].iloc[daily_index]
            high_price = daily_data['high'].iloc[daily_index]
            low_price = daily_data['low'].iloc[daily_index]
            close_price = daily_data['close'].iloc[daily_index]

            
            update_positions(current_time, open_price, close_price, high_price, low_price)
            
            active_areas = touch_area_collection.get_active_areas(current_time)
            for area in active_areas:
                if balance <= 0:
                    break
                if open_positions:
                    break
                
                if (area.is_long and do_longs) or (not area.is_long and do_shorts):

                    if place_stop_market_buy(area, current_time, open_price, high_price, low_price, close_price, prev_close):
                        break  # Exit the loop after placing a position
            
            daily_index += 1
        elif current_time >= market_close or i >= len(df)-1:
            debug_print(f"\n{current_time} - Market Close")
            close_all_positions(current_time, df['close'].iloc[i])

# ....
