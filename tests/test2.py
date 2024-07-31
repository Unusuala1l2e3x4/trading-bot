
# in touch_detection_area_v6.ipynb:
# 
def backtest_strategy(...): # NOT in TradePosition class
#...
    
    def calculate_max_position_size(current_price):
        nonlocal balance, use_margin, is_marginable, times_buying_power
        
        if use_margin and is_marginable:
            initial_margin_requirement = 0.5  # 50% for marginable securities
            overall_margin_multiplier = min(times_buying_power, 4.0)
        else:
            initial_margin_requirement = 1.0  # 100% for non-marginable securities
            overall_margin_multiplier = min(times_buying_power, 1.0)  # Allow for times_buying_power < 1
            
        actual_margin_multiplier = min(overall_margin_multiplier, 1.0/initial_margin_requirement)
        
        # Calculate max shares without considering entry cost
        max_shares_before_costs = math.floor(balance * overall_margin_multiplier / current_price)
        
        # Estimate entry cost
        estimated_entry_cost = TradePosition.estimate_entry_cost(max_shares_before_costs) * overall_margin_multiplier
        
        # currently, estimate_entry_cost maxes out at FINRA_TAF_MAX. this means we have a mismatch when overall_margin_multiplier > 2, since TradePosition would have 2 sub positions
        
        
        # Adjust max_shares to account for entry cost
        available_balance = (balance * overall_margin_multiplier) - estimated_entry_cost
        max_shares = math.floor(available_balance / current_price)
        
        estimated_entry_cost = TradePosition.estimate_entry_cost(max_shares) * overall_margin_multiplier
        
        debug3_print(f'{max_shares}, {available_balance:.4f}, {estimated_entry_cost:.4f}')

        if max_shares < max_shares_before_costs:
            debug3_print('max_shares < max_shares_before_costs',max_shares, max_shares_before_costs)
        else:
            assert max_shares == max_shares_before_costs, (max_shares, max_shares_before_costs)

        # Ensure max_shares is a multiple of 2 when times_buying_power > 2 and is_marginable
        if times_buying_power > 2 and is_marginable and max_shares % 2 == 1:
            debug3_print('max_shares = max_shares - (max_shares % 2)', max_shares, max_shares - (max_shares % 2))
            max_shares -= 1
        
        return max_shares, actual_margin_multiplier, overall_margin_multiplier, estimated_entry_cost

    
    def place_stop_market_buy(area: TouchArea, timestamp: datetime, open_price: float, high_price: float, low_price: float, close_price: float, prev_close: float):
        nonlocal balance, current_id, total_account_value, open_positions, trades_executed

        if open_positions or balance <= 0:
            return NO_POSITION_OPENED

        # debug_print(f"Attempting order: {'Long' if area.is_long else 'Short'} at {area.get_buy_price:.4f}")
        # debug_print(f"  Balance: {balance:.4f}, Total Account Value: {total_account_value:.4f}")

        # Check if the stop buy would have executed based on high/low. close price unnecessary (i think)
        if area.is_long:
            if prev_close > area.get_buy_price:
                # debug_print(f"  Rejected: Previous close ({prev_close:.4f}) above buy price, likey re-entering area ({area.get_buy_price:.4f})")
                return NO_POSITION_OPENED
            if high_price < area.get_buy_price or close_price > high_price:
                # debug_print(f"  Rejected: High price ({high_price:.4f}) didn't reach buy price ({area.get_buy_price:.4f})")
                return NO_POSITION_OPENED
        else:  # short
            if prev_close < area.get_buy_price:
                # debug_print(f"  Rejected: Previous close ({prev_close:.4f}) below buy price, likey re-entering area ({area.get_buy_price:.4f})")
                return NO_POSITION_OPENED
            if low_price > area.get_buy_price or close_price < low_price:
                # debug_print(f"  Rejected: Low price ({low_price:.4f}) didn't reach buy price ({area.get_buy_price:.4f})")
                return NO_POSITION_OPENED

        # Stop buy (placed at time of min_touches) would have executed ############### IMPORTANT
        execution_price = area.get_buy_price
        # execution_price = close_price

        # Calculate position size, etc...
        max_shares, actual_margin_multiplier, overall_margin_multiplier, estimated_entry_cost = calculate_max_position_size(execution_price)
            
        # Recalculate invest_amount based on rounded max_shares
        invest_amount = max_shares * execution_price
        actual_cash_used = invest_amount / overall_margin_multiplier
        
        if actual_cash_used > balance:
            debug3_print(f"  Order rejected: Insufficient balance ({actual_cash_used:.4f} > {balance:.4f})")
            return NO_POSITION_OPENED

        debug_print(f"  Shares: {max_shares}, Invest Amount: {invest_amount:.4f}")
        debug_print(f"  Margin Multiplier: {actual_margin_multiplier:.2f}")
        debug_print(f"  execution_price: {execution_price:.2f}")
        
        # Create the position
        position = TradePosition(
            id=current_id,
            area=area,
            is_long=area.is_long,
            entry_time=timestamp,
            initial_balance=actual_cash_used,
            initial_shares=max_shares,
            entry_price=execution_price,
            use_margin=use_margin,
            is_marginable=is_marginable,
            times_buying_power=overall_margin_multiplier, # important: overall_margin_multiplier == TradePosition.times_buying_power 
            actual_margin_multiplier=actual_margin_multiplier,
            current_stop_price=high_price - area.get_range if area.is_long else low_price + area.get_range,
            max_price=high_price if area.is_long else None,
            min_price=low_price if not area.is_long else None,
        )
        
        debug3_print(f'Balance {balance:.4f}, invest_amount {invest_amount:.4f}, actual_cash_used {actual_cash_used:.4f}')
        cash_needed, fees = position.initial_entry()
        
        debug3_print(f'INITIAL entry fees estimated {estimated_entry_cost:.4f}, actual {fees:.4f}')
        debug3_print(f'  cash needed {cash_needed:.4f}')
        
        
        current_id += 1
        
        # Add to open positions
        open_positions[area.id] = position
        
        debug2_print(f"{'res' if area.is_long else 'sup'} area {area.id}: {current_id} {timestamp} - Enter {'Long ' if area.is_long else 'Short'} at {execution_price:.4f}. "
              f"Shares: {max_shares}, Amount: ${invest_amount:.4f} (Margin: {actual_margin_multiplier:.2f}x, Overall: {overall_margin_multiplier:.2f}x, Sub-positions: {len(position.sub_positions)})")
        # debug2_print(cash_needed + fees, fees)
        rebalance(-cash_needed - fees, close_price)
        return POSITION_OPENED

    def update_positions(timestamp:datetime, open_price, high_price, low_price, close_price):
        #...

        for area_id, position in open_positions.items():
            #...
            if target_shares < position.shares:
                #...
            elif target_shares > position.shares:
                shares_to_adjust = target_shares - position.shares
                debug2_print(f"  Initiating partial entry - Shares to buy: {shares_to_adjust}")
                if shares_to_adjust > 0:
                    debug2_print(f"    Current -> Target percentage: {current_pct*100:.2f}% ({position.shares}) -> {target_pct*100:.2f}% ({target_shares})")
                    
                    # Calculate the maximum number of shares we can buy with the current balance
                    max_shares, _, _, estimated_entry_cost = calculate_max_position_size(price_at_action)

                    # Adjust the number of shares to buy if necessary
                    shares_to_buy = min(shares_to_adjust, max_shares)
                    
                    if shares_to_buy > 0:
                        cash_needed, fees = position.partial_entry(timestamp, price_at_action, shares_to_buy)
                        rebalance(-cash_needed - fees, price_at_action)
                        debug2_print(f"    Partial enter: Bought {shares_to_buy} shares at {price_at_action:.4f}")
                        
                        if shares_to_buy < shares_to_adjust:
                            debug2_print(f"    WARNING: Could only buy {shares_to_buy} out of {shares_to_adjust} desired shares due to insufficient balance.")
                            warning_count_insuf_some += 1
                    else:
                        debug2_print(f"    WARNING: Insufficient balance to buy any shares. Desired: {shares_to_adjust}, Max possible: {max_shares}")
                        warning_count_insuf_none += 1

# ...


# in TradePosition.py:

@dataclass
class Transaction:
    timestamp: datetime
    shares: int
    price: float
    is_entry: bool
    transaction_cost: float
    value: float  # This will be the cost (negative) or revenue (positive)
    realized_pnl: Optional[float] = None 

@dataclass
class SubPosition:
    entry_time: datetime
    entry_price: float
    shares: int
    cash_committed: float
    market_value: float = field(init=False)
    unrealized_pnl: float = field(init=False)
    realized_pnl: float = 0
    transactions: List[Transaction] = field(default_factory=list)
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None

    def __post_init__(self):
        self.update_market_value(self.entry_price)

    def update_market_value(self, current_price: float):
        self.market_value = self.shares * current_price
        self.unrealized_pnl = self.market_value - (self.shares * self.entry_price)

    def add_transaction(self, transaction: Transaction):
        self.transactions.append(transaction)

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
    entry_price: float
    market_value: float = 0
    shares: int = 0
    partial_entry_count: int = 0
    partial_exit_count: int = 0
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    sub_positions: List[SubPosition] = field(default_factory=list)
    transactions: List[Transaction] = field(default_factory=list)
    current_stop_price: Optional[float] = None
    max_price: Optional[float] = None
    min_price: Optional[float] = None
    last_price: float = field(default=0.0)
    cash_committed: float = field(init=False)
    unrealized_pnl: float = field(default=0.0)
    realized_pnl: float = 0
    stock_borrow_rate: float = 0.03
    
    SEC_FEE_RATE = 0.000008  # $8 per $1,000,000
    FINRA_TAF_RATE = 0.000166  # $166 per 1,000,000 shares
    FINRA_TAF_MAX = 8.30  # Maximum $8.30 per trade
    
#...
    def calculate_num_sub_positions(self, total_shares: int) -> int:
        if self.times_buying_power <= 2:
            return 1
        else:
            return 2  # We'll always use 2 sub-positions when times_buying_power > 2

    def calculate_shares_per_sub(self, total_shares: int, num_subs: int) -> List[int]:
        if self.times_buying_power <= 2:
            assert num_subs == 1
            return [total_shares]
        else:
            assert num_subs == 2
            assert total_shares % 2 == 0
            # Always split evenly between two sub-positions
            half_shares = total_shares // 2
            return [half_shares, total_shares - half_shares]


    def partial_entry(self, entry_time: datetime, entry_price: float, shares_to_buy: int):
        # Ensure we maintain an even number of shares when times_buying_power > 2
        if self.times_buying_power > 2 and (self.shares + shares_to_buy) % 2 != 0:
            debug_print(f"WARNING: shares_to_buy adjusted to ensure even shares")
            shares_to_buy -= 1
            
        debug_print(f"DEBUG: Entering partial_entry - Time: {entry_time}, Price: {entry_price:.4f}, Shares to buy: {shares_to_buy}")
        debug_print(f"DEBUG: Current position - Shares: {self.shares}, Cash committed: {self.cash_committed:.2f}")

        new_total_shares = self.shares + shares_to_buy
        new_num_subs = self.calculate_num_sub_positions(new_total_shares)

        additional_cash_committed = (shares_to_buy * entry_price) / self.times_buying_power

        active_sub_positions = [sp for sp in self.sub_positions if sp.shares > 0]
        target_shares = self.calculate_shares_per_sub(new_total_shares, new_num_subs)

        fees = 0
        shares_added = 0

        debug_print(f"DEBUG: Target shares per sub-position: {target_shares}")

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
                    # sp.update_market_value(entry_price)
                    fees += self.add_transaction(entry_time, shares_to_add, entry_price, is_entry=True, sub_position=sp)
                    shares_added += shares_to_add
                    debug_print(f"DEBUG: Adding to sub-position {i} - Entry price: {sp.entry_price:.4f}, Shares added: {shares_to_add}, "
                        f"Cash committed: {sub_cash_committed:.2f}, Old shares: {old_shares}, New shares: {sp.shares}")
            else:
                # New sub-position
                sub_cash_committed = (target * entry_price) / self.times_buying_power
                new_sub = SubPosition(entry_time, entry_price, target, sub_cash_committed)
                self.sub_positions.append(new_sub)
                fees += self.add_transaction(entry_time, target, entry_price, is_entry=True, sub_position=new_sub)
                shares_added += target
                debug_print(f"DEBUG: Created new sub-position {i} - Entry price: {entry_price:.4f}, Shares: {target}, "
                    f"Cash committed: {sub_cash_committed:.2f}")

        self.shares += shares_added
        self.cash_committed += additional_cash_committed
        self.update_market_value(entry_price)
        self.partial_entry_count += 1

        assert abs(self.cash_committed - sum(sp.cash_committed for sp in self.sub_positions if sp.shares > 0)) < 1e-8, \
            f"Cash committed mismatch: {self.cash_committed:.2f} != {sum(sp.cash_committed for sp in self.sub_positions if sp.shares > 0):.2f}"
        assert self.shares == sum(sp.shares for sp in self.sub_positions if sp.shares > 0), \
            f"Shares mismatch: {self.shares} != {sum(sp.shares for sp in self.sub_positions if sp.shares > 0)}"

        debug_print(f"DEBUG: Partial entry complete - Shares added: {shares_added}, New total shares: {self.shares}, "
            f"New cash committed: {self.cash_committed:.2f}")
        debug_print("DEBUG: Current sub-positions:")
        for i, sp in enumerate(self.sub_positions):
            if sp.shares > 0:
                debug_print(f"  Sub-position {i}: Shares: {sp.shares}, Entry price: {sp.entry_price:.4f}")

        return additional_cash_committed, fees

    @staticmethod
    def estimate_entry_cost(shares: int):
        finra_taf = min(TradePosition.FINRA_TAF_RATE * shares, TradePosition.FINRA_TAF_MAX)
        return finra_taf
    
    def calculate_transaction_cost(self, shares: int, price: float, is_entry: bool, timestamp: datetime, sub_position: SubPosition) -> float:
        finra_taf = min(self.FINRA_TAF_RATE * shares, self.FINRA_TAF_MAX)
        sec_fee = 0
        if not is_entry:  # SEC fee only applies to exits
            trade_value = price * shares
            sec_fee = self.SEC_FEE_RATE * trade_value
        
        stock_borrow_cost = 0
        if not self.is_long and not is_entry:  # Stock borrow cost applies only to short position exits
            daily_borrow_rate = self.stock_borrow_rate / 360
            # Calculate holding time from transactions
            total_days_held = 0
            
            # Walk backwards to find the earliest share included in current sub_position.shares
            cumulative_shares = 0
            for transaction in reversed(sub_position.transactions):
                if transaction.is_entry:
                    cumulative_shares += transaction.shares
                if cumulative_shares >= sub_position.shares:
                    earliest_relevant_timestamp = transaction.timestamp
                    break
            
            # Walk forwards to calculate the cost for the shares being removed
            shares_to_remove = shares
            for transaction in sub_position.transactions:
                if transaction.is_entry and transaction.timestamp >= earliest_relevant_timestamp:
                    holding_time = timestamp - transaction.timestamp
                    days_held = holding_time.total_seconds() / (24 * 60 * 60)
                    shares_to_calculate = min(shares_to_remove, transaction.shares)
                    total_days_held += shares_to_calculate * days_held
                    shares_to_remove -= shares_to_calculate
                    
                    if shares_to_remove <= 0:
                        break
            
            stock_borrow_cost = shares * price * daily_borrow_rate * total_days_held
            
        return finra_taf + sec_fee + stock_borrow_cost



    def add_transaction(self, timestamp: datetime, shares: int, price: float, is_entry: bool, sub_position: SubPosition, sp_realized_pnl: Optional[float] = None):
        transaction_cost = self.calculate_transaction_cost(shares, price, is_entry, timestamp, sub_position)
        value = -shares * price if is_entry else shares * price
        
        transaction = Transaction(timestamp, shares, price, is_entry, transaction_cost, value, sp_realized_pnl)
        self.transactions.append(transaction)
        sub_position.add_transaction(transaction)
        
        if not is_entry:
            if sp_realized_pnl is None:
                raise ValueError("sp_realized_pnl must be provided for exit transactions")
            self.realized_pnl += sp_realized_pnl
            
        debug_print(f"DEBUG: Transaction added - {'Entry' if is_entry else 'Exit'}, Shares: {shares}, Price: {price:.4f}, "
            f"Value: {value:.2f}, Cost: {transaction_cost:.4f}, Realized PnL: {sp_realized_pnl if sp_realized_pnl is not None else 'N/A'}")


        return transaction_cost
