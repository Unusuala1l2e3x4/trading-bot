from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from datetime import datetime
from alpaca.trading.models import Order
from alpaca.trading.enums import OrderStatus, TradeEvent
import asyncio
from TradingStrategy import IntendedOrder


@dataclass
class OrderFill:
    """Represents fill details for reconciliation"""
    qty: float
    avg_price: float
    transaction_time: datetime
    event: TradeEvent
    intended_order: IntendedOrder

@dataclass
class OrderState:
    """Tracks the state of an individual order"""
    order_id: str
    client_order_id: str
    intended_order: IntendedOrder
    pre_submit_timestamp: datetime  # For SlippageData
    quote_timestamp: datetime  # For SlippageData
    quote_bid_price: float  # For SlippageData
    quote_ask_price: float  # For SlippageData
    filled_qty: float = 0
    filled_avg_price: float = 0
    status: OrderStatus = OrderStatus.NEW
    filled_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    canceled_at: Optional[datetime] = None
    is_complete: bool = False
    fills: List[OrderFill] = field(default_factory=list)

    def add_fill(self, fill: OrderFill) -> None:
        self.fills.append(fill)
        self.filled_qty = fill.qty  # Updated total filled quantity
        self.filled_avg_price = fill.avg_price  # Updated average price

class OrderTracker:
    """Tracks order state and fill information"""
    def __init__(self):
        self.orders: Dict[str, OrderState] = {}  # order_id -> OrderState
        self.client_orders: Dict[str, str] = {}  # client_order_id -> order_id
        self._lock = asyncio.Lock()
        self.pending_fills: Set[str] = set()  # order_ids with unreconciled fills

    def add_order(self, order: Order, intended_order: IntendedOrder, 
                 pre_submit_timestamp: datetime, 
                 quote_timestamp: datetime,
                 quote_bid_price: float,
                 quote_ask_price: float) -> None:
        """Register a new order with timestamps for slippage analysis"""
        order_state = OrderState(
            order_id=str(order.id),
            client_order_id=order.client_order_id,
            intended_order=intended_order,
            pre_submit_timestamp=pre_submit_timestamp,
            quote_timestamp=quote_timestamp,
            quote_bid_price=quote_bid_price,
            quote_ask_price=quote_ask_price
        )
        
        self.orders[str(order.id)] = order_state
        self.client_orders[order.client_order_id] = str(order.id)

    async def process_trade_update(self, event: TradeEvent, order: Order, transaction_time: datetime) -> None:
        """Process a trade update event"""
        async with self._lock:
            order_id = str(order.id)
            if order_id not in self.orders:
                return  # Ignore updates for unknown orders
                
            order_state = self.orders[order_id]
            
            if event in (TradeEvent.FILL, TradeEvent.PARTIAL_FILL):
                fill = OrderFill(
                    qty=float(order.filled_qty),
                    avg_price=float(order.filled_avg_price),
                    transaction_time=transaction_time,
                    event=event,
                    intended_order=order_state.intended_order
                )
                
                order_state.add_fill(fill)
                
                if event == TradeEvent.FILL:
                    order_state.is_complete = True
                    order_state.filled_at = order.filled_at
                    
                self.pending_fills.add(order_id)
                    
            elif event == TradeEvent.CANCELED:
                order_state.is_complete = True
                order_state.canceled_at = order.canceled_at
                
            elif event == TradeEvent.NEW:
                pass  # Order accepted by exchange
                
            # Update overall status
            order_state.status = order.status

    async def get_fills_to_reconcile(self) -> List[Tuple[str, OrderFill]]:
        """Get list of fills ready for reconciliation"""
        async with self._lock:
            fills_to_reconcile = []
            
            for order_id in list(self.pending_fills):
                order_state = self.orders[order_id]
                
                # Add all unreconciled fills
                for fill in order_state.fills:
                    fills_to_reconcile.append((order_id, fill))
                    
            return fills_to_reconcile

    async def mark_fills_reconciled(self, order_ids: Set[str]) -> None:
        """Mark fills as reconciled after strategy update"""
        async with self._lock:
            for order_id in order_ids:
                self.pending_fills.discard(order_id)
                
    def get_order_state(self, order_id: str) -> Optional[OrderState]:
        """Get current state for an order"""
        return self.orders.get(str(order_id))