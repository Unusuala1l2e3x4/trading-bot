from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
from alpaca.trading.models import Order
from alpaca.trading.enums import OrderSide

import pandas as pd
import os
from pathlib import Path


@dataclass
class SlippageData:
    """Records trade execution data for slippage analysis.
    
    This class captures intended trade details, actual execution details, and market 
    conditions to analyze execution quality and calibrate slippage models.
    """
    # Order identifiers
    order_id: str
    client_order_id: str
    symbol: str
    
    # Intended trade details
    intended_qty: float
    is_entry: bool  # True for entries, False for exits
    is_long: bool   # True for long positions, False for shorts
    
    # Latest quote before submission
    quote_bid_price: float
    quote_ask_price: float
    quote_timestamp: datetime
    
    # Actual order details
    filled_qty: float
    filled_avg_price: float
    submitted_timestamp: datetime  # When we attempted to submit
    created_at: datetime      # When Alpaca created the order
    submitted_at: datetime    # When Alpaca submitted to exchange
    updated_at: datetime      # Last update timestamp
    filled_at: Optional[datetime]  # When order was completely filled
    
    # Market conditions
    rolling_atr: float  # ATR at time of trade
    avg_volume: float   # Volume EMA at time of trade
    minute_volume: float  # Volume in the last minute
    
    @classmethod
    def from_order(cls, order: Order, intended_qty: float, is_entry: bool, is_long: bool,
                  quote_bid_price: float, quote_ask_price: float, quote_timestamp: datetime,
                  submitted_timestamp: datetime, rolling_atr: float, avg_volume: float,
                  minute_volume: float) -> 'SlippageData':
        """Creates SlippageData instance from an Alpaca Order object and trade details.
        
        Args:
            order: The Alpaca Order object after fill
            intended_qty: Original quantity we attempted to trade
            is_entry: Whether this was an entry (True) or exit (False)
            is_long: Whether this was for a long (True) or short (False) position
            quote_bid_price: Latest bid price before submission
            quote_ask_price: Latest ask price before submission
            quote_timestamp: When the quotes were recorded
            submitted_timestamp: When we attempted to submit the order
            rolling_atr: ATR value at time of trade
            avg_volume: Volume EMA at time of trade
            minute_volume: Volume in the last minute
        """
        return cls(
            order_id=str(order.id),
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            intended_qty=intended_qty,
            is_entry=is_entry,
            is_long=is_long,
            quote_bid_price=quote_bid_price,
            quote_ask_price=quote_ask_price,
            quote_timestamp=quote_timestamp,
            filled_qty=float(order.filled_qty or 0),
            filled_avg_price=float(order.filled_avg_price or 0),
            submitted_timestamp=submitted_timestamp,
            created_at=order.created_at,
            submitted_at=order.submitted_at,
            updated_at=order.updated_at,
            filled_at=order.filled_at,
            rolling_atr=rolling_atr,
            avg_volume=avg_volume,
            minute_volume=minute_volume
        )
    
    @property
    def expected_price(self) -> float:
        """Expected execution price based on quote and trade direction."""
        if (self.is_long and self.is_entry) or (not self.is_long and not self.is_entry):
            return self.quote_ask_price  # Buying at ask
        else:
            return self.quote_bid_price  # Selling at bid
    
    @property
    def actual_slippage(self) -> float:
        """Calculate actual price slippage experienced (can be negative)."""
        if self.filled_qty == 0 or self.expected_price == 0:
            return 0.0
            
        # For shorts or exits, flip the sign
        direction = 1 if (
            (self.is_long and self.is_entry) or 
            (not self.is_long and not self.is_entry)
        ) else -1
        
        return (self.filled_avg_price - self.expected_price) * direction
    
    @property
    def fill_ratio(self) -> float:
        """Ratio of filled quantity to intended quantity."""
        return self.filled_qty / self.intended_qty if self.intended_qty != 0 else 0.0
    
    @property
    def latency_to_create(self) -> timedelta:
        """Time between our submission and Alpaca creating the order."""
        return self.created_at - self.submitted_timestamp
    
    @property
    def latency_to_submit(self) -> timedelta:
        """Time between Alpaca creating and submitting the order."""
        return self.submitted_at - self.created_at
    
    @property
    def time_to_fill(self) -> Optional[timedelta]:
        """Total time from our submission to complete fill."""
        if self.filled_at is None:
            return None
        return self.filled_at - self.submitted_timestamp
    
    @property
    def quote_age(self) -> timedelta:
        """How old the quote was when we submitted."""
        return self.submitted_timestamp - self.quote_timestamp
    
    def calculate_cost_impact(self) -> float:
        """Calculate total cost impact of slippage in dollars."""
        return self.actual_slippage * self.filled_qty
    
    def to_dict(self) -> dict:
        """Convert to dictionary for analysis/storage."""
        base_dict = {
            'order_id': self.order_id,
            'client_order_id': self.client_order_id,
            'symbol': self.symbol,
            'intended_qty': self.intended_qty,
            'is_entry': self.is_entry,
            'is_long': self.is_long,
            'quote_bid_price': self.quote_bid_price,
            'quote_ask_price': self.quote_ask_price,
            'quote_timestamp': self.quote_timestamp.isoformat(),
            'filled_qty': self.filled_qty,
            'filled_avg_price': self.filled_avg_price,
            'submitted_timestamp': self.submitted_timestamp.isoformat(),
            'created_at': self.created_at.isoformat(),
            'submitted_at': self.submitted_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'filled_at': self.filled_at.isoformat() if self.filled_at else None,
            'rolling_atr': self.rolling_atr,
            'avg_volume': self.avg_volume,
            'minute_volume': self.minute_volume,
            'actual_slippage': self.actual_slippage,
            'fill_ratio': self.fill_ratio,
            'latency_to_create_ms': self.latency_to_create.total_seconds() * 1000,
            'latency_to_submit_ms': self.latency_to_submit.total_seconds() * 1000,
            'time_to_fill_ms': self.time_to_fill.total_seconds() * 1000 if self.time_to_fill else None,
            'quote_age_ms': self.quote_age.total_seconds() * 1000,
            'cost_impact': self.calculate_cost_impact()
        }
        return base_dict
        
    

    def append_to_csv(self, filepath: str) -> None:
        """Appends the SlippageData to a CSV file. Creates file with headers if it doesn't exist.
        
        Args:
            filepath: Path to the CSV file
        """
        # Convert to single-row DataFrame
        df = pd.DataFrame([self.to_dict()])
        
        # Create directory if it doesn't exist
        Path(os.path.dirname(filepath)).mkdir(parents=True, exist_ok=True)
        
        # If file doesn't exist, write with header. Otherwise, append without header
        if not os.path.exists(filepath):
            df.to_csv(filepath, index=False)
        else:
            df.to_csv(filepath, mode='a', header=False, index=False)