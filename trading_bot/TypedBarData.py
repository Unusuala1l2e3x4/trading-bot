from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import pandas as pd
import numpy as np

@dataclass
class TypedBarData:
    """Wrapper for bar data with proper type hints."""
    # Index fields
    symbol: str
    timestamp: datetime
    
    # Bar data fields
    open: float
    high: float
    low: float
    close: float
    volume: float
    trade_count: float
    vwap: float
    MACD: float
    MACD_signal: float
    MACD_hist: float
    MACD_hist_roc: float
    RSI: float
    RSI_roc: float
    MFI: float
    MFI_roc: float
    central_value: float
    is_res: bool
    # shares_per_trade: float
    avg_volume: float # (EMA)
    avg_trade_count: float # (EMA)
    log_return: float
    volatility: float # (rolling)
    
    # New bar data fields
    H_L: float  # High-Low range
    # H_PC: float  # High-Previous Close
    # L_PC: float  # Low-Previous Close
    # TR: float  # True Range
    ATR: float  # Average True Range (EMA)
    MTR: float  # Median True Range
    rolling_range_min_4: float  # 4-bar minimum range
    rolling_range_min_7: float  # 7-bar minimum range
    rolling_ATR: float  # rolling ATR average
    
    # Store original row for access to any additional columns
    _row: pd.Series = field(repr=False)
    
    # Add threshold parameters as class attributes with defaults
    doji_ratio_threshold: float = 0.1
    wick_ratio_threshold: float = 2.0
    atr_compression_threshold: float = 0.9
    volume_compression_threshold: float = 0.7
    min_indecision_signals: int = 2
    

    
    @property
    def shares_per_trade(self) -> float:
        return self.volume / self.trade_count
    
    @property
    def body_size(self) -> float:
        """Calculate the absolute size of the candle's body."""
        return abs(self.close - self.open)
    
    @property
    def range_size(self) -> float:
        """Calculate the total range of the candle (high minus low)."""
        return self.high - self.low
    
    @property
    def doji_ratio(self) -> float:
        """Calculate the ratio of the body size to the total range."""
        if self.range_size == 0:
            return 0
        return (self.close - self.open) / self.range_size
    
    @property
    def doji_ratio_abs(self) -> float:
        """Calculate the ratio of the body size to the total range."""
        return abs(self.doji_ratio)
    
    @property
    def wick_ratio(self) -> float:
        """Calculate the ratio of the total range to the body size."""
        if self.body_size == 0:
            return float('inf')
        return self.range_size / self.body_size

    @property
    def is_doji(self) -> bool:
        """
        Determine if the bar is a doji candle.

        For 1-minute bars, a threshold of 0.1 may be more appropriate to capture more doji patterns,
        as small price movements can be significant.

        :param threshold: The maximum ratio of body size to range size to consider as a doji.
        :return: True if the bar is a doji, False otherwise.
        """
        return self.doji_ratio_abs < self.doji_ratio_threshold
    
    @property
    def has_long_wicks(self) -> bool:
        """
        Determine if the bar has long wicks relative to its body.

        For 1-minute bars, a lower threshold like 1.5-2.0 may be more suitable,
        as wicks can be proportionally larger on shorter timeframes.

        :param threshold: The minimum ratio of range size to body size to consider as having long wicks.
        :return: True if the bar has long wicks, False otherwise.
        """
        return self.wick_ratio > self.wick_ratio_threshold
    
    
    @property
    def nr4_hl_diff(self) -> bool:
        return self.H_L - self.rolling_range_min_4
    
    @property
    def nr7_hl_diff(self) -> bool:
        return self.H_L - self.rolling_range_min_7
    
    @property
    def is_nr4(self) -> bool:
        """Check if the current bar is a Narrow Range 4 (NR4) bar."""
        return self.nr4_hl_diff <= 0
    
    @property
    def is_nr7(self) -> bool:
        """Check if the current bar is a Narrow Range 7 (NR7) bar."""
        return self.nr7_hl_diff <= 0
    
    @property
    def ATR_ratio(self) -> float:
        """ATR relative to rolling avg."""
        if self.rolling_ATR == 0:
            return 0
        return self.ATR / self.rolling_ATR
    
    @property
    def is_atr_compressed(self) -> bool:
        """
        Check if the ATR is compressed relative to the recent average ATR.

        For 1-minute bars, adjusting the threshold to 0.7-0.8 may better capture periods of reduced volatility.

        :param threshold: The maximum ratio of current ATR to rolling ATR to consider as compressed.
        :return: True if the ATR is compressed, False otherwise.
        """
        return self.ATR_ratio < self.atr_compression_threshold

    @property
    def volume_ratio(self) -> float:
        """Volume relative to recent average."""
        if self.avg_volume == 0:
            return 0
        return self.volume / self.avg_volume
    
    @property
    def is_volume_compressed(self) -> bool:
        """Check if volume shows indecision (lower than average)."""
        return self.volume_ratio < self.volume_compression_threshold

    def get_indecision_signals(self): # -> List[Tuple[str, bool]]:
        """
        Get list of tuples containing (description, boolean) for each indecision signal.
        """
        return [
            (f"Doji (ratio={self.doji_ratio:.3f} < {self.doji_ratio_threshold})", 
             self.is_doji),
            
            (f"Long Wicks (ratio={self.wick_ratio:.2f} > {self.wick_ratio_threshold})", 
             self.has_long_wicks),
            
            (f"NR4 (range={self.H_L:.3f} <= {self.rolling_range_min_4:.3f})", 
             self.is_nr4),
            
            (f"NR7 (range={self.H_L:.3f} <= {self.rolling_range_min_7:.3f})", 
             self.is_nr7),
            
            (f"ATR Compressed (ratio={self.ATR_ratio:.2f} < {self.atr_compression_threshold})", 
             self.is_atr_compressed),
            
            (f"Volume Compressed (ratio={self.volume_ratio:.2f} < {self.volume_compression_threshold})", 
             self.is_volume_compressed)
        ]

    @property
    def indecision_count(self) -> int:
        """Count how many indecision indicators are present."""
        return sum(signal[1] for signal in self.get_indecision_signals())
    
    @property
    def describe_indecision(self) -> str:
        """Get a readable description of all indecision signals."""
        signals = self.get_indecision_signals()
        active_signals = [desc for desc, is_active in signals if is_active]
        return f"{len(active_signals)} signals active:\n" + "\n".join(
            f"  - {desc}" for desc in active_signals
        )

    @property
    def shows_indecision(self) -> bool:
        """
        Check if enough indecision signals are present.

        For 1-minute bars, reducing the minimum required signals to 2-3 may capture more opportunities.

        :param min_signals: The minimum number of indecision indicators to consider as indecision.
        :return: True if enough indecision signals are present, False otherwise.
        """
        return self.indecision_count >= self.min_indecision_signals
    
    @property
    def has_divergence(self) -> bool:
        """Check if price and indicators are showing divergence."""
        price_trending_up = self.close > self.open
        rsi_trending_down = self.RSI_roc < 0
        mfi_trending_down = self.MFI_roc < 0
        
        price_trending_down = self.close < self.open
        rsi_trending_up = self.RSI_roc > 0
        mfi_trending_up = self.MFI_roc > 0
        
        return (price_trending_up and rsi_trending_down and mfi_trending_down) or \
            (price_trending_down and rsi_trending_up and mfi_trending_up)

    def shows_reversal_potential(self, area_is_long, rsi_overbought: float = 70, rsi_oversold: float = 30, mfi_overbought: float = 80, mfi_oversold: float = 20) -> bool:
        """
        Check if the bar shows potential for reversal based on multiple factors.
        
        Args:
            rsi_overbought: RSI threshold for overbought condition (default: 70)
            rsi_oversold: RSI threshold for oversold condition (default: 30)
            mfi_overbought: MFI threshold for overbought condition (default: 80)
            mfi_oversold: MFI threshold for oversold condition (default: 20)
        
        Returns:
            bool: True if either:
                - Oversold conditions with bullish price action, or
                - Overbought conditions with bearish price action
                AND either shows indecision or has indicator divergence
        """
        # Check if oversold (looking for bullish reversal)
        oversold_reversal = (
            not area_is_long
            and
            self.MFI <= mfi_oversold
            # and  # Primary indicator
            # self.RSI <= rsi_oversold
            # and   # Confirmation
            # self.close >= self.open          # Current bar showing strength
        )
        
        # Check if overbought (looking for bearish reversal)
        overbought_reversal = (
            area_is_long
            and
            self.MFI >= mfi_overbought
            # and  # Primary indicator
            # self.RSI >= rsi_overbought
            # and  # Confirmation
            # self.close <= self.open                 # Current bar showing weakness
        )

        # Base conditions for reversal
        base_reversal = (oversold_reversal or overbought_reversal) # mututally exclusive so never both true
        
        
        if base_reversal:
            if oversold_reversal:
                assert not overbought_reversal
                # assert not area_is_long, (area_is_long, self.RSI, self.MFI)
            if overbought_reversal:
                assert not oversold_reversal
                # assert area_is_long, (area_is_long, self.RSI, self.MFI)
                
        # Enhanced reversal signal requires:
        # 1. Base reversal conditions
        # 2. Enough indecision signals
        # 3. Either divergence or high indecision
        return base_reversal and (
            self.shows_indecision or 
            self.has_divergence
        )
        
    def describe_reversal_potential(self, area_was_long, rsi_overbought: float = 70, rsi_oversold: float = 30, 
                                mfi_overbought: float = 80, mfi_oversold: float = 20) -> str:
        """Get a readable description of reversal potential conditions."""
        conditions = []
        
        # # Check oversold conditions
        if self.MFI <= mfi_oversold:
            # assert not area_was_long
            conditions.append(f"MFI oversold ({self.MFI:.1f} <= {mfi_oversold})")
        if not area_was_long and self.RSI <= rsi_oversold:
            # assert not area_was_long
            conditions.append(f"RSI oversold ({self.RSI:.1f} <= {rsi_oversold})")
        
        # # Check overbought conditions
        if self.MFI >= mfi_overbought:
            # assert area_was_long
            conditions.append(f"MFI overbought ({self.MFI:.1f} >= {mfi_overbought})")
        if area_was_long and self.RSI >= rsi_overbought:
            # assert area_was_long
            conditions.append(f"RSI overbought ({self.RSI:.1f} >= {rsi_overbought})")
        
        # # Check price action
        # if self.close >= self.open:
        #     conditions.append(f"Bullish price action (O:{self.open:.2f} -> C:{self.close:.2f})")
        # elif self.close <= self.open:
        #     conditions.append(f"Bearish price action (O:{self.open:.2f} -> C:{self.close:.2f})")
        
        # Check divergence
        if self.has_divergence:
            conditions.append(f"Divergence detected (Price: {'up' if self.close > self.open else 'down'}, " 
                            f"RSI: {'down' if self.RSI_roc < 0 else 'up'}, "
                            f"MFI: {'down' if self.MFI_roc < 0 else 'up'})"
                            )
        
        # Check indecision
        if self.shows_indecision:
            # conditions.append(f"Shows indecision ({self.indecision_count} signals)")
            conditions.append(f"Shows indecision - {self.describe_indecision}")
        
        if not conditions:
            return "No reversal conditions detected"
        
        # Determine overall reversal potential
        has_potential = self.shows_reversal_potential(area_was_long, rsi_overbought, rsi_oversold, 
                                                    mfi_overbought, mfi_oversold)
        
        return (f"{'POTENTIAL REVERSAL' if has_potential else 'Partial conditions'} detected:\n" + 
                "\n".join(f"- {cond}" for cond in conditions))
        
    
    def update_thresholds(self,
                         doji_ratio_threshold: Optional[float] = None,
                         wick_ratio_threshold: Optional[float] = None,
                         atr_compression_threshold: Optional[float] = None,
                         volume_compression_threshold: Optional[float] = None,
                         min_indecision_signals: Optional[int] = None):
        """Update indecision thresholds."""
        if doji_ratio_threshold is not None:
            self.doji_ratio_threshold = doji_ratio_threshold
        if wick_ratio_threshold is not None:
            self.wick_ratio_threshold = wick_ratio_threshold
        if atr_compression_threshold is not None:
            self.atr_compression_threshold = atr_compression_threshold
        if volume_compression_threshold is not None:
            self.volume_compression_threshold = volume_compression_threshold
        if min_indecision_signals is not None:
            self.min_indecision_signals = min_indecision_signals
            
    @classmethod
    def from_row(cls, row: pd.Series) -> 'TypedBarData':
        """Create TypedBarData instance from DataFrame row."""
        # Extract symbol and timestamp from index
        symbol = row.name[0] if isinstance(row.name, tuple) else None
        timestamp = row.name[1] if isinstance(row.name, tuple) else row.name
        
        # Create kwargs dict with all available fields
        kwargs = {
            'symbol': symbol,
            'timestamp': timestamp,
            '_row': row
        }
        
        # Add all known fields, handling missing or NaN values
        known_fields = {k: v for k, v in cls.__annotations__.items() 
                    if not k.endswith('_threshold') and k != 'min_indecision_signals'}
        
        for field in known_fields:
            if field in ['symbol', 'timestamp', '_row']:
                continue
            
            value = row.get(field)
            kwargs[field] = None if pd.isna(value) else value

        return cls(**kwargs)
    
    def to_dict(self):
        """Convert bar data to dictionary."""
        return {
            field: getattr(self, field) 
            for field in self.__annotations__ 
            if not field.startswith('_')
        }

    def __str__(self):
        """String representation showing all attributes."""
        fields = [
            f"{field}: {getattr(self, field)}" 
            for field in self.__annotations__ 
            if not field.startswith('_')
        ]
        return f"TypedBarData({', '.join(fields)})"
    
    @staticmethod
    def get_field_names() -> list[str]:
        """Get list of field names from class annotations, excluding _row."""
        return [field for field in TypedBarData.__annotations__ if field != '_row']

    @staticmethod
    def to_dataframe(bars: list['TypedBarData']) -> pd.DataFrame:
        """Convert a list of TypedBarData objects to a DataFrame.
        
        Args:
            bars: List of TypedBarData objects
            
        Returns:
            pd.DataFrame: DataFrame where index attributes become columns
        """
        # Create list of dictionaries from each bar's Series
        data = []
        for bar in bars:
            row_dict = bar._row.to_dict()
            # Add index values if they exist (for multi-index)
            if isinstance(bar._row.name, tuple):
                row_dict['symbol'] = bar._row.name[0]
                row_dict['timestamp'] = bar._row.name[1]
            else:
                row_dict['timestamp'] = bar._row.name
            data.append(row_dict)
        
        # Create DataFrame from list of dictionaries
        df = pd.DataFrame(data)
        
        # # Set the index if needed
        # if 'symbol' in df.columns:
        #     df.set_index(['symbol', 'timestamp'], inplace=True)
        # else:
        #     df.set_index('timestamp', inplace=True)
            
        return df
