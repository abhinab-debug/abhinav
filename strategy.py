import pandas as pd
import numpy as np

def calculate_avwap(df: pd.DataFrame, anchor_index: int = 0) -> pd.Series:
    """Calculates Anchored Volume Weighted Average Price (AVWAP)."""
    if df.empty:
        return pd.Series()

    df_slice = df.iloc[anchor_index:].copy()
    avwap = (df_slice['Rate'] * df_slice['Quantity']).cumsum() / df_slice['Quantity'].cumsum()
    return avwap

def get_smc_zones(close_series: pd.Series, avwap: float):
    """
    Defines Tiers based on price relative to Swing and AVWAP.
    TIER 1: APEX ZONE (> 75% of Swing)
    TIER 2: PREMIUM ZONE (AVWAP to 75%)
    TIER 3: DISCOUNT ZONE (Base to AVWAP)
    """
    if close_series.empty:
        return "UNKNOWN"

    swing_high = close_series.max()
    swing_low = close_series.min()
    current_price = close_series.iloc[-1]

    apex_threshold = swing_low + 0.75 * (swing_high - swing_low)

    if current_price > apex_threshold:
        return "TIER 1: APEX ZONE"
    elif current_price > avwap:
        return "TIER 2: PREMIUM ZONE"
    else:
        return "TIER 3: DISCOUNT ZONE"

def check_execution_triggers(
    htf_trend: str,
    ltf_trend: str,
    idm_sweep: bool,
    ob_fvg_mitigated: bool,
    choch_bos: bool,
    wnp: float,
    kyles_lambda: float
) -> bool:
    """
    SMC Execution Triggers (The Buyer Rules).
    1. HTF-LTF Coherence
    2. Inducement (IDM) Sweep
    3. OB/FVG Mitigation
    4. CHoCH/BOS Confirmation (backed by microstructure)
    """
    # Basic logic gate
    if not (htf_trend == ltf_trend):
        return False
    if not idm_sweep:
        return False
    if not ob_fvg_mitigated:
        return False
    if not choch_bos:
        return False

    # Backed by 14 pillars (e.g., high WNP, low Kyle's Lambda)
    # Thresholds should be adaptive, but for now we use placeholders
    if wnp > 0 and kyles_lambda < 0.1:
        return True

    return False

def calculate_oli(overnight_up_vol: float, overnight_down_vol: float, median_opening_vol: float) -> float:
    """24-Hour Off-Hour Order Imbalance (OLI)."""
    if median_opening_vol == 0:
        return 0.0
    return (overnight_up_vol - overnight_down_vol) / median_opening_vol

def post_halt_momentum_rebound(first_100_vols: np.ndarray, first_100_sides: np.ndarray) -> bool:
    """Exploits index circuit breaker rules."""
    if len(first_100_vols) < 100:
        return False
    # If large blocks match with zero latency
    whale_threshold = np.percentile(first_100_vols, 90)
    whale_buys = np.sum(first_100_vols[(first_100_vols >= whale_threshold) & (first_100_sides == 1)])
    if whale_buys > np.sum(first_100_vols) * 0.5:
        return True
    return False

def avellaneda_kelly_size(price: float, volatility: float, inventory: int, risk_aversion: float = 0.1) -> float:
    """Avellaneda-Stoikov Inventory Model / Kelly size parameter proxy."""
    # Simplified version for position sizing
    # Reduce size as inventory increases or volatility increases
    base_size = 1.0
    inventory_penalty = np.exp(-risk_aversion * inventory)
    vol_penalty = 1.0 / (1.0 + volatility)
    return base_size * inventory_penalty * vol_penalty
