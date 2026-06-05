import pandas as pd
import numpy as np

def calculate_avwap(df: pd.DataFrame, anchor_id: int) -> pd.Series:
    """
    True Volume-Weighted Anchored VWAP (AVWAP) from structural anchor event (anchor_id).
    AVWAP = sum(P * V) / sum(V)
    """
    if df.empty:
        return pd.Series()

    # Filter from anchor
    df_anchor = df[df['Contract_ID'] >= anchor_id].copy().sort_values('Contract_ID')
    if df_anchor.empty:
        return pd.Series([df['Rate'].iloc[-1]] * len(df))

    df_anchor['pv'] = df_anchor['Rate'] * df_anchor['Quantity']
    df_anchor['cum_pv'] = df_anchor['pv'].cumsum()
    df_anchor['cum_vol'] = df_anchor['Quantity'].cumsum()
    df_anchor['avwap'] = df_anchor['cum_pv'] / df_anchor['cum_vol']

    # Re-align with original dataframe if needed, but here we return for the anchor period
    return df_anchor['avwap']

def scan_fvgs(df: pd.DataFrame):
    """
    Automated Fair Value Gap (FVG) Scanner.
    Identify three-candle structural gaps where Candle 1 High < Candle 3 Low (Bullish)
    or Candle 1 Low > Candle 3 High (Bearish).
    """
    fvgs = []
    if len(df) < 3:
        return fvgs

    # Assuming 'Rate' is price, and we use Information Tick Buckets (ohlc-like)
    # Since Information Tick Buckets were aggregated, we use 'Rate' as close.
    # For a true FVG, we'd need High/Low of the bucket.
    # Let's assume the aggregation provides 'High' and 'Low'.
    # If not, we approximate using consecutive Rates.

    rates = df['Rate'].values
    for i in range(1, len(rates) - 1):
        # Bullish FVG: Low[i+1] > High[i-1]
        # Using Rate as proxy for High/Low if they don't exist
        if rates[i+1] > rates[i-1]:
            fvgs.append({'type': 'BULLISH', 'start_id': df['Contract_ID'].iloc[i-1], 'end_id': df['Contract_ID'].iloc[i+1], 'top': rates[i+1], 'bottom': rates[i-1]})
        # Bearish FVG: High[i+1] < Low[i-1]
        elif rates[i+1] < rates[i-1]:
            fvgs.append({'type': 'BEARISH', 'start_id': df['Contract_ID'].iloc[i-1], 'end_id': df['Contract_ID'].iloc[i+1], 'top': rates[i-1], 'bottom': rates[i+1]})

    return fvgs

def scan_order_blocks(df: pd.DataFrame):
    """
    Order Block (OB) Scanner.
    Identify final down-candle volume signature prior to an aggressive upward expansion leg.
    """
    obs = []
    if len(df) < 5: return obs

    rates = df['Rate'].values
    vols = df['Quantity'].values

    for i in range(1, len(rates) - 2):
        # Bullish OB: Price drops (Rate[i] < Rate[i-1]) then aggressively expands (Rate[i+2] >> Rate[i])
        if rates[i] < rates[i-1] and rates[i+2] > rates[i] * 1.02:
            obs.append({
                'type': 'BULLISH',
                'id': df['Contract_ID'].iloc[i],
                'price': rates[i],
                'volume': vols[i]
            })
    return obs

def get_smc_zones(current_price: float, swing_low: float, swing_high: float, avwap: float):
    """
    SMC Spatial Tiers.
    TIER 1: APEX ZONE (> 75% of Swing)
    TIER 2: PREMIUM ZONE (AVWAP to 75%)
    TIER 3: DISCOUNT ZONE (Base to AVWAP)
    """
    apex_threshold = swing_low + 0.75 * (swing_high - swing_low)

    if current_price > apex_threshold:
        return "TIER 1: APEX ZONE"
    elif current_price > avwap:
        return "TIER 2: PREMIUM ZONE"
    else:
        return "TIER 3: DISCOUNT ZONE"

def check_execution_triggers(
    is_htf_ltf_coherent: bool,
    is_idm_swept: bool,
    is_ob_fvg_tapped: bool,
    is_choch: bool,
    wnp: float,
    kyles_lambda: float
) -> bool:
    """Execution Sequence Triggers."""
    if not (is_htf_ltf_coherent and is_idm_swept and is_ob_fvg_tapped and is_choch):
        return False
    # Validate with microstructure
    if wnp > 0 and kyles_lambda < 0.1:
        return True
    return False

def calculate_oli(overnight_up_vol: float, overnight_down_vol: float, median_opening_vol: float) -> float:
    """24-Hour Off-Hour Order Imbalance (OLI)."""
    if median_opening_vol == 0: return 0.0
    return (overnight_up_vol - overnight_down_vol) / median_opening_vol

def post_halt_momentum_rebound(first_100_vols: np.ndarray, first_100_sides: np.ndarray) -> bool:
    """Post-Halt Momentum Rebound (Rm)."""
    if len(first_100_vols) < 100: return False
    # High velocity if top 10% transactions are > 50% of volume
    top_10_threshold = np.percentile(first_100_vols, 90)
    whale_vol = np.sum(first_100_vols[first_100_vols >= top_10_threshold])
    if whale_vol > np.sum(first_100_vols) * 0.5:
        # Check if buyers dominant
        if np.sum(first_100_sides) > 0:
            return True
    return False

def avellaneda_stoikov_scaling(inventory: int, risk_aversion: float, vol: float, time_horizon: float) -> float:
    """
    Avellaneda-Stoikov Inventory Scaling.
    Adjust entry sizing based on current portfolio exposure.
    """
    # Simplified reservation price adjustment
    # delta_p = - inventory * gamma * sigma^2 * (T - t)
    adjustment = - inventory * risk_aversion * (vol**2) * time_horizon
    return adjustment

def concave_square_root_sizing(total_vol: float, adv: float, sigma: float, y: float = 0.5) -> float:
    """
    Concave Square-Root Sizing Law.
    Market impact = sigma * (order_size / ADV)^y
    Used to downscale inside illiquid matrices.
    """
    if adv == 0: return 0.0
    # To keep impact below a certain threshold (e.g., 10bps), we solve for order_size
    target_impact = 0.001
    order_size = adv * (target_impact / (sigma + 1e-9))**(1/y)
    return order_size
