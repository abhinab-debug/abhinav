import numpy as np
import pandas as pd
from typing import List, Dict

class SMCScanner:
    """
    NEPSE Sovereign Quantitative Inference Terminal (v15.0)
    File 3: SMC Spatial Geometry & Information Tick-Bar Engine.
    Rejecting retail indicators for transaction-anchored spatial liquidity.
    """

    @staticmethod
    def generate_information_tick_bars(df: pd.DataFrame, alpha: float = 0.5) -> pd.DataFrame:
        """
        Compresses raw transaction ledger into uniform Information Tick-Bars.
        A new bucket is formed when cumulative volume exceeds V_threshold.
        V_threshold = alpha * Median(V_20-day, rolling).
        """
        if df.empty:
            return pd.DataFrame()

        # Ensure ledger is sorted by sequencing
        df = df.sort_values('Contract_ID')

        # Calculate Rolling 20-day Median Volume (Cold-Start Immortality)
        daily_vols = df.groupby('Date')['Quantity'].sum().reset_index()
        lookback = min(20, len(daily_vols))

        # Median of daily volumes over the lookback period
        median_daily_vol = daily_vols['Quantity'].rolling(window=lookback, min_periods=1).median().iloc[-1]
        v_threshold = alpha * median_daily_vol

        if v_threshold <= 0:
            v_threshold = df['Quantity'].median() * 100 # Emergency fallback

        # Bucketing Logic: Vectorized cumulative volume slicing
        df['cum_vol'] = df['Quantity'].cumsum()
        df['bucket'] = (df['cum_vol'] // v_threshold).astype(int)

        # Aggregate buckets into high-fidelity OHLCV Bars
        bars = df.groupby('bucket').agg(
            ID_Start=('Contract_ID', 'first'),
            ID_End=('Contract_ID', 'last'),
            Open=('Rate', 'first'),
            High=('Rate', 'max'),
            Low=('Rate', 'min'),
            Close=('Rate', 'last'),
            Volume=('Quantity', 'sum'),
            Date=('Date', 'last'),
            Timestamp=('Timestamp', 'last')
        ).reset_index(drop=True)

        return bars

    @staticmethod
    def calculate_tpi(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Transaction Proximity Index (TPI): Detects Synthetic Institutional Blocks.
        TPI = (var_vol / mean_vol) * ln(delta_C + 1).
        Stabilization at 1e-9 to prevent division by zero in illiquid states.
        """
        if len(df) < 2:
            return pd.Series(0.0, index=df.index)

        # Rolling stats for transaction intensity
        vol_mean = df['Quantity'].rolling(window=window, min_periods=1).mean()
        vol_var = df['Quantity'].rolling(window=window, min_periods=1).var().fillna(0)

        # Sequence delta between contracts
        delta_c = df['Contract_ID'].diff().fillna(1)

        # TPI Calculation with 1e-9 stabilization
        tpi = (vol_var / (vol_mean + 1e-9)) * np.log(delta_c + 1)
        return tpi

    @staticmethod
    def scan_fvgs(bars: pd.DataFrame) -> pd.DataFrame:
        """
        SMC Engine - Fair Value Gaps (FVG).
        A Bullish FVG exists at index t if Low(t) > High(t-2).
        Returns unmitigated zone boundaries.
        """
        fvgs = []
        if len(bars) < 3:
            return pd.DataFrame()

        for t in range(2, len(bars)):
            # Bullish FVG detection
            if bars['Low'].iloc[t] > bars['High'].iloc[t-2]:
                fvgs.append({
                    'Type': 'BULLISH',
                    'Top': bars['Low'].iloc[t],
                    'Bottom': bars['High'].iloc[t-2],
                    'ID_Start': bars['ID_End'].iloc[t-2],
                    'ID_End': bars['ID_Start'].iloc[t],
                    'Date': bars['Date'].iloc[t],
                    'Mitigated': False
                })

        return pd.DataFrame(fvgs)

    @staticmethod
    def scan_idm_ob(bars: pd.DataFrame) -> List[Dict]:
        """
        SMC Engine - Inducement (IDM) Sweeps & Order Blocks (OB).
        IDM: Wick pierces localized swing low, but Close stays above.
        OB: Absolute range of the down-close candle immediately preceding the sweep.
        """
        obs = []
        if len(bars) < 5:
            return obs

        # Fractal-based localized swing low detection
        for i in range(2, len(bars) - 2):
            is_swing_low = (bars['Low'].iloc[i] < bars['Low'].iloc[i-1] and
                            bars['Low'].iloc[i] < bars['Low'].iloc[i-2] and
                            bars['Low'].iloc[i] < bars['Low'].iloc[i+1] and
                            bars['Low'].iloc[i] < bars['Low'].iloc[i+2])

            if is_swing_low:
                swing_low = bars['Low'].iloc[i]

                # Monitor for IDM sweep in the transaction stream
                for j in range(i+1, len(bars)):
                    # Wick pierce check: Low < Swing Low AND Close > Swing Low
                    if bars['Low'].iloc[j] < swing_low and bars['Close'].iloc[j] > swing_low:
                        # Found IDM Sweep at bar j
                        # Search for the down-close candle range immediately preceding
                        for k in range(j-1, -1, -1):
                            if bars['Close'].iloc[k] < bars['Open'].iloc[k]:
                                obs.append({
                                    'Type': 'BULLISH_OB',
                                    'Price_High': bars['High'].iloc[k],
                                    'Price_Low': bars['Low'].iloc[k],
                                    'ID_Start': bars['ID_Start'].iloc[k],
                                    'ID_End': bars['ID_End'].iloc[k],
                                    'Swing_Low': swing_low,
                                    'Sweep_Bar_Idx': j
                                })
                                break
                        break

                    # Invalidation: Full candle close below swing low (BOS)
                    if bars['Close'].iloc[j] < swing_low:
                        break
        return obs

    @staticmethod
    def calculate_avwap_equilibrium(df: pd.DataFrame, bars: pd.DataFrame) -> Dict:
        """
        AVWAP Equilibrium: Anchored to rolling 20-day swing high/low.
        Establishes Premium, Discount, and Equilibrium zones for spatial routing.
        """
        if df.empty or bars.empty:
            return {"Equilibrium": 0.0, "Premium": 0.0, "Discount": 0.0, "High": 0.0, "Low": 0.0}

        # Identify Swing Extremes (Cold-start protected)
        lookback = min(20, len(bars))
        recent_bars = bars.iloc[-lookback:]

        swing_high_idx = recent_bars['High'].idxmax()
        swing_low_idx = recent_bars['Low'].idxmin()

        # Anchor to the origin of the active swing leg
        anchor_idx = min(swing_high_idx, swing_low_idx)
        anchor_id = bars['ID_Start'].iloc[anchor_idx]

        # Extract transaction subset for AVWAP
        df_anchor = df[df['Contract_ID'] >= anchor_id].copy()
        if df_anchor.empty:
            current_avwap = df['Rate'].iloc[-1]
        else:
            # AVWAP = Sum(P * V) / Sum(V)
            current_avwap = (df_anchor['Rate'] * df_anchor['Quantity']).sum() / df_anchor['Quantity'].sum()

        leg_high = recent_bars['High'].max()
        leg_low = recent_bars['Low'].min()

        # Establishing Spatial Zones
        equilibrium = current_avwap
        premium = equilibrium + (leg_high - equilibrium) * 0.5
        discount = leg_low + (equilibrium - leg_low) * 0.5

        return {
            "Equilibrium": float(equilibrium),
            "Premium": float(premium),
            "Discount": float(discount),
            "High": float(leg_high),
            "Low": float(leg_low)
        }

if __name__ == "__main__":
    print("SMC Spatial Geometry Engine (v15.0) Loaded.")
