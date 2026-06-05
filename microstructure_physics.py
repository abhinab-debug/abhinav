import numpy as np
import pandas as pd
from scipy import stats
from typing import List

class MicrostructurePhysics:
    """
    NEPSE Sovereign Quantitative Inference Terminal (v15.0)
    Phase 2: Pure Transaction Physics & Structural Microstructure Pillars.
    """

    @staticmethod
    def reconstruct_tick_test(df: pd.DataFrame) -> pd.Series:
        """
        Reconstructed Tick-Test Algorithm: Infers trade initiation side.
        Includes 60-minute session reset and vectorized forward-fill.
        """
        if df.empty:
            return pd.Series(dtype='int32')

        # Ensure sorted by Contract_ID
        df = df.sort_values('Contract_ID')

        # Calculate Price Jump Velocity (ΔP)
        df['price_diff'] = df['Rate'].diff()

        # Deterministic Side Assignment
        # 1: Buy (Up-tick), -1: Sell (Down-tick), 0: Zero-tick
        df['Side'] = np.where(df['price_diff'] > 0, 1,
                              np.where(df['price_diff'] < 0, -1, 0))

        # Vectorized Forward-Fill for Zero-Tick Stacks
        # Replaces 0 with NaN so ffill() can propagate the last known trade side
        df['Side'] = df['Side'].replace(0, np.nan).ffill().fillna(1).astype('int32')

        # Session Reset Logic: Reset to neutral if time gap > 60 minutes
        if 'Timestamp' in df.columns:
            df['dt'] = pd.to_datetime(df['Date'] + ' ' + df['Timestamp'])
            df['time_gap'] = df['dt'].diff().dt.total_seconds() / 60.0
            df.loc[df['time_gap'] > 60, 'Side'] = 0 # Reset to neutral on session gap

        return df['Side']

    @staticmethod
    def calculate_wnp(volumes: np.ndarray, sides: np.ndarray) -> float:
        """
        Whale Net Pressure (WNP): Aggregates block buys vs block sells (>95th percentile).
        WNP = sum(V_Whale,Buy) - sum(V_Whale,Sell)
        """
        if len(volumes) == 0:
            return 0.0

        threshold = np.percentile(volumes, 95)
        whale_mask = volumes >= threshold

        wnp = np.sum(volumes[whale_mask & (sides == 1)]) - np.sum(volumes[whale_mask & (sides == -1)])
        return float(wnp)

    @staticmethod
    def calculate_gini_coefficient(volumes: np.ndarray) -> float:
        """
        Gini Coefficient of Volume (G): Measures trade size inequality.
        G = (2 * sum(i * v_i)) / (n * sum(v_i)) - (n+1)/n
        """
        n = len(volumes)
        if n < 2 or np.sum(volumes) == 0:
            return 0.0

        sorted_v = np.sort(volumes)
        index = np.arange(1, n + 1)

        gini = (2 * np.sum(index * sorted_v)) / (n * np.sum(sorted_v)) - (n + 1) / n
        return float(gini)

    @staticmethod
    def calculate_vpin(volumes: np.ndarray, sides: np.ndarray, num_buckets: int = 50) -> float:
        """
        VPIN Fractional Slicing Loop: Volume-Synchronized Probability of Toxicity.
        VPIN = sum(|V_tau_B - V_tau_S|) / (N * V_bucket)
        """
        total_vol = np.sum(volumes)
        if total_vol == 0:
            return 0.0

        v_bucket = total_vol / num_buckets

        # Slicing fractional trades into buckets
        buys = volumes * (sides == 1)
        sells = volumes * (sides == -1)

        cum_vol = np.cumsum(volumes)
        bucket_indices = (cum_vol // v_bucket).clip(0, num_buckets - 1).astype(int)

        bucket_df = pd.DataFrame({'bucket': bucket_indices, 'buy': buys, 'sell': sells})
        bucket_agg = bucket_df.groupby('bucket').sum()

        imbalance = np.abs(bucket_agg['buy'] - bucket_agg['sell'])
        vpin = imbalance.sum() / (num_buckets * v_bucket)

        return float(vpin)

    @staticmethod
    def calculate_tsrv(prices: np.ndarray, k: int = 15) -> float:
        """
        Two-Scale Realized Variance (TSRV): Microstructure noise-adjusted variance.
        Fast (1-tick) vs Slow (k-tick) sampling with bias correction.
        """
        n = len(prices)
        if n <= k:
            return 0.0

        # 1. Fast Scale (1-tick)
        rv_fast = np.sum(np.diff(prices)**2)

        # 2. Slow Scale (k-tick average)
        rv_slow_sum = 0
        for i in range(k):
            sub_prices = prices[i::k]
            if len(sub_prices) > 1:
                rv_slow_sum += np.sum(np.diff(sub_prices)**2)

        rv_slow = rv_slow_sum / k

        # 3. Bias Correction
        # Adj = (1 - ((n - k + 1) / n))^-1
        adj = 1.0 / (1.0 - ((n - k + 1) / n))
        tsrv = adj * (rv_slow - (1.0/k) * rv_fast)

        return float(max(0, tsrv))

    @staticmethod
    def calculate_hasbrouck_impact(prices: np.ndarray, signed_volumes: np.ndarray) -> float:
        """
        Hasbrouck Permanent Impact (I_P): Isolates price impact from signed volume.
        Regress ΔP_t against lagged ΔP_t-1 and signed_volume_t-1.
        """
        if len(prices) < 3 or len(signed_volumes) < 3:
            return 0.0

        dp = np.diff(prices)
        y = dp[1:] # ΔP_t, length N-2
        x1 = dp[:-1] # ΔP_t-1, length N-2
        x2 = signed_volumes[1:-1] # S_t-1 * V_t-1, length N-2

        # Manual OLS: β = (X^T X)^-1 X^T Y
        X = np.column_stack([np.ones(len(y)), x1, x2])
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            return float(beta[2]) # Impact coefficient of signed volume
        except:
            return 0.0

    @staticmethod
    def calculate_hurst_exponent(prices: np.ndarray) -> float:
        """
        Hurst Exponent (H): Rescaled Range (R/S) analysis for memory regimes.
        Analyzed over variable lags [2, 4, 8, 16].
        """
        if len(prices) < 32:
            return 0.5

        lags = [2, 4, 8, 16]
        rs_values = []

        for lag in lags:
            # Rescaled Range calculation
            # This is a simplified R/S for the specific requested lags
            diffs = np.diff(prices)
            # Split into chunks of 'lag'
            num_chunks = len(diffs) // lag
            if num_chunks == 0: continue

            chunk_rs = []
            for i in range(num_chunks):
                chunk = diffs[i*lag : (i+1)*lag]
                mean_adj = np.cumsum(chunk - np.mean(chunk))
                r = np.max(mean_adj) - np.min(mean_adj)
                s = np.std(chunk)
                if s > 0:
                    chunk_rs.append(r / s)

            if chunk_rs:
                rs_values.append(np.mean(chunk_rs))
            else:
                rs_values.append(1.0) # Neutral fallback

        # Hurst is the slope of log(R/S) vs log(lag)
        if len(rs_values) < 2: return 0.5
        h, _ = np.polyfit(np.log(lags), np.log(rs_values), 1)
        return float(h)

    @staticmethod
    def calculate_kyles_lambda(prices: np.ndarray, signed_volumes: np.ndarray) -> float:
        """
        Kyle's Lambda (λ): Measures price impact of signed order flow.
        ΔP_t = μ + λ * (Signed Volume_t) + ε
        """
        if len(prices) < 2:
            return 0.0

        dp = np.diff(prices)
        sv = signed_volumes[1:]

        if np.all(sv == 0) or np.std(sv) == 0:
            return 0.0

        slope, _, _, _, _ = stats.linregress(sv, dp)
        return float(slope)

    @staticmethod
    def calculate_svke(sides: np.ndarray, volumes: np.ndarray, price_velocity: float) -> float:
        """
        Signed Volume Kinetic Energy (SVKE): Directional force of transaction stream.
        SVKE = 0.5 * sum(S_t * v_t) * Velocity^2
        """
        force = np.sum(sides * volumes)
        svke = 0.5 * force * (price_velocity ** 2)
        return float(svke)

    @staticmethod
    def calculate_amihud_illiquidity(returns: np.ndarray, amounts: np.ndarray) -> float:
        """
        Amihud Illiquidity (A_t): Absolute return per unit of monetary volume.
        A_t = |R_t| / Amount_t
        """
        if len(amounts) == 0 or np.sum(amounts) == 0:
            return 0.0

        # Avoid division by zero
        valid = amounts > 0
        if not np.any(valid): return 0.0

        amihud = np.mean(np.abs(returns[valid]) / amounts[valid])
        return float(amihud)

    @staticmethod
    def calculate_tib(sides: np.ndarray) -> float:
        """Tick Imbalance Sign Divergence (TIB): Sum of signed trade flags."""
        return float(np.sum(sides))

    @staticmethod
    def calculate_jensens_alpha(symbol_returns: np.ndarray, market_returns: np.ndarray) -> float:
        """Jensen's Alpha: Tick-by-tick outperformance coefficient."""
        if len(symbol_returns) < 5 or len(market_returns) < 5:
            return 0.0
        n = min(len(symbol_returns), len(market_returns))
        slope, intercept, _, _, _ = stats.linregress(market_returns[:n], symbol_returns[:n])
        return float(intercept)

    @staticmethod
    def detect_icebergs(prices: np.ndarray, volumes: np.ndarray) -> bool:
        """
        Iceberg Detection: Audits queue depletion anomalies.
        Flag where volume spikes (>90th percentile) but price remains static.
        """
        if len(prices) < 5: return False

        dp = np.abs(np.diff(prices))
        vol_threshold = np.percentile(volumes, 90)

        # Detect clusters of high volume with zero price movement
        for i in range(len(dp) - 2):
            if np.all(dp[i:i+3] == 0) and np.any(volumes[i:i+3] > vol_threshold):
                return True
        return False

    @staticmethod
    def calculate_liquidity_commonality(symbol_liq: np.ndarray, market_liq: np.ndarray) -> float:
        """Liquidity Commonality: Correlation with systematic liquidity factor."""
        if len(symbol_liq) < 5 or len(market_liq) < 5:
            return 0.0
        n = min(len(symbol_liq), len(market_liq))
        corr, _ = stats.pearsonr(symbol_liq[:n], market_liq[:n])
        return float(corr)

    @staticmethod
    def calculate_gtv(total_contracts: int, minutes: float) -> float:
        """Global Transaction Velocity (GTV): Contracts processed per minute."""
        if minutes <= 0: return 0.0
        return float(total_contracts / minutes)

if __name__ == "__main__":
    print("Microstructure Physics Engine (v15.0) Loaded.")
