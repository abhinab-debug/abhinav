import numpy as np
import pandas as pd
from scipy import stats

def reconstruct_sides(rates: np.ndarray) -> np.ndarray:
    """
    Vectorized Reconstructed Tick Test.
    S_t = 1 if ΔP > 0, -1 if ΔP < 0, S_{t-1} if ΔP = 0.
    """
    if len(rates) == 0:
        return np.array([])

    sides = np.zeros(len(rates))
    # Seed the first side
    if len(rates) > 1:
        i = 1
        while i < len(rates) and rates[i] == rates[0]:
            i += 1
        if i < len(rates):
            sides[0] = 1 if rates[i] > rates[0] else -1
        else:
            sides[0] = 1
    else:
        sides[0] = 1

    # Using a simple loop for the recursive carry-forward logic as it's inherently sequential
    for i in range(1, len(rates)):
        diff = rates[i] - rates[i-1]
        if diff > 0:
            sides[i] = 1
        elif diff < 0:
            sides[i] = -1
        else:
            sides[i] = sides[i-1]
    return sides

def calculate_wnp(quantities: np.ndarray, sides: np.ndarray) -> float:
    """Whale Net Pressure (WNP): Filters > 95th percentile transactions."""
    if len(quantities) == 0:
        return 0.0
    threshold = np.percentile(quantities, 95)
    whale_mask = quantities >= threshold
    wnp = np.sum(quantities[whale_mask & (sides == 1)]) - np.sum(quantities[whale_mask & (sides == -1)])
    return float(wnp)

def calculate_gini(volumes: np.ndarray) -> float:
    """Gini Coefficient of Volume: Measures trade size inequality."""
    vols = np.sort(volumes)
    n = len(vols)
    if n == 0 or np.sum(vols) == 0:
        return 0.0
    index_array = np.arange(1, n + 1)
    gini = (2 * np.sum(index_array * vols)) / (n * np.sum(vols)) - ((n + 1) / n)
    return float(gini)

def calculate_tib(sides: np.ndarray) -> float:
    """Tick Imbalance Sign Divergence (TIB)."""
    return float(np.sum(sides))

def calculate_vpin(buys: np.ndarray, sells: np.ndarray, v_bucket: float) -> float:
    """
    Volume-Synchronized Probability of Toxicity (VPIN).
    VPIN = sum(|V_buy - V_sell|) / (N * V_bucket)
    """
    if v_bucket <= 0 or len(buys) == 0:
        return 0.0
    imbalance = np.abs(buys - sells)
    vpin = np.sum(imbalance) / (len(buys) * v_bucket)
    return float(vpin)

def calculate_hurst_rolling(series: np.ndarray) -> float:
    """
    Rolling Hurst Exponent (H) via Rescaled Range (R/S) analysis.
    """
    if len(series) < 10:
        return 0.5

    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]

    # Filter out zeros to avoid log issues
    valid_indices = [i for i, t in enumerate(tau) if t > 0]
    if not valid_indices: return 0.5

    log_lags = np.log([lags[i] for i in valid_indices])
    log_tau = np.log([tau[i] for i in valid_indices])

    poly = np.polyfit(log_lags, log_tau, 1)
    return float(poly[0] * 2.0)

def calculate_kyles_lambda(prices: np.ndarray, signed_volumes: np.ndarray) -> float:
    """
    Kyle's Lambda (λ): Rolling OLS of price change over signed transaction volume.
    Handle 15% circuit breaker zero-variance boundaries.
    """
    if len(prices) < 2 or len(signed_volumes) < 2:
        return 0.0

    price_changes = np.diff(prices)
    # Align lengths: Signed volume for the period leading to the price change
    x = signed_volumes[1:]
    y = price_changes

    if np.all(x == 0) or np.std(x) == 0:
        return 0.0

    slope, _, _, _, _ = stats.linregress(x, y)
    return float(slope)

def calculate_svke(sides: np.ndarray, volumes: np.ndarray, price_velocity: float) -> float:
    """Signed Volume Kinetic Energy (SVKE): directional net force of total order flow."""
    force = np.sum(sides * volumes)
    svke = 0.5 * force * (price_velocity ** 2)
    return float(svke)

def calculate_amihud(returns: np.ndarray, amounts: np.ndarray) -> float:
    """Amihud Illiquidity (At): Absolute ratio of tick returns to monetary turnover."""
    if len(amounts) == 0 or np.sum(amounts) == 0:
        return 0.0
    # Aligned lengths
    n = min(len(returns), len(amounts))
    if n == 0: return 0.0
    amihud = np.mean(np.abs(returns[:n]) / amounts[:n])
    return float(amihud)

def calculate_tsrv(prices: np.ndarray) -> float:
    """
    Two-Scale Realized Variance (TSRV).
    Subsamples overlapping tick grids to filter microstructure noise.
    """
    if len(prices) < 10:
        return 0.0

    # Scale 1: Full grid
    rv_full = np.sum(np.diff(prices)**2)

    # Scale 2: Subsampled grid (e.g., every 5 ticks)
    k = 5
    rv_avg = 0
    for i in range(k):
        sub_prices = prices[i::k]
        if len(sub_prices) > 1:
            rv_avg += np.sum(np.diff(sub_prices)**2)
    rv_avg /= k

    # TSRV formula: Bias correction
    n = len(prices)
    n_avg = (n - k + 1) / k
    tsrv = rv_avg - (n_avg / n) * rv_full
    return float(max(0, tsrv))

def calculate_jensens_alpha(returns: np.ndarray, market_returns: np.ndarray) -> float:
    """Jensen's Alpha (Ja): Tick-by-tick market outperformance."""
    if len(returns) < 5 or len(market_returns) < 5:
        return 0.0
    # Assuming risk-free rate is 0 in tick-space
    n = min(len(returns), len(market_returns))
    slope, intercept, _, _, _ = stats.linregress(market_returns[:n], returns[:n])
    return float(intercept)

def detect_icebergs(rates: np.ndarray, volumes: np.ndarray) -> bool:
    """
    Iceberg Detection Engine: Audits queue depletion anomalies.
    Flag where volume expands but price velocity drops to zero.
    """
    if len(rates) < 10: return False

    # Calculate price velocity (diff)
    velocity = np.abs(np.diff(rates))
    # Thresholds for 'expanding volume' and 'zero velocity'
    vol_threshold = np.percentile(volumes, 90)

    # Look for sequential ticks with high volume and zero price movement
    for i in range(len(velocity) - 3):
        if np.all(velocity[i:i+3] == 0) and np.any(volumes[i:i+3] > vol_threshold):
            return True
    return False

def calculate_liquidity_commonality(symbol_liq: np.ndarray, market_liq: np.ndarray) -> float:
    """Liquidity Commonality (βL)."""
    if len(symbol_liq) < 5 or len(market_liq) < 5:
        return 0.0
    n = min(len(symbol_liq), len(market_liq))
    corr, _ = stats.pearsonr(symbol_liq[:n], market_liq[:n])
    return float(corr)

def calculate_gtv(total_contracts: int, minutes: float) -> float:
    """Global Transaction Velocity (GTV): Contract ID issuance per minute."""
    if minutes <= 0: return 0.0
    return float(total_contracts / minutes)

def calculate_hasbrouck_impact(prices: np.ndarray, signed_volumes: np.ndarray) -> float:
    """
    Hasbrouck Permanent Impact (I_P) using VAR-like proxy.
    Isolates structural price changes from microstructure noise.
    """
    if len(prices) < 5 or len(signed_volumes) < 5:
        return 0.0

    returns = np.diff(prices) / prices[:-1]
    # Simple VAR proxy: Impact of signed volume(t) on returns(t+1)
    x = signed_volumes[:-1]
    y = returns[1:]

    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0

    slope, _, _, _, _ = stats.linregress(x, y)
    return float(slope)
