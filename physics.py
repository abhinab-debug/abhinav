import numpy as np
import pandas as pd
from scipy import stats

def reconstruct_sides(rates: np.ndarray) -> np.ndarray:
    """
    Reconstructed Tick-Test Algorithm to infer order-flow directionality.
    S_t = 1 if ΔP > 0, -1 if ΔP < 0, S_{t-1} if ΔP = 0.
    """
    if len(rates) == 0:
        return np.array([])

    sides = np.zeros(len(rates))
    # Seed the first side based on the first price change
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
    """Whale Net Pressure (WNP): Aggressive Block Buys minus Block Sells (>95th percentile)."""
    if len(quantities) == 0:
        return 0.0
    threshold = np.percentile(quantities, 95)
    whale_mask = quantities >= threshold
    whale_buys = np.sum(quantities[whale_mask & (sides == 1)])
    whale_sells = np.sum(quantities[whale_mask & (sides == -1)])
    return float(whale_buys - whale_sells)

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
    """Tick Imbalance Sign Divergence (TIB): Cumulative aggressive execution strings."""
    return float(np.sum(sides))

def calculate_vpin(buys: np.ndarray, sells: np.ndarray, v_bucket: float) -> float:
    """Volume-Synchronized Probability of Toxicity (VPIN)."""
    if v_bucket == 0:
        return 0.0
    imbalance = np.abs(buys - sells)
    vpin = np.sum(imbalance) / (len(buys) * v_bucket)
    return float(vpin)

def calculate_hurst(series: np.ndarray) -> float:
    """Hurst Exponent (H): Measures fractal trend persistence."""
    if len(series) < 10:
        return 0.5
    lags = range(2, 20)
    tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
    # Filter out zeros for log calculation
    valid = [(l, t) for l, t in zip(lags, tau) if t > 0]
    if not valid: return 0.5
    lags, tau = zip(*valid)
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return float(poly[0] * 2.0)

def calculate_kyles_lambda(prices: np.ndarray, signed_volumes: np.ndarray) -> float:
    """Kyle's Lambda (λ): Regression-based market impact."""
    if len(prices) < 2 or len(signed_volumes) < 1:
        return 0.0
    price_changes = np.diff(prices)
    # Align lengths
    y = price_changes
    x = signed_volumes[1:] # Skip first as diff reduces length by 1
    if len(x) == 0 or len(y) == 0: return 0.0
    slope, _, _, _, _ = stats.linregress(x, y)
    return float(slope)

def calculate_hasbrouck_impact(prices: np.ndarray, signed_volumes: np.ndarray) -> float:
    """Hasbrouck Permanent Impact (I_P) - simplified version."""
    # Permanent impact is often modeled as the long-term price change relative to trade
    # Here we use a simple proxy: correlation of signed volume and future returns
    if len(prices) < 2: return 0.0
    returns = np.diff(prices) / prices[:-1]
    if len(returns) == 0: return 0.0
    corr, _ = stats.pearsonr(signed_volumes[:-1], returns)
    return float(corr)

def calculate_svke(sides: np.ndarray, volumes: np.ndarray, price_velocity: float) -> float:
    """Signed Volume Kinetic Energy (SVKE): directional net force of total order flow."""
    force = np.sum(sides * volumes)
    svke = 0.5 * force * (price_velocity ** 2)
    return float(svke)

def calculate_amihud(returns: np.ndarray, amounts: np.ndarray) -> float:
    """Amihud Illiquidity (At): Absolute ratio of tick returns to monetary turnover."""
    if len(amounts) == 0 or np.sum(amounts) == 0:
        return 0.0
    amihud = np.mean(np.abs(returns) / amounts)
    return float(amihud)

def calculate_tsrv(prices: np.ndarray) -> float:
    """Two-Scale Realized Variance (TSRV): Filters out microstructure noise."""
    if len(prices) < 2: return 0.0
    # Simplified TSRV using average of variances at different scales
    diff1 = np.diff(prices)
    rv1 = np.sum(diff1**2)

    # Scale 2
    if len(prices) < 3: return float(rv1)
    diff2 = prices[2:] - prices[:-2]
    rv2 = np.sum(diff2**2) / 2.0

    return float(rv2 - (len(prices)/2.0)/len(prices) * rv1)

def calculate_jensens_alpha(returns: np.ndarray, market_returns: np.ndarray) -> float:
    """Jensen's Alpha (Ja): Outperformance relative to baseline index."""
    if len(returns) == 0 or len(market_returns) == 0: return 0.0
    # Assuming risk-free rate is 0 for simplicity
    beta, alpha, _, _, _ = stats.linregress(market_returns, returns)
    return float(alpha)

def detect_icebergs(quantities: np.ndarray) -> bool:
    """Iceberg Detection Engine: Audits transaction sequence queue depletion anomalies."""
    if len(quantities) < 5: return False
    # Simplified: look for repeated identical small volumes in sequence
    # which might indicate a sliced larger order
    for i in range(len(quantities) - 3):
        if len(set(quantities[i:i+4])) == 1:
            return True
    return False

def calculate_liquidity_commonality(symbol_liquidity: np.ndarray, market_liquidity: np.ndarray) -> float:
    """Liquidity Commonality (βL): Systemic co-movements in liquidity."""
    if len(symbol_liquidity) < 2 or len(market_liquidity) < 2: return 0.0
    corr, _ = stats.pearsonr(symbol_liquidity, market_liquidity)
    return float(corr)

def calculate_gtv(total_contracts: int, minutes: float) -> float:
    """Global Transaction Velocity (GTV): Rate of total Contract ID issuance per minute."""
    if minutes == 0: return 0.0
    return float(total_contracts / minutes)
