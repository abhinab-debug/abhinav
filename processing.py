import pandas as pd
import numpy as np
import re

def sanitize_floorsheet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deterministic purification matrix:
    1. Instrument Exclusion (Mutual Funds, Promoter Shares, Debentures)
    2. Temporal Session Purge (Jan 3rd)
    """
    if df.empty:
        return df

    # 1. Instrument Exclusion Matrix
    mf_regex = r'MF$|SF$'
    promoter_regex = r'PO$|P$|-PO'
    debenture_regex = r'D\d{2}|BD|D80'
    combined_regex = f"{mf_regex}|{promoter_regex}|{debenture_regex}"

    df = df[~df['Symbol'].str.contains(combined_regex, regex=True, na=False)]

    # 2. Temporal Session Purge (2026-01-03)
    # Ensuring date comparison is robust
    df['Date_tmp'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    df = df[df['Date_tmp'] != '2026-01-03']
    df = df.drop(columns=['Date_tmp'])

    return df

def generate_information_tick_bars(df: pd.DataFrame, v_threshold: float) -> pd.DataFrame:
    """
    Compresses transaction records into uniform Information Tick Buckets.
    """
    if df.empty or v_threshold <= 0:
        return df

    df = df.sort_values('Contract_ID')
    df['cum_vol'] = df['Quantity'].cumsum()
    df['bucket'] = (df['cum_vol'] // v_threshold).astype(int)

    tick_bars = df.groupby('bucket').agg({
        'Contract_ID': 'last',
        'Symbol': 'first',
        'Rate': 'last',
        'Quantity': 'sum',
        'Amount': 'sum',
        'Date': 'last'
    }).reset_index(drop=True)

    # Calculate returns inside the bucketed space
    tick_bars['Return'] = tick_bars['Rate'].pct_change().fillna(0)

    return tick_bars

def apply_transaction_proximity_slicing(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Collapses fragmented trades into Synthetic Institutional Blocks.
    TPI = (var_volume / mean_volume) * ln(delta_C + 1)
    """
    if len(df) < window:
        return df

    df = df.sort_values('Contract_ID')

    # Vectorized TPI-like logic for window-based clustering
    # This is a simplified version of the TPI logic to find clusters
    vol_mean = df['Quantity'].rolling(window).mean()
    vol_var = df['Quantity'].rolling(window).var()
    delta_c = df['Contract_ID'].diff(window-1)

    tpi = (vol_var / vol_mean) * np.log(delta_c + 1)

    # Flag clusters where TPI is extremely low
    df['is_cluster'] = (tpi < 1.0) & (tpi >= 0)

    # For now, we flag them. Actual 'collapsing' would require defining cluster boundaries.
    # In a production system, we'd group sequential 'is_cluster' rows.

    return df
