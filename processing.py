import pandas as pd
import numpy as np
import re

def sanitize_floorsheet(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executes deterministic filtering to purge noise, non-equity instruments,
    and corrupted sessions.
    """
    if df.empty:
        return df

    # 1. Instrument Exclusion Matrix
    # Debentures & Fixed-Income Bonds
    debenture_regex = r'D\d{2}|BD|D\d{2}/\d{2}'
    # Mutual Funds
    mf_regex = r'MF$|SF$'
    # Promoter Shares
    promoter_regex = r'PO$|P$|-PO'

    combined_regex = f"{debenture_regex}|{mf_regex}|{promoter_regex}"

    # Filter out instruments
    df = df[~df['Symbol'].str.contains(combined_regex, regex=True, na=False)]

    # 2. Temporal Corruption Purge: Unconditionally drop Jan 3rd
    # Assuming 'Date' column is present and can be converted to datetime
    df['Date_dt'] = pd.to_datetime(df['Date'])
    df = df[~((df['Date_dt'].dt.month == 1) & (df['Date_dt'].dt.day == 3))]
    df = df.drop(columns=['Date_dt'])

    return df

def generate_information_tick_bars(df: pd.DataFrame, v_threshold: float) -> pd.DataFrame:
    """
    Compresses transaction records into uniform Information Tick Buckets based on volume threshold.
    V_threshold = alpha * Median(V_20-day, rolling)
    """
    if df.empty or v_threshold <= 0:
        return df

    df = df.sort_values('Contract_ID')
    df['cum_vol'] = df['Quantity'].cumsum()
    df['bucket'] = (df['cum_vol'] // v_threshold).astype(int)

    # Aggregate by bucket
    tick_bars = df.groupby('bucket').agg({
        'Contract_ID': 'last',
        'Symbol': 'first',
        'Rate': 'last', # Close of the bucket
        'Quantity': 'sum',
        'Amount': 'sum',
        'Date': 'last'
    }).reset_index(drop=True)

    return tick_bars

def calculate_tpi(window_df: pd.DataFrame) -> float:
    """
    Calculates the Transaction Proximity Index (TPI).
    TPI = (var_volume / mean_volume) * ln(delta_C + 1)
    """
    if window_df.empty or len(window_df) < 2:
        return 0.0

    vol_var = window_df['Quantity'].var()
    vol_mean = window_df['Quantity'].mean()
    delta_c = window_df['Contract_ID'].max() - window_df['Contract_ID'].min()

    if vol_mean == 0:
        return 0.0

    tpi = (vol_var / vol_mean) * np.log(delta_c + 1)
    return tpi

def load_raw_xlsx(filepath: str) -> pd.DataFrame:
    """Loads raw floorsheet data from Excel file."""
    return pd.read_excel(filepath)
