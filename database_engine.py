import sqlite3
import pandas as pd
import numpy as np
import re
from typing import List, Optional, Dict

class NEPSEDatabaseEngine:
    """
    NEPSE Sovereign Quantitative Inference Terminal (v15.0)
    Core Database Engine: Handles Ingestion, Sanitization, and Feature Storage.
    """

    def __init__(self, db_path: str = "nepse_clean.db"):
        self.db_path = db_path
        self._initialize_infrastructure()

    def _initialize_infrastructure(self):
        """Configures SQLite WAL mode and initializes the high-performance schema."""
        conn = sqlite3.connect(self.db_path, timeout=60.0)
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")

        # Primary Floorsheet Ledger
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS floorsheet (
                Contract_ID INTEGER PRIMARY KEY,
                Symbol TEXT NOT NULL,
                Quantity INTEGER NOT NULL,
                Rate REAL NOT NULL,
                Amount REAL NOT NULL,
                Timestamp TEXT,
                Date TEXT NOT NULL
            )
        """)

        # Quantitative Feature Store (Materialized Results)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quant_features (
                Symbol TEXT NOT NULL,
                Date TEXT NOT NULL,
                VPIN REAL,
                WNP REAL,
                Gini REAL,
                K_Lambda REAL,
                SVKE REAL,
                Regime TEXT,
                Equilibrium REAL,
                Premium REAL,
                Discount REAL,
                PRIMARY KEY (Symbol, Date)
            )
        """)

        # Optimized Indexing
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol_date ON floorsheet (Symbol, Date);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_feat_symbol_date ON quant_features (Symbol, Date);")

        conn.commit()
        conn.close()

    @staticmethod
    def downcast_memory(df: pd.DataFrame) -> pd.DataFrame:
        """Deterministic memory downcasting for high-frequency parsing."""
        fcols = df.select_dtypes('float').columns
        icols = df.select_dtypes('integer').columns
        df[fcols] = df[fcols].astype('float32')
        df[icols] = df[icols].astype('int32')
        return df

    def apply_regulatory_sanitization(self, df: pd.DataFrame) -> pd.DataFrame:
        """Regulatory Compliance Filter: Drops non-equities and Jan 3rd session."""
        if df.empty: return df
        exclusion_regex = r'MF$|SF$|PO$|P$|-PO|D\d{2}|BD|D80'
        df = df[~df['Symbol'].str.contains(exclusion_regex, regex=True, na=False)]
        df['Date_parsed'] = pd.to_datetime(df['Date'])
        df = df[~((df['Date_parsed'].dt.month == 1) & (df['Date_parsed'].dt.day == 3))]
        df = df.drop(columns=['Date_parsed'])
        return df

    @staticmethod
    def apply_10_paisa_rounding(df: pd.DataFrame) -> pd.DataFrame:
        """Vectorized 10-Paisa Grid Rounding."""
        if 'Rate' in df.columns:
            df['Rate'] = np.round(df['Rate'] * 10) / 10
        return df

    def run_auto_healing_check(self, symbol: str, date: str) -> List[int]:
        """Verifies continuous sequence order of Contract_IDs."""
        conn = sqlite3.connect(self.db_path)
        query = "SELECT Contract_ID FROM floorsheet WHERE Symbol = ? AND Date = ? ORDER BY Contract_ID"
        df = pd.read_sql_query(query, conn, params=(symbol, date))
        conn.close()
        if df.empty: return []
        ids = df['Contract_ID'].values
        full_range = np.arange(ids[0], ids[-1] + 1)
        return np.setdiff1d(full_range, ids).tolist()

    def ingest_floorsheet(self, raw_df: pd.DataFrame):
        """Primary Ingestion Pipeline using Staging Table Merge."""
        df = self.apply_regulatory_sanitization(raw_df)
        df = self.apply_10_paisa_rounding(df)
        df = self.downcast_memory(df)
        if df.empty: return

        conn = sqlite3.connect(self.db_path, timeout=60.0)
        try:
            df.to_sql("floorsheet_temp", conn, if_exists="replace", index=False)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO floorsheet
                (Contract_ID, Symbol, Quantity, Rate, Amount, Timestamp, Date)
                SELECT Contract_ID, Symbol, Quantity, Rate, Amount, Timestamp, Date
                FROM floorsheet_temp
            """)
            conn.commit()
        finally:
            conn.close()

    def upsert_quant_features(self, df: pd.DataFrame):
        """Bulk persistence for pre-calculated quantitative metrics."""
        if df.empty: return
        conn = sqlite3.connect(self.db_path, timeout=60.0)
        try:
            df.to_sql("quant_temp", conn, if_exists="replace", index=False)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO quant_features
                (Symbol, Date, VPIN, WNP, Gini, K_Lambda, SVKE, Regime, Equilibrium, Premium, Discount)
                SELECT Symbol, Date, VPIN, WNP, Gini, K_Lambda, SVKE, Regime, Equilibrium, Premium, Discount
                FROM quant_temp
            """)
            conn.commit()
        finally:
            conn.close()

    def fetch_transaction_stream(self, symbol: str, lookback_days: int = 14) -> pd.DataFrame:
        """Retrieves raw transaction ledger subset."""
        conn = sqlite3.connect(self.db_path)
        date_query = "SELECT DISTINCT Date FROM floorsheet WHERE Symbol = ? ORDER BY Date DESC LIMIT ?"
        dates_df = pd.read_sql_query(date_query, conn, params=(symbol, lookback_days))
        if dates_df.empty:
            conn.close()
            return pd.DataFrame()
        target_dates = dates_df['Date'].tolist()
        query = f"SELECT * FROM floorsheet WHERE Symbol = ? AND Date IN ({','.join(['?']*len(target_dates))}) ORDER BY Contract_ID ASC"
        df = pd.read_sql_query(query, conn, params=[symbol] + target_dates)
        conn.close()
        return df

    def fetch_quant_features(self, symbol: str, lookback_days: int = 30) -> pd.DataFrame:
        """Retrieves pre-calculated features for UI visualization."""
        conn = sqlite3.connect(self.db_path)
        query = """
            SELECT * FROM quant_features
            WHERE Symbol = ?
            ORDER BY Date DESC LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(symbol, lookback_days))
        conn.close()
        return df

if __name__ == "__main__":
    engine = NEPSEDatabaseEngine()
    print("Feature Store Infrastructure Initialized.")
