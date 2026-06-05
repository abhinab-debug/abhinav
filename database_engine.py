import sqlite3
import pandas as pd
import numpy as np
import re
from typing import List, Optional

class NEPSEDatabaseEngine:
    """
    NEPSE Sovereign Quantitative Inference Terminal (v15.0)
    Core Database Engine: Handles Ingestion, Sanitization, and Gap Healing.
    """

    def __init__(self, db_path: str = "nepse_clean.db"):
        self.db_path = db_path
        self._initialize_infrastructure()

    def _initialize_infrastructure(self):
        """Configures SQLite WAL mode and initializes the high-performance schema."""
        conn = sqlite3.connect(self.db_path, timeout=60.0)
        cursor = conn.cursor()
        # Enable Write-Ahead Logging for concurrent read/write performance
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

        # Optimized Multi-Level Indexing
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol_date ON floorsheet (Symbol, Date);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_contract_seq ON floorsheet (Contract_ID);")

        conn.commit()
        conn.close()

    @staticmethod
    def downcast_memory(df: pd.DataFrame) -> pd.DataFrame:
        """
        Deterministic memory downcasting to optimize terminal processing speed.
        Reduces memory footprint by ~50% for high-frequency transaction parsing.
        """
        fcols = df.select_dtypes('float').columns
        icols = df.select_dtypes('integer').columns

        df[fcols] = df[fcols].astype('float32')
        df[icols] = df[icols].astype('int32')

        return df

    def apply_regulatory_sanitization(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Strict Regulatory Compliance Filter:
        1. Drops Mutual Funds, Promoter Shares, and Debentures via Regex.
        2. Unconditionally drops January 3rd (Corrupted Session Data).
        """
        if df.empty:
            return df

        # Regex Matrix for Instrument Exclusion
        # MF$|SF$ -> Mutual Funds
        # PO$|P$|-PO -> Promoter Shares
        # D\d{2}|BD|D80 -> Debentures/Bonds
        exclusion_regex = r'MF$|SF$|PO$|P$|-PO|D\d{2}|BD|D80'

        df = df[~df['Symbol'].str.contains(exclusion_regex, regex=True, na=False)]

        # Temporal Session Purge (Jan 3rd)
        df['Date_parsed'] = pd.to_datetime(df['Date'])
        df = df[~((df['Date_parsed'].dt.month == 1) & (df['Date_parsed'].dt.day == 3))]
        df = df.drop(columns=['Date_parsed'])

        return df

    @staticmethod
    def apply_10_paisa_rounding(df: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized 10-Paisa Grid Rounding.
        Eliminates floating-point calculation anomalies from exchange floorsheets.
        """
        if 'Rate' in df.columns:
            df['Rate'] = np.round(df['Rate'] * 10) / 10
        return df

    def run_auto_healing_check(self, symbol: str, date: str) -> List[int]:
        """
        Auto-Healing Gap Checker: Verifies continuous sequence order of Contract_IDs.
        Returns a list of missing Contract_IDs that may require re-ingestion.
        """
        conn = sqlite3.connect(self.db_path)
        query = "SELECT Contract_ID FROM floorsheet WHERE Symbol = ? AND Date = ? ORDER BY Contract_ID"
        df = pd.read_sql_query(query, conn, params=(symbol, date))
        conn.close()

        if df.empty:
            return []

        ids = df['Contract_ID'].values
        full_range = np.arange(ids[0], ids[-1] + 1)
        missing_ids = np.setdiff1d(full_range, ids)

        return missing_ids.tolist()

    def ingest_floorsheet(self, raw_df: pd.DataFrame):
        """
        Primary Ingestion Pipeline: Sanitization -> Rounding -> Downcasting -> Insertion.
        """
        # 1. Regulatory Sanitization
        df = self.apply_regulatory_sanitization(raw_df)

        # 2. Vectorized Grid Rounding
        df = self.apply_10_paisa_rounding(df)

        # 3. Deterministic Downcasting
        df = self.downcast_memory(df)

        if df.empty:
            return

        # 4. Multi-Row WAL Insertion
        conn = sqlite3.connect(self.db_path, timeout=60.0)
        try:
            df.to_sql("floorsheet", conn, if_exists="append", index=False, method="multi", chunksize=1000)
        except sqlite3.IntegrityError:
            # Handle duplicates if re-running ingestion for the same IDs
            pass
        finally:
            conn.close()

    def fetch_transaction_stream(self, symbol: str, lookback_days: int = 14) -> pd.DataFrame:
        """
        Retrieves cleansed transaction ledger for analysis.
        Implements Cold-Start lookback logic.
        """
        conn = sqlite3.connect(self.db_path)

        # Determine the date range
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

if __name__ == "__main__":
    # Internal component test logic
    engine = NEPSEDatabaseEngine()
    print("Database Engine Initialized in WAL Mode.")
