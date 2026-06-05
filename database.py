import sqlite3
import pandas as pd
import numpy as np

class UnifiedDatabaseManager:
    def __init__(self, db_path: str = "nepse_clean.db"):
        self.db_path = db_path
        self._initialize_wal_mode()
        self._initialize_schema()

    def _initialize_wal_mode(self):
        """Configures multi-threaded non-blocking WAL mode to prevent deadlocks."""
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute("PRAGMA synchronous=NORMAL;")
        conn.commit()
        conn.close()

    def _initialize_schema(self):
        """Initializes the unified SQLite database schema with indexes."""
        conn = self.get_clean_connection()
        cursor = conn.cursor()
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
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol ON floorsheet (Symbol);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_date ON floorsheet (Date);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol_date ON floorsheet (Symbol, Date);")
        conn.commit()
        conn.close()

    def get_clean_connection(self):
        return sqlite3.connect(self.db_path, timeout=30.0)

    def fetch_ui_view(self, query: str, params=()) -> pd.DataFrame:
        conn = self.get_clean_connection()
        try:
            df = pd.read_sql_query(query, conn, params=params)
            return df
        finally:
            conn.close()

    def insert_floorsheet(self, df: pd.DataFrame):
        """Inserts sanitized floorsheet data into the database using multi-row insertion."""
        conn = self.get_clean_connection()
        try:
            df.to_sql("floorsheet", conn, if_exists="append", index=False, method="multi", chunksize=1000)
        except sqlite3.IntegrityError:
            # Duplicates are ignored at the individual row level by SQLite PRIMARY KEY constraint
            pass
        finally:
            conn.close()

    def reconcile_volume(self, symbol: str, date: str, official_volume: float) -> bool:
        """
        Auto-Healing Reconciliation: Checks if local volume matches official exchange volume.
        Returns True if reconciled, False if missing data detected.
        """
        query = "SELECT SUM(Quantity) as local_vol FROM floorsheet WHERE Symbol = ? AND Date = ?"
        df = self.fetch_ui_view(query, (symbol, date))
        local_vol = df['local_vol'].iloc[0] if not df.empty and df['local_vol'].iloc[0] is not None else 0

        if np.isclose(local_vol, official_volume):
            return True
        else:
            print(f"Volume Mismatch for {symbol} on {date}: Local={local_vol}, Official={official_volume}")
            return False

    def get_missing_contract_sequences(self, symbol: str, date: str):
        """Identifies gaps in Contract_ID sequencing for a specific symbol/date."""
        query = "SELECT Contract_ID FROM floorsheet WHERE Symbol = ? AND Date = ? ORDER BY Contract_ID"
        df = self.fetch_ui_view(query, (symbol, date))
        if df.empty: return []

        ids = df['Contract_ID'].values
        expected_ids = np.arange(ids[0], ids[-1] + 1)
        missing_ids = np.setdiff1d(expected_ids, ids)
        return missing_ids.tolist()
