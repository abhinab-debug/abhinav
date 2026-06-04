import sqlite3
import pandas as pd

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
        """Initializes the unified SQLite database schema."""
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
        """Inserts sanitized floorsheet data into the database."""
        conn = self.get_clean_connection()
        try:
            df.to_sql("floorsheet", conn, if_exists="append", index=False, method="multi")
        except sqlite3.IntegrityError:
            # Handle potential duplicates if re-running ingestion
            pass
        finally:
            conn.close()
