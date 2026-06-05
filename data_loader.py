import pandas as pd
import numpy as np
import sqlite3
import os
import glob
from typing import List, Optional
from database_engine import NEPSEDatabaseEngine

class BulkDataLoader:
    """
    NEPSE Sovereign Quantitative Inference Terminal (v15.0)
    File 6: Integrated Bulk Data Ingestion Pipeline (Refactored).
    Directly addresses Schema Mismatch and Memory Optimization.
    """

    def __init__(self, db_engine: NEPSEDatabaseEngine):
        self.db_engine = db_engine
        self.db_path = db_engine.db_path
        self.raw_data_dir = "./raw_data/"

    def _standardize_and_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Force header standardization, 10-paisa rounding, and strict downcasting.
        """
        # 1. Precise Header Mapping
        mapping = {
            'Contract No': 'Contract_ID',
            'Transaction No': 'Contract_ID',
            'id': 'Contract_ID',
            'Stock Symbol': 'Symbol',
            'Symbol': 'Symbol',
            'Vol': 'Quantity',
            'Quantity': 'Quantity',
            'Rate': 'Rate',
            'Price': 'Rate',
            'Amount': 'Amount',
            'Date': 'Date',
            'Time': 'Timestamp'
        }

        # Rename based on existing columns
        rename_map = {col: mapping[col] for col in df.columns if col in mapping}
        df = df.rename(columns=rename_map)

        if df.empty:
            return df

        # 2. Vectorized 10-Paisa Micro-Tick Rounding Structure
        if 'Rate' in df.columns:
            df['Rate'] = (df['Rate'] * 10).round() / 10.0

        # 3. Temporal Purge (Jan 3rd session data - generic)
        if 'Date' in df.columns:
            df['Date_tmp'] = pd.to_datetime(df['Date'])
            df = df[~((df['Date_tmp'].dt.month == 1) & (df['Date_tmp'].dt.day == 3))]
            df = df.drop(columns=['Date_tmp'])

        # 4. Strict Memory Downcasting & Type Enforcement
        if 'Contract_ID' in df.columns:
            df['Contract_ID'] = df['Contract_ID'].astype('int64')
        if 'Quantity' in df.columns:
            df['Quantity'] = df['Quantity'].astype('int32')
        if 'Rate' in df.columns:
            df['Rate'] = df['Rate'].astype('float32')
        if 'Amount' in df.columns:
            df['Amount'] = df['Amount'].astype('float32')

        return df

    def run_ingestion_pipeline(self, chunksize: int = 100000):
        """
        Ingest raw files from ./raw_data/ via memory-safe chunking.
        Ensures zero RAM exhaustion during multi-gigabyte loads.
        """
        search_pattern = os.path.join(self.raw_data_dir, "*.*")
        files = glob.glob(search_pattern)

        if not files:
            print(f"Ingestion Alert: No raw data files detected in {self.raw_data_dir}")
            return

        for file_path in files:
            print(f"Processing File: {file_path}")

            # Streaming Matrix for Memory Protection
            try:
                if file_path.endswith('.csv'):
                    reader = pd.read_csv(file_path, chunksize=chunksize)
                elif file_path.endswith('.txt'):
                    reader = pd.read_csv(file_path, sep='\t', chunksize=chunksize)
                else:
                    # Non-chunkable formats (Excel) are passed directly to engine
                    df = pd.read_excel(file_path)
                    std_df = self._standardize_and_clean(df)
                    self.db_engine.ingest_floorsheet(std_df)
                    continue

                for chunk in reader:
                    std_chunk = self._standardize_and_clean(chunk)
                    # Pointing EXCLUSIVELY to the unified floorsheet table
                    self.db_engine.ingest_floorsheet(std_chunk)

            except Exception as e:
                print(f"Ingestion Error on {file_path}: {e}")

        # Post-Ingestion Closure Routine
        self.establish_high_performance_indexes()

    def establish_high_performance_indexes(self):
        """
        Creates multi-level database indexes on (Symbol, Date) using native SQL.
        Runs database optimizations for near-instantaneous UI queries.
        """
        print("Executing Post-Ingestion Database Optimization...")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            # Unified schema indexing
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol_date_optimized ON floorsheet (Symbol, Date);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_contract_full ON floorsheet (Contract_ID);")

            # Database Tuning
            cursor.execute("ANALYZE;")
            cursor.execute("VACUUM;")

            conn.commit()
            print("Database Optimized. Multi-level Indexes Established.")
        except sqlite3.Error as e:
            print(f"Database Optimization Failure: {e}")
        finally:
            conn.close()

if __name__ == "__main__":
    # Internal component initialization
    engine = NEPSEDatabaseEngine()
    loader = BulkDataLoader(engine)
    loader.run_ingestion_pipeline()
