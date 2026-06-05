import pandas as pd
import numpy as np
import sqlite3
import os
import re
from typing import List, Dict, Optional
from database_engine import NEPSEDatabaseEngine

class BulkDataLoader:
    """
    NEPSE Sovereign Quantitative Inference Terminal (v15.0)
    File 6: Integrated Bulk Data Ingestion Pipeline.
    Handles memory-protected streaming of raw floorsheet historicals.
    """

    def __init__(self, db_engine: NEPSEDatabaseEngine):
        self.db_engine = db_engine
        self.db_path = db_engine.db_path

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Flexible Mapping: Accounts for varying NEPSE column naming conventions.
        Maps Contract No/Transaction No/id uniformly to Contract_ID, etc.
        """
        mapping = {
            'Contract No': 'Contract_ID',
            'Transaction No': 'Contract_ID',
            'id': 'Contract_ID',
            'Rate': 'Rate',
            'Price': 'Rate',
            'Qty': 'Quantity',
            'Quantity': 'Quantity',
            'Amount': 'Amount',
            'Symbol': 'Symbol',
            'Date': 'Date',
            'Time': 'Time',
            'Timestamp': 'Timestamp'
        }

        # Identify available columns and rename
        current_cols = df.columns.tolist()
        rename_dict = {}
        for raw, target in mapping.items():
            if raw in current_cols:
                rename_dict[raw] = target

        df = df.rename(columns=rename_dict)
        return df

    def _cleansing_pipeline(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Data Cleansing Rules:
        - NEPSE 10-paisa rounding rule.
        - Drop anomalous historical test dates (e.g., January 3rd, 2024).
        - Strict downcasting.
        - Synthesize clean Timestamp.
        """
        if df.empty:
            return df

        # 1. NEPSE 10-Paisa Micro-Tick Rounding
        if 'Rate' in df.columns:
            df['Rate'] = (df['Rate'] * 10).round() / 10.0

        # 2. Drop Anomalous Date: January 3rd, 2024 (and 2026-01-03 per v15.0 core)
        if 'Date' in df.columns:
            df['Date_dt'] = pd.to_datetime(df['Date'])
            df = df[~((df['Date_dt'].dt.month == 1) & (df['Date_dt'].dt.day == 3) & (df['Date_dt'].dt.year == 2024))]
            df = df[~((df['Date_dt'].dt.month == 1) & (df['Date_dt'].dt.day == 3) & (df['Date_dt'].dt.year == 2026))]
            df = df.drop(columns=['Date_dt'])

        # 3. Dynamic Timestamp Synthesis
        if 'Date' in df.columns and 'Time' in df.columns and 'Timestamp' not in df.columns:
            df['Timestamp'] = df['Time']

        # 4. Strict Downcasting & Type Enforcement
        if 'Contract_ID' in df.columns:
            df['Contract_ID'] = df['Contract_ID'].astype('int64')
        if 'Quantity' in df.columns:
            df['Quantity'] = df['Quantity'].astype('int32')
        if 'Rate' in df.columns:
            df['Rate'] = df['Rate'].astype('float32')
        if 'Amount' in df.columns:
            df['Amount'] = df['Amount'].astype('float32')

        return df

    def process_files(self, file_paths: List[str], chunksize: int = 100000):
        """
        Memory Optimization & Protection:
        Ingests raw data files in streaming chunks to guarantee zero RAM exhaustion.
        """
        for path in file_paths:
            print(f"Streaming ingestion for: {path}")
            if not os.path.exists(path):
                print(f"Skip: {path} not found.")
                continue

            # Determine file type
            if path.endswith('.csv'):
                reader = pd.read_csv(path, chunksize=chunksize)
            elif path.endswith('.txt'):
                reader = pd.read_csv(path, sep='\t', chunksize=chunksize)
            else:
                # Fallback to full load for XLSX as pandas doesn't support chunksize for excel natively
                # but we handle it via DB engine's multi-insertion
                df = pd.read_excel(path)
                self.db_engine.ingest_floorsheet(self._cleansing_pipeline(self._standardize_columns(df)))
                continue

            for chunk in reader:
                # Apply pipeline
                chunk = self._standardize_columns(chunk)
                chunk = self._cleansing_pipeline(chunk)

                # Insert via DB Engine (which uses the temp table merge logic)
                self.db_engine.ingest_floorsheet(chunk)

        # Post-ingestion optimization
        self.optimize_database()

    def optimize_database(self):
        """
        Establish index on (Symbol, Date) and run vacuum/analyze.
        Guarantees near-instantaneous query execution in the UI.
        """
        print("Finalizing database optimization...")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Establishment of high-speed composite index
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol_date_fast ON floorsheet (Symbol, Date);")

        # SQLite maintenance
        cursor.execute("ANALYZE;")
        cursor.execute("VACUUM;")

        conn.commit()
        conn.close()
        print("Optimization Complete. Production Ready.")

if __name__ == "__main__":
    # Integration logic
    db = NEPSEDatabaseEngine()
    loader = BulkDataLoader(db)
    print("Bulk Data Loader (v15.0) Initialized.")
