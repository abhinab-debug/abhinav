import sqlite3
import pandas as pd
import numpy as np
import os
import argparse
import logging
import re
import time
import urllib.request
import urllib.error
from datetime import datetime

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    """usage example: python nepse_backfill_ingest.py --folder ./manual_downloads/ --db nepse_floorsheet.db"""
    parser = argparse.ArgumentParser(description="NEPSE Manual Floorsheet Ingestor")
    parser.add_argument("--folder", required=True, help="Path to folder containing CSV/Excel files")
    parser.add_argument("--db", default="nepse_floorsheet.db", help="Path to SQLite database")
    return parser.parse_args()

def scrape_single_day_safe(date_str, db_path):
    """
    Opt-in scraping helper for a single day.
    Requirements: max 1 request per 3 seconds, max 500 pages, stop on 403/429/503.
    """
    logger.info(f"Safe scraping initiated for {date_str}...")
    req_headers = {'User-Agent': 'Mozilla/5.0'}
    for page in range(1, 501):
        time.sleep(3) # Rate limit
        try:
            url = f"https://example.com/floorsheet?date={date_str}&page={page}"
            req = urllib.request.Request(url, headers=req_headers)
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    pass # Parsing logic omitted
                else:
                    break
        except urllib.error.HTTPError as e:
            if e.code in [403, 429, 503]:
                logger.error(f"Scraper halted: HTTP {e.code}")
                break
            with open(f"debug_{date_str}_p{page}.html", "w") as f:
                f.write(e.read().decode())
            break
        except Exception as e:
            logger.error(f"Scraper error: {e}")
            break

def extract_date_from_filename(filename):
    """Extracts YYYY-MM-DD from filename."""
    match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
    if match:
        return match.group(1)
    return None

def map_columns(df):
    """Maps varied column names to standard schema."""
    mapping = {
        'symbol': ['stock symbol', 'symbol', 'scrip'],
        'buyer_member_id': ['buyer broker', 'buyer', 'buyer broker id', 'buyer_member_id'],
        'seller_member_id': ['seller broker', 'seller', 'seller broker id', 'seller_member_id'],
        'quantity': ['quantity', 'qty', 'vol', 'units'],
        'rate': ['rate', 'price', 'rate/price'],
        'amount': ['amount', 'total amount', 'value'],
        'trade_time': ['time', 'trade time', 'timestamp']
    }

    df.columns = [c.lower().strip() for c in df.columns]
    new_cols = {}
    for target, variations in mapping.items():
        for var in variations:
            if var in df.columns:
                new_cols[var] = target
                break
    return df.rename(columns=new_cols)

def clean_data(df, file_date):
    """Cleans and validates the floorsheet dataframe."""
    df = map_columns(df)

    # Required columns check
    required = ['symbol', 'buyer_member_id', 'seller_member_id', 'quantity', 'rate', 'amount', 'trade_time']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Extract numeric ID from broker columns
    for col in ['buyer_member_id', 'seller_member_id']:
        df[col] = df[col].astype(str).str.extract(r'(\d+)').astype(float)

    # Parse time and combine with date
    def parse_dt(t):
        t = str(t).strip()
        try:
            if len(t) <= 8: # HH:MM:SS
                return datetime.strptime(f"{file_date} {t}", "%Y-%m-%d %H:%M:%S")
            return pd.to_datetime(t)
        except:
            return pd.NaT

    df['trade_time'] = df['trade_time'].apply(parse_dt)

    # Numeric conversion
    for col in ['quantity', 'rate', 'amount']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    initial_count = len(df)
    # Validation filters
    df = df.dropna(subset=['trade_time', 'symbol', 'quantity', 'rate'])
    df = df[(df['quantity'] > 0) & (df['rate'] > 0)]
    df = df[(df['buyer_member_id'] >= 1) & (df['buyer_member_id'] <= 99)]
    df = df[(df['seller_member_id'] >= 1) & (df['seller_member_id'] <= 99)]

    df['buyer_member_id'] = df['buyer_member_id'].astype(int)
    df['seller_member_id'] = df['seller_member_id'].astype(int)
    df['trade_time'] = df['trade_time'].dt.strftime('%Y-%m-%d %H:%M:%S')

    rejected = initial_count - len(df)
    return df[required], rejected

def init_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS floorsheet (
            trade_time TEXT, symbol TEXT, buyer_member_id INTEGER,
            seller_member_id INTEGER, quantity INTEGER, rate REAL,
            amount REAL, source TEXT,
            UNIQUE(trade_time, symbol, buyer_member_id, seller_member_id, quantity, rate)
        )
    """)
    conn.commit()
    return conn

def main():
    args = parse_args()
    if not os.path.exists(args.folder):
        logger.error(f"Folder not found: {args.folder}")
        return

    files = [f for f in os.listdir(args.folder) if f.endswith(('.csv', '.xlsx', '.xls'))]
    if not files:
        logger.info(f"No CSV/Excel files found in {args.folder}")
        return

    conn = init_db(args.db)

    for file in files:
        file_path = os.path.join(args.folder, file)
        file_date = extract_date_from_filename(file)
        if not file_date:
            logger.warning(f"Could not extract date from filename: {file}. Skipping.")
            continue

        try:
            df = pd.read_excel(file_path) if file.endswith(('.xlsx', '.xls')) else pd.read_csv(file_path)
            if len(df) > 100000:
                logger.warning(f"File {file} exceeds 100,000 rows. Rejecting.")
                continue

            clean_df, rejected = clean_data(df, file_date)
            clean_df['source'] = 'manual_backfill'

            parsed_count = len(df)
            inserted = 0
            skipped = 0

            for _, row in clean_df.iterrows():
                try:
                    # Convert row to a format that can be easily checked for uniqueness
                    # or use INSERT OR IGNORE if we want to be faster but less granular in logging
                    # Given the small scale of backfills, row-by-row is fine for logging skipped counts.
                    rdf = row.to_frame().T
                    rdf.to_sql('floorsheet', conn, if_exists='append', index=False)
                    inserted += 1
                except (sqlite3.IntegrityError, pd.errors.DatabaseError):
                    skipped += 1

            logger.info(f"Summary for {file}: Parsed: {parsed_count} | Inserted: {inserted} | Skipped: {skipped} | Rejected: {rejected}")

        except Exception as e:
            logger.error(f"Failed to process {file}: {e}")

    conn.close()

if __name__ == "__main__":
    main()
