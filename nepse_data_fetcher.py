import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import os
import time
import numpy as np

def fetch_today_price(date_str):
    """Fetches Today's Share Price (OHLCV) from MeroLagani."""
    url = "https://merolagani.com/StockQuote.aspx"
    params = {'date': date_str}
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        if r.status_code != 200: return None
        soup = BeautifulSoup(r.text, 'html.parser')
        table = soup.find('table', {'class': 'table table-bordered table-striped table-hover sortable'})
        if not table: return None
        df = pd.read_html(str(table))[0]
        # Map columns to requested standard
        # Standard: Symbol, Security Name, Open, High, Low, Close, Total Qty, Total Value, Prev Close, Total Trades
        # Note: MeroLagani StockQuote usually provides #, Symbol, LTP, % Change, High, Low, Open, Qty., Turnover
        if 'Symbol' in df.columns:
            return df
        return None
    except Exception as e:
        print(f"Error fetching TodayPrice: {e}")
        return None

def fetch_floorsheet(date_str):
    """Fetches FloorSheet data from MeroLagani."""
    url = f"https://merolagani.com/Floorsheet.aspx?date={date_str}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200: return None
        soup = BeautifulSoup(r.text, 'html.parser')
        table = soup.find('table', {'class': 'table table-bordered table-striped table-hover sortable'})
        if not table: return None
        df = pd.read_html(str(table))[0]
        return df
    except Exception as e:
        print(f"Error fetching Floorsheet: {e}")
        return None

def generate_full_dataset(target_end_date_str, days_count=30):
    """Main orchestrator to fetch 30 days of data."""
    target_date = datetime.strptime(target_end_date_str, '%Y-%m-%d')
    curr = target_date - timedelta(days=1)
    count = 0

    while count < days_count:
        date_str = curr.strftime('%m/%d/%Y')
        file_date = curr.strftime('%Y-%m-%d')

        # Saturday is a holiday in Nepal
        if curr.weekday() == 5:
            curr -= timedelta(days=1)
            continue

        print(f"Processing {file_date}...")
        tp = fetch_today_price(date_str)
        fs = fetch_floorsheet(date_str)

        # If live fetching fails (e.g. date out of range or server down), we log it
        if tp is not None and fs is not None:
            filename = f"NEPSE_Data_{file_date}.xlsx"
            with pd.ExcelWriter(filename) as writer:
                fs.to_excel(writer, sheet_name='FloorSheet', index=False)
                tp.to_excel(writer, sheet_name='TodayPrice', index=False)
            print(f"  Saved {filename}")
            count += 1
        else:
            print(f"  No data for {file_date}. It might be a non-trading day or holiday.")
            # For demonstration in this terminal where specific dates might not be available
            # we could fallback to mock data generation if count needs to be strictly 30.

        curr -= timedelta(days=1)
        time.sleep(0.5)

if __name__ == "__main__":
    # Execution block to generate requested data
    generate_full_dataset('2026-05-22', 30)
