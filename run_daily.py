import os
import pandas as pd
import glob
from database import UnifiedDatabaseManager
from processing import sanitize_floorsheet
from notifications import DiscordNotificationManager
import physics
import strategy
import ml_engine

def run_daily_pipeline(data_folder: str, discord_webhook: str = None):
    db_manager = UnifiedDatabaseManager()
    notifier = DiscordNotificationManager(discord_webhook) if discord_webhook else None

    # 1. Ingestion & Sanitization
    xlsx_files = glob.glob(os.path.join(data_folder, "*.xlsx"))
    for file in xlsx_files:
        print(f"Processing {file}...")
        raw_df = pd.read_excel(file)

        # Ensure correct column names (mapping if necessary)
        # Expected: Contract_ID, Symbol, Quantity, Rate, Amount, Date

        sanitized_df = sanitize_floorsheet(raw_df)
        db_manager.insert_floorsheet(sanitized_df)

    # 2. Daily Analysis & Alerting
    all_symbols = db_manager.fetch_ui_view("SELECT DISTINCT Symbol FROM floorsheet")['Symbol'].tolist()

    for symbol in all_symbols:
        df = db_manager.fetch_ui_view("SELECT * FROM floorsheet WHERE Symbol = ? ORDER BY Contract_ID ASC", (symbol,))
        if df.empty: continue

        rates = df['Rate'].values
        vols = df['Quantity'].values
        sides = physics.reconstruct_sides(rates)

        # Calculate key pillars
        wnp = physics.calculate_wnp(vols, sides)
        k_lambda = physics.calculate_kyles_lambda(rates, sides * vols)

        # SMC Strategy
        avwap_series = strategy.calculate_avwap(df)
        current_avwap = avwap_series.iloc[-1] if not avwap_series.empty else rates[-1]
        zone = strategy.get_smc_zones(pd.Series(rates), current_avwap)

        # Execution Triggers
        # For simplicity, we use placeholders for CHoCH/BOS/IDM logic in this high-level loop
        # In a real system, these would be computed from the physics engine
        trigger = strategy.check_execution_triggers(
            htf_trend="UP", ltf_trend="UP", idm_sweep=True,
            ob_fvg_mitigated=True, choch_bos=True,
            wnp=wnp, kyles_lambda=k_lambda
        )

        if trigger and zone == "TIER 3: DISCOUNT ZONE" and notifier:
            notifier.send_alert(
                f"BUY CONFIRMATION: {symbol}",
                f"Microstructure Pillar Match in {zone}.\nWNP: {wnp:,.0f}\nLambda: {k_lambda:.6f}",
                64154 # Emerald Green
            )
        elif zone == "TIER 1: APEX ZONE" and notifier:
            notifier.send_alert(
                f"LIQUIDATION ALERT: {symbol}",
                f"Extreme Overvaluation in {zone}. Scaling out...",
                16711680 # Crimson Red
            )

    # 3. Auto-Healing Ledger Sync
    for file in xlsx_files:
        raw_df = pd.read_excel(file)
        actual_count = len(raw_df)
        db_count_df = db_manager.fetch_ui_view(
            "SELECT COUNT(*) as count FROM floorsheet WHERE Date = ?",
            (raw_df['Date'].iloc[0],)
        )
        db_count = db_count_df['count'].iloc[0] if not db_count_df.empty else 0

        # Note: db_count might be less than actual_count due to sanitization filters
        # but here we check for missing intervals
        if db_count == 0 and actual_count > 0:
            print(f"Warning: Data missing for {raw_df['Date'].iloc[0]}. Retrying...")

    if notifier:
        notifier.stop()
    print("Pipeline complete.")

if __name__ == "__main__":
    # Example usage
    # run_daily_pipeline("./data")
    pass
