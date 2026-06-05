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
        print(f"Ingesting {file}...")
        raw_df = pd.read_excel(file)

        # Mandatory sanitization purge
        sanitized_df = sanitize_floorsheet(raw_df)
        db_manager.insert_floorsheet(sanitized_df)

        # Auto-Healing Reconciliation
        # Assuming we can fetch official volume for the symbol/date
        # For each symbol in the day's data:
        symbols = sanitized_df['Symbol'].unique()
        date = sanitized_df['Date'].iloc[0]
        for sym in symbols:
            # In a real scenario, official_vol would be fetched from an API
            # Here we just show the call to the reconciliation logic
            # db_manager.reconcile_volume(sym, date, official_vol)
            pass

    # 2. Sequential Analysis Loop
    all_symbols = db_manager.fetch_ui_view("SELECT DISTINCT Symbol FROM floorsheet")['Symbol'].tolist()

    for symbol in all_symbols:
        df = db_manager.fetch_ui_view("SELECT * FROM floorsheet WHERE Symbol = ? ORDER BY Contract_ID ASC", (symbol,))
        if df.empty: continue

        rates = df['Rate'].values
        vols = df['Quantity'].values
        sides = physics.reconstruct_sides(rates)
        signed_vols = sides * vols

        # Pillar Analysis
        wnp = physics.calculate_wnp(vols, sides)
        k_lambda = physics.calculate_kyles_lambda(rates, signed_vols)

        # SMC Geometry
        anchor_id = df['Contract_ID'].iloc[0]
        avwap_series = strategy.calculate_avwap(df, anchor_id)
        current_avwap = avwap_series.iloc[-1]
        zone = strategy.get_smc_zones(rates[-1], rates.min(), rates.max(), current_avwap)

        # Execution Triggers (Mocking logic for FVG/OB tap)
        is_ob_tapped = len(strategy.scan_order_blocks(df)) > 0
        trigger = strategy.check_execution_triggers(
            is_htf_ltf_coherent=True,
            is_idm_swept=True,
            is_ob_fvg_tapped=is_ob_tapped,
            is_choch=True,
            wnp=wnp,
            kyles_lambda=k_lambda
        )

        if trigger and zone == "TIER 3: DISCOUNT ZONE" and notifier:
            notifier.send_alert(
                f"BUY CONFIRMATION: {symbol}",
                f"Structural Microstructure Alignment in DISCOUNT.\nWNP: {wnp:,.0f} | λ: {k_lambda:.6f}",
                64154 # Emerald Green
            )
        elif zone == "TIER 1: APEX ZONE" and notifier:
            notifier.send_alert(
                f"APEX LIQUIDATION: {symbol}",
                f"Price has reached APEX zone ({rates[-1]:.2f}). Liquidate exposure.",
                16711680 # Crimson Red
            )

    if notifier:
        notifier.stop()
    print("Sequential Execution Complete.")

if __name__ == "__main__":
    # run_daily_pipeline("./data")
    pass
