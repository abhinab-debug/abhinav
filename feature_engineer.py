import pandas as pd
import numpy as np
from database_engine import NEPSEDatabaseEngine
from microstructure_physics import MicrostructurePhysics
from smc_spatial_geometry import SMCScanner
from ml_state_machine import MLStateMachine

class QuantitativeFeatureBridge:
    """
    Backend ETL Calculation Bridge.
    Pre-calculates microstructure and SMC vectors to populate the Feature Store.
    """
    def __init__(self, db_engine: NEPSEDatabaseEngine):
        self.db = db_engine
        self.physics = MicrostructurePhysics()
        self.smc = SMCScanner()

    def generate_daily_features(self, symbol: str, lookback_days: int = 60):
        """Processes raw floorsheet data into persistent quant vectors."""
        print(f"Engineered Feature Generation for {symbol}...")

        # 1. Fetch historical raw stream
        df = self.db.fetch_transaction_stream(symbol, lookback_days=lookback_days)
        if df.empty:
            return

        # 2. Iterate by Unique Dates for Daily Aggregation
        unique_dates = sorted(df['Date'].unique())
        daily_features = []

        for date in unique_dates:
            day_df = df[df['Date'] == date].copy()
            if day_df.empty: continue

            # Physics Calculation
            sides = self.physics.reconstruct_tick_test(day_df)
            vols = day_df['Quantity'].values
            rates = day_df['Rate'].values
            signed_vols = sides.values * vols

            wnp = self.physics.calculate_wnp(vols, sides.values)
            gini = self.physics.calculate_gini_coefficient(vols)
            vpin = self.physics.calculate_vpin(vols, sides.values)
            k_lambda = self.physics.calculate_kyles_lambda(rates, signed_vols)

            # SVKE Proxy
            price_velocity = np.diff(rates, prepend=rates[0])
            svke = 0.5 * np.sum(signed_vols * (price_velocity**2))

            # SMC Calculation (using full history for better anchoring)
            full_until_now = df[df['Date'] <= date]
            bars = self.smc.generate_information_tick_bars(full_until_now)
            avwap_data = self.smc.calculate_avwap_equilibrium(full_until_now, bars)

            # ML Regime Classification
            ml_engine = MLStateMachine(data_length_days=len(unique_dates))
            # HMM Features for the specific day
            hmm_feat = np.column_stack([
                np.full(len(rates), k_lambda),
                0.5 * signed_vols * (price_velocity**2),
                np.full(len(rates), vpin)
            ])
            regime = ml_engine.route_inference(hmm_feat)

            daily_features.append({
                'Symbol': symbol,
                'Date': date,
                'VPIN': vpin,
                'WNP': wnp,
                'Gini': gini,
                'K_Lambda': k_lambda,
                'SVKE': svke,
                'Regime': regime,
                'Equilibrium': avwap_data.get('Equilibrium', 0),
                'Premium': avwap_data.get('Premium', 0),
                'Discount': avwap_data.get('Discount', 0)
            })

        # 3. Batch Upsert to Feature Store
        if daily_features:
            feature_df = pd.DataFrame(daily_features)
            self.db.upsert_quant_features(feature_df)
            print(f"Successfully Persisted {len(daily_features)} Feature Vectors.")

if __name__ == "__main__":
    db = NEPSEDatabaseEngine()
    bridge = QuantitativeFeatureBridge(db)
    # Trigger for specific symbol if run standalone
    # bridge.generate_daily_features("NABIL")
