import pandas as pd
import numpy as np
from database_engine import NEPSEDatabaseEngine
from microstructure_physics import MicrostructurePhysics
from smc_spatial_geometry import SMCScanner
from ml_state_machine import MLStateMachine

class QuantitativeFeatureBridge:
    """
    Backend ETL Calculation Bridge.
    Vectorized computation of microstructure and SMC features for persistence.
    """
    def __init__(self, db_engine: NEPSEDatabaseEngine):
        self.db = db_engine
        self.physics = MicrostructurePhysics()
        self.smc = SMCScanner()

    def generate_daily_features(self, symbol: str, lookback_days: int = 90):
        """Processes historical ledger data into pre-calculated quant vectors."""
        print(f"Executing Feature Engineering Matrix for: {symbol}")

        # 1. Extraction: High-fidelity ledger stream
        df = self.db.fetch_transaction_stream(symbol, lookback_days=lookback_days)
        if df.empty:
            return

        # 2. Transformation: Vectorized compute blocks per session
        unique_dates = sorted(df['Date'].unique())
        daily_features = []

        for date in unique_dates:
            day_df = df[df['Date'] == date].copy()
            if day_df.empty: continue

            # Physics Transcription
            sides = self.physics.reconstruct_tick_test(day_df)
            vols = day_df['Quantity'].values
            rates = day_df['Rate'].values
            signed_vols = sides.values * vols

            wnp = self.physics.calculate_wnp(vols, sides.values)
            gini = self.physics.calculate_gini_coefficient(vols)
            vpin = self.physics.calculate_vpin(vols, sides.values)
            k_lambda = self.physics.calculate_kyles_lambda(rates, signed_vols)

            # Kinetic Energy Proxy
            price_velocity = np.diff(rates, prepend=rates[0])
            svke = 0.5 * np.sum(signed_vols * (price_velocity**2))

            # SMC Spatial Projection
            full_until_now = df[df['Date'] <= date]
            bars = self.smc.generate_information_tick_bars(full_until_now)
            avwap_data = self.smc.calculate_avwap_equilibrium(full_until_now, bars)

            # ML State Inference (Cold-start safety enabled)
            ml_engine = MLStateMachine(data_length_days=len(unique_dates))
            hmm_feat = np.column_stack([
                np.full(len(rates), k_lambda),
                0.5 * signed_vols * (price_velocity**2),
                np.full(len(rates), vpin)
            ])
            regime = ml_engine.route_inference(hmm_feat)

            daily_features.append({
                'Symbol': symbol,
                'Date': date,
                'VPIN': float(vpin),
                'WNP': float(wnp),
                'Gini': float(gini),
                'K_Lambda': float(k_lambda),
                'SVKE': float(svke),
                'Regime': str(regime),
                'Equilibrium': float(avwap_data.get('Equilibrium', 0)),
                'Premium': float(avwap_data.get('Premium', 0)),
                'Discount': float(avwap_data.get('Discount', 0))
            })

        # 3. Loading: Upsert to Quant Feature Store
        if daily_features:
            feature_df = pd.DataFrame(daily_features)
            self.db.upsert_quant_features(feature_df)
            print(f"Institutional Persistence Complete: {len(daily_features)} vectors stored.")

if __name__ == "__main__":
    db = NEPSEDatabaseEngine()
    bridge = QuantitativeFeatureBridge(db)
    # bridge.generate_daily_features("NABIL")
