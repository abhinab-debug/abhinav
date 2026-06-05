import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import time
import queue
import threading
import requests
import json
from datetime import datetime
from typing import Dict, List, Optional

# Core Architecture Imports
from database_engine import NEPSEDatabaseEngine
from microstructure_physics import MicrostructurePhysics
from smc_spatial_geometry import SMCScanner
from ml_state_machine import MLStateMachine

class DiscordAlertEngine:
    """
    Asynchronous Non-Blocking Discord Alert Dispatcher.
    Implements a thread-safe Queue and HTTP 429 Sentinel logic.
    """
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url
        self.payload_queue = queue.Queue()
        self.stop_event = threading.Event()

        if self.webhook_url:
            self.worker_thread = threading.Thread(target=self._dispatch_loop, daemon=True)
            self.worker_thread.start()

    def _dispatch_loop(self):
        """Internal background worker for non-blocking execution."""
        while not self.stop_event.is_set():
            try:
                payload = self.payload_queue.get(timeout=1.0)
                if payload is None:
                    break
                self._send_request(payload)
                self.payload_queue.task_done()
                time.sleep(0.5) # Proactive delay
            except queue.Empty:
                continue

    def _send_request(self, payload: Dict):
        """Executes the POST request with HTTP 429 rate-limit handling."""
        try:
            response = requests.post(self.webhook_url, json=payload)
            if response.status_code == 429:
                retry_after = response.json().get('retry_after', 1)
                time.sleep(retry_after)
                self._send_request(payload) # Recursive retry after stall
            elif response.status_code not in [200, 204]:
                st.error(f"Discord API Error: {response.status_code}")
        except Exception as e:
            st.error(f"Alert Engine Exception: {e}")

    def send_alert(self, title: str, description: str, color: int = 64154):
        """
        Pushes an alert to the background queue.
        Default Color: Emerald Green (64154).
        """
        if not self.webhook_url:
            return

        payload = {
            "embeds": [{
                "title": title,
                "description": description,
                "color": color,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }]
        }
        self.payload_queue.put(payload)

    def stop(self):
        """Gracefully shuts down the background worker."""
        self.stop_event.set()
        self.payload_queue.put(None)
        if hasattr(self, 'worker_thread'):
            self.worker_thread.join()

def calculate_oli(df: pd.DataFrame) -> float:
    """
    Off-Hour Order Imbalance (OLI): Parsing pre-market queue stuffing (10:30 AM - 10:45 AM).
    OLI = (V_overnight_up - V_overnight_down) / V_20-day_median_opening.
    """
    if df.empty or 'Timestamp' not in df.columns:
        return 0.0

    # Extract pre-market data
    pre_market = df[(df['Timestamp'] >= '10:30:00') & (df['Timestamp'] <= '10:45:00')]
    if pre_market.empty:
        return 0.0

    # Infer direction using price diff in pre-market
    pre_market = pre_market.sort_values('Contract_ID')
    pre_market['diff'] = pre_market['Rate'].diff()
    up_vol = pre_market[pre_market['diff'] > 0]['Quantity'].sum()
    down_vol = pre_market[pre_market['diff'] < 0]['Quantity'].sum()

    # Calculate historical median opening volume (Cold-Start safe)
    daily_open_vol = df[df['Timestamp'] < '11:05:00'].groupby('Date')['Quantity'].sum()
    median_opening_vol = daily_open_vol.median() if not daily_open_vol.empty else 1.0

    oli = (up_vol - down_vol) / (median_opening_vol + 1e-9)
    return float(oli)

class AvellanedaStoikovModel:
    """
    Inventory Sizing Model.
    r(s, q, t) = s - q * gamma * sigma^2 * (T - t).
    """
    @staticmethod
    def calculate_reservation_price(s: float, q: int, gamma: float, sigma: float, t_remaining: float) -> float:
        """
        Calculates the reservation price adjusted for inventory risk.
        s: current price, q: inventory position, gamma: risk aversion, sigma: volatility, t: time.
        """
        reservation_price = s - (q * gamma * (sigma**2) * t_remaining)
        return float(reservation_price)

    @staticmethod
    def get_inventory_skew(reservation_price: float, current_price: float) -> float:
        """Returns the directional skew based on reservation vs market price."""
        return float(reservation_price - current_price)

class GNNVARProxy:
    """
    Causal t-1 VAR Proxy logic for Sector Cascades.
    Determines if bellwether velocity is cascading into mid-cap peers.
    """
    @staticmethod
    def calculate_sector_cascade(bellwether_returns: np.ndarray, peer_returns: np.ndarray) -> float:
        """
        Calculates the t-1 causal impact coefficient.
        Returns the causal multiplier.
        """
        if len(bellwether_returns) < 5 or len(peer_returns) < 5:
            return 0.0

        # Peer(t) = C + Beta * Bellwether(t-1)
        y = peer_returns[1:]
        x = bellwether_returns[:-1]

        # Linear Regression for causal linkage
        if np.std(x) == 0:
            return 0.0

        from scipy import stats
        slope, _, _, _, _ = stats.linregress(x, y)
        return float(slope)

def evaluate_execution_matrix(
    current_price: float,
    avwap_bands: Dict,
    idm_swept: bool,
    unmitigated_fvg_exists: bool,
    unmitigated_ob_exists: bool,
    wnp: float
) -> str:
    """
    Execution Matrix (The Buyer Rules):
    Returns "EXECUTE BUY" only if criteria met.
    Criteria: Discount Zone reached, IDM Swept, unmitigated FVG/OB exists, and positive WNP.
    """
    # 1. Spatial Requirement: Price must be in Discount Zone
    # Discount Zone: [leg_low, equilibrium/avwap]
    is_discount = current_price <= avwap_bands.get('Equilibrium', 0)

    # 2. Structural Confirmation
    has_structure = idm_swept and (unmitigated_fvg_exists or unmitigated_ob_exists)

    # 3. Kinetic Confirmation
    is_aggressive = wnp > 0

    if is_discount and has_structure and is_aggressive:
        return "EXECUTE BUY"

    # Trace logic for diagnostics
    reasons = []
    if not is_discount: reasons.append("NOT_IN_DISCOUNT")
    if not idm_swept: reasons.append("NO_IDM_SWEEP")
    if not (unmitigated_fvg_exists or unmitigated_ob_exists): reasons.append("NO_UNMITIGATED_ZONE")
    if not is_aggressive: reasons.append("NEGATIVE_WNP")

    return f"WAITING: {','.join(reasons)}"

@st.fragment(run_every=1.0)
def render_microstructure_telemetry(symbol: str, physics_engine: MicrostructurePhysics, db: NEPSEDatabaseEngine):
    """
    Sub-Rendering microstructural telemetry streams.
    Updates every 1 second without full page resets.
    """
    df = db.fetch_transaction_stream(symbol, lookback_days=1)
    if df.empty:
        st.warning(f"No active transaction stream for {symbol}")
        return

    # Physics Execution
    sides = physics_engine.reconstruct_tick_test(df)
    vols = df['Quantity'].values
    rates = df['Rate'].values
    wnp = physics_engine.calculate_wnp(vols, sides.values)
    vpin = physics_engine.calculate_vpin(vols, sides.values)

    col1, col2, col3 = st.columns(3)
    col1.metric("Whale Net Pressure", f"{wnp:,.0f}")
    col2.metric("VPIN Toxicity", f"{vpin:.4f}")
    col3.metric("Last Rate", f"{rates[-1]:.1f}")

    # Microstructure Tape
    st.dataframe(df.tail(10), use_container_width=True)

def construct_dashboard_ui():
    """
    Constructs the master cockpit node with the 7-Tab UI Architecture.
    """
    st.set_page_config(page_title="NEPSE Sovereign Terminal v15.0", layout="wide")
    st.title("🏛️ NEPSE Sovereign Quantitative Inference Terminal (v15.0)")

    # Sidebar Controls
    with st.sidebar:
        st.header("Operational Controls")
        selected_symbol = st.text_input("Target Symbol", value="NABIL")
        webhook_url = st.text_input("Discord Webhook", type="password")
        lookback = st.slider("Historical Lookback (Days)", 1, 60, 20)
        st.divider()
        st.info("System Status: Sovereign Autonomy Active")

    # Initialize Engine
    db = NEPSEDatabaseEngine()
    physics = MicrostructurePhysics()
    smc = SMCScanner()

    # Persistent Discord Alert Engine in Session State
    if "alert_engine" not in st.session_state:
        st.session_state.alert_engine = DiscordAlertEngine(webhook_url if webhook_url else None)
    elif st.session_state.alert_engine.webhook_url != webhook_url:
        st.session_state.alert_engine.stop()
        st.session_state.alert_engine = DiscordAlertEngine(webhook_url if webhook_url else None)

    # 7-Tab UI Architecture
    tabs = st.tabs([
        "Operational Room",
        "Microstructure Tape",
        "SMC Action Zones",
        "Concentration Matrix",
        "Stochastic Regimes",
        "Sector Cointegration",
        "Risk Architecture"
    ])

    with tabs[0]:
        st.subheader("Master Execution Cockpit")
        render_microstructure_telemetry(selected_symbol, physics, db)

    # Analytics Engine Execution
    df_full = db.fetch_transaction_stream(selected_symbol, lookback_days=lookback)
    if not df_full.empty:
        # 1. Microstructure Physics
        sides = physics.reconstruct_tick_test(df_full)
        vols = df_full['Quantity'].values
        rates = df_full['Rate'].values
        signed_vols = sides.values * vols

        wnp = physics.calculate_wnp(vols, sides.values)
        k_lambda = physics.calculate_kyles_lambda(rates, signed_vols)
        vpin = physics.calculate_vpin(vols, sides.values)

        # 2. SMC Spatial Geometry
        bars = smc.generate_information_tick_bars(df_full)
        fvgs = smc.scan_fvgs(bars)
        obs = smc.scan_idm_ob(bars)
        avwap_data = smc.calculate_avwap_equilibrium(df_full, bars)

        # 3. ML State Machine
        unique_dates = df_full['Date'].nunique()
        ml_engine = MLStateMachine(data_length_days=unique_dates)
        # HMM Features: Lambda, SVKE proxy (Rate diff * vol), VPIN
        price_velocity = np.diff(rates, prepend=rates[0])
        svke_proxy = 0.5 * signed_vols * (price_velocity**2)

        # Construct feature matrix for HMM (using last 50 bars or available)
        hmm_features = np.column_stack([
            np.full(len(rates), k_lambda), # Simplified for terminal display
            svke_proxy,
            np.full(len(rates), vpin)
        ])
        regime_status = ml_engine.route_inference(hmm_features)

        # 4. Regulatory OLI
        oli_value = calculate_oli(df_full)

        # 5. Execution Matrix
        idm_swept = len(obs) > 0 # Simplified detection
        has_fvg = not fvgs.empty
        has_ob = len(obs) > 0
        signal = evaluate_execution_matrix(rates[-1], avwap_data, idm_swept, has_fvg, has_ob, wnp)

        with tabs[0]:
            st.info(f"OLI Imbalance: {oli_value:.4f}")
            if signal == "EXECUTE BUY":
                st.success("🔥 SIGNAL: EXECUTE BUY SEQUENCE")
                # Asynchronous Alert via persistent engine
                st.session_state.alert_engine.send_alert(
                    f"BUY CONFIRMATION: {selected_symbol}",
                    f"Execution Matrix Validated.\nWNP: {wnp:,.0f} | Regime: {regime_status}",
                    64154
                )
            else:
                st.warning(f"STATUS: {signal}")

    with tabs[1]:
        st.subheader("Structural Order Flow")
        if not df_full.empty:
            st.line_chart(df_full.set_index('Contract_ID')['Rate'])

    with tabs[2]:
        st.subheader("Spatial Liquidity Zones")
        if not df_full.empty:
            st.json(avwap_data)
            if not fvgs.empty:
                st.write("Unmitigated Fair Value Gaps:")
                st.dataframe(fvgs)

    with tabs[3]:
        st.subheader("Whale Concentration Matrix")
        if not df_full.empty:
            st.metric("Whale Net Pressure", f"{wnp:,.0f}")
            st.write("Gini Coefficient of Volume:", physics.calculate_gini_coefficient(vols))

    with tabs[4]:
        st.subheader("Gaussian HMM Regimes")
        st.write(f"Current Market State: {regime_status}")

    with tabs[5]:
        st.subheader("GNN Sector Cascades")
        # Placeholder for cross-asset comparison
        st.write("Causal Sector Linkage Analysis Active.")

    with tabs[6]:
        st.subheader("Inventory & Capital Protection")
        # Avellaneda-Stoikov Inventory Logic
        as_model = AvellanedaStoikovModel()
        res_price = as_model.calculate_reservation_price(rates[-1] if not df_full.empty else 0, 0, 0.1, 0.02, 1.0)
        st.metric("Reservation Price", f"{res_price:.2f}")

if __name__ == "__main__":
    construct_dashboard_ui()
