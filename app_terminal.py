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
from scipy import stats

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
                pass # Silent fail to prevent Streamlit noise in background
        except Exception:
            pass

    def send_alert(self, title: str, description: str, color: int = 64154):
        """Pushes an alert to the background queue."""
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
    """Off-Hour Order Imbalance (OLI) for pre-market queue stuffing."""
    if df.empty or 'Timestamp' not in df.columns:
        return 0.0
    pre_market = df[(df['Timestamp'] >= '10:30:00') & (df['Timestamp'] <= '10:45:00')]
    if pre_market.empty:
        return 0.0
    pre_market = pre_market.sort_values('Contract_ID')
    pre_market['diff'] = pre_market['Rate'].diff()
    up_vol = pre_market[pre_market['diff'] > 0]['Quantity'].sum()
    down_vol = pre_market[pre_market['diff'] < 0]['Quantity'].sum()
    daily_open_vol = df[df['Timestamp'] < '11:05:00'].groupby('Date')['Quantity'].sum()
    median_opening_vol = daily_open_vol.median() if not daily_open_vol.empty else 1.0
    return float((up_vol - down_vol) / (median_opening_vol + 1e-9))

class AvellanedaStoikovModel:
    """Inventory Sizing Model for risk-adjusted reservation price."""
    @staticmethod
    def calculate_reservation_price(s: float, q: int, gamma: float, sigma: float, t_remaining: float) -> float:
        return float(s - (q * gamma * (sigma**2) * t_remaining))

class GNNVARProxy:
    """Causal t-1 VAR Proxy for Sector Cascades."""
    @staticmethod
    def calculate_sector_cascade(bellwether_returns: np.ndarray, peer_returns: np.ndarray) -> float:
        if len(bellwether_returns) < 5 or len(peer_returns) < 5:
            return 0.0
        y = peer_returns[1:]
        x = bellwether_returns[:-1]
        if np.std(x) == 0:
            return 0.0
        slope, _, _, _, _ = stats.linregress(x, y)
        return float(slope)

def evaluate_execution_matrix(current_price: float, avwap_bands: Dict, idm_swept: bool, has_fvg: bool, has_ob: bool, wnp: float) -> str:
    """The Buyer Rules: Tier 3 Discount, IDM Swept, Unmitigated Zone, Positive WNP."""
    is_discount = current_price <= avwap_bands.get('Equilibrium', 0)
    has_structure = idm_swept and (has_fvg or has_ob)
    is_aggressive = wnp > 0
    if is_discount and has_structure and is_aggressive:
        return "EXECUTE BUY"
    reasons = []
    if not is_discount: reasons.append("NOT_IN_DISCOUNT")
    if not idm_swept: reasons.append("NO_IDM_SWEEP")
    if not (has_fvg or has_ob): reasons.append("NO_UNMITIGATED_ZONE")
    if not is_aggressive: reasons.append("NEGATIVE_WNP")
    return f"WAITING: {','.join(reasons)}"

@st.fragment(run_every=1.0)
def render_microstructure_telemetry(symbol: str, physics_engine: MicrostructurePhysics, db: NEPSEDatabaseEngine):
    """Seamless sub-rendering for microstructural telemetry."""
    df = db.fetch_transaction_stream(symbol, lookback_days=1)
    if df.empty:
        st.warning(f"AWAITING DATA INGESTION (COLD START) FOR {symbol}")
        return
    sides = physics_engine.reconstruct_tick_test(df)
    vols = df['Quantity'].values
    rates = df['Rate'].values
    wnp = physics_engine.calculate_wnp(vols, sides.values)
    vpin = physics_engine.calculate_vpin(vols, sides.values)
    c1, c2, col3 = st.columns(3)
    c1.metric("Whale Net Pressure", f"{wnp:,.0f}")
    c2.metric("VPIN Toxicity", f"{vpin:.4f}")
    col3.metric("Last Rate", f"{rates[-1]:.1f}")
    st.dataframe(df.tail(10), use_container_width=True)

def construct_dashboard_ui():
    """Master Cockpit Node: 7-Tab UI Architecture with Data Desert Protection."""
    st.set_page_config(page_title="NEPSE Sovereign Terminal v15.0", layout="wide", initial_sidebar_state="expanded")
    st.title("🏛️ NEPSE Sovereign Quantitative Inference Terminal (v15.0)")

    with st.sidebar:
        st.header("Operational Controls")
        selected_symbol = st.text_input("Target Symbol", value="NABIL").upper()
        webhook_url = st.text_input("Discord Webhook", type="password")
        lookback = st.slider("Historical Lookback (Days)", 1, 60, 20)
        st.divider()
        st.info("System Status: Sovereign Autonomy Active")

    db = NEPSEDatabaseEngine()
    physics = MicrostructurePhysics()
    smc = SMCScanner()

    if "alert_engine" not in st.session_state:
        st.session_state.alert_engine = DiscordAlertEngine(webhook_url if webhook_url else None)
    elif st.session_state.alert_engine.webhook_url != webhook_url:
        st.session_state.alert_engine.stop()
        st.session_state.alert_engine = DiscordAlertEngine(webhook_url if webhook_url else None)

    # 1. Initialize Default Fallback Variables (Data Desert Protection)
    df_full = pd.DataFrame()
    wnp = 0.0
    vpin = 0.0
    k_lambda = 0.0
    oli_value = 0.0
    regime_status = "AWAITING DATA INGESTION (COLD START)"
    signal = "WAITING: NO_DATA"
    avwap_data = {"Equilibrium": 0.0, "Premium": 0.0, "Discount": 0.0, "High": 0.0, "Low": 0.0}
    fvgs = pd.DataFrame()
    obs = []
    last_rate = 0.0
    res_price = 0.0

    # 2. Analytics Execution
    df_full = db.fetch_transaction_stream(selected_symbol, lookback_days=lookback)
    if not df_full.empty:
        sides = physics.reconstruct_tick_test(df_full)
        vols = df_full['Quantity'].values
        rates = df_full['Rate'].values
        last_rate = rates[-1]
        signed_vols = sides.values * vols
        wnp = physics.calculate_wnp(vols, sides.values)
        k_lambda = physics.calculate_kyles_lambda(rates, signed_vols)
        vpin = physics.calculate_vpin(vols, sides.values)

        bars = smc.generate_information_tick_bars(df_full)
        fvgs = smc.scan_fvgs(bars)
        obs = smc.scan_idm_ob(bars)
        avwap_data = smc.calculate_avwap_equilibrium(df_full, bars)

        ml_engine = MLStateMachine(data_length_days=df_full['Date'].nunique())
        price_velocity = np.diff(rates, prepend=rates[0])
        svke_proxy = 0.5 * signed_vols * (price_velocity**2)
        hmm_features = np.column_stack([np.full(len(rates), k_lambda), svke_proxy, np.full(len(rates), vpin)])
        regime_status = ml_engine.route_inference(hmm_features)

        oli_value = calculate_oli(df_full)
        idm_swept = len(obs) > 0
        signal = evaluate_execution_matrix(last_rate, avwap_data, idm_swept, not fvgs.empty, idm_swept, wnp)

        as_model = AvellanedaStoikovModel()
        res_price = as_model.calculate_reservation_price(last_rate, 0, 0.1, 0.02, 1.0)

        if signal == "EXECUTE BUY":
            st.session_state.alert_engine.send_alert(f"BUY: {selected_symbol}", f"WNP: {wnp:,.0f} | Regime: {regime_status}")

    # 3. 7-Tab UI Architecture
    tabs = st.tabs(["Operational Room", "Microstructure Tape", "SMC Action Zones", "Concentration Matrix", "Stochastic Regimes", "Sector Cointegration", "Risk Architecture"])

    with tabs[0]:
        st.subheader("Master Execution Cockpit")
        st.info(f"OLI Imbalance: {oli_value:.4f}")
        if "EXECUTE BUY" in signal: st.success(f"🔥 SIGNAL: {signal}")
        else: st.warning(f"STATUS: {signal}")
        render_microstructure_telemetry(selected_symbol, physics, db)

    with tabs[1]:
        st.subheader("Structural Order Flow")
        if not df_full.empty: st.line_chart(df_full.set_index('Contract_ID')['Rate'])
        else: st.write("Awaiting tape data...")

    with tabs[2]:
        st.subheader("Spatial Liquidity Zones")
        st.json(avwap_data)
        if not fvgs.empty: st.dataframe(fvgs)

    with tabs[3]:
        st.subheader("Whale Concentration Matrix")
        st.metric("Whale Net Pressure", f"{wnp:,.0f}")
        if not df_full.empty: st.write("Gini Coefficient:", physics.calculate_gini_coefficient(df_full['Quantity'].values))

    with tabs[4]:
        st.subheader("Gaussian HMM Regimes")
        st.write(f"Current State: {regime_status}")

    with tabs[5]:
        st.subheader("GNN Sector Cascades")
        if not df_full.empty and selected_symbol != "NABIL":
            b_df = db.fetch_transaction_stream("NABIL", lookback_days=lookback)
            if not b_df.empty:
                n = min(len(b_df), len(df_full))
                b_ret = np.diff(b_df['Rate'].values[:n])
                p_ret = np.diff(df_full['Rate'].values[:n])
                mult = GNNVARProxy.calculate_sector_cascade(b_ret, p_ret)
                st.metric("Causal Multiplier (vs NABIL)", f"{mult:.4f}")
        else: st.write("Causal sector linkage analysis active.")

    with tabs[6]:
        st.subheader("Inventory & Capital Protection")
        st.metric("Reservation Price", f"{res_price:.2f}")

if __name__ == "__main__":
    construct_dashboard_ui()
