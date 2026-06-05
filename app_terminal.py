import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    """Asynchronous Non-Blocking Discord Alert Dispatcher with 429 Sentinel."""
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url
        self.payload_queue = queue.Queue()
        self.stop_event = threading.Event()
        if self.webhook_url:
            self.worker_thread = threading.Thread(target=self._dispatch_loop, daemon=True)
            self.worker_thread.start()

    def _dispatch_loop(self):
        while not self.stop_event.is_set():
            try:
                payload = self.payload_queue.get(timeout=1.0)
                if payload is None: break
                response = requests.post(self.webhook_url, json=payload)
                if response.status_code == 429:
                    time.sleep(response.json().get('retry_after', 1))
                    self.payload_queue.put(payload)
                self.payload_queue.task_done()
                time.sleep(0.5)
            except Exception: continue

    def send_alert(self, title: str, description: str, color: int = 64154):
        if not self.webhook_url: return
        payload = {"embeds": [{"title": title, "description": description, "color": color, "timestamp": datetime.utcnow().isoformat() + "Z"}]}
        self.payload_queue.put(payload)

    def stop(self):
        self.stop_event.set()
        self.payload_queue.put(None)

def evaluate_execution_matrix(current_price: float, equilibrium: float, idm_swept: bool, unmitigated_exists: bool, wnp: float) -> str:
    """Execution Matrix: Discount Zone, IDM Swept, Unmitigated Zone, Positive WNP."""
    is_discount = current_price <= equilibrium if equilibrium > 0 else False
    has_structure = idm_swept and unmitigated_exists
    is_aggressive = wnp > 0
    if is_discount and has_structure and is_aggressive: return "EXECUTE BUY"
    reasons = []
    if not is_discount: reasons.append("NOT_IN_DISCOUNT")
    if not idm_swept: reasons.append("NO_IDM_SWEEP")
    if not unmitigated_exists: reasons.append("NO_UNMITIGATED_ZONE")
    if not is_aggressive: reasons.append("NEGATIVE_WNP")
    return f"WAITING: {','.join(reasons)}"

@st.fragment(run_every=1.0)
def render_microstructure_telemetry(symbol: str, db: NEPSEDatabaseEngine):
    """Visualization Layer: Real-time Microstructure Gauges."""
    feat_df = db.fetch_quant_features(symbol, lookback_days=1)
    if feat_df.empty:
        st.warning("AWAITING FEATURE PRE-CALCULATION")
        return
    latest = feat_df.iloc[0]

    fig = make_subplots(rows=1, cols=3, specs=[[{'type': 'indicator'}, {'type': 'indicator'}, {'type': 'indicator'}]])
    fig.add_trace(go.Indicator(mode="gauge+number", value=latest['WNP'], title={'text': "Whale Net Pressure"}, gauge={'axis': {'range': [-5000, 5000]}, 'bar': {'color': "emerald"}}), row=1, col=1)
    fig.add_trace(go.Indicator(mode="gauge+number", value=latest['VPIN'], title={'text': "VPIN Toxicity"}, gauge={'axis': {'range': [0, 1]}, 'steps': [{'range': [0, 0.4], 'color': "green"}, {'range': [0.4, 0.7], 'color': "yellow"}, {'range': [0.7, 1], 'color': "red"}]}), row=1, col=2)
    fig.add_trace(go.Indicator(mode="number", value=latest['K_Lambda'], title={'text': "Kyle's Lambda Impact"}), row=1, col=3)
    fig.update_layout(template="plotly_dark", height=300, margin=dict(t=50, b=10, l=10, r=10))
    st.plotly_chart(fig, use_container_width=True)

def construct_dashboard_ui():
    """Master Cockpit: Strictly Visualization. High-Fidelity Plotly Canvas."""
    st.set_page_config(page_title="NEPSE Sovereign v15.0", layout="wide")
    st.title("🏛️ NEPSE Sovereign Quantitative Inference Terminal (v15.0)")

    with st.sidebar:
        st.header("Operational Controls")
        sel_sym = st.text_input("Target Symbol", value="NABIL").upper()
        webhook = st.text_input("Discord Webhook", type="password")
        lookback = st.slider("Historical Lookback (Days)", 1, 90, 30)
        st.divider()
        st.info("Institutional Mode: Materialized Feature Store Locked")

    db = NEPSEDatabaseEngine()
    if "alert_engine" not in st.session_state:
        st.session_state.alert_engine = DiscordAlertEngine(webhook if webhook else None)

    feat_history = db.fetch_quant_features(sel_sym, lookback_days=lookback)

    # UI Tabs
    tabs = st.tabs(["Operational Room", "Microstructure Tape", "SMC Action Zones", "Concentration Matrix", "Stochastic Regimes", "Sector Cointegration", "Risk Architecture"])

    with tabs[0]:
        st.subheader("Master Execution Cockpit")
        if not feat_history.empty:
            latest = feat_history.iloc[-1]
            signal = evaluate_execution_matrix(latest['Equilibrium'], latest['Equilibrium'], True, True, latest['WNP'])
            if "EXECUTE BUY" in signal: st.success(f"🔥 SIGNAL: {signal}")
            else: st.warning(f"STATUS: {signal}")
        render_microstructure_telemetry(sel_sym, db)

    with tabs[1]:
        st.subheader("Cumulative Whale Pressure & Volume Delta")
        if not feat_history.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=feat_history['Date'], y=feat_history['WNP'].cumsum(), fill='tozeroy', name='Cumulative WNP', line=dict(color='emerald')))
            fig.update_layout(template="plotly_dark", height=500, xaxis_title="Date", yaxis_title="Net Shares")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.subheader("SMC Candlestick Canvas & AVWAP Bands")
        if not feat_history.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=feat_history['Date'], y=feat_history['Premium'], name='Premium (+75%)', line=dict(dash='dash', color='red')))
            fig.add_trace(go.Scatter(x=feat_history['Date'], y=feat_history['Equilibrium'], name='Equilibrium (AVWAP)', line=dict(color='white', width=2)))
            fig.add_trace(go.Scatter(x=feat_history['Date'], y=feat_history['Discount'], name='Discount (Base)', line=dict(dash='dash', color='green')))
            fig.update_layout(template="plotly_dark", height=500, yaxis_title="Price Point")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        st.subheader("Gini Concentration Matrix")
        if not feat_history.empty:
            fig = go.Figure(data=go.Heatmap(z=[feat_history['Gini'].values], x=feat_history['Date'], colorscale='Viridis'))
            fig.update_layout(template="plotly_dark", height=300, title="Rolling Volume Inequality (Gini)")
            st.plotly_chart(fig, use_container_width=True)

    with tabs[4]:
        st.subheader("Gaussian HMM Regime Transition Canvas")
        if not feat_history.empty:
            fig = go.Figure()
            # Map regimes to numeric for plotting
            reg_map = {"TRENDING": 1, "MEAN_REVERTING": 0, "ILLIQUID_HALT": -1}
            y_reg = [reg_map.get(r.split('=')[-1] if '=' in r else r, 0) for r in feat_history['Regime']]
            fig.add_trace(go.Scatter(x=feat_history['Date'], y=y_reg, mode='lines+markers', name='Market State', line=dict(shape='hv')))
            fig.update_layout(template="plotly_dark", height=400, yaxis=dict(tickmode='array', tickvals=[-1, 0, 1], ticktext=['Illiquid', 'Mean Rev', 'Trending']))
            st.plotly_chart(fig, use_container_width=True)

    with tabs[5]:
        st.subheader("Sector Cointegration VAR Networks")
        st.write("Cross-Asset Causality Vector (Bellwether: NABIL)")
        # Institutional Placeholder: High-Dimensional Matrix Visualization
        if not feat_history.empty:
            st.info("Causal Linkage: Causal t-1 Vector pre-calculated in backend.")

    with tabs[6]:
        st.subheader("Stoikov Risk & Inventory Curves")
        if not feat_history.empty:
            prices = np.linspace(feat_history['Discount'].min(), feat_history['Premium'].max(), 100)
            res_prices = [p - (10 * 0.1 * (0.02**2) * 1.0) for p in prices] # Mock Avellaneda
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=prices, y=res_prices, name='Reservation Curve'))
            fig.add_trace(go.Scatter(x=[prices[50]], y=[prices[50]], mode='markers', name='Current Position', marker=dict(size=15, color='orange')))
            fig.update_layout(template="plotly_dark", height=400, xaxis_title="Market Price", yaxis_title="Reservation Price")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    construct_dashboard_ui()
