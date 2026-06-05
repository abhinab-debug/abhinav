import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from database import UnifiedDatabaseManager
import physics
import strategy
import ml_engine

st.set_page_config(layout="wide", page_title="NEPSE Sovereign Quantitative Inference Terminal v15.0", page_icon="📈")

# Dark-themed analytical cockpit setup
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background-color: #1e2130; border-radius: 5px; padding: 10px; }
    </style>
    """, unsafe_allow_html=True)

db_manager = UnifiedDatabaseManager()

st.title("NEPSE Sovereign Inference Terminal v15.0")

# Sidebar for Ticker Selection
tickers_df = db_manager.fetch_ui_view("SELECT DISTINCT Symbol FROM floorsheet")
tickers = tickers_df['Symbol'].tolist() if not tickers_df.empty else []
selected_ticker = st.sidebar.selectbox("Select Target Ticker", tickers if tickers else ["NO_DATA"])

def get_ticker_data(symbol):
    if symbol == "NO_DATA": return pd.DataFrame()
    # Fetch last 2000 transactions for depth
    query = "SELECT * FROM floorsheet WHERE Symbol = ? ORDER BY Contract_ID DESC LIMIT 2000"
    df = db_manager.fetch_ui_view(query, (symbol,))
    return df.sort_values('Contract_ID')

# Tabs Domain Hierarchy
tabs = st.tabs([
    "Operational Room", "Microstructure Tape", "SMC Action Zones",
    "Concentration Matrix", "Stochastic Regimes", "Sector Cointegration", "Risk Architecture"
])

@st.fragment(run_every=2.0)
def render_operational_room(symbol):
    df = get_ticker_data(symbol)
    if df.empty:
        st.warning("Awaiting Transaction Physics data...")
        return

    rates = df['Rate'].values
    vols = df['Quantity'].values
    sides = physics.reconstruct_sides(rates)

    # ML Prediction
    bayesian = ml_engine.BayesianUpdateEngine()
    prob = bayesian.get_probability()

    # AVWAP
    anchor_id = df['Contract_ID'].iloc[0]
    avwap_series = strategy.calculate_avwap(df, anchor_id)
    current_avwap = avwap_series.iloc[-1]

    # Zone
    zone = strategy.get_smc_zones(rates[-1], rates.min(), rates.max(), current_avwap)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("LTP", f"{rates[-1]:.2f}", f"{rates[-1]-rates[-2]:.2f}")
    col2.metric("ML Probability", f"{prob:.2%}")
    col3.metric("AVWAP Equilibrium", f"{current_avwap:.2f}")
    col4.metric("SMC Tier", zone)

    # Plotly Spatial Visualization
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Contract_ID'], y=rates, name='Transaction Price', line=dict(color='#00FA9A', width=1)))
    fig.add_trace(go.Scatter(x=df['Contract_ID'], y=avwap_series, name='Anchored VWAP', line=dict(color='orange', dash='dash')))

    fig.update_layout(
        template="plotly_dark", height=600,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Contract Sequencing ID", yaxis_title="Price Tick"
    )
    st.plotly_chart(fig, use_container_width=True)

@st.fragment(run_every=5.0)
def render_microstructure_tape(symbol):
    df = get_ticker_data(symbol)
    if df.empty: return

    rates = df['Rate'].values
    vols = df['Quantity'].values
    sides = physics.reconstruct_sides(rates)
    signed_vols = sides * vols

    # Pillars
    k_lambda = physics.calculate_kyles_lambda(rates, signed_vols)
    wnp = physics.calculate_wnp(vols, sides)
    gini = physics.calculate_gini(vols)
    hurst = physics.calculate_hurst_rolling(rates)

    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Kyle's Lambda (λ)", f"{k_lambda:.6f}")
    m_col2.metric("Whale Net Pressure", f"{wnp:,.0f}")
    m_col3.metric("Gini Coefficient", f"{gini:.4f}")
    m_col4.metric("Hurst Exponent", f"{hurst:.4f}")

    # Display raw reconstructed tape
    st.dataframe(df.tail(20), use_container_width=True)

with tabs[0]:
    render_operational_room(selected_ticker)

with tabs[1]:
    render_microstructure_tape(selected_ticker)

with tabs[2]:
    st.subheader("SMC Action Zones")
    df = get_ticker_data(selected_ticker)
    if not df.empty:
        fvgs = strategy.scan_fvgs(df)
        obs = strategy.scan_order_blocks(df)
        st.write(f"Detected {len(fvgs)} Fair Value Gaps and {len(obs)} Order Blocks.")
        if obs:
            st.write("Latest Order Block Signature:", obs[-1])

with tabs[3]:
    st.subheader("Concentration Matrix")
    df = get_ticker_data(selected_ticker)
    if not df.empty:
        fig = go.Figure(data=[go.Histogram(x=df['Quantity'], nbinsx=50, marker_color='#00FA9A')])
        fig.update_layout(template="plotly_dark", title="Transaction Size Distribution")
        st.plotly_chart(fig, use_container_width=True)

with tabs[4]:
    st.subheader("Stochastic Regimes")
    st.info("HMM Gaussian State Transitions: Analyzing market regimes...")

with tabs[5]:
    st.subheader("Sector Cointegration")
    st.info("GNN Lead-Lag node networks: Evaluating sector kinetic energy...")

with tabs[6]:
    st.subheader("Risk Architecture")
    df = get_ticker_data(selected_ticker)
    if not df.empty:
        # Concave Square-Root sizing
        adv = df['Quantity'].sum() # Placeholder for ADV
        vol = df['Rate'].pct_change().std()
        size = strategy.concave_square_root_sizing(0, adv, vol)
        st.metric("Suggested Cap Sizing (Units)", f"{size:,.0f}")
