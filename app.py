import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from database import UnifiedDatabaseManager
import physics
import strategy

st.set_page_config(layout="wide", page_title="NEPSE Sovereign Quantitative Inference Terminal v15.0")

db_manager = UnifiedDatabaseManager()

st.title("NEPSE Sovereign Quantitative Inference Terminal v15.0")

# Sidebar for Ticker Selection
tickers_df = db_manager.fetch_ui_view("SELECT DISTINCT Symbol FROM floorsheet")
tickers = tickers_df['Symbol'].tolist() if not tickers_df.empty else []
selected_ticker = st.sidebar.selectbox("Select Ticker", tickers if tickers else ["No Data"])

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Operational Room",
    "Microstructure Tape",
    "SMC Action Zones",
    "Concentration Matrix",
    "Stochastic Regimes",
    "Sector Cointegration",
    "Risk Architecture"
])

def get_data(ticker):
    if not ticker or ticker == "No Data":
        return pd.DataFrame()
    query = "SELECT * FROM floorsheet WHERE Symbol = ? ORDER BY Contract_ID DESC LIMIT 1000"
    df = db_manager.fetch_ui_view(query, (ticker,))
    if df.empty:
        return pd.DataFrame()
    return df.sort_values('Contract_ID')

@st.fragment(run_every=1.0)
def render_operational_room():
    df = get_data(selected_ticker)
    if df.empty:
        st.warning("No data available.")
        return

    rates = df['Rate'].values
    sides = physics.reconstruct_sides(rates)
    wnp = physics.calculate_wnp(df['Quantity'].values, sides)
    gini = physics.calculate_gini(df['Quantity'].values)

    avwap_series = strategy.calculate_avwap(df)
    current_avwap = avwap_series.iloc[-1] if not avwap_series.empty else rates[-1]
    zone = strategy.get_smc_zones(pd.Series(rates), current_avwap)

    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", rates[-1])
    col2.metric("Whale Net Pressure", f"{wnp:,.0f}")
    col3.metric("Gini Coefficient", f"{gini:.4f}")

    st.subheader(f"Market Regime: {zone}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Contract_ID'], y=rates, name='Price', line=dict(color='white')))
    fig.add_trace(go.Scatter(x=df['Contract_ID'], y=avwap_series, name='AVWAP', line=dict(color='orange', dash='dash')))
    fig.update_layout(template="plotly_dark", height=500, margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig, use_container_width=True)

@st.fragment(run_every=5.0)
def render_microstructure_tape():
    df = get_data(selected_ticker)
    if df.empty: return

    rates = df['Rate'].values
    sides = physics.reconstruct_sides(rates)
    signed_vols = sides * df['Quantity'].values

    k_lambda = physics.calculate_kyles_lambda(rates, signed_vols)
    amihud = physics.calculate_amihud(np.diff(rates)/rates[:-1], df['Amount'].values[1:])
    hurst = physics.calculate_hurst(rates)

    col1, col2, col3 = st.columns(3)
    col1.metric("Kyle's Lambda", f"{k_lambda:.6f}")
    col2.metric("Amihud Index", f"{amihud:.8f}")
    col3.metric("Hurst Exponent", f"{hurst:.4f}")

with tab1:
    render_operational_room()

with tab2:
    render_microstructure_tape()

with tab3:
    st.header("SMC Action Zones")
    st.info("Visualizing Unmitigated Order Blocks and Fair Value Gaps...")

with tab4:
    st.header("Concentration Matrix")
    df = get_data(selected_ticker)
    if not df.empty:
        vols = df['Quantity'].values
        fig = go.Figure(data=[go.Histogram(x=vols, nbinsx=50, marker_color='teal')])
        fig.update_layout(template="plotly_dark", title="Volume Distribution")
        st.plotly_chart(fig, use_container_width=True)

with tab5:
    st.header("Stochastic Regimes")
    st.write("Gaussian Hidden Markov Model (HMM) State Transitions")

with tab6:
    st.header("Sector Cointegration")
    st.write("GNN Lead-Lag node networks")

with tab7:
    st.header("Risk Architecture")
    st.write("EVT-CVaR metrics and Avellaneda-Kelly size parameters")
