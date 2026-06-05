# NEPSE Sovereign Quantitative Inference Terminal (v15.0)

🏛️ **Master Transaction Physics & Systemic Arbitrage Engine for the Nepal Stock Exchange.**

The terminal operates on pure market microstructure, spatial liquidity (SMC), and machine learning regimes. It strictly rejects retail indicators (RSI, MACD, EMA) in favor of high-fidelity transaction analysis.

---

## 🚀 Quick Start: Running Your Own Database

The terminal uses a dynamically generated SQLite database (**nepse_clean.db**) running in high-speed WAL Mode. Follow these steps to initialize and populate your own local environment:

### 1. Prepare Your Data
Place your raw NEPSE floorsheet files (CSV, XLSX, or TXT) into the `./raw_data/` folder.
The system is designed with a **Flexible Mapping Matrix** to automatically handle various exchange header formats (e.g., *Contract No*, *Transaction No*, *Stock Symbol*, *Qty*, *Price*, etc.).

### 2. Run the Ingestion Pipeline
Execute the bulk data loader to process your files, apply 10-paisa rounding, enforce regulatory sanitization, and establish high-performance multi-level indexes.

```bash
python3 data_loader.py
```

### 3. Launch the Cockpit
Once the database is populated, start the Streamlit terminal to visualize the microstructure tape, spatial liquidity zones, and ML regimes.

```bash
streamlit run app_terminal.py
```

---

## 📐 System Architecture Matrix

1.  **`database_engine.py`**: High-performance SQLite ledger with WAL mode and strict merge logic.
2.  **`microstructure_physics.py`**: Vectorized implementation of the 14 microstructure pillars (VPIN, TSRV, Hurst, Kyle's Lambda, etc.).
3.  **`smc_spatial_geometry.py`**: Spatial routing using Information Tick-Bars, Fair Value Gaps (FVG), and Inducement (IDM) sweeps.
4.  **`ml_state_machine.py`**: Dual-regime inference (Bayesian Cold-Start & Gaussian HMM/XGBoost Autonomy).
5.  **`app_terminal.py`**: Master execution cockpit with `@st.fragment` sub-rendering and asynchronous Discord alerting.
6.  **`data_loader.py`**: Integrated bulk ingestion pipeline with memory-safe streaming (chunksize=100000).

---

## ⚡ Mathematical Compliance Guardrails
- **10-Paisa Grid Rounding**: All price data is normalized to eliminate floating-point anomalies.
- **Regulatory Purge**: Automatic removal of Mutual Funds, Promoter Shares, Debentures, and corrupted session data (January 3rd).
- **Zero Placeholders**: 100% of the codebase is functional and production-ready.

---

**Terminal Status**: Sovereign Autonomy Active. Ready for Ingestion.
