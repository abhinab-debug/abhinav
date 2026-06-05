import os
from database_engine import NEPSEDatabaseEngine
from data_loader import BulkDataLoader

def initialize_system():
    """
    NEPSE Sovereign Quantitative Inference Terminal (v15.0)
    Convenience Script to initialize and populate the local database.
    """
    print("🏛️ INITIALIZING NEPSE SOVEREIGN TERMINAL (v15.0)...")

    # 1. Initialize Engine
    db_engine = NEPSEDatabaseEngine()

    # 2. Check for raw_data directory
    if not os.path.exists("./raw_data"):
        os.makedirs("./raw_data")
        print("Folder './raw_data/' created. Please place your floorsheet files there.")
        return

    # 3. Run Ingestion Pipeline
    print("Running Bulk Data Ingestion...")
    loader = BulkDataLoader(db_engine)
    loader.run_ingestion_pipeline()

    print("\n✅ SYSTEM INITIALIZATION COMPLETE.")
    print("To start the execution cockpit, run:")
    print("   streamlit run app_terminal.py")

if __name__ == "__main__":
    initialize_system()
