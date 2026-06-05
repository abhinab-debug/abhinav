import numpy as np
import pandas as pd
from hmmlearn import hmm
from typing import Tuple, List, Generator
from scipy import stats

class BayesianLikelihoodEngine:
    """Phase 1: Kinetic Cold-Start (Days 1-14) - Bayesian Update Engine."""
    def __init__(self, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.alpha = prior_alpha
        self.beta = prior_beta

    def update(self, success: bool):
        """Updates the posterior based on new transaction evidence."""
        if success:
            self.alpha += 1
        else:
            self.beta += 1

    def get_posterior_mean(self) -> float:
        """Calculates the expected probability of the signal being valid."""
        return self.alpha / (self.alpha + self.beta)

class GaussianHMMRegime:
    """
    Gaussian Hidden Markov Model (HMM) for Regime Classification.
    3 States: Mean Reverting, Trending, Illiquid/Halt.
    """
    def __init__(self):
        self.model = hmm.GaussianHMM(
            n_components=3,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        self.is_trained = False

    def fit(self, features: np.ndarray):
        """Fits the HMM against Kyle's Lambda, SVKE, and VPIN features."""
        if len(features) < 20:
            return
        self.model.fit(features)
        self.is_trained = True

    def predict_state(self, features: np.ndarray) -> np.ndarray:
        """Predicts the hidden market regime for the given features."""
        if not self.is_trained:
            return np.zeros(len(features))
        return self.model.predict(features)

class PurgedTimeSeriesSplit:
    """
    Purged Walk-Forward Cross Validation.
    Drops a 2-day transaction window between train and test sets to suppress leakage.
    """
    def __init__(self, n_splits: int = 5, purge_window_days: int = 2):
        self.n_splits = n_splits
        self.purge_window_days = purge_window_days

    def split(self, df: pd.DataFrame) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generates purged train/test indices based on Date column."""
        unique_dates = sorted(df['Date'].unique())
        n_dates = len(unique_dates)

        if n_dates < self.n_splits + self.purge_window_days + 1:
            yield np.arange(len(df)), np.arange(len(df))
            return

        test_size = n_dates // (self.n_splits + 1)

        for i in range(self.n_splits):
            train_end_idx = (i + 1) * test_size
            test_start_idx = train_end_idx + self.purge_window_days
            test_end_idx = test_start_idx + test_size

            if test_end_idx > n_dates:
                test_end_idx = n_dates

            train_dates = unique_dates[:train_end_idx]
            test_dates = unique_dates[test_start_idx:test_end_idx]

            if not test_dates:
                break

            train_indices = df[df['Date'].isin(train_dates)].index.values
            test_indices = df[df['Date'].isin(test_dates)].index.values

            yield train_indices, test_indices

class GNNSectorCascades:
    """
    Causal t-1 VAR Node Networks.
    Measures how kinetic velocity in sector bellwethers bleeds into mid-cap peers.
    """
    @staticmethod
    def calculate_causal_multiplier(bellwether_velocity: np.ndarray, peer_velocity: np.ndarray) -> float:
        """Calculates the time-lagged causal tracking multiplier using OLS."""
        if len(bellwether_velocity) < 5 or len(peer_velocity) < 5:
            return 1.0

        # Align for t-1 lag
        # Peer(t) = alpha + beta * Bellwether(t-1)
        y = peer_velocity[1:]
        x = bellwether_velocity[:-1]

        if np.std(x) == 0:
            return 1.0

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        # Return slope as the causal tracking multiplier
        return float(slope)

class MLStateMachine:
    """
    NEPSE Sovereign Quantitative Inference Terminal (v15.0)
    File 4: Predictive Inference & Regime State Machine.
    """
    def __init__(self, data_length_days: int):
        self.data_length_days = data_length_days
        self.bayesian_engine = BayesianLikelihoodEngine()
        self.hmm_engine = GaussianHMMRegime()

    def route_inference(self, features: np.ndarray) -> str:
        """
        Dynamic Cold-Start Switcher.
        Routes to Bayesian engine if < 15 days, otherwise unlocks HMM regime.
        """
        if self.data_length_days < 15:
            # Cold-Start Regime
            prob = self.bayesian_engine.get_posterior_mean()
            return f"KINETIC_COLD_START: BAYESIAN_PROB={prob:.4f}"
        else:
            # Sovereign Autonomy Regime
            if not self.hmm_engine.is_trained:
                self.hmm_engine.fit(features)

            current_regime = self.hmm_engine.predict_state(features[-1:])
            regime_map = {0: "MEAN_REVERTING", 1: "TRENDING", 2: "ILLIQUID_HALT"}
            return f"SOVEREIGN_AUTONOMY: REGIME={regime_map.get(current_regime[0], 'UNKNOWN')}"

if __name__ == "__main__":
    print("ML State Machine (v15.0) Initialized.")
