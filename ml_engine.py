import numpy as np
import xgboost as xgb
from hmmlearn import hmm

class BayesianUpdateEngine:
    """Phase 1: Kinetic Cold-Start (Days 1-14)."""
    def __init__(self):
        self.prior_win_prob = 0.5
        self.successes = 0
        self.trials = 0

    def update(self, success: bool):
        self.trials += 1
        if success:
            self.successes += 1

    def get_probability(self) -> float:
        # Beta distribution posterior mean
        alpha = 1 + self.successes
        beta = 1 + (self.trials - self.successes)
        return alpha / (alpha + beta)

class SovereignAutonomyEngine:
    """Phase 2: Sovereign Autonomy (Day 15+)."""
    def __init__(self):
        self.xgb_model = xgb.XGBClassifier()
        self.hmm_model = hmm.GaussianHMM(n_components=3, covariance_type="full")
        self.is_trained = False

    def train_xgboost(self, X, y):
        self.xgb_model.fit(X, y)
        self.is_trained = True

    def predict_probability(self, X):
        if not self.is_trained:
            return 0.5
        return self.xgb_model.predict_proba(X)[:, 1]

    def train_hmm(self, X):
        self.hmm_model.fit(X)

    def predict_state(self, X):
        return self.hmm_model.predict(X)

def get_ml_regime(data_length: int):
    if data_length < 15:
        return "COLD-START"
    else:
        return "SOVEREIGN-AUTONOMY"
