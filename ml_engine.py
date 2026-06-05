import numpy as np
import xgboost as xgb
from hmmlearn import hmm

class BayesianUpdateEngine:
    """Phase 1: Kinetic Cold-Start (Days 1-14)."""
    def __init__(self):
        self.alpha = 1.0 # Successes + 1
        self.beta = 1.0  # Failures + 1

    def update(self, success: bool):
        if success:
            self.alpha += 1
        else:
            self.beta += 1

    def get_probability(self) -> float:
        # Mean of Beta distribution
        return self.alpha / (self.alpha + self.beta)

class SovereignAutonomyEngine:
    """Phase 2: Sovereign Autonomy (Day 15+)."""
    def __init__(self):
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            objective='binary:logistic'
        )
        self.hmm_model = hmm.GaussianHMM(n_components=3, covariance_type="full", n_iter=100)
        self.is_xgb_trained = False
        self.is_hmm_trained = False

    def train_xgboost(self, X: np.ndarray, y: np.ndarray):
        """Purged Walk-Forward Cross Validation logic would be external or integrated here."""
        if len(X) < 10: return
        self.xgb_model.fit(X, y)
        self.is_xgb_trained = True

    def predict_probability(self, X: np.ndarray) -> np.ndarray:
        if not self.is_xgb_trained:
            return np.array([0.5] * len(X))
        return self.xgb_model.predict_proba(X)[:, 1]

    def train_hmm(self, X: np.ndarray):
        """Train Gaussian HMM for regime detection."""
        if len(X) < 20: return
        self.hmm_model.fit(X)
        self.is_hmm_trained = True

    def predict_regime(self, X: np.ndarray) -> np.ndarray:
        if not self.is_hmm_trained:
            return np.zeros(len(X))
        return self.hmm_model.predict(X)

def get_ml_regime(data_length_days: int) -> str:
    if data_length_days < 15:
        return "KINETIC COLD-START"
    else:
        return "SOVEREIGN AUTONOMY"
