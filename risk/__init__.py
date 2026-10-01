"""
CityTwin AI — Risk Assessment Package (Unit I: Feedforward Neural Network)
==========================================================================
Estimates urban disruption severity (risk score 0–100 and risk level
LOW/MEDIUM/HIGH/CRITICAL) by fusing current city state observations
with LSTM future traffic predictions.

Pipeline Position
-----------------
  Current CityState (14 features)  +  LSTM Forecast (5-step summary & trends)
                 │                                   │
                 └─────────────────┬─────────────────┘
                                   ▼
                   Risk Feature Builder (23 floats)
                                   ▼
                            RiskMLP Network
                                   ▼
                         RiskPredictor Output:
                { "risk_score": 0–100, "risk_level": "..." }

Public API
----------
  RiskFeatureBuilder             — 23-dim feature fusion builder
  RISK_FEATURE_NAMES             — list of feature names
  RISK_FEATURE_DIM               — constant (23)
  generate_synthetic_risk_target — synthetic disruption severity generator
  classify_risk_level            — score-to-level mapping (LOW/MEDIUM/HIGH/CRITICAL)
  RiskLevel                      — Enum of discrete levels
  RiskMLP                        — PyTorch Feedforward Neural Network
  RiskPredictor                  — High-level inference API
"""

from .feature_builder import (
    RiskFeatureBuilder,
    RISK_FEATURE_NAMES,
    RISK_FEATURE_DIM,
    generate_synthetic_risk_target,
    classify_risk_level,
    RiskLevel,
)
from .risk_model import RiskMLP
from .predictor import RiskPredictor

__all__ = [
    "RiskFeatureBuilder",
    "RISK_FEATURE_NAMES",
    "RISK_FEATURE_DIM",
    "generate_synthetic_risk_target",
    "classify_risk_level",
    "RiskLevel",
    "RiskMLP",
    "RiskPredictor",
]
