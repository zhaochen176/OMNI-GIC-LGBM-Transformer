# OMNI-GIC-LGBM-Transformer

This repository provides a reproducible research codebase for geomagnetically induced current (GIC) prediction using OMNI solar wind data and a two-stage LightGBM–Transformer framework, as described in the accompanying manuscript.

---

## Key Features

- End-to-end GIC prediction from solar wind parameters to local GIC measurements  
- Two-stage hybrid framework:
  - LightGBM for large-scale trend modeling
  - Transformer for residual correction and short-term spike enhancement
- Physics-informed feature construction (e.g., Akasofu ε, SYM-H derivatives, temporal lag features)
- Unified benchmarking with Random Forest (RF), CNN, and LSTM baselines
- Reproducible experiments controlled by configuration files

---

## Workflow (as Described in the Manuscript)

### Stage 1: Feature Construction

OMNI solar wind parameters are ingested and time-aligned with GIC observations.  
Physics-informed coupling functions and temporal lag features are constructed to form a unified feature set.

### Stage 2: LightGBM Trend Prediction

A LightGBM model is trained to capture the dominant trend and baseline variability of GIC under solar wind forcing.

### Stage 3: Transformer Residual Learning

Prediction residuals from the LightGBM stage are used as targets for a Transformer model, which focuses on short-term nonlinear variations and spike-like behavior.

### Stage 4: Evaluation

Model performance is evaluated using MAE, RMSE, and sMAPE.  
Metrics are reported both overall and within amplitude bins (0–10, 10–20, 20–30, >30 A).