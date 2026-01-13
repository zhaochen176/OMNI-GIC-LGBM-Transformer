# OMNI-GIC-LGBM-Transformer

This repository provides the code structure and (optionally) processed datasets for geomagnetically induced current (GIC) prediction using OMNI solar wind data and a two-stage LightGBM–Transformer framework. Baseline models (RF, CNN, and LSTM) are included for benchmarking under the same feature set and evaluation protocol.

## Workflow (as described in the manuscript)
1. **OMNI → Feature construction**: ingest OMNI solar wind parameters, align them to GIC observations, and construct physics-informed and temporal features (e.g., Akasofu ε, SYM-H derivatives, lag features).
2. **LightGBM stage**: train a LightGBM model to obtain an initial GIC prediction.
3. **Residual learning stage (Transformer)**: compute prediction residuals and train a Transformer to correct the residuals.
4. **Evaluation**: report MAE, RMSE, and sMAPE, including bin-wise statistics (0–10, 10–20, 20–30, >30, ALL).
5. **Baselines**: Random Forest (RF), CNN, and LSTM baselines using the same tabular feature set.

## Repository Structure
- `configs/`: experiment configurations (paper run, baselines, quick run)
- `src/`: data preprocessing, feature engineering, model training, and evaluation
- `data/`: data description and (optionally) example/processed datasets
- `outputs/`: generated models, metrics, figures, and logs (ignored by default)

## Quick Start
> This repository is being organized for reproducibility. The following command will become the unified entry point.

```bash
python main.py --config configs/paper.yaml --model all
