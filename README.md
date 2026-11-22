# Kaudulla Reservoir 7-Day Effective Storage Forecast

Predicts the effective water storage (in acre-feet) of **Kaudulla Reservoir** seven calendar days ahead using daily records from Sri Lanka's Irrigation Department (via Macro Lab).

## Project Files
- `Kaudulla_Reservoir_Forecast.ipynb` — Data preprocessing, missing-date alignment (`pd.Timedelta(days=7)`), chronological train/validation/test splits, model training, tuning, and evaluation.
- `LK.WEATHER.RESERVOIR.KAUDULLA.EFFECTIVE_STORAGE.D.csv` — Daily reservoir storage dataset (`2020-10-01` to `2026-11-20`).
- `app.py` — Interactive Streamlit dashboard.
- `forecast_model.joblib` — Trained models (`Random Forest`, `XGBoost`, `LightGBM`, `CatBoost`, `Stacking`, `RF + ANN`), feature importances, and historical lookup table.
- `model_scores.csv` — 2025 Validation MAE and 2026 Test MAE comparison.
- `test_forecasts.csv` — 2026 chronological test predictions.
- `defaults.json` — Default input readings (`2026-11-20`).
- `requirements.txt` — Python dependencies for Streamlit deployment.

## Model Summary (MAE in acre-feet)
- **Baseline (Persistence):** Validation `3,761.0` | Test `3,129.1`
- **CatBoost (Validation-selected ML model):** Validation `3,870.2` | Test `2,789.8` (`-10.8%` vs. Baseline)
- **LightGBM (Lowest 2026 Test MAE):** Validation `4,445.1` | Test `2,532.7` (`-19.1%` vs. Baseline)

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
