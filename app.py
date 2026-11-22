from datetime import date, timedelta
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent

st.set_page_config(page_title="Kaudulla Reservoir Forecast", layout="wide")
st.title("Kaudulla Reservoir Forecast")
st.write("Predict effective water storage (acre-feet) seven calendar days after a reading.")

@st.cache_resource
def load_bundle():
    return joblib.load(ROOT / "forecast_model.joblib")

bundle = load_bundle()
defaults = json.loads((ROOT / "defaults.json").read_text())
scores = pd.read_csv(ROOT / "model_scores.csv").set_index("Model")
history = pd.read_csv(ROOT / "test_forecasts.csv", parse_dates=["forecast_date", "target_date"])
history_lookup = bundle.get("history_lookup", {})
all_models = bundle.get("models", {bundle["name"]: bundle["model"]})
features = bundle["features"]

def get_model_importance(model_name):
    m = all_models.get(model_name)
    if m is None or not hasattr(m, "feature_importances_"):
        return None
    arr = np.asarray(m.feature_importances_, dtype=float)
    total = arr.sum()
    if total <= 0:
        return None
    return pd.Series(arr / total, index=features).sort_values()

model_options = list(scores.index)
default_idx = model_options.index(bundle["name"]) if bundle["name"] in model_options else 0

st.subheader("Try a forecast")

col1, col2, col3, col4 = st.columns(4)

with col1:
    selected_model = st.selectbox("Model", options=model_options, index=default_idx)

with col2:
    reading_date = st.date_input(
        "Reading date",
        value=date.fromisoformat(defaults["reading_date"]),
        min_value=date(2020, 10, 8),
        max_value=date(2027, 12, 31)
    )

date_key = reading_date.isoformat()
if date_key in history_lookup:
    default_current = history_lookup[date_key]["storage"]
    default_previous = history_lookup[date_key]["previous_storage"]
else:
    default_current = defaults["storage"]
    default_previous = defaults["previous_storage"]

with col3:
    current = st.number_input(
        f"Storage on {reading_date} (acre-feet)",
        min_value=0.0,
        value=float(default_current),
        step=500.0
    )

with col4:
    previous = st.number_input(
        f"Storage on {reading_date - timedelta(days=7)} (acre-feet)",
        min_value=0.0,
        value=float(default_previous),
        step=500.0
    )

change_past_7d = current - previous
inputs = pd.DataFrame([{
    "value": current,
    "change_past_7d": change_past_7d
}])

if selected_model == "Baseline":
    estimate = current
elif selected_model in all_models and all_models[selected_model] is not None:
    estimate = float(all_models[selected_model].predict(inputs[features])[0])
else:
    estimate = float(bundle["model"].predict(inputs[features])[0])

target_date = reading_date + timedelta(days=7)
baseline_test_mae = float(scores.loc["Baseline", "Test MAE"])
selected_test_mae = float(scores.loc[selected_model, "Test MAE"])
mae_reduction_pct = 100.0 * (baseline_test_mae - selected_test_mae) / baseline_test_mae

m1, m2, m3 = st.columns(3)
with m1:
    st.metric(
        f"Forecast for {target_date}",
        f"{estimate:,.0f} acre-feet",
        delta=f"{estimate - current:+,.0f} vs reading date"
    )
with m2:
    st.metric(
        "Past 7-day change",
        f"{change_past_7d:+,.0f} acre-feet"
    )
with m3:
    st.metric(
        f"{selected_model} Test MAE",
        f"{selected_test_mae:,.1f} acre-feet",
        delta=f"{mae_reduction_pct:+.1f}% vs Baseline"
    )

tab1, tab2, tab3 = st.tabs([
    "2026 Test Forecasts",
    "Model Scores",
    "Feature Importance & Errors"
])

if selected_model == "Baseline":
    plot_col = "baseline"
elif selected_model in history.columns:
    plot_col = selected_model
else:
    plot_col = "selected_forecast"

with tab1:
    chart_df = history.set_index("target_date")[["actual", "baseline"]].copy()
    if selected_model != "Baseline":
        chart_df[selected_model] = history.set_index("target_date")[plot_col]
    st.line_chart(chart_df)

with tab2:
    st.dataframe(scores.round(1), use_container_width=True)
    st.caption("MAE is the average absolute error in acre-feet. Lower is better.")

with tab3:
    c1, c2 = st.columns([1, 1.4])
    with c1:
        imp_series = get_model_importance(selected_model)
        if imp_series is not None:
            st.write(f"**Feature Importance ({selected_model})**")
            st.bar_chart(imp_series.rename("Importance").to_frame())
        else:
            st.write("**Feature Importance**")
            st.caption("Available for single tree-based models (Random Forest, XGBoost, LightGBM, CatBoost).")

    with c2:
        st.write(f"**Top 5 Largest Errors ({selected_model})**")
        err_df = history[["forecast_date", "target_date", "actual", "baseline"]].copy()
        err_df["forecast"] = history[plot_col]
        err_df["absolute_error"] = (err_df["actual"] - err_df["forecast"]).abs()
        top5 = err_df.nlargest(5, "absolute_error").copy()
        top5["forecast_date"] = top5["forecast_date"].dt.strftime("%Y-%m-%d")
        top5["target_date"] = top5["target_date"].dt.strftime("%Y-%m-%d")
        st.dataframe(top5.round(1), use_container_width=True, hide_index=True)

st.caption("Data source: Sri Lanka Irrigation Department, via Macro Lab.")
