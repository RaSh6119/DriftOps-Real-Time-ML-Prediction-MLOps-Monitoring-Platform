import os
import sys
import time
import joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))       # backend/app
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))   # driftops root
ML_DIR = os.path.join(PROJECT_ROOT, "ml")
sys.path.insert(0, ML_DIR)  # reuse ml/features.py and ml/explain.py, don't duplicate logic

from features import FEATURE_COLUMNS, get_output_feature_names
from explain import explain_prediction

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "latest_model.joblib")
_bundle = None

def load_bundle():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load(MODEL_PATH)
    return _bundle


def _lookup(key_cols, key_values, value_col, lookup_df, global_fallback):
    match = lookup_df
    for col, val in zip(key_cols, key_values):
        match = match[match[col] == val]
    if len(match) == 0:
        return float(global_fallback)
    return float(match.iloc[0][value_col])


def build_feature_row(request) -> pd.DataFrame:
    lookups = load_bundle()["lookups"]

    traffic_index = _lookup(["Origin", "hour_of_day"], [request.origin, request.scheduled_hour],
                             "traffic_index", lookups["traffic_lookup"], lookups["traffic_global"])
    weather_severity = _lookup(["Origin", "Month"], [request.origin, request.month],
                                "weather_severity", lookups["weather_lookup"], lookups["weather_global"])
    historical_demand = _lookup(["Origin", "hour_of_day"], [request.origin, request.scheduled_hour],
                                 "historical_demand", lookups["demand_lookup"], lookups["demand_global"])
    driver_availability = _lookup(["Reporting_Airline"], [request.carrier],
                                   "driver_availability", lookups["carrier_lookup"], lookups["carrier_global"])

    return pd.DataFrame([{
        "Origin": request.origin, "Dest": request.destination, "DayOfWeek": request.day_of_week,
        "Distance": request.distance_miles, "hour_of_day": request.scheduled_hour,
        "weather_severity": weather_severity, "traffic_index": traffic_index,
        "historical_demand": historical_demand, "driver_availability": driver_availability,
    }])


def predict(request) -> dict:
    bundle = load_bundle()
    row = build_feature_row(request)

    start = time.time()
    X = bundle["preprocessor"].transform(row[FEATURE_COLUMNS])
    proba = float(bundle["classifier"].predict_proba(X)[:, 1][0])
    minutes = float(np.clip(bundle["regressor"].predict(X), 0, None)[0])
    latency_ms = (time.time() - start) * 1000

    distance_from_midpoint = abs(proba - 0.5)
    confidence = "High" if distance_from_midpoint > 0.3 else "Medium" if distance_from_midpoint > 0.15 else "Low"

    top_features = explain_prediction(bundle["preprocessor"], bundle["classifier"], row, top_n=3)

    return {
        "delay_probability": round(proba, 4),
        "predicted_delay_minutes": round(minutes, 1),
        "confidence": confidence,
        "top_features": top_features,
        "model_version": bundle.get("version"),
        "latency_ms": round(latency_ms, 2),
    }

if __name__ == "__main__":
    import shap
    from schemas import PredictionRequest
    from features import get_output_feature_names
    from explain import _source_column

    req = PredictionRequest(origin="ATL", destination="ORD", carrier="DL",
                             distance_miles=606, scheduled_hour=17, day_of_week=5, month=3)
    print(predict(req))

    # bundle = load_bundle()
    # row = build_feature_row(req)

    # # Build X exactly the way explain_prediction() does: dense, once.
    # X = bundle["preprocessor"].transform(row[FEATURE_COLUMNS])
    # if hasattr(X, "toarray"):
    #     X = X.toarray()

    # # Fresh explainer, used once for everything below.
    # explainer = shap.TreeExplainer(bundle["classifier"])
    # raw_margin = bundle["classifier"].predict(X, output_margin=True)
    # shap_vals = explainer.shap_values(X)
    # if isinstance(shap_vals, list):
    #     shap_vals = shap_vals[1]
    # shap_vals = np.array(shap_vals, dtype=np.float64).reshape(-1)

    # print("raw margin:", raw_margin)
    # print("expected_value + sum(shap):", explainer.expected_value + shap_vals.sum())

    # feature_names = get_output_feature_names(bundle["preprocessor"])
    # grouped = {}
    # for name, val in zip(feature_names, shap_vals):
    #     col = _source_column(name)
    #     grouped[col] = grouped.get(col, 0.0) + val

    # for k, v in sorted(grouped.items(), key=lambda kv: -abs(kv[1])):
    #     print(f"{k}: {v:.4f}")
    # print("sum:", sum(grouped.values()))