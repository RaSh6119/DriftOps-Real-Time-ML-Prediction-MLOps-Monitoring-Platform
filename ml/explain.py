import numpy as np
import shap
import os
from features import FEATURE_COLUMNS, get_output_feature_names, CATEGORICAL_COLUMNS

FRIENDLY_NAMES = {
    "Distance": "Trip distance",
    "hour_of_day": "Time of day",
    "weather_severity": "Weather-related delay history at this airport",
    "traffic_index": "Historical congestion at this hour",
    "historical_demand": "Scheduled flight volume",
    "driver_availability": "Carrier's on-time reliability",
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))   # .../driftops/ml
PROJECT_ROOT = os.path.dirname(BASE_DIR)                 # .../driftops


def _source_column(raw_name: str) -> str:
    for cat_col in CATEGORICAL_COLUMNS:
        if raw_name.startswith(f"{cat_col}_"):
            return cat_col
    return raw_name  # numeric columns map to themselves

def _friendly_label(col: str, value=None) -> str:
    if col == "Origin":
        return f"Departing from {value}"
    if col == "Dest":
        return f"Arriving at {value}"
    if col == "DayOfWeek":
        return f"Day of week: {value}"
    return FRIENDLY_NAMES.get(col, col)

def explain_prediction(preprocessor, classifier, input_row, top_n=5):
    X = preprocessor.transform(input_row[FEATURE_COLUMNS])
    # if hasattr(X, "toarray"):
    #     X = X.toarray()

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    feature_names = get_output_feature_names(preprocessor)
    row_shap = np.array(shap_values, dtype=np.float64).reshape(-1)

    grouped = {}
    for name, val in zip(feature_names, row_shap):
        col = _source_column(name)
        grouped[col] = grouped.get(col, 0.0) + val

    total_abs = sum(abs(v) for v in grouped.values()) or 1.0
    top = sorted(grouped.items(), key=lambda kv: -abs(kv[1]))[:top_n]

    return [
        {
            "feature": col,
            "label": _friendly_label(col, input_row[col].iloc[0] if col in CATEGORICAL_COLUMNS else None),
            "impact_pct": round(100 * abs(val) / total_abs, 1),
            "direction": "increases" if val > 0 else "decreases",
        }
        for col, val in top
    ]

if __name__ == "__main__":
    import joblib
    import pandas as pd

    bundle = joblib.load(os.path.join(PROJECT_ROOT, "models", "latest_model.joblib"))
    test_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "test_processed.csv"))

    sample = test_df.iloc[[0]]
    result = explain_prediction(bundle["preprocessor"], bundle["classifier"], sample)
    for r in result:
        print(f"{r['label']}: {r['direction']} risk by {r['impact_pct']}%")