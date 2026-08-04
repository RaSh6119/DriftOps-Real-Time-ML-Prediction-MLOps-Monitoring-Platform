import os

import numpy as np
import pandas as pd
from features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS, NUMERIC_COLUMNS

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

PSI_LOW = 0.1       # standard industry PSI thresholds
PSI_MODERATE = 0.25


def _psi_numeric(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    ref, cur = reference.dropna(), current.dropna()
    breakpoints = np.linspace(0, 100, bins + 1)
    quantiles = np.percentile(ref, breakpoints)
    quantiles[0], quantiles[-1] = -np.inf, np.inf
    quantiles = np.unique(quantiles)

    ref_counts, _ = np.histogram(ref, bins=quantiles)
    cur_counts, _ = np.histogram(cur, bins=quantiles)
    ref_pct = np.clip(ref_counts / max(len(ref), 1), 1e-4, None)
    cur_pct = np.clip(cur_counts / max(len(cur), 1), 1e-4, None)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def _psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    categories = set(reference.unique()) | set(current.unique())
    ref_counts = reference.value_counts(normalize=True)
    cur_counts = current.value_counts(normalize=True)

    psi = 0.0
    for cat in categories:
        ref_pct = max(ref_counts.get(cat, 0.0), 1e-4)
        cur_pct = max(cur_counts.get(cat, 0.0), 1e-4)
        psi += (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    return float(psi)


def _severity(psi: float) -> str:
    if psi < PSI_LOW:
        return "none"
    if psi < PSI_MODERATE:
        return "moderate"
    return "high"


def compute_drift_report(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> dict:
    per_feature = {}
    for col in NUMERIC_COLUMNS:
        psi = _psi_numeric(reference_df[col], current_df[col])
        per_feature[col] = {
            "psi": round(psi, 4), "severity": _severity(psi),
            "reference_mean": round(float(reference_df[col].mean()), 3),
            "current_mean": round(float(current_df[col].mean()), 3),
        }
    for col in CATEGORICAL_COLUMNS:
        psi = _psi_categorical(reference_df[col], current_df[col])
        per_feature[col] = {"psi": round(psi, 4), "severity": _severity(psi),
                             "reference_mean": None, "current_mean": None}

    drifted_features = [c for c, v in per_feature.items() if v["severity"] != "none"]
    overall_psi = max(v["psi"] for v in per_feature.values())
    overall_severity = _severity(overall_psi)
    recommendation = ("retrain model" if overall_severity == "high"
                       else "monitor closely" if overall_severity == "moderate"
                       else "no action needed")

    return {
        "drift_detected": len(drifted_features) > 0,
        "severity": overall_severity,
        "overall_psi": round(overall_psi, 4),
        "drifted_features": drifted_features,
        "per_feature": per_feature,
        "recommendation": recommendation,
    }


def generate_html_report(reference_df: pd.DataFrame, current_df: pd.DataFrame, output_path: str):
    from evidently import Report
    from evidently.presets import DataDriftPreset

    report = Report([DataDriftPreset(method="psi")], include_tests=True)
    result = report.run(reference_df[FEATURE_COLUMNS], current_df[FEATURE_COLUMNS])
    result.save_html(output_path)


if __name__ == "__main__":
    import json

    reference_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "train_processed.csv"))
    current_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "test_processed.csv"))

    summary = compute_drift_report(reference_df, current_df)
    print(json.dumps(summary, indent=2))

    try:
        generate_html_report(reference_df, current_df, os.path.join(PROJECT_ROOT, "data", "drift_report.html"))
        print("HTML report saved to data/drift_report.html")
    except Exception as e: # noqa: BLE001 -- HTML report is optional/best-effort, must not break the core drift summary
        print(f"Evidently HTML generation failed (non-fatal, core JSON summary above is unaffected): {e}")