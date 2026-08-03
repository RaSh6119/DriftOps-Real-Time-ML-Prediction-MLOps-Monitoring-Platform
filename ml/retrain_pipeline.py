import json
import mlflow

from train import train_and_log, REGISTERED_MODEL_NAME, PRODUCTION_ALIAS, MLFLOW_DB_PATH


def _get_current_production_metrics():
    mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
    client = mlflow.MlflowClient()
    try:
        current = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, PRODUCTION_ALIAS)
    except Exception:
        return None
    run = client.get_run(current.run_id)
    return {"version": str(current.version), "roc_auc": run.data.metrics.get("roc_auc")}


def run_retraining() -> dict:
    current = _get_current_production_metrics()

    # always train and register a new version for audit trail; decide promotion separately
    candidate = train_and_log(run_name="retrain-candidate", promote=False)
    candidate_roc_auc = candidate["metrics"]["roc_auc"]
    current_roc_auc = current["roc_auc"] if current else None

    should_promote = current is None or candidate_roc_auc > current_roc_auc

    if should_promote:
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, PRODUCTION_ALIAS, int(candidate["version"]))
        # re-run with promote=True to also refresh the served joblib bundle
        train_and_log(run_name="retrain-candidate-promoted", promote=True)

    return {
        "decision": "promoted" if should_promote else "kept_existing",
        "candidate_version": candidate["version"],
        "candidate_metrics": candidate["metrics"],
        "current_production_version": current["version"] if current else None,
        "current_production_roc_auc": current_roc_auc,
    }


if __name__ == "__main__":
    print(json.dumps(run_retraining(), indent=2))