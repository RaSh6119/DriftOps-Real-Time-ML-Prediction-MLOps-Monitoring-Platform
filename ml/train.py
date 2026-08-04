import os
from datetime import datetime, timezone

import joblib
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from data_prep import add_aggregate_features, add_targets, load_and_clean, split_by_date
from features import (
    CLASSIFICATION_TARGET,
    FEATURE_COLUMNS,
    REGRESSION_TARGET,
    build_preprocessor,
    get_output_feature_names,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier, XGBRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # .../driftops/ml
PROJECT_ROOT = os.path.dirname(BASE_DIR)                # .../driftops
MLFLOW_DB_PATH = os.path.join(PROJECT_ROOT, "mlflow.db")

mlflow.set_tracking_uri(f"sqlite:///{MLFLOW_DB_PATH}")
mlflow.set_experiment("driftops-delay-prediction")

REGISTERED_MODEL_NAME = "driftops-delay-model"
PRODUCTION_ALIAS = "production"

class DriftOpsModel(mlflow.pyfunc.PythonModel):
    def __init__(self, preprocessor, classifier, regressor):
        self.preprocessor = preprocessor
        self.classifier = classifier
        self.regressor = regressor

    def predict(self, context, model_input: pd.DataFrame):
        X = self.preprocessor.transform(model_input[FEATURE_COLUMNS])
        proba = self.classifier.predict_proba(X)[:, 1]
        minutes = np.clip(self.regressor.predict(X), 0, None)
        return pd.DataFrame({"delay_probability": proba, "delay_minutes": minutes})


def train_and_log(run_name="training-run", register=True, promote=True):
    df = load_and_clean()
    train_df, test_df = split_by_date(df)
    train_df, test_df, lookups = add_aggregate_features(train_df, test_df)
    train_df = add_targets(train_df)
    test_df = add_targets(test_df)

    preprocessor = build_preprocessor()
    X_train = preprocessor.fit_transform(train_df[FEATURE_COLUMNS])
    X_test = preprocessor.transform(test_df[FEATURE_COLUMNS])

    neg = (train_df[CLASSIFICATION_TARGET] == 0).sum()
    pos = (train_df[CLASSIFICATION_TARGET] == 1).sum()
    params = {
        "n_estimators": 250,
        "max_depth": 6,
        "learning_rate": 0.08,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42,
        "scale_pos_weight": neg / pos,
    }

    clf = XGBClassifier(**params, eval_metric="logloss")
    clf.fit(X_train, train_df[CLASSIFICATION_TARGET])
    reg = XGBRegressor(**params)
    reg.fit(X_train, train_df[REGRESSION_TARGET])

    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    metrics = {
        "accuracy": accuracy_score(test_df[CLASSIFICATION_TARGET], y_pred),
        "roc_auc": roc_auc_score(test_df[CLASSIFICATION_TARGET], y_proba),
        "f1": f1_score(test_df[CLASSIFICATION_TARGET], y_pred),
        "precision": precision_score(test_df[CLASSIFICATION_TARGET], y_pred),
        "recall": recall_score(test_df[CLASSIFICATION_TARGET], y_pred),
        "delay_minutes_mae": mean_absolute_error(test_df[REGRESSION_TARGET], np.clip(reg.predict(X_test), 0, None)),
    }

    version = None
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        bundle = DriftOpsModel(preprocessor, clf, reg)
        mlflow.pyfunc.log_model(artifact_path="model", python_model=bundle,
                                 registered_model_name=REGISTERED_MODEL_NAME if register else None)
        run_id = run.info.run_id

        if register:
            client = mlflow.MlflowClient()
            version = next(mv.version for mv in client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
                            if mv.run_id == run_id)
            if promote:
                client.set_registered_model_alias(REGISTERED_MODEL_NAME, PRODUCTION_ALIAS, version)

    result = {"run_id": run_id, "version": str(version) if version else None, "metrics": metrics}

    if promote:
        models_dir = os.path.join(PROJECT_ROOT, "models")
        os.makedirs(models_dir, exist_ok=True)
        joblib.dump(
            {"preprocessor": preprocessor, "classifier": clf, "regressor": reg,
             "feature_names": get_output_feature_names(preprocessor), "metrics": metrics,
             "lookups": lookups, "version": version,
             "trained_at": datetime.now(timezone.utc).isoformat()},
            os.path.join(models_dir, "latest_model.joblib"),
        )

    return result


if __name__ == "__main__":
    import json
    print(json.dumps(train_and_log(run_name="initial-training"), indent=2))