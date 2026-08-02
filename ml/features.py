from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL_COLUMNS = ["Origin", "Dest", "DayOfWeek"]
NUMERIC_COLUMNS = [
    "Distance",
    "hour_of_day",
    "weather_severity",
    "traffic_index",
    "historical_demand",
    "driver_availability",
]
FEATURE_COLUMNS = CATEGORICAL_COLUMNS + NUMERIC_COLUMNS

CLASSIFICATION_TARGET = "is_delayed"
REGRESSION_TARGET = "delay_minutes"


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLUMNS),
            ("num", StandardScaler(), NUMERIC_COLUMNS),
        ]
    )


def get_output_feature_names(preprocessor: ColumnTransformer) -> list:
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_COLUMNS))
    return cat_names + NUMERIC_COLUMNS

if __name__ == "__main__":
    from data_prep import load_and_clean, split_by_date, add_aggregate_features, add_targets

    df = load_and_clean()
    train_df, test_df = split_by_date(df)
    train_df, test_df = add_aggregate_features(train_df, test_df)
    train_df = add_targets(train_df)

    pre = build_preprocessor()
    X = pre.fit_transform(train_df[FEATURE_COLUMNS])
    print(X.shape)