import pandas as pd

RAW_CSV_PATH = "data/raw/flight_raw.csv"

COLUMNS_NEEDED = [
    "Year", "Month", "DayOfWeek", "FlightDate",
    "Reporting_Airline", "Origin", "Dest",
    "CRSDepTime", "DepDelay", "CRSArrTime", "ArrDelay",
    "Cancelled", "Diverted", "Distance", "WeatherDelay",
]


def load_and_clean(path: str = RAW_CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=COLUMNS_NEEDED)
    df.columns = [c.strip() for c in df.columns]

    df = df[(df["Cancelled"] == 0) & (df["Diverted"] == 0)]
    df = df.dropna(subset=["DepDelay", "ArrDelay"])

    df["CRSDepTime"] = df["CRSDepTime"].replace(2400, 0)
    df["hour_of_day"] = (df["CRSDepTime"] // 100).clip(0, 23)

    if "FlightDate" in df.columns:
        df["FlightDate"] = pd.to_datetime(df["FlightDate"])
    else:
        # some exports don't include a combined FlightDate column
        df["FlightDate"] = pd.to_datetime(
            df["Year"].astype(str) + "-" + df["Month"].astype(str) + "-" + df["DayofMonth"].astype(str)
        )
    return df

def split_by_date(df: pd.DataFrame, test_days: int = 7):
    cutoff = df["FlightDate"].max() - pd.Timedelta(days=test_days)
    train_df = df[df["FlightDate"] < cutoff].copy()
    test_df = df[df["FlightDate"] >= cutoff].copy()
    return train_df, test_df

def _smoothed_group_stat(train_df, group_cols, value_series, k=20):
    """Shrinks small-sample group means toward the global mean.
    k = how many 'prior' flights worth of trust to give the global mean —
    a group with far fewer than k flights gets pulled hard toward the global average."""
    tmp = train_df[group_cols].copy()
    tmp["_val"] = value_series.values
    agg = tmp.groupby(group_cols)["_val"].agg(["mean", "count"]).reset_index()
    global_mean = value_series.mean()
    agg["smoothed"] = (agg["count"] * agg["mean"] + k * global_mean) / (agg["count"] + k)
    return agg[group_cols + ["smoothed"]], global_mean

def build_feature_lookups(train_df: pd.DataFrame, k: int = 20):
    clipped_dep_delay = train_df["DepDelay"].clip(upper=180)
    traffic_lookup, traffic_global = _smoothed_group_stat(train_df, ["Origin", "hour_of_day"], clipped_dep_delay, k)
    traffic_lookup = traffic_lookup.rename(columns={"smoothed": "traffic_index"})

    had_weather_delay = (train_df["WeatherDelay"].fillna(0) > 0).astype(float)
    weather_lookup, weather_global = _smoothed_group_stat(train_df, ["Origin", "Month"], had_weather_delay, k)
    weather_lookup = weather_lookup.rename(columns={"smoothed": "weather_severity"})

    demand = train_df.groupby(["Origin", "hour_of_day"]).size().rename("historical_demand").reset_index()
    demand_global = demand["historical_demand"].mean()

    carrier_reliability = (
        train_df.assign(on_time=lambda d: d["ArrDelay"] <= 15)
        .groupby("Reporting_Airline")["on_time"].mean()
        .rename("driver_availability").reset_index()
    )
    carrier_global = carrier_reliability["driver_availability"].mean()

    return {
        "traffic_lookup": traffic_lookup, "traffic_global": traffic_global,
        "weather_lookup": weather_lookup, "weather_global": weather_global,
        "demand_lookup": demand, "demand_global": demand_global,
        "carrier_lookup": carrier_reliability, "carrier_global": carrier_global,
    }

def add_aggregate_features(train_df, test_df, k=20):
    lookups = build_feature_lookups(train_df, k)
    train_df, test_df = train_df.copy(), test_df.copy()

    for df in (train_df, test_df):
        df.drop(columns=["traffic_index", "weather_severity", "historical_demand", "driver_availability"],
                 errors="ignore", inplace=True)

    for split_name, df in (("train", train_df), ("test", test_df)):
        pass

    train_df = train_df.merge(lookups["traffic_lookup"], on=["Origin", "hour_of_day"], how="left")
    train_df = train_df.merge(lookups["weather_lookup"], on=["Origin", "Month"], how="left")
    train_df = train_df.merge(lookups["demand_lookup"], on=["Origin", "hour_of_day"], how="left")
    train_df = train_df.merge(lookups["carrier_lookup"], on="Reporting_Airline", how="left")

    test_df = test_df.merge(lookups["traffic_lookup"], on=["Origin", "hour_of_day"], how="left")
    test_df = test_df.merge(lookups["weather_lookup"], on=["Origin", "Month"], how="left")
    test_df = test_df.merge(lookups["demand_lookup"], on=["Origin", "hour_of_day"], how="left")
    test_df = test_df.merge(lookups["carrier_lookup"], on="Reporting_Airline", how="left")

    fallbacks = {
        "traffic_index": lookups["traffic_global"], "weather_severity": lookups["weather_global"],
        "historical_demand": lookups["demand_global"], "driver_availability": lookups["carrier_global"],
    }
    for col, fallback in fallbacks.items():
        train_df[col] = train_df[col].fillna(fallback)
        test_df[col] = test_df[col].fillna(fallback)

    return train_df, test_df, lookups

def add_targets(df: pd.DataFrame, delay_cap_minutes: int = 240) -> pd.DataFrame:
    df = df.copy()
    df["is_delayed"] = (df["ArrDelay"] > 15).astype(int)
    df["delay_minutes"] = df["ArrDelay"].clip(lower=0, upper=delay_cap_minutes)
    return df

if __name__ == "__main__":
    df = load_and_clean()
    train_df, test_df = split_by_date(df)
    train_df, test_df, lookups = add_aggregate_features(train_df, test_df)
    train_df = add_targets(train_df)
    test_df = add_targets(test_df)
    # print(pd.read_csv(RAW_CSV_PATH, nrows=0).columns.tolist())
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    print(f"Train delay rate: {train_df['is_delayed'].mean():.3f}, Test delay rate: {test_df['is_delayed'].mean():.3f}")
    print(train_df.head())
    train_df.to_csv("data/train_processed.csv", index=False)
    test_df.to_csv("data/test_processed.csv", index=False)

    # 1. NaN check on the new engineered columns — should all be 0
    new_cols = ["traffic_index", "weather_severity", "historical_demand", "driver_availability"]
    print("NaNs in train:\n", train_df[new_cols].isna().sum())
    print("NaNs in test:\n", test_df[new_cols].isna().sum())

    # 2. Range/sanity check on the engineered features
    print(train_df[new_cols].describe())

    # 3. Delay rate — should be roughly 15-30%, not near 0% or 100%
    print("Train delay rate:", train_df["is_delayed"].value_counts(normalize=True))
    print("Test delay rate:", test_df["is_delayed"].value_counts(normalize=True))

    # 4. delay_minutes should be 0 for every non-delayed row, and >0 only for delayed ones
    print(train_df.loc[train_df["is_delayed"] == 0, "delay_minutes"].max())  # should be 0 or very close
    print(train_df.loc[train_df["is_delayed"] == 1, "delay_minutes"].describe())

    counts = train_df.groupby(["Origin", "hour_of_day"]).size()
    print(counts.sort_values().head(10))  # smallest-sample buckets — likely where traffic_index is noisy

    small_buckets = [("FAI", 18), ("TVC", 18), ("CRP", 19), ("LEX", 20)]
    check = train_df[train_df[["Origin", "hour_of_day"]].apply(tuple, axis=1).isin(small_buckets)]
    print(check[["Origin", "hour_of_day", "traffic_index"]].drop_duplicates())