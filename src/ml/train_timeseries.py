"""
Task 2.2: Time-Series Model Development
Uses Azure Predictive Maintenance Dataset with temporal sequence features.
MLPClassifier serves as a neural-network-based sequence model (LSTM-equivalent
for tabular data) since TensorFlow/PyTorch are unavailable on Python 3.14.
"""
import polars as pl
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, roc_auc_score
import xgboost as xgb
import joblib
import os

AZURE_PATH = r"C:\Users\Shiti\.cache\kagglehub\datasets\arnabbiswas1\microsoft-azure-predictive-maintenance\versions\3"

def load_and_merge_azure():
    """Load all Azure CSVs and perform temporal joins."""
    print("[2.2] Loading Azure dataset...")
    telemetry = pl.read_csv(os.path.join(AZURE_PATH, "PdM_telemetry.csv"))
    telemetry = telemetry.with_columns(pl.col("datetime").str.to_datetime("%Y-%m-%d %H:%M:%S"))

    errors = pl.read_csv(os.path.join(AZURE_PATH, "PdM_errors.csv"))
    errors = errors.with_columns(pl.col("datetime").str.to_datetime("%Y-%m-%d %H:%M:%S"))

    maint = pl.read_csv(os.path.join(AZURE_PATH, "PdM_maint.csv"))
    maint = maint.with_columns(pl.col("datetime").str.to_datetime("%Y-%m-%d %H:%M:%S"))

    failures = pl.read_csv(os.path.join(AZURE_PATH, "PdM_failures.csv"))
    failures = failures.with_columns(pl.col("datetime").str.to_datetime("%Y-%m-%d %H:%M:%S"))

    machines = pl.read_csv(os.path.join(AZURE_PATH, "PdM_machines.csv"))

    print(f"  Telemetry: {telemetry.height} rows | Errors: {errors.height} | Failures: {failures.height}")
    return telemetry, errors, maint, failures, machines


def engineer_temporal_features(telemetry, errors, maint, failures, machines):
    """
    Build lag/rolling features that capture temporal patterns (LSTM-equivalent).
    This is the 'Muscle' of the system.
    """
    print("[2.2] Engineering temporal features...")

    # Sort telemetry
    tel = telemetry.sort(["machineID", "datetime"])

    # --- Rolling Statistics (3h = 3 rows, 24h = 24 rows at hourly intervals) ---
    for window in [3, 24]:
        tel = tel.with_columns([
            pl.col("volt").rolling_mean(window_size=window).over("machineID").alias(f"volt_avg_{window}h"),
            pl.col("rotate").rolling_mean(window_size=window).over("machineID").alias(f"rotate_avg_{window}h"),
            pl.col("pressure").rolling_mean(window_size=window).over("machineID").alias(f"pressure_avg_{window}h"),
            pl.col("vibration").rolling_mean(window_size=window).over("machineID").alias(f"vibration_avg_{window}h"),
            pl.col("volt").rolling_std(window_size=window).over("machineID").alias(f"volt_std_{window}h"),
            pl.col("rotate").rolling_std(window_size=window).over("machineID").alias(f"rotate_std_{window}h"),
            pl.col("pressure").rolling_std(window_size=window).over("machineID").alias(f"pressure_std_{window}h"),
            pl.col("vibration").rolling_std(window_size=window).over("machineID").alias(f"vibration_std_{window}h"),
        ])

    # --- Rate of Change (Diff) ---
    tel = tel.with_columns([
        pl.col("volt").diff().over("machineID").alias("volt_diff"),
        pl.col("rotate").diff().over("machineID").alias("rotate_diff"),
        pl.col("pressure").diff().over("machineID").alias("pressure_diff"),
        pl.col("vibration").diff().over("machineID").alias("vibration_diff"),
    ])

    # --- Error Count Features ---
    # Count errors per machine per datetime
    error_counts = (
        errors.group_by(["datetime", "machineID"])
        .agg(pl.col("errorID").count().alias("error_count"))
    )
    tel = tel.join(error_counts, on=["datetime", "machineID"], how="left").with_columns(
        pl.col("error_count").fill_null(0)
    )

    # --- Machine Metadata ---
    tel = tel.join(machines, on="machineID", how="left")

    # --- Failure Labels (Binary: failure within next 24 hours) ---
    # For each failure, mark the preceding 24 rows as "will fail"
    failure_set = set()
    for row in failures.iter_rows(named=True):
        mid = row["machineID"]
        dt = row["datetime"]
        failure_set.add((mid, dt))

    # Create label column
    tel = tel.with_columns(pl.lit(0).alias("target"))

    # Convert to pandas for label propagation (more flexible for this operation)
    pdf = tel.to_pandas()

    for mid, fdt in failure_set:
        mask = (pdf["machineID"] == mid) & (pdf["datetime"] <= fdt) & (pdf["datetime"] > fdt - pd.Timedelta(hours=24))
        pdf.loc[mask, "target"] = 1

    print(f"  Failure rate: {pdf['target'].mean()*100:.2f}%")
    print(f"  Feature columns: {len(pdf.columns)}")

    return pdf


def train_time_series_models(pdf):
    """Train MLP (neural net) and Random Forest on temporal features."""
    os.makedirs("models", exist_ok=True)

    # Drop non-feature columns
    drop_cols = ["datetime", "machineID", "model", "target"]
    feature_cols = [c for c in pdf.columns if c not in drop_cols]

    # Drop rows with NaN from rolling windows
    pdf = pdf.dropna(subset=feature_cols)

    X = pdf[feature_cols].values
    y = pdf["target"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale for neural network
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- 1. MLPClassifier (Neural Network / LSTM-Equivalent) ---
    print("\n[2.2] Training MLPClassifier (Neural Network)...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        max_iter=100,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=True
    )
    mlp.fit(X_train_scaled, y_train)

    y_pred_mlp = mlp.predict(X_test_scaled)
    y_prob_mlp = mlp.predict_proba(X_test_scaled)[:, 1]

    print("\n--- MLP (Neural Network) Results ---")
    print(classification_report(y_test, y_pred_mlp))
    print(f"F1 Score: {f1_score(y_test, y_pred_mlp):.4f}")
    print(f"ROC-AUC:  {roc_auc_score(y_test, y_prob_mlp):.4f}")

    joblib.dump(mlp, "models/mlp_timeseries.pkl")
    joblib.dump(scaler, "models/azure_scaler.pkl")
    joblib.dump(feature_cols, "models/azure_feature_cols.pkl")
    print("Saved: models/mlp_timeseries.pkl")

    # --- 2. Random Forest (Ensemble Baseline) ---
    print("\n[2.2] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]

    print("\n--- Random Forest Results ---")
    print(classification_report(y_test, y_pred_rf))
    print(f"F1 Score: {f1_score(y_test, y_pred_rf):.4f}")
    print(f"ROC-AUC:  {roc_auc_score(y_test, y_prob_rf):.4f}")

    joblib.dump(rf, "models/rf_timeseries.pkl")
    print("Saved: models/rf_timeseries.pkl")

    # --- 3. XGBoost on Time-Series Features (for comparison) ---
    print("\n[2.2] Training XGBoost on Azure features...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        tree_method='hist',
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        random_state=42
    )
    xgb_model.fit(X_train, y_train)

    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

    print("\n--- XGBoost (Azure) Results ---")
    print(classification_report(y_test, y_pred_xgb))
    print(f"F1 Score: {f1_score(y_test, y_pred_xgb):.4f}")
    print(f"ROC-AUC:  {roc_auc_score(y_test, y_prob_xgb):.4f}")

    joblib.dump(xgb_model, "models/xgb_timeseries.pkl")
    print("Saved: models/xgb_timeseries.pkl")

    return X_test, X_test_scaled, y_test, feature_cols, xgb_model, rf, mlp


def train_ttf_regression(pdf):
    """
    Time-To-Failure (TTF) Regression: Predict how many hours until next failure.
    """
    print("\n[2.2] Training Time-to-Failure (TTF) Regression...")

    # Calculate TTF for each record
    pdf = pdf.sort_values(["machineID", "datetime"])
    pdf["ttf"] = np.nan

    for mid in pdf["machineID"].unique():
        machine_data = pdf[pdf["machineID"] == mid].copy()
        failure_times = machine_data[machine_data["target"] == 1]["datetime"]

        if len(failure_times) == 0:
            continue

        for ft in failure_times:
            mask = (pdf["machineID"] == mid) & (pdf["datetime"] <= ft)
            # Calculate hours until this specific failure
            time_diff = (ft - pdf.loc[mask, "datetime"]).dt.total_seconds() / 3600
            
            # Get existing ttf for these rows
            existing_ttf = pdf.loc[mask, "ttf"]
            
            # Only update where ttf is NaN or where new ttf is smaller (closer to failure)
            # Use .values to avoid alignment issues if needed, or ensure indices match
            update_mask_in_subset = existing_ttf.isna() | (time_diff < existing_ttf)
            
            # Combine with main mask to get global update mask
            final_update_mask = mask.copy()
            final_update_mask[mask] = update_mask_in_subset
            
            pdf.loc[final_update_mask, "ttf"] = time_diff[update_mask_in_subset]

    # Drop records with no TTF
    ttf_data = pdf.dropna(subset=["ttf"])
    ttf_data = ttf_data[ttf_data["ttf"] <= 168]  # Cap at 1 week (168 hours)

    drop_cols = ["datetime", "machineID", "model", "target", "ttf"]
    feature_cols = [c for c in ttf_data.columns if c not in drop_cols]
    ttf_data = ttf_data.dropna(subset=feature_cols)

    X = ttf_data[feature_cols].values
    y = ttf_data["ttf"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    ttf_model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        tree_method='hist',
        random_state=42
    )
    ttf_model.fit(X_train, y_train)

    y_pred = ttf_model.predict(X_test)
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

    print(f"  TTF MAE:  {mae:.2f} hours")
    print(f"  TTF RMSE: {rmse:.2f} hours")

    joblib.dump(ttf_model, "models/ttf_regressor.pkl")
    print("Saved: models/ttf_regressor.pkl")

    return ttf_model


if __name__ == "__main__":
    tel, err, maint, fail, mach = load_and_merge_azure()
    pdf = engineer_temporal_features(tel, err, maint, fail, mach)
    X_test, X_test_scaled, y_test, feature_cols, xgb_m, rf_m, mlp_m = train_time_series_models(pdf)
    ttf_model = train_ttf_regression(pdf)
    print("\n✅ Task 2.2 Complete: All time-series models trained and saved.")
