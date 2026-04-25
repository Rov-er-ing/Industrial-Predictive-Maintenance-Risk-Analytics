"""
Task 3.2: Model Validation & Sanity Check
Traces a specific machine leading up to a failure to verify:
1. Probability increases as failure nears.
2. TTF (Time-To-Failure) decreases logically.
"""
import joblib
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

def run_expert_validation():
    print("[3.2] Running Domain Expert Sanity Check...")
    
    # 1. Load Data and Models
    from train_timeseries import load_and_merge_azure, engineer_temporal_features
    tel, err, maint, fail, mach = load_and_merge_azure()
    pdf = engineer_temporal_features(tel, err, maint, fail, mach)
    
    # 2. Pick a machine that actually failed
    failed_machines = fail["machineID"].unique()
    target_machine = failed_machines[0]
    failure_time = fail.filter(pl.col("machineID") == target_machine)["datetime"][0]
    
    # 3. Get the 48 hours leading up to failure
    trace = pdf[(pdf["machineID"] == target_machine) & 
                (pdf["datetime"] <= failure_time) & 
                (pdf["datetime"] > failure_time - pd.Timedelta(hours=48))]
    
    print(f"  Tracing Machine {target_machine} leading to failure at {failure_time}")
    
    # 4. Run Predictions
    model = joblib.load("models/xgb_optimized.pkl")
    ttf_model = joblib.load("models/ttf_regressor.pkl")
    
    drop_cols = ["datetime", "machineID", "model", "target", "ttf"]
    feature_cols = [c for c in pdf.columns if c not in drop_cols]
    X_trace = trace[feature_cols].values
    
    probs = model.predict_proba(X_trace)[:, 1]
    ttfs = ttf_model.predict(X_trace)
    
    # 5. Visualize for "Domain Expert" Review
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(range(len(probs)), probs, color='red', lw=2)
    plt.title(f"Expert Review: Failure Probability (48h lead-up)")
    plt.ylabel("Probability")
    plt.axhline(0.5, color='black', linestyle='--')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(range(len(ttfs)), ttfs, color='blue', lw=2)
    plt.title(f"Expert Review: Predicted Time-To-Failure (TTF)")
    plt.ylabel("Hours")
    plt.xlabel("Hours into trace")
    plt.grid(True)
    
    os.makedirs("models/validation", exist_ok=True)
    plt.tight_layout()
    plt.savefig("models/validation/expert_sanity_check.png")
    
    print("  ✅ Validation Plot saved: models/validation/expert_sanity_check.png")
    print("  Logic Check: Probability should TREND UP, TTF should TREND DOWN.")

if __name__ == "__main__":
    import polars as pl
    run_expert_validation()
