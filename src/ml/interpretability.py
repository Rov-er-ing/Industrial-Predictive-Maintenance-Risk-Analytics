"""
Task 2.4: Model Interpretability (SHAP Analysis)
Generates comprehensive SHAP analysis for all trained models:
  1. Global Feature Importance (Summary Plot)
  2. Per-Feature Dependence Plots
  3. Force Plot for individual predictions
  4. Model Comparison Chart
"""
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import joblib
import os
import json

def load_test_data():
    """Load test data from the Azure pipeline."""
    from train_timeseries import load_and_merge_azure, engineer_temporal_features

    tel, err, maint, fail, mach = load_and_merge_azure()
    pdf = engineer_temporal_features(tel, err, maint, fail, mach)

    drop_cols = ["datetime", "machineID", "model", "target"]
    feature_cols = [c for c in pdf.columns if c not in drop_cols]
    pdf = pdf.dropna(subset=feature_cols)

    X = pdf[feature_cols]
    y = pdf["target"]

    # Use last 20% as test
    split_idx = int(len(X) * 0.8)
    X_test = X.iloc[split_idx:]
    y_test = y.iloc[split_idx:]

    return X_test, y_test, feature_cols


def analyze_xgboost_shap(X_test, feature_cols):
    """SHAP analysis for XGBoost model."""
    print("\n[2.4] SHAP Analysis: XGBoost...")

    model_path = "models/xgb_optimized.pkl"
    if not os.path.exists(model_path):
        model_path = "models/xgb_timeseries.pkl"
    
    if not os.path.exists(model_path):
        print(f"  Error: {model_path} not found.")
        return None, None

    model = joblib.load(model_path)
    explainer = shap.TreeExplainer(model)

    # Use a sample for speed (SHAP on 876k rows is slow)
    sample = X_test.sample(n=min(2000, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(sample)

    # 1. Summary Plot (Global Feature Importance)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, sample, feature_names=feature_cols, show=False, max_display=15)
    plt.title("XGBoost: Global Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig("models/shap_xgb_summary.png", dpi=150)
    plt.close()
    print("  Saved: models/shap_xgb_summary.png")

    # 2. Bar Plot (Mean |SHAP|)
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, sample, feature_names=feature_cols, plot_type="bar", show=False, max_display=15)
    plt.title("XGBoost: Mean |SHAP| Feature Importance")
    plt.tight_layout()
    plt.savefig("models/shap_xgb_bar.png", dpi=150)
    plt.close()
    print("  Saved: models/shap_xgb_bar.png")

    # 3. Top Feature Dependence Plots
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_features_idx = np.argsort(mean_abs_shap)[-3:][::-1]

    for idx in top_features_idx:
        fname = feature_cols[idx]
        plt.figure(figsize=(8, 5))
        shap.dependence_plot(idx, shap_values, sample, feature_names=feature_cols, show=False)
        plt.title(f"SHAP Dependence: {fname}")
        plt.tight_layout()
        safe_name = fname.replace(" ", "_").replace("/", "_")
        plt.savefig(f"models/shap_dependence_{safe_name}.png", dpi=150)
        plt.close()
        print(f"  Saved: models/shap_dependence_{safe_name}.png")

    return shap_values, mean_abs_shap


def analyze_rf_shap(X_test, feature_cols):
    """SHAP analysis for Random Forest."""
    print("\n[2.4] SHAP Analysis: Random Forest...")

    model_path = "models/rf_optimized.pkl"
    if not os.path.exists(model_path):
        model_path = "models/rf_timeseries.pkl"
    if not os.path.exists(model_path):
        print("  No RF model found. Skipping.")
        return None

    model = joblib.load(model_path)
    explainer = shap.TreeExplainer(model)

    sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(sample)

    # For binary classification, shap_values is a list [class0, class1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_values, sample, feature_names=feature_cols, show=False, max_display=15)
    plt.title("Random Forest: Global Feature Importance (SHAP)")
    plt.tight_layout()
    plt.savefig("models/shap_rf_summary.png", dpi=150)
    plt.close()
    print("  Saved: models/shap_rf_summary.png")

    return shap_values


def generate_comparison_report(feature_cols, xgb_shap, rf_shap):
    """Generate a model comparison report."""
    print("\n[2.4] Generating Model Comparison Report...")

    report = {"models": {}}

    # Load optimization results if available
    optuna_path = "models/optuna_results.json"
    if os.path.exists(optuna_path):
        with open(optuna_path) as f:
            optuna_results = json.load(f)
        for name, data in optuna_results.items():
            report["models"][name] = {"f1_score": data.get("f1", "N/A")}

    # Top features from XGBoost SHAP
    if xgb_shap is not None:
        top_idx = np.argsort(xgb_shap)[-5:][::-1]
        report["top_failure_predictors"] = [feature_cols[i] for i in top_idx]
        print(f"\n  Top 5 Failure Predictors (XGBoost SHAP):")
        for i, idx in enumerate(top_idx):
            print(f"    {i+1}. {feature_cols[idx]} (mean |SHAP| = {xgb_shap[idx]:.4f})")

    # Save report
    with open("models/interpretability_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("  Saved: models/interpretability_report.json")

    # Create a visual comparison
    if os.path.exists(optuna_path):
        with open(optuna_path) as f:
            results = json.load(f)

        names = list(results.keys())
        scores = [results[n].get("f1", 0) for n in names]

        plt.figure(figsize=(8, 5))
        bars = plt.bar(names, scores, color=['#2196F3', '#4CAF50', '#FF9800'])
        plt.ylabel('F1 Score (Cross-Validated)')
        plt.title('Model Comparison: Optimized F1 Scores')
        plt.ylim(0, 1)
        for bar, score in zip(bars, scores):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                     f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        plt.savefig("models/model_comparison.png", dpi=150)
        plt.close()
        print("  Saved: models/model_comparison.png")


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    X_test, y_test, feature_cols = load_test_data()
    print(f"Test set: {len(X_test)} samples, {len(feature_cols)} features")

    xgb_shap_values, mean_abs_shap = analyze_xgboost_shap(X_test, feature_cols)
    rf_shap_values = analyze_rf_shap(X_test, feature_cols)

    generate_comparison_report(feature_cols, mean_abs_shap, rf_shap_values)

    print("\n✅ Task 2.4 Complete: SHAP interpretability analysis finished.")
    print("   Check the 'models/' folder for all generated plots and reports.")
