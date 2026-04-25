"""
Phase 2 Master Runner: Executes Tasks 2.2, 2.3, and 2.4 in sequence.
Run from project root: python src/ml/run_phase2.py
"""
import sys
import os
import time

# Add src/ml to path so imports work
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def main():
    start = time.time()

    print("=" * 60)
    print("  PHASE 2: MODEL DEVELOPMENT - MASTER RUNNER")
    print("=" * 60)

    # ============================================
    # Task 2.1: Already completed (XGBoost baseline)
    # ============================================
    print("\n[2.1] ✅ Baseline Model (XGBoost on AI4I) — Already completed.")

    # ============================================
    # Task 2.2: Time-Series Model Development
    # ============================================
    print("\n" + "=" * 60)
    print("  TASK 2.2: TIME-SERIES MODEL DEVELOPMENT")
    print("=" * 60)

    from train_timeseries import load_and_merge_azure, engineer_temporal_features, train_time_series_models, train_ttf_regression

    tel, err, maint, fail, mach = load_and_merge_azure()
    pdf = engineer_temporal_features(tel, err, maint, fail, mach)
    X_test, X_test_scaled, y_test, feature_cols, xgb_m, rf_m, mlp_m = train_time_series_models(pdf)
    ttf_model = train_ttf_regression(pdf)

    print("\n✅ Task 2.2 DONE")

    # ============================================
    # Task 2.3: Hyperparameter Optimization
    # ============================================
    print("\n" + "=" * 60)
    print("  TASK 2.3: HYPERPARAMETER OPTIMIZATION (OPTUNA)")
    print("=" * 60)

    from optimize_models import load_prepared_data, optimize_xgboost, optimize_random_forest, optimize_mlp, retrain_best_models

    X, y, feat_cols = load_prepared_data()

    # Use fewer trials for speed (increase for production)
    xgb_best, xgb_score = optimize_xgboost(X, y, n_trials=15)
    rf_best, rf_score = optimize_random_forest(X, y, n_trials=10)
    mlp_best, mlp_score = optimize_mlp(X, y, n_trials=8)

    import json
    os.makedirs("models", exist_ok=True)
    results = {
        "xgboost": {"params": xgb_best, "f1": xgb_score},
        "random_forest": {"params": rf_best, "f1": rf_score},
        "mlp": {"params": mlp_best, "f1": mlp_score}
    }
    with open("models/optuna_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    retrain_best_models(X, y, xgb_best, rf_best, mlp_best)

    print("\n✅ Task 2.3 DONE")

    # ============================================
    # Task 2.4: Model Interpretability (SHAP)
    # ============================================
    print("\n" + "=" * 60)
    print("  TASK 2.4: MODEL INTERPRETABILITY (SHAP)")
    print("=" * 60)

    from interpretability import load_test_data, analyze_xgboost_shap, analyze_rf_shap, generate_comparison_report

    X_test_shap, y_test_shap, feat_cols_shap = load_test_data()
    xgb_shap, mean_abs_shap = analyze_xgboost_shap(X_test_shap, feat_cols_shap)
    rf_shap = analyze_rf_shap(X_test_shap, feat_cols_shap)
    generate_comparison_report(feat_cols_shap, mean_abs_shap, rf_shap)

    print("\n✅ Task 2.4 DONE")

    # ============================================
    # Summary
    # ============================================
    elapsed = time.time() - start
    print("\n" + "=" * 60)
    print("  PHASE 2 COMPLETE")
    print("=" * 60)
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Models saved in: models/")
    print(f"  SHAP plots saved in: models/")
    print(f"\n  Trained Models:")
    print(f"    - models/xgb_timeseries.pkl")
    print(f"    - models/rf_timeseries.pkl")
    print(f"    - models/mlp_timeseries.pkl")
    print(f"    - models/ttf_regressor.pkl")
    print(f"    - models/xgb_optimized.pkl")
    print(f"    - models/rf_optimized.pkl")
    print(f"    - models/mlp_optimized.pkl")
    print(f"\n  Analysis Outputs:")
    print(f"    - models/optuna_results.json")
    print(f"    - models/interpretability_report.json")
    print(f"    - models/shap_xgb_summary.png")
    print(f"    - models/shap_rf_summary.png")
    print(f"    - models/model_comparison.png")


if __name__ == "__main__":
    main()
