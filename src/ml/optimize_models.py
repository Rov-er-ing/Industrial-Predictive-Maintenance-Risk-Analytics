"""
Task 2.3: Hyperparameter Optimization (Optuna)
Unified optimization for all models: XGBoost, Random Forest, and MLP.
Uses the Azure dataset with temporal features for maximum signal.
"""
import polars as pl
import pandas as pd
import numpy as np
import optuna
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import f1_score
import xgboost as xgb
import joblib
import os
import json

# Suppress Optuna info logs for cleaner output
optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_prepared_data():
    """Load the Azure dataset and prepare features (reuses train_timeseries logic)."""
    from train_timeseries import load_and_merge_azure, engineer_temporal_features

    tel, err, maint, fail, mach = load_and_merge_azure()
    pdf = engineer_temporal_features(tel, err, maint, fail, mach)

    drop_cols = ["datetime", "machineID", "model", "target"]
    feature_cols = [c for c in pdf.columns if c not in drop_cols]
    pdf = pdf.dropna(subset=feature_cols)

    X = pdf[feature_cols].values
    y = pdf["target"].values

    return X, y, feature_cols


def optimize_xgboost(X, y, n_trials=30):
    """Optimize XGBoost hyperparameters."""
    print("\n[2.3] Optimizing XGBoost...")

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'scale_pos_weight': (y == 0).sum() / max((y == 1).sum(), 1),
            'tree_method': 'hist',
            'random_state': 42
        }
        model = xgb.XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction='maximize', study_name='xgboost_opt')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"  Best XGBoost F1: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params, study.best_value


def optimize_random_forest(X, y, n_trials=20):
    """Optimize Random Forest hyperparameters."""
    print("\n[2.3] Optimizing Random Forest...")

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
        model = RandomForestClassifier(**params)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction='maximize', study_name='rf_opt')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"  Best RF F1: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params, study.best_value


def optimize_mlp(X, y, n_trials=15):
    """Optimize MLP Neural Network hyperparameters."""
    print("\n[2.3] Optimizing MLP Neural Network...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    def objective(trial):
        n_layers = trial.suggest_int('n_layers', 1, 4)
        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_int(f'n_units_l{i}', 16, 256))

        params = {
            'hidden_layer_sizes': tuple(layers),
            'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
            'alpha': trial.suggest_float('alpha', 1e-5, 1e-1, log=True),
            'learning_rate_init': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'solver': 'adam',
            'max_iter': 150,
            'early_stopping': True,
            'random_state': 42
        }
        model = MLPClassifier(**params)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='f1', n_jobs=-1)
        return scores.mean()

    study = optuna.create_study(direction='maximize', study_name='mlp_opt')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"  Best MLP F1: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params, study.best_value


def retrain_best_models(X, y, xgb_params, rf_params, mlp_params):
    """Retrain all models with optimized parameters and save."""
    os.makedirs("models", exist_ok=True)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # XGBoost
    xgb_params['tree_method'] = 'hist'
    xgb_params['scale_pos_weight'] = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    best_xgb = xgb.XGBClassifier(**xgb_params, random_state=42)
    best_xgb.fit(X_train, y_train)
    joblib.dump(best_xgb, "models/xgb_optimized.pkl")

    # Random Forest
    rf_params['class_weight'] = 'balanced'
    rf_params['n_jobs'] = -1
    best_rf = RandomForestClassifier(**rf_params, random_state=42)
    best_rf.fit(X_train, y_train)
    joblib.dump(best_rf, "models/rf_optimized.pkl")

    # MLP
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    n_layers = mlp_params.pop('n_layers', 2)
    layers = tuple(mlp_params.pop(f'n_units_l{i}', 64) for i in range(n_layers))
    lr = mlp_params.pop('lr', 0.001)
    best_mlp = MLPClassifier(
        hidden_layer_sizes=layers,
        learning_rate_init=lr,
        solver='adam',
        max_iter=200,
        early_stopping=True,
        random_state=42,
        **{k: v for k, v in mlp_params.items() if k in ['activation', 'alpha']}
    )
    best_mlp.fit(X_train_s, y_train)
    joblib.dump(best_mlp, "models/mlp_optimized.pkl")
    joblib.dump(scaler, "models/azure_scaler_optimized.pkl")

    print("\n✅ All optimized models saved to models/")
    return best_xgb, best_rf, best_mlp


if __name__ == "__main__":
    X, y, feature_cols = load_prepared_data()
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {y.mean()*100:.2f}% positive class")

    xgb_best, xgb_score = optimize_xgboost(X, y, n_trials=30)
    rf_best, rf_score = optimize_random_forest(X, y, n_trials=20)
    mlp_best, mlp_score = optimize_mlp(X, y, n_trials=15)

    # Save optimization results
    os.makedirs("models", exist_ok=True)
    results = {
        "xgboost": {"params": xgb_best, "f1": xgb_score},
        "random_forest": {"params": rf_best, "f1": rf_score},
        "mlp": {"params": mlp_best, "f1": mlp_score}
    }
    with open("models/optuna_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n--- Optimization Summary ---")
    print(f"  XGBoost F1:       {xgb_score:.4f}")
    print(f"  Random Forest F1: {rf_score:.4f}")
    print(f"  MLP (Neural) F1:  {mlp_score:.4f}")
    print(f"  Winner: {'XGBoost' if xgb_score >= max(rf_score, mlp_score) else 'RF' if rf_score >= mlp_score else 'MLP'}")

    retrain_best_models(X, y, xgb_best, rf_best, mlp_best)
    print("\n✅ Task 2.3 Complete: Hyperparameter optimization finished.")
