"""
Phase 3: Model Evaluation & Validation
Generates:
  1. Confusion Matrices
  2. Precision-Recall Curves
  3. ROC Curves
  4. Detailed Performance Metrics (JSON)
"""
import joblib
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    precision_recall_curve, roc_curve, auc, f1_score
)
import json

def load_data_and_models():
    """Load test data and all optimized models."""
    from train_timeseries import load_and_merge_azure, engineer_temporal_features
    
    print("[3.1] Loading test data and models for evaluation...")
    tel, err, maint, fail, mach = load_and_merge_azure()
    pdf = engineer_temporal_features(tel, err, maint, fail, mach)
    
    drop_cols = ["datetime", "machineID", "model", "target"]
    feature_cols = [c for c in pdf.columns if c not in drop_cols]
    pdf = pdf.dropna(subset=feature_cols)
    
    # Use same split as training
    split_idx = int(len(pdf) * 0.8)
    X_test = pdf[feature_cols].iloc[split_idx:].values
    y_test = pdf["target"].iloc[split_idx:].values
    
    # Load Models
    models = {}
    if os.path.exists("models/mlp_optimized.pkl"):
        models["MLP (Neural)"] = joblib.load("models/mlp_optimized.pkl")
    if os.path.exists("models/xgb_optimized.pkl"):
        models["XGBoost"] = joblib.load("models/xgb_optimized.pkl")
    
    # Load Scaler for MLP
    scaler = None
    if os.path.exists("models/azure_scaler_optimized.pkl"):
        scaler = joblib.load("models/azure_scaler_optimized.pkl")
        
    return X_test, y_test, models, scaler

def evaluate_models(X_test, y_test, models, scaler):
    """Run full evaluation suite."""
    os.makedirs("models/evaluation", exist_ok=True)
    summary = {}
    
    plt.figure(figsize=(15, 10))
    
    for i, (name, model) in enumerate(models.items()):
        print(f"  Evaluating {name}...")
        
        # Scale if it's the MLP
        X_eval = X_test
        if "MLP" in name and scaler:
            X_eval = scaler.transform(X_test)
            
        y_pred = model.predict(X_eval)
        y_prob = model.predict_proba(X_eval)[:, 1]
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.subplot(2, 2, i+1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix: {name}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # 2. Precision-Recall Curve
        prec, rec, thresholds = precision_recall_curve(y_test, y_prob)
        
        # 3. Store Metrics
        summary[name] = {
            "f1_score": float(f1_score(y_test, y_pred)),
            "precision": float(cm[1,1] / (cm[1,1] + cm[0,1])) if (cm[1,1] + cm[0,1]) > 0 else 0,
            "recall": float(cm[1,1] / (cm[1,1] + cm[1,0])),
            "false_alarms": int(cm[0,1])
        }
    
    plt.tight_layout()
    plt.savefig("models/evaluation/confusion_matrices.png")
    plt.close()
    
    # 4. Global ROC Curve
    plt.figure(figsize=(10, 7))
    for name, model in models.items():
        X_eval = X_test
        if "MLP" in name and scaler:
            X_eval = scaler.transform(X_test)
        y_prob = model.predict_proba(X_eval)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc(fpr, tpr):.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves: Optimized Models')
    plt.legend()
    plt.grid(True)
    plt.savefig("models/evaluation/roc_curves.png")
    plt.close()
    
    with open("models/evaluation/evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print("\n✅ Evaluation Summary:")
    print(json.dumps(summary, indent=2))
    print("\nPlots saved in models/evaluation/")

if __name__ == "__main__":
    X, y, mods, sc = load_data_and_models()
    if mods:
        evaluate_models(X, y, mods, sc)
    else:
        print("No models found to evaluate. Please run Phase 2 training first.")
