"""
Emergency Fix: Correcting Overfitting to "Normal" State
1. Applies Random Oversampling to balance failure classes.
2. Constrains max_depth to improve generalization.
3. Implements stronger regularization.
"""
import polars as pl
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# Import existing feature engineering logic
import sys
sys.path.append('src/ml')
from train_timeseries import load_and_merge_azure, engineer_temporal_features

def fix_and_retrain():
    print("[FIX] Loading and engineering data...")
    tel, err, maint, fail, mach = load_and_merge_azure()
    pdf = engineer_temporal_features(tel, err, maint, fail, mach)
    
    drop_cols = ["datetime", "machineID", "model", "target"]
    feature_cols = [c for c in pdf.columns if c not in drop_cols]
    pdf = pdf.dropna(subset=feature_cols)
    
    # --- Class Balancing (Random Oversampling) ---
    print("[FIX] Applying Random Oversampling to balance classes...")
    df_normal = pdf[pdf.target == 0]
    df_failure = pdf[pdf.target == 1]
    
    # Upsample minority class to 10% of majority class (instead of <1%)
    # This provides enough signal without destroying the real-world distribution
    df_failure_upsampled = resample(df_failure, 
                                   replace=True, 
                                   n_samples=len(df_normal) // 10, 
                                   random_state=42)
    
    df_balanced = pd.concat([df_normal, df_failure_upsampled])
    print(f"      Original Failure Rate: {len(df_failure)/len(pdf)*100:.2f}%")
    print(f"      New Balanced Failure Rate: {len(df_failure_upsampled)/len(df_balanced)*100:.2f}%")
    
    X = df_balanced[feature_cols].values
    y = df_balanced["target"].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # --- Constrained XGBoost (Generalization Focused) ---
    print("[FIX] Training Generalized XGBoost (max_depth=6)...")
    model = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=6,             # Reduced from 12 to prevent overfitting
        learning_rate=0.05,
        min_child_weight=5,       # Higher value prevents over-specialization
        gamma=0.2,               # Minimum loss reduction required for a split
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=1,      # Already balanced via oversampling
        tree_method='hist',
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Save the new "Optimized" model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/xgb_optimized.pkl")
    print("✅ Successfully saved models/xgb_optimized.pkl with generalization fixes.")

if __name__ == "__main__":
    fix_and_retrain()
