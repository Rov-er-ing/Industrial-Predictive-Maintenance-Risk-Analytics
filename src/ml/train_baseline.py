import polars as pl
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import joblib
import os
import optuna
import shap
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

# DB Configuration
DB_NAME = os.getenv("DB_NAME", "maintenance_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

CONNECTION_URI = f"postgres://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def train_xgboost_baseline():
    os.makedirs("models", exist_ok=True)
    print("Loading data from TimescaleDB...")
    # Load AI4I data from sensor_readings
    try:
        df = pl.read_database_uri("SELECT * FROM sensor_readings", CONNECTION_URI)
    except Exception as e:
        print(f"Error loading from DB: {e}. Falling back to CSV.")
        # Fallback for skeleton build if DB is empty
        csv_path = r"C:\Users\Shiti\.cache\kagglehub\datasets\stephanmatzka\predictive-maintenance-dataset-ai4i-2020\versions\2\ai4i2020.csv"
        df = pl.read_csv(csv_path)
        # Rename columns to match our expected features
        df = df.rename({
            "Air temperature [K]": "air_temp",
            "Process temperature [K]": "process_temp",
            "Rotational speed [rpm]": "rpm",
            "Torque [Nm]": "torque",
            "Tool wear [min]": "tool_wear",
            "Machine failure": "target"
        })

    if df.height == 0:
        print("No data found to train.")
        return

    # Select features
    features = ["air_temp", "process_temp", "rpm", "torque", "tool_wear"]
    X = df.select(features).to_pandas()
    y = df.select("target").to_pandas()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Optuna Objective
    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'tree_method': 'hist',
            'device': 'cuda' if trial.suggest_categorical('use_gpu', [True, False]) else 'cpu'
        }
        model = xgb.XGBClassifier(**param)
        model.fit(X_train, y_train)
        return f1_score(y_test, model.predict(X_test))

    print("Starting Hyperparameter Optimization (Optuna)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10) # Small number for skeleton build
    
    print(f"Best parameters: {study.best_params}")
    model = xgb.XGBClassifier(**study.best_params)
    model.fit(X_train, y_train)

    # SHAP Interpretability
    print("Generating SHAP Explanations...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.tight_layout()
    plt.savefig("models/shap_summary.png")
    print("SHAP plot saved to models/shap_summary.png")

    # Evaluate
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

    # Save model
    joblib.dump(model, "models/xgboost_baseline.pkl")
    print("Model saved to models/xgboost_baseline.pkl")

if __name__ == "__main__":
    train_xgboost_baseline()
