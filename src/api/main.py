from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os
import sys

import logging
from datetime import datetime

from src.config import Config

# Ensure log directory exists
os.makedirs(Config.MODEL_DIR / "monitoring", exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.MODEL_DIR / "monitoring" / "api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("PredictionAPI")

# Standardized imports from src
try:
    from src.monitoring.guardian import log_prediction, initialize_monitoring
    initialize_monitoring()
    logger.info("Guardian monitoring system initialized.")
except ImportError:
    logger.warning("Guardian monitoring system not found. Logging will be disabled.")
    def log_prediction(m, r, t): pass

app = FastAPI(title="Industrial Predictive Maintenance API", version="2.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Model Loading ---
MODELS = {}
PATHS = {
    "skeleton": "models/xgboost_baseline.pkl",
    "muscle": Config.XGB_OPTIMIZED,
    "ttf": Config.TTF_MODEL
}

for name, path in PATHS.items():
    if os.path.exists(path):
        MODELS[name] = joblib.load(path)
        logger.info(f"Loaded {name} model from {path}")

# --- Data Models ---
class SkeletonReading(BaseModel):
    air_temp: float
    process_temp: float
    rpm: float
    torque: float
    tool_wear: float

class MuscleReading(BaseModel):
    volt: float
    rotate: float
    pressure: float
    vibration: float
    volt_avg_3h: float
    rotate_avg_3h: float
    pressure_avg_3h: float
    vibration_avg_3h: float
    volt_avg_24h: float
    rotate_avg_24h: float
    pressure_avg_24h: float
    vibration_avg_24h: float
    volt_std_3h: float
    rotate_std_3h: float
    pressure_std_3h: float
    vibration_std_3h: float
    volt_std_24h: float
    rotate_std_24h: float
    pressure_std_24h: float
    vibration_std_24h: float
    volt_diff: float
    rotate_diff: float
    pressure_diff: float
    vibration_diff: float
    error_count: int
    age: int

@app.get("/")
def read_root():
    return {
        "status": "online",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": list(MODELS.keys())
    }

@app.get("/health")
def health_check():
    health = {
        "status": "healthy",
        "uptime": "active",
        "models": {name: "READY" for name in MODELS.keys()},
        "missing_models": [name for name, path in PATHS.items() if name not in MODELS]
    }
    if not MODELS:
        health["status"] = "degraded"
        logger.error("API Health: Degraded - No models loaded")
    return health

@app.post("/predict/skeleton")
def predict_skeleton(reading: SkeletonReading):
    if "skeleton" not in MODELS:
        raise HTTPException(status_code=503, detail="Skeleton model not loaded")
    try:
        features = np.array([[reading.air_temp, reading.process_temp, reading.rpm, reading.torque, reading.tool_wear]])
        prob = MODELS["skeleton"].predict_proba(features)[0].tolist()
        pred = int(MODELS["skeleton"].predict(features)[0])
        return {
            "model": "XGBoost-Skeleton",
            "failure_predicted": bool(pred),
            "failure_probability": round(prob[1], 4),
            "risk_level": "High" if prob[1] > 0.3 else "Medium" if prob[1] > 0.1 else "Low"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/muscle")
def predict_muscle(reading: MuscleReading):
    if "muscle" not in MODELS:
        raise HTTPException(status_code=503, detail="Muscle model not loaded")
    try:
        features = np.array([[
            reading.volt, reading.rotate, reading.pressure, reading.vibration,
            reading.volt_avg_3h, reading.rotate_avg_3h, reading.pressure_avg_3h, reading.vibration_avg_3h,
            reading.volt_std_3h, reading.rotate_std_3h, reading.pressure_std_3h, reading.vibration_std_3h,
            reading.volt_avg_24h, reading.rotate_avg_24h, reading.pressure_avg_24h, reading.vibration_avg_24h,
            reading.volt_std_24h, reading.rotate_std_24h, reading.pressure_std_24h, reading.vibration_std_24h,
            reading.volt_diff, reading.rotate_diff, reading.pressure_diff, reading.vibration_diff,
            reading.error_count, reading.age
        ]])
        
        prob = MODELS["muscle"].predict_proba(features)[0].tolist()
        ttf = -1.0
        if "ttf" in MODELS:
            ttf = float(MODELS["ttf"].predict(features)[0])
            
        res = {
            "model": "XGBoost-Optimized-Muscle",
            "failure_predicted_24h": bool(prob[1] > 0.05),
            "failure_probability": round(prob[1], 4),
            "risk_level": "High" if prob[1] > 0.05 else "Medium" if prob[1] > 0.02 else "Low",
            "time_to_failure_hours": round(ttf, 2) if ttf > 0 else "Normal Operation"
        }

        # --- Heuristic Safety Override (Industrial Best Practice) ---
        # If sensors are in a physically dangerous range, override AI for safety.
        if reading.vibration > Config.VIBRATION_CRITICAL_THRESHOLD or reading.error_count >= Config.ERROR_COUNT_CRITICAL:
            res["risk_level"] = "High"
            res["failure_predicted_24h"] = True
            res["failure_probability"] = max(res["failure_probability"], 0.95)
            if res["time_to_failure_hours"] == "Normal Operation":
                res["time_to_failure_hours"] = 12.0
        elif reading.vibration > Config.VIBRATION_WARNING_THRESHOLD or reading.error_count >= 2:
            if res["risk_level"] == "Low":
                res["risk_level"] = "Medium"
                res["failure_probability"] = max(res["failure_probability"], 0.45)

        # Log for monitoring
        log_prediction("SIM-MACHINE-01", res["risk_level"], res["time_to_failure_hours"] if isinstance(res["time_to_failure_hours"], float) else 0)
        
        return res
    except Exception as e:
        logger.error(f"Muscle Prediction Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/alerts")
def get_alerts():
    try:
        from src.monitoring.guardian import get_active_alerts
        return {"alerts": get_active_alerts()}
    except Exception as e:
        logger.error(f"Alerts Fetch Error: {str(e)}")
        return {"alerts": [], "error": str(e)}

@app.post("/alerts/resolve")
def resolve_alerts():
    try:
        from src.monitoring.guardian import resolve_all_alerts
        resolve_all_alerts()
        return {"status": "success", "message": "All alerts resolved"}
    except Exception as e:
        logger.error(f"Alerts Resolve Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
