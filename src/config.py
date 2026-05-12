import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

class Config:
    # Base paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    MODEL_DIR = PROJECT_ROOT / os.getenv("MODEL_DIR", "models")
    
    # Dataset
    AZURE_PATH = os.getenv("AZURE_DATASET_PATH", "")
    
    # Model Specific Paths
    XGB_OPTIMIZED = PROJECT_ROOT / os.getenv("XGB_OPTIMIZED_PATH", "models/xgb_optimized.pkl")
    RF_OPTIMIZED = PROJECT_ROOT / os.getenv("RF_OPTIMIZED_PATH", "models/rf_optimized.pkl")
    TTF_MODEL = PROJECT_ROOT / os.getenv("TTF_MODEL_PATH", "models/ttf_regressor.pkl")
    SCALER_PATH = PROJECT_ROOT / os.getenv("SCALER_PATH", "models/azure_scaler.pkl")
    
    # API Configuration
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))
    API_URL = os.getenv("API_URL", "http://localhost:8000")
    
    # Industrial Constraints (Safe Ranges)
    VIBRATION_CRITICAL_THRESHOLD = 75.0
    VIBRATION_WARNING_THRESHOLD = 60.0
    ERROR_COUNT_CRITICAL = 4
    
    @classmethod
    def validate_paths(cls):
        """Checks if critical files exist and prints warnings if not."""
        critical_files = [cls.XGB_OPTIMIZED, cls.TTF_MODEL, cls.SCALER_PATH]
        for f in critical_files:
            if not f.exists():
                print(f"Warning: Critical model file not found at {f}")

if __name__ == "__main__":
    print(f"Project Root: {Config.PROJECT_ROOT}")
    print(f"Dataset Path: {Config.AZURE_PATH}")
    Config.validate_paths()
