import kagglehub
import os

try:
    print("Downloading dataset...")
    path = kagglehub.dataset_download("arnabbiswas1/microsoft-azure-predictive-maintenance")
    print(f"Dataset downloaded to: {path}")
    with open("azure_path.txt", "w") as f:
        f.write(path)
except Exception as e:
    print(f"Error: {e}")
