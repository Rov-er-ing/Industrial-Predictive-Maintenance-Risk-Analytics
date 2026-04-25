import polars as pl
import os
import matplotlib.pyplot as plt
import seaborn as sns

class DataProfiler:
    def __init__(self, ai4i_path, azure_path):
        self.ai4i_path = ai4i_path
        self.azure_path = azure_path

    def profile_ai4i(self):
        print("Profiling AI4I 2020 Dataset...")
        df = pl.read_csv(self.ai4i_path)
        
        # Summary Stats
        stats = df.describe()
        print(stats)
        
        # Class Imbalance
        failure_counts = df.group_by("Machine failure").count()
        print("Failure Distribution:")
        print(failure_counts)
        
        # Correlations (numeric only)
        numeric_cols = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
        corr = df.select(numeric_cols).to_pandas().corr()
        print("Correlation Matrix:")
        print(corr)

    def profile_azure(self):
        print("\nProfiling Microsoft Azure Dataset (Telemetry)...")
        tel_path = os.path.join(self.azure_path, "PdM_telemetry.csv")
        fail_path = os.path.join(self.azure_path, "PdM_failures.csv")
        
        telemetry = pl.read_csv(tel_path)
        failures = pl.read_csv(fail_path)
        
        print(f"Telemetry Records: {telemetry.height}")
        print(f"Failure Events: {failures.height}")
        
        # Check machine distribution
        machine_dist = telemetry.group_by("machineID").count()
        print(f"Unique Machines: {machine_dist.height}")
        
        # Failure types
        fail_types = failures.group_by("failure").count()
        print("Failure Type Distribution:")
        print(fail_types)

if __name__ == "__main__":
    AI4I_CSV = r"C:\Users\Shiti\.cache\kagglehub\datasets\stephanmatzka\predictive-maintenance-dataset-ai4i-2020\versions\2\ai4i2020.csv"
    AZURE_DIR = r"C:\Users\Shiti\.cache\kagglehub\datasets\arnabbiswas1\microsoft-azure-predictive-maintenance\versions\3"
    
    profiler = DataProfiler(AI4I_CSV, AZURE_DIR)
    profiler.profile_ai4i()
    profiler.profile_azure()
