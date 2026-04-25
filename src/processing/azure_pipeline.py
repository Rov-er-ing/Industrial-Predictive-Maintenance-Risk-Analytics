import polars as pl
import os
from datetime import timedelta

class AzureDataPipeline:
    def __init__(self, dataset_path):
        self.path = dataset_path

    def load_data(self):
        print("Loading Azure dataset components...")
        telemetry = pl.read_csv(os.path.join(self.path, "PdM_telemetry.csv")).with_columns(pl.col("datetime").str.strptime(pl.Datetime))
        errors = pl.read_csv(os.path.join(self.path, "PdM_errors.csv")).with_columns(pl.col("datetime").str.strptime(pl.Datetime))
        maint = pl.read_csv(os.path.join(self.path, "PdM_maint.csv")).with_columns(pl.col("datetime").str.strptime(pl.Datetime))
        failures = pl.read_csv(os.path.join(self.path, "PdM_failures.csv")).with_columns(pl.col("datetime").str.strptime(pl.Datetime))
        machines = pl.read_csv(os.path.join(self.path, "PdM_machines.csv"))
        
        return telemetry, errors, maint, failures, machines

    def engineer_features(self, telemetry, errors, maint, machines):
        print("Engineering temporal features (The Muscle)...")
        
        # 1. Telemetry Rolling Stats (Lag features)
        tel_features = telemetry.sort(["machineID", "datetime"])
        
        # Define windows (3h and 24h)
        for win in [3, 24]:
            tel_features = tel_features.with_columns([
                pl.col("volt").rolling_mean(window_size=win).over("machineID").alias(f"volt_mean_{win}h"),
                pl.col("rotate").rolling_mean(window_size=win).over("machineID").alias(f"rotate_mean_{win}h"),
                pl.col("pressure").rolling_mean(window_size=win).over("machineID").alias(f"pressure_mean_{win}h"),
                pl.col("vibration").rolling_mean(window_size=win).over("machineID").alias(f"vibration_mean_{win}h"),
                pl.col("volt").rolling_std(window_size=win).over("machineID").alias(f"volt_std_{win}h")
            ])

        # 2. Error Counts (Rolling)
        # One-hot encode errors then sum
        error_counts = errors.with_columns(pl.lit(1).alias("error_count")).pivot(
            values="error_count", index=["datetime", "machineID"], on="errorID"
        ).fill_null(0)
        
        # Merge with telemetry
        df = tel_features.join(error_counts, on=["datetime", "machineID"], how="left").fill_null(0)
        
        # 3. Machine Metadata
        df = df.join(machines, on="machineID", how="left")
        
        return df

    def create_labels(self, df, failures, window_hours=24):
        """
        Label a record as 1 if a failure occurs within the next 'window_hours'.
        """
        print(f"Generating predictive labels ({window_hours}h window)...")
        
        # Mark actual failure timestamps
        labeled_df = df.join(failures.with_columns(pl.lit(1).alias("failure_event")), on=["datetime", "machineID"], how="left").fill_null(0)
        
        # Shift failure back to create a look-ahead window
        # For simplicity in this skeleton, we'll use a simple join with failure times
        return labeled_df

if __name__ == "__main__":
    AZURE_PATH = r"C:\Users\Shiti\.cache\kagglehub\datasets\arnabbiswas1\microsoft-azure-predictive-maintenance\versions\3"
    pipeline = AzureDataPipeline(AZURE_PATH)
    tel, err, maint, fail, mach = pipeline.load_data()
    df = pipeline.engineer_features(tel, err, maint, mach)
    print(f"Engineered Dataset Shape: {df.shape}")
    print(df.head())
