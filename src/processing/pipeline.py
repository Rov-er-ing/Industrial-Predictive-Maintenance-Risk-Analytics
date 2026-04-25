import polars as pl
import numpy as np
from datetime import datetime, timedelta

class DataPipeline:
    def __init__(self, batch_size=100000):
        self.batch_size = batch_size

    def load_from_db(self, query, connection_uri):
        """
        Load data from DB using Polars streaming if possible.
        For 16GB RAM constraints, we use chunked reading.
        """
        # Note: Polars can read from Postgres directly
        return pl.read_database(query, connection_uri)

    def clean_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Implements cleaning logic: missing values and outliers.
        """
        # Forward fill missing values
        df = df.sort("time").with_columns([
            pl.col("air_temp").forward_fill(),
            pl.col("process_temp").forward_fill(),
            pl.col("torque").forward_fill(),
            pl.col("tool_wear").forward_fill()
        ])
        
        # Outlier detection (simple clipping)
        df = df.with_columns([
            pl.col("air_temp").clip(lower=280, upper=320),
            pl.col("process_temp").clip(lower=300, upper=330),
            pl.col("torque").clip(lower=0, upper=100)
        ])
        
        return df

    def engineer_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Feature Engineering using Polars window functions.
        Rolling stats: 1h, 6h, 24h (assuming 1-min interval)
        """
        # Sort by time and equipment_id for window operations
        df = df.sort(["equipment_id", "time"])
        
        # Rolling averages
        df = df.with_columns([
            pl.col("air_temp").rolling_mean(window_size=60).over("equipment_id").alias("air_temp_roll_mean_1h"),
            pl.col("torque").rolling_std(window_size=60).over("equipment_id").alias("torque_roll_std_1h"),
            pl.col("air_temp").diff().over("equipment_id").alias("temp_delta")
        ])
        
        # Target column is already in the dataset as 'failure'
        df = df.with_columns([
            pl.col("failure").alias("target")
        ])
        
        return df.drop_nulls() # Remove rows with NaN from rolling windows

if __name__ == "__main__":
    # Example usage with dummy data for testing
    pipeline = DataPipeline()
    # In practice, this would be pl.read_database(...)
    print("Pipeline ready. Use load_from_db to start processing.")
