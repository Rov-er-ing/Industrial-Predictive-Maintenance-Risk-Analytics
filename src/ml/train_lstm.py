import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler
import os

class LSTMTrainer:
    def __init__(self, sequence_length=24):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()

    def create_sequences(self, data, target):
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(target[i + self.sequence_length])
        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        model = Sequential([
            LSTM(64, input_shape=input_shape, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, df: pl.DataFrame):
        print("Preparing data for LSTM sequences...")
        # Select features for LSTM
        features = ["volt", "rotate", "pressure", "vibration"]
        data = df.select(features).to_numpy()
        target = df.select("failure_event").to_numpy() # Assuming failure_event column exists
        
        # Scale
        data_scaled = self.scaler.fit_transform(data)
        
        X, y = self.create_sequences(data_scaled, target)
        
        print(f"Sequence Data Shape: {X.shape}")
        
        # Build and Train
        model = self.build_model((X.shape[1], X.shape[2]))
        
        # Optimized for 4GB VRAM
        model.fit(X, y, epochs=10, batch_size=64, validation_split=0.2)
        
        os.makedirs("models", exist_ok=True)
        model.save("models/lstm_azure_v1.h5")
        import joblib
        joblib.dump(self.scaler, "models/azure_scaler.pkl")
        print("LSTM Model and Scaler saved.")

if __name__ == "__main__":
    print("LSTM training script ready. Requires engineered Azure dataframe.")
