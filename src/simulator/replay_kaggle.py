import pandas as pd
import paho.mqtt.client as mqtt
import json
import time
from datetime import datetime
import os

# Dataset Path (Update this if needed)
DATASET_PATH = r"C:\Users\Shiti\.cache\kagglehub\datasets\stephanmatzka\predictive-maintenance-dataset-ai4i-2020\versions\2\ai4i2020.csv"

# MQTT Configuration
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "industrial/sensors"

def on_connect(client, userdata, flags, rc):
    print(f"Connected to MQTT Broker with code {rc}")

def replay_dataset(interval=1.0):
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Dataset not found at {DATASET_PATH}")
        return

    df = pd.read_csv(DATASET_PATH)
    
    client = mqtt.Client()
    client.on_connect = on_connect
    
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
    except Exception as e:
        print(f"Failed to connect to MQTT: {e}")
        return

    client.loop_start()

    print(f"Replaying dataset to {MQTT_TOPIC}...")
    
    for _, row in df.iterrows():
        # Determine failure type
        failure_type = "None"
        if row['TWF'] == 1: failure_type = "TWF"
        elif row['HDF'] == 1: failure_type = "HDF"
        elif row['PWF'] == 1: failure_type = "PWF"
        elif row['OSF'] == 1: failure_type = "OSF"
        elif row['RNF'] == 1: failure_type = "RNF"

        data = {
            "timestamp": datetime.now().isoformat(),
            "equipment_id": row['Product ID'],
            "type": row['Type'],
            "air_temp": float(row['Air temperature [K]']),
            "process_temp": float(row['Process temperature [K]']),
            "rpm": float(row['Rotational speed [rpm]']),
            "torque": float(row['Torque [Nm]']),
            "tool_wear": float(row['Tool wear [min]']),
            "failure": int(row['Machine failure']),
            "failure_type": failure_type
        }
        
        client.publish(MQTT_TOPIC, json.dumps(data))
        time.sleep(interval)

    client.loop_stop()
    client.disconnect()

if __name__ == "__main__":
    replay_dataset(interval=0.1) # Replay faster for testing
