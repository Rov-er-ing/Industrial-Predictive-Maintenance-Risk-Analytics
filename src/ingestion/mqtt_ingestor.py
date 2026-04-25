import paho.mqtt.client as mqtt
import json
import psycopg2
import os
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# DB Configuration
DB_NAME = os.getenv("DB_NAME", "maintenance_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

# MQTT Configuration
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPICS = [
    ("industrial/sensors", 0),
    ("industrial/telemetry", 0),
    ("industrial/errors", 0),
    ("industrial/maint", 0),
    ("industrial/failures", 0)
]

def get_db_connection():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        host=DB_HOST,
        port=DB_PORT
    )

def on_connect(client, userdata, flags, rc):
    print(f"Connected with result code {rc}")
    client.subscribe(MQTT_TOPICS)

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        conn = get_db_connection()
        cur = conn.cursor()
        
        topic = msg.topic
        if topic == "industrial/sensors":
            cur.execute("""
                INSERT INTO sensor_readings (time, equipment_id, type, air_temp, process_temp, rpm, torque, tool_wear, failure, failure_type)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (payload['timestamp'], payload['equipment_id'], payload['type'], payload['air_temp'], 
                  payload['process_temp'], payload['rpm'], payload['torque'], payload['tool_wear'], 
                  payload['failure'], payload['failure_type']))
            
        elif topic == "industrial/telemetry":
            cur.execute("""
                INSERT INTO azure_telemetry (datetime, machineID, volt, rotate, pressure, vibration)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (payload['datetime'], payload['machineID'], payload['volt'], payload['rotate'], 
                  payload['pressure'], payload['vibration']))
            
        elif topic == "industrial/errors":
            cur.execute("""
                INSERT INTO azure_errors (datetime, machineID, errorID)
                VALUES (%s, %s, %s)
            """, (payload['datetime'], payload['machineID'], payload['errorID']))
            
        elif topic == "industrial/maint":
            cur.execute("""
                INSERT INTO azure_maint (datetime, machineID, comp)
                VALUES (%s, %s, %s)
            """, (payload['datetime'], payload['machineID'], payload['comp']))
            
        elif topic == "industrial/failures":
            cur.execute("""
                INSERT INTO azure_failures (datetime, machineID, failure)
                VALUES (%s, %s, %s)
            """, (payload['datetime'], payload['machineID'], payload['failure']))
        
        conn.commit()
        cur.close()
        conn.close()
        
    except Exception as e:
        print(f"Error processing message on {msg.topic}: {e}")

def start_ingestor():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
    except Exception as e:
        print(f"Error connecting to MQTT: {e}")
        return

    print(f"Ingestor listening on multiple industrial topics...")
    client.loop_forever()

if __name__ == "__main__":
    start_ingestor()
