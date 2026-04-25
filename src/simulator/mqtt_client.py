import paho.mqtt.client as mqtt
import json
import time
from engine import IndustrialEquipmentSimulator

# MQTT Configuration
MQTT_BROKER = "localhost"  # Change to your broker IP if needed
MQTT_PORT = 1883
MQTT_TOPIC = "industrial/sensors"

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT Broker!")
    else:
        print(f"Failed to connect, return code {rc}")

def start_publisher(num_equipment=5, interval=1.0):
    client = mqtt.Client()
    client.on_connect = on_connect
    
    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
    except Exception as e:
        print(f"Error connecting to MQTT Broker: {e}")
        return

    client.loop_start()

    simulators = [IndustrialEquipmentSimulator(f"EQ_{i:03d}") for i in range(num_equipment)]
    
    print(f"Starting Industrial Simulator on topic: {MQTT_TOPIC}")
    try:
        while True:
            for sim in simulators:
                data = sim.generate_reading()
                payload = json.dumps(data)
                client.publish(MQTT_TOPIC, payload)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Stopping publisher...")
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    start_publisher()
