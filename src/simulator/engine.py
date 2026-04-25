import numpy as np
import pandas as pd
import time
import json
import random
from datetime import datetime, timedelta

class IndustrialEquipmentSimulator:
    def __init__(self, equipment_id, base_temp=60, base_vibration=2.5, base_pressure=100):
        self.equipment_id = equipment_id
        self.base_temp = base_temp
        self.base_vibration = base_vibration
        self.base_pressure = base_pressure
        self.state = "NORMAL"  # NORMAL, DEGRADING, FAILURE
        self.cycles_to_failure = random.randint(1000, 5000)
        self.current_cycle = 0
        self.degradation_start = self.cycles_to_failure * 0.7
        
    def generate_reading(self):
        self.current_cycle += 1
        
        # Base noise
        temp_noise = np.random.normal(0, 0.5)
        vib_noise = np.random.normal(0, 0.1)
        press_noise = np.random.normal(0, 1.0)
        
        # Degradation logic
        if self.current_cycle > self.degradation_start:
            self.state = "DEGRADING"
            # Linear increase in temp and vibration as failure approaches
            deg_factor = (self.current_cycle - self.degradation_start) / (self.cycles_to_failure - self.degradation_start)
            temp_noise += deg_factor * 15  # Up to 15 degrees increase
            vib_noise += deg_factor * 2.0   # Up to 2.0 Hz increase
            
        if self.current_cycle >= self.cycles_to_failure:
            self.state = "FAILURE"
            # Reset or flatline logic can be added here
            self.current_cycle = 0 # Simulate repair and restart
            self.state = "NORMAL"
            self.cycles_to_failure = random.randint(1000, 5000)
            self.degradation_start = self.cycles_to_failure * 0.7

        reading = {
            "timestamp": datetime.now().isoformat(),
            "equipment_id": self.equipment_id,
            "temperature": round(self.base_temp + temp_noise, 2),
            "vibration": round(self.base_vibration + vib_noise, 3),
            "pressure": round(self.base_pressure + press_noise, 2),
            "rpm": round(1500 + np.random.normal(0, 10), 0),
            "state": self.state
        }
        return reading

def run_simulation(num_equipment=5, interval=1.0):
    simulators = [IndustrialEquipmentSimulator(f"EQ_{i:03d}") for i in range(num_equipment)]
    
    print(f"Starting simulation for {num_equipment} units...")
    try:
        while True:
            for sim in simulators:
                data = sim.generate_reading()
                # In a real scenario, we'd publish to MQTT here
                print(f"PUBLISH: {json.dumps(data)}")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Simulation stopped.")

if __name__ == "__main__":
    run_simulation()
