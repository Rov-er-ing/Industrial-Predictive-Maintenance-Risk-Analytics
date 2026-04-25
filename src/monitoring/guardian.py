"""
Task 4.3: Monitoring & Alerting (The "Guardian" System)
Handles:
  1. Alert Persistence (Local DB/JSON)
  2. Alert Thresholding (Time-based filtering)
  3. Production Logging
"""
import json
import os
from datetime import datetime

ALERT_LOG = "models/monitoring/alerts.json"
PERFORMANCE_LOG = "models/monitoring/performance.csv"

def initialize_monitoring():
    os.makedirs("models/monitoring", exist_ok=True)
    if not os.path.exists(ALERT_LOG):
        with open(ALERT_LOG, "w") as f:
            json.dump([], f)
    
    if not os.path.exists(PERFORMANCE_LOG):
        with open(PERFORMANCE_LOG, "w") as f:
            f.write("timestamp,machineID,risk_level,ttf_prediction\n")

def log_prediction(machine_id, risk_level, ttf):
    """Log prediction for audit/monitoring."""
    initialize_monitoring()
    timestamp = datetime.now().isoformat()
    
    with open(PERFORMANCE_LOG, "a") as f:
        f.write(f"{timestamp},{machine_id},{risk_level},{ttf}\n")
    
    # Simple Alerting Logic: If Risk is High, create an Alert Ticket
    if risk_level == "High":
        create_alert(machine_id, risk_level, ttf)

def create_alert(machine_id, risk_level, ttf):
    """Generate a maintenance alert ticket."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alert = {
        "alert_id": f"ALT-{int(datetime.now().timestamp())}",
        "timestamp": timestamp,
        "machine_id": machine_id,
        "severity": "CRITICAL",
        "message": f"High risk of failure detected. Estimated TTF: {ttf} hours.",
        "status": "OPEN"
    }
    
    with open(ALERT_LOG, "r+") as f:
        alerts = json.load(f)
        # Check if we already have an open alert for this machine to avoid spam
        recent_open = [a for a in alerts if a["machine_id"] == machine_id and a["status"] == "OPEN"]
        if not recent_open:
            alerts.append(alert)
            f.seek(0)
            json.dump(alerts, f, indent=2)
            print(f"🚨 [ALERT] Critical ticket created for {machine_id}")

def get_active_alerts():
    if not os.path.exists(ALERT_LOG):
        return []
    with open(ALERT_LOG, "r") as f:
        return [a for a in json.load(f) if a["status"] == "OPEN"]

def resolve_all_alerts():
    """Clear all active alerts (for demo purposes)."""
    if not os.path.exists(ALERT_LOG):
        return
    with open(ALERT_LOG, "r+") as f:
        alerts = json.load(f)
        for a in alerts:
            a["status"] = "RESOLVED"
        f.seek(0)
        json.dump(alerts, f, indent=2)
        f.truncate()

if __name__ == "__main__":
    # Test simulation
    print("Testing Guardian Monitoring System...")
    log_prediction("MACHINE-001", "High", 12.5)
    print("Active Alerts:", len(get_active_alerts()))
