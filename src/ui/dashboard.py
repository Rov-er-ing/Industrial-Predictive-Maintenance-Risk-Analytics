import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

from src.config import Config

# --- Page Config ---
st.set_page_config(
    page_title="Industrial Health Control Center",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Theme / CSS ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    .status-card {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 10px;
        border: 1px solid #30363d;
    }
    .low-risk { background-color: rgba(35, 134, 54, 0.1); border-left: 5px solid #238636; }
    .medium-risk { background-color: rgba(210, 153, 34, 0.1); border-left: 5px solid #d29922; }
    .high-risk { background-color: rgba(218, 54, 51, 0.1); border-left: 5px solid #da3633; }
    </style>
""", unsafe_allow_html=True)

# --- Helpers ---
API_URL = Config.API_URL

def get_prediction_muscle(data):
    try:
        response = requests.post(f"{API_URL}/predict/muscle", json=data)
        return response.json()
    except:
        return None

def get_prediction_skeleton(data):
    try:
        response = requests.post(f"{API_URL}/predict/skeleton", json=data)
        return response.json()
    except:
        return None

def get_system_alerts():
    try:
        response = requests.get(f"{API_URL}/alerts")
        return response.json().get("alerts", [])
    except:
        return []

def load_interpretability():
    try:
        import json
        with open("models/interpretability_report.json", "r") as f:
            return json.load(f)
    except:
        return {"top_failure_predictors": ["rotate_avg_24h", "volt_avg_24h", "pressure_avg_24h"]}

# --- Data Load ---
report = load_interpretability()
top_predictors = report.get("top_failure_predictors", [])

def generate_diagnostic_summary(res, input_data):
    risk = res['risk_level']
    ttf = res['time_to_failure_hours']
    
    # What Happened
    if risk == "Low":
        what = "System is operating within normal parameters. No immediate failure risk detected."
    elif risk == "Medium":
        what = "System is showing early signs of degradation. Operating efficiency may be reduced."
    else:
        what = "CRITICAL: System health is severely compromised. Immediate failure is highly probable."
        
    # Why Happened
    why_list = []
    if input_data['rotate'] > 2500: why_list.append("Excessive rotation speed (RPM) causing mechanical stress.")
    if input_data['vibration'] > 7: why_list.append("High vibration levels suggesting bearing wear or misalignment.")
    if input_data['pressure'] > 130: why_list.append("Over-pressurization in the system.")
    if input_data['age'] > 60: why_list.append("Extended equipment age (fatigue).")
    
    why = " ".join(why_list) if why_list else "Degradation is likely due to cumulative operational load over time."
    
    # How to Resolve
    if risk == "Low":
        resolve = "Continue standard monitoring schedule. No corrective action required."
    elif risk == "Medium":
        resolve = "Schedule inspection of mechanical components. Check lubrication and sensor calibration."
    else:
        resolve = "SHUTDOWN RECOMMENDED. Perform immediate overhaul of affected asset to prevent catastrophic failure."
        
    # Detailed Analysis
    analysis = []
    if input_data['volt'] > input_data['volt_avg_24h'] * 1.05:
        analysis.append(f"Voltage instability: {((input_data['volt']/input_data['volt_avg_24h'])-1)*100:.1f}% surge over 24h average.")
    if input_data['rotate'] < input_data['rotate_avg_24h'] * 0.95:
        analysis.append(f"Performance dip: RPM is {((1-input_data['rotate']/input_data['rotate_avg_24h']))*100:.1f}% below baseline.")
    if input_data['vibration'] > input_data['vibration_avg_24h'] * 1.2:
        analysis.append(f"Vibration anomaly: Magnitude is {((input_data['vibration']/input_data['vibration_avg_24h'])-1)*100:.1f}% above historical mean.")
    
    analysis_text = " ".join(analysis) if analysis else "Sensor data indicates system is currently following stable historical patterns."
    
    return what, why, resolve, analysis_text

# --- Sidebar ---
st.sidebar.image("https://img.icons8.com/fluency/96/factory.png", width=80)
st.sidebar.title("Asset Management")
engine_id = st.sidebar.selectbox("Select Asset", [f"ENGINE-X-{i:03d}" for i in range(1, 10)])
mode = st.sidebar.radio("Analysis Depth", ["Deep Insight (Muscle)", "Quick Check (Skeleton)"])

st.sidebar.markdown("---")
st.sidebar.subheader("System Health")
st.sidebar.progress(85)
st.sidebar.caption("Data Pipeline: ACTIVE")
st.sidebar.caption("Last Sync: " + datetime.now().strftime("%H:%M:%S"))

if st.sidebar.button("Resolve All Alerts"):
    try:
        response = requests.post(f"{API_URL}/alerts/resolve")
        if response.status_code == 200:
            st.sidebar.success("Alerts resolved!")
            time.sleep(1) # Brief pause to show success
            st.rerun()
        else:
            st.sidebar.error(f"Error: {response.status_code}")
    except Exception as e:
        st.sidebar.error("Could not connect to API.")

# --- Main Dashboard ---
st.title("🏭 Industrial Health Control Center")
st.caption(f"Monitoring Asset: {engine_id} | Location: Production Line A")

# --- Alert Banner ---
alerts = get_system_alerts()
if alerts:
    with st.expander(f"🚨 ACTIVE SYSTEM ALERTS ({len(alerts)})", expanded=True):
        for alert in alerts:
            st.error(f"**[{alert['severity']}]** {alert['timestamp']}: {alert['message']} (Machine: {alert['machine_id']})")

# --- KPI Row ---
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric("Vibration (RMS)", "4.2 mm/s", "+0.2")
with kpi2:
    st.metric("Operating Temp", "308.5 K", "-1.2")
with kpi3:
    st.metric("Efficiency", "94.2%", "-0.5%")
with kpi4:
    st.metric("Power Quality", "Stable", "")

# --- Analysis Tabs ---
tab1, tab2, tab3 = st.tabs(["🔴 Real-time Monitoring", "🧠 Predictive Analysis", "📅 Maintenance Log"])

with tab1:
    col_l, col_r = st.columns([2, 1])
    
    with col_l:
        st.subheader("Telemetry Streams")
        # Simulated stream
        chart_data = pd.DataFrame(
            np.random.randn(50, 3) + [4.0, 308, 40],
            columns=['Vibration', 'Temp', 'Torque']
        )
        st.line_chart(chart_data, height=350)
        
    with col_r:
        st.subheader("Operational Settings")
        with st.form("telemetry_input"):
            if mode == "Deep Insight (Muscle)":
                v = st.slider("Voltage [V]", 100, 250, 170)
                r = st.slider("Rotation [RPM]", 200, 800, 450)
                p = st.slider("Pressure [psi]", 50, 200, 100)
                vib = st.slider("Vibration [mm/s]", 10, 100, 40)
            else:
                v = st.slider("Air Temp [K]", 290, 310, 300)
                r = st.slider("Rotation [RPM]", 1000, 3000, 1500)
                p = st.slider("Torque [Nm]", 10, 100, 40)
                vib = st.slider("Process Temp [K]", 300, 320, 310)
            
            errors = st.slider("Active Error Codes / Tool Wear", 0, 5, 0) if mode == "Deep Insight (Muscle)" else st.slider("Tool Wear [min]", 0, 250, 0)
            age = st.number_input("Machine Age [months]", 0, 120, 12)
            submit = st.form_submit_button("Run Diagnostics")

if tab2:
    if submit:
        # Prepare "Muscle" data with dummy temporal stats for the UI demo
        # In a real app, these would come from the live TimescaleDB stream
        with st.spinner("Analyzing temporal patterns..."):
            if mode == "Deep Insight (Muscle)":
                input_data = {
                    "volt": float(v), "rotate": float(r), "pressure": float(p), "vibration": float(vib),
                    "volt_avg_3h": float(v * 1.02), 
                    "rotate_avg_3h": float(r * 0.98), 
                    "pressure_avg_3h": float(p * 1.01), 
                    "vibration_avg_3h": float(vib * 1.05),
                    "volt_std_3h": 2.1 if v < 220 else 12.0, 
                    "rotate_std_3h": 45.0 if r < 2500 else 150.0, 
                    "pressure_std_3h": 1.5 if p < 130 else 8.5, 
                    "vibration_std_3h": 0.4 if vib < 7 else 3.2,
                    "volt_avg_24h": float(v * 0.95), # Follow the slider trend
                    "rotate_avg_24h": float(r * 0.92), 
                    "pressure_avg_24h": float(p * 0.94), 
                    "vibration_avg_24h": float(vib * 0.9), 
                    "volt_std_24h": 5.2 if v < 220 else 15.0, 
                    "rotate_std_24h": 88.0 if r < 2500 else 180.0, 
                    "pressure_std_24h": 3.4 if p < 130 else 10.0, 
                    "vibration_std_24h": 1.1 if vib < 7 else 4.5,
                    "volt_diff": 0.5, "rotate_diff": -10.0, "pressure_diff": 1.2, "vibration_diff": 0.1,
                    "error_count": int(errors), "age": int(age)
                }
                res = get_prediction_muscle(input_data)
            else:
                # Use the sliders directly as they now match the mode
                res = get_prediction_skeleton({
                    "air_temp": float(v),
                    "process_temp": float(vib),
                    "rpm": float(r),
                    "torque": float(p),
                    "tool_wear": float(errors)
                })
                # Adjust res format for Skeleton
                if res:
                    res['failure_predicted_24h'] = res.get('failure_predicted', False)
                    res['time_to_failure_hours'] = "N/A (Classification Only)"
            
            if res:
                risk = res['risk_level']
                risk_class = "low-risk" if risk == "Low" else "medium-risk" if risk == "Medium" else "high-risk"
                
                st.markdown(f"""
                    <div class="status-card {risk_class}">
                        <h2 style='margin:0'>Risk Level: {risk}</h2>
                        <p style='margin:0'>Failure Probability (24h): {res['failure_probability']*100:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)
                
                c1, c2 = st.columns(2)
                with c1:
                    st.info(f"🔮 **AI Prediction**: {'Failure Warning' if res['failure_predicted_24h'] else 'Safe Operation'}")
                with c2:
                    st.warning(f"⏱️ **Time to Failure**: {res['time_to_failure_hours']} hours")
                
                st.subheader("Root Cause Analysis (SHAP)")
                st.write("Top environmental drivers identified by the model:")
                
                # Show dynamic metrics based on top_predictors
                cols = st.columns(min(len(top_predictors), 4))
                for i, pred in enumerate(top_predictors[:4]):
                    # Simplified logic to highlight if value is "concerning"
                    # In production, this would compare against historical thresholds
                    val = input_data.get(pred, "N/A")
                    label = pred.replace("_", " ").title()
                    status = "Normal"
                    cols[i % len(cols)].metric(label, val, delta=status if status != "Normal" else None, delta_color="inverse")
                
                st.markdown("---")
                st.subheader("📝 Diagnostic Summary")
                what, why, resolve, detail = generate_diagnostic_summary(res, input_data)
                
                sum_col1, sum_col2, sum_col3 = st.columns(3)
                with sum_col1:
                    st.write("### What Happened?")
                    st.write(what)
                with sum_col2:
                    st.write("### Why it Happened?")
                    st.write(why)
                with sum_col3:
                    st.write("### Recommended Action")
                    st.write(resolve)
                
                st.info(f"**Detailed Technical Analysis**: {detail}")
            else:
                st.error("Could not connect to Prediction Engine. Ensure API is running at localhost:8000")
    else:
        st.info("👈 Use the 'Real-time Monitoring' tab to adjust settings and run diagnostics.")

with tab3:
    st.subheader("Upcoming Maintenance Schedule")
    schedule = [
        {"Task": "Oil Filter Replacement", "Date": "2026-04-25", "Status": "Scheduled"},
        {"Task": "Vibration Sensor Calibration", "Date": "2026-05-12", "Status": "Planned"},
        {"Task": "Full Mechanical Overhaul", "Date": "2026-12-15", "Status": "Waitlist"}
    ]
    st.table(schedule)
    
st.markdown("---")
