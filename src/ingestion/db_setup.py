import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

DB_NAME = os.getenv("DB_NAME", "maintenance_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")

def init_db():
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        host=DB_HOST,
        port=DB_PORT
    )
    cur = conn.cursor()

    # Enable TimescaleDB extension if not already enabled
    cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")

    # Equipment Metadata Table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS equipment (
        id VARCHAR(50) PRIMARY KEY,
        name VARCHAR(100),
        model VARCHAR(100),
        install_date DATE,
        location VARCHAR(100)
    );
    """)

    # Sensor Readings Table (Hypertable)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS sensor_readings (
        time TIMESTAMPTZ NOT NULL,
        equipment_id VARCHAR(50) NOT NULL,
        type VARCHAR(10),
        air_temp FLOAT,
        process_temp FLOAT,
        rpm FLOAT,
        torque FLOAT,
        tool_wear FLOAT,
        failure INT,
        failure_type VARCHAR(20)
    );
    """)

    # Convert to hypertable
    try:
        cur.execute("SELECT create_hypertable('sensor_readings', 'time', if_not_exists => TRUE);")
    except Exception as e:
        print(f"Note: {e}")

    # Azure Machines Table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS azure_machines (
        machineID INT PRIMARY KEY,
        model VARCHAR(50),
        age INT
    );
    """)

    # Azure Telemetry (Hypertable)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS azure_telemetry (
        datetime TIMESTAMPTZ NOT NULL,
        machineID INT NOT NULL,
        volt FLOAT,
        rotate FLOAT,
        pressure FLOAT,
        vibration FLOAT
    );
    """)

    try:
        cur.execute("SELECT create_hypertable('azure_telemetry', 'datetime', if_not_exists => TRUE);")
    except Exception as e:
        print(f"Note: {e}")

    # Azure Errors (Hypertable)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS azure_errors (
        datetime TIMESTAMPTZ NOT NULL,
        machineID INT NOT NULL,
        errorID VARCHAR(20)
    );
    """)

    try:
        cur.execute("SELECT create_hypertable('azure_errors', 'datetime', if_not_exists => TRUE);")
    except Exception as e:
        print(f"Note: {e}")

    # Azure Maintenance
    cur.execute("""
    CREATE TABLE IF NOT EXISTS azure_maint (
        datetime TIMESTAMPTZ NOT NULL,
        machineID INT NOT NULL,
        comp VARCHAR(20)
    );
    """)

    # Azure Failures
    cur.execute("""
    CREATE TABLE IF NOT EXISTS azure_failures (
        datetime TIMESTAMPTZ NOT NULL,
        machineID INT NOT NULL,
        failure VARCHAR(20)
    );
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("Database initialized successfully.")

if __name__ == "__main__":
    init_db()
