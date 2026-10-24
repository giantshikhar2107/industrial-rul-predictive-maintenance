import sqlite3
import os


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


DATABASE = os.path.join(
    PROJECT_ROOT,
    "database",
    "machine_data.db"
)


def get_connection():

    connection = sqlite3.connect(
        DATABASE
    )

    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    return connection


def initialize_database():

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS machines (
        machine_id TEXT PRIMARY KEY,
        machine_name TEXT,
        machine_type TEXT,
        location TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS sensor_readings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        machine_id TEXT,
        timestamp TEXT,
        cycle INTEGER,
        temperature REAL,
        vibration REAL,
        current REAL,
        pressure REAL,
        rpm REAL,
        rul REAL,
        predicted_rul REAL,
        FOREIGN KEY (machine_id)
        REFERENCES machines(machine_id)
    )
    """)

    cursor.execute("""
    PRAGMA table_info(sensor_readings)
    """)

    columns = [
        row[1]
        for row in cursor.fetchall()
    ]

    if "predicted_rul" not in columns:

        cursor.execute("""
        ALTER TABLE sensor_readings
        ADD COLUMN predicted_rul REAL
        """)

    connection.commit()
    connection.close()


def insert_machine(
    machine_id,
    machine_name,
    machine_type,
    location
):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
    INSERT OR IGNORE INTO machines
    (machine_id, machine_name, machine_type, location)
    VALUES (?, ?, ?, ?)
    """, (
        machine_id,
        machine_name,
        machine_type,
        location
    ))

    connection.commit()
    connection.close()


def insert_sensor_reading(data):

    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
    INSERT INTO sensor_readings
    (machine_id, timestamp, cycle, temperature,
     vibration, current, pressure, rpm, rul,
     predicted_rul)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["machine_id"],
        data["timestamp"],
        data["cycle"],
        data["temperature"],
        data["vibration"],
        data["current"],
        data["pressure"],
        data["rpm"],
        data["rul"],
        data.get("predicted_rul")
    ))

    connection.commit()
    connection.close()