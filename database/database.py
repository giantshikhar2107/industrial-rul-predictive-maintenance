import sqlite3

DATABASE = "database/machine_data.db"


def get_connection():
    return sqlite3.connect(DATABASE)


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
        FOREIGN KEY (machine_id) REFERENCES machines(machine_id)
    )
    """)

    connection.commit()
    connection.close()


def insert_machine(machine_id, machine_name, machine_type, location):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
    INSERT OR IGNORE INTO machines
    (machine_id, machine_name, machine_type, location)
    VALUES (?, ?, ?, ?)
    """, (machine_id, machine_name, machine_type, location))

    connection.commit()
    connection.close()


def insert_sensor_reading(data):
    connection = get_connection()
    cursor = connection.cursor()

    cursor.execute("""
    INSERT INTO sensor_readings
    (machine_id, timestamp, cycle, temperature,
     vibration, current, pressure, rpm, rul)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        data["machine_id"],
        data["timestamp"],
        data["cycle"],
        data["temperature"],
        data["vibration"],
        data["current"],
        data["pressure"],
        data["rpm"],
        data["rul"]
    ))

    connection.commit()
    connection.close()