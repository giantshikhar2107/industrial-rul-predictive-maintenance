import sqlite3
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE = os.path.join(PROJECT_ROOT, "database", "machine_data.db")


def get_connection():
    os.makedirs(os.path.dirname(DATABASE), exist_ok=True)
    connection = sqlite3.connect(DATABASE)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database():
    connection = get_connection()

    try:
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
                dataset TEXT,
                engine_id INTEGER,
                max_cycles INTEGER,
                setting_1 REAL,
                setting_2 REAL,
                setting_3 REAL,
                FOREIGN KEY (machine_id) REFERENCES machines(machine_id)
            )
        """)

        cursor.execute("PRAGMA table_info(sensor_readings)")
        existing_columns = {
            row[1] for row in cursor.fetchall()
        }

        required_columns = {
            "predicted_rul": "REAL",
            "dataset": "TEXT",
            "engine_id": "INTEGER",
            "max_cycles": "INTEGER",
            "setting_1": "REAL",
            "setting_2": "REAL",
            "setting_3": "REAL",
        }

        for i in range(1, 22):
            required_columns[f"sensor_{i}"] = "REAL"

        for column, data_type in required_columns.items():
            if column not in existing_columns:
                cursor.execute(
                    f"ALTER TABLE sensor_readings "
                    f"ADD COLUMN {column} {data_type}"
                )

        connection.commit()

    finally:
        connection.close()


def insert_machine(machine_id, machine_name, machine_type, location):
    connection = get_connection()

    try:
        connection.execute("""
            INSERT OR IGNORE INTO machines (
                machine_id,
                machine_name,
                machine_type,
                location
            )
            VALUES (?, ?, ?, ?)
        """, (
            machine_id,
            machine_name,
            machine_type,
            location
        ))

        connection.commit()

    finally:
        connection.close()


def insert_sensor_reading(data):
    connection = get_connection()

    try:
        connection.execute("""
            INSERT INTO sensor_readings (
                machine_id,
                timestamp,
                cycle,
                temperature,
                vibration,
                current,
                pressure,
                rpm,
                rul,
                predicted_rul
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            data["machine_id"],
            data["timestamp"],
            data["cycle"],
            data.get("temperature"),
            data.get("vibration"),
            data.get("current"),
            data.get("pressure"),
            data.get("rpm"),
            data.get("rul"),
            data.get("predicted_rul")
        ))

        connection.commit()

    finally:
        connection.close()


def insert_nasa_sensor_reading(data):
    machine_id = data["machine_id"]

    insert_machine(
        machine_id=machine_id,
        machine_name=f"NASA Engine {data['engine_id']}",
        machine_type=data.get(
            "machine_type",
            "NASA C-MAPSS Turbofan Engine"
        ),
        location="NASA C-MAPSS Dataset"
    )

    columns = [
        "machine_id",
        "timestamp",
        "cycle",
        "rul",
        "predicted_rul",
        "dataset",
        "engine_id",
        "max_cycles",
        "setting_1",
        "setting_2",
        "setting_3"
    ]

    values = [
        machine_id,
        data["timestamp"],
        data["cycle"],
        data.get("actual_rul", data.get("rul")),
        data.get("predicted_rul"),
        data.get("dataset", "NASA_CMAPSS_FD001"),
        data["engine_id"],
        data["max_cycles"],
        data.get("setting_1"),
        data.get("setting_2"),
        data.get("setting_3")
    ]

    for i in range(1, 22):
        column = f"sensor_{i}"
        columns.append(column)
        values.append(data.get(column))

    placeholders = ", ".join(["?"] * len(columns))
    column_names = ", ".join(columns)

    query = f"""
        INSERT INTO sensor_readings ({column_names})
        VALUES ({placeholders})
    """

    connection = get_connection()

    try:
        connection.execute(query, values)
        connection.commit()

    finally:
        connection.close()


def get_recent_sensor_readings(limit=50, machine_id="M001"):
    if not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")

    connection = get_connection()
    connection.row_factory = sqlite3.Row

    try:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT *
            FROM sensor_readings
            WHERE machine_id = ?
            ORDER BY id DESC
            LIMIT ?
        """, (machine_id, limit))

        readings = [dict(row) for row in cursor.fetchall()]
        readings.reverse()

        return readings

    finally:
        connection.close()


def get_sensor_reading_count(machine_id="M001"):
    connection = get_connection()

    try:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT COUNT(*)
            FROM sensor_readings
            WHERE machine_id = ?
        """, (machine_id,))

        return cursor.fetchone()[0]

    finally:
        connection.close()


def get_latest_sensor_reading(machine_id="M001"):
    connection = get_connection()
    connection.row_factory = sqlite3.Row

    try:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT *
            FROM sensor_readings
            WHERE machine_id = ?
            ORDER BY id DESC
            LIMIT 1
        """, (machine_id,))

        reading = cursor.fetchone()

        return dict(reading) if reading else None

    finally:
        connection.close()


initialize_database()