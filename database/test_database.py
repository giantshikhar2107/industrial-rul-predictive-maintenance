from database import initialize_database, insert_machine, insert_sensor_reading

initialize_database()

insert_machine(
    "M001",
    "Motor-01",
    "Industrial Motor",
    "Production Line 1"
)

data = {
    "machine_id": "M001",
    "timestamp": "2026-09-30 10:00:00",
    "cycle": 100,
    "temperature": 62.5,
    "vibration": 1.8,
    "current": 7.2,
    "pressure": 4.5,
    "rpm": 1470,
    "rul": 9900
}

insert_sensor_reading(data)

print("Test sensor reading inserted successfully")