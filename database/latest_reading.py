import sqlite3

connection = sqlite3.connect("database/machine_data.db")

cursor = connection.cursor()

cursor.execute("""
SELECT *
FROM sensor_readings
ORDER BY id DESC
LIMIT 1
""")

reading = cursor.fetchone()

print("Latest reading:")
print(reading)

connection.close()
