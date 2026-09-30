import sqlite3

connection = sqlite3.connect("database/machine_data.db")

cursor = connection.cursor()

cursor.execute("SELECT COUNT(*) FROM sensor_readings")

count = cursor.fetchone()[0]

print("Total sensor readings:", count)

connection.close()