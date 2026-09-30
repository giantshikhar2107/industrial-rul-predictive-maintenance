import sqlite3

connection = sqlite3.connect("database/machine_data.db")

cursor = connection.cursor()

cursor.execute("SELECT * FROM machines")

rows = cursor.fetchall()

for row in rows:
    print(row)

connection.close()