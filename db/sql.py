import mysql.connector

conn = mysql.connector.connect(
  host="localhost",
  user="myuser",
  password="mypassword",
  database="mydatabase"
)

cursor = conn.cursor()
cursor.execute("SHOW TABLES")

for table in cursor:
    print(table)

conn.close()