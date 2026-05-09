import mysql.connector

def get_connection():
    conn = mysql.connector.connect(
        host="localhost",
        user="root",              # change if your MySQL user is different
        password="123456",  # replace with your MySQL password
        database="heart_db",
    )
    return conn
