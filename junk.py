import psycopg2

# Replace these values with your database credentials
host = "localhost"          # PostgreSQL server host
port = "5432"               # PostgreSQL server port
dbname = "book"     # Database name
user = "user"           # Username
password = "admin"  # Password
tableName = "bence_books"

# Connect to the PostgreSQL database
try:
    connection = psycopg2.connect(
        host=host,
        port=port,
        dbname=dbname,
        user=user,
        password=password
    )
    
    cursor = connection.cursor()

    # Query to select all data from a specific table (e.g., 'your_table_name')
    query = "SELECT * FROM bence_books"
    
    cursor.execute(query)
    
    # Fetch all rows from the executed query
    rows = cursor.fetchall()

    #Print all rows
    for row in rows:
        print(row)
    
    insert_query = f"INSERT INTO {tableName} (title , comment) VALUES (%s, %s)"

    data_list = [
    ("Uma Musume", "Best Anime ever"),
    ("Pretty Derby", "Cannot stop reading"),
    ("Oguri Cap", "Book of the year")
]
    
    cursor.executemany(insert_query, data_list)
    connection.commit()

    # Close the cursor and connection
    cursor.close()
    connection.close()

except Exception as error:
    print(f"Error: {error}")