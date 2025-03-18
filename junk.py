import psycopg2

# Replace these values with your database credentials
host = "localhost"          # PostgreSQL server host
port = "5432"               # PostgreSQL server port
dbname = "book"     # Database name
user = "user"           # Username
password = "admin"  # Password
bookTable = "book_title"
commentTable = "comments"

def insert_comment(id, comment):
    try:
        # Connect to PostgreSQL
        connection = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )
        cursor = connection.cursor()

        # Insert the comment linked to book_id
        insert_comment_query = f"INSERT INTO {commentTable} (fk_book_title, comment) VALUES (%s, %s)"
        cursor.execute(insert_comment_query, (id, comment))

        # Commit changes
        connection.commit()
        print("✅ Book and comment inserted successfully!")

    except psycopg2.Error as e:
        print(f"Error occurred: {e}")
        connection.rollback()  # Rollback the transaction in case of failure
    finally:
        cursor.close()
        connection.close()

def insert_book(title, comment):
    try:
        # Connect to PostgreSQL
        connection = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )
        cursor = connection.cursor()

        # Insert the book title (if not exists)
        insert_book_query = f"""
        INSERT INTO {bookTable} (title) 
        VALUES (%s)
        ON CONFLICT (title) DO NOTHING RETURNING id;
        """
        cursor.execute(insert_book_query, (title,))
        
        # Get book_id (if inserted, fetch it; otherwise, get the existing one)
        book_id = cursor.fetchone()
        
        if book_id is None:
            cursor.execute(f"SELECT id FROM {bookTable} WHERE title = %s", (title,))
            book_id = cursor.fetchone()[0]
        else:
            book_id = book_id[0]

        # Insert the comment linked to book_id
        insert_comment_query = f"INSERT INTO {commentTable} (fk_book_title, comment) VALUES (%, %s)"
        cursor.execute(insert_comment_query, (book_id, comment))

        # Commit changes
        connection.commit()
        print("✅ Book and comment inserted successfully!")

    except psycopg2.Error as e:
        print(f"Error occurred: {e}")
        connection.rollback()  # Rollback the transaction in case of failure
    finally:
        cursor.close()
        connection.close()

def search_book(title):
    try:
        # Connect to PostgreSQL
        connection = psycopg2.connect(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password
        )
        cursor = connection.cursor()

        # Ensure title is not empty
        if not title.strip():
            return []  # Return empty list if title is empty

        # Query to search for the book and include the id
        search_query = f"SELECT id, title FROM {bookTable} WHERE title ILIKE %s"
        cursor.execute(search_query, ('%' + title + '%',))  # Case-insensitive search

        # Fetch results
        results = cursor.fetchall()

        if not results:
            print("No books found with that title.")
            return []  # Return empty list if no results found
        
        return results  # Returns a list of tuples with (id, title)

    except psycopg2.Error as e:
        print(f"Error occurred during search: {e}")
        connection.rollback()  # Rollback the transaction in case of failure
        return None  # Return None to indicate an error occurred
    finally:
        cursor.close()
        connection.close()
