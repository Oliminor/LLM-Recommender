import openai
import psycopg2
import numpy as np

# Replace with your actual PostgreSQL credentials
host = "localhost"
port = "5432"
dbname = "book"
user = "user"
password = "admin"

bookTable = "book_title"
commentTable = "comments"


# PostgreSQL connection settings
DB_CONFIG = {
    "dbname": dbname,
    "user": user,
    "password": password,
    "host": host,
    "port": port
}


def get_openai_embedding(text):
    """Generate embedding for a given text using OpenAI API."""
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]  # Returns a list of 1536 floats


def store_book_embedding(book_id, title):
    """Store book title embedding in the database."""
    embedding = get_openai_embedding(title)
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    print(f"Storing embedding for book: {book_id}, title: {title}")

    cur.execute(
        "UPDATE book_title SET embedding = %s WHERE id = %s;",
        (embedding, book_id)
    )
    
    conn.commit()
    cur.close()
    conn.close()


def store_comment_embedding(comment_id, comment):
    """Store comment embedding in the database."""
    embedding = get_openai_embedding(comment)
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    print(f"Storing embedding for comment: {comment_id}, comment: {comment}")

    cur.execute(
        "UPDATE comments SET embedding = %s WHERE id = %s;",
        (embedding, comment_id)
    )
    
    conn.commit()
    cur.close()
    conn.close()


def vectorize_all_books():
    """Convert all book titles into embeddings and store them in the DB."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Fetch all books that don't have an embedding yet
    cur.execute("SELECT id, title FROM book_title WHERE embedding IS NULL;")
    books = cur.fetchall()

    for book_id, title in books:
        embedding = get_openai_embedding(title)  # Convert title to vector
        cur.execute("UPDATE book_title SET embedding = %s WHERE id = %s;", (embedding, book_id))

    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ Stored {len(books)} book embeddings.")


def vectorize_all_comments():
    """Convert all comments into embeddings and store them in the DB."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Fetch all comments that don't have an embedding yet
    cur.execute("SELECT id, comment FROM comments WHERE embedding IS NULL;")
    comments = cur.fetchall()

    for comment_id, comment in comments:
        embedding = get_openai_embedding(comment)  # Convert comment to vector
        cur.execute("UPDATE comments SET embedding = %s WHERE id = %s;", (embedding, comment_id))

    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ Stored {len(comments)} comment embeddings.")


def search_books(query):
    """Search books based on query vector similarity."""
    query_embedding = get_openai_embedding(query)

    # Convert the query embedding into the correct format (numpy array -> list)
    query_embedding = np.array(query_embedding).tolist()

    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    print(f"Searching for query: {query} with embedding: {query_embedding[:5]}...")  # Only show part of the vector for readability

    # Find the most similar books using cosine similarity with explicit vector type cast
    cur.execute("""
        SELECT id, title, 1 - (embedding <=> %s::vector) AS similarity
        FROM book_title
        ORDER BY similarity DESC
        LIMIT 5;
    """, (query_embedding,))

    results = cur.fetchall()
    cur.close()
    conn.close()

    return results


# Run the vectorization functions to ensure all data is embedded
if __name__ == "__main__":
    print("Vectorizing all books...")
    vectorize_all_books()
    
    print("Vectorizing all comments...")
    vectorize_all_comments()
    
    # Example search
    print("\n🔍 Searching for books related to 'Uma Musume':")
    results = search_books("Uma")
    
    for book in results:
        print(f"📖 Book: {book[1]} (Similarity: {book[2]:.2f})")