import openai
import psycopg2
import numpy as np


host = "localhost"
port = "5432"
dbname = "Article"
user = "user"
password = "admin"

username_table = "username"
article_table = "article"

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
        model="text-embedding-3-large"
    )
    return response["data"][0]["embedding"]  # Returns a list of 1536 floats

def store_username_embedding(user_id, username):
    """Store username embedding in the database."""
    embedding = get_openai_embedding(username)
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    print(f"Storing embedding for user_id: {user_id}, username: {username}")

    cur.execute(
        f"UPDATE {username_table} SET embedding = %s WHERE id = %s;",
        (embedding, user_id)
    )
    
    conn.commit()
    cur.close()
    conn.close()


def store_article_embedding(article_id, article_title, article_body):
    """Store article embedding in the database."""
    embedding = get_openai_embedding(article_title + " " + article_body)
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    print(f"Storing embedding for article_id: {article_id}, article: {article_title}")

    cur.execute(
        f"UPDATE {article_table} SET embedding = %s WHERE id = %s;",
        (embedding, article_id)
    )
    
    conn.commit()
    cur.close()
    conn.close()


def vectorize_all_username():
    """Convert all usernames into embeddings and store them in the DB."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Fetch all usernames that don't have an embedding yet
    cur.execute(f"SELECT id, username FROM {username_table} WHERE embedding IS NULL;")
    usernames = cur.fetchall()

    # Loop through each user and vectorize their username
    for username_id, user in usernames:
        embedding = get_openai_embedding(user)  # Convert username to vector
        cur.execute(f"UPDATE {username_table} SET embedding = %s WHERE id = %s;", (embedding, username_id))

    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ Stored {len(usernames)} username embeddings.")


def vectorize_all_articles():
    """Convert all articles into embeddings and store them in the DB."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Fetch all articles that don't have an embedding yet
    cur.execute(f"SELECT id, article_title, article_body FROM {article_table} WHERE embedding IS NULL;")
    articles = cur.fetchall()

    for article_id, title, body in articles:
        combined_text = f"{title} {body}"  # Concatenate title and body
        embedding = get_openai_embedding(combined_text)  # Convert to vector
        cur.execute(f"UPDATE {article_table} SET embedding = %s WHERE id = %s;", (embedding, article_id))

    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ Stored {len(articles)} article embeddings.")


def search_similar_articles(article_id, exclude_user_id):
    """Search for articles similar to a given article while ignoring articles from a specific user."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Get the embedding of the selected article
    cur.execute(f"SELECT embedding FROM {article_table} WHERE id = %s;", (article_id,))
    article_embedding = cur.fetchone()

    if not article_embedding:
        print(f"⚠️ No embedding found for article ID {article_id}.")
        return []

    article_embedding = np.array(article_embedding[0]).tolist()  # Convert to list

    print(f"Searching for articles similar to ID {article_id} (excluding user {exclude_user_id})...")

    # Find the most similar articles, excluding those by the selected user
    cur.execute(f"""
        SELECT id, article_title, 1 - (embedding <=> %s::vector) AS similarity
        FROM {article_table}
        WHERE fk_username != %s
        ORDER BY similarity DESC
        LIMIT 5;
    """, (article_embedding, exclude_user_id))

    results = cur.fetchall()

    # Debugging: Print similarity score for each result
    for result in results:
        print(f"Article ID: {result[0]} - Title: {result[1]} - Similarity: {result[2]:.4f}")

    cur.close()
    conn.close()

    return results


def add_article(username, article_title, article_body):
    """Add a new article to the database. If the user does not exist, add them first."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Check if the user exists
    cur.execute(f"SELECT id FROM {username_table} WHERE username = %s;", (username,))
    user = cur.fetchone()

    if user:
        user_id = user[0]  # Existing user ID
    else:
        # Insert new user and get the ID
        cur.execute(f"INSERT INTO {username_table} (username) VALUES (%s) RETURNING id;", (username,))
        user_id = cur.fetchone()[0]
        print(f"✅ Added new user: {username} (ID: {user_id})")

    # Insert the article into the database
    cur.execute(f"""
        INSERT INTO {article_table} (fk_username, article_title, article_body)
        VALUES (%s, %s, %s);
    """, (user_id, article_title, article_body))

    conn.commit()
    cur.close()
    conn.close()
    return f"✅ Article '{article_title}' added successfully!"  # Return a success message

def get_user_id(username):
    """Fetch user ID based on the provided username."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    cur.execute(f"SELECT id FROM {username_table} WHERE username = %s;", (username,))
    user = cur.fetchone()
    
    cur.close()
    conn.close()
    
    if user:
        return user[0]
    else:
        return None  # User not found
    
def get_user_articles(user_id):
    """Fetch all articles written by a user."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    cur.execute(f"SELECT id, article_title FROM {article_table} WHERE fk_username = %s;", (user_id,))
    articles = cur.fetchall()
    
    cur.close()
    conn.close()
    
    return articles
