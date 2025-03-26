import openai
import psycopg2
import numpy as np
import spacy
import re
from peewee import *


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

# PostgreSQL database connection
db = PostgresqlDatabase('Article', user='user', password='admin', host='localhost', port=5432)

# Define the Username model
class Username(Model):
    id = AutoField()  # Auto incrementing ID
    username = CharField(unique=True)

    class Meta:
        database = db  # Define the database for this model


# Define the Article model
class Article(Model):
    id = AutoField()  # Auto incrementing ID
    fk_username = ForeignKeyField(Username, backref='articles')  # Foreign Key to 'Username' table
    article_title = CharField()
    article_body = TextField()
    embedding_title = BlobField(null=True)  # Store embedding 
    embedding_body = BlobField(null=True)   # Store embedding 

    class Meta:
        database = db  # Define the database for this model

nlp = spacy.load("en_core_web_sm")

def get_openai_embedding(text):
    """Generate embedding for a given text using OpenAI API."""
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-3-large"
    )
    return response["data"][0]["embedding"]  # Returns a list of 1536 floats


def store_article_embedding(article_id, article_title, article_body):
    """Store article embedding for title and body separately in the database."""
    # Get embedding for title
    title_embedding = get_openai_embedding(article_title)
    
    # Get embedding for body
    body_embedding = get_openai_embedding(article_body)
    
    # Connect to the database
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    print(f"Storing embedding for article_id: {article_id}, article: {article_title}")

    # Update title and body embeddings in the database
    cur.execute(
        f"UPDATE {article_table} SET embedding_title = %s, embedding_body = %s WHERE id = %s;",
        (title_embedding, body_embedding, article_id)
    )
    
    # Commit changes and close connection
    conn.commit()
    cur.close()
    conn.close()

def preprocess_text(text):
    """Filter out stopwords and keep relevant words while maintaining sentence structure."""
    doc = nlp(text)
    filtered_tokens = [token.text for token in doc if not token.is_stop]  # Remove stopwords
    return " ".join(filtered_tokens)  # Reconstruct the sentence

def vectorize_all_articles():
    """Convert all articles into embeddings (title and body separately) after preprocessing."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Fetch all articles that don't have embeddings for title or body yet
    cur.execute(f"SELECT id, article_title, article_body FROM {article_table} WHERE embedding_title IS NULL OR embedding_body IS NULL;")
    articles = cur.fetchall()

    for article_id, title, body in articles:
        # Preprocess text to remove stopwords while keeping context
        filtered_title = preprocess_text(title)
        filtered_body = preprocess_text(body)

        # Generate embeddings separately for the filtered title and body
        title_embedding = get_openai_embedding(filtered_title)
        body_embedding = get_openai_embedding(filtered_body)

        # Update the database with the separate embeddings
        cur.execute(f"UPDATE {article_table} SET embedding_title = %s, embedding_body = %s WHERE id = %s;", 
                    (title_embedding, body_embedding, article_id))

    conn.commit()
    cur.close()
    conn.close()
    print(f"✅ Processed and stored {len(articles)} article embeddings.")


def search_similar_articles(article_id, exclude_user_id):
    """Search for articles similar to a given article by title and body separately while ignoring articles from a specific user.
       Then, send the results to ChatGPT for final filtering based on relevance.
    """
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Get the title and body embeddings of the selected article
    cur.execute(f"SELECT article_title, article_body, embedding_title, embedding_body FROM {article_table} WHERE id = %s;", (article_id,))
    result = cur.fetchone()

    if not result:
        print(f"⚠️ No embeddings found for article ID {article_id}.")
        return []

    target_title, target_body, title_embedding, body_embedding = result

    # Check for None values and convert to list if valid
    title_embedding = np.array(title_embedding).tolist() if title_embedding is not None else None
    body_embedding = np.array(body_embedding).tolist() if body_embedding is not None else None

    if not title_embedding and not body_embedding:
        print(f"⚠️ Both title and body embeddings are missing for article ID {article_id}.")
        return []

    print(f"Searching for articles similar to ID {article_id} (excluding user {exclude_user_id})...")

    # Run a single query to get similarity based on both title and body
    cur.execute(f"""
        SELECT 
            id, 
            article_title, 
            article_body, 
            1 - (embedding_title <=> %s::vector) AS title_similarity,
            1 - (embedding_body <=> %s::vector) AS body_similarity
        FROM {article_table}
        WHERE fk_username != %s
        ORDER BY 
            (1 - (embedding_title <=> %s::vector)) + (1 - (embedding_body <=> %s::vector)) DESC
        LIMIT 5;
    """, (title_embedding, body_embedding, exclude_user_id, title_embedding, body_embedding))

    results = cur.fetchall()

    if not results:
        print(f"⚠️ No similar articles found for article ID {article_id}.")
        cur.close()
        conn.close()
        return []

    cur.close()
    conn.close()

    # Prepare articles for ChatGPT filtering
    articles_list = []
    for article_id, article_title, article_body, title_similarity, body_similarity in results:
        articles_list.append({
            "id": article_id,
            "title": article_title,
            "body": article_body,
            "title_similarity": title_similarity,
            "body_similarity": body_similarity
        })

    # Send articles to ChatGPT for filtering
    return filter_relevant_articles(target_title, target_body, articles_list)

def filter_relevant_articles(target_title, target_body, articles_list):
    """Uses ChatGPT to filter the most relevant articles from the top 5 returned by similarity."""
    
    # Construct the prompt
    prompt = f"""
    You are an AI that helps recommend the most relevant articles based on a target article. 

    The target article:
    - Title: "{target_title}"
    - Content: "{target_body[:1000]}..."  (truncated for efficiency)

    Here are 5 articles ranked by similarity scores:

    {''.join([f"- ID: {a['id']}, Title: \"{a['title']}\"\n  Content: \"{a['body'][:300]}...\"\n\n" for a in articles_list])}

    Please select the top maximum 3 most relevant articles (could be less) based on content similarity and conceptual alignment with the target article.
    Return the article IDs and their Titles in the following format:

    - ID: <article_id>, Title: "<article_title>"

    If there are no relevant articles, please return "none"
    """

    # Call OpenAI API
    response = openai.ChatCompletion.create(
        #model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are an intelligent article recommender."},
                  {"role": "user", "content": prompt}]
    )

    raw_response = response['choices'][0]['message']['content']

    print("🔍 Raw Response:\n", raw_response)  # Debugging output

    # ✅ Improved Regex to correctly extract the IDs and Titles
    matches = re.findall(r'- ID: (\d+), Title: "(.*?)"', raw_response)

    filtered_articles = [{"id": int(article_id), "title": article_title} for article_id, article_title in matches]

    return filtered_articles  # ✅ Returns a correctly formatted list


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

    # Insert the article into the database and get its ID
    cur.execute(f"""
        INSERT INTO {article_table} (fk_username, article_title, article_body)
        VALUES (%s, %s, %s) RETURNING id;
    """, (user_id, article_title, article_body))
    
    article_id = cur.fetchone()[0]  # Retrieve the newly inserted article ID
    conn.commit()
    
    cur.close()
    conn.close()

    # Generate and store embeddings
    store_article_embedding(article_id, article_title, article_body)

    return f"✅ Article '{article_title}' added successfully with embeddings!"

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
    
def get_user_article_titles(user_id):
    """Fetch all articles title written by a user."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    cur.execute(f"SELECT id, article_title FROM {article_table} WHERE fk_username = %s;", (user_id,))
    articles = cur.fetchall()
    
    cur.close()
    conn.close()
    
    return articles

def get_user_article_body(user_id):
    """Fetch all articles body written by a user."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    cur.execute(f"SELECT id, article_body FROM {article_table} WHERE fk_username = %s;", (user_id,))
    articles = cur.fetchall()
    
    cur.close()
    conn.close()
    
    return articles

def get_article_body_by_id(article_id):
    """Fetch the body of an article by its ID."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Query to fetch the article body by article_id
    cur.execute(f"SELECT article_body FROM {article_table} WHERE id = %s;", (article_id,))
    result = cur.fetchone()  # Use fetchone since you expect only one result
    
    cur.close()
    conn.close()

    if result:
        return result[0]  # Return the article body
    else:
        return None  # In case the article ID doesn't exist
    
def get_article_title_by_id(article_id):
    """Fetch the body of an article by its ID."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Query to fetch the article body by article_id
    cur.execute(f"SELECT article_title FROM {article_table} WHERE id = %s;", (article_id,))
    result = cur.fetchone()  # Use fetchone since you expect only one result
    
    cur.close()
    conn.close()

    if result:
        return result[0]  # Return the article body
    else:
        return None  # In case the article ID doesn't exist
    
def get_user_by_article_id(article_id):
    """Fetch the username of the user who wrote the article based on the article ID."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    cur.execute(f"""
        SELECT u.username
        FROM {article_table} a
        JOIN {username_table} u ON a.fk_username = u.id
        WHERE a.id = %s;
    """, (article_id,))
    
    # Fetch the result (we expect a single row with the username)
    user = cur.fetchone()
    
    cur.close()
    conn.close()
    
    if user:
        return user[0]  # Return the username
    else:
        return None  # Return None if no user is found