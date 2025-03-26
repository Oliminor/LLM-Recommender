import openai
import numpy as np
import spacy
import re
from peewee import *
from pgvector.peewee import VectorField

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
    embedding_title = VectorField(dimensions=3072)  # Store embedding 
    embedding_body = VectorField(dimensions=3072)   # Store embedding 

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

    # Log the storing action
    print(f"Storing embedding for article_id: {article_id}, article: {article_title}")

    # Use Peewee to update the article's embeddings
    try:
        # Find the article by its ID and update the embeddings
        article = Article.get(Article.id == article_id)
        article.embedding_title = title_embedding
        article.embedding_body = body_embedding
        article.save()  # Save the updated article with the new embeddings
        print(f"✅ Embedding stored successfully for article_id: {article_id}")

    except Article.DoesNotExist:
        print(f"⚠️ Article with ID {article_id} does not exist.")
        

def preprocess_text(text):
    """Filter out stopwords and keep relevant words while maintaining sentence structure."""
    doc = nlp(text)
    filtered_tokens = [token.text for token in doc if not token.is_stop]  # Remove stopwords
    return " ".join(filtered_tokens)  # Reconstruct the sentence

def vectorize_all_articles():
    """Convert all articles into embeddings (title and body separately) after preprocessing."""
    # Fetch all articles that don't have embeddings for title or body yet
    articles = Article.select().where((Article.embedding_title.is_null(True)) | (Article.embedding_body.is_null(True)))

    for article in articles:
        # Preprocess text to remove stopwords while keeping context
        filtered_title = preprocess_text(article.article_title)
        filtered_body = preprocess_text(article.article_body)

        # Generate embeddings separately for the filtered title and body
        title_embedding = get_openai_embedding(filtered_title)
        body_embedding = get_openai_embedding(filtered_body)

        # Update the database with the separate embeddings
        article.embedding_title = title_embedding
        article.embedding_body = body_embedding
        article.save()  # Save the updated article with the embeddings

    print(f"✅ Processed and stored {len(articles)} article embeddings.")


def search_similar_articles(article_id, exclude_user_id):
    """Search for articles similar to a given article by title and body separately while ignoring articles from a specific user.
       Then, send the results to ChatGPT for final filtering based on relevance.
    """
    try:
        # Fetch target article and its embeddings
        target_article = Article.get_or_none(Article.id == article_id)
        if not target_article:
            print(f"⚠️ Article ID {article_id} not found.")
            return []

        target_title = target_article.article_title
        target_body = target_article.article_body
        title_embedding = target_article.embedding_title
        body_embedding = target_article.embedding_body

        # Convert to list if valid, else set to None
        title_embedding = title_embedding.tolist() if title_embedding is not None else None
        body_embedding = body_embedding.tolist() if body_embedding is not None else None

        # Ensure at least one valid embedding exists
        if title_embedding is None and body_embedding is None:
            print(f"⚠️ Both title and body embeddings are missing for article ID {article_id}.")
            return []

        print(f"🔍 Searching for articles similar to ID {article_id} (excluding user {exclude_user_id})...")

        # Query articles similar based on embeddings using pgvector
        similar_articles = (
            Article.select(
                Article.id,
                Article.article_title,
                Article.article_body,
                (1 - Article.embedding_title.cosine_distance(title_embedding)).alias("title_similarity"),
                (1 - Article.embedding_body.cosine_distance(body_embedding)).alias("body_similarity"),
            )
            .where(
                (Article.fk_username != exclude_user_id) &
                (Article.embedding_title.is_null(False)) & (Article.embedding_body.is_null(False))
            )
            .order_by(
                ((1 - Article.embedding_title.cosine_distance(title_embedding)) + 
                 (1 - Article.embedding_body.cosine_distance(body_embedding))).desc()
            )
            .limit(5)
        )

        # Convert query results to a list
        articles_list = [
            {
                "id": article.id,
                "title": article.article_title,
                "body": article.article_body,
                "title_similarity": article.title_similarity,
                "body_similarity": article.body_similarity,
            }
            for article in similar_articles
        ]

        if not articles_list:
            print(f"⚠️ No similar articles found for article ID {article_id}.")
            return []

        return filter_relevant_articles(target_title, target_body, articles_list)

    except Exception as e:
        print(f"❌ Error while searching for similar articles: {e}")
        return []

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
        model="gpt-4o-mini",
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
    # Check if the user exists using Peewee ORM
    user = Username.get_or_none(Username.username == username)

    if user:
        # Use the existing user
        user_id = user
    else:
        # Insert new user using Peewee ORM
        user = Username.create(username=username)
        user_id = user
        print(f"✅ Added new user: {username} (ID: {user_id.id})")

    # Insert the article into the database using Peewee ORM
    article = Article.create(fk_username=user, article_title=article_title, article_body=article_body)
    article_id = article.id  # Retrieve the newly inserted article ID

    # Generate and store embeddings
    store_article_embedding(article_id, article_title, article_body)

    return f"✅ Article '{article_title}' added successfully with embeddings!"


def get_user_id(username):
    """Fetch user ID based on the provided username using Peewee."""
    try:
        # Query the user from the Username model
        user = Username.get(Username.username == username)
        return user.id
    except Username.DoesNotExist:
        return None  # User not found
    
def get_user_article_titles(user_id):
    """Fetch all article titles written by a user using Peewee."""
    articles = Article.select(Article.id, Article.article_title).where(Article.fk_username == user_id)
    return [(article.id, article.article_title) for article in articles]

def get_user_article_body(user_id):
    """Fetch all article bodies written by a user using Peewee."""
    articles = Article.select(Article.id, Article.article_body).where(Article.fk_username == user_id)
    return [(article.id, article.article_body) for article in articles]

def get_article_body_by_id(article_id):
    """Fetch the body of an article by its ID using Peewee."""
    try:
        article = Article.get(Article.id == article_id)
        return article.article_body
    except Article.DoesNotExist:
        return None  # Return None if the article doesn't exist
    
def get_article_title_by_id(article_id):
    """Fetch the title of an article by its ID using Peewee."""
    try:
        article = Article.get(Article.id == article_id)
        return article.article_title
    except Article.DoesNotExist:
        return None  # Return None if the article doesn't exist
    
def get_user_by_article_id(article_id):
    """Fetch the username of the user who wrote the article based on the article ID using Peewee."""
    try:
        article = Article.get(Article.id == article_id)
        return article.fk_username.username  # Accessing the related user's username
    except Article.DoesNotExist:
        return None  # Return None if the article doesn't exist