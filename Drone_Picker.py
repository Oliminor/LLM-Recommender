import openai
import PyPDF2
import os
import io
import json
from peewee import *
from typing import Optional, Dict, List
from dotenv import load_dotenv
from playhouse.postgres_ext import ArrayField

load_dotenv()

model_name = os.getenv("OPENAI_CHAT_MODEL")

# Initialize PostgreSQL database connection
db = PostgresqlDatabase(
    "Drone",
    user="user",
    password="admin",
    host="localhost",
    port=5432
)

class BaseModel(Model):
    class Meta:
        database = db

class Drone(BaseModel):
    drone_model = TextField(null=False)
    weight_g = IntegerField()
    battery_mah = IntegerField(null=True)
    flight_time_min = IntegerField(null=True)
    max_range_km = DecimalField(max_digits=5, decimal_places=1, null=True)
    camera = TextField(null=True)
    geofencing = BooleanField(default=False)
    noise_db = IntegerField(null=True)
    max_height_km = DoubleField(null=True)  # double precision
    payload_weight_g = IntegerField(null=True)
    description = TextField(null=True)

    class Meta:
        table_name = "drones"

class Location(BaseModel):
    name = CharField(max_length=100, unique=True)
    coordinates = ArrayField(DoubleField)  # Stores [latitude, longitude]

class DronesLocation(BaseModel):
    fk_location_id = ForeignKeyField(Location, backref="drones", on_delete="CASCADE")
    fk_drones_id = ForeignKeyField(Drone, backref="locations", on_delete="CASCADE")

    class Meta:
        table_name = "drones_location"  


def uploaded_file_to_bytes(uploaded_file) -> bytes:
    """Convert Streamlit UploadedFile to bytes"""
    return uploaded_file.getvalue()

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extracts text content from PDF bytes"""
    text = ""
    with io.BytesIO(pdf_bytes) as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def generate_sql_from_natural_query(user_query: str) -> str:
    """Generate an SQL query using OpenAI based on natural language input"""
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an expert SQL assistant with knowledge of a drones database."},
            {"role": "system", "content": "The 'drones' table has a column 'drone_model' that represent the drone name character VARCHAR and should be drone_model IS NOT NULL AND drone_model LIKE."},
            {"role": "system", "content": "The 'drones' table has a column 'weight_g' for drone weight in grams."},
            {"role": "system", "content": "The 'drones' table has a column 'payload_weight_g' for payload weight in grams."},
            {"role": "system", "content": "The 'drones' table has a column 'battery_mah' for battery in mah."},
            {"role": "system", "content": "The 'drones' table has a column 'flight_time_min' for flight time in minutes."},
            {"role": "system", "content": "The 'drones' table has a column 'max_range_km' for flight distance in kilometer."},
            {"role": "system", "content": "The 'drones' table has a column 'camera' for camera type and resolution in character VARCHAR and should be camera IS NOT NULL AND camera LIKE"},
            {"role": "system", "content": "The 'drones' table has a column 'noise_db' for noise volume in decibel."},
            {"role": "system", "content": "The 'drones' table has a column 'max_height_km' for max flight height in kilometer."},
            {"role": "system", "content": "The 'drones' table has a column 'max_height_km' for max flight height in kilometer."},
            {"role": "user", "content": f"Convert the following natural language query into a SQL query for a PostgreSQL database. Only return the SQL query with no explanation or additional (SQL: ```sql) text: {user_query}"}
        ]
    )

    generated_sql = response["choices"][0]["message"]["content"]
    print("Generated SQL:", generated_sql)  # Print generated SQL
    return generated_sql

def execute_sql_query(sql_query: str):
    """Executes a raw SQL query using Peewee and returns the results."""
    try:
        cursor = db.cursor()  # Get a cursor explicitly from the db object
        cursor.execute(sql_query)  # Use cursor's execute method
        results_list = cursor.fetchall()  # Fetch results

        return results_list
    except Exception as e:
        print(f"SQL Execution Error: {e}")
        return None
    finally:
        cursor.close()  # Make sure to close the cursor
    
def search_drones_with_ai(user_query: str):
    """Search drones based on user natural language query"""
    sql_query = generate_sql_from_natural_query(user_query)
    results = execute_sql_query(sql_query)

    if results:
        return results
    else:
        print("No results found or query failed.")