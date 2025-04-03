import openai
import PyPDF2
import os
import io
import re
from peewee import *
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

def extract_table_names(sql_query: str) -> list:
    """Extracts table names from an SQL query."""
    # This regex captures table names after FROM or JOIN
    table_pattern = r'\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    tables = re.findall(table_pattern, sql_query, re.IGNORECASE)
    return tables

def generate_sql_from_natural_query(user_query: str) -> str:
    """Generate an SQL query using OpenAI based on natural language input"""
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an expert SQL assistant with knowledge of a drones, location and drones_location bridge table database."},
            {"role": "system", "content": "The 'drones' table has a column 'drone_model' that represent the drone name character VARCHAR and should be drone_model IS NOT NULL AND drone_model ILIKE."},
            {"role": "system", "content": "The 'drones' table has a column 'weight_g' for drone weight in grams."},
            {"role": "system", "content": "The 'drones' table has a column 'payload_weight_g' for payload weight in grams."},
            {"role": "system", "content": "The 'drones' table has a column 'battery_mah' for battery in mah."},
            {"role": "system", "content": "The 'drones' table has a column 'flight_time_min' for flight time in minutes."},
            {"role": "system", "content": "The 'drones' table has a column 'max_range_km' for flight distance in kilometer."},
            {"role": "system", "content": "The 'drones' table has a column 'camera' for camera type and resolution in character VARCHAR and should be camera IS NOT NULL AND camera ILIKE"},
            {"role": "system", "content": "The 'drones' table has a column 'noise_db' for noise volume in decibel."},
            {"role": "system", "content": "The 'drones' table has a column 'max_height_km' for max flight height in kilometer."},

            {"role": "system", "content": "The 'location' table has a column 'name' that represent the location name as text and must be name IS NOT NULL AND name ILIKE"},
            {"role": "system", "content": "The 'location' table has a column 'coordinates' that represent the location latitude and longtitude in double precision array."},

            {"role": "system", "content": "The 'drones_location' table has a column 'fk_location_id' that represent the location id in integer and it is a foreign key for the 'location' table."},
            {"role": "system", "content": "The 'drones_location' table has a column 'fk_drones_id' that represent the drones id in integer and it is a foreign key for the 'drones' table."},

            {"role": "system", "content": "If search for location, it should always search for the name first to make sure it exits using ILIKE"},
            {"role": "system", "content": "Only search for numberic based column if numeric value represented if not, dont even generate SQL query for it"},

            {"role": "user", "content": f"Convert the following natural language query into a SQL query for a PostgreSQL database. Only return the SQL query with no explanation or additional text: {user_query}"},
            {"role": "user", "content": f"Never include 'SQL: ```sql' '```' and similar additional text in the SQL query: {user_query}"}
        
        ]
    )

    generated_sql = response["choices"][0]["message"]["content"]
    print("Generated SQL:", generated_sql)  # Print generated SQL
    return generated_sql

def execute_sql_query(sql_query: str):
    """Executes a raw SQL query and fetches results dynamically."""
    try:
        print(f"Executing query: {sql_query}")  # Debugging
        cursor = db.cursor()  
        cursor.execute(sql_query)
        results = cursor.fetchall()
        column_names = [desc[0] for desc in cursor.description]  # Get column names dynamically
        cursor.close()

        # Check if rows are being fetched correctly
        #print("Column Names:", column_names)  # Debug
        #print("Raw Query Results:", results)  # Debug

        # Ensure results are mapped correctly
        formatted_results = [dict(zip(column_names, row)) for row in results] if results else []

        #print("Formatted Results:", formatted_results)  # Debugging
        return formatted_results
    except Exception as e:
        print(f"SQL Execution Error: {e}")
        return None
    
def search_drones_with_ai(user_query: str):
    """Search drones based on user natural language query and format results dynamically."""
    sql_query = generate_sql_from_natural_query(user_query)
    table_names = extract_table_names(sql_query)  # Extract table names
    results = execute_sql_query(sql_query)

    if results:
        #print(f"Query used tables: {table_names}")  # Debugging
        #print("Raw Query Results:", results)  # Debugging

        formatted_results = []

        for row in results:
            #print(f"Row Content: {row}")  # Debugging
            
            if isinstance(row, dict):  
                row_dict = row  # ✅ Just use the dictionary directly
            else:
                row_dict = {}

                if "drones" in table_names:
                    row_dict["Drone Model"] = row[0]
                    row_dict["Weight (g)"] = row[1]
                    row_dict["Battery (mAh)"] = row[2]
                    row_dict["Flight Time (min)"] = row[3]
                    row_dict["Max Range (km)"] = row[4]
                    row_dict["Camera"] = row[5]
                    row_dict["Geofencing"] = row[6]
                    row_dict["Noise (dB)"] = row[7]
                    row_dict["Max Height (km)"] = row[8]
                    row_dict["Payload Weight (g)"] = row[9]
                    row_dict["Description"] = row[10]

                if "location" in table_names:
                    row_dict["Location Name"] = row.get("name", "Unknown")  
                    row_dict["Coordinates"] = row.get("coordinates", "Unknown")

                if "drones_location" in table_names:
                    row_dict["Location ID"] = row[0]
                    row_dict["Drone ID"] = row[1]

            #print(f"Formatted Row: {row_dict}")  # Debugging

            if row_dict:
                formatted_results.append(row_dict)

        return formatted_results
    else:
        print("No results found or query failed.")
        return []