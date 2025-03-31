import openai
import PyPDF2
import io
import json
from peewee import *
from typing import Optional, Dict, List

# Initialize PostgreSQL database connection
db = PostgresqlDatabase(
    "Drone",
    user="user",
    password="admin",
    host="localhost",
    port=5432
)

class Drone(Model):
    # Note: Peewee automatically adds an auto-incrementing 'id' primary key
    drone_model = CharField(max_length=100)
    weight_g = IntegerField()
    battery_mah = IntegerField(null=True)  # Allows NULL values
    flight_time_min = IntegerField(null=True)
    max_range_km = DecimalField(max_digits=5, decimal_places=1, null=True)
    camera = CharField(max_length=100, null=True)
    geofencing = BooleanField(default=False)
    noise_db = IntegerField(null=True)
    
    class Meta:
        database = db
        table_name = "drones"

from peewee import *

def search_drones(**filters):
    """
    Search for drones using dynamic filters.
    You can pass in multiple keyword arguments to filter by specific fields.

    Example usage:
    - search_drones(drone_model="DJI")  # Search by model name
    - search_drones(weight_g__gte=500, weight_g__lte=1500)  # Search by weight range
    - search_drones(flight_time_min__gte=20)  # Search by minimum flight time
    - search_drones(camera="4K")  # Search by camera type
    """

    query = Drone.select()

    for key, value in filters.items():
        if "__" in key:
            field, operation = key.split("__", 1)

            # Get the field from the model
            field_attr = getattr(Drone, field, None)
            if not field_attr:
                raise ValueError(f"Invalid field: {field}")

            # Mapping operations to Peewee filters
            operations = {
                "gte": field_attr >= value,  # Greater than or equal to
                "lte": field_attr <= value,  # Less than or equal to
                "gt": field_attr > value,  # Greater than
                "lt": field_attr < value,  # Less than
                "contains": field_attr.contains(value),  # Case-insensitive search
                "exact": field_attr == value,  # Exact match
            }

            if operation not in operations:
                raise ValueError(f"Invalid operation: {operation}")

            query = query.where(operations[operation])
        else:
            field_attr = getattr(Drone, key, None)
            if not field_attr:
                raise ValueError(f"Invalid field: {key}")

            query = query.where(field_attr == value)

    return list(query)

def search_drones_by_weight(min_weight: int, max_weight: int) -> List[Drone]:
    """Search for drones within a specific weight range."""
    return Drone.select().where((Drone.weight_g >= min_weight) & (Drone.weight_g <= max_weight))

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

def extract_restrictions_from_pdf(pdf_bytes: bytes) -> Dict[str, List[str]]:
    """
    Extracts drone restrictions from PDF bytes using OpenAI's API.
    Uses your existing function with proper bytes conversion.
    """
    text = extract_text_from_pdf(pdf_bytes)
    
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """
             You are an expert in drone regulations. Extract all drone model restrictions 
             from the provided text and return as JSON.

             if there are weight specific requirement, please returns with in gram and the number only and state the minimum and maximum weight limit in that category
             example: if under 25 kg it should return minimum 0, maximum 25000

             return back with every restrictions per category
             the restrictions should be separated by wieght class category
             if possible, must figure out the minimum weight limit based on the other weight categories, if not just use 0
             every restriction should be listed out per weight category
             if a restriction true for multiple category, should be listed on those category
             """},
            {"role": "user", "content": text[:15000]}
        ],
        response_format={"type": "json_object"},
        temperature=0.3
    )
    
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {"error": "Failed to parse restrictions"}
