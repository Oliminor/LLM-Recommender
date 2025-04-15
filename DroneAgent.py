import os
import io
import openai
import PyPDF2
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from peewee import PostgresqlDatabase, Model, TextField, IntegerField, DecimalField, BooleanField, DoubleField, ForeignKeyField
from playhouse.postgres_ext import ArrayField
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_community.chat_models import ChatOpenAI
from langchain_experimental.tools import PythonREPLTool

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_CHAT_MODEL")

# Initialize DB
db = PostgresqlDatabase(
    "Drone",
    user="user",
    password="admin",
    host="localhost",
    port=5432
)

# Peewee models
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
    max_height_km = DoubleField(null=True)
    payload_weight_g = IntegerField(null=True)
    description = TextField(null=True)

    class Meta:
        table_name = "drones"

class Location(BaseModel):
    name = TextField()
    coordinates = ArrayField(DoubleField)

class DronesLocation(BaseModel):
    fk_location_id = ForeignKeyField(Location, backref="drones")
    fk_drones_id = ForeignKeyField(Drone, backref="locations")

    class Meta:
        table_name = "drones_location"

def describe_drone_schema(query=""):
    return """
    TABLE: drones
    COLUMNS:
      - drone_model: text
      - weight_g: int
      - battery_mah: int
      - flight_time_min: int
      - max_range_km: decimal
      - camera: text
      - geofencing: boolean
      - noise_db: int
      - max_height_km: float
      - payload_weight_g: int
      - description: text

      TABLE: location
      - name: TextField
      - coordinates: ArrayField(DoubleField) 

      TABLE: drones_location
      - fk_location_id: ForeignKeyField(Location, backref="drones")
      - fk_drones_id: ForeignKeyField(Drone, backref="locations")
    """

drone_schema_tool = Tool(
    name="describe_drone_schema",
    func=describe_drone_schema,
    description="Use this to see what tables and columns exist in the drone database"
)

def distance_schema(query=""):
    return """
    Try to search for coordinates in the tables and if no result or part, use other tools to figure out
    """

distance_schema_tool = Tool(
    name="distance_schema",
    func=distance_schema,
    description="Use this tool to see if the prompt distance related"
)

# Helper functions
def extract_text_from_pdf(uploaded_file) -> str:
    text = ""
    with io.BytesIO(uploaded_file.read()) as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"
    return text

def run_sql(query):
    try:
        cursor = db.cursor()
        cursor.execute(query)
        columns = [desc[0] for desc in cursor.description]
        results = cursor.fetchall()
        cursor.close()

        # Store as global for previewing in Streamlit
        st.session_state["last_sql_data"] = {
            "columns": columns,
            "rows": results
        }

        # Markdown for the agent's reasoning output
        header = "| " + " | ".join(columns) + " |"
        separator = "| " + " | ".join(["---"] * len(columns)) + " |"
        rows = ["| " + " | ".join(map(str, row)) + " |" for row in results]
        return "\n".join([header, separator] + rows)

    except Exception as e:
        return f"Error: {str(e)}"

# LangChain tool
sql_tool = Tool(
    name="run_sql",
    func=run_sql,
    description="Use this to query the drone database with SQL"
)

pdf_context = ""

def pdf_lookup(query: str) -> str:
    if not pdf_context:
        return "No PDF uploaded."
    return f"PDF context: {pdf_context}..."

pdf_tool = Tool(
    name="read_pdf",
    func=pdf_lookup,
    description="Use this to answer questions based on uploaded PDF content"
)

# Agent setup
llm = ChatOpenAI(temperature=0, model=model_name)
agent_executor = initialize_agent(
    tools=[sql_tool, pdf_tool, drone_schema_tool, distance_schema_tool, PythonREPLTool()],
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    return_intermediate_steps=True,
    handle_parsing_errors=True,
    llm=llm,
    agent_kwargs={
        "prefix": (
            "You are an intelligent assistant that can use the following tools:\n\n"
            "When choosing a tool, do not use () or quotes like a function. Just specify the tool name."
            "- run_sql: Use to run SQL queries on the drone database.\n"
            "- read_pdf: Drone regulations PDF.\n"
            "- describe_drone_schema: Use to get the structure of the drone database.\n"
            "- distance_schema: If the prompt distance related.\n"
            "- Python_REPL: Use to calculate or transform values.\n\n"
        )
    }
)

# Streamlit UI setup
st.set_page_config(page_title="Drone Agent 🤖", layout="wide")
st.title("🤖 Natural Language Drone & PDF Agent")

# PDF Upload
pdf_file = st.file_uploader("Upload a PDF (optional)", type=["pdf"])
if pdf_file:
    pdf_context = extract_text_from_pdf(pdf_file)
    st.success("📄 PDF uploaded and processed.")

# User Query
query = st.text_input("Ask a question about drones or documents:", placeholder="Which drones can fly over 20km?")

st.markdown(
        """
        <style>
            .block-container {
                max-width: 65% !important;
                margin: 20px;
            }
        </style>
        """,
        unsafe_allow_html=True)

# Run Agent
if st.button("Run Agent") and query:
    with st.spinner("Thinking..."):
        result = agent_executor({"input": query})
        final_answer = result.get("output", "")
        steps = result.get("intermediate_steps", [])

    # Final Answer Section
    st.success("✅ Final Answer")
    st.markdown(final_answer)

    # Optional SQL Table Display
    if "last_sql_data" in st.session_state:
        st.markdown("### 📊 Drone Data Table")
        df = pd.DataFrame(
            st.session_state["last_sql_data"]["rows"],
            columns=st.session_state["last_sql_data"]["columns"]
        )
        
        # Hide ID and similar columns
        df_display = df.drop(columns=['id', 'internal_id'], errors='ignore')
        
        # Display with auto-widths
        st.dataframe(
            df_display,
            use_container_width=False,  # Best for responsive design
            hide_index=True
        )

    # Show Reasoning Toggle
    with st.expander("🧠 Show Agent Thought Process"):
        for i, step in enumerate(steps):
            tool_name = step[0].tool
            tool_input = step[0].tool_input
            tool_output = step[1]

            st.markdown(f"**Step {i+1}: Tool Used – `{tool_name}`**")
            st.code(tool_input, language="sql" if "select" in tool_input.lower() else "text")
            st.markdown(f"**Tool Result:**")
            st.write(tool_output)

# Helpful Query Examples
st.markdown("---")
st.markdown("### 💡 Tips for Writing Good Queries")
st.markdown("- Ask direct questions like **'Which drones support geofencing?'**")
st.markdown("- Ask document-related questions if PDF is uploaded, e.g., **'Summarize the battery section.'**")
st.markdown("- Combine both, e.g., **'Compare drones from the document vs database.'**")