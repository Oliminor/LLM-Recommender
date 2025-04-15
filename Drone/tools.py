from Drone.Drone_Picker import execute_sql_query  # from your shared code

def run_sql(input_query: str):
    return execute_sql_query(input_query)

AGENT_TOOLS = {
    "run_sql": run_sql,
}
