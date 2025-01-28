from typing import Dict

from agent.agent_state import AgentState
from agent.tools_context import bigquery_tools
from langchain_core.messages import HumanMessage
from utils.tracers import trace_calls

@trace_calls
def validate_source_tables(state: AgentState, parsed_result: Dict) -> Dict:
    """
    Validates the source tables in the parsed request.
    Checks if the dataset and table exist and if they are valid.
    """
    source_tables = parsed_result.get("source_tables", [])
    messages = state.messages
    
    for source in source_tables:
        dataset_name = source.get("dataset")
        table_name = source.get("table")

        # Check if dataset and table name are provided
        if not dataset_name or not table_name:
            return {
                "messages": [
                    HumanMessage(content="Source table information is incomplete. Please specify both dataset and table name.")
                ]
            }

        # Check if the dataset exists
        if not bigquery_tools.dataset_exists(dataset_name):
            return {
                "messages": [
                    HumanMessage(content=f"Dataset '{dataset_name}' not found. Please provide a valid dataset name.")
                ]
            }

        # Check if the table exists
        if not bigquery_tools.table_exists(dataset_name, table_name):
            return {
                "messages": [
                    HumanMessage(content=f"Table '{table_name}' not found in dataset '{dataset_name}'. Please provide a valid table name.")
                ]
            }
        
        # Retrieve and add column information for the table
        try:
            columns = bigquery_tools.query_information_schema(
                dataset_name=dataset_name, table_name=table_name
            )
            source["columns"] = columns
        except Exception as e:
            return {
                "messages": [
                    HumanMessage(content=f"Error retrieving information for table '{table_name}' in dataset '{dataset_name}': {e}")
                ]
            }

    return {"messages": messages}  # Return updated messages if no error