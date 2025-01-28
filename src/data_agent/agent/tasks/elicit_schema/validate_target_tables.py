from typing import Dict

from agent.agent_state import AgentState
from agent.tools_context import bigquery_tools
from langchain_core.messages import HumanMessage
from utils.tracers import trace_calls

@trace_calls
def validate_target_tables(state: AgentState, parsed_result: Dict) -> Dict:
    """
    Validates the target tables in the parsed request.
    Checks if the dataset exists (if provided).
    """
    target_tables = parsed_result.get("target_tables", [])
    messages = state.messages

    for target in target_tables:
        dataset_name = target.get("dataset")

        # Check if dataset name is provided
        if not dataset_name:
            return {
                "messages": [
                    HumanMessage(content="Target table information is incomplete. Please specify the dataset name.")
                ]
            }

        # Check if the dataset exists
        if not bigquery_tools.dataset_exists(dataset_name):
            return {
                "messages": [
                    HumanMessage(content=f"Dataset '{dataset_name}' not found. Please provide a valid dataset name.")
                ]
            }

    return {"messages": messages}