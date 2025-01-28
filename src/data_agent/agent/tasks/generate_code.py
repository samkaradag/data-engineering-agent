import json
from typing import Dict, List, Optional, Any


from langchain_core.messages import AIMessage

from agent.agent_state import AgentState
from utils.tracers import trace_calls
from agent.tools_context import tools_executor, bigquery_tools  # Import tools_executor

# Assuming tools_executor is available globally, as initialized in agent_executor.py
# For example:
# tools_executor = {
#     "generate_pipeline_code": lambda x: {"pipeline_code": "some code"}, # Replace with actual implementation
# }

@trace_calls
def generate_code(state: AgentState) -> Dict:
    """
    Generates the data pipeline code based on extracted information.
    """
    print("Generating pipeline code...")
    source_tables = state.source_tables
    target_tables: Optional[List[Dict[str, Any]]] = state.target_tables or [] # Handle empty list case
    transformations = state.transformations
    intermediate_tables = state.intermediate_tables
    data_quality_checks = state.data_quality_checks

    result = tools_executor["generate_pipeline_code"].invoke(
        {
            "source_tables": source_tables,
            "target_table": target_tables,
            "transformations": transformations,
            "intermediate_tables": intermediate_tables,
            "data_quality_checks": data_quality_checks,
        }
    )

    try:
        # Load the JSON string into a Python dictionary
        json_result = json.loads(result)

        # Extract the 'pipeline_code' from the dictionary
        pipeline_code = json_result.get("pipeline_code", "")

        if not pipeline_code:
            raise ValueError("pipeline_code is missing or empty in the result")

    except (json.JSONDecodeError, ValueError) as e:
        error_message = f"Error processing generated code: {e}"
        print(error_message)
        state.messages.append(AIMessage(content=error_message))
        return {"messages": state.messages, "error": error_message}

    state.messages.append(AIMessage(content=pipeline_code))
    state.pipeline_code = pipeline_code
    return {"messages": state.messages, "pipeline_code": pipeline_code, "next": "identify_dataform_files"}

