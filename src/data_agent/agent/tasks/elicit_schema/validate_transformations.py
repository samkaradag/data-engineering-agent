from typing import Dict

from agent.agent_state import AgentState
from agent.tools_context import tools_executor
from langchain_core.messages import HumanMessage
from utils.tracers import trace_calls

@trace_calls
def validate_transformations(state: AgentState, parsed_result: Dict) -> Dict:
    """
    Validates the transformations in the parsed request.
    Asks the user for clarification if transformations are not clear.
    """
    transformations = parsed_result.get("transformations", {})
    messages = state.messages

    if not transformations:
        return {
            "messages": messages
        }

    # Example: Check if transformations are empty or not detailed enough
    if not transformations or all(not value for value in transformations.values()):
        return {
            "messages": [
                HumanMessage(content="Transformations are not specified or not clear. Can you provide more details?")
            ]
        }

    # Basic validation: check for unsupported operations
    unsupported_operations = ["UPDATE", "DELETE", "DROP"]
    for transformation_name, details in transformations.items():
        if any(op in details.upper() for op in unsupported_operations):
            return {
                "messages": [
                    HumanMessage(content=f"Transformation '{transformation_name}' contains unsupported operations. Only SELECT operations are supported.")
                ]
            }

    return {"messages": messages}