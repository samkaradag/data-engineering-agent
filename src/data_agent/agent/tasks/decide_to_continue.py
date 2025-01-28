from typing import Dict
from agent.agent_state import AgentState
from utils.tracers import trace_calls
from langgraph.graph import END

@trace_calls
def decide_to_continue(state: AgentState) -> str:
    """
    Decides the next action to take based on the current state.
    """
    print("Checking state...")
    print(f"state...{state}")

    if state.next == "generate_code":
        print("Generate Code...")
        return "generate_code"
    elif state.next == "ask_clarifications":
        print("Asking for clarifications...")
        return "ask_clarifications"
    elif state.next == "identify_dataform_files":
        print("Identifying dataform files...")
        return "identify_dataform_files"
    elif state.next == "upload_files":
        print("Uploading files...")
        return "upload_files"
    elif state.next == "fix_errors":
        print("Fixing errors...")
        return "fix_errors"
    elif state.next == "handle_errors":
        print("Handling errors...")
        return "handle_errors"
    elif state.next == "validate_data":
        print("Validating data...")
        return "validate_data"
    elif state.next == "elicit_schema":
        print("Elicit Schema...")
        return "elicit_schema"
    else:
        print("Ending conversation...")
        return END