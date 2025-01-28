from typing import Dict, List
from agent.agent_state import AgentState
from utils.tracers import trace_calls
from langchain_core.messages import HumanMessage

@trace_calls
def get_initial_user_request(state: AgentState) -> Dict:
    """
    Gets the initial user request.
    """
    print("Getting initial user request...")
    # This is a placeholder function.
    # You may need to implement the logic for getting the user request.
    # For now, we will just return the input from the state.
    return {
        "messages": [HumanMessage(content=state.input),HumanMessage(content=state.input)]
    }