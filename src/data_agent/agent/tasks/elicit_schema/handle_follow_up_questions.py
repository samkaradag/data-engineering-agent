from typing import Dict

from agent.agent_state import AgentState
from langchain_core.messages import HumanMessage
from utils.tracers import trace_calls

@trace_calls
def handle_follow_up_questions(state: AgentState, follow_up_message: str) -> Dict:
    """
    Handles follow-up questions identified in the parsed result.
    """
    messages = state.messages
    messages.append(HumanMessage(content=follow_up_message))
    return {
        "messages": messages,
        "source_tables": state.source_tables,
        "target_tables": state.target_tables,
        "transformations": state.transformations,
        "intermediate_tables": state.intermediate_tables,
        "data_quality_checks": state.data_quality_checks, 
        "next": "ask_clarifications"
    }
