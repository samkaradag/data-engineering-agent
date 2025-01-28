from typing import Dict
from langgraph.graph import END  # Import END here

from langchain_core.messages import AIMessage

from agent.agent_state import AgentState
from utils.tracers import trace_calls
from agent.tools_context import tools_executor, bigquery_tools  # Import tools_executor


@trace_calls
def ask_for_further_input(state: AgentState) -> Dict:
    """
    Asks the user if they have any further requests or modifications.
    """
    print("Asking for further input...")
    messages = state.messages
    messages.append(
        AIMessage(
            content="The data pipeline has been generated and compiled successfully. Is there anything else you would like me to do? (e.g., modify the pipeline, add data quality checks, etc.)"
        )
    )
    return {"messages": messages, "next": END}  # Update state.next to END
