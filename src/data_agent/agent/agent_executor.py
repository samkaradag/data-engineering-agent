from typing import Dict, List, Optional
from copy import deepcopy
from IPython.display import display, Image

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langgraph.graph import END, StateGraph
from collections import defaultdict

from agent.agent_state import AgentState
from tools.dataform import DataformTools
from tools.bigquery import BigQueryTools
from tools.vertex_ai import VertexAITools
from utils.tracers import trace_calls
from utils.validations import *
from utils.prompt_loader import load_prompt  # Import the function
from agent.tools_context import tools_executor, bigquery_tools  # Import tools_executor and bigquery_tools


import json

# Import task functions
from agent.tasks.get_initial_user_request import get_initial_user_request
from agent.tasks.elicit_schema import elicit_schema
from agent.tasks.generate_code import generate_code
from agent.tasks.identify_dataform_files import identify_dataform_files
from agent.tasks.upload_files import upload_files
from agent.tasks.ask_clarifications import ask_clarifications
from agent.tasks.ask_for_further_input import ask_for_further_input
from agent.tasks.decide_to_continue import decide_to_continue

# Initialize tool implementations
dataform_tools = DataformTools(project_id="samets-ai-playground")
bigquery_tools = BigQueryTools(project_id="samets-ai-playground")
vertexai_tools = VertexAITools(project_id="samets-ai-playground", location="us-central1")

# Define the workflow graph
workflow = StateGraph(AgentState)

# Add nodes for each step in the process
workflow.add_node("get_initial_user_request", get_initial_user_request)
workflow.add_node("elicit_schema", elicit_schema)
workflow.add_node("generate_code", generate_code)
workflow.add_node("identify_dataform_files", identify_dataform_files)
workflow.add_node("upload_files", upload_files)
workflow.add_node("ask_clarifications", ask_clarifications)
workflow.add_node("ask_for_further_input", ask_for_further_input)
workflow.add_node("decide_to_continue", decide_to_continue)

# Set the entry point
workflow.set_entry_point("get_initial_user_request")

workflow.add_edge("ask_for_further_input", END)
workflow.add_edge("get_initial_user_request", "elicit_schema")
workflow.add_edge("ask_clarifications", "elicit_schema")

# Define the conditional edges based on the output of `decide_to_continue`
workflow.add_conditional_edges("elicit_schema", decide_to_continue)
workflow.add_conditional_edges("ask_clarifications", decide_to_continue)
workflow.add_conditional_edges("generate_code", decide_to_continue)
workflow.add_conditional_edges("identify_dataform_files", decide_to_continue)
workflow.add_conditional_edges("upload_files", decide_to_continue)


# Compile the graph
graph = workflow.compile()

display(Image(graph.get_graph().draw_mermaid_png()))


@trace_calls
def update_agent_state(
    state: AgentState,
    output: Dict
) -> AgentState:
    """
    Updates the agent state properly by creating a deep copy of the state
    and updating the required keys.
    """
    new_state = deepcopy(state)  # Create a deep copy to avoid modifying the original state directly

    for key, value in output.items():
        if key != "__end__":
            if key == "messages":
                new_state.messages.extend(
                    m for m in value if not any(sm.content == m.content for sm in new_state.messages)
                )
            elif key == "source_tables":
                # Handle source_tables update
                if value is not None:
                    new_state.source_tables = value
            elif key == "target_tables":
                # Handle target_tables update
                if value is not None:
                    new_state.target_tables = value
            elif key == "next":
                new_state.next = value
            elif hasattr(new_state, key):
                setattr(new_state, key, value)

    return new_state


# Interactive Mode Function
@trace_calls
def interactive_mode():
    """
    Runs the agent in interactive mode, allowing for user input at each step.
    """
    print("Welcome to the Dynamic Data Pipeline Agent!")
    print("Describe your pipeline requirements, and I’ll help you step by step.\n")
    iteration_count = 0
    while True:
        # Initialize the state at the beginning of each iteration
        state = AgentState(
            input="", messages=[], target_tables=[]
        )  # Initialize target_tables to []

        while True:
            iteration_count += 1
            
            print(f"Iteration: {iteration_count}")
            if not state.messages:
                # Get the initial user request
                user_request = input("Your request: ")
                if user_request.lower() in ["exit", "quit"]:
                    print("Goodbye! See you next time!")
                    return  # Exit the entire function
                state.input = user_request
                state.messages.append(HumanMessage(content=user_request))

            # Use stream to get intermediate steps
            try:
                for output in graph.stream(
                    state,
                    {
                        "recursion_limit": 20,
                        "output_keys": [
                            "next",
                            "messages",
                            "files",
                            "last_compilation_results",
                            "pipeline_code",
                            "source_tables",
                            "target_tables",
                            "transformations",
                            "intermediate_tables",
                            "data_quality_checks",
                            "validation_results",
                        ],
                    },
                ):
                    # Handle streaming output
                    if "__end__" in output:
                        # Update the state with the final output
                        state = output["__end__"]
                    else:
                        # state = update_agent_state(state, output)
                        for key, value in output.items():
                            if key != "__end__":
                                # Optionally, print intermediate messages
                                if isinstance(value, dict) and "messages" in value:
                                    for msg in value["messages"]:
                                        if not any(
                                            m.content == msg.content
                                            for m in state.messages
                                        ):
                                            if isinstance(msg, HumanMessage):
                                                print(f"Human: {msg.content}")
                                            elif isinstance(msg, AIMessage):
                                                print(f"AI: {msg.content}")
                                            elif isinstance(msg, SystemMessage):
                                                print(f"System: {msg.content}")
                                            else:
                                                print(f"Unknown message type: {msg}")
                                # Update state based on intermediate output

                    if state.next == "ask_for_further_input":
                        state.next = END
                    elif state.next == END:
                        print("Pipeline execution completed. Starting over...")
                        break

                    user_request_dummy = input("Continue? (yes/no):) ")

            except KeyError as e:
                if "branch:upload_files:wrapper:end" in str(e):
                    print(
                        "Encountered a known LangGraph issue. Restarting the process..."
                    )
                    break  # Break the inner loop to restart
                else:
                    raise  # Re-raise other KeyErrors
            
            