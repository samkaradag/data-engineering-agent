import json
from typing import Dict, List, Optional

from agent.agent_state import AgentState
# from utils.config import get_vertexai_model
from langchain_core.messages import HumanMessage
from utils.tracers import trace_calls

@trace_calls
def ask_clarifications(state: AgentState) -> Dict:
    """
    Asks the user for clarifications based on the follow-up questions.
    """
    # model = get_vertexai_model()
    print("Asking for clarifications...")
    messages = state.messages
    # Find the HumanMessage with the missing info list
    # Make sure that it is the last human message
    missing_info_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage) and isinstance(msg.content, list):
            missing_info_message = msg
            break

    if missing_info_message is None:
        print("No missing information found.")
        return {"messages": messages}

    missing_info = missing_info_message.content
    
    follow_up_questions = []

    if "source_table_dataset" in missing_info:
        follow_up_questions.append("Which dataset contains the source table?")

    if "target_table_dataset" in missing_info:
        follow_up_questions.append("Which dataset should the target tables be in?")
    
    # Add more specific questions based on missing_info
    if "source_tables" in missing_info:
        follow_up_questions.append("Which source tables should be used?")
    if "target_tables" in missing_info:
        follow_up_questions.append(
            "What are the desired target tables for dimensions and facts? Please provide names, types (dimension or fact), and a brief description for each."
        )
    if "transformations" in missing_info:
        follow_up_questions.append(
            "What specific transformations should be applied to the data?"
        )
    if "intermediate_tables" in missing_info:
        follow_up_questions.append(
            "Are there any intermediate tables needed? If so, what are their names, datasets, and purposes?"
        )
    if "data_quality_checks" in missing_info:
        follow_up_questions.append(
            "What data quality checks should be applied? (e.g., uniqueness, non-nullity, specific ranges)"
        )

    # If no specific missing information is provided, ask general questions
    if not missing_info:
        follow_up_questions.extend(
            [
                "What is the desired output format (e.g., table, view, materialized view)?",
                "Are there any performance considerations or specific requirements for the pipeline?",
                "Where should the pipeline be deployed (e.g., Dataform, BigQuery)?",
                "How should the pipeline be scheduled (e.g., daily, hourly)?",
            ]
        )
    
    # Collect user responses for each question
    user_responses = []
    for question in follow_up_questions:
        response = input(f"{question}\nYour response: ")
        user_responses.append(response)

    # Update the messages in the state with the collected responses
    messages.append(HumanMessage(content=json.dumps(user_responses)))

    # Extract and update missing information based on user responses
    updated_source_tables = state.source_tables
    updated_target_tables = state.target_tables

    for i, info_type in enumerate(missing_info):
        if i < len(user_responses):
            user_response = user_responses[i]
            if info_type == "source_table_dataset":
                for table in updated_source_tables:
                    table["dataset"] = user_response
            elif info_type == "target_table_dataset":
                for table in updated_target_tables:
                    table["dataset"] = user_response

    return {
        "messages": messages,
        "source_tables": updated_source_tables,
        "target_tables": updated_target_tables,
        "next": "elicit_schema"
    }