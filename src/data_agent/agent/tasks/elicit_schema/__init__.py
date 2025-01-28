import json
from typing import Dict

from agent.agent_state import AgentState
from agent.tasks.elicit_schema.identify_missing_information import (
    identify_missing_information,
)
from agent.tasks.elicit_schema.validate_source_tables import validate_source_tables
from agent.tasks.elicit_schema.validate_target_tables import validate_target_tables
from agent.tasks.elicit_schema.validate_transformations import validate_transformations
from agent.tasks.elicit_schema.handle_follow_up_questions import (
    handle_follow_up_questions,
)
from agent.tools_context import tools_executor
from langchain_core.messages import HumanMessage, SystemMessage
from utils.tracers import trace_calls

@trace_calls
def elicit_schema(state: AgentState) -> Dict:
    """
    Orchestrates the elicitation and validation of the data schema.
    """
    print("Eliciting data schema...")
    user_request = state.input
    messages = state.messages

    # Check if the last message is a response to follow-up questions
    if state.messages and isinstance(state.messages[-1].content, str) and state.source_tables and state.target_tables:
        try:
            user_responses = json.loads(state.messages[-1].content)
            if isinstance(user_responses, list):
                # Assuming the order of responses corresponds to the order of missing info
                missing_info = identify_missing_information(state, {}) # Pass empty dict as we are already populating information in the state
                for i, info_type in enumerate(missing_info):
                    if i < len(user_responses):
                        user_response = user_responses[i]
                        if info_type == "source_table_dataset":
                            # Update each source table with the provided dataset
                            for table in state.source_tables:
                                table["dataset"] = user_response
                        elif info_type == "target_table_dataset":
                            # Update each target table with the provided dataset
                            for table in state.target_tables:
                                table["dataset"] = user_response
        except json.JSONDecodeError:
            print("Error decoding user responses JSON.")

    # Only call structure_transformation_request if source_tables or target_tables are not already populated
    if state.source_tables is None or state.target_tables is None:
        # Structure the transformation request
        result = tools_executor["structure_transformation_request"].invoke(
            {"user_request": user_request}
        )

        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError as e:
                print(f"Error decoding result JSON. JSON:{result} Error:{e}")
                return {
                    "messages": [
                        SystemMessage(
                            content = f"Error decoding result JSON. JSON:{result} Error:{e}"
                        )
                    ],
                    "next": None
                }

        parsed_result = result
        
        # Update state with structured information
        state.source_tables = parsed_result.get("source_tables", [])
        state.target_tables = parsed_result.get("target_tables", [])
        state.transformations = parsed_result.get("transformations", {})
        state.intermediate_tables = parsed_result.get("intermediate_tables", [])
        state.data_quality_checks = parsed_result.get("data_quality_checks", {})


    else:
        # If source_tables and target_tables are already populated, use the existing information in the state
        parsed_result = {
            "source_tables": state.source_tables,
            "target_tables": state.target_tables,
            "transformations": state.transformations,
            "intermediate_tables": state.intermediate_tables,
            "data_quality_checks": state.data_quality_checks
        }

    missing_info = identify_missing_information(state, parsed_result)
    if missing_info:
        return handle_follow_up_questions(state, missing_info)

    # Validate source tables
    validation_result = validate_source_tables(state, parsed_result)
    if validation_result.get("error"):
        return validation_result

    # Validate target tables
    validation_result = validate_target_tables(state, parsed_result)
    if validation_result.get("error"):
        return validation_result

    # Validate transformations
    validation_result = validate_transformations(state, parsed_result)
    if validation_result.get("error"):
        return validation_result

    # Merge messages from validation results into the state's messages
    messages = state.messages
    if "messages" in validation_result:
        messages.extend(validation_result["messages"])

    # Append a message to indicate successful processing
    messages.append(
        HumanMessage(
            content=f"Processing request: Source Tables: {state.source_tables}, Target Table: {state.target_tables}, Transformations: {state.transformations}, Intermediate Tables: {state.intermediate_tables}, Data Quality Checks: {state.data_quality_checks}"
        )
    )

    return {
        "messages": messages,  # Return the merged messages
        "source_tables": state.source_tables,
        "target_tables": state.target_tables,
        "transformations": state.transformations,
        "intermediate_tables": state.intermediate_tables,
        "data_quality_checks": state.data_quality_checks,
        "next": "generate_code"
    }