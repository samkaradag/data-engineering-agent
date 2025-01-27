from typing import Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langgraph.graph import END, StateGraph

from agent.agent_state import AgentState
from tools.dataform import DataformTools
from tools.bigquery import BigQueryTools
from tools.vertex_ai import VertexAITools
from utils.tracers import trace_calls
from utils.validations import *
from utils.prompt_loader import load_prompt # Import the function

import json

# Initialize tool implementations
dataform_tools = DataformTools(project_id="samets-ai-playground")
bigquery_tools = BigQueryTools(project_id="samets-ai-playground")
vertexai_tools = VertexAITools(project_id="samets-ai-playground", location="us-central1")

# Define tool configurations
tools_config = {
    "query_information_schema": {
        "function": bigquery_tools.query_information_schema,
        "input_model": InformationSchemaInput,
        "output_model": InformationSchemaOutput,
    },
    "find_relevant_dataset": {
        "function": bigquery_tools.find_relevant_dataset,
        "input_model": FindRelevantDatasetInput,
        "output_model": FindRelevantDatasetOutput,
    },
    "parse_llm_output": {
        "function": vertexai_tools.parse_llm_output,
        "input_model": ParseLLMOutputInput,
        "output_model": ParseLLMOutputOutput,
    },
    "upload_and_compile_files": {
        "function": dataform_tools.upload_and_compile_files,
        "input_model": UploadAndCompileFilesInput,
        "output_model": UploadAndCompileFilesOutput,
    },
    "fix_compilation_errors": {
        "function": dataform_tools.fix_compilation_errors,
        "input_model": FixCompilationErrorsInput,
        "output_model": FixCompilationErrorsOutput,
    },
    "handle_user_request": {
        "function": vertexai_tools.handle_user_request,
        "input_model": HandleUserRequestInput,
        "output_model": HandleUserRequestOutput,
    },
    "generate_pipeline_code": {
        "function": vertexai_tools.generate_pipeline_code,
        "input_model": GeneratePipelineCodeInput,
        "output_model": GeneratePipelineCodeOutput,
    },
    "refine_llm_response": {
        "function": vertexai_tools.refine_llm_response,
        "input_model": RefineLLMResponseInput,
        "output_model": RefineLLMResponseOutput,
    },
    "ask_for_clarifications": {
        "function": vertexai_tools.ask_for_clarifications,
        "input_model": AskForClarificationsInput,
        "output_model": AskForClarificationsOutput,
    },
    "validate_data": {
        "function": bigquery_tools.validate_data,
        "input_model": ValidateDataInput,
        "output_model": ValidateDataOutput,
    },
}

# Create an empty dictionary for tools_executor
tools_executor = {}

# Populate tools_executor
for tool_name, tool_config in tools_config.items():
    tools_executor[tool_name] = vertexai_tools.create_runnable_tool(tool_name, tool_config)


# Define the nodes of the LangGraph
@trace_calls
def handle_request(state: AgentState) -> Dict:
    """
    Handles the initial user request, parsing and extracting relevant information.
    """
    print("Handling user request...")
    user_request = state.input
    result = tools_executor["handle_user_request"].invoke({"user_request": user_request})

    try:
        parsed_result = json.loads(result)
    except json.JSONDecodeError:
        # Handle JSON decoding error
        return {
            "messages": [
                HumanMessage(
                    content="I couldn't understand the response. Can you please clarify?"
                )
            ]
        }

    messages = state.messages

    if "follow_up" in parsed_result:
        messages.append(HumanMessage(content=parsed_result["follow_up"]))
        return {"messages": messages}

    # Extract and store information from parsed_request
    source_tables = parsed_result.get("source_tables", [])
    target_tables = parsed_result.get("target_tables", "")
    transformations = parsed_result.get("transformations", {})
    intermediate_tables = parsed_result.get("intermediate_tables", [])
    data_quality_checks = parsed_result.get("data_quality_checks", {})

    # Retrieve schema for source tables
    for source in source_tables:
        dataset_name = source.get("dataset")
        table_name = source.get("table")
        columns = bigquery_tools.query_information_schema(
            dataset_name=dataset_name, table_name=table_name
        )
        source["columns"] = columns

    messages.append(
        HumanMessage(
            content=f"Processing request: Source Tables: {source_tables}, Target Table: {target_tables}, Transformations: {transformations}, Intermediate Tables: {intermediate_tables}, Data Quality Checks: {data_quality_checks}"
        )
    )

    # Store in state
    state.source_tables = source_tables
    state.target_tables = target_tables
    state.transformations = transformations
    state.intermediate_tables = intermediate_tables
    state.data_quality_checks = data_quality_checks

    return {
        "messages": messages,
        "source_tables": source_tables,
        "target_tables": target_tables,
        "transformations": transformations,
        "intermediate_tables": intermediate_tables,
        "data_quality_checks": data_quality_checks,
    }

# Define the nodes of the LangGraph
# @trace_calls
# def handle_request(state: AgentState) -> Dict:
#     """
#     Handles the initial user request, parsing and extracting relevant information.
#     """
#     print("Handling user request...")
#     user_request = state.input
#     result = tools_executor["handle_user_request"].invoke({"user_request": user_request})

#     try:
#         parsed_result = json.loads(result)
#     except json.JSONDecodeError:
#         # Handle JSON decoding error
#         return {
#             "messages": [
#                 HumanMessage(
#                     content="I couldn't understand the response. Can you please clarify?"
#                 )
#             ]
#         }

#     messages = state.messages

#     if "follow_up" in parsed_result:
#         messages.append(HumanMessage(content=parsed_result["follow_up"]))
#         return {"messages": messages}

#     # Extract and store information from parsed_request
#     source_tables = parsed_result.get("source_tables", [])
#     target_tables = parsed_result.get("target_tables", "")
#     transformations = parsed_result.get("transformations", {})
#     intermediate_tables = parsed_result.get("intermediate_tables", [])
#     data_quality_checks = parsed_result.get("data_quality_checks", {})

#     # Retrieve schema for source tables
#     for source in source_tables:
#         dataset_name = source.get("dataset")
#         table_name = source.get("table")
#         columns = bigquery_tools.query_information_schema(
#             dataset_name=dataset_name, table_name=table_name
#         )
#         source["columns"] = columns

#     messages.append(
#         HumanMessage(
#             content=f"Processing request: Source Tables: {source_tables}, Target Table: {target_tables}, Transformations: {transformations}, Intermediate Tables: {intermediate_tables}, Data Quality Checks: {data_quality_checks}"
#         )
#     )

#     # Store in state
#     state.source_tables = source_tables
#     state.target_tables = target_tables
#     state.transformations = transformations
#     state.intermediate_tables = intermediate_tables
#     state.data_quality_checks = data_quality_checks

#     return {
#         "messages": messages,
#         "source_tables": source_tables,
#         "target_tables": target_tables,
#         "transformations": transformations,
#         "intermediate_tables": intermediate_tables,
#         "data_quality_checks": data_quality_checks,
#     }

@trace_calls
def confirm_tables(state: AgentState) -> Dict:
    """
    Confirms the inferred target tables and dataset with the user.
    If the dataset is incorrect, asks the user for the correct one or suggests the source dataset.
    """
    print("Confirming target tables and dataset...")
    messages = state.messages
    target_tables = state.target_tables
    source_dataset = (
        state.source_tables[0]["dataset"] if state.source_tables else None
    )  # Get source dataset if available

    if target_tables:
        # Check if all target tables have the same dataset
        first_dataset = target_tables[0]["dataset"]
        all_same_dataset = all(table["dataset"] == first_dataset for table in target_tables)

        if all_same_dataset:
            table_descriptions = "\n".join(
                [
                    f"- {table['table']} (Type: {table['type']}, Dataset: {table['dataset']}): {table['description']}"
                    for table in target_tables
                ]
            )
            confirmation_message = (
                f"I have inferred the following target tables in dataset '{first_dataset}':\n{table_descriptions}\n\n"
                f"1. Is '{first_dataset}' the correct dataset for these tables?\n"
                f"2. Are the table names and types correct?\n"
                f"Please provide any corrections in the format 'dataset: new_dataset, table_name: new_table_name, type: new_type'."
            )
            if (
                source_dataset
                and source_dataset != first_dataset
            ):
                confirmation_message += (
                    f"\nAlternatively, should I use the source dataset '{source_dataset}' for the fact table?"
                )
        else:
            table_descriptions = "\n".join(
                [
                    f"- Dataset: {table['dataset']}, Table: {table['table']} (Type: {table['type']}): {table['description']}"
                    for table in target_tables
                ]
            )
            confirmation_message = (
                f"I have inferred the following target tables:\n{table_descriptions}\n\n"
                f"Please confirm if the datasets, table names, and types are correct.\n"
                f"Provide any corrections in the format 'dataset: new_dataset, table_name: new_table_name, type: new_type'."
            )

        messages.append(AIMessage(content=confirmation_message))
    # else:
    #     messages.append(
    #         AIMessage(
    #             content="I couldn't infer any target tables. Please specify them along with their types and the target dataset."
    #         )
    #     )
    # Skip confirmation if no target tables were inferred and directly go to asking follow up questions
    else:
        return {"messages": messages, "next": "ask_clarifications"}

    return {"messages": messages}

@trace_calls
def handle_user_response_to_table_confirmation(state: AgentState) -> Dict:
    """
    Handles the user's response to the target table confirmation.
    Updates the target_tables in the AgentState based on user feedback.
    """
    print("Handling user response to table confirmation...")
    messages = state.messages
    user_response = state.input
    target_tables = state.target_tables

    # Simple parsing for corrections (can be improved with more robust parsing logic)
    corrections = {}
    if user_response.lower() not in ["yes", "y", "correct", "ok"]:
        for item in user_response.split(","):
            item = item.strip()
            if ":" in item:
                key, value = item.split(":", 1)
                corrections[key.strip()] = value.strip()

    # Update target tables based on corrections
    for table in target_tables:
        if "dataset" in corrections:
            table["dataset"] = corrections["dataset"]
        if table["table"] in corrections:
            table["table"] = corrections[f"{table['table']}"]
        if f"{table['table']}_type" in corrections:  # Check for type corrections
            table["type"] = corrections[f"{table['table']}_type"]
            
    if "dataset" in corrections or any(f"{table['table']}_type" in corrections for table in target_tables) or any(table["table"] in corrections for table in target_tables):
        messages.append(
            AIMessage(
                content=f"Target tables updated based on your feedback: {json.dumps(target_tables, indent=2)}"
            )
        )
    elif user_response.lower() in ["yes", "y", "correct", "ok"]:
        messages.append(AIMessage(content="Great! I'll proceed with the current target tables."))
    else:
        messages.append(
            AIMessage(
                content=f"Please confirm the target tables again. Current target tables: {json.dumps(target_tables, indent=2)}"
            )
        )

    return {"messages": messages, "target_tables": target_tables}

@trace_calls
def generate_code(state: AgentState) -> Dict:
    """
    Generates the data pipeline code based on extracted information.
    """
    print("Generating pipeline code...")
    source_tables = state.source_tables
    target_table = state.target_tables
    transformations = state.transformations
    intermediate_tables = state.intermediate_tables
    data_quality_checks = state.data_quality_checks

    result = tools_executor["generate_pipeline_code"].invoke(
        {
            "source_tables": source_tables,
            "target_table": target_table,
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
    return {"messages": state.messages, "pipeline_code": pipeline_code}


@trace_calls
def parse_output(state: AgentState) -> Dict:
    """
    Parses the LLM output to identify files for upload to Dataform.
    """
    print("Parsing LLM output...")
    llm_output = state.pipeline_code
    result = tools_executor["parse_llm_output"].invoke({"llm_output": llm_output})

    try:
        parsed_result = json.loads(result)
        files = parsed_result.get("files", [])  # Extract the 'files' list
        state.files = files
        state.messages.append(AIMessage(content=f"Parsed output: Found {len(files)} files."))
    except json.JSONDecodeError:
        state.messages.append(AIMessage(content="Error parsing LLM output. Please revise."))

    return {"messages": state.messages, "files": state.files}


@trace_calls
def upload_files(state: AgentState) -> Dict:
    """
    Uploads and compiles the files in Dataform.
    """
    print("Uploading and compiling files...")
    files = state.files
    # Use a default workspace name or make it configurable
    workspace_name = "agent"
    result_str = tools_executor["upload_and_compile_files"].invoke(
        {"files": files, "workspace_name": workspace_name}
    )
    print(f"Compilation Result: {result_str}")

    # Safely parse the result string as JSON
    try:
        result = json.loads(result_str)
    except json.JSONDecodeError:
        print(f"Error parsing JSON from result from upload_and_compile_files: {result_str}")
        state.messages.append(
            AIMessage(
                content=f"Error parsing the result from upload_and_compile_files: {result_str}"
            )
        )
        state.error = f"Error parsing JSON from result from upload_and_compile_files: {result_str}"  # Store error message
        return {
            "messages": state.messages,
            "error": state.error,  # Return the error
        }

    state.last_compilation_results = result

    # Check if the result contains compilation results
    if "name" in result and "compilation_errors" in result:
        # Update last_compilation_results with the result
        state.last_compilation_results = result

        # Check for compilation errors
        if result["compilation_errors"] != []:
            error_messages = [error["message"] for error in result["compilation_errors"]]
            error_message = "\n".join(error_messages)
            state.messages.append(
                AIMessage(content=f"Error during compilation: {error_message}")
            )
            state.error = error_message
            return {
                "messages": state.messages,
                "last_compilation_results": result,
                "error": error_message,
            }
        else:
            state.messages.append(AIMessage(content="Files uploaded and compiled successfully."))
            return {"messages": state.messages, "last_compilation_results": result}
    else:
        # Handle unexpected result type
        error_message = "Unexpected result format from upload_and_compile_files"
        state.messages.append(AIMessage(content=error_message))
        return {
            "messages": state.messages,
            "last_compilation_results": result,
            "error": error_message,
        }



@trace_calls
def fix_errors(state: AgentState) -> Dict:
    """
    Handles compilation errors by invoking the fix_compilation_errors tool.
    """
    print("Fixing compilation errors...")
    files = state.files
    errors = state.last_compilation_results
    result = tools_executor["fix_compilation_errors"].invoke({"files": files, "errors": errors})

    try:
        parsed_result = json.loads(result)
        fixed_files = parsed_result.get("files", [])
        state.files = fixed_files
        state.messages.append(
            AIMessage(content="Compilation errors fixed. Re-uploading files.")
        )
    except json.JSONDecodeError:
        state.messages.append(
            AIMessage(content="Error parsing LLM output for error fixes. Please revise.")
        )

    return {"messages": state.messages, "files": state.files}

@trace_calls
def validate_data_node(state: AgentState) -> Dict:
    """
    Validates the data in the specified tables after the pipeline execution.
    """
    print("Validating data...")
    validation_results_list = []  # Initialize as a list

    # Validate target table
    if state.target_tables:
        for target_table in state.target_tables:
            table_name = target_table["table"]
            table_id = f"{target_table['dataset']}.{table_name}"

            if state.data_quality_checks:
                results = bigquery_tools.validate_data(table_id, state.data_quality_checks)
                validation_results_list.extend(
                    [
                        {
                            "table": table_name,
                            "dataset": target_table["dataset"],
                            "type": target_table["type"],
                            "validation_result": result
                        }
                        for result in results
                    ]
                )
                state.messages.append(
                    AIMessage(content=f"Validated target table: {table_name}")
                )
                for result in results:
                    state.messages.append(
                        AIMessage(
                            content=f"  Rule: {result['rule']}, Result: {result['result']}, Details: {result['details']}"
                        )
                    )

    # Validate intermediate tables
    if state.intermediate_tables:
        for intermediate_table in state.intermediate_tables:
            table_name = intermediate_table["table"]
            table_id = f"{intermediate_table['dataset']}.{table_name}"
            if state.data_quality_checks:
                results = bigquery_tools.validate_data(table_id, state.data_quality_checks)
                validation_results_list.extend(
                    [
                        {
                            "table": table_name,
                            "dataset": intermediate_table["dataset"],
                            "type": "intermediate",
                            "validation_result": result,
                        }
                        for result in results
                    ]
                )
                state.messages.append(
                    AIMessage(content=f"Validated intermediate table: {table_name}")
                )
                for result in results:
                    state.messages.append(
                        AIMessage(
                            content=f"  Rule: {result['rule']}, Result: {result['result']}, Details: {result['details']}"
                        )
                    )

    state.validation_results = validation_results_list
    return {
        "messages": state.messages,
        "validation_results": validation_results_list,  # Return as a list
    }

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

@trace_calls
def decide_to_continue(state: AgentState) -> str:
    """
    Decides whether to continue with code generation or ask the user for clarification.
    """

    print(f"Checking state...{state}")

    # Access attributes directly
    if any(
        isinstance(msg, HumanMessage) and "follow_up" in msg.content
        for msg in state.messages
    ):
        # If there are follow-up questions, go to "ask_clarifications"
        print("Asking for clarifications...")
        return "ask_clarifications"

    # Prioritize handling user response after confirmation
    if state.target_tables is not None and any(
        isinstance(msg, AIMessage) and "Do these look correct" in msg.content
        for msg in state.messages
    ):
        print("Handling user response to confirmation...")
        return "handle_user_response"

    if state.target_tables is not None and any(
        isinstance(msg, AIMessage) and "Would you like to:" in msg.content
        for msg in state.messages
    ):
        print("Handling user response to confirmation...")
        return "handle_user_response"

    if state.target_tables is None:
        # If target tables haven't been confirmed, go to "confirm_tables"
        print("Confirming tables...")
        return "confirm_tables"

    if state.error:  # Check for the 'error' field in the state
        print("Handling specific error...")
        return "handle_errors"

    if not state.files and state.pipeline_code:
        print("Going to parse output")
        return "parse_output"

    print(
        f"Decide to continue: Last compilation results: {state.last_compilation_results}"
    )
    if not state.last_compilation_results and state.files:
        # If no compilation results yet, go to "upload_files"
        print("Uploading files...")
        return "upload_files"

    # Check for compilation errors (updated condition)
    if (
        state.last_compilation_results
        and state.last_compilation_results.get("compilation_errors") != []
    ):
        print("Fixing errors...")
        return "fix_errors"

    # Check if compilation is successful and all files are uploaded(updated condition)
    if (
        state.last_compilation_results
        and state.last_compilation_results.get("compilation_errors") == []
        and state.files
    ):
        print("Asking for further input...")
        return "ask_for_further_input"

    print("Generating code...")
    return "generate_code"


@trace_calls
def ask_clarifications(state: AgentState) -> Dict:
    """
    Asks the user for clarifications based on the follow-up questions.
    """
    print("Asking for clarifications...")
    user_request = state.input

    # Identify missing information by comparing current state with expected information
    missing_information = []
    if not state.source_tables:
        missing_information.append("source_tables")
    if not state.target_table:
        missing_information.append("target_table")
    if not state.transformations:
        missing_information.append("transformations")
    if not state.intermediate_tables:
        missing_information.append("intermediate_tables")
    if not state.data_quality_checks:
        missing_information.append("data_quality_checks")

    # Pass missing_information to ask_for_clarifications
    result = tools_executor["ask_for_clarifications"].invoke(
        {"user_request": user_request, "missing_information": missing_information}
    )

    # Assuming the result contains a list of follow-up questions
    follow_up_questions = result
    state.messages.append(AIMessage(content=f"Clarification questions: {follow_up_questions}"))

    return {"messages": state.messages}


@trace_calls
def handle_errors(state: AgentState) -> Dict:
    """
    Handles errors encountered during the process, such as JSON parsing errors or compilation issues.
    """
    print("Handling errors...")

    # Find the most recent error message, prioritizing AIMessage content
    # Access the error from the state
    error_message = state.error or "Unknown error"

    # Append a HumanMessage to prompt for the next action
    state.messages.append(
        HumanMessage(content=f"An error occurred: {error_message}. What should we do next?")
    )
     # Clear the error to avoid re-handling the same error
    state.error = None
    return {"messages": state.messages}


# Define the workflow graph
workflow = StateGraph(AgentState)

# Add nodes for each step in the process
workflow.add_node("handle_request", handle_request)
workflow.add_node("confirm_tables", confirm_tables)
workflow.add_node("handle_user_response", handle_user_response_to_table_confirmation)

workflow.add_node("generate_code", generate_code)
workflow.add_node("parse_output", parse_output)
workflow.add_node("upload_files", upload_files)
workflow.add_node("fix_errors", fix_errors)
workflow.add_node("ask_clarifications", ask_clarifications)
workflow.add_node("handle_errors", handle_errors)
# workflow.add_node("validate_data", validate_data_node)
workflow.add_node("ask_for_further_input", ask_for_further_input)

# Set the entry point
workflow.set_entry_point("handle_request")

# Define the conditional edges based on the output of `decide_to_continue`
workflow.add_conditional_edges("handle_request", decide_to_continue)
workflow.add_edge("confirm_tables", "handle_user_response")
workflow.add_edge("handle_user_response", "generate_code")

workflow.add_conditional_edges("ask_clarifications", decide_to_continue)
workflow.add_conditional_edges("generate_code", decide_to_continue)
workflow.add_conditional_edges("parse_output", decide_to_continue)
workflow.add_conditional_edges("upload_files", decide_to_continue)
workflow.add_conditional_edges("fix_errors", decide_to_continue)
workflow.add_conditional_edges("handle_errors", decide_to_continue)
# workflow.add_conditional_edges("validate_data", decide_to_continue)


# Add unconditional edge from fix_errors back to upload_files for re-upload
workflow.add_edge("fix_errors", "upload_files")
# workflow.add_edge("validate_data", END)
workflow.add_edge("ask_for_further_input", END)

# Add an edge to the end
# workflow.add_edge("upload_files", END)


# Compile the graph
graph = workflow.compile()

# Interactive Mode Function
@trace_calls
def interactive_mode():
    """
    Runs the agent in interactive mode, allowing for user input at each step.
    """
    print("Welcome to the Dynamic Data Pipeline Agent!")
    print("Describe your pipeline requirements, and I’ll help you step by step.\n")

    while True:
        # Initialize the state at the beginning of each iteration
        state = AgentState(input="", messages=[])

        while True:
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
                            "validation_results"
                        ],
                    },
                ):
                    # Handle streaming output
                    if "__end__" in output:
                        # Update the state with the final output
                        state = output["__end__"]
                    else:
                        for key, value in output.items():
                            if key != "__end__":
                                # print(f"Intermediate Step: {key}") # Remove to avoid too much verbosity
                                # Optionally, print intermediate messages
                                if isinstance(value, dict) and "messages" in value:
                                    for msg in value["messages"]:
                                        if not any(
                                            m.content == msg.content for m in state.messages
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
                                state.messages.extend(
                                    m
                                    for m in value.get("messages", [])
                                    if not any(sm.content == m.content for sm in state.messages)
                                )
                                state.files = value.get("files", state.files)
                                state.last_compilation_results = value.get(
                                    "last_compilation_results", state.last_compilation_results
                                )
                                state.pipeline_code = value.get(
                                    "pipeline_code", state.pipeline_code
                                )
                                state.source_tables = value.get(
                                    "source_tables", state.source_tables
                                )
                                state.target_tables = value.get(
                                    "target_tables", state.target_tables
                                )
                                state.transformations = value.get(
                                    "transformations", state.transformations
                                )
                                state.intermediate_tables = value.get("intermediate_tables",state.intermediate_tables)
                                state.data_quality_checks = value.get("data_quality_checks",state.data_quality_checks)
                                state.validation_results = value.get("validation_results", state.validation_results)
                                state.next = value.get("next", state.next)

                    

                # Get user input for the next step based on state.next or other relevant conditions
                if state.next in ["ask_clarifications","handle_errors", "handle_user_response"]:
                    user_input = input("Your response (or type 'exit' to quit): ")
                    if user_input.lower() in ["exit"]:
                        print("Goodbye! See you next time!")
                        return  # Exit the entire function
                    state.input = user_input
                    state.messages.append(HumanMessage(content=user_input))
                elif state.next == "ask_for_further_input":
                    state.next = END

                # Check if the graph execution has reached the end
                if END in state.next:
                    print("Pipeline execution completed. Starting over...")
                    break

            except KeyError as e:
                if "branch:upload_files:wrapper:end" in str(e):
                    print("Encountered a known LangGraph issue. Restarting the process...")
                    break  # Break the inner loop to restart
                else:
                    raise  # Re-raise other KeyErrors