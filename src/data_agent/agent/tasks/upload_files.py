import json
from typing import Dict
from langchain_core.messages import AIMessage
from agent.agent_state import AgentState
from utils.tracers import trace_calls
from agent.tools_context import tools_executor, bigquery_tools  # Import tools_executor

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
            return {"messages": state.messages, "last_compilation_results": result, "next": "ask_for_further_input"}
    else:
        # Handle unexpected result type
        error_message = "Unexpected result format from upload_and_compile_files"
        state.messages.append(AIMessage(content=error_message))
        return {
            "messages": state.messages,
            "last_compilation_results": result,
            "error": error_message
        }
