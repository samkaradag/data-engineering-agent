import json
from typing import Dict
from langchain_core.messages import AIMessage
from agent.agent_state import AgentState
from utils.tracers import trace_calls
from agent.tools_context import tools_executor, bigquery_tools  # Import tools_executor

# from utils.config import get_vertexai_model
# from utils.prompt_loader import load_prompt

@trace_calls
def identify_dataform_files(state: AgentState) -> Dict:
    """
    Parses the LLM output to identify files for upload to Dataform.
    """
    print("Parsing LLM output...")
    llm_output = state.pipeline_code
    result = tools_executor["identify_dataform_files"].invoke({"llm_output": llm_output})

    try:
        parsed_result = json.loads(result)
        files = parsed_result.get("files", [])  # Extract the 'files' list
        state.files = files
        state.messages.append(AIMessage(content=f"Parsed output: Found {len(files)} files."))
    except json.JSONDecodeError:
        state.messages.append(AIMessage(content="Error parsing LLM output. Please revise."))

    return {"messages": state.messages, "files": state.files, "next": "upload_files"}
