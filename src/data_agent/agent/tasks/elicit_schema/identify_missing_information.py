from typing import Dict, List, Optional
from agent.agent_state import AgentState
from utils.tracers import trace_calls

@trace_calls
def identify_missing_information(state: AgentState, parsed_result: Dict) -> Optional[List[str]]:
    """
    Identifies missing information in the parsed request, including missing datasets.
    """
    missing_info = []
    
    
    # Check for missing source tables
    # if not parsed_result.get("source_tables") and state.source_tables is None:
    if not parsed_result.get("source_tables"):
        missing_info.append("source_tables")
    else:
        # Check for missing datasets in source tables
        for source in parsed_result.get("source_tables"):
            if not source.get("dataset"):
                missing_info.append("source_table_dataset")
                break  # Exit the loop if a missing dataset is found

    # Check for missing target tables
    # if not parsed_result.get("target_tables") and state.target_tables is None:
    if not parsed_result.get("target_tables"):
        missing_info.append("target_tables")
    else:
        # Check for missing datasets in target tables
        for target in parsed_result.get("target_tables"):
            if not target.get("dataset"):
                missing_info.append("target_table_dataset")
                break  # Exit the loop if a missing dataset is found

    # Check for missing transformations
    if not parsed_result.get("transformations")  and state.transformations is None:
        missing_info.append("transformations")

    return list(set(missing_info))  # Remove duplicates