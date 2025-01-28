from typing import List, Optional, Dict, Any
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

class AgentState(BaseModel):
    """
    Represents the state of the agent.

    Attributes:
        input (str): The user's input.
        messages (List[BaseMessage]): A list of messages in the conversation.
        next (Optional[str]): The next node to run.
        source_tables (Optional[List[Dict[str, Any]]]): List of source tables.
        target_tables (Optional[List[Dict[str, Any]]]): List of target tables. Defaults to an empty list.
        intermediate_tables (Optional[List[Dict[str, Any]]]): List of intermediate tables.
        files (Optional[List[Dict[str, str]]]): List of files.
        last_compilation_results (Optional[Dict[str, Any]]): Results from the last compilation.
        error (Optional[str]): Any error message.
        pipeline_code (Optional[str]): Generated pipeline code.
        transformations (Optional[Dict[str, Any]]): Transformations to be applied.
        data_quality_checks (Optional[Dict[str, Any]]): Data quality checks to be performed.
        validation_results (Optional[List[Dict[str, Any]]]): Results of data validation.
    """

    input: str
    messages: List[BaseMessage]
    next: Optional[str] = None
    source_tables: Optional[List[Dict[str, Any]]] = None
    target_tables: Optional[List[Dict[str, Any]]] = Field(default_factory=list)  # Initialize as empty list
    intermediate_tables: Optional[List[Dict[str, Any]]] = None
    files: Optional[List[Dict[str, str]]] = None
    last_compilation_results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    pipeline_code: Optional[str] = None
    transformations: Optional[Dict[str, Any]] = None
    data_quality_checks: Optional[Dict[str, Any]] = None
    validation_results: Optional[List[Dict[str, Any]]] = None