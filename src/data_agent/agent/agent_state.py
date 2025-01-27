from typing import List, Optional, Dict
from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """
    Represents the state of the agent in the pipeline.

    Attributes:
        input (str): The initial input or prompt provided to the agent.
        messages (List[BaseMessage]): A list of messages representing the conversation history.
        files: A list of files parsed from LLM output, ready for upload to Dataform.
        last_compilation_results: A dictionary of compilation results from Dataform.
        next (str, optional): The next operation or step to be executed, if determined.
        pipeline_code (str, optional): The generated data pipeline code.
        source_tables (List[Dict], optional): List of source tables.
        target_table (str, optional): The target table.
        transformations (Dict, optional): The transformations to apply.
        intermediate_tables (List[Dict], optional): List of intermediate tables
        data_quality_checks (Dict, optional): List of data quality checks to be applied
        validation_results (List[Dict], optional): List of validation results

    """

    input: str
    messages: List[BaseMessage]
    files: Optional[List[Dict]] = None
    last_compilation_results: Optional[Dict] = None
    next: Optional[str] = None
    pipeline_code: Optional[str] = None
    source_tables: Optional[List[Dict]] = None
    target_tables: Optional[List[Dict]] = None  # List of dictionaries for dimension and fact tables
    error: Optional[str] = None  # Field to store error messages

    transformations: Optional[Dict] = None
    intermediate_tables: Optional[List[Dict]] = None
    data_quality_checks: Optional[Dict] = None
    validation_results: Optional[List[Dict]] = None

    class Config:
        arbitrary_types_allowed = True