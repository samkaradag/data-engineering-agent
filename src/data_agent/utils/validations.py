from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Define tool schemas for LangChain
class InformationSchemaInput(BaseModel):
    dataset_name: Optional[str] = Field(None, description="The name of the dataset to query")
    table_name: Optional[str] = Field(None, description="The name of the table to query")


class InformationSchemaOutput(BaseModel):
    result: List[Dict[str, Any]] = Field(..., description="The result of the information schema query")


class FindRelevantDatasetInput(BaseModel):
    table_name: str = Field(
        ..., description="The name of the table to find the relevant dataset for"
    )
    user_request: str = Field(..., description="The original user request")


class FindRelevantDatasetOutput(BaseModel):
    dataset: Optional[str] = Field(None, description="The name of the relevant dataset")


class ParseLLMOutputInput(BaseModel):
    llm_output: str = Field(..., description="The output from the LLM to be parsed")


class ParseLLMOutputOutput(BaseModel):
    files: List[Dict[str, Any]] = Field(
        ..., description="A JSON object containing file paths and contents"
    )


class UploadAndCompileFilesInput(BaseModel):
    files: List[Dict[str, Any]] = Field(..., description="A list of files to upload and compile")
    workspace_name: str = Field(..., description="The name of the Dataform workspace")


class UploadAndCompileFilesOutput(BaseModel):
    compilation_results: Dict = Field(..., description="The Dataform compilation results")


class FixCompilationErrorsInput(BaseModel):
    files: List[Dict[str, Any]] = Field(..., description="The current files")
    errors: Dict = Field(..., description="The compilation errors")


class FixCompilationErrorsOutput(BaseModel):
    files: List[Dict[str, Any]] = Field(..., description="The fixed files")


class HandleUserRequestInput(BaseModel):
    user_request: str = Field(..., description="The user's request")


class HandleUserRequestOutput(BaseModel):
    parsed_request: Dict[str, Any] = Field(..., description="The parsed request")
    target_tables: Optional[List[Dict]] = Field(
        None, description="The inferred target tables for dimensions and facts"
    )


class GeneratePipelineCodeInput(BaseModel):
    source_tables: List[Dict[str, Any]] = Field(..., description="The source tables")
    target_tables: Optional[List[Dict]] = Field(
        None, description="The target tables for dimensions and facts"
    )  # Now optional
    transformations: Dict[str, Any] = Field(..., description="The transformations to apply")
    intermediate_tables: Optional[List[Dict]] = Field(None, description="The intermediate tables")
    data_quality_checks: Optional[Dict] = Field(
        None, description="Data quality checks to be applied"
    )


class GeneratePipelineCodeOutput(BaseModel):
    pipeline_code: str = Field(..., description="The generated pipeline code")


class RefineLLMResponseInput(BaseModel):
    prompt: str = Field(..., description="The original prompt")
    previous_response: str = Field(..., description="The previous LLM response")
    error: str = Field(..., description="The error encountered")


class RefineLLMResponseOutput(BaseModel):
    refined_response: str = Field(..., description="The refined LLM response")


class AskForClarificationsInput(BaseModel):
    user_request: str = Field(..., description="The user's request")
    missing_information: Optional[List[str]] = Field(
        None,
        description="List of missing information identified in the user request or in the previous turns",
    )


class AskForClarificationsOutput(BaseModel):
    follow_up_questions: List[str] = Field(..., description="Follow-up clarification questions")


class ValidateDataInput(BaseModel):
    table: str = Field(..., description="The table to validate")
    validation_rules: Dict = Field(..., description="The validation rules")


class ValidateDataOutput(BaseModel):
    validation_results: List[Dict] = Field(..., description="The validation results")