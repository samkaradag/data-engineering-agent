# agent/tools_context.py

from tools.dataform import DataformTools
from tools.bigquery import BigQueryTools
from tools.vertex_ai import VertexAITools
from utils.validations import *

# Initialize tool implementations
dataform_tools = DataformTools(project_id="samets-ai-playground")
bigquery_tools = BigQueryTools(project_id="samets-ai-playground")
vertexai_tools = VertexAITools(
    project_id="samets-ai-playground", location="us-central1"
)

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
    "identify_dataform_files": {
        "function": dataform_tools.identify_dataform_files,
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
    "structure_transformation_request": {
        "function": vertexai_tools.structure_transformation_request,
        "input_model": HandleUserRequestInput,
        "output_model": HandleUserRequestOutput,
    },
    "generate_pipeline_code": {
        "function": dataform_tools.generate_pipeline_code,
        "input_model": GeneratePipelineCodeInput,
        "output_model": GeneratePipelineCodeOutput,
    },
    "fix_json_parse_errors": {
        "function": vertexai_tools.fix_json_parse_errors,
        "input_model": RefineLLMResponseInput,
        "output_model": RefineLLMResponseOutput,
    },
    # "ask_for_clarifications": {
    #     "function": vertexai_tools.ask_for_clarifications,
    #     "input_model": AskForClarificationsInput,
    #     "output_model": AskForClarificationsOutput,
    # },
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
    tools_executor[tool_name] = vertexai_tools.create_runnable_tool(
        tool_name, tool_config
    )