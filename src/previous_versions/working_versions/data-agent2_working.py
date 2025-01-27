from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import json
import re
import os

from google.cloud import bigquery
from google.cloud import dataform_v1beta1
import vertexai
from vertexai.generative_models import GenerativeModel, Tool, FunctionDeclaration  

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

from langchain_core.runnables import (
    ConfigurableField,
    ConfigurableFieldSpec,
    Runnable,
    RunnableBinding,
    RunnableConfig,
    RunnableLambda,
    RunnableMap,
    RunnablePassthrough,
    RunnableWithFallbacks,
    ensure_config,
)
from langchain_core.runnables.base import (
    Other,
    RunnableLike,
    coerce_to_runnable,
)

from langchain_core.language_models import BaseChatModel, BaseLanguageModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError

from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolExecutor, ToolInvocation

from langchain_google_vertexai import ChatVertexAI
# from langchain_google_vertexai.functions_utils import _format_pydantic_to_vertex_function # Removed
from langchain_core.utils.function_calling import convert_to_openai_function
# No need for FunctionDescription import as convert_to_openai_function handles it

def trace_calls(func):
    def wrapper(*args, **kwargs):
        # print(f"Calling {func.__name__} with args={args} kwargs={kwargs}")
        # print(f"Calling {func.__name__} ")
        result = func(*args, **kwargs)
        # print(f"{func.__name__} returned {result}")
        return result
    return wrapper

# Define the state of the agent
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
    """

    input: str
    messages: List[BaseMessage]
    files: Optional[List[Dict]] = None
    last_compilation_results: Optional[Dict] = None
    next: Optional[str] = None
    pipeline_code: Optional[str] = None
    source_tables: Optional[List[Dict]] = None
    target_table: Optional[str] = None
    transformations: Optional[Dict] = None

    class Config:
        arbitrary_types_allowed = True

# Define tool schemas for LangChain
class InformationSchemaInput(BaseModel):
    dataset_name: Optional[str] = Field(None, description="The name of the dataset to query")
    table_name: Optional[str] = Field(None, description="The name of the table to query")

class InformationSchemaOutput(BaseModel):
    result: List[Dict[str, Any]] = Field(..., description="The result of the information schema query")

class FindRelevantDatasetInput(BaseModel):
    table_name: str = Field(..., description="The name of the table to find the relevant dataset for")
    user_request: str = Field(..., description="The original user request")

class FindRelevantDatasetOutput(BaseModel):
    dataset: Optional[str] = Field(None, description="The name of the relevant dataset")

class ParseLLMOutputInput(BaseModel):
    llm_output: str = Field(..., description="The output from the LLM to be parsed")

class ParseLLMOutputOutput(BaseModel):
    files: List[Dict[str, Any]] = Field(..., description="A JSON object containing file paths and contents")

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

class GeneratePipelineCodeInput(BaseModel):
    source_tables: List[Dict[str, Any]] = Field(..., description="The source tables")
    target_table: str = Field(..., description="The target table")
    transformations: Dict[str, Any] = Field(..., description="The transformations to apply")

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

class AskForClarificationsOutput(BaseModel):
    follow_up_questions: List[str] = Field(..., description="Follow-up clarification questions")

# Define custom tool implementations
class CustomToolImplementations:
    @trace_calls
    def __init__(self, project_id, location="us-central1", model_name="gemini-2.0-flash-exp"):
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)

        self.model = GenerativeModel(model_name)
        self.bigquery_client = bigquery.Client(project=project_id)
        self.project_id = project_id
    @trace_calls
    def validate_dataset_name(self, dataset_name):
        """
        Validates a BigQuery dataset name.
        """
        if not dataset_name:
            return False
        # Dataset names must start with a letter or underscore, and can contain letters, numbers, and underscores
        pattern = r"^[a-zA-Z_][a-zA-Z0-9_]*$"
        return bool(re.fullmatch(pattern, dataset_name))

    @trace_calls
    def validate_table_name(self, table_name):
        """
        Validates a BigQuery table name.
        """
        if not table_name:
            return False
        # Table names can contain letters, numbers, and underscores
        pattern = r"^[a-zA-Z0-9_]+$"
        return bool(re.fullmatch(pattern, table_name))
    
    @trace_calls
    def query_information_schema(self, dataset_name=None, table_name=None):
        """
        Queries BigQuery information_schema to retrieve datasets, tables, and columns.
        """
        if not dataset_name:
            query = f"""
            SELECT schema_name 
            FROM `{self.project_id}.region-us.INFORMATION_SCHEMA.SCHEMATA`
            """
            return [row.schema_name for row in self.bigquery_client.query(query)]

        if not table_name:
            query = f"""
            SELECT table_name
            FROM `{self.project_id}.{dataset_name}.INFORMATION_SCHEMA.TABLES`
            """
            return [row.table_name for row in self.bigquery_client.query(query)]

        query = f"""
        SELECT column_name, data_type
        FROM `{self.project_id}.{dataset_name}.INFORMATION_SCHEMA.COLUMNS`
        WHERE table_name = '{table_name}'
        """
        return [{"name": row.column_name, "type": row.data_type} for row in self.bigquery_client.query(query)]

    @trace_calls
    def find_relevant_dataset(self, table_name, user_request):
        """
        Finds the relevant dataset containing the given table by querying 
        INFORMATION_SCHEMA and analyzing the user request.
        Handles wildcards and partial matches in table names.
        """
        print(f"Finding relevant dataset for table '{table_name}'...")
        datasets = self.query_information_schema()
        for dataset in datasets:
            tables = self.query_information_schema(dataset_name=dataset)
            for table in tables:
                # Use regex to match table names with wildcards
                if re.match(table_name.replace("*", ".*"), table):
                    print(f"Found table '{table}' in dataset '{dataset}'")
                    return dataset
        return None

    @trace_calls
    def parse_llm_output(self, llm_output):
        """
        Uses the LLM to parse the output and identify the files to be uploaded to Dataform.
        Constructs a JSON object with file paths and contents.
        """

        prompt = f"""
        You generated the following Dataform code:

        ```
        {llm_output}
        ```

        Now, your task is to parse this code and extract the individual files. 

        Structure your response as a JSON object.
        Example output:
        {{
        "files": [
            {{
            "path": "path/to/file1.sqlx",
            "content": "SQLX code for file1"
            }},
            {{
            "path": "path/to/file2.sqlx",
            "content": "SQLX code for file2"
            }},
            ...
        ]
        }}

        Make sure to accurately capture the file paths and their corresponding code content.
        """

        response = self.model.generate_content(prompt)
        response_content = response.text.strip()
        print(f"Parsed LLM Output:{response_content}")
        try:
            files_json = json.loads(response_content
                .replace("```json\n", "")
                .replace("```", "")
                .replace("  \n", ""))
            return files_json  # Return the JSON object directly
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM output: {e}")
            # Handle the error, e.g., ask for clarification or retry
            return None

    @trace_calls
    def upload_and_compile_files(self, files, workspace_name):
        """
        Uploads and compiles the files in Dataform.
        """
        client = dataform_v1beta1.DataformClient()
        repository_path = client.repository_path(self.project_id, "us-central1", "agent")
        workspace_path = client.workspace_path(
            self.project_id,
            "us-central1",  # Replace with your Dataform region
            "agent",  # Replace with your Dataform repository name
            workspace_name,
        )

        for file in files:
            file_path = file.get("path")
            file_content = file.get("content")
            print(f"Uploading file: {file_path} ")
            try:
                # Upload the code file
                request = dataform_v1beta1.WriteFileRequest(
                    workspace=workspace_path,
                    path=file_path,
                    contents=file_content.encode("utf-8"),  # Encode content as bytes
                )
                client.write_file(request=request)
            except Exception as e:
                print(f"Error uploading file '{file_path}': {e}")
        
        # Attempt to fix compilation errors
        print(f"Compiling...")
        for _ in range(3):  # Iterate 3 times

                compilation_result = dataform_v1beta1.CompilationResult()
                compilation_result.git_commitish = "main"
                compilation_result.workspace = (
                        f"{workspace_path}"
                    )


                request = dataform_v1beta1.CreateCompilationResultRequest(
                    parent=repository_path,
                    compilation_result=compilation_result,
                )


                compilation_results = client.create_compilation_result(request=request)

                print("Compilation results:")
                print(compilation_results)
                if hasattr(compilation_results, 'compilation_errors') and compilation_results.compilation_errors:
                    print("Compilation errors found!")

                    # Ask LLM to fix the errors
                    try:
                        fixed_files = self.fix_compilation_errors(files, compilation_results)
                        if not fixed_files:
                            print("LLM couldn't fix the errors.")
                            break

                        # Update files with fixed code
                        for fixed_file in fixed_files:
                            file_path = fixed_file["path"]
                            file_content = fixed_file["content"]
                            print(f"Uploading fixed file: {file_path}")
                            request = dataform_v1beta1.WriteFileRequest(
                                workspace=workspace_path,
                                path=file_path,
                                contents=file_content.encode("utf-8"),
                            )
                            client.write_file(request=request)
                    except Exception as e:
                        print(f"Error compiling or fixing files: {e}")
                else:
                    print("Compilation Successfull")
                    # Convert CompilationResult to a dictionary
                    compilation_result_dict = {
                        "name": compilation_results.name,
                        "workspace": compilation_results.workspace,
                        "compilation_errors": [
                            {
                                "message": error.message,
                                "path": error.path,
                                "stack": error.stack,
                            }
                            for error in compilation_results.compilation_errors
                        ]
                        if hasattr(compilation_results, "compilation_errors")
                        else [],
                        # Add any other fields you need
                    }
                    return compilation_result_dict

        # If loop finishes without success, return None or raise an exception
        print("Compilation failed after multiple attempts.")
        return None

    @trace_calls
    def refine_llm_response(self, prompt, previous_response, error):
        """Refines the LLM prompt based on the previous response.
        """
        prompt = f"""
        For the following prompt:
        {prompt}

        LLM returned:
        {previous_response}

        Which caused the following error during parsing the output:
        {error}
        
        Refine and correct the LLM response.
        Only respond with the JSON formatted text, nothing else.
        """

        # Do not put unnecessary escape.
        response = self.model.generate_content(prompt)

        response_content = response.text.strip().replace("`json\n", "").replace("`", "")
        print(f"Parsed LLM")
        return response_content

    @trace_calls
    def fix_compilation_errors(self, files, errors):
        """
        Asks the LLM to fix the compilation errors.
        """

        print("Fixing the issues..")
        with open("dataform_examples.txt", "r") as f:
            troubleshooting_guide = f.read()

        prompt = f"""
        The Dataform code has the following compilation errors:

        {errors}

        Here are the files:

        {files}

        Please provide the corrected code in the same format as before, 
        ensuring that the errors are fixed. 
        
        Structure your response as a JSON object with the following format:

        {{
        "files": [
            {{
            "path": "path/to/file1.sqlx",
            "content": "SQLX code for file1"
            }},
            {{
            "path": "path/to/file2.sqlx",
            "content": "SQLX code for file2"
            }},
            ...
        ]
        }}

        Here are some of the examples for troubleshooting:

        {troubleshooting_guide}


        Only include the files that need to be fixed and any additional files needed.
        Only return the JSON formatted response, nothing else.
        """

        # Do not put unnecessary escape.
        response = self.model.generate_content(prompt)

        response_content = response.text.strip().replace("`json\n", "").replace("`", "")
        print(f"Parsed LLM Output:{response_content}")
        
        while True:
            try:
                # Directly parse the JSON response from the LLM
                fixed_files_json = json.loads(response_content)
                return fixed_files_json.get("files", [])
            except json.JSONDecodeError as e:
                print(f"Error parsing LLM output for fixes: {e}")
                print("Refinining JSON...")
                # Consider refining the LLM prompt or providing more context
                response_content = self.refine_llm_response(prompt, response_content, e)

    @trace_calls
    def handle_user_request(self, user_request):
        """
        Handles the user request using Vertex AI to extract details 
        and ask for missing information.

        Assumptions:
        - Relies on the LLM to accurately interpret the user request 
          and extract the relevant details.
        """
        prompt = f"""
        You are a data engineering assistant. Analyze the following 
        user request and extract the following information in JSON format: 

        {{
        "source_tables": [
            {{
            "dataset": "dataset_name",
            "table": "table_name"
            }},
            {{
            "dataset": "dataset_name",
            "table": "table_name"
            }}
        ],
        "target_table": "dataset_name.table_name",
        "transformations": {{
            "transformation_description": "transformation_details"
        }}
        }}

        Request: "{user_request}"

        Example:
        Request: "Create a table in dataset 'my_dataset' named 
        'joined_data' by joining 'table1' from 'dataset1' and 
        'table2'  on the 'id' column."
        JSON Output:
        {{
        "source_tables": [
            {{
            "dataset": "dataset1",
            "table": "table1"
            }},
            {{
            "dataset": "",
            "table": "table2"
            }}
        ],
        "target_table": "my_dataset.joined_data",
        "transformations": {{
            "join": "Join 'table1' and 'table2' on the 'id' column."
        }}
        }}

        Only return the JSON.
        """
        

        # Call the Vertex AI model
        response = self.model.generate_content(prompt)
        

        # Extract clean text content
        response_content = response.text.strip().replace("`json\n", "").replace("`", "")
        print(response_content) 

        # Check if the response requires follow-up (improved)
        if not ("source_tables" in response_content 
                and "target_table" in response_content 
                and "transformations" in response_content): 
            return {"follow_up": response_content} 

        # If the content starts with '{' and ends with '}', it's likely in 
        # JSON format
        if response_content.startswith("{") and response_content.endswith("}"):
            try:
                parsed_response = json.loads(response_content)

                # If dataset is missing, try to find the relevant dataset
                for table in parsed_response.get("source_tables", []):
                    if not table.get("dataset"):
                        dataset = self.find_relevant_dataset(table.get("table"), user_request)
                        if dataset:
                            table["dataset"] = dataset
                        else:
                            # If no relevant dataset is found, return follow-up
                            return {"follow_up": f"Which dataset contains the table '{table.get('table')}'?"}
                return parsed_response
            except json.JSONDecodeError:
                return {"error": "Unable to parse the response from LLM. "
                                "Please refine the input or try again."}
        else:
            # If raw text, just return as a message
            parsed_response = json.loads(
                response_content
                .replace("```json\n", "")
                .replace("```", "")
                .replace("  \n", "")
            )
            return parsed_response
        
    @trace_calls
    def generate_pipeline_code(self, source_tables, target_table, transformations):
        """
        Generates a multi-layer data pipeline using Dataform SQLX.
        """
        # Load Dataform examples from the text file
        with open("dataform_examples.txt", "r") as f:
            dataform_examples = f.read()

        prompt = f"""
        You are a data engineering assistant. 
        Generate a multi-layer data pipeline using Dataform SQLX with the following details:

        1. Source tables: {json.dumps(source_tables, indent=2)}
        2. Target table (destination): {target_table}
        3. Transformations: {json.dumps(transformations, indent=2)}

        Ensure the pipeline:
        - Processes data from the source tables with necessary transformations.
        - Outputs the data into the target table.
        - Is modular and follows best practices for maintainability.

        Here are some Dataform examples to help you generate the code:
        {dataform_examples}

        Provide SQLX code for each layer.
        """
        # print("Prompt being sent to LLM:\n", prompt)
        
        try:
            response = self.model.generate_content(prompt)
            pipeline_code = response.text.strip()
            return {"pipeline_code": pipeline_code}

        except Exception as e:
            error_message = f"Error generating pipeline code: {e}"
            print(error_message)
            return {"error": error_message}

    @trace_calls
    def ask_for_clarifications(self, user_request):
        """
        Asks follow-up clarification questions to refine the pipeline request based on the task.
        The goal is to understand what details are necessary to proceed.
        """
        follow_up_questions = []

        # Identify common data engineering needs based on the request
        if "join" in user_request.lower():
            follow_up_questions.append("What tables would you like to join? Please specify the join keys.")
            follow_up_questions.append("Are there any specific filtering conditions or constraints for the join?")

        if "aggregation" in user_request.lower():
            follow_up_questions.append("What aggregation functions would you like to apply (e.g., SUM, AVG)?")
            follow_up_questions.append("Which columns should be grouped in the aggregation?")

        if "transformation" in user_request.lower():
            follow_up_questions.append("What specific transformations would you like to perform on the data?")
            follow_up_questions.append("Should any new columns be added based on the transformations?")

        if "date" in user_request.lower() or "timestamp" in user_request.lower():
            follow_up_questions.append("Are there any specific date ranges or time periods you would like to focus on?")
            follow_up_questions.append("Should the data be aggregated by a particular time window (e.g., daily, weekly)?")

        if "filter" in user_request.lower():
            follow_up_questions.append("What filter conditions would you like to apply to the data?")
            follow_up_questions.append("Are there any columns you want to exclude or filter out from the dataset?")

        if "schema" in user_request.lower() or "structure" in user_request.lower():
            follow_up_questions.append("Do you have any specific requirements for the table schema or structure?")
            follow_up_questions.append("Would you like to include any foreign key relationships or indexes in the output table?")

        # General follow-up questions
        follow_up_questions.extend([
            "What is the desired output format (e.g., table, view, materialized view)?",
            "Are there any performance considerations or specific requirements for the pipeline?",
            "Where should the pipeline be deployed (e.g., Dataform, BigQuery)?",
            "How should the pipeline be scheduled (e.g., daily, hourly)?"
        ])

        return follow_up_questions

# Initialize tool implementations
tool_impls = CustomToolImplementations(project_id="samets-ai-playground")

# Define tool configurations
tools_config = {
    "query_information_schema": {
        "function": tool_impls.query_information_schema,
        "input_model": InformationSchemaInput,
        "output_model": InformationSchemaOutput,
    },
    "find_relevant_dataset": {
        "function": tool_impls.find_relevant_dataset,
        "input_model": FindRelevantDatasetInput,
        "output_model": FindRelevantDatasetOutput,
    },
    "parse_llm_output": {
        "function": tool_impls.parse_llm_output,
        "input_model": ParseLLMOutputInput,
        "output_model": ParseLLMOutputOutput,
    },
    "upload_and_compile_files": {
        "function": tool_impls.upload_and_compile_files,
        "input_model": UploadAndCompileFilesInput,
        "output_model": UploadAndCompileFilesOutput,
    },
    "fix_compilation_errors": {
        "function": tool_impls.fix_compilation_errors,
        "input_model": FixCompilationErrorsInput,
        "output_model": FixCompilationErrorsOutput,
    },
    "handle_user_request": {
        "function": tool_impls.handle_user_request,
        "input_model": HandleUserRequestInput,
        "output_model": HandleUserRequestOutput,
    },
    "generate_pipeline_code": {
        "function": tool_impls.generate_pipeline_code,
        "input_model": GeneratePipelineCodeInput,
        "output_model": GeneratePipelineCodeOutput,
    },
    "refine_llm_response": {
        "function": tool_impls.refine_llm_response,
        "input_model": RefineLLMResponseInput,
        "output_model": RefineLLMResponseOutput,
    },
    "ask_for_clarifications": {
        "function": tool_impls.ask_for_clarifications,
        "input_model": AskForClarificationsInput,
        "output_model": AskForClarificationsOutput,
    },
}

# Define Vertex AI model
MODEL_NAME = "gemini-2.0-flash-exp"
llm = ChatVertexAI(model_name=MODEL_NAME, temperature=0.0, max_output_tokens=8192)

# Create LangChain tools
VERTEX_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "query_information_schema",
            "description": "Queries BigQuery information_schema to retrieve datasets, tables, and columns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string",
                        "description": "The name of the dataset to query"
                    },
                    "table_name": {
                        "type": "string",
                        "description": "The name of the table to query"
                    }
                },
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find_relevant_dataset",
            "description": "Finds the relevant dataset containing the given table by querying INFORMATION_SCHEMA and analyzing the user request.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {
                        "type": "string",
                        "description": "The name of the table to find the relevant dataset for"
                    },
                    "user_request": {
                        "type": "string",
                        "description": "The original user request"
                    }
                },
                "required": ["table_name", "user_request"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "parse_llm_output",
            "description": "Uses the LLM to parse the output and identify the files to be uploaded to Dataform.",
            "parameters": {
                "type": "object",
                "properties": {
                    "llm_output": {
                        "type": "string",
                        "description": "The output from the LLM to be parsed"
                    }
                },
                "required": ["llm_output"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "upload_and_compile_files",
            "description": "Uploads and compiles the files in Dataform.",
            "parameters": {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "The path of the file"
                                },
                                "content": {
                                    "type": "string",
                                    "description": "The content of the file"
                                }
                            }
                        },
                        "description": "A list of files to upload and compile"
                    },
                    "workspace_name": {
                        "type": "string",
                        "description": "The name of the Dataform workspace"
                    }
                },
                "required": ["files", "workspace_name"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "fix_compilation_errors",
            "description": "Asks the LLM to fix the compilation errors.",
            "parameters": {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {
                                    "type": "string",
                                    "description": "The path of the file"
                                },
                                "content": {
                                    "type": "string",
                                    "description": "The content of the file"
                                }
                            }
                        },
                        "description": "The current files"
                    },
                    "errors": {
                        "type": "object",
                        "description": "The compilation errors"
                    }
                },
                "required": ["files", "errors"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "handle_user_request",
            "description": "Handles the user request using Vertex AI to extract details and ask for missing information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_request": {
                        "type": "string",
                        "description": "The user's request"
                    }
                },
                "required": ["user_request"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_pipeline_code",
            "description": "Generates a multi-layer data pipeline using Dataform SQLX.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source_tables": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "dataset": {
                                    "type": "string",
                                    "description": "The dataset of the source table"
                                },
                                "table": {
                                    "type": "string",
                                    "description": "The name of the source table"
                                }
                            }
                        },
                        "description": "The source tables"
                    },
                    "target_table": {
                        "type": "string",
                        "description": "The target table"
                    },
                    "transformations": {
                        "type": "object",
                        "description": "The transformations to apply"
                    }
                },
                # "required": ["source_tables", "target_table", "transformations"]
                "required": ["source_tables", "transformations"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "refine_llm_response",
            "description": "Refines the LLM prompt based on the previous response.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The original prompt"
                    },
                    "previous_response": {
                        "type": "string",
                        "description": "The previous LLM response"
                    },
                    "error": {
                        "type": "string",
                        "description": "The error encountered"
                    }
                },
                "required": ["prompt", "previous_response", "error"]
            },
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ask_for_clarifications",
            "description": "Asks follow-up clarification questions to refine the pipeline request based on the task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_request": {
                        "type": "string",
                        "description": "The user's request"
                    }
                },
                "required": ["user_request"]
            },
        }
    },
]

# Convert LangChain tools to Vertex AI tools
vertex_tools = []
for tool in VERTEX_TOOLS:
    function_declaration = tool["function"]
    vertex_tools.append(
        Tool(
            function_declarations=[
                FunctionDeclaration(
                    name=function_declaration["name"],
                    description=function_declaration["description"],
                    parameters=function_declaration["parameters"],
                )
            ]
        )
    )

# Function to create a runnable tool from configuration
def create_runnable_tool(tool_name, tool_config):
    tool_func = tool_config["function"]
    tool_input_model = tool_config["input_model"]
    tool_output_model = tool_config["output_model"]

    def _convert_to_langchain_tool_input(tool_input: dict) -> BaseModel:
        return tool_input_model(**tool_input)

    def _invoke_tool(tool_input: BaseModel) -> str:
        try:
            output = tool_func(**tool_input.dict())
            # Check if the output is a BaseModel and needs to be dumped into JSON
            if isinstance(output, BaseModel):
                return output.model_dump_json()
            # Check if the output is a dictionary
            elif isinstance(output, dict):
                # Directly return the output if it's a dictionary
                return json.dumps(output)
            else:
                # Handle other types of output
                return json.dumps({"result": output})
        except Exception as e:
            print(f"Error invoking tool: {e}")
            return json.dumps({"error": str(e)})


    convert_to_input_model = RunnableLambda(_convert_to_langchain_tool_input)
    invoke_tool_lambda = RunnableLambda(_invoke_tool)

    runnable_tool = convert_to_input_model | invoke_tool_lambda | StrOutputParser()
    runnable_tool = runnable_tool.with_config(
        run_name=tool_name,
        name=tool_name,
    )

    return runnable_tool

# Create an empty dictionary for tools_executor
tools_executor = {}

# Populate tools_executor
for tool_name, tool_config in tools_config.items():
    tools_executor[tool_name] = create_runnable_tool(tool_name, tool_config)

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
        return {"messages": [HumanMessage(content="I couldn't understand the response. Can you please clarify?")]}

    messages = state.messages

    if "follow_up" in parsed_result:
        messages.append(HumanMessage(content=parsed_result["follow_up"]))
        return {"messages": messages}

    # Extract and store information from parsed_request
    source_tables = parsed_result.get("source_tables", [])
    target_table = parsed_result.get("target_table", "")
    transformations = parsed_result.get("transformations", {})

    # Retrieve schema for source tables
    for source in source_tables:
        dataset_name = source.get("dataset")
        table_name = source.get("table")
        columns = tool_impls.query_information_schema(dataset_name=dataset_name, table_name=table_name)
        source["columns"] = columns

    messages.append(HumanMessage(content=f"Processing request: Source Tables: {source_tables}, Target Table: {target_table}, Transformations: {transformations}"))

    # Store in state
    state.source_tables = source_tables
    state.target_table = target_table
    state.transformations = transformations

    return {
        "messages": messages,
        "source_tables": source_tables,
        "target_table": target_table,
        "transformations": transformations,
    }

@trace_calls
def generate_code(state: AgentState) -> Dict:
    """
    Generates the data pipeline code based on extracted information.
    """
    print("Generating pipeline code...")
    source_tables = state.source_tables
    target_table = state.target_table
    transformations = state.transformations

    result = tools_executor["generate_pipeline_code"].invoke({
        "source_tables": source_tables,
        "target_table": target_table,
        "transformations": transformations,
    })

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

    return {
        "messages": state.messages,
        "files": state.files
    }

@trace_calls
def upload_files(state: AgentState) -> Dict:
    """
    Uploads and compiles the files in Dataform.
    """
    print("Uploading and compiling files...")
    files = state.files
    # Use a default workspace name or make it configurable
    workspace_name = "agent" 
    result_str = tools_executor["upload_and_compile_files"].invoke({"files": files, "workspace_name": workspace_name})
    print (f"Compilation Result: {result_str}")
    
    # Safely parse the result string as JSON
    try:
        result = json.loads(result_str)
    except json.JSONDecodeError:
        print(f"Error parsing JSON from result from upload_and_compile_files: {result_str}")
        state.messages.append(AIMessage(content=f"Error parsing the result from upload_and_compile_files: {result_str}"))
        return {"messages": state.messages, "error": "Failed to parse result from upload_and_compile_files"}


    # # Check if the result indicates an error
    # if "error" in result:
    #     state.messages.append(AIMessage(content=f"Error during upload and compile: {result['error']}"))
    #     return {"messages": state.messages, "error": result["error"]}
    
    
    # Convert compilation_results to a dictionary for storage in AgentState
    # compilation_results_dict = {
    #     "name": result.name,
    #     "workspace": result.workspace,
    #     # Add other relevant fields as needed
    # }
    state.last_compilation_results = result

    # Check if the result contains compilation results
    if "name" in result and "compilation_errors" in result:
        # Update last_compilation_results with the result
        state.last_compilation_results = result

        # Check for compilation errors
        if result["compilation_errors"] != []:
            error_messages = [error["message"] for error in result["compilation_errors"]]
            error_message = "\n".join(error_messages)
            state.messages.append(AIMessage(content=f"Error during compilation: {error_message}"))
            return {"messages": state.messages, "last_compilation_results": result, "error": error_message}
        else:
            state.messages.append(AIMessage(content="Files uploaded and compiled successfully."))
            return {"messages": state.messages, "last_compilation_results": result}
    else:
        # Handle unexpected result type
        error_message = "Unexpected result format from upload_and_compile_files"
        state.messages.append(AIMessage(content=error_message))
        return {"messages": state.messages, "last_compilation_results": result, "error": error_message}
    # Check for errors in compilation_results
    # if hasattr(result, 'compilation_errors') and result.compilation_errors:
    #     error_messages = [error.message for error in result.compilation_errors]
    #     error_message = "\n".join(error_messages)
    #     state.messages.append(AIMessage(content=f"Error during compilation: {error_message}"))
    #     return {"messages": state.messages, "compilation_results": compilation_results_dict, "error": error_message}
    # else:
    #     state.messages.append(AIMessage(content="Files uploaded and compiled successfully."))
    #     return {"messages": state.messages, "compilation_results": compilation_results_dict}

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
        state.messages.append(AIMessage(content="Compilation errors fixed. Re-uploading files."))
    except json.JSONDecodeError:
        state.messages.append(AIMessage(content="Error parsing LLM output for error fixes. Please revise."))

    return {"messages": state.messages, "files": state.files}

@trace_calls
def decide_to_continue(state: AgentState) -> str:
    """
    Decides whether to continue with code generation or ask the user for clarification.
    """

    print(f"Checking state...{state}")

    # Access attributes directly
    if any(isinstance(msg, HumanMessage) and "follow_up" in msg.content for msg in state.messages):
        # If there are follow-up questions, go to "ask_clarifications"
        print("Asking for clarifications...")
        return "ask_clarifications"

    if any(isinstance(msg, AIMessage) and "error" in msg.content.lower() for msg in state.messages):
        # If errors are present, go to "handle_errors"
        print("Handling errors...")
        return "handle_errors"

    if not state.files and state.pipeline_code:
        print("Going to parse output")
        return "parse_output"
        
    print(f"Decide to continue: Last compilation results: {state.last_compilation_results}")
    if not state.last_compilation_results  and state.files:
        # If no compilation results yet, go to "upload_files"
        print("Uploading files...")
        return "upload_files"
    
    # Check for compilation errors (updated condition)
    if state.last_compilation_results and  state.last_compilation_results.get("compilation_errors") != []:
        print("Fixing errors...")
        return "fix_errors"

    # Check if compilation is successful (updated condition)
    if state.last_compilation_results and state.last_compilation_results.get("compilation_errors") == []:
        print("Ending the process...")
        return "end"

    # if state.last_compilation_results is not None and any(isinstance(msg, AIMessage) and "error" in msg.content.lower() for msg in state.messages):
    #     # If compilation errors, go to "fix_errors"
    #     print("Fixing errors...")
    #     return "fix_errors"

    # if state.last_compilation_results is not None and not any(isinstance(msg, AIMessage) and "error" in msg.content.lower() for msg in state.messages):
    #     # If compilation is successful, end
    #     print("Ending the process...")
    #     return "end"
    
    print("Generating code...")
    return "generate_code"

@trace_calls
def ask_clarifications(state: AgentState) -> Dict:
    """
    Asks the user for clarifications based on the follow-up questions.
    """
    print("Asking for clarifications...")
    user_request = state.input 
    result = tools_executor["ask_for_clarifications"].invoke({"user_request": user_request})

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
    error_message = "Unknown error"
    for msg in reversed(state.messages):
        if isinstance(msg, AIMessage) and "error" in msg.content.lower():
            error_message = msg.content
            break
        elif isinstance(msg, HumanMessage) and "error" in msg.content.lower():
            error_message = msg.content
            break

    # Append a HumanMessage to prompt for the next action
    state.messages.append(HumanMessage(content=f"An error occurred: {error_message}. What should we do next?"))

    return {"messages": state.messages}

# Define the workflow graph
workflow = StateGraph(AgentState)

# Add nodes for each step in the process
workflow.add_node("handle_request", handle_request)
workflow.add_node("generate_code", generate_code)
workflow.add_node("parse_output", parse_output)
workflow.add_node("upload_files", upload_files)
workflow.add_node("fix_errors", fix_errors)
workflow.add_node("ask_clarifications", ask_clarifications)
workflow.add_node("handle_errors", handle_errors)

# Set the entry point
workflow.set_entry_point("handle_request")

# Define the conditional edges based on the output of `decide_to_continue`
workflow.add_conditional_edges("handle_request", decide_to_continue)
workflow.add_conditional_edges("generate_code", decide_to_continue)
workflow.add_conditional_edges("parse_output", decide_to_continue)
workflow.add_conditional_edges("upload_files", decide_to_continue)
workflow.add_conditional_edges("fix_errors", decide_to_continue)
workflow.add_conditional_edges("ask_clarifications", decide_to_continue)
workflow.add_conditional_edges("handle_errors", decide_to_continue)

# Add unconditional edge from fix_errors back to upload_files for re-upload
workflow.add_edge("fix_errors", "upload_files")

# Add an edge to the end
workflow.add_edge("upload_files", END)


# Compile the graph
graph = workflow.compile()

# Interactive Mode Function
@trace_calls

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
                for output in graph.stream(state, {"recursion_limit": 150, "output_keys": ["next","messages","files","last_compilation_results","pipeline_code","source_tables","target_table","transformations"]}):
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
                                        if not any(m.content == msg.content for m in state.messages):
                                            if isinstance(msg, HumanMessage):
                                                print(f"Human: {msg.content}")
                                            elif isinstance(msg, AIMessage):
                                                print(f"AI: {msg.content}")
                                            elif isinstance(msg, SystemMessage):
                                                print(f"System: {msg.content}")
                                            else:
                                                print(f"Unknown message type: {msg}")
                                # Update state based on intermediate output
                                state.messages.extend(m for m in value.get("messages", []) if not any(sm.content == m.content for sm in state.messages))
                                state.files = value.get("files", state.files)
                                state.last_compilation_results = value.get("last_compilation_results", state.last_compilation_results)
                                state.pipeline_code = value.get("pipeline_code", state.pipeline_code)
                                state.source_tables = value.get("source_tables", state.source_tables)
                                state.target_table = value.get("target_table", state.target_table)
                                state.transformations = value.get("transformations", state.transformations)
                                state.next = value.get("next", state.next)

                # Check if the graph execution has reached the end
                if END in state.next:
                    print("Pipeline execution completed. Starting over...")
                    break

                # Get user input for the next step based on state.next or other relevant conditions
                if state.next == "ask_clarifications":
                    user_input = input("Your response (or type 'exit' to quit): ")
                    if user_input.lower() in ["exit"]:
                        print("Goodbye! See you next time!")
                        return  # Exit the entire function
                    state.input = user_input
                    state.messages.append(HumanMessage(content=user_input))

            except KeyError as e:
                if "branch:upload_files:wrapper:end" in str(e):
                    print("Encountered a known LangGraph issue. Restarting the process...")
                    break  # Break the inner loop to restart
                else:
                    raise  # Re-raise other KeyErrors

# def interactive_mode():
#     """
#     Runs the agent in interactive mode, allowing for user input at each step.
#     """
#     print("Welcome to the Dynamic Data Pipeline Agent!")
#     print("Describe your pipeline requirements, and I’ll help you step by step.\n")

#     # Initialize the state
#     state = AgentState(input="", messages=[])

#     while True:
#         if not state.messages:
#             # Get the initial user request
#             user_request = input("Your request: ")
#             if user_request.lower() in ["exit", "quit"]:
#                 print("Goodbye! See you next time!")
#                 break
#             state.input = user_request
#             state.messages.append(HumanMessage(content=user_request))

#         # Invoke the graph with the current state
#         result = graph.invoke(state)

#         # Update the state with the result
#         state = AgentState(
#             input=state.input,
#             messages=result["messages"],
#             files=result.get("files"),
#             last_compilation_results=result.get("last_compilation_results"),
#             pipeline_code=result.get("pipeline_code"),
#             source_tables=result.get("source_tables"),
#             target_table=result.get("target_table"),
#             transformations=result.get("transformations"),
#             next=result.get("next")
#         )

#         # Print the messages
#         for message in result["messages"]:
#             if isinstance(message, HumanMessage):
#                 print(f"Human: {message.content}")
#             elif isinstance(message, AIMessage):
#                 print(f"AI: {message.content}")
#             elif isinstance(message, SystemMessage):
#                 print(f"System: {message.content}")
#             else:
#                 print(f"Unknown message type: {message}")

#         # Check if the graph execution has reached the end
#         if result.get("next") == END:
#             break

#         # Get user input for the next step
#         if result.get("next") == "ask_clarifications":
#              user_input = input("Your response (or type 'exit' to quit): ")
#              if user_input.lower() in ["exit"]:
#                 print("Goodbye! See you next time!")
#                 break
#              state.input = user_input
#              state.messages.append(HumanMessage(content=user_input))

#         # other conditions to be handled if necessary

# Run the interactive mode
interactive_mode()