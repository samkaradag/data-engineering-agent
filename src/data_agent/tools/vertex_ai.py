from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import json
import re
import os

import vertexai
from vertexai.generative_models import GenerativeModel, Tool, FunctionDeclaration

from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from pydantic import BaseModel, Field, ValidationError
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

from langchain_google_vertexai import ChatVertexAI
from utils.tracers import trace_calls
from utils.prompt_loader import load_prompt # Import the function

class VertexAITools:
    @trace_calls
    def __init__(self, project_id, location="us-central1", model_name="gemini-2.0-flash-exp"):
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)

        self.model = GenerativeModel(model_name)
        self.project_id = project_id

    @trace_calls
    def parse_llm_output(self, llm_output):
        """
        Uses the LLM to parse the output and identify the files to be uploaded to Dataform.
        Constructs a JSON object with file paths and contents.
        """

        # prompt = f"""
        # You generated the following Dataform code:

        # ```
        # {llm_output}
        # ```

        # Now, your task is to parse this code and extract the individual files. 

        # Structure your response as a JSON object.
        # Example output:
        # {{
        # "files": [
        #     {{
        #     "path": "path/to/file1.sqlx",
        #     "content": "SQLX code for file1"
        #     }},
        #     {{
        #     "path": "path/to/file2.sqlx",
        #     "content": "SQLX code for file2"
        #     }},
        #     ...
        # ]
        # }}

        # Make sure to accurately capture the file paths and their corresponding code content.
        # """

        # Load the prompt template
        prompt_template = load_prompt("parse_llm_output")

        # Fill the prompt template
        prompt = prompt_template.format(llm_output=llm_output)

        response = self.model.generate_content(prompt)
        response_content = response.text.strip()
        print(f"Parsed LLM Output:{response_content}")
        try:
            files_json = json.loads(
                response_content.replace("`json\n", "").replace("`", "").replace(" \n", "")
            )
            return files_json  # Return the JSON object directly
        except json.JSONDecodeError as e:
            print(f"Error parsing LLM output: {e}")
            # Handle the error, e.g., ask for clarification or retry
            return None

    @trace_calls
    def refine_llm_response(self, prompt, previous_response, error):
        """Refines the LLM prompt based on the previous response.
        """
        # prompt = f"""
        # For the following prompt:
        # {prompt}

        # LLM returned:
        # {previous_response}

        # Which caused the following error during parsing the output:
        # {error}
        
        # Refine and correct the LLM response.
        # Only respond with the JSON formatted text, nothing else.
        # """
        
        # Load the prompt template from YAML
        prompt_template = load_prompt("refine_llm_response")

        # Fill the placeholders in the template
        prompt = prompt_template.format(
            prompt=prompt,
            previous_response=previous_response,
            error=error
        )

        # Do not put unnecessary escape.
        response = self.model.generate_content(prompt)

        response_content = response.text.strip().replace("`json\n", "").replace("`", "")
        print(f"Parsed LLM")
        return response_content

    @trace_calls
    def handle_user_request(self, user_request):
        """
        Handles the user request using Vertex AI to extract details 
        and ask for missing information.

        Assumptions:
        - Relies on the LLM to accurately interpret the user request 
        and extract the relevant details.
        """
        from tools.bigquery import BigQueryTools
        bigquery_tools = BigQueryTools(project_id=self.project_id)
        # Load the prompt template
        prompt_template = load_prompt("handle_user_request")

        # Fill the prompt template
        prompt = prompt_template.format(user_request=user_request)
        # prompt = f"""
        # You are a data engineering assistant. Analyze the following 
        # user request and extract the following information in JSON format: 

        # {{
        # "source_tables": [
        #     {{
        #     "dataset": "dataset_name",
        #     "table": "table_name"
        #     }}
        # ],
        # "transformations": {{
        #     "transformation_description": "transformation_details"
        # }},
        # "target_tables": [
        #     {{
        #     "dataset": "dataset_name",
        #     "table": "table_name",
        #     "type": "dimension | fact",
        #     "description": "short description"
        #     }}
        # ],
        # "intermediate_tables": [
        #     {{
        #     "dataset": "dataset_name",
        #     "table": "table_name",
        #     "description": "description of intermediate table"
        #     }}
        # ],
        # "data_quality_checks": {{
        #     "check_description": "description of the data quality check"
        # }}
        # }}

        # Request: "{user_request}"

        # Example:
        # Request: "Create a star schema from the table 'sales_data'. I need dimensions for date, product, and customer, and a fact table for sales."
        # JSON Output:
        # {{
        # "source_tables": [
        #     {{
        #     "dataset": "",
        #     "table": "sales_data"
        #     }}
        # ],
        # "transformations": {{
        #     "star_schema": "Create a star schema with dimensions and fact table"
        # }},
        # "target_tables": [
        #     {{
        #     "dataset": "",
        #     "table": "dim_date",
        #     "type": "dimension",
        #     "description": "Date dimension table"
        #     }},
        #     {{
        #     "dataset": "",
        #     "table": "dim_product",
        #     "type": "dimension",
        #     "description": "Product dimension table"
        #     }},
        #     {{
        #     "dataset": "",
        #     "table": "dim_customer",
        #     "type": "dimension",
        #     "description": "Customer dimension table"
        #     }},
        #     {{
        #     "dataset": "",
        #     "table": "fact_sales",
        #     "type": "fact",
        #     "description": "Sales fact table"
        #     }}
        # ],
        # "intermediate_tables": [],
        # "data_quality_checks": {{}}
        # }}
        
        # Only return the JSON.
        # """
        # If dataset of the table is not specified for the source table return dataset empty otherwise provide the dataset.
        
        # Call the Vertex AI model
        # print (f"Calling Vertex AI model {prompt}")
        response = self.model.generate_content(prompt)


        # Extract clean text content
        response_content = response.text.strip().replace("`json\n", "").replace("`", "")
        print(response_content)

        # Check if the response requires follow-up (improved)
        if not (
            "source_tables" in response_content
            and "target_tables" in response_content
            and "transformations" in response_content
        ):
            return {"follow_up": response_content}

        # If the content starts with '{' and ends with '}', it's likely in
        # JSON format
        if response_content.startswith("{") and response_content.endswith("}"):
            try:
                parsed_response = json.loads(response_content)
                source_tables = parsed_response.get("source_tables", [])
                for table in source_tables:
                    if not table.get("dataset"):
                        # Try to find the dataset
                        dataset = bigquery_tools.find_relevant_dataset(table.get("table"), user_request)
                        if dataset:
                            table["dataset"] = dataset
                        else:
                            # If no relevant dataset is found, return follow-up
                            return {
                                "follow_up": f"Which dataset contains the table '{table.get('table')}'?"
                            }
                    else:
                        # If dataset is provided, validate it
                        try:
                            bigquery_tools.bigquery_client.get_dataset(table.get("dataset"))
                        except Exception as e:
                            print(
                                f"Error: Provided dataset '{table.get('dataset')}' not found: {e}"
                            )
                            return {
                                "follow_up": f"Could not find dataset '{table.get('dataset')}'. Please confirm the source dataset."
                            }
                # If dataset is missing, try to find the relevant dataset
                for table in parsed_response.get("source_tables", []):
                    if not table.get("dataset"):
                        dataset = bigquery_tools.find_relevant_dataset(table.get("table"), user_request)
                        if dataset:
                            table["dataset"] = dataset
                        else:
                            # If no relevant dataset is found, return follow-up
                            return {
                                "follow_up": f"Which dataset contains the table '{table.get('table')}'?"
                            }
                # Check if target_tables is present in the parsed response and add it to the output
                if "target_tables" in parsed_response:
                    return {
                        "parsed_request": parsed_response,
                        "target_tables": parsed_response["target_tables"],
                    }
                else:
                    return parsed_response
            except json.JSONDecodeError:
                return {
                    "error": "Unable to parse the response from LLM. "
                    "Please refine the input or try again."
                }
        else:
            # If raw text, just return as a message
            parsed_response = json.loads(
                response_content.replace("`json\n", "").replace("`", "").replace(" \n", "")
            )
            return parsed_response
    # @trace_calls
    # def handle_user_request(self, user_request):
    #     """
    #     Handles the user request using Vertex AI to extract details 
    #     and ask for missing information.

    #     Assumptions:
    #     - Relies on the LLM to accurately interpret the user request 
    #       and extract the relevant details.
    #     """
    #     prompt = f"""
    #     You are a data engineering assistant. Analyze the following 
    #     user request and extract the following information in JSON format: 

    #     {{
    #     "source_tables": [
    #         {{
    #         "dataset": "dataset_name",
    #         "table": "table_name"
    #         }}
    #     ],
    #     "transformations": {{
    #         "transformation_description": "transformation_details"
    #     }},
    #     "target_tables": [
    #         {{
    #         "dataset": "dataset_name",
    #         "table": "table_name",
    #         "type": "dimension | fact",
    #         "description": "short description"
    #         }}
    #     ],
    #     "intermediate_tables": [
    #         {{
    #         "dataset": "dataset_name",
    #         "table": "table_name",
    #         "description": "description of intermediate table"
    #         }}
    #     ],
    #     "data_quality_checks": {{
    #         "check_description": "description of the data quality check"
    #     }}
    #     }}

    #     Request: "{user_request}"

    #     Example:
    #     Request: "Create a star schema from the table 'sales_data'. I need dimensions for date, product, and customer, and a fact table for sales."
    #     JSON Output:
    #     {{
    #     "source_tables": [
    #         {{
    #         "dataset": "",
    #         "table": "sales_data"
    #         }}
    #     ],
    #     "transformations": {{
    #         "star_schema": "Create a star schema with dimensions and fact table"
    #     }},
    #     "target_tables": [
    #         {{
    #         "dataset": "",
    #         "table": "dim_date",
    #         "type": "dimension",
    #         "description": "Date dimension table"
    #         }},
    #         {{
    #         "dataset": "",
    #         "table": "dim_product",
    #         "type": "dimension",
    #         "description": "Product dimension table"
    #         }},
    #         {{
    #         "dataset": "",
    #         "table": "dim_customer",
    #         "type": "dimension",
    #         "description": "Customer dimension table"
    #         }},
    #         {{
    #         "dataset": "",
    #         "table": "fact_sales",
    #         "type": "fact",
    #         "description": "Sales fact table"
    #         }}
    #     ],
    #     "intermediate_tables": [],
    #     "data_quality_checks": {{}}
    #     }}

    #     Only return the JSON.
    #     """

    #     # Call the Vertex AI model
    #     # print (f"Calling Vertex AI model {prompt}")
    #     response = self.model.generate_content(prompt)

    #     # Extract clean text content
    #     response_content = response.text.strip().replace("`json\n", "").replace("`", "")
    #     print(response_content)

    #     # Check if the response requires follow-up (improved)
    #     if not (
    #         "source_tables" in response_content
    #         and "target_tables" in response_content
    #         and "transformations" in response_content
    #     ):
    #         return {"follow_up": response_content}

    #     # If the content starts with '{' and ends with '}', it's likely in
    #     # JSON format
    #     if response_content.startswith("{") and response_content.endswith("}"):
    #         try:
    #             parsed_response = json.loads(response_content)
    #             source_tables = parsed_response.get("source_tables", [])
    #             for table in source_tables:
    #                 if not table.get("dataset"):
    #                     # Try to find the dataset
    #                     dataset = self.find_relevant_dataset(table.get("table"), user_request)
    #                     if dataset:
    #                         table["dataset"] = dataset
    #                     else:
    #                         # If no relevant dataset is found, return follow-up
    #                         return {
    #                             "follow_up": f"Which dataset contains the table '{table.get('table')}'?"
    #                         }
    #                 else:
    #                     # If dataset is provided, validate it
    #                     try:
    #                         self.bigquery_client.get_dataset(table.get("dataset"))
    #                     except Exception as e:
    #                         print(
    #                             f"Error: Provided dataset '{table.get('dataset')}' not found: {e}"
    #                         )
    #                         return {
    #                             "follow_up": f"Could not find dataset '{table.get('dataset')}'. Please confirm the source dataset."
    #                         }
    #             # If dataset is missing, try to find the relevant dataset
    #             for table in parsed_response.get("source_tables", []):
    #                 if not table.get("dataset"):
    #                     dataset = self.find_relevant_dataset(table.get("table"), user_request)
    #                     if dataset:
    #                         table["dataset"] = dataset
    #                     else:
    #                         # If no relevant dataset is found, return follow-up
    #                         return {
    #                             "follow_up": f"Which dataset contains the table '{table.get('table')}'?"
    #                         }
    #             # Check if target_tables is present in the parsed response and add it to the output
    #             if "target_tables" in parsed_response:
    #                 return {
    #                     "parsed_request": parsed_response,
    #                     "target_tables": parsed_response["target_tables"],
    #                 }
    #             else:
    #                 return parsed_response
    #         except json.JSONDecodeError:
    #             return {
    #                 "error": "Unable to parse the response from LLM. "
    #                 "Please refine the input or try again."
    #             }
    #     else:
    #         # If raw text, just return as a message
    #         parsed_response = json.loads(
    #             response_content.replace("`json\n", "").replace("`", "").replace("  \n", "")
    #         )
    #         return parsed_response

    @trace_calls
    def generate_pipeline_code(
        self, source_tables, target_tables, transformations, intermediate_tables, data_quality_checks
    ):
        """
        Generates a multi-layer data pipeline using Dataform SQLX.
        Now it also handles intermediate tables and data quality checks.
        """
        # Load Dataform examples from the text file
        with open("prompts/guides/dataform_examples.txt", "r") as f:
            dataform_examples = f.read()
        
        prompt_template = load_prompt("generate_pipeline_code")

        # Fill the prompt template
        prompt = prompt_template.format(
            source_tables=json.dumps(source_tables, indent=2),
            target_tables=json.dumps(target_tables, indent=2),
            transformations=json.dumps(transformations, indent=2),
            intermediate_tables=json.dumps(intermediate_tables, indent=2),
            data_quality_checks=json.dumps(data_quality_checks, indent=2),
            dataform_examples=dataform_examples
        )
        # print("Prompt being sent to LLM:\n", prompt)

        # prompt = f"""
        # You are a data engineering assistant. 
        # Generate a multi-layer data pipeline using Dataform SQLX with the following details:

        # 1. Source tables: {json.dumps(source_tables, indent=2)}
        # 2. Target table (destination): {json.dumps(target_tables, indent=2)}
        # 3. Transformations: {json.dumps(transformations, indent=2)}
        # 4. Intermediate tables (if any): {json.dumps(intermediate_tables, indent=2)}
        # 5. Data quality checks (if any): {json.dumps(data_quality_checks, indent=2)}

        # Ensure the pipeline:
        # - Processes data from the source tables with necessary transformations.
        # - Uses intermediate tables if specified
        # - Applies data quality checks if specified
        # - Outputs the data into the target table.
        # - Is modular and follows best practices for maintainability.

        # Here are some Dataform examples to help you generate the code:
        # {dataform_examples}

        # Provide SQLX code for each layer.
        # """
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
    def ask_for_clarifications(self, user_request, missing_information: Optional[List[str]] = None):
        """
        Asks follow-up clarification questions to refine the pipeline request based on the task.
        The goal is to understand what details are necessary to proceed.

        Handles missing_information by prioritizing questions about those specific details.
        """
        follow_up_questions = []

        # Prioritize questions based on missing information
        if missing_information:
            if "source_tables" in missing_information:
                follow_up_questions.append("Which source tables should be used?")
            if "target_tables" in missing_information:
                follow_up_questions.append(
                    "What are the desired target tables for dimensions and facts? Please provide names, types (dimension or fact), and a brief description for each."
                )
            if "transformations" in missing_information:
                follow_up_questions.append(
                    "What specific transformations should be applied to the data?"
                )
            if "intermediate_tables" in missing_information:
                follow_up_questions.append(
                    "Are there any intermediate tables needed? If so, what are their names, datasets, and purposes?"
                )
            if "data_quality_checks" in missing_information:
                follow_up_questions.append(
                    "What data quality checks should be applied? (e.g., uniqueness, non-nullity, specific ranges)"
                )

        # If no specific missing information is provided, ask general questions
        if not missing_information:
            # Identify common data engineering needs based on the request
            if "join" in user_request.lower():
                follow_up_questions.append(
                    "What tables would you like to join? Please specify the join keys."
                )
                follow_up_questions.append(
                    "Are there any specific filtering conditions or constraints for the join?"
                )

            if "aggregation" in user_request.lower():
                follow_up_questions.append(
                    "What aggregation functions would you like to apply (e.g., SUM, AVG)?"
                )
                follow_up_questions.append("Which columns should be grouped in the aggregation?")

            if "transformation" in user_request.lower():
                follow_up_questions.append(
                    "What specific transformations would you like to perform on the data?"
                )
                follow_up_questions.append(
                    "Should any new columns be added based on the transformations?"
                )

            if "date" in user_request.lower() or "timestamp" in user_request.lower():
                follow_up_questions.append(
                    "Are there any specific date ranges or time periods you would like to focus on?"
                )
                follow_up_questions.append(
                    "Should the data be aggregated by a particular time window (e.g., daily, weekly)?"
                )

            if "filter" in user_request.lower():
                follow_up_questions.append("What filter conditions would you like to apply to the data?")
                follow_up_questions.append(
                    "Are there any columns you want to exclude or filter out from the dataset?"
                )

            if "schema" in user_request.lower() or "structure" in user_request.lower():
                follow_up_questions.append(
                    "Do you have any specific requirements for the table schema or structure?"
                )
                follow_up_questions.append(
                    "Would you like to include any foreign key relationships or indexes in the output table?"
                )

        # General follow-up questions (always ask these)
        follow_up_questions.extend(
            [
                "What is the desired output format (e.g., table, view, materialized view)?",
                "Are there any performance considerations or specific requirements for the pipeline?",
                "Where should the pipeline be deployed (e.g., Dataform, BigQuery)?",
                "How should the pipeline be scheduled (e.g., daily, hourly)?",
            ]
        )

        return follow_up_questions
    
     # Function to create a runnable tool from configuration
    def create_runnable_tool(self, tool_name, tool_config):
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