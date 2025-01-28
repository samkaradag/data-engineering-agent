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
    def fix_json_parse_errors(self, prompt, previous_response, error):
        """Refines the LLM prompt based on the previous response.
        """

        # Load the prompt template from YAML
        prompt_template = load_prompt("fix_json_parse_errors")

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
    def structure_transformation_request(self, user_request):
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
        prompt_template = load_prompt("structure_transformation_request")

        # Fill the prompt template
        prompt = prompt_template.format(user_request=user_request)

        # Call the Vertex AI model
        # print (f"Calling Vertex AI model {prompt}")
        response = self.model.generate_content(prompt)


        # Extract clean text content
        response_content = response.text.strip().replace("`json\n", "").replace("`", "")
        print(response_content)

        parsed_response = json.loads(
                response_content.replace("`json\n", "").replace("`", "").replace(" \n", "")
            )
        return parsed_response
    
    # @trace_calls
    # def ask_for_clarifications(self, user_request, missing_information: Optional[List[str]] = None):
    #     """
    #     Asks follow-up clarification questions to refine the pipeline request based on the task.
    #     The goal is to understand what details are necessary to proceed.

    #     Handles missing_information by prioritizing questions about those specific details.
    #     """
    #     follow_up_questions = []

    #     # Prioritize questions based on missing information
    #     if missing_information:
    #         if "source_tables" in missing_information:
    #             follow_up_questions.append("Which source tables should be used?")
    #         if "target_tables" in missing_information:
    #             follow_up_questions.append(
    #                 "What are the desired target tables for dimensions and facts? Please provide names, types (dimension or fact), and a brief description for each."
    #             )
    #         if "transformations" in missing_information:
    #             follow_up_questions.append(
    #                 "What specific transformations should be applied to the data?"
    #             )
    #         if "intermediate_tables" in missing_information:
    #             follow_up_questions.append(
    #                 "Are there any intermediate tables needed? If so, what are their names, datasets, and purposes?"
    #             )
    #         if "data_quality_checks" in missing_information:
    #             follow_up_questions.append(
    #                 "What data quality checks should be applied? (e.g., uniqueness, non-nullity, specific ranges)"
    #             )

    #     # If no specific missing information is provided, ask general questions
    #     if not missing_information:
    #         # Identify common data engineering needs based on the request
    #         if "join" in user_request.lower():
    #             follow_up_questions.append(
    #                 "What tables would you like to join? Please specify the join keys."
    #             )
    #             follow_up_questions.append(
    #                 "Are there any specific filtering conditions or constraints for the join?"
    #             )

    #         if "aggregation" in user_request.lower():
    #             follow_up_questions.append(
    #                 "What aggregation functions would you like to apply (e.g., SUM, AVG)?"
    #             )
    #             follow_up_questions.append("Which columns should be grouped in the aggregation?")

    #         if "transformation" in user_request.lower():
    #             follow_up_questions.append(
    #                 "What specific transformations would you like to perform on the data?"
    #             )
    #             follow_up_questions.append(
    #                 "Should any new columns be added based on the transformations?"
    #             )

    #         if "date" in user_request.lower() or "timestamp" in user_request.lower():
    #             follow_up_questions.append(
    #                 "Are there any specific date ranges or time periods you would like to focus on?"
    #             )
    #             follow_up_questions.append(
    #                 "Should the data be aggregated by a particular time window (e.g., daily, weekly)?"
    #             )

    #         if "filter" in user_request.lower():
    #             follow_up_questions.append("What filter conditions would you like to apply to the data?")
    #             follow_up_questions.append(
    #                 "Are there any columns you want to exclude or filter out from the dataset?"
    #             )

    #         if "schema" in user_request.lower() or "structure" in user_request.lower():
    #             follow_up_questions.append(
    #                 "Do you have any specific requirements for the table schema or structure?"
    #             )
    #             follow_up_questions.append(
    #                 "Would you like to include any foreign key relationships or indexes in the output table?"
    #             )

    #     # General follow-up questions (always ask these)
    #     follow_up_questions.extend(
    #         [
    #             "What is the desired output format (e.g., table, view, materialized view)?",
    #             "Are there any performance considerations or specific requirements for the pipeline?",
    #             "Where should the pipeline be deployed (e.g., Dataform, BigQuery)?",
    #             "How should the pipeline be scheduled (e.g., daily, hourly)?",
    #         ]
    #     )

    #     return follow_up_questions
    
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