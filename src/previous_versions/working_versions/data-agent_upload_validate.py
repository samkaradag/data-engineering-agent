import vertexai
from vertexai.generative_models import GenerativeModel
from google.cloud import bigquery
import json
import re
from google.cloud import dataform_v1beta1
import os



# gemini-1.5-flash-002
# gemini-1.5-pro-002
# gemini-2.0-flash-exp
# gemini-pro-experimental
class DynamicPipelineAgent:
    def __init__(self, project_id, location="us-central1", model_name="gemini-2.0-flash-exp"):
    # def __init__(self, project_id, location="us-central1", model_name="gemini-pro-experimental"):
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)

        self.model = GenerativeModel(model_name)
        self.bigquery_client = bigquery.Client(project=project_id)
        self.project_id = project_id

    
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
        for _ in range(7):  # Iterate 3 times

                # compilation_results = client.list_compilation_results(
                #     parent=repository_path,
                # )
                # Initialize request argument(s)
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
                # if True:
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
                    break

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

        response_content = response.text.strip().replace("```json\n", "").replace("```", "")
        print(f"Parsed LLM Output:{response_content}")
        
        return response_content

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

        response_content = response.text.strip().replace("```json\n", "").replace("```", "")
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
        # TODO: Add more sophisticated logic to analyze the user request
        # and identify the most relevant dataset based on keywords, context, etc.
        return None

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
        response_content = response.text.strip().replace("```json\n", "").replace("```", "")
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


    def generate_pipeline_code(self, source_tables, columns, target_table, transformations):
        """
        Generates a multi-layer data pipeline using Dataform SQLX.

        Assumptions:
        - Assumes the user wants to use Dataform SQLX for the pipeline.
        - Assumes the user wants a multi-layer pipeline.
        - Relies on the LLM to generate valid and efficient SQLX code.
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
        response = self.model.generate_content(prompt)
        return response.text.strip()

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

    def interactive_mode(self):
        """
        Runs an interactive mode to handle user requests iteratively.
        """
        print("Welcome to the Dynamic Data Pipeline Agent!")
        print("Describe your pipeline requirements, and I’ll help you step by step.\n")

        while True:
            user_request = input("Your request: ")

            if user_request.lower() in ["exit", "quit"]:
                print("Goodbye! See you next time!")
                break

            # Parse the request with LLM and handle any follow-up questions
            parsed_request = self.handle_user_request(user_request)
            print(f"Parsed request:{parsed_request}")
            if "follow_up" in parsed_request:
                follow_up_questions = self.ask_for_clarifications(user_request)
                print("\nFollow-up Questions:")
                for question in follow_up_questions:
                    print(f"- {question}")
                continue

            if "error" in parsed_request:
                print(parsed_request["error"])
                continue

            # Extract details from the parsed request
            source_tables = parsed_request.get("source_tables", [])
            target_table = parsed_request.get("target_table", "")
            transformations = parsed_request.get("transformations", {})
            print(f"I understand that - source: {source_tables}, target: {target_table}, transformations:{transformations}")

            # Validate source tables and retrieve their schema
            for source in source_tables:
                dataset_name = source.get("dataset")
                table_name = source.get("table")
                if not dataset_name:
                    datasets = self.query_information_schema()
                    if datasets:
                        print(f"Available datasets: {datasets}")
                        source["dataset"] = datasets[0]
                        dataset_name  = datasets[0]
                        
                columns = self.query_information_schema(dataset_name=dataset_name, table_name = table_name)
                source["columns"] = columns
               

            # Generate pipeline code
            print(f"\nRetrieved the following from INFORMATION_SCHEMA:\nSource tables:{source_tables}\nTarget table:{target_table}\nGenerating dataform pipeline code... ")
            pipeline_code = self.generate_pipeline_code(source_tables, columns, target_table, transformations)
            print("\nHere’s your pipeline code:\n")
            print(pipeline_code)

            # Allow user to upload to dataform
            refine = input("\nWould you like to upload and validate this pipeline? (yes/no): ")
            if refine.lower() in ["yes", "y"]:
                # Parse the LLM output
                files_json = self.parse_llm_output(pipeline_code)
                print(files_json)
                 # Upload and compile the files
                self.upload_and_compile_files(files_json["files"], "agent")

            elif refine.lower() in ["no", "n"]:
                print("Pipeline creation completed! Exiting interactive mode.")
                break


            # Allow user to refine the pipeline -->todo
            refine = input("\nWould you like to refine this pipeline? (yes/no): ")
            if refine.lower() in ["no", "n"]:
                print("Pipeline creation completed! Exiting interactive mode.")
                break


# Initialize the agent
agent = DynamicPipelineAgent(project_id="samets-ai-playground")

# Start interactive mode
agent.interactive_mode()