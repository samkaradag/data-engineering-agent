import vertexai
from vertexai.generative_models import GenerativeModel
from google.cloud import bigquery
import json
import re


class DynamicPipelineAgent:
    def __init__(self, project_id, location="us-central1", model_name="gemini-pro-experimental"):
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)
        self.model = GenerativeModel(model_name)
        self.bigquery_client = bigquery.Client(project=project_id)
        self.project_id = project_id
        self.conversation_history = []  # Initialize conversation history

    def query_information_schema(self, dataset_name=None, table_name=None):
        """
        Queries BigQuery information_schema to validate datasets, tables, and columns.

        Assumptions:
        - Assumes the BigQuery project is in the 'region-europe-west4' location. 
          This should be made configurable or automatically detected.
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
    
        # query = f"""
        # SELECT table_schema,table_name,column_name, data_type
        # FROM `{self.project_id}.{dataset_name}.INFORMATION_SCHEMA.COLUMNS`
        # WHERE table_name = '{table_name}'
        # """
        # return [{"dataset": row.table_schema, "table_name": row.table_name,"column_name": row.column_name, "type": row.data_type} for row in self.bigquery_client.query(query)]
    
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
        - Assumes the user request is in English.
        - Assumes the user request is clear and unambiguous.
        - Relies on the LLM to accurately interpret the user request 
          and extract the relevant details.
        """
        # Add user request to history
        self.conversation_history.append(f"User: {user_request}")

        # Construct prompt with history
        prompt = f"""Conversation history: 
        {self.conversation_history}

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
        'table2' from 'dataset2' on the 'id' column."
        JSON Output:
        {{
        "source_tables": [
            {{
            "dataset": "dataset1",
            "table": "table1"
            }},
            {{
            "dataset": "dataset2",
            "table": "table2"
            }}
        ],
        "target_table": "my_dataset.joined_data",
        "transformations": {{
            "join": "Join 'table1' and 'table2' on the 'id' column."
        }}
        }}
        """

        # Call the Vertex AI model
        response = self.model.generate_content(prompt)

        # Extract clean text content
        response_content = response.text.strip()

        # Add agent response to history
        self.conversation_history.append(f"Agent: {response_content}") 

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


    def generate_pipeline_code(self, source_tables, target_table, transformations):
        """
        Generates a multi-layer data pipeline using Dataform SQLX.

        Assumptions:
        - Assumes the user wants to use Dataform SQLX for the pipeline.
        - Assumes the user wants a multi-layer pipeline.
        - Relies on the LLM to generate valid and efficient SQLX code.
        """
        prompt = f"""
        You are a data engineering assistant. Generate a multi-layer data pipeline using Dataform SQLX with the following details:

        1. Source tables: {json.dumps(source_tables, indent=2)}
        2. Target table (destination): {target_table}
        3. Transformations: {json.dumps(transformations, indent=2)}

        Ensure the pipeline:
        - Processes data from the source tables with necessary transformations.
        - Outputs the data into the target table.
        - Is modular and follows best practices for maintainability.

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
                datasets = self.query_information_schema()
                if datasets:
                    print(f"Available datasets: {datasets}")
                    dataset_name = datasets[0]
                # if dataset_name not in datasets:
                #     # print(f"The dataset '{dataset_name}' does not exist. Available datasets: {datasets}")
                #     print(f"Available datasets: {datasets}")
                #     continue

                tables = self.query_information_schema(dataset_name=dataset_name)
                if tables:
                    print(f"Available datasets: {datasets}")
                    target_table = dataset_name + '.' + tables[0]
                # if table_name not in tables:
                #     print(f"The table '{table_name}' does not exist in dataset '{dataset_name}'. Available tables: {tables}")
                #     continue

            # Validate target table and retrieve its schema
            if target_table:
                target_dataset = target_table.split('.')[0]
                target_table_name = target_table.split('.')[1]
                if target_dataset not in datasets:
                    print(f"The dataset '{target_dataset}' does not exist.")
                    continue
                target_table_list = self.query_information_schema(dataset_name=target_dataset)
                if target_table_name not in target_table_list:
                    print(f"The target table '{target_table_name}' does not exist in the dataset '{target_dataset}'. Available tables: {target_table_list}")
                    continue

            # Generate pipeline code
            print("\nGenerating dynamic pipeline code...")
            pipeline_code = self.generate_pipeline_code(source_tables, target_table, transformations)
            print("\nHere’s your pipeline code:\n")
            print(pipeline_code)

            # Allow user to refine the pipeline
            refine = input("\nWould you like to refine this pipeline? (yes/no): ")
            if refine.lower() in ["no", "n"]:
                print("Pipeline creation completed! Exiting interactive mode.")
                break


# Initialize the agent
agent = DynamicPipelineAgent(project_id="samets-ai-playground")

# Start interactive mode
agent.interactive_mode()