import vertexai
from vertexai.generative_models import GenerativeModel
from google.cloud import bigquery
import json
import re
from google.cloud import dataform_v1beta1
import os

# Configuration (Consider moving these to environment variables or a config file)
PROJECT_ID = os.environ.get("PROJECT_ID") # Replace with your GCP project ID
PROJECT_ID = "samets-ai-playground"
LOCATION = "us-central1"  # Replace with your preferred location
MODEL_NAME = "gemini-2.0-flash-exp"  # or "gemini-pro-vision" if needed

class DataPipelineMedicAgent:
    def __init__(self, project_id=PROJECT_ID, location=LOCATION, model_name=MODEL_NAME):
        vertexai.init(project=project_id, location=location)
        self.model = GenerativeModel(model_name)
        self.bigquery_client = bigquery.Client(project=project_id)
        self.dataform_client = dataform_v1beta1.DataformClient()
        self.project_id = project_id
        self.location = location

    def analyze_logs(self, logs):
        """Analyzes pipeline logs using the LLM to identify errors and potential causes."""
        prompt = f"""Analyze the following pipeline logs and identify any errors, warnings, or potential issues. Provide a summary of the findings and suggest possible causes.

        Logs:
        ```
        {logs}
        ```
        """
        response = self.model.generate_content(prompt)
        return response.text

    def analyze_sql(self, sql_code):
        """Analyzes SQL code using the LLM to identify potential issues (syntax, logic, performance)."""
        prompt = f"""Analyze the following SQL code for potential issues, including syntax errors, logical errors, and performance bottlenecks. Provide suggestions for improvement.

        SQL Code:
        ```sql
        {sql_code}
        ```
        """
        response = self.model.generate_content(prompt)
        return response.text

    def suggest_fixes(self, error_message, code=None, context=None):
        """Suggests fixes for a given error message, optionally with code and context."""
        prompt = f"""Suggest fixes for the following error:

        Error Message:
        ```
        {error_message}
        ```

        """
        if code:
            prompt += f"""Code:
            ```
            {code}
            ```
            """
        if context:
            prompt += f"""Context:
            {context}
            """
        response = self.model.generate_content(prompt)
        return response.text

    #Dataform related functions
    def upload_and_compile_files(self, files, workspace_name, repository_name="agent"):
        """Uploads and compiles files in Dataform."""
        repository_path = self.dataform_client.repository_path(self.project_id, self.location, repository_name)
        workspace_path = self.dataform_client.workspace_path(self.project_id, self.location, repository_name, workspace_name)

        try:
            for file_path, file_content in files.items():
                request = dataform_v1beta1.WriteFileRequest(
                    workspace=workspace_path, path=file_path, contents=file_content.encode("utf-8")
                )
                self.dataform_client.write_file(request=request)

            compilation_result = dataform_v1beta1.CompilationResult()
            compilation_result.git_commitish = "main"  # Or a specific commit/branch
            compilation_result.workspace = workspace_path
            request = dataform_v1beta1.CreateCompilationResultRequest(parent=repository_path, compilation_result=compilation_result)
            compilation_results = self.dataform_client.create_compilation_result(request=request)

            if hasattr(compilation_results, 'compilation_errors') and compilation_results.compilation_errors:
                return compilation_results.compilation_errors #Return errors to be handled
            else:
                return "Compilation successful."
        except Exception as e:
            return f"Error during Dataform operations: {e}"
    
    def fix_dataform_compilation_errors(self, files, errors):
        """Uses LLM to fix Dataform compilation errors."""
        prompt = f"""You are a Dataform expert. The following Dataform code has compilation errors:

        Errors:
        ```
        {errors}
        ```

        Files:
        ```json
        {json.dumps(files, indent=2)}
        ```

        Correct the code to resolve the errors. Return the corrected files in the same JSON structure. Only return the json.
        """
        response = self.model.generate_content(prompt)
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            return f"Failed to parse LLM response: {response.text}"

    def analyze_logs_with_context(self, code=None, pipeline_name=None, error_message=None):
        """Analyzes logs with SQL code, pipeline name, and error message context."""
        prompt = f"""Analyze the following pipeline logs, providing context from the associated code, pipeline name, and error message. Identify errors, warnings, and potential issues. Suggest possible causes and fixes.

        Pipeline Name: {pipeline_name or "N/A"}
        Error Message:{error_message or "N/A"}

        """
        if code:
            prompt += f"""Code:
            ```
            {code}
            ```
            """
             # Extract table names from SQL code (improved regex)
            table_names = re.findall(r"`([^`]+)`", code)
            schemas = {}
            for table_name in table_names:
                schemas[table_name] = self.get_table_schema(table_name)
            
            if schemas:
                prompt += f"""
                Table Schemas:
                ```json
                {json.dumps(schemas, indent=2)}
                ```
                """
        response = self.model.generate_content(prompt)
        return response.text

    def get_table_schema(self, table_ref):
        """Retrieves the schema of a BigQuery table, including data types."""
        try:
            table = self.bigquery_client.get_table(table_ref)
            schema = {}  # Use a dictionary to store name and type
            for field in table.schema:
                schema[field.name] = field.field_type
            return schema
        except Exception as e:
            print(f"Error getting table schema for {table_ref}: {e}")
            return None

    def get_multiline_input(self, prompt):
        """Gets multi-line input from the user."""
        print(prompt)
        lines = []
        while True:
            line = input()
            if line.strip() == "":  # Empty line signals end of input
                break
            lines.append(line)
        return "\n".join(lines)  # Join lines with newline characters

    def get_job_details(self, job_id):
        """Retrieves details of a BigQuery job, including query and errors."""
        try:
            job = self.bigquery_client.get_job(job_id)
            query = job.query
            errors = job.error_result
            if errors:
                error_message = f"{errors['message']}"
            else:
                error_message = None
            return query, error_message
        except Exception as e:
            print(f"Error getting job details: {e}")
            return None, None, None

    
    def suggest_code_fix(self, error_message, code):
        """Suggests a code fix using the LLM, including schema context."""

        # Extract table names and retrieve schemas
        table_names = re.findall(r"`([^`]+)`", code)
        schemas = {}
        for table_name in table_names:
            schemas[table_name] = self.get_table_schema(table_name)
        
        print("Schemas:", schemas)

        prompt = f"""The following SQL code produced an error. Suggest a corrected version of the code.

        Error:
        ```
        {error_message}
        ```

        Code:
        ```sql
        {code}
        ```
        """
        if schemas:
            prompt += f"""
            Table Schemas:
            ```json
            {json.dumps(schemas, indent=2)}
            ```
            """
        response = self.model.generate_content(prompt)
        return response.text

    def interactive_troubleshooting(self):
        while True:
            print("\nWhat type of issue are you facing? (logs/sql/job/dataform/exit)")
            issue_type = input("> ").lower()

            if issue_type == "job":
                job_id = input("Enter the BigQuery job ID:\n")
                query, error_message = self.get_job_details(job_id)
                if query:
                    print("Query:", query)
                if error_message:
                    print("Error:", error_message)
                    analysis = self.analyze_logs_with_context( query, job_id, error_message)
                    print("Analysis:\n", analysis)

                    if error_message and query:
                        fix = self.suggest_code_fix(error_message, query)
                        print("\nSuggested Fix:\n", fix)
                else:
                    print("Could not retrieve job logs.")
            elif issue_type == "logs":
                logs = self.get_multiline_input("Paste the pipeline logs (press Enter twice to finish):\n")
                code_available = input("Do you have the associated code? (yes/no): ").lower()
                code = None
                if code_available == "yes":
                    code_type = input("Is it SQL or Dataform code? (sql/dataform): ").lower()
                    code = self.get_multiline_input(f"Paste the {code_type.upper()} code (press Enter twice to finish):\n")
                analysis = self.analyze_logs_with_context(logs, code)
                print(analysis)

            elif issue_type == "sql":
                sql_code = self.get_multiline_input("Paste the SQL code (press Enter twice to finish):\n")
                analysis = self.analyze_sql(sql_code)
                print(analysis)
                fix = self.suggest_code_fix("Manual request", sql_code)
                print("\nSuggested Fix:\n", fix)

            elif issue_type == "dataform": # No changes needed here
                workspace_name = input("Enter Dataform workspace name: ")
                files = {}
                while True:
                    file_path = input("Enter file path (or type 'done'): ")
                    if file_path.lower() == "done":
                        break
                    file_content = self.get_multiline_input(f"Enter content for {file_path} (press Enter twice to finish):\n")
                    files[file_path] = file_content
                compile_result = self.upload_and_compile_files(files, workspace_name)
                if isinstance(compile_result, list):  # If there are compilation errors
                    fixed_files = self.fix_dataform_compilation_errors(files, compile_result)
                    if isinstance(fixed_files, dict):
                        compile_result = self.upload_and_compile_files(fixed_files, workspace_name)
                        print(compile_result)
                    else:
                        print(fixed_files)
                else:
                    print(compile_result)

            elif issue_type == "exit":
                break
            else:
                print("Invalid input.")

if __name__ == "__main__":
    agent = DataPipelineMedicAgent()
    agent.interactive_troubleshooting()