from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import json
import re
import os

from google.cloud import dataform_v1beta1
from utils.tracers import trace_calls
from utils.prompt_loader import load_prompt # Import the function


class DataformTools:
    @trace_calls
    def __init__(self, project_id, location="us-central1"):
        self.project_id = project_id
        self.location = location
        self.client = dataform_v1beta1.DataformClient()

    @trace_calls
    def upload_and_compile_files(self, files, workspace_name):
        """
        Uploads and compiles the files in Dataform.
        """
        
        repository_path = self.client.repository_path(self.project_id, self.location, "agent")
        workspace_path = self.client.workspace_path(
            self.project_id,
            self.location,  # Replace with your Dataform region
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
                self.client.write_file(request=request)
            except Exception as e:
                print(f"Error uploading file '{file_path}': {e}")

        # Attempt to fix compilation errors
        print(f"Compiling...")
        for _ in range(3):  # Iterate 3 times
            compilation_result = dataform_v1beta1.CompilationResult()
            compilation_result.git_commitish = "main"
            compilation_result.workspace = f"{workspace_path}"

            request = dataform_v1beta1.CreateCompilationResultRequest(
                parent=repository_path, compilation_result=compilation_result
            )

            compilation_results = self.client.create_compilation_result(request=request)

            print("Compilation results:")
            print(compilation_results)
            if (
                hasattr(compilation_results, "compilation_errors")
                and compilation_results.compilation_errors
            ):
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
                        self.client.write_file(request=request)
                except Exception as e:
                    print(f"Error compiling or fixing files: {e}")
            else:
                print("Compilation Successful")
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
    def fix_compilation_errors(self, files, errors):
        """
        Asks the LLM to fix the compilation errors.
        """
        from data_agent.tools.vertex_ai import VertexAITools
        vertexai_tools = VertexAITools(project_id=self.project_id)

        print("Fixing the issues..")
        with open("prompts/guides/dataform_examples.txt", "r") as f:
            troubleshooting_guide = f.read()

        # prompt = f"""
        # The Dataform code has the following compilation errors:

        # {errors}

        # Here are the files:

        # {files}

        # Please provide the corrected code in the same format as before, 
        # ensuring that the errors are fixed. 

        # Structure your response as a JSON object with the following format:

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

        # Here are some of the examples for troubleshooting:

        # {troubleshooting_guide}


        # Only include the files that need to be fixed and any additional files needed.
        # Only return the JSON formatted response, nothing else.
        # """
        # Load the prompt template
        prompt_template = load_prompt("fix_compilation_errors")

        # Fill the prompt template
        prompt = prompt_template.format(errors=errors, files=files)

        # Do not put unnecessary escape.
        response = vertexai_tools.model.generate_content(prompt)

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
                response_content = vertexai_tools.refine_llm_response(prompt, response_content, e)