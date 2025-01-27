from google.cloud import bigquery
import vertexai

from vertexai.generative_models import GenerativeModel
import csv
import os
import random
import time

class EvalSetGenerator:
    def __init__(self, project_id, dataset_id, gemini_api_key):
        self.client = bigquery.Client(project=project_id)
        self.dataset_id = dataset_id
        self.dataset_ref = self.client.dataset(dataset_id)
        vertexai.init(project=project_id, location="us-central1")
        model_name = "gemini-2.0-flash-exp"
        self.model = GenerativeModel(model_name)
        # self.model = genai.GenerativeModel('gemini-pro')
        self.mapping_data = self.load_and_structure_mapping_data()

    def get_table_names(self, suffix):
        """Fetches table names from the dataset that end with a specific suffix."""
        tables = self.client.list_tables(self.dataset_ref)
        return [table.table_id for table in tables if table.table_id.endswith(suffix)]

    def load_and_structure_mapping_data(self):
        """Loads all mapping tables and structures the data in a dictionary."""
        mapping_tables = self.get_table_names("_mapping")
        mapping_data = {}

        for mapping_table in mapping_tables:
            print("Mapping table:" + mapping_table)
            table_ref = self.dataset_ref.table(mapping_table)
            table = self.client.get_table(table_ref)
            polluted_table = mapping_table.replace("_mapping", "_polluted")
            clean_table = mapping_table.replace("_mapping", "")

            query = f"SELECT * FROM `{self.dataset_id}.{mapping_table}`"
            query_job = self.client.query(query)
            results = query_job.result()

            mapping_data[polluted_table] = {
                "clean_table": clean_table,
                "mappings": [],
            }

            for row in results:
                row_keys = [field.name.lower() for field in results.schema]
                # Check for both "type" and "Type" variations
                if "type" in row_keys:
                    type_value_index = row_keys.index("type")
                    type_value = list(row.values())[type_value_index]
                elif "Type" in row_keys:
                    type_value_index = row_keys.index("Type")
                    type_value = list(row.values())[type_value_index]
                else:
                    type_value = None


                mapping_data[polluted_table]["mappings"].append(
                    {
                        "polluted_column": row.polluted_column,
                        "clean_column": row.clean_column,
                        "type": type_value,
                        "transformation_type": row.transformation_type,
                    }
                )

        return mapping_data

    def generate_prompts_with_gemini(self, num_prompts_per_table):
      """Generates evaluation prompts using the Gemini API based on mapping data."""
      eval_set = []

      for polluted_table, table_data in self.mapping_data.items():
          for _ in range(num_prompts_per_table):
              if not table_data["mappings"]:
                  continue  # Skip if no mappings for this table

              # Select a random mapping for prompt generation
              mapping = random.choice(table_data["mappings"])
              polluted_column = mapping["polluted_column"]
              clean_column = mapping["clean_column"]
              transformation_type = mapping["transformation_type"]
              type_ = mapping["type"]
              clean_table = table_data["clean_table"]

              # Construct a prompt for Gemini
              gemini_prompt = f"""
              You are a data engineer tasked with creating data cleaning pipelines. Generate a prompt to instruct a data engineering agent to transform the polluted table '{self.dataset_id}.{polluted_table}'. 
              The specific task is related to the column '{polluted_column}'. 

              Here are the details from the mapping table:
              - Polluted Table: {polluted_table}
              - Polluted Column: {polluted_column}
              - Clean Column: {clean_column}
              - Transformation Type: {transformation_type}
              - Type: {type_}
              - Clean Table: {clean_table}

              The generated prompt should:
              1. Clearly state the transformation needed based on 'transformation_type' and 'type'.
              2. Instruct to write the results to a new table or view with a specific naming pattern (e.g., '{polluted_table}_result' or '{polluted_table}_view').
              3. Only use information available in the polluted table.
              4. Be concise and specific, suitable for a data engineering agent to understand and execute.
              5. {self.dataset_id}.{polluted_table} should be the table name

              Example:
              Mapping row has: 	
                    years_of_experience
                    years_of_experience
                    transformation
                    data_type_anomalies
              Output Prompt: Standardize the column years of experience on table {polluted_table} so it has the same data type, a number. Round off decimals to the nearest whole number. Write the results in a new table named org_employee_table13_result01.
              
              Generate ONE prompt for the data engineering agent.
              """

              # Exponential backoff implementation
              retries = 0
              max_retries = 5  # Maximum number of retries
              delay = 60  # Initial delay in seconds

              while retries < max_retries:
                try:
                    # Get response from Gemini
                    response = self.model.generate_content(gemini_prompt)
                    prompt = response.text
                    answer = f"Verify the transformation on column '{polluted_column}' in table '{polluted_table}' as per type '{type_}' and transformation '{transformation_type}'. Check the results in the new table or view."
                    eval_set.append({"table": polluted_table, "prompt": prompt, "answer": answer})
                    break  # Success, exit the retry loop

                except Exception as e:
                    if "429 Quota exceeded" in str(e):
                        print(
                            f"Quota exceeded for {polluted_table}, attempt {retries + 1}. Retrying in {delay} seconds..."
                        )
                        time.sleep(delay)
                        retries += 1
                        delay *= 2  # Exponentially increase delay
                    else:
                        print(f"Error generating prompt for {polluted_table}: {e}")
                        break # Exit the retry loop for non-quota errors

      return eval_set

    def generate_eval_set(self, num_prompts_per_table):
        """Generates the complete evaluation set and saves it to a CSV file."""
        eval_set = self.generate_prompts_with_gemini(num_prompts_per_table)

        # Save to CSV
        with open(
            "data_engineering_eval_set.csv", "w", newline="", encoding="utf-8"
        ) as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["prompt", "answer"])  # Write header row
            for item in eval_set:
                writer.writerow([item["prompt"], item["answer"]])

        return eval_set

# Example Usage
project_id = "samets-ai-playground"  # Replace with your GCP project ID
dataset_id = "dataduo"  # Your dataset ID
gemini_api_key = os.environ.get("GEMINI_API_KEY")

eval_generator = EvalSetGenerator(project_id, dataset_id, gemini_api_key)
eval_set = eval_generator.generate_eval_set(num_prompts_per_table=2)

print("Evaluation set generated and saved to data_engineering_eval_set.csv")