from google.cloud import bigquery
import random

class EvalSetGenerator:
    def __init__(self, project_id, dataset_id):
        self.client = bigquery.Client(project=project_id)
        self.dataset_id = dataset_id
        self.dataset_ref = self.client.dataset(dataset_id)

    def get_table_names(self, suffix):
        """Fetches table names from the dataset that end with a specific suffix."""
        tables = self.client.list_tables(self.dataset_ref)
        return [table.table_id for table in tables if table.table_id.endswith(suffix)]

    def get_column_names(self, table_id):
        """Fetches column names for a given table."""
        table_ref = self.dataset_ref.table(table_id)
        table = self.client.get_table(table_ref)
        return [field.name for field in table.schema]

    def generate_mapping_prompts(self, num_prompts):
        """Generates prompts related to table and column mapping."""
        mapping_tables = self.get_table_names("_mapping")
        prompts = []

        for _ in range(num_prompts):
            mapping_table = random.choice(mapping_tables)
            polluted_table = mapping_table.replace("_mapping", "_polluted")
            clean_table = mapping_table.replace("_mapping", "")

            # Scenario 1: Identify Mapping (Table Level)
            if random.random() < 0.5:
                prompt = f"Given the polluted table `{self.dataset_id}.{polluted_table}`, find its corresponding clean table using the mapping information in `{self.dataset_id}.{mapping_table}`."
                prompts.append({"prompt": prompt, "answer": f"`{self.dataset_id}.{clean_table}`"})

            # Scenario 2: Column Mapping within a table
            else:
                prompt = f"What are the column mappings between the polluted table `{self.dataset_id}.{polluted_table}` and the clean table `{self.dataset_id}.{clean_table}`? Use the mapping table `{self.dataset_id}.{mapping_table}` to identify the mappings."

                # Fetch column mappings from the mapping table
                query = f"""
                    SELECT polluted_column, clean_column
                    FROM `{self.dataset_id}.{mapping_table}`
                    WHERE clean_column IS NOT NULL
                """
                query_job = self.client.query(query)
                results = query_job.result()
                column_mappings = {row.polluted_column: row.clean_column for row in results}

                answer = ""
                if column_mappings:
                    answer = "Column mappings:\n"
                    for polluted_col, clean_col in column_mappings.items():
                        answer += f"- `{polluted_col}` maps to `{clean_col}`\n"
                else:
                    answer = "No column mappings found."

                prompts.append({"prompt": prompt, "answer": answer})

        return prompts

    def generate_cleaning_prompts(self, num_prompts):
        """Generates prompts related to data cleaning operations."""
        mapping_tables = self.get_table_names("_mapping")
        prompts = []

        for _ in range(num_prompts):
            mapping_table = random.choice(mapping_tables)
            polluted_table = mapping_table.replace("_mapping", "_polluted")
            clean_table = mapping_table.replace("_mapping", "")

            # Fetch column names for polluted and clean tables
            polluted_columns = self.get_column_names(polluted_table)
            clean_columns = self.get_column_names(clean_table)

            if not polluted_columns or not clean_columns:
                continue  # Skip if no columns found

            # Scenario 3: Column Renaming
            if random.random() < 0.3:
                prompt = f"Identify column renaming operations needed to transform `{self.dataset_id}.{polluted_table}` to `{self.dataset_id}.{clean_table}` based on the mappings in `{self.dataset_id}.{mapping_table}`. Specifically, focus on columns that have a 'merge' type and are not null in the 'clean_column' field. Provide the SQL code or logic to perform these renamings."

                # Find column renamings from the mapping table
                query = f"""
                    SELECT polluted_column, clean_column
                    FROM `{self.dataset_id}.{mapping_table}`
                    WHERE Type = 'merge' AND clean_column IS NOT NULL
                """
                query_job = self.client.query(query)
                results = query_job.result()
                renamings = {row.polluted_column: row.clean_column for row in results}

                answer = ""
                if renamings:
                    answer = "Column renamings:\n"
                    for polluted, clean in renamings.items():
                        answer += f"- Rename `{polluted}` to `{clean}`\n"
                    # Example SQL
                    answer += "\nExample SQL:\n"
                    answer += f"ALTER TABLE `{self.dataset_id}.{polluted_table}`\n"
                    for polluted, clean in renamings.items():
                        answer += f"RENAME COLUMN `{polluted}` TO `{clean}`,\n"
                    answer = answer.rstrip(",\n")
                else:
                    answer = "No column renamings needed."

                prompts.append({"prompt": prompt, "answer": answer})

            # Scenario 4: Data Type Correction
            elif random.random() < 0.6:
                # Get schema details for polluted and clean tables
                polluted_table_ref = self.dataset_ref.table(polluted_table)
                polluted_table_schema = self.client.get_table(polluted_table_ref).schema

                clean_table_ref = self.dataset_ref.table(clean_table)
                clean_table_schema = self.client.get_table(clean_table_ref).schema

                polluted_types = {field.name: field.field_type for field in polluted_table_schema}
                clean_types = {field.name: field.field_type for field in clean_table_schema}

                prompt = f"Identify data type discrepancies between `{self.dataset_id}.{polluted_table}` and `{self.dataset_id}.{clean_table}` that are marked as 'data_type_anomalies' in the `transformation_type` column of the mapping table `{self.dataset_id}.{mapping_table}`. Suggest corrections to ensure data type consistency based on the clean table."

                # Find data type differences
                type_corrections = {}
                query = f"""
                    SELECT polluted_column, clean_column
                    FROM `{self.dataset_id}.{mapping_table}`
                    WHERE transformation_type = 'data_type_anomalies'
                """
                query_job = self.client.query(query)
                results = query_job.result()

                for row in results:
                    polluted_col = row.polluted_column
                    clean_col_name = row.clean_column

                    if polluted_col in polluted_types:
                         if clean_col_name in clean_types and polluted_types[polluted_col] != clean_types[clean_col_name]:
                             type_corrections[polluted_col] = clean_types[clean_col_name]

                answer = ""
                if type_corrections:
                    answer = "Data type corrections needed:\n"
                    for col, clean_type in type_corrections.items():
                        answer += f"- Column `{col}` should be changed from `{polluted_types[col]}` to `{clean_type}`\n"
                    # Example SQL
                    answer += "\nExample SQL for data type corrections:\n"
                    answer += f"ALTER TABLE `{self.dataset_id}.{polluted_table}`\n"
                    for col, clean_type in type_corrections.items():
                        answer += f"ALTER COLUMN `{col}` SET DATA TYPE {clean_type},\n"
                    answer = answer.rstrip(",\n")
                else:
                    answer = "No data type corrections needed."

                prompts.append({"prompt": prompt, "answer": answer})
            
            # Scenario 5: Data Normalization
            elif random.random() < 0.9:
                prompt = f"Identify columns in `{self.dataset_id}.{polluted_table}` that require normalization based on the `{self.dataset_id}.{mapping_table}`. Provide specific steps or SQL code to perform the normalization to match the format of the corresponding columns in `{self.dataset_id}.{clean_table}`."
                
                # Find columns that need normalization from the mapping table
                query = f"""
                    SELECT polluted_column, clean_column
                    FROM `{self.dataset_id}.{mapping_table}`
                    WHERE transformation_type = 'normalization'
                """
                query_job = self.client.query(query)
                results = query_job.result()
                normalization_tasks = {row.polluted_column: row.clean_column for row in results}

                answer = ""
                if normalization_tasks:
                    answer = "Normalization tasks:\n"
                    for polluted, clean in normalization_tasks.items():
                        answer += f"- Normalize data in column `{polluted}` to match the format of column `{clean}`\n"
                    # Example SQL (This will be highly dependent on the specific type of normalization needed)
                    answer += "\nExample SQL for normalization (adjust based on specific needs):\n"
                    answer += f"UPDATE `{self.dataset_id}.{polluted_table}`\n"
                    for polluted, clean in normalization_tasks.items():
                        answer += f"SET `{polluted}` = (CASE WHEN condition THEN normalized_value ELSE `{polluted}` END),\n" # Replace condition and normalized_value with your logic
                    answer = answer.rstrip(",\n")
                else:
                    answer = "No normalization tasks identified."

                prompts.append({"prompt": prompt, "answer": answer})

            # Scenario 6: Data Format Standardization, Scenario 7: Handling Missing Values, Scenario 8: Deduplication, Scenario 9: Outlier Treatment are also to be added here.
            # Add more scenarios as needed

        return prompts

    def generate_eval_set(self, num_mapping_prompts, num_cleaning_prompts):
        """Generates the complete evaluation set."""
        eval_set = []
        eval_set.extend(self.generate_mapping_prompts(num_mapping_prompts))
        eval_set.extend(self.generate_cleaning_prompts(num_cleaning_prompts))
        return eval_set

# Example Usage
project_id = "samets-ai-playground"  # Replace with your GCP project ID
dataset_id = "dataduo"  # Your dataset ID

eval_generator = EvalSetGenerator(project_id, dataset_id)
eval_set = eval_generator.generate_eval_set(num_mapping_prompts=10, num_cleaning_prompts=20)

# Print the evaluation set
for item in eval_set:
    print(item)