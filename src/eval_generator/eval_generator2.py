from google.cloud import bigquery
import random
import csv


class EvalSetGenerator:
    def __init__(self, project_id, dataset_id):
        self.client = bigquery.Client(project=project_id)
        self.dataset_id = dataset_id
        self.dataset_ref = self.client.dataset(dataset_id)

    def get_table_names(self, suffix):
        """Fetches table names from the dataset that end with a specific suffix."""
        tables = self.client.list_tables(self.dataset_ref)
        return [table.table_id for table in tables if table.table_id.endswith(suffix)]

    def generate_data_type_standardization_prompts(self, num_prompts):
        """Generates prompts related to standardizing data types."""
        polluted_tables = self.get_table_names("_polluted")
        prompts = []

        for _ in range(num_prompts):
            table_name = random.choice(polluted_tables)
            table_ref = self.dataset_ref.table(table_name)
            table = self.client.get_table(table_ref)
            
            columns_with_issues = []
            for field in table.schema:
                # Example criteria for identifying columns needing standardization
                if field.field_type in ("STRING", "FLOAT", "INTEGER"): 
                    columns_with_issues.append(field)
            
            if not columns_with_issues:
                continue

            column = random.choice(columns_with_issues)

            if column.field_type == "STRING":
                target_type = random.choice(["INTEGER", "FLOAT", "DATE", "BOOLEAN"]) # Example target types
                prompt = f"Transform {self.dataset_id}.{table_name} Standardize the column `{column.name}` in table `{table_name}` so it has the data type {target_type}. Write the results to a new table named `{table_name.replace('_polluted', '')}_result{_:02d}`."

            elif column.field_type == "FLOAT":
                prompt = f"Transform {self.dataset_id}.{table_name} Standardize the column `{column.name}` in table `{table_name}` so it has the data type INTEGER, rounding off decimals to the nearest whole number. Write the results in a new table named `{table_name.replace('_polluted', '')}_result{_:02d}`."
            elif column.field_type == "INTEGER":
                prompt = f"Transform {self.dataset_id}.{table_name} Standardize the column `{column.name}` in table `{table_name}` so it has the data type FLOAT, Write the results in a new table named `{table_name.replace('_polluted', '')}_result{_:02d}`."

            else:
                # Handle other types or skip
                continue
            
            prompts.append({"prompt": prompt, "answer": f"Check if data type is changed to {target_type}"})

        return prompts

    def generate_data_extraction_prompts(self, num_prompts):
        """Generates prompts related to extracting data from columns."""
        polluted_tables = self.get_table_names("_polluted")
        prompts = []

        for _ in range(num_prompts):
            table_name = random.choice(polluted_tables)
            table_ref = self.dataset_ref.table(table_name)
            table = self.client.get_table(table_ref)
            
            
            string_columns = [field.name for field in table.schema if field.field_type == "STRING" and field.name not in ["uuid", "id"]] # Consider only string columns that are not IDs

            if not string_columns:
                continue

            column = random.choice(string_columns)

            # Example extractions (add more as needed)
            if "email" in column.lower():
                prompt = f"Transform {self.dataset_id}.{table_name} Extract the domain name from the field `{column}` in table `{table_name}` and store it in a column called `domain`. Write all columns with the result to a new table named `{table_name.replace('_polluted', '')}_result{_:02d}`."
                prompts.append({"prompt": prompt, "answer": f"Check if domain names are extracted correctly from column {column}"})
            elif "name" in column.lower() or "city" in column.lower():
                prompt = f"Transform {self.dataset_id}.{table_name} Extract the first word from the field `{column}` in table `{table_name}` and store it in a column called `first_part`. Write all columns with the result to a new table named `{table_name.replace('_polluted', '')}_result{_:02d}`."
                prompts.append({"prompt": prompt, "answer": f"Check if first word is extracted correctly from column {column}"})
            else:
                # Add more extraction patterns or skip
                continue
        return prompts

    def generate_data_merging_prompts(self, num_prompts):
        """Generates prompts related to merging data from multiple columns."""
        polluted_tables = self.get_table_names("_polluted")
        prompts = []

        for _ in range(num_prompts):
            table_name = random.choice(polluted_tables)
            table_ref = self.dataset_ref.table(table_name)
            table = self.client.get_table(table_ref)

            # Select columns to merge (add more logic if needed)
            columns_to_merge = random.sample([field.name for field in table.schema if field.name not in ["uuid", "id"] and field.field_type == "STRING"], min(3, len([field.name for field in table.schema if field.field_type == "STRING"])))

            if len(columns_to_merge) < 2:
                continue

            # Create prompt
            new_column_name = "_".join(columns_to_merge) + "_merged"
            prompt = f"Transform {self.dataset_id}.{table_name} Take the columns `{', '.join(columns_to_merge)}` from the table `{table_name}` and merge them into a new field called `{new_column_name}`. Separate the values with a delimiter of your choice. Write all columns to a new table named `{table_name.replace('_polluted', '')}_result{_:02d}`."

            prompts.append({"prompt": prompt, "answer": f"Check if columns {', '.join(columns_to_merge)} are merged correctly into column {new_column_name}"})

        return prompts
    
    def generate_value_replacement_prompts(self, num_prompts):
        """Generates prompts related to replacing values in columns."""
        polluted_tables = self.get_table_names("_polluted")
        prompts = []

        for _ in range(num_prompts):
            table_name = random.choice(polluted_tables)
            table_ref = self.dataset_ref.table(table_name)
            table = self.client.get_table(table_ref)

            columns = [field.name for field in table.schema]
            if not columns:
                continue

            column = random.choice(columns)

            # Example replacements (add more as needed)
            prompt = f"Transform {self.dataset_id}.{table_name} Create a view called `{table_name.replace('_polluted', '')}_view{_:02d}` that replaces all blank values in the column `{column}` of table `{table_name}` with a zero."
            prompts.append({"prompt": prompt, "answer": f"Check if blank values in column {column} are replaced with zero in the view"})

        return prompts

    def generate_data_cleaning_prompts(self, num_prompts):
        """Generates prompts related to cleaning data in columns."""
        polluted_tables = self.get_table_names("_polluted")
        prompts = []

        for _ in range(num_prompts):
            table_name = random.choice(polluted_tables)
            table_ref = self.dataset_ref.table(table_name)
            table = self.client.get_table(table_ref)

            columns = [field.name for field in table.schema]
            if not columns:
                continue

            column = random.choice(columns)

            # Example cleaning operations (add more as needed)
            if column in ["taxes", "salary", "amount"]: # Example columns that might need number cleaning
                prompt = f"Transform {self.dataset_id}.{table_name} Create a view that removes all non-numerical characters from the field `{column}` in table `{table_name}`. Name the view `{table_name.replace('_polluted', '')}_view{_:02d}`."
                prompts.append({"prompt": prompt, "answer": f"Check if all non-numerical characters are removed from column {column} in the view"})
            else:
                # Add more cleaning operations based on column name or type
                continue

        return prompts

    def generate_eval_set(self, num_prompts_per_category):
        """Generates the complete evaluation set."""
        eval_set = []
        eval_set.extend(self.generate_data_type_standardization_prompts(num_prompts_per_category))
        eval_set.extend(self.generate_data_extraction_prompts(num_prompts_per_category))
        eval_set.extend(self.generate_data_merging_prompts(num_prompts_per_category))
        eval_set.extend(self.generate_value_replacement_prompts(num_prompts_per_category))
        eval_set.extend(self.generate_data_cleaning_prompts(num_prompts_per_category))
        # Save to CSV
        with open("data_engineering_eval_set.csv", "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["prompt", "answer"])  # Write header row
            for item in eval_set:
                writer.writerow([item["prompt"], item["answer"]])
        return eval_set

# Example Usage
project_id = "samets-ai-playground"  # Replace with your GCP project ID
dataset_id = "dataduo"  # Your dataset ID

eval_generator = EvalSetGenerator(project_id, dataset_id)
eval_set = eval_generator.generate_eval_set(num_prompts_per_category=2)

# Print the evaluation set
for item in eval_set:
    print(item)