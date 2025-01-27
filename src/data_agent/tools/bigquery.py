from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import json
import re
import os

from google.cloud import bigquery
from utils.tracers import trace_calls

class BigQueryTools:
    @trace_calls
    def __init__(self, project_id):
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
        return [
            {"name": row.column_name, "type": row.data_type}
            for row in self.bigquery_client.query(query)
        ]

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
    def validate_data(self, table, validation_rules):
        """
        Validates the data in the given table based on the provided validation rules.
        """
        print(f"Validating data in table '{table}'...")

        validation_results = []
        for rule_name, rule_details in validation_rules.items():
            print(f"Applying validation rule: {rule_name}")
            try:
                if rule_details["type"] == "not_null":
                    column = rule_details["column"]
                    query = f"SELECT COUNT(*) FROM {table} WHERE {column} IS NULL"
                    result = self.bigquery_client.query(query).result()
                    count = [row[0] for row in result][0]
                    validation_results.append(
                        {
                            "rule": rule_name,
                            "result": "PASSED" if count == 0 else "FAILED",
                            "details": f"{count} null values found in column '{column}'",
                        }
                    )

                elif rule_details["type"] == "unique":
                    column = rule_details["column"]
                    query = f"SELECT {column}, COUNT(*) FROM {table} GROUP BY {column} HAVING COUNT(*) > 1"
                    result = self.bigquery_client.query(query).result()
                    duplicates = [row for row in result]
                    validation_results.append(
                        {
                            "rule": rule_name,
                            "result": "PASSED" if not duplicates else "FAILED",
                            "details": f"{len(duplicates)} duplicate values found in column '{column}'",
                        }
                    )

                elif rule_details["type"] == "accepted_values":
                    column = rule_details["column"]
                    accepted_values = rule_details["values"]
                    # Build the quoted values list separately
                    quoted_values = [f"'{v}'" for v in accepted_values]
                    query = f"SELECT COUNT(*) FROM {table} WHERE {column} NOT IN ({', '.join(quoted_values)})"
                    result = self.bigquery_client.query(query).result()
                    count = [row[0] for row in result][0]
                    validation_results.append(
                        {
                            "rule": rule_name,
                            "result": "PASSED" if count == 0 else "FAILED",
                            "details": f"{count} values not in accepted values in column '{column}'",
                        }
                    )
                # Add more validation rule types as needed

                else:
                    validation_results.append(
                        {
                            "rule": rule_name,
                            "result": "ERROR",
                            "details": f"Unknown validation rule type: {rule_details['type']}",
                        }
                    )

            except Exception as e:
                print(f"Error validating data with rule '{rule_name}': {e}")
                validation_results.append(
                    {
                        "rule": rule_name,
                        "result": "ERROR",
                        "details": str(e),
                    }
                )

        return validation_results