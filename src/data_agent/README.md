# Data Pipeline Agent

This project implements an intelligent data pipeline agent that automates the creation, modification, and management of data pipelines using Dataform, BigQuery, and Vertex AI.

## Project Structure

The project is structured as follows:

- `main.py`: Main entry point for running the agent in interactive mode.
- `agent/`: Contains core agent logic, including state management, execution, and prompts.
    - `__init__.py`: Marks the directory as a Python package.
    - `agent_executor.py`: Defines the agent's execution logic using LangGraph, orchestrating the flow of actions based on user input and system state.
    - `agent_state.py`: Defines the `AgentState` class, which manages the agent's state, including messages, files, compilation results, and other relevant data.
    - `prompts.py`: Contains prompt templates used for interacting with the Vertex AI model.
- `tools/`: Contains modules that wrap interactions with external services like Dataform, BigQuery, and Vertex AI.
    - `__init__.py`: Marks the directory as a Python package.
    - `dataform.py`: Implements functions for interacting with Dataform, such as uploading files and managing workspaces.
    - `bigquery.py`: Provides functions for interacting with BigQuery, such as querying information schema and validating data.
    - `vertex_ai.py`: Contains functions for interacting with Vertex AI's GenerativeModel and related functionalities.
- `utils/`: Contains utility functions and resources used across the project.
    - `__init__.py`: Marks the directory as a Python package.
    - `tracers.py`: Implements a tracing function for debugging.
    - `dataform_examples.txt`: A text file containing examples of Dataform SQLX code used for generating pipeline code.
    - `validations.py`: Defines Pydantic models for input and output validation of tool functions.

## Getting Started

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Set up Environment:**
    -   Ensure you have a Google Cloud project set up with the necessary APIs enabled (Dataform, BigQuery, Vertex AI).
    -   Configure your environment with appropriate credentials (e.g., using `gcloud auth application-default login`).

3.  **Run the Agent:**
    ```bash
    python main.py
    ```

## Interactive Mode

The agent runs in interactive mode, guiding you through the process of creating or modifying a data pipeline. It will ask you for your requirements, and then use its tools to generate code, manage Dataform workspaces, query BigQuery metadata, and validate the pipeline.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests to enhance the agent's capabilities.