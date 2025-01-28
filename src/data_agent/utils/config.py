# agent/task_utils.py
from agent.tools_context import vertexai_tools  # Import vertexai_tools here

def get_vertexai_model():
    """
    Returns the Vertex AI model instance.
    """
    return vertexai_tools.model