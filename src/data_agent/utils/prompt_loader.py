# data_pipeline_agent/utils/prompt_loader.py
import yaml
from string import Template
import os

def load_prompt(prompt_name: str, data: dict = None) -> str:
    """Loads a prompt template from a YAML file and fills it with data.

    Args:
        prompt_name: The name of the prompt to load.
        data: Optional data dictionary to fill in placeholders in the prompt.

    Returns:
        The prompt string with placeholders filled.
    """
    prompts_dir = os.path.join(os.path.dirname(__file__), "..", "prompts")
    file_path = os.path.join(prompts_dir, f"{prompt_name}.yaml")

    with open(file_path, "r") as f:
        prompt_data = yaml.safe_load(f)

    template = Template(prompt_data["template"])

    # Add dataform examples to the data dictionary if prompt name is generate_pipeline_code
    if prompt_name == "generate_pipeline_code":
        with open(os.path.join(os.path.dirname(__file__), "../prompts/guides/dataform_examples.txt"), "r") as examples_file:
            dataform_examples = examples_file.read()
        if data is None:
            data = {}
        data["dataform_examples"] = dataform_examples

    # Add troubleshooting guide to the data dictionary if prompt name is fix_compilation_errors
    if prompt_name == "fix_compilation_errors":
        with open(os.path.join(os.path.dirname(__file__), "../prompts/guides/dataform_examples.txt"), "r") as examples_file:
            troubleshooting_guide = examples_file.read()
        if data is None:
            data = {}
        data["troubleshooting_guide"] = troubleshooting_guide

    # Add additional logic for ask_for_clarifications prompt
    if prompt_name == "ask_for_clarifications":
        missing_info = data.get("missing_information", [])
        user_request = data.get("user_request", "")

        missing_information_prompt = ""
        if missing_info:
            missing_information_prompt += "I am missing the following information:\n"
            for info in missing_info:
                missing_information_prompt += f"- {info}\n"

        user_request_prompt = ""
        if user_request:
            user_request_prompt += f"The user request is: {user_request}\n"

        general_follow_up_prompt = (
            "Please provide additional information regarding:\n"
            "- Specific transformations\n"
            "- Desired output format\n"
            "- Performance considerations\n"
            "- Pipeline deployment location\n"
            "- Scheduling requirements\n"
        )
        if data is None:
            data = {}

        data["missing_information_prompt"] = missing_information_prompt
        data["user_request_prompt"] = user_request_prompt
        data["general_follow_up_prompt"] = general_follow_up_prompt

    if data:
        return template.substitute(data)
    else:
        return template.template