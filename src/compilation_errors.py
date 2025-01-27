from google.cloud import dataform_v1beta1

client = dataform_v1beta1.DataformClient()
repository_path = client.repository_path("samets-ai-playground", "us-central1", "agent")
workspace_path = client.workspace_path(
    "samets-ai-playground",
    "us-central1",  # Replace with your Dataform region
    "agent",  # Replace with your Dataform repository name
    "agent",
)

# compilation_result = client.create_compilation_result(
#     parent=repository_path,
# )

compilation_result = dataform_v1beta1.CompilationResult()
compilation_result.git_commitish = "main"

# request = dataform_v1beta1.get_compilation_result(
#     parent=repository_path,
# )


# Initialize request argument(s)
compilation_result = dataform_v1beta1.CompilationResult()
compilation_result.git_commitish = "main"
compilation_result.workspace = (
        f"{workspace_path}"
    )

# compilation_result={
#     "git_commitish": "main",
#     "workspace": (
#         f"{repository_path}"
#         f"workspaces/{workspace_path}"
#     ),
# },

request = dataform_v1beta1.CreateCompilationResultRequest(
    parent=repository_path,
    compilation_result=compilation_result,
)

# Make the request
compilation_results = client.create_compilation_result(request=request)

# compilation_results = client.list_compilation_results(
#         parent=repository_path,
#     )
    # print(f"Compilation result:")
    # print(compilation_result)
    # errors = compilation_result
    
    # if not errors:
    #     print("Compilation successful!")
    #     break
print("Compilation results:")
print(compilation_results)