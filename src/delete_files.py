from google.cloud import dataform_v1beta1

location = 'us-central1'
def delete_files_in_dataform_definitions_folder(project_id, region, repository_id):
    """Deletes all files in the definitions folder of a Dataform repository.

    Args:
      project_id: The Google Cloud project ID.
      region: The region of the Dataform repository.
      repository_id: The ID of the Dataform repository.
    """

    client = dataform_v1beta1.DataformClient()
    repository_path = client.repository_path(project_id, region, repository_id)
    print(repository_path)
    # List all workspaces in the repository
    workspaces = client.list_workspaces(parent=repository_path)
    print(workspaces)
    for workspace in workspaces:
        workspace_path = workspace.name
        print(workspace_path)
        request = dataform_v1beta1.ListCompilationResultsRequest(
            parent=workspace_path,
        )
        # List all files in the definitions folder of the workspace
        # files = client.remove_directory(request=workspace_path + "/files/definitions")
        request = dataform_v1beta1.RemoveDirectoryRequest(
            workspace=workspace_path,
            path="definitions",
        )
        files = client.remove_directory(request=request)
        print(files)
        # Delete each file in the definitions folder
        request = dataform_v1beta1.MakeDirectoryRequest(
            workspace=workspace_path,
            path="definitions",
        )

        files = client.make_directory(request=request)
        print(files)

    print(
        f"All files in the definitions folder of Dataform repository {repository_id} have been deleted."
    )


# Replace with your actual values
project_id = "samets-ai-playground"
region = "us-central1"  # e.g., 'us-central1'
repository_id = "agent"

delete_files_in_dataform_definitions_folder(project_id, region, repository_id)