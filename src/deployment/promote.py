"""Promote the winning run's model to Production."""

import mlflow
from mlflow.tracking import MlflowClient

from src.deployment.config import MODEL_NAME, PRODUCTION_ALIAS, PYFUNC_ARTIFACT_PATH


def get_or_register_version(client: MlflowClient, run_id: str, model_name: str = MODEL_NAME) -> str:
    """Return the registered version for run_id, registering the model if needed.

    The training pipeline already registers every run, so normally this just
    finds that version.
    """
    versions = [mv for mv in client.search_model_versions(f"name='{model_name}'") if mv.run_id == run_id]
    if versions:
        return str(max(versions, key=lambda mv: int(mv.version)).version)

    print(f"Run {run_id} is not registered yet, registering it")
    return str(mlflow.register_model(f"runs:/{run_id}/{PYFUNC_ARTIFACT_PATH}", model_name).version)


def promote_to_production(client: MlflowClient, run_id: str, model_name: str = MODEL_NAME) -> str:
    """Move the run's model version through Staging to Production. Returns the version."""
    version = get_or_register_version(client, run_id, model_name)
    print(f"Promoting {model_name} version {version}")

    # Always via Staging, like the notebook. Training already puts new versions there.
    if client.get_model_version(model_name, version).current_stage != "Staging":
        client.transition_model_version_stage(model_name, version, "Staging", archive_existing_versions=True)

    client.transition_model_version_stage(model_name, version, "Production", archive_existing_versions=True)
    client.set_registered_model_alias(model_name, PRODUCTION_ALIAS, version)
    client.set_model_version_tag(model_name, version, "deployed_to", "production")

    print(f"{model_name} version {version} is now in Production")
    return version
