"""Look up the model that is currently in production."""

from typing import NamedTuple

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.deployment.config import MODEL_NAME, PRODUCTION_ALIAS


class ProductionModel(NamedTuple):
    uri: str
    version: str
    run_id: str


def get_production_model(client: MlflowClient, model_name: str = MODEL_NAME) -> ProductionModel | None:
    """Return the production model, or None if nothing is in production yet.

    Checks the "Production" alias first and falls back to the older
    "Production" stage, like the notebook did.
    """
    try:
        mv = client.get_model_version_by_alias(name=model_name, alias=PRODUCTION_ALIAS)
        print(f"Production model: {model_name} version {mv.version} (alias)")
        return ProductionModel(f"models:/{model_name}@{PRODUCTION_ALIAS}", mv.version, mv.run_id)
    except MlflowException:
        # No such alias, or the model isn't registered at all
        pass

    for mv in client.search_model_versions(f"name='{model_name}'"):
        if mv.current_stage == "Production":
            print(f"Production model: {model_name} version {mv.version} (stage)")
            return ProductionModel(f"models:/{model_name}/{mv.version}", mv.version, mv.run_id)

    print("No model in production")
    return None
