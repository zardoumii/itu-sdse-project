"""Run the deployment step: `python -m src.deployment.run`.

Run the training pipeline first (`python -m src.training.run`). This picks the
best run, promotes it if it beats the production model, and exports the
production model to models/model for the serving image.
"""

from mlflow.tracking import MlflowClient

from src.common.mlflow_utils import setup_mlflow_tracking
from src.deployment.compare import choose_run_to_promote
from src.deployment.config import EXPERIMENT_NAME, MLFLOW_STATE_DIR
from src.deployment.export import export_production_model
from src.deployment.production import get_production_model
from src.deployment.promote import promote_to_production
from src.deployment.select_model import get_best_run


def main() -> None:
    setup_mlflow_tracking(MLFLOW_STATE_DIR, EXPERIMENT_NAME)
    client = MlflowClient()

    best_run = get_best_run()
    production = get_production_model(client)
    run_id = choose_run_to_promote(best_run, production, client)
    if run_id is not None:
        promote_to_production(client, run_id)

    # Export even when nothing changed, so models/model always holds the production model
    export_production_model(client)


if __name__ == "__main__":
    main()
