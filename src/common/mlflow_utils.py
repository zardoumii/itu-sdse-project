"""MLflow setup shared by training and deployment."""

from pathlib import Path

import mlflow


def setup_mlflow_tracking(state_dir, experiment_name=None):
    """Point MLflow at a local state dir and create the experiment if needed.

    Runs and the model registry are stored in `<state_dir>/mlflow.db` (SQLite)
    and run artifacts in `<state_dir>/artifacts`, so no MLflow server has to
    be running.
    """
    state_dir = Path(state_dir).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{state_dir / 'mlflow.db'}")

    if experiment_name is None:
        return

    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(experiment_name, artifact_location=str(state_dir / "artifacts"))
    mlflow.set_experiment(experiment_name)
