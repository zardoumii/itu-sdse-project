"""Find the best training run of the experiment."""

import mlflow
import pandas as pd

from src.deployment.config import EXPERIMENT_NAME


def get_best_run(experiment_name: str = EXPERIMENT_NAME) -> pd.Series:
    """Return the run with the highest test F1-score."""
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found, has the training pipeline run?")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.test_f1_score DESC"],
        max_results=1,
    )
    if runs.empty:
        raise ValueError(f"Experiment '{experiment_name}' has no runs")

    best_run = runs.iloc[0]
    print(f"Best run: {best_run['run_id']} (test F1: {best_run.get('metrics.test_f1_score')})")
    return best_run
