"""Compare the best new run with the production model and decide which one wins."""

import math

import pandas as pd
from mlflow.tracking import MlflowClient

from src.deployment.production import ProductionModel


def _clean(score) -> float | None:
    # MLflow gives NaN (not None) for a metric a run never logged
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return None
    return float(score)


def get_score(metrics: dict) -> float | None:
    """Test F1-score of a run, falling back to f1_score like the notebook did."""
    score = _clean(metrics.get("test_f1_score"))
    if score is None:
        score = _clean(metrics.get("f1_score"))
    return score


def should_promote(new_score: float | None, prod_score: float | None) -> bool:
    """The new model wins if there is no scored production model, or if it scores higher."""
    if prod_score is None:
        return True
    if new_score is None:
        return False
    return new_score > prod_score


def choose_run_to_promote(
    best_run: pd.Series,
    production: ProductionModel | None,
    client: MlflowClient,
) -> str | None:
    """Return the run id to promote, or None if the production model should stay."""
    new_score = get_score({
        "test_f1_score": best_run.get("metrics.test_f1_score"),
        "f1_score": best_run.get("metrics.f1_score"),
    })

    if production is None:
        print("No model in production, the best run will be promoted")
        return best_run["run_id"]

    prod_score = get_score(client.get_run(production.run_id).data.metrics)
    print(f"New model: {new_score} | Production model: {prod_score}")

    if should_promote(new_score, prod_score):
        print("New model is better, promoting it")
        return best_run["run_id"]

    print("Production model is at least as good, keeping it")
    return None
