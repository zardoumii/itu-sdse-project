from types import SimpleNamespace

import pandas as pd
import pytest

from src.deployment.compare import choose_run_to_promote, get_score, should_promote
from src.deployment.production import ProductionModel


class FakeClient:
    """Stands in for MlflowClient: get_run returns a run with the given metrics."""

    def __init__(self, metrics):
        self.metrics = metrics

    def get_run(self, run_id):
        return SimpleNamespace(data=SimpleNamespace(metrics=self.metrics))


def best_run(run_id="new-run", f1=0.85):
    return pd.Series({"run_id": run_id, "metrics.test_f1_score": f1})


PRODUCTION = ProductionModel("models:/cnn_model@Production", "1", "prod-run")


@pytest.mark.parametrize(
    "new_score, prod_score, expected",
    [
        (0.9, 0.8, True),
        (0.7, 0.8, False),
        (0.8, 0.8, False),  # a tie keeps the production model
        (0.1, 0.0, True),  # 0.0 is a real score, not a missing one
        (None, 0.5, False),
    ],
)
def test_should_promote_picks_higher_score(new_score, prod_score, expected):
    assert should_promote(new_score, prod_score) is expected


def test_should_promote_when_production_has_no_score():
    assert should_promote(0.5, None) is True


def test_get_score_falls_back_to_f1_score():
    assert get_score({"test_f1_score": float("nan"), "f1_score": 0.6}) == 0.6
    assert get_score({}) is None


def test_promotes_best_run_when_no_production_model():
    assert choose_run_to_promote(best_run(), None, FakeClient({})) == "new-run"


def test_promotes_best_run_when_it_beats_production():
    client = FakeClient({"test_f1_score": 0.80})
    assert choose_run_to_promote(best_run(f1=0.85), PRODUCTION, client) == "new-run"


def test_keeps_production_when_it_is_better():
    client = FakeClient({"test_f1_score": 0.90})
    assert choose_run_to_promote(best_run(f1=0.85), PRODUCTION, client) is None
