"""Drift detection over the requests logged by the served model.

Two-sample KS tests compare the logged confidences and predicted classes with
the baseline statistics written by the training pipeline.
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)

INFERENCE_LOG_PATH = Path("artifacts/inference_log.json")
DRIFT_BASELINE_STATS_PATH = Path("artifacts/drift_baseline_stats.json")

DEFAULT_DAYS = 7
KS_PVALUE_THRESHOLD = 0.05
MEAN_DIFF_THRESHOLD = 0.1

# The baseline only has summary statistics, so we draw samples from it.
# A fixed seed means /drift gives the same answer for the same logs.
SEED = 42


def _no_drift():
    return {"ks_statistic": 0.0, "ks_pvalue": 1.0, "drift_detected": False, "threshold_ks": KS_PVALUE_THRESHOLD}


def load_inference_logs(log_path: Path = INFERENCE_LOG_PATH) -> list[dict]:
    """Load the inference log. Returns [] if it is missing or unreadable."""
    if not log_path.exists():
        logger.warning("Inference log not found at %s", log_path)
        return []
    try:
        logs = json.loads(log_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Failed to load inference log: %s", e)
        return []
    return logs if isinstance(logs, list) else []


def load_baseline_stats(baseline_path: Path = DRIFT_BASELINE_STATS_PATH) -> dict | None:
    """Load the baseline written during training. Returns None if missing or unreadable."""
    if not baseline_path.exists():
        logger.warning("Baseline stats not found at %s", baseline_path)
        return None
    try:
        return json.loads(baseline_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Failed to load baseline stats: %s", e)
        return None


def filter_logs_by_time(logs: list[dict], days: int = DEFAULT_DAYS) -> list[dict]:
    """Keep the entries from the last `days` days.

    Entries without a timestamp are dropped. Entries whose timestamp can't
    be parsed are kept, same as in the notebook.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    filtered = []
    for log in logs:
        timestamp = log.get("timestamp")
        if not timestamp:
            continue
        try:
            if datetime.fromisoformat(timestamp) >= cutoff:
                filtered.append(log)
        except (ValueError, TypeError):
            filtered.append(log)
    return filtered


def confidence_stats(confidences: list[float]) -> dict:
    if not confidences:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    values = np.array(confidences)
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def check_confidence_drift(confidences: list[float], baseline_stats: dict, rng: np.random.Generator) -> dict:
    """KS test of the logged confidences against the baseline, plus a mean-shift check."""
    if "confidence_statistics" not in baseline_stats or len(confidences) < 2:
        return _no_drift()

    baseline = baseline_stats["confidence_statistics"]
    current = np.array(confidences)
    current_mean = float(current.mean())

    baseline_samples = rng.normal(baseline["mean"], baseline["std"], size=len(current))
    ks_statistic, ks_pvalue = ks_2samp(baseline_samples, current)
    mean_diff = abs(current_mean - baseline["mean"])

    return {
        "ks_statistic": float(ks_statistic),
        "ks_pvalue": float(ks_pvalue),
        "mean_diff": mean_diff,
        "current_mean": current_mean,
        "baseline_mean": baseline["mean"],
        "drift_detected": bool(ks_pvalue < KS_PVALUE_THRESHOLD or mean_diff > MEAN_DIFF_THRESHOLD),
        "threshold_ks": KS_PVALUE_THRESHOLD,
        "threshold_mean": MEAN_DIFF_THRESHOLD,
    }


def check_class_distribution_drift(predictions: list[int], baseline_stats: dict, rng: np.random.Generator) -> dict:
    """KS test of the logged predictions against the predictions made on the baseline data.

    The notebook compared against `class_distribution`, which counts the true
    labels. The model's predictions should be compared with its own earlier
    predictions, which training stores as `prediction_distribution`.
    """
    baseline = baseline_stats.get("prediction_distribution")
    if not baseline or not baseline["total"] or not predictions:
        return _no_drift()

    total = baseline["total"]
    p = [baseline["pred_0_count"] / total, baseline["pred_1_count"] / total]
    baseline_samples = rng.choice([0, 1], size=total, p=p)
    ks_statistic, ks_pvalue = ks_2samp(baseline_samples, predictions)

    return {
        "ks_statistic": float(ks_statistic),
        "ks_pvalue": float(ks_pvalue),
        "drift_detected": bool(ks_pvalue < KS_PVALUE_THRESHOLD),
        "threshold_ks": KS_PVALUE_THRESHOLD,
    }


def detect_drift(logs: list[dict], baseline_stats: dict | None = None, days: int = DEFAULT_DAYS) -> dict:
    """Run the confidence and class-distribution checks over the recent logs."""
    recent = filter_logs_by_time(logs, days=days)
    if not recent:
        logger.warning("No inference logs found in the last %d days", days)
        return {
            "overall_drift_detected": False,
            "num_samples": 0,
            "confidence_drift": None,
            "class_distribution_drift": None,
            "error": "No logs found",
        }

    # Failed requests are logged with None, skip them
    confidences = [float(log["confidence"]) for log in recent if log.get("confidence") is not None]
    predictions = [int(log["prediction"]) for log in recent if log.get("prediction") is not None]

    baseline_stats = baseline_stats or {}
    rng = np.random.default_rng(SEED)
    confidence_drift = check_confidence_drift(confidences, baseline_stats, rng)
    class_drift = check_class_distribution_drift(predictions, baseline_stats, rng)

    return {
        "overall_drift_detected": confidence_drift["drift_detected"] or class_drift["drift_detected"],
        "num_samples": len(recent),
        "confidence_drift": confidence_drift,
        "class_distribution_drift": class_drift,
        "confidence_stats": confidence_stats(confidences),
        "class_distribution": {
            "class_0_count": predictions.count(0),
            "class_1_count": predictions.count(1),
            "total": len(predictions),
        },
    }
