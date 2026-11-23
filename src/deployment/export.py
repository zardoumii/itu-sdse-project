"""Export the Production model to a plain directory for the serving image.

The serving container does not use MLflow tracking, so the model is downloaded
to EXPORT_DIR together with:
- version.json: model version and training time, for the /version endpoint
- drift_baseline_stats.json: the baseline of this model's own training run,
  for the /drift endpoint
"""

import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from src.deployment.config import EXPORT_DIR, MODEL_NAME, PRODUCTION_ALIAS

# Where the training run logs the baseline (save_baseline_statistics in src/training/drift_baseline.py)
BASELINE_ARTIFACT = "drift_detection/drift_baseline_stats.json"


def export_production_model(client: MlflowClient, export_dir: Path = EXPORT_DIR, model_name: str = MODEL_NAME) -> Path:
    mv = client.get_model_version_by_alias(model_name, PRODUCTION_ALIAS)
    model_uri = f"models:/{model_name}@{PRODUCTION_ALIAS}"
    print(f"Exporting {model_name} version {mv.version} to {export_dir}")

    # Download next to the target first, so a failed download doesn't wipe the old export
    export_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=export_dir.parent) as tmp:
        downloaded = mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=tmp)
        if export_dir.exists():
            shutil.rmtree(export_dir)
        shutil.move(downloaded, export_dir)

    run = client.get_run(mv.run_id)
    trained_at = datetime.fromtimestamp(run.info.start_time / 1000, tz=timezone.utc).isoformat()
    version_info = {"model_version": str(mv.version), "trained_at": trained_at}
    (export_dir / "version.json").write_text(json.dumps(version_info, indent=2))
    print(f"Version info: {version_info}")

    try:
        mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path=BASELINE_ARTIFACT, dst_path=str(export_dir))
        shutil.move(export_dir / BASELINE_ARTIFACT, export_dir / "drift_baseline_stats.json")
        shutil.rmtree(export_dir / "drift_detection")
    except (MlflowException, OSError) as e:
        print(f"Warning: no drift baseline found for run {mv.run_id}, /drift will report no drift: {e}")

    return export_dir
