"""Download the vial images and load them as (image_path, label) pairs.

The metadata csvs are tracked with DVC, so run `dvc pull` first. Then
`python -m src.training.data` downloads the images listed in them.
"""

import csv
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
import pandas as pd
from botocore import UNSIGNED
from botocore.config import Config

from src.training.config import (
    IMAGE_DATA_DIR,
    METADATA_PATH,
    S3_BUCKET,
    S3_ENDPOINT,
    TEST_DATA_DIR,
    TEST_METADATA_PATH,
)

logger = logging.getLogger(__name__)


def _s3_client():
    """S3 client for the public bucket (unsigned requests, no credentials)."""
    return boto3.client(
        "s3",
        endpoint_url=S3_ENDPOINT,
        aws_access_key_id="",
        aws_secret_access_key="",
        config=Config(signature_version=UNSIGNED),
    )


def _download_file(s3_client, key, output_path):
    """Download one file. Returns "downloaded", "skipped" or "failed"."""
    if output_path.exists():
        return "skipped"
    try:
        s3_client.download_file(S3_BUCKET, key, str(output_path))
        return "downloaded"
    except Exception as e:
        logger.warning("Failed to download %s: %s", key, e)
        return "failed"


def download_files_from_metadata(metadata_path, output_dir, prefix=""):
    """Download every file listed in a metadata csv into output_dir."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(metadata_path) as f:
        filenames = [row["filename"] for row in csv.DictReader(f)]

    s3_client = _s3_client()
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(
                lambda name: _download_file(s3_client, f"{prefix}{name}", output_dir / name),
                filenames,
            )
        )

    counts = {status: results.count(status) for status in ("downloaded", "skipped", "failed")}
    logger.info(
        "Downloaded %d images, skipped %d existing, failed %d",
        counts["downloaded"], counts["skipped"], counts["failed"],
    )
    return counts


def download_data():
    """Download the training and test images from the object store."""
    download_files_from_metadata(METADATA_PATH, IMAGE_DATA_DIR, "images/raw/")
    download_files_from_metadata(TEST_METADATA_PATH, TEST_DATA_DIR, "test/")


def load_image_dataset(metadata_path=METADATA_PATH, image_dir=IMAGE_DATA_DIR):
    """Return (image_path, label) pairs for every listed image that exists on disk."""
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found at {metadata_path} (run `dvc pull`)")

    metadata = pd.read_csv(metadata_path)
    image_dataset = []
    missing = 0
    for filename, label in zip(metadata["filename"], metadata["class"]):
        image_path = Path(image_dir) / filename
        if image_path.exists():
            image_dataset.append((image_path, int(label)))
        else:
            missing += 1

    logger.info("Loaded %d images with labels", len(image_dataset))
    if missing:
        logger.warning("%d image files listed in the metadata were not found", missing)
    return image_dataset


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    download_data()
