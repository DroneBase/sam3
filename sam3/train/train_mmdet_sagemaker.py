#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "dotenv==0.9.9",
#   "opencv-python==4.11.0.86",
#   "numpy==1.26.4",
#   "sagemaker==2.253.1",
#   "pandas==2.3.3",
#   "requests==2.32.5",
#   "pillow==12.0.0",
#   "openpyxl==3.1.5",
#   "cloudpickle==3.1.1",
# ]
# ///

import argparse
import logging
import os
from pathlib import Path

import dotenv
import sagemaker
from sagemaker.remote_function import remote

dotenv.load_dotenv()

PROJECT_NAME = "turbine-crack-detection-mmdet"
WANDB_API_KEY = os.environ["WANDB_API_KEY"]
INSTANCE_TYPE = os.environ.get("INSTANCE_TYPE", "ml.g5.xlarge")
MAX_RUNTIME_SECONDS = int(os.environ.get("MAX_RUNTIME_SECONDS", 24 * 60 * 60))  # Default to 24 hours
INSTANCE_ATTRS = {
    "ml.g4dn.xlarge": {"gpu_count": 1, "gpu_type": "NVIDIA T4"},
    "ml.g4dn.2xlarge": {"gpu_count": 1, "gpu_type": "NVIDIA T4"},
    "ml.g4dn.4xlarge": {"gpu_count": 1, "gpu_type": "NVIDIA T4"},
    "ml.g4dn.8xlarge": {"gpu_count": 1, "gpu_type": "NVIDIA T4"},
    "ml.g4dn.16xlarge": {"gpu_count": 1, "gpu_type": "NVIDIA T4"},
    "ml.g4dn.12xlarge": {"gpu_count": 4, "gpu_type": "NVIDIA T4"},
    "ml.g5.xlarge": {"gpu_count": 1, "gpu_type": "NVIDIA A10G"},
    "ml.g5.2xlarge": {"gpu_count": 1, "gpu_type": "NVIDIA A10G"},
    "ml.g5.4xlarge": {"gpu_count": 1, "gpu_type": "NVIDIA A10G"},
    "ml.g5.8xlarge": {"gpu_count": 1, "gpu_type": "NVIDIA A10G"},
    "ml.g5.16xlarge": {"gpu_count": 1, "gpu_type": "NVIDIA A10G"},
    "ml.g5.12xlarge": {"gpu_count": 4, "gpu_type": "NVIDIA A10G"},
}
session = sagemaker.Session(default_bucket="zeitview-aiml-research-artifacts", default_bucket_prefix="sagemaker")

# NOTE: __file__ does not work with remote functions because it indexes host system path, so we use a relative path
RELATIVE_TRAIN_MMDET_PATH = (Path(__file__).parent / "train_mmdet.py").relative_to(Path.cwd())


@remote(
    sagemaker_session=session,
    instance_type=INSTANCE_TYPE,
    image_uri="757639335249.dkr.ecr.us-east-1.amazonaws.com/aiml/cuda:v1.18.1_py310",
    role="arn:aws:iam::757639335249:role/aiml/aiml-sagemaker-exec-us-east-1",
    dependencies=str(Path(__file__).parent / "sagemaker-requirements.txt"),
    environment_variables={"WANDB_API_KEY": WANDB_API_KEY, "RELATIVE_TRAIN_MMDET_PATH": str(RELATIVE_TRAIN_MMDET_PATH)},
    job_name_prefix=PROJECT_NAME,
    include_local_workdir=True,
    max_runtime_in_seconds=MAX_RUNTIME_SECONDS,
)
def main(config_content: str) -> None:
    from PIL import Image, ImageFile

    # Hack to get around PIL ImageFile loading truncated images
    # This is necessary for some datasets that may have corrupted images
    # See: https://github.com/DroneBase/ml-generalized-models/pull/223
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # Remove decompression warning
    Image.MAX_IMAGE_PIXELS = 400000000

    import mmengine
    from ml_dronebase_data_utils.s3 import sync_dir

    config = mmengine.Config.fromstring(config_content, file_format=".py")

    Path(config.train_dataloader.dataset.data_root).mkdir(parents=True, exist_ok=True)
    Path(config.val_dataloader.dataset.data_root).mkdir(parents=True, exist_ok=True)

    # Download data from s3 URIs
    sync_dir(config.train_uri, config.train_dataloader.dataset.data_root)
    sync_dir(config.val_uri, config.val_dataloader.dataset.data_root)

    # Download pre-trained model checkpoint if specified
    if hasattr(config, "load_from_uri") and config.load_from_uri is not None:
        from ml_dronebase_data_utils.s3 import download_file

        checkpoint_path = os.path.join("/tmp", "pretrained_checkpoint.pth")
        logging.info(f"Downloading pre-trained model checkpoint from {config.load_from_uri} to {checkpoint_path}")
        download_file(config.load_from_uri, checkpoint_path)
        config.load_from = checkpoint_path

    # Show some stats about the data for logging and debugging
    def get_dataset_stats(root_path, dataset_name):
        """Get comprehensive statistics about a dataset directory."""
        root = Path(root_path)

        # Count different file types
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
        annotation_extensions = {".xml", ".txt", ".json"}

        image_files = []
        annotation_files = []

        for ext in image_extensions:
            image_files.extend(list(root.rglob(f"*{ext}")))
            image_files.extend(list(root.rglob(f"*{ext.upper()}")))

        for ext in annotation_extensions:
            annotation_files.extend(list(root.rglob(f"*{ext}")))

        # Calculate total size
        total_size = sum(f.stat().st_size for f in root.rglob("*") if f.is_file())
        total_size_mb = total_size / (1024 * 1024)
        total_size_gb = total_size_mb / 1024

        # Get image dimensions if possible
        image_dims = []
        sample_size = min(100, len(image_files))  # Sample first 100 images for efficiency
        for img_file in image_files[:sample_size]:
            try:
                with Image.open(img_file) as img:
                    image_dims.append(img.size)
            except Exception:
                continue

        stats = {
            "dataset_name": dataset_name,
            "root_path": str(root),
            "total_files": len(list(root.rglob("*"))),
            "image_count": len(image_files),
            "annotation_count": len(annotation_files),
            "xml_annotations": len([f for f in annotation_files if f.suffix.lower() == ".xml"]),
            "total_size_mb": round(total_size_mb, 2),
            "total_size_gb": round(total_size_gb, 2),
            "sample_image_count": len(image_dims),
            "avg_image_width": round(sum(w for w, h in image_dims) / len(image_dims), 2) if image_dims else 0,
            "avg_image_height": round(sum(h for w, h in image_dims) / len(image_dims), 2) if image_dims else 0,
        }

        return stats

    # Get and log statistics for both datasets
    train_stats = get_dataset_stats(config.train_dataloader.dataset.data_root, "Training")
    val_stats = get_dataset_stats(config.val_dataloader.dataset.data_root, "Validation")

    logging.info("=" * 80)
    logging.info("DATASET STATISTICS")
    logging.info("=" * 80)

    for stats in [train_stats, val_stats]:
        logging.info(f"\n{stats['dataset_name']} Dataset:")
        logging.info(f"  Root Path: {stats['root_path']}")
        logging.info(f"  Total Files: {stats['total_files']:,}")
        logging.info(f"  Images: {stats['image_count']:,}")
        logging.info(f"  Annotations: {stats['annotation_count']:,}")
        logging.info(f"  XML Annotations: {stats['xml_annotations']:,}")
        logging.info(f"  Dataset Size: {stats['total_size_mb']:,} MB ({stats['total_size_gb']:.2f} GB)")
        if stats["sample_image_count"] > 0:
            logging.info(f"  Average Image Dimensions: {stats['avg_image_width']:.0f} x {stats['avg_image_height']:.0f} pixels")
            logging.info(f"  (Based on sample of {stats['sample_image_count']} images)")

    total_images = train_stats["image_count"] + val_stats["image_count"]
    total_annotations = train_stats["annotation_count"] + val_stats["annotation_count"]
    total_size_gb = train_stats["total_size_gb"] + val_stats["total_size_gb"]

    logging.info("\nCombined Dataset Summary:")
    logging.info(f"  Total Images: {total_images:,}")
    logging.info(f"  Total Annotations: {total_annotations:,}")
    logging.info(f"  Total Size: {total_size_gb:.2f} GB")
    if total_images > 0:
        logging.info(
            f"  Train/Val Split: {train_stats['image_count']:,} / {val_stats['image_count']:,} ({train_stats['image_count'] / total_images * 100:.1f}% / {val_stats['image_count'] / total_images * 100:.1f}%)"
        )
    logging.info("=" * 80)

    logging.info("Starting Training")
    train_mmdet_path = RELATIVE_TRAIN_MMDET_PATH
    config_path = "config.py"
    config.dump(config_path)
    original_config_path = "config_original.py"
    with open(original_config_path, "w") as f:
        f.write(config_content)
    if INSTANCE_ATTRS[INSTANCE_TYPE]["gpu_count"] == 1:
        os.system(f"python {train_mmdet_path} {config_path}")
    else:
        os.system(
            f"python -m torch.distributed.launch --nnodes=1 --node_rank=0 --master_addr='127.0.0.1' --nproc_per_node={INSTANCE_ATTRS[INSTANCE_TYPE]['gpu_count']} --master_port=29500 {train_mmdet_path} {config_path} --launcher pytorch"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to mmdet config.py file")
    args = parser.parse_args()

    # Read the config file
    with open(args.config) as f:
        config_content = f.read()

    main(config_content)