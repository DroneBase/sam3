"""
SAM3 SageMaker Fine-tuning Script for Solar Array Segmentation

This script launches SAM3 fine-tuning jobs on AWS SageMaker using the remote function
decorator. It downloads data from S3, runs the segmentation training, and uploads
results back to S3.

Usage:
    python sam3/train/fine_tune_sam3_sagemaker_seg.py --config sam3/train/configs/solar_array/solar_array_seg_finetune_sagemaker.yaml
"""

import argparse
import logging
import os
import random
from pathlib import Path

import dotenv
import sagemaker
from sagemaker.remote_function import remote

# Load .env from project root and from the script's directory
dotenv.load_dotenv()  # project root .env
dotenv.load_dotenv(Path(__file__).parent / ".env")  # sam3/train/.env

# ============================================================================
# Configuration
# ============================================================================
PROJECT_NAME = "sam3-solar-array-segmentation"

# API Keys
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "sam3-solar-array-segmentation")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "zeitview")  # Your WandB team/entity name

# HuggingFace Token - Required to download SAM3 model weights from facebook/sam3
# Get your token from: https://huggingface.co/settings/tokens
# You must accept the model license at: https://huggingface.co/facebook/sam3
HF_TOKEN = os.environ.get("HF_TOKEN", os.environ.get("HUGGING_FACE_HUB_TOKEN", ""))

print(f"HuggingFace Token found: {'yes' if HF_TOKEN else 'no'}")
print(HF_TOKEN)

# SageMaker Instance Configuration
INSTANCE_TYPE = os.environ.get("INSTANCE_TYPE", "ml.g5.16xlarge")
MAX_RUNTIME_SECONDS = int(os.environ.get("MAX_RUNTIME_SECONDS", 195 * 60 * 60))  # Default to 195 hours

INSTANCE_ATTRS = {
    "ml.g4dn.xlarge": {"gpu_count": 1, "gpu_type": "NVIDIA T4", "memory_gb": 16},
    "ml.g4dn.2xlarge": {"gpu_count": 1, "gpu_type": "NVIDIA T4", "memory_gb": 32},
    "ml.g4dn.4xlarge": {"gpu_count": 1, "gpu_type": "NVIDIA T4", "memory_gb": 64},
    "ml.g4dn.8xlarge": {"gpu_count": 1, "gpu_type": "NVIDIA T4", "memory_gb": 128},
    "ml.g4dn.16xlarge": {"gpu_count": 1, "gpu_type": "NVIDIA T4", "memory_gb": 256},
    "ml.g4dn.12xlarge": {"gpu_count": 4, "gpu_type": "NVIDIA T4", "memory_gb": 192},
    "ml.g5.xlarge": {"gpu_count": 1, "gpu_type": "NVIDIA A10G", "memory_gb": 24},
    "ml.g5.2xlarge": {"gpu_count": 1, "gpu_type": "NVIDIA A10G", "memory_gb": 32},
    "ml.g5.4xlarge": {"gpu_count": 1, "gpu_type": "NVIDIA A10G", "memory_gb": 64},
    "ml.g5.8xlarge": {"gpu_count": 1, "gpu_type": "NVIDIA A10G", "memory_gb": 128},
    "ml.g5.16xlarge": {"gpu_count": 1, "gpu_type": "NVIDIA A10G", "memory_gb": 256},
    "ml.g5.12xlarge": {"gpu_count": 4, "gpu_type": "NVIDIA A10G", "memory_gb": 192},
    "ml.g5.24xlarge": {"gpu_count": 4, "gpu_type": "NVIDIA A10G", "memory_gb": 384},
    "ml.g5.48xlarge": {"gpu_count": 8, "gpu_type": "NVIDIA A10G", "memory_gb": 768},
    "ml.p4d.24xlarge": {"gpu_count": 8, "gpu_type": "NVIDIA A100", "memory_gb": 320},
    "ml.p4de.24xlarge": {"gpu_count": 8, "gpu_type": "NVIDIA A100", "memory_gb": 640},
}

# SageMaker session configuration
session = sagemaker.Session(
    default_bucket="zeitview-aiml-research-artifacts",
    default_bucket_prefix="sagemaker/sam3"
)

# NOTE: __file__ does not work with remote functions because it indexes host system path
# so we use a relative path
RELATIVE_TRAIN_PATH = (Path(__file__).parent / "train.py").relative_to(Path.cwd())
PRE_EXEC_SCRIPT = Path(__file__).parent / "pre_exec.sh"


@remote(
    sagemaker_session=session,
    instance_type=INSTANCE_TYPE,
    image_uri="757639335249.dkr.ecr.us-east-1.amazonaws.com/aiml/cuda:v1.18.1_py310",
    role="arn:aws:iam::757639335249:role/aiml/aiml-sagemaker-exec-us-east-1",
    dependencies=str(Path(__file__).parent / "sam3-sagemaker-requirements.txt"),
    pre_execution_script=str(PRE_EXEC_SCRIPT),
    environment_variables={
        # WandB configuration
        "WANDB_API_KEY": WANDB_API_KEY,
        "WANDB_PROJECT": WANDB_PROJECT,
        "WANDB_ENTITY": WANDB_ENTITY,
        # HuggingFace configuration - needed to download SAM3 weights
        "HF_TOKEN": HF_TOKEN,
        "HUGGING_FACE_HUB_TOKEN": HF_TOKEN,  # Alternative env var name
        # Training configuration
        "RELATIVE_TRAIN_PATH": str(RELATIVE_TRAIN_PATH),
        "HYDRA_FULL_ERROR": "1",
        # CUDA memory optimization to reduce fragmentation on A10G
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    },
    job_name_prefix=PROJECT_NAME,
    include_local_workdir=True,
    max_runtime_in_seconds=MAX_RUNTIME_SECONDS,
)
def main(config_content: str, config_name: str) -> dict:
    """
    Main training function that runs on SageMaker for solar array segmentation.

    Args:
        config_content: YAML configuration file content as string
        config_name: Name of the config (used for logging)

    Returns:
        Dictionary containing training results and metrics
    """
    import importlib
    import subprocess
    import sys
    import tempfile
    import yaml
    from PIL import Image, ImageFile

    # =========================================================================
    # Setup workspace path - MUST be added for sam3 imports and Hydra config loading
    # =========================================================================
    workspace_dir = Path("/workspace/sagemaker_remote_function_workspace")
    workspace_str = str(workspace_dir)
    if workspace_str not in sys.path:
        sys.path.insert(0, workspace_str)
        print(f"Added {workspace_str} to sys.path")

    # Hack to get around PIL ImageFile loading truncated images
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = 400000000

    # Parse the config to extract S3 URIs
    config = yaml.safe_load(config_content)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)

    # =========================================================================
    # HuggingFace Authentication
    # Required to download SAM3 model weights from facebook/sam3
    # =========================================================================
    hf_token = os.environ.get("HF_TOKEN", os.environ.get("HUGGING_FACE_HUB_TOKEN", ""))
    if hf_token:
        try:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
            logger.info("Successfully authenticated with HuggingFace Hub")
        except Exception as e:
            logger.warning(f"Failed to authenticate with HuggingFace Hub: {e}")
            logger.warning("Model download may fail if SAM3 weights require authentication")
    else:
        logger.warning("HF_TOKEN not set - SAM3 model weights may fail to download")
        logger.warning("Set HF_TOKEN environment variable with your HuggingFace token")
        logger.warning("Get token from: https://huggingface.co/settings/tokens")
        logger.warning("Accept license at: https://huggingface.co/facebook/sam3")

    logger.info("=" * 80)
    logger.info("SAM3 SAGEMAKER TRAINING JOB - SOLAR ARRAY SEGMENTATION")
    logger.info("=" * 80)
    logger.info(f"Instance Type: {INSTANCE_TYPE}")
    logger.info(f"GPU Count: {INSTANCE_ATTRS.get(INSTANCE_TYPE, {}).get('gpu_count', 'unknown')}")
    logger.info(f"GPU Type: {INSTANCE_ATTRS.get(INSTANCE_TYPE, {}).get('gpu_type', 'unknown')}")
    logger.info(f"Task: Segmentation (enable_segmentation=True)")
    logger.info("=" * 80)

    # WandB setup
    # NOTE: We do NOT call wandb.init() here because training runs as a
    # subprocess (subprocess.run).  WandB runs are per-process, so the
    # trainer's WandBLogger (instantiated via Hydra config) will create and
    # own the real WandB run inside the subprocess.  We only store the
    # config values so we can do a short-lived post-training artifact upload.
    wandb_config = config.get("wandb", {})
    wandb_enabled = bool(WANDB_API_KEY) and wandb_config.get("enabled", True)
    if wandb_enabled:
        logger.info("WandB is enabled — training metrics will be logged by the trainer subprocess")
    else:
        logger.info("WandB logging disabled (no API key or disabled in config)")

    # Download data from S3 if URIs are specified
    s3_config = config.get("sagemaker", {})

    if s3_config.get("train_data_uri"):
        from ml_dronebase_data_utils.s3 import sync_dir

        local_train_path = config["paths"]["dataset_root"] + "/train"
        logger.info(f"Downloading training data from {s3_config['train_data_uri']} to {local_train_path}")
        Path(local_train_path).mkdir(parents=True, exist_ok=True)
        sync_dir(s3_config["train_data_uri"], local_train_path)

    if s3_config.get("val_data_uri"):
        from ml_dronebase_data_utils.s3 import sync_dir

        local_val_path = config["paths"]["dataset_root"] + "/test"
        logger.info(f"Downloading validation data from {s3_config['val_data_uri']} to {local_val_path}")
        Path(local_val_path).mkdir(parents=True, exist_ok=True)
        sync_dir(s3_config["val_data_uri"], local_val_path)

    # Download pretrained checkpoint if specified
    if s3_config.get("pretrained_checkpoint_uri"):
        from ml_dronebase_data_utils.s3 import download_file

        checkpoint_path = "/tmp/pretrained_checkpoint.pth"
        logger.info(f"Downloading pretrained checkpoint from {s3_config['pretrained_checkpoint_uri']}")
        download_file(s3_config["pretrained_checkpoint_uri"], checkpoint_path)
        # Update config to use the downloaded checkpoint
        if "trainer" in config and "checkpoint" in config["trainer"]:
            config["trainer"]["checkpoint"]["load_from"] = checkpoint_path

    # Log dataset statistics
    log_dataset_stats(config["paths"]["dataset_root"], logger)

    # Verify segmentation masks are present in annotations
    verify_segmentation_annotations(config["paths"]["dataset_root"], logger)

    # Download BPE tokenizer file from S3 if specified
    bpe_filename = "bpe_simple_vocab_16e6.txt.gz"

    # Define potential locations for the BPE file
    bpe_search_locations = [
        workspace_dir / "sam3" / "assets" / bpe_filename,
        Path("/tmp") / bpe_filename,
        Path.cwd() / "sam3" / "assets" / bpe_filename,
    ]

    if s3_config.get("bpe_s3_uri"):
        from ml_dronebase_data_utils.s3 import download_file

        bpe_local_path = f"/tmp/{bpe_filename}"
        logger.info(f"Downloading BPE tokenizer from {s3_config['bpe_s3_uri']} to {bpe_local_path}")
        download_file(s3_config["bpe_s3_uri"], bpe_local_path)
        config["paths"]["bpe_path"] = bpe_local_path
        logger.info(f"Updated bpe_path to: {bpe_local_path}")
    elif "paths" in config and "bpe_path" in config["paths"]:
        # Fix paths for SageMaker environment
        bpe_path = config["paths"]["bpe_path"]
        logger.info(f"Original bpe_path from config: '{bpe_path}'")

        # Add the config-specified path to search locations
        if os.path.isabs(bpe_path):
            bpe_search_locations.insert(0, Path(bpe_path))
        else:
            bpe_search_locations.insert(0, workspace_dir / bpe_path)

        # Find the BPE file from search locations
        found_bpe_path = None
        for search_path in bpe_search_locations:
            logger.info(f"Checking for BPE file at: {search_path}")
            if search_path.exists():
                found_bpe_path = str(search_path)
                logger.info(f"Found BPE file at: {found_bpe_path}")
                break

        # Try pkg_resources as fallback (if sam3 is installed as package)
        if not found_bpe_path:
            try:
                import pkg_resources
                pkg_bpe_path = pkg_resources.resource_filename("sam3", f"assets/{bpe_filename}")
                logger.info(f"Checking pkg_resources path: {pkg_bpe_path}")
                if os.path.exists(pkg_bpe_path):
                    found_bpe_path = pkg_bpe_path
                    logger.info(f"Found BPE file via pkg_resources at: {found_bpe_path}")
            except Exception as e:
                logger.warning(f"pkg_resources lookup failed: {e}")

        # Try importlib.resources as another fallback
        if not found_bpe_path:
            try:
                import importlib.resources
                with importlib.resources.files("sam3.assets").joinpath(bpe_filename) as bpe_resource:
                    if bpe_resource.exists():
                        found_bpe_path = str(bpe_resource)
                        logger.info(f"Found BPE file via importlib.resources at: {found_bpe_path}")
            except Exception as e:
                logger.warning(f"importlib.resources lookup failed: {e}")

        if found_bpe_path:
            config["paths"]["bpe_path"] = found_bpe_path
            logger.info(f"Set bpe_path to: {found_bpe_path}")
        else:
            # List workspace contents for debugging
            logger.error(f"BPE file '{bpe_filename}' not found in any search location!")
            logger.error(f"Search locations tried: {[str(p) for p in bpe_search_locations]}")
            if workspace_dir.exists():
                logger.error(f"Workspace root contents: {[p.name for p in workspace_dir.iterdir()]}")
                sam3_dir = workspace_dir / "sam3"
                if sam3_dir.exists():
                    logger.error(f"sam3/ contents: {[p.name for p in sam3_dir.iterdir()]}")
                    assets_dir = sam3_dir / "assets"
                    if assets_dir.exists():
                        logger.error(f"sam3/assets/ contents: {[p.name for p in assets_dir.iterdir()]}")
                    else:
                        logger.error("sam3/assets/ directory does not exist!")
            raise FileNotFoundError(
                f"BPE tokenizer file '{bpe_filename}' not found. "
                f"Searched locations: {[str(p) for p in bpe_search_locations]}. "
                "Please ensure the sam3/assets directory is included in the SageMaker workspace."
            )

    # Save the config to a temp directory for Hydra's initialize_config_dir
    # train.py supports --config-path and --config-name for SageMaker usage
    temp_config_dir = "/tmp/sam3_config"
    Path(temp_config_dir).mkdir(parents=True, exist_ok=True)
    temp_config_name = "sagemaker_config"  # Without .yaml extension
    temp_config_path = f"{temp_config_dir}/{temp_config_name}.yaml"

    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f)

    logger.info(f"Config saved to: {temp_config_path}")

    # Install sam3 package now that torch is upgraded
    # (pre_exec.sh runs before pip install, so sam3 import would fail with old torch)
    # We MUST install via pip (not just add to sys.path) for Hydra's pkg:// URLs to work
    logger.info("Setting up sam3 package for Hydra config loading...")

    # Check if pyproject.toml exists - SageMaker sometimes doesn't copy it
    pyproject_path = workspace_dir / "pyproject.toml"
    if not pyproject_path.exists():
        logger.warning(f"pyproject.toml not found at {pyproject_path}, creating minimal version...")
        # List workspace contents for debugging
        if workspace_dir.exists():
            logger.info(f"Workspace contents: {[p.name for p in workspace_dir.iterdir()]}")
        # Create minimal pyproject.toml for pip install
        minimal_pyproject = '''[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "sam3"
version = "0.1.0"
description = "SAM3 (Segment Anything Model 3) implementation"
requires-python = ">=3.8"

[tool.setuptools.packages.find]
where = ["."]
include = ["sam3*"]

[tool.setuptools.package-data]
"*" = ["*.yaml", "*.yml", "*.json", "*.txt", "*.gz"]
'''
        pyproject_path.write_text(minimal_pyproject)
        logger.info(f"Created minimal pyproject.toml at {pyproject_path}")

    # Ensure __init__.py files exist in all config directories for Hydra's pkg://
    # resource loading. Hydra uses importlib.resources which requires __init__.py
    # in Python < 3.11
    sam3_train_dir = workspace_dir / "sam3" / "train"
    configs_dir = sam3_train_dir / "configs"

    # Create __init__.py in configs directories if they don't exist
    for dir_path in [configs_dir]:
        if dir_path.exists():
            init_file = dir_path / "__init__.py"
            if not init_file.exists():
                init_file.write_text("# Auto-generated for Hydra config loading\n")
                logger.info(f"Created {init_file}")
            # Also create in subdirectories
            for subdir in dir_path.iterdir():
                if subdir.is_dir():
                    sub_init = subdir / "__init__.py"
                    if not sub_init.exists():
                        sub_init.write_text("# Auto-generated for Hydra config loading\n")
                        logger.info(f"Created {sub_init}")

    # Verify sam3 can be imported
    try:
        # Clear any cached imports first
        for mod_name in list(sys.modules.keys()):
            if mod_name == 'sam3' or mod_name.startswith('sam3.'):
                del sys.modules[mod_name]
        importlib.invalidate_caches()

        import sam3
        import sam3.train
        logger.info(f"sam3 package available at: {sam3.__file__}")
        logger.info(f"sam3.train package available at: {sam3.train.__file__}")

        # Verify configs directory exists
        if configs_dir.exists():
            config_files = list(configs_dir.rglob("*.yaml"))
            logger.info(f"Found {len(config_files)} config files in {configs_dir}")
            for cf in config_files[:5]:  # Show first 5
                logger.info(f"  - {cf.relative_to(workspace_dir)}")
    except ImportError as e:
        logger.error(f"sam3 import failed: {e}")
        logger.error(f"sys.path: {sys.path[:5]}...")  # Show first 5 paths

    logger.info("Starting SAM3 Segmentation Training...")

    # Determine number of GPUs
    num_gpus = INSTANCE_ATTRS.get(INSTANCE_TYPE, {}).get("gpu_count", 1)

    # Run the training script using --config-path and --config-name for Hydra
    if num_gpus == 1:
        # Single GPU training
        cmd = [
            sys.executable,
            str(RELATIVE_TRAIN_PATH),
            "--config-path", temp_config_dir,
            "--config-name", temp_config_name,
            "--use-cluster", "0"
        ]
    else:
        # Multi-GPU distributed training
        cmd = [
            sys.executable, "-m", "torch.distributed.launch",
            "--nnodes=1",
            "--node_rank=0",
            "--master_addr=127.0.0.1",
            f"--nproc_per_node={num_gpus}",
            "--master_port=29500",
            str(RELATIVE_TRAIN_PATH),
            "--config-path", temp_config_dir,
            "--config-name", temp_config_name,
            "--use-cluster", "0",
            f"--num-gpus", str(num_gpus)
        ]

    logger.info(f"Running command: {' '.join(cmd)}")

    # Set up environment with PYTHONPATH to include workspace
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    workspace_path = str(Path("/workspace/sagemaker_remote_function_workspace"))
    if current_pythonpath:
        env["PYTHONPATH"] = f"{workspace_path}:{current_pythonpath}"
    else:
        env["PYTHONPATH"] = workspace_path
    logger.info(f"PYTHONPATH set to: {env['PYTHONPATH']}")

    # Set SAM3_CONFIG_DIR for directory-based Hydra config loading
    # This is required in SageMaker where package resources aren't available
    config_dir = str(workspace_dir / "sam3" / "train" / "configs")
    env["SAM3_CONFIG_DIR"] = config_dir
    logger.info(f"SAM3_CONFIG_DIR set to: {config_dir}")

    # Ensure WandB env vars are available in the training subprocess
    for wandb_var in ["WANDB_API_KEY", "WANDB_PROJECT", "WANDB_ENTITY"]:
        val = os.environ.get(wandb_var, "")
        if val:
            env[wandb_var] = val

    # Run the training
    result = subprocess.run(cmd, capture_output=False, text=True, env=env)

    if result.returncode != 0:
        logger.error(f"Training failed with return code: {result.returncode}")
        raise RuntimeError(f"Training failed with return code: {result.returncode}")

    logger.info("Segmentation training completed successfully!")

    # Upload ONLY the best checkpoint to WandB as an artifact.
    # The trainer saves best-metric checkpoints as e.g.
    #   detection_segm_AP_0.50_0.95.pt  (segmentation AP)
    #   detection_bbox_AP_0.50_0.95.pt  (bbox AP)
    # We prefer the segm one; fall back to bbox; ignore periodic checkpoints.
    if wandb_enabled:
        try:
            import wandb

            experiment_log_dir = config["paths"]["experiment_log_dir"]
            checkpoint_dir = Path(experiment_log_dir) / "checkpoints"

            best_ckpt = None
            if checkpoint_dir.exists():
                # Prioritized search: prefer segm AP checkpoint over bbox AP
                for pattern in ["detection_segm*.pt", "detection_bbox*.pt"]:
                    candidates = sorted(checkpoint_dir.glob(pattern))
                    if candidates:
                        best_ckpt = candidates[-1]  # latest (only one expected per meter)
                        break

            if best_ckpt is not None:
                wandb_project = wandb_config.get("project", WANDB_PROJECT)
                wandb_entity = wandb_config.get("entity", WANDB_ENTITY)

                # Open a short-lived run just for artifact upload
                artifact_run = wandb.init(
                    project=wandb_project,
                    entity=wandb_entity,
                    name=f"{config_name}_artifact_upload",
                    job_type="artifact-upload",
                    tags=["artifact-upload"],
                )
                artifact = wandb.Artifact(
                    name=f"{config_name.replace('/', '_')}_best_model",
                    type="model",
                    metadata={
                        "config_name": config_name,
                        "instance_type": INSTANCE_TYPE,
                        "task": "segmentation",
                        "checkpoint_file": best_ckpt.name,
                    },
                )
                artifact.add_file(str(best_ckpt))
                artifact_run.log_artifact(artifact, aliases=["best"])
                logger.info(f"Uploaded best checkpoint to WandB: {best_ckpt.name}")
                wandb.finish()
                logger.info("WandB artifact upload run finished")
            else:
                logger.info(
                    "No best-metric checkpoint found in %s — skipping WandB artifact upload. "
                    "Only periodic checkpoints exist (checkpoint.pt, checkpoint_N.pt). "
                    "Ensure save_best_meters is set in the YAML config.",
                    checkpoint_dir,
                )
        except Exception as e:
            logger.warning(f"Failed to upload artifact to WandB: {e}")

    # Upload results to S3 if output URI is specified
    if s3_config.get("output_uri"):
        from ml_dronebase_data_utils.s3 import sync_dir

        experiment_log_dir = config["paths"]["experiment_log_dir"]
        logger.info(f"Uploading results from {experiment_log_dir} to {s3_config['output_uri']}")
        sync_dir(experiment_log_dir, s3_config["output_uri"])

    return {
        "status": "success",
        "config_name": config_name,
        "instance_type": INSTANCE_TYPE,
        "num_gpus": num_gpus,
        "task": "segmentation",
        "wandb_enabled": wandb_enabled,
    }


def verify_segmentation_annotations(root_path: str, logger: logging.Logger) -> None:
    """Verify that COCO annotations contain segmentation masks (required for seg training)."""
    import json

    root = Path(root_path)

    for split in ["train", "test"]:
        ann_file = root / split / "_annotations.coco.json"
        if not ann_file.exists():
            logger.warning(f"Annotation file not found: {ann_file}")
            continue

        try:
            with open(ann_file, "r") as f:
                coco_data = json.load(f)

            annotations = coco_data.get("annotations", [])
            total_anns = len(annotations)

            if total_anns == 0:
                logger.warning(f"No annotations found in {ann_file}")
                continue

            # Check how many annotations have segmentation masks
            anns_with_seg = sum(
                1 for ann in annotations
                if ann.get("segmentation") and len(ann["segmentation"]) > 0
            )
            anns_with_area = sum(1 for ann in annotations if ann.get("area", 0) > 0)

            categories = coco_data.get("categories", [])
            images = coco_data.get("images", [])

            logger.info(f"  [{split}] Annotation Statistics:")
            logger.info(f"    Images: {len(images):,}")
            logger.info(f"    Annotations: {total_anns:,}")
            logger.info(f"    Categories: {len(categories)} - {[c.get('name', c.get('id')) for c in categories]}")
            logger.info(f"    Annotations with segmentation: {anns_with_seg:,} ({anns_with_seg/total_anns*100:.1f}%)")
            logger.info(f"    Annotations with area > 0: {anns_with_area:,} ({anns_with_area/total_anns*100:.1f}%)")

            if anns_with_seg == 0:
                logger.error(
                    f"  [{split}] WARNING: No segmentation masks found in annotations! "
                    "Segmentation training requires polygon/RLE masks in COCO format. "
                    "Please ensure your annotations include 'segmentation' fields."
                )
            elif anns_with_seg < total_anns:
                logger.warning(
                    f"  [{split}] Only {anns_with_seg}/{total_anns} annotations have segmentation masks. "
                    "Some annotations may be skipped during segmentation training."
                )
            else:
                logger.info(f"  [{split}] All annotations have segmentation masks ✓")

        except Exception as e:
            logger.warning(f"Failed to verify annotations in {ann_file}: {e}")


def log_dataset_stats(root_path: str, logger: logging.Logger) -> None:
    """Log comprehensive statistics about the dataset."""
    from PIL import Image

    root = Path(root_path)

    if not root.exists():
        logger.warning(f"Dataset root {root_path} does not exist yet")
        return

    # Count different file types
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    annotation_extensions = {".json"}

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

    # Get image dimensions sample
    image_dims = []
    sample_size = min(100, len(image_files))
    for img_file in image_files[:sample_size]:
        try:
            with Image.open(img_file) as img:
                image_dims.append(img.size)
        except Exception:
            continue

    logger.info("=" * 80)
    logger.info("DATASET STATISTICS")
    logger.info("=" * 80)
    logger.info(f"  Root Path: {root_path}")
    logger.info(f"  Total Files: {len(list(root.rglob('*'))):,}")
    logger.info(f"  Images: {len(image_files):,}")
    logger.info(f"  Annotations (JSON): {len(annotation_files):,}")
    logger.info(f"  Dataset Size: {total_size_mb:,.2f} MB ({total_size_gb:.2f} GB)")

    if image_dims:
        avg_width = sum(w for w, h in image_dims) / len(image_dims)
        avg_height = sum(h for w, h in image_dims) / len(image_dims)
        logger.info(f"  Average Image Dimensions: {avg_width:.0f} x {avg_height:.0f} pixels")
        logger.info(f"  (Based on sample of {len(image_dims)} images)")

    # Check train/test split
    train_images = list((root / "train" / "images").rglob("*")) if (root / "train" / "images").exists() else []
    test_images = list((root / "test" / "images").rglob("*")) if (root / "test" / "images").exists() else []

    if train_images or test_images:
        train_count = len([f for f in train_images if f.suffix.lower() in image_extensions])
        test_count = len([f for f in test_images if f.suffix.lower() in image_extensions])
        total = train_count + test_count
        if total > 0:
            logger.info(f"  Train/Test Split: {train_count:,} / {test_count:,} "
                        f"({train_count/total*100:.1f}% / {test_count/total*100:.1f}%)")

    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch SAM3 segmentation fine-tuning job on AWS SageMaker"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to SAM3 YAML config file (e.g., sam3/train/configs/solar_array/solar_array_seg_finetune_sagemaker.yaml)"
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default=None,
        help="Config name for Hydra (default: derived from config path)"
    )
    args = parser.parse_args()

    # Read the config file
    with open(args.config, 'r') as f:
        config_content = f.read()

    # Derive config name from path if not provided
    if args.config_name is None:
        # Convert path like "sam3/train/configs/solar_array/solar_array_seg_finetune_sagemaker.yaml"
        # to "solar_array/solar_array_seg_finetune_sagemaker" (relative to configs directory)
        config_path = Path(args.config)
        if "configs" in config_path.parts:
            configs_idx = config_path.parts.index("configs")
            # Get parts AFTER "configs", not including "configs" itself
            config_name = "/".join(config_path.parts[configs_idx + 1:])
            config_name = config_name.rsplit(".", 1)[0]  # Remove .yaml extension
        else:
            config_name = config_path.stem
    else:
        config_name = args.config_name

    print(f"Launching SAM3 SageMaker segmentation training job with config: {args.config}")
    print(f"Config name: {config_name}")
    print(f"Instance type: {INSTANCE_TYPE}")

    result = main(config_content, config_name)
    print(f"Job completed with result: {result}")
