"""
SAM3 SageMaker Inference Script for Solar Array Segmentation

This script launches SAM3 inference jobs on AWS SageMaker using the remote function
decorator. It downloads images and a fine-tuned checkpoint from S3 (or uses local paths),
runs inference with text prompts, and saves:
  - predictions.json  — COCO-style detection coordinates, scores, labels
  - masks/            — per-image binary segmentation mask PNGs
  - overlays/         — per-image overlay visualizations (image + masks + bboxes)

All outputs are uploaded to S3 (or saved locally).

Usage:
    # Basic (uses config defaults):
    python sam3/train/infer_sam3_sagemaker_segmentation.py \
        --config sam3/train/configs/solar_array/solar_array_seg_inference_sagemaker.yaml

    # Override checkpoint and images:
    python sam3/train/infer_sam3_sagemaker_segmentation.py \
        --config sam3/train/configs/solar_array/solar_array_seg_inference_sagemaker.yaml \
        --checkpoint-uri s3://zeitview-aiml-research-artifacts/solar/ir_hotspot_and_diode_failure_detection_v1/datasets/solar_array_mini_v0/artifacts_3/checkpoints/checkpoint_30.pt \
        --images-uri s3://zeitview-aiml-research-artifacts/solar/ir_hotspot_and_diode_failure_detection_v1/datasets/anomaly_panel_offline_albatross_tiled_norm_v0_vlm/valid/ \
        --output-uri s3://zeitview-aiml-research-artifacts/solar/ir_hotspot_and_diode_failure_detection_v1/datasets/solar_array_mini_v0/infer_2/

    # Run locally (no SageMaker, for testing):
    python sam3/train/infer_sam3_sagemaker_segmentation.py \
        --config sam3/train/configs/solar_array/solar_array_seg_inference_sagemaker.yaml \
        --local
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
PROJECT_NAME = "sam3-solar-array-seg-infer"

# API Keys
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "sam3-solar-array-segmentation")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "zeitview")

# HuggingFace Token — Required to download SAM3 model weights from facebook/sam3
HF_TOKEN = os.environ.get("HF_TOKEN", os.environ.get("HUGGING_FACE_HUB_TOKEN", ""))

print(f"HuggingFace Token found: {'yes' if HF_TOKEN else 'no'}")

# SageMaker Instance Configuration
# Inference is less GPU-intensive; a single A10G is usually sufficient
INSTANCE_TYPE = os.environ.get("INSTANCE_TYPE", "ml.g5.2xlarge")
MAX_RUNTIME_SECONDS = int(os.environ.get("MAX_RUNTIME_SECONDS", 12 * 60 * 60))  # Default 12 hours

INSTANCE_ATTRS = {
    "ml.g4dn.xlarge": {"gpu_count": 1, "gpu_type": "NVIDIA T4", "memory_gb": 16},
    "ml.g4dn.2xlarge": {"gpu_count": 1, "gpu_type": "NVIDIA T4", "memory_gb": 32},
    "ml.g4dn.4xlarge": {"gpu_count": 1, "gpu_type": "NVIDIA T4", "memory_gb": 64},
    "ml.g5.xlarge": {"gpu_count": 1, "gpu_type": "NVIDIA A10G", "memory_gb": 24},
    "ml.g5.2xlarge": {"gpu_count": 1, "gpu_type": "NVIDIA A10G", "memory_gb": 32},
    "ml.g5.4xlarge": {"gpu_count": 1, "gpu_type": "NVIDIA A10G", "memory_gb": 64},
    "ml.g5.8xlarge": {"gpu_count": 1, "gpu_type": "NVIDIA A10G", "memory_gb": 128},
    "ml.g5.16xlarge": {"gpu_count": 1, "gpu_type": "NVIDIA A10G", "memory_gb": 256},
    "ml.p4d.24xlarge": {"gpu_count": 8, "gpu_type": "NVIDIA A100", "memory_gb": 320},
}

# SageMaker session configuration
session = sagemaker.Session(
    default_bucket="zeitview-aiml-research-artifacts",
    default_bucket_prefix="sagemaker/sam3",
)

PRE_EXEC_SCRIPT = Path(__file__).parent / "pre_exec.sh"


# ============================================================================
# Helper: determine if a URI is S3
# ============================================================================
def _is_s3_uri(uri: str) -> bool:
    return uri is not None and uri.strip().startswith("s3://")


# ============================================================================
# SageMaker Remote Function
# ============================================================================
@remote(
    sagemaker_session=session,
    instance_type=INSTANCE_TYPE,
    image_uri="757639335249.dkr.ecr.us-east-1.amazonaws.com/aiml/cuda:v1.18.1_py310",
    role="arn:aws:iam::757639335249:role/aiml/aiml-sagemaker-exec-us-east-1",
    dependencies=str(Path(__file__).parent / "sam3-sagemaker-requirements.txt"),
    pre_execution_script=str(PRE_EXEC_SCRIPT),
    environment_variables={
        "WANDB_API_KEY": WANDB_API_KEY,
        "WANDB_PROJECT": WANDB_PROJECT,
        "WANDB_ENTITY": WANDB_ENTITY,
        "HF_TOKEN": HF_TOKEN,
        "HUGGING_FACE_HUB_TOKEN": HF_TOKEN,
        "HYDRA_FULL_ERROR": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    },
    job_name_prefix=PROJECT_NAME,
    include_local_workdir=True,
    max_runtime_in_seconds=MAX_RUNTIME_SECONDS,
)
def main(config_content: str, config_name: str) -> dict:
    """
    Main inference function that runs on SageMaker.

    Args:
        config_content: YAML configuration file content as string.
        config_name: Name of the config (used for logging).

    Returns:
        Dictionary with inference summary (num images, num detections, output path).
    """
    return _run_inference(config_content, config_name)


def _run_inference(config_content: str, config_name: str) -> dict:
    """
    Core inference logic — shared by both SageMaker remote and local modes.
    """
    import importlib
    import json
    import sys
    import time

    import numpy as np
    import torch
    import yaml
    from PIL import Image, ImageFile

    # =========================================================================
    # Setup workspace path (SageMaker)
    # =========================================================================
    workspace_dir = Path("/workspace/sagemaker_remote_function_workspace")
    workspace_str = str(workspace_dir)
    if workspace_dir.exists() and workspace_str not in sys.path:
        sys.path.insert(0, workspace_str)
        print(f"Added {workspace_str} to sys.path")

    ImageFile.LOAD_TRUNCATED_IMAGES = True
    Image.MAX_IMAGE_PIXELS = 400000000

    config = yaml.safe_load(config_content)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # =========================================================================
    # HuggingFace Authentication
    # =========================================================================
    hf_token = os.environ.get("HF_TOKEN", os.environ.get("HUGGING_FACE_HUB_TOKEN", ""))
    if hf_token:
        try:
            from huggingface_hub import login

            login(token=hf_token, add_to_git_credential=False)
            logger.info("Successfully authenticated with HuggingFace Hub")
        except Exception as e:
            logger.warning(f"Failed to authenticate with HuggingFace Hub: {e}")
    else:
        logger.warning("HF_TOKEN not set — SAM3 model weights may fail to download")

    logger.info("=" * 80)
    logger.info("SAM3 SAGEMAKER INFERENCE JOB — SOLAR ARRAY SEGMENTATION")
    logger.info("=" * 80)
    logger.info(f"Instance Type: {INSTANCE_TYPE}")
    logger.info(f"Config Name: {config_name}")
    logger.info("=" * 80)

    s3_config = config.get("sagemaker", {})
    paths_config = config.get("paths", {})
    infer_config = config.get("inference", {})

    # =========================================================================
    # 1. Download / resolve images
    # =========================================================================
    images_uri = s3_config.get("images_uri", "")
    images_dir = paths_config.get("images_dir", "/tmp/solar_array_inference/images")

    if _is_s3_uri(images_uri):
        from ml_dronebase_data_utils.s3 import sync_dir

        Path(images_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading images from {images_uri} to {images_dir}")
        sync_dir(images_uri, images_dir)
    elif images_uri and Path(images_uri).is_dir():
        images_dir = images_uri
        logger.info(f"Using local images directory: {images_dir}")
    else:
        logger.info(f"Using images_dir from paths config: {images_dir}")

    # =========================================================================
    # 2. Download / resolve checkpoint
    # =========================================================================
    checkpoint_uri = s3_config.get("pretrained_checkpoint_uri")
    checkpoint_path = paths_config.get("checkpoint_path")

    if _is_s3_uri(checkpoint_uri):
        from ml_dronebase_data_utils.s3 import download_file

        checkpoint_path = "/tmp/inference_checkpoint.pth"
        logger.info(f"Downloading checkpoint from {checkpoint_uri}")
        download_file(checkpoint_uri, checkpoint_path)
    elif checkpoint_uri and Path(checkpoint_uri).is_file():
        checkpoint_path = checkpoint_uri
    # else: checkpoint_path stays as-is from paths config (or None → HF download)

    if checkpoint_path:
        logger.info(f"Using checkpoint: {checkpoint_path}")
    else:
        logger.info("No checkpoint specified — will use default HuggingFace weights")

    # =========================================================================
    # 3. Resolve BPE tokenizer
    # =========================================================================
    bpe_path = _resolve_bpe_path(config, workspace_dir, logger)

    # =========================================================================
    # 4. Resolve output directory
    # =========================================================================
    output_dir = paths_config.get("output_dir", "/tmp/solar_array_inference/output")
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    masks_dir = os.path.join(output_dir, "masks")
    instance_masks_dir = os.path.join(output_dir, "instance_masks")
    overlays_dir = os.path.join(output_dir, "overlays")

    save_masks = infer_config.get("save_masks", True)
    save_instance_masks = infer_config.get("save_instance_masks", False)
    save_overlays = infer_config.get("save_overlays", True)

    if save_masks:
        Path(masks_dir).mkdir(parents=True, exist_ok=True)
    if save_instance_masks:
        Path(instance_masks_dir).mkdir(parents=True, exist_ok=True)
    if save_overlays:
        Path(overlays_dir).mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # 5. Collect image files
    # =========================================================================
    image_extensions = set(infer_config.get(
        "image_extensions",
        [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"],
    ))
    image_files = sorted([
        p for p in Path(images_dir).rglob("*")
        if p.suffix.lower() in image_extensions and p.is_file()
    ])

    if not image_files:
        logger.error(f"No images found in {images_dir}")
        raise FileNotFoundError(f"No images found in {images_dir} with extensions {image_extensions}")

    logger.info(f"Found {len(image_files)} images in {images_dir}")

    # =========================================================================
    # 6. Build model
    # =========================================================================
    enable_segmentation = infer_config.get("enable_segmentation", True)

    logger.info("Building SAM3 image model...")
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(
        bpe_path=bpe_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        eval_mode=True,
        checkpoint_path=checkpoint_path,
        load_from_HF=(checkpoint_path is None),
        enable_segmentation=enable_segmentation,
    )
    logger.info("Model built successfully")

    # =========================================================================
    # 7. Create processor
    # =========================================================================
    from sam3.model.sam3_image_processor import Sam3Processor

    confidence_threshold = infer_config.get("confidence_threshold", 0.3)
    resolution = infer_config.get("resolution", 1008)
    text_prompts = infer_config.get("text_prompts", ["solar array"])

    processor = Sam3Processor(
        model,
        resolution=resolution,
        device="cuda" if torch.cuda.is_available() else "cpu",
        confidence_threshold=confidence_threshold,
    )

    logger.info(f"Text prompts: {text_prompts}")
    logger.info(f"Confidence threshold: {confidence_threshold}")
    logger.info(f"Resolution: {resolution}")
    logger.info(f"Enable segmentation: {enable_segmentation}")

    # =========================================================================
    # 8. Run inference
    # =========================================================================
    overlay_alpha = infer_config.get("overlay_alpha", 0.5)
    all_predictions = []
    total_detections = 0
    start_time = time.time()

    with torch.inference_mode(), torch.autocast(
        "cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()
    ):
        for img_idx, img_path in enumerate(image_files):
            logger.info(f"[{img_idx + 1}/{len(image_files)}] Processing: {img_path.name}")

            try:
                pil_image = Image.open(img_path).convert("RGB")
            except Exception as e:
                logger.warning(f"Failed to open {img_path}: {e}")
                continue

            img_w, img_h = pil_image.size

            # Run inference for each text prompt and merge results
            image_boxes = []
            image_scores = []
            image_masks = []
            image_prompt_ids = []

            for prompt_idx, prompt in enumerate(text_prompts):
                state = processor.set_image(pil_image)
                state = processor.set_text_prompt(prompt, state)

                if "boxes" not in state or len(state["boxes"]) == 0:
                    continue

                boxes_xyxy = state["boxes"].cpu().float().numpy()  # (N, 4) in pixel coords
                scores = state["scores"].cpu().float().numpy()  # (N,)
                masks = state["masks"].cpu().float().numpy() if "masks" in state else None  # (N, 1, H, W)

                for det_idx in range(len(scores)):
                    image_boxes.append(boxes_xyxy[det_idx])
                    image_scores.append(float(scores[det_idx]))
                    image_prompt_ids.append(prompt_idx)
                    if masks is not None:
                        image_masks.append(masks[det_idx])

            num_dets = len(image_boxes)
            total_detections += num_dets

            # Build per-image prediction record
            image_pred = {
                "image_id": img_idx,
                "file_name": img_path.name,
                "width": img_w,
                "height": img_h,
                "detections": [],
            }

            for det_idx in range(num_dets):
                x1, y1, x2, y2 = image_boxes[det_idx].tolist()
                det_record = {
                    "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                    "bbox_xywh": [
                        round(x1, 2),
                        round(y1, 2),
                        round(x2 - x1, 2),
                        round(y2 - y1, 2),
                    ],
                    "score": round(image_scores[det_idx], 4),
                    "prompt": text_prompts[image_prompt_ids[det_idx]],
                    "prompt_id": image_prompt_ids[det_idx],
                }

                # Encode mask as RLE for JSON serialization
                if image_masks and det_idx < len(image_masks):
                    mask_binary = image_masks[det_idx].squeeze()  # (H, W)
                    import pycocotools.mask as mask_utils

                    rle = mask_utils.encode(
                        np.asfortranarray(mask_binary.astype(np.uint8))
                    )
                    rle["counts"] = rle["counts"].decode("utf-8")
                    det_record["segmentation"] = rle

                image_pred["detections"].append(det_record)

            all_predictions.append(image_pred)

            # -----------------------------------------------------------------
            # Save binary mask (all instances merged)
            # -----------------------------------------------------------------
            if save_masks and image_masks:
                merged_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                for m in image_masks:
                    merged_mask = np.maximum(merged_mask, m.squeeze().astype(np.uint8) * 255)
                mask_out_path = os.path.join(masks_dir, f"{img_path.stem}_mask.png")
                Image.fromarray(merged_mask).save(mask_out_path)

            # -----------------------------------------------------------------
            # Save per-instance masks
            # -----------------------------------------------------------------
            if save_instance_masks and image_masks:
                img_instance_dir = os.path.join(instance_masks_dir, img_path.stem)
                Path(img_instance_dir).mkdir(parents=True, exist_ok=True)
                for inst_idx, m in enumerate(image_masks):
                    inst_mask = (m.squeeze().astype(np.uint8)) * 255
                    inst_path = os.path.join(img_instance_dir, f"instance_{inst_idx:03d}.png")
                    Image.fromarray(inst_mask).save(inst_path)

            # -----------------------------------------------------------------
            # Save overlay visualization
            # -----------------------------------------------------------------
            if save_overlays:
                overlay_path = os.path.join(overlays_dir, f"{img_path.stem}_overlay.jpg")
                _save_overlay(
                    pil_image,
                    image_boxes,
                    image_scores,
                    image_masks,
                    [text_prompts[pid] for pid in image_prompt_ids],
                    overlay_path,
                    alpha=overlay_alpha,
                )

            if (img_idx + 1) % 50 == 0 or (img_idx + 1) == len(image_files):
                elapsed = time.time() - start_time
                rate = (img_idx + 1) / elapsed
                logger.info(
                    f"  Progress: {img_idx + 1}/{len(image_files)} images, "
                    f"{total_detections} detections, {rate:.1f} img/s"
                )

    # =========================================================================
    # 9. Save predictions JSON
    # =========================================================================
    predictions_path = os.path.join(output_dir, "predictions.json")
    predictions_output = {
        "info": {
            "description": "SAM3 inference predictions",
            "config_name": config_name,
            "text_prompts": text_prompts,
            "confidence_threshold": confidence_threshold,
            "checkpoint": checkpoint_uri or checkpoint_path or "huggingface-default",
            "num_images": len(image_files),
            "total_detections": total_detections,
            "inference_time_seconds": round(time.time() - start_time, 2),
        },
        "predictions": all_predictions,
    }

    with open(predictions_path, "w") as f:
        json.dump(predictions_output, f, indent=2)

    logger.info(f"Predictions saved to: {predictions_path}")
    logger.info(f"Total images: {len(image_files)}, Total detections: {total_detections}")

    # =========================================================================
    # 10. Upload results to S3
    # =========================================================================
    output_uri = s3_config.get("output_uri", "")
    if _is_s3_uri(output_uri):
        from ml_dronebase_data_utils.s3 import sync_dir

        logger.info(f"Uploading results from {output_dir} to {output_uri}")
        sync_dir(output_dir, output_uri)
        logger.info("Upload complete")

    # Log output summary
    _log_output_summary(output_dir, logger)

    return {
        "status": "success",
        "config_name": config_name,
        "instance_type": INSTANCE_TYPE,
        "num_images": len(image_files),
        "total_detections": total_detections,
        "inference_time_seconds": round(time.time() - start_time, 2),
        "output_uri": output_uri or output_dir,
    }


# ============================================================================
# Helper Functions
# ============================================================================


def _resolve_bpe_path(config: dict, workspace_dir: Path, logger: logging.Logger) -> str:
    """Resolve BPE tokenizer path from S3, workspace, or package resources."""
    s3_config = config.get("sagemaker", {})
    bpe_filename = "bpe_simple_vocab_16e6.txt.gz"

    bpe_search_locations = [
        workspace_dir / "sam3" / "assets" / bpe_filename,
        Path("/tmp") / bpe_filename,
        Path.cwd() / "sam3" / "assets" / bpe_filename,
    ]

    if s3_config.get("bpe_s3_uri"):
        from ml_dronebase_data_utils.s3 import download_file

        bpe_local_path = f"/tmp/{bpe_filename}"
        logger.info(f"Downloading BPE tokenizer from {s3_config['bpe_s3_uri']}")
        download_file(s3_config["bpe_s3_uri"], bpe_local_path)
        return bpe_local_path

    if "paths" in config and "bpe_path" in config["paths"] and config["paths"]["bpe_path"]:
        bpe_path = config["paths"]["bpe_path"]
        if os.path.isabs(bpe_path):
            bpe_search_locations.insert(0, Path(bpe_path))
        else:
            bpe_search_locations.insert(0, workspace_dir / bpe_path)

    for search_path in bpe_search_locations:
        logger.info(f"Checking for BPE file at: {search_path}")
        if search_path.exists():
            logger.info(f"Found BPE file at: {search_path}")
            return str(search_path)

    # Package resource fallbacks
    try:
        import pkg_resources

        pkg_bpe_path = pkg_resources.resource_filename("sam3", f"assets/{bpe_filename}")
        if os.path.exists(pkg_bpe_path):
            logger.info(f"Found BPE file via pkg_resources: {pkg_bpe_path}")
            return pkg_bpe_path
    except Exception:
        pass

    try:
        import importlib.resources

        with importlib.resources.files("sam3.assets").joinpath(bpe_filename) as bpe_resource:
            if bpe_resource.exists():
                logger.info(f"Found BPE file via importlib.resources: {bpe_resource}")
                return str(bpe_resource)
    except Exception:
        pass

    raise FileNotFoundError(
        f"BPE tokenizer file '{bpe_filename}' not found. "
        f"Searched: {[str(p) for p in bpe_search_locations]}"
    )


def _save_overlay(
    pil_image,
    boxes: list,
    scores: list,
    masks: list,
    labels: list,
    output_path: str,
    alpha: float = 0.5,
) -> None:
    """Save an overlay visualization with masks and bounding boxes on the image."""
    import cv2
    import numpy as np

    img_np = np.array(pil_image).copy()  # (H, W, 3) RGB

    # Pre-generate distinct colors
    n_colors = max(len(boxes), 1)
    np.random.seed(42)
    colors = []
    for i in range(n_colors):
        # Use HSV space for perceptually distinct colors, then convert to RGB
        hue = (i * 137.508) % 360  # golden angle for good spread
        import colorsys

        r, g, b = colorsys.hsv_to_rgb(hue / 360.0, 0.85, 0.95)
        colors.append((int(r * 255), int(g * 255), int(b * 255)))

    for det_idx in range(len(boxes)):
        color = colors[det_idx % len(colors)]
        color_bgr = (color[2], color[1], color[0])  # BGR for OpenCV

        # Draw mask overlay
        if masks and det_idx < len(masks):
            mask_binary = masks[det_idx].squeeze().astype(bool)
            overlay = img_np.copy()
            overlay[mask_binary] = color
            img_np = cv2.addWeighted(overlay, alpha, img_np, 1 - alpha, 0)

        # Draw bounding box
        x1, y1, x2, y2 = [int(c) for c in boxes[det_idx].tolist()]
        cv2.rectangle(img_np, (x1, y1), (x2, y2), color_bgr, 2)

        # Draw label
        label_text = f"{labels[det_idx]} {scores[det_idx]:.2f}"
        font_scale = 0.5
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        # Background rectangle for text
        cv2.rectangle(img_np, (x1, y1 - th - 6), (x1 + tw + 4, y1), color_bgr, -1)
        cv2.putText(
            img_np, label_text, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )

    # Save as JPEG (convert RGB → BGR for cv2)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])


def _log_output_summary(output_dir: str, logger: logging.Logger) -> None:
    """Log a summary of output files."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return

    logger.info("=" * 60)
    logger.info("INFERENCE OUTPUT SUMMARY")
    logger.info("=" * 60)
    total_files = 0
    total_size = 0
    for subdir in ["", "masks", "instance_masks", "overlays"]:
        d = output_path / subdir if subdir else output_path
        if not d.exists():
            continue
        files = [f for f in d.iterdir() if f.is_file()]
        if files:
            dir_size = sum(f.stat().st_size for f in files)
            total_size += dir_size
            total_files += len(files)
            label = subdir if subdir else "root"
            logger.info(f"  {label}: {len(files)} files ({dir_size / 1024:.1f} KB)")

    logger.info(f"  TOTAL: {total_files} files ({total_size / (1024 * 1024):.2f} MB)")
    logger.info("=" * 60)


# ============================================================================
# Local Execution (no SageMaker)
# ============================================================================
def run_local(config_content: str, config_name: str) -> dict:
    """
    Run inference locally without SageMaker.

    Useful for testing on a local GPU machine.
    """
    return _run_inference(config_content, config_name)


# ============================================================================
# CLI Entry Point
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Launch SAM3 segmentation inference job on AWS SageMaker (or locally)"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to SAM3 inference YAML config file",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default=None,
        help="Config name for logging (default: derived from config path)",
    )
    parser.add_argument(
        "--checkpoint-uri",
        type=str,
        default=None,
        help="Override S3 URI (or local path) of the checkpoint to use",
    )
    parser.add_argument(
        "--images-uri",
        type=str,
        default=None,
        help="Override S3 URI (or local path) of the images folder",
    )
    parser.add_argument(
        "--output-uri",
        type=str,
        default=None,
        help="Override S3 URI (or local path) for output",
    )
    parser.add_argument(
        "--text-prompts",
        type=str,
        nargs="+",
        default=None,
        help='Override text prompts (e.g., --text-prompts "solar panel" "hotspot")',
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=None,
        help="Override confidence threshold (0.0-1.0)",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run locally instead of on SageMaker",
    )

    args = parser.parse_args()

    # Read the config file
    import yaml

    with open(args.config, "r") as f:
        config_content_raw = f.read()

    # Apply CLI overrides
    config_data = yaml.safe_load(config_content_raw)

    if args.checkpoint_uri:
        config_data.setdefault("sagemaker", {})["pretrained_checkpoint_uri"] = args.checkpoint_uri
        print(f"Overriding checkpoint URI: {args.checkpoint_uri}")

    if args.images_uri:
        config_data.setdefault("sagemaker", {})["images_uri"] = args.images_uri
        print(f"Overriding images URI: {args.images_uri}")

    if args.output_uri:
        config_data.setdefault("sagemaker", {})["output_uri"] = args.output_uri
        print(f"Overriding output URI: {args.output_uri}")

    if args.text_prompts:
        config_data.setdefault("inference", {})["text_prompts"] = args.text_prompts
        print(f"Overriding text prompts: {args.text_prompts}")

    if args.confidence_threshold is not None:
        config_data.setdefault("inference", {})["confidence_threshold"] = args.confidence_threshold
        print(f"Overriding confidence threshold: {args.confidence_threshold}")

    config_content = yaml.dump(config_data)

    # Derive config name
    if args.config_name is None:
        config_path = Path(args.config)
        if "configs" in config_path.parts:
            configs_idx = config_path.parts.index("configs")
            config_name = "/".join(config_path.parts[configs_idx + 1:])
            config_name = config_name.rsplit(".", 1)[0]
        else:
            config_name = config_path.stem
    else:
        config_name = args.config_name

    print(f"Config: {args.config}")
    print(f"Config name: {config_name}")
    print(f"Instance type: {INSTANCE_TYPE}")
    print(f"Mode: {'local' if args.local else 'SageMaker'}")

    if args.local:
        result = run_local(config_content, config_name)
    else:
        result = main(config_content, config_name)

    print(f"Job completed: {result}")
