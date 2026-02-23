# SAM3 SageMaker Segmentation Fine-tuning — Debug & Development Log

**Date:** February 10–11, 2026
**Objective:** Create a SageMaker script to fine-tune SAM3 for solar array **segmentation** (with mask loss), based on an existing detection fine-tuning reference script.

---

## Table of Contents

1. [Files Created](#1-files-created)
2. [Files Modified](#2-files-modified)
3. [Bug #1 — Hydra Config Loading (Config Args Mismatch)](#3-bug-1--hydra-config-loading-config-args-mismatch)
4. [Bug #2 — Hydra Struct Key Error (`launcher` not in struct)](#4-bug-2--hydra-struct-key-error-launcher-not-in-struct)
5. [Bug #3 — DDP Unused Parameters](#5-bug-3--ddp-unused-parameters)
6. [Bug #4 — CUDA OOM (First Attempt — Wrong Resolution Fix)](#6-bug-4--cuda-oom-first-attempt--wrong-resolution-fix)
7. [Bug #5 — RoPE Assertion Error (Caused by Resolution Change)](#7-bug-5--rope-assertion-error-caused-by-resolution-change)
8. [Bug #6 — Collate Empty Chunks](#8-bug-6--collate-empty-chunks)
9. [Bug #7 — CUDA OOM (Second Attempt — Point Sampling Fix)](#9-bug-7--cuda-oom-second-attempt--point-sampling-fix)
10. [Bug #8 — CUDA OOM (Third Attempt — Batch Size Reduction)](#10-bug-8--cuda-oom-third-attempt--batch-size-reduction)
11. [Bug #9 — Validation DecodeRle Missing](#11-bug-9--validation-decoderle-missing)
12. [Final Working Configuration](#12-final-working-configuration)
13. [Key Lessons Learned](#13-key-lessons-learned)

---

## 1. Files Created

### 1.1 `sam3/train/fine_tune_sam3_sagemaker_seg.py` (NEW — 690 lines)

**Purpose:** Main SageMaker launch script for solar array segmentation fine-tuning.

**Based on:** Reference detection script at `/Users/sayandebroy/Developer/zeitview/codes/sam3_det_1/sam3/train/fine_tune_sam3_sagemaker_det.py`

**Key differences from the detection reference:**
- `PROJECT_NAME` changed to `"sam3-solar-array-segmentation"`
- Added `verify_segmentation_annotations()` helper to validate COCO polygon/RLE masks
- Config passed as YAML string to `main(config_content, config_name)` via `@remote` decorator
- Config saved to `/tmp/sam3_config/` on SageMaker instance, then `train.py` is invoked with `--config-path /tmp/sam3_config/ --config-name <name>`
- Added `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to environment variables (reduces CUDA memory fragmentation)
- Includes WandB initialization, S3 data download, result upload

**Key components:**
- `@remote` decorator with SageMaker session, IAM role, Docker image, dependencies
- `main()` function: downloads data from S3, writes config to disk, runs `train.py` subprocess, uploads results to S3
- Instance configuration: `ml.g5.16xlarge` (NVIDIA A10G, 24GB VRAM)
- Docker image: `757639335249.dkr.ecr.us-east-1.amazonaws.com/aiml/cuda:v1.18.1_py310`

---

### 1.2 `sam3/train/configs/solar_array/solar_array_seg_finetune_sagemaker.yaml` (NEW — ~480 lines)

**Purpose:** Full Hydra YAML configuration for segmentation fine-tuning on SageMaker.

**Key sections:**
- **SageMaker:** S3 URIs for train/val data, output, BPE tokenizer
- **WandB:** Project `sam3-solar-array-segmentation`, entity `zeitview`
- **Paths:** `/tmp/solar_array_data` (dataset), `/tmp/experiments/solar_array_segmentation` (logs)
- **Transforms:** Train + Val pipelines with `DecodeRle`, resize, pad, normalize
- **Loss:** Boxes + IABCEMdetr + **Masks** (with point sampling — see Bug #7)
- **Scratch:** `enable_segmentation: True`, resolution 1008, batch sizes, learning rates
- **Trainer:** DDP with `static_graph: True`, AMP bfloat16, 30 epochs
- **Meters:** Both `bbox` and `segm` COCO evaluation

---

### 1.3 `sam3/train/pre_exec.sh` (NEW)

**Purpose:** SageMaker pre-execution script that runs before pip install. Verifies workspace structure.

---

### 1.4 `sam3/train/sam3-sagemaker-requirements.txt` (NEW)

**Purpose:** Python dependencies for the SageMaker instance.

**Key packages:**
- `torch==2.6.0+cu124`, `torchvision==0.21.0+cu124`
- `hydra-core>=1.3.0`, `pycocotools>=2.0.0`, `wandb>=0.15.0`
- `ml-dronebase-data-utils>=0.1.0`, `sagemaker==2.251.1`
- `huggingface_hub>=0.34.0` (for SAM3 weight downloads)

---

## 2. Files Modified

### 2.1 `sam3/train/train.py` (MODIFIED)

**What changed:** Added support for 3 config loading methods (previously only supported package-based).

**Lines modified:** `__main__` block (bottom of file, ~lines 320–369)

**Before:**
```python
parser.add_argument("-c", "--config", required=True, type=str)
# ...
initialize_config_module("sam3.train", version_base="1.2")
register_omegaconf_resolvers()
main(args)
```

**After:**
```python
parser.add_argument("-c", "--config", required=False, type=str, default=None)
parser.add_argument("--config-path", type=str, default=None)
parser.add_argument("--config-name", type=str, default=None)
# ...
# Method 1: Explicit --config-path / --config-name (for SageMaker)
if args.config_path and args.config_name:
    initialize_config_dir(config_dir=args.config_path, version_base="1.2")
    args.config = args.config_name
# Method 2: SAM3_CONFIG_DIR environment variable
elif config_dir := os.environ.get("SAM3_CONFIG_DIR"):
    initialize_config_dir(config_dir=config_dir, version_base="1.2")
# Method 3: Package-based (default for local development)
else:
    initialize_config_module("sam3.train", version_base="1.2")
```

**Also added import:** `from hydra import compose, initialize_config_dir, initialize_config_module`

**Why:** SageMaker writes config to `/tmp/sam3_config/` at runtime. The original `initialize_config_module` only loads configs from within the `sam3.train` Python package, which doesn't include dynamically generated files. `initialize_config_dir` loads from an arbitrary filesystem directory.

---

## 3. Bug #1 — Hydra Config Loading (Config Args Mismatch)

| | |
|---|---|
| **SageMaker Job** | #1 |
| **Error** | `unrecognized arguments: --config-path, --config-name` |
| **Root Cause** | The SageMaker script passed `--config-path` and `--config-name` to `train.py`, but `train.py` only accepted `-c/--config` |
| **Investigation** | Read `train.py` argument parser — only had `-c/--config` as a required argument |
| **Fix Attempt 1** | Save config into `sam3/train/configs/sagemaker_runtime_config.yaml` inside the Python package, use `-c configs/sagemaker_runtime_config` |
| **Result** | This partially worked but led to Bug #2 |

---

## 4. Bug #2 — Hydra Struct Key Error (`launcher` not in struct)

| | |
|---|---|
| **SageMaker Job** | #2 |
| **Error** | `omegaconf.errors.ConfigAttributeError: Key 'launcher' is not in struct` |
| **Root Cause** | When `yaml.dump()` writes a plain dict to a YAML file and `initialize_config_module` loads it, OmegaConf treats it as a strict struct. Keys like `launcher` that are defined at the top level of the config but expected by Hydra's resolution mechanism aren't found because `@package _global_` directive semantics are lost. |
| **Investigation** | Compared `train.py` in the seg repo vs the detection repo. The detection repo's `train.py` had been patched to support `initialize_config_dir()` which loads configs from an arbitrary directory without package constraints. |
| **Fix** | Patched `train.py` to support 3 config loading methods (matching the detection reference). Reverted SageMaker script to use `--config-path /tmp/sam3_config/ --config-name <name>` approach. |
| **Files Changed** | `sam3/train/train.py` (added `--config-path`, `--config-name` args + 3-method config resolution), `fine_tune_sam3_sagemaker_seg.py` (reverted to `--config-path`/`--config-name`) |

---

## 5. Bug #3 — DDP Unused Parameters

| | |
|---|---|
| **SageMaker Job** | #3 |
| **Error** | `RuntimeError: Expected to have finished reduction in the prior iteration before starting a new one. Parameter indices which did not receive grad for rank 0: 1088 1089` |
| **Root Cause** | With `enable_segmentation: True`, the model has mask prediction head parameters in **auxiliary decoder layers**. The Masks loss has `compute_aux: false`, meaning mask loss is NOT computed on auxiliary outputs. Therefore, mask head parameters in aux layers (indices 1088, 1089) **never receive gradients**. Combined with `gradient_accumulation_steps: 2`, DDP's bucket reduction fails between successive forward passes because it expects all parameters to participate. |
| **Investigation** | Checked the loss config — `compute_aux: false` on Masks loss. Checked the model architecture — auxiliary decoder layers contain mask head parameters. Checked DDP config — `find_unused_parameters: True` was set but not sufficient with gradient accumulation. |
| **How Diagnosed** | Parameter indices 1088, 1089 consistently didn't receive gradients across all forward passes. These are the mask head weights/biases in auxiliary decoder layers that are skipped when `compute_aux: false`. |
| **Fix** | Added `static_graph: True` to the DDP distributed config. This tells PyTorch DDP to determine the computation graph topology after the first iteration and handle consistently-unused parameters correctly across gradient accumulation steps. |
| **File Changed** | `solar_array_seg_finetune_sagemaker.yaml` |
| **Config Change** | |
```yaml
distributed:
  backend: nccl
  find_unused_parameters: True
  gradient_as_bucket_view: True
  static_graph: True  # <-- ADDED
```

---

## 6. Bug #4 — CUDA OOM (First Attempt — Wrong Resolution Fix)

| | |
|---|---|
| **SageMaker Job** | #4 |
| **Error** | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 610.00 MiB. GPU 0 has a total capacity of 22.30 GiB of which 196.69 MiB is free.` |
| **Root Cause** | Segmentation mask loss computed on **full 1008×1008 masks** (~1M pixels per mask) consumed too much VRAM on A10G (24GB). The sigmoid focal loss backward pass needed 610MB but only 196MB was free. |
| **Investigation** | Training logs showed memory at 22GB out of 22.3GB. The dice loss had already OOM'd and fallen back to CPU (`GPU OOM, computing dice loss on CPU`), then the focal loss backward also OOM'd fatally. |
| **Fix Attempt** | Reduced `resolution` from 1008 → 672, reduced `train_batch_size` from 8 → 1, increased `gradient_accumulation_steps` from 2 → 4, added `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| **Result** | This led to Bug #5 (RoPE assertion) and Bug #6 (empty chunks) |

---

## 7. Bug #5 — RoPE Assertion Error (Caused by Resolution Change)

| | |
|---|---|
| **SageMaker Job** | #5 |
| **Error** | `AssertionError` at `vitdet.py:65`: `assert freqs_cis.shape == (x.shape[-2], x.shape[-1])` |
| **Root Cause** | SAM3's Vision Transformer uses **RoPE (Rotary Position Embeddings)** with frequencies precomputed for the pretrained resolution. At 1008 resolution: 1008 ÷ 14 (patch size) = **72 patches per side**, so RoPE is shape `(72×72, dim)`. At 672: 672 ÷ 14 = **48 patches**, but RoPE was still `(72×72, dim)` → shape mismatch. |
| **Investigation** | Read `vitdet.py` lines 60-95 — `reshape_for_broadcast()` asserts `freqs_cis.shape == (x.shape[-2], x.shape[-1])`. The RoPE frequencies are computed in `_setup_rope_freqs()` based on `self.input_size` which comes from the model config, not the runtime image size. |
| **Key Insight** | **Resolution MUST stay at 1008** — it's tied to the pretrained positional embeddings and cannot be changed without re-computing/interpolating RoPE frequencies (which the model doesn't support dynamically for training). |
| **Fix** | Reverted `resolution` back to 1008. Pursued alternative memory optimizations (point sampling — see Bug #7). |
| **File Changed** | `solar_array_seg_finetune_sagemaker.yaml` — reverted `resolution: 1008` |

---

## 8. Bug #6 — Collate Empty Chunks

| | |
|---|---|
| **SageMaker Job** | #5 (same job as Bug #5, different error on a different run attempt) |
| **Error** | `ValueError: max() arg is an empty sequence` at `collator.py:151` |
| **Root Cause** | `train_batch_size: 1` with `gradient_accumulation_steps: 4` meant the collator tried to split **1 sample into 4 chunks**. 3 of the 4 chunks were empty lists. The collate function did `max(q.query_processing_order for data in batch for q in data.find_queries)` on an empty batch → `max()` of empty sequence. |
| **Key Constraint** | `train_batch_size` **must be ≥** `gradient_accumulation_steps` because `collate_fn_api_with_chunking` divides the DataLoader batch into `num_chunks` equal sub-batches. |
| **Fix** | Set `train_batch_size: 4` with `gradient_accumulation_steps: 4` (then later adjusted to `train_batch_size: 2` with `gradient_accumulation_steps: 2`). |
| **File Changed** | `solar_array_seg_finetune_sagemaker.yaml` |

---

## 9. Bug #7 — CUDA OOM (Second Attempt — Point Sampling Fix)

| | |
|---|---|
| **SageMaker Job** | #6 |
| **Error** | `torch.OutOfMemoryError` again at 22GB, same focal loss backward |
| **Root Cause** | Even with batch=1 per GPU step, computing mask loss on **full 1008×1008 masks** with ~40+ annotations per image was too memory-intensive. Each matched mask pair: 1008 × 1008 = 1,016,064 float values for both prediction and target, plus gradients. |
| **Investigation** | Read `sam3/train/loss/loss_fns.py` — the `Masks` class has a built-in **point sampling** mechanism (Mask2Former style). When `num_sample_points` is `None` (default), loss is computed on the FULL mask. When set, it samples a fixed number of points using importance sampling. |
| **Fix** | Enabled point-sampled mask loss in the config: |
```yaml
- _target_: sam3.train.loss.loss_fns.Masks
  num_sample_points: 12544    # <-- ADDED (was null)
  oversample_ratio: 3.0        # <-- ADDED
  importance_sample_ratio: 0.75 # <-- ADDED
```
| **Memory Impact** | Reduced mask loss memory by ~**80×** (12,544 sampled points vs 1,016,064 full pixels). Importance sampling focuses points on uncertain boundary regions for better gradient signal. |
| **File Changed** | `solar_array_seg_finetune_sagemaker.yaml` |

---

## 10. Bug #8 — CUDA OOM (Third Attempt — Batch Size Reduction)

| | |
|---|---|
| **SageMaker Job** | #7 |
| **Error** | `torch.OutOfMemoryError: Tried to allocate 642.00 MiB` at `loss_fns.py:677` — `target_masks = target_masks[keep]` |
| **Root Cause** | Point sampling fixed the **backward** pass OOM, but the OOM now occurred **before** sampling — just indexing and selecting the full target masks. With `train_batch_size: 4` / `gradient_accumulation_steps: 2`, each GPU step processed **2 images** at 1008×1008 with ~40+ annotations each. The target mask tensor indexing alone exceeded available memory. |
| **Fix** | Reduced to **1 sample per GPU step**: |
```yaml
train_batch_size: 2     # DataLoader loads 2
gradient_accumulation_steps: 2  # Chunks into 2 → 1 per step
max_ann_per_img: 100    # Capped from 200 (avg is ~41, so most images unaffected)
```
| **Memory Result** | Peak memory dropped to **20-21GB** (well within 22.3GB A10G limit) |
| **File Changed** | `solar_array_seg_finetune_sagemaker.yaml` |

---

## 11. Bug #9 — Validation DecodeRle Missing

| | |
|---|---|
| **SageMaker Job** | #8 |
| **Error** | `KeyError: (None, None)` at `basic_for_api.py:212` — `obj.segment = F.resize(obj.segment[None, None], size).squeeze()` |
| **Training Status** | **Training was working!** Completed 2 full epochs successfully (memory stable at 20-21GB). Crashed when entering **validation** at epoch 2. |
| **Root Cause** | The `val_transforms` pipeline was missing the `DecodeRle` transform. During validation, segmentation masks were still stored as RLE dicts (from COCO JSON). When the resize transform tried to do `obj.segment[None, None]` (tensor indexing), it was actually doing `dict[None, None]` → `KeyError`. The `train_transforms` pipeline had `DecodeRle` but it was forgotten in `val_transforms`. |
| **Investigation** | Compared `train_transforms` vs `val_transforms` — the train pipeline had `DecodeRle` as the 3rd transform, the val pipeline did not. |
| **Fix** | Added `DecodeRle` as the first transform in `val_transforms`: |
```yaml
val_transforms:
  - _target_: sam3.train.transforms.basic_for_api.ComposeAPI
    transforms:
      - _target_: sam3.train.transforms.segmentation.DecodeRle  # <-- ADDED
      - _target_: sam3.train.transforms.basic_for_api.RandomResizeAPI
        # ...
```
| **File Changed** | `solar_array_seg_finetune_sagemaker.yaml` |

---

## 12. Final Working Configuration

After all 9 bugs were resolved, the final configuration that successfully trains:

| Parameter | Value | Reason |
|---|---|---|
| `resolution` | **1008** | Must match pretrained RoPE (1008/14=72 patches) |
| `train_batch_size` | **2** | DataLoader loads 2, chunked by collator |
| `gradient_accumulation_steps` | **2** | 2 chunks of 1 → 1 sample per GPU step |
| `val_batch_size` | **1** | Minimal for validation |
| `max_ann_per_img` | **100** | Caps mask memory (dataset avg ~41) |
| `num_sample_points` | **12544** | Point-sampled mask loss (80× memory reduction) |
| `oversample_ratio` | **3.0** | Mask2Former standard |
| `importance_sample_ratio` | **0.75** | Focus on uncertain regions |
| `static_graph` | **True** | Handles unused mask head params in aux layers |
| `find_unused_parameters` | **True** | Required with `enable_segmentation` + `compute_aux: false` |
| `compute_aux` (Masks) | **false** | Don't compute mask loss on auxiliary decoder outputs |
| `amp_dtype` | **bfloat16** | Better numerical stability than fp16 |
| `PYTORCH_CUDA_ALLOC_CONF` | **expandable_segments:True** | Reduces CUDA memory fragmentation |
| Val `DecodeRle` | **present** | Decode RLE masks before resize in validation |

**Memory usage:** ~20-21 GB peak on A10G (22.3 GB available)
**Batch time:** ~2.3 seconds per step after warmup
**Estimated training time:** ~40 minutes for 30 epochs

---

## 13. Key Lessons Learned

1. **RoPE is resolution-locked:** SAM3's ViT uses 2D Rotary Position Embeddings computed at init time for a specific spatial resolution. Changing resolution without re-computing RoPE causes an assertion failure. Always check if the model has resolution-dependent positional encodings before changing input size.

2. **Point-sampled mask loss is essential for memory:** Computing loss on full H×W masks is prohibitively expensive. The Mask2Former-style point sampling (`num_sample_points: 12544`) reduces memory by ~80× with minimal quality impact, as importance sampling focuses on uncertain boundary regions.

3. **`train_batch_size ≥ gradient_accumulation_steps`:** The chunking collator divides the DataLoader batch into `num_chunks` equal sub-batches. If batch_size < num_chunks, some chunks are empty → `max()` on empty sequence.

4. **`static_graph: True` for DDP with unused parameters:** When some parameters consistently don't receive gradients (e.g., mask head in aux layers with `compute_aux: false`), standard DDP fails during gradient accumulation. `static_graph: True` tells DDP to learn the graph topology after the first iteration and handle this correctly.

5. **Don't forget transforms in val pipelines:** The val_transforms must include the same data preprocessing steps (like `DecodeRle`) as train_transforms, minus the augmentation steps. It's easy to forget a transform that converts the data format (RLE → tensor) when the training pipeline works fine.

6. **Hydra config loading matters for SageMaker:** `initialize_config_module()` only loads configs from within Python packages. For SageMaker where configs are written to `/tmp/` at runtime, `initialize_config_dir()` is required.

7. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`:** When PyTorch reports "reserved but unallocated" memory, this setting helps reduce fragmentation by allowing CUDA memory segments to grow dynamically.
