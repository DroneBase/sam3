# SAM3 Setup Guide for macOS (Apple Silicon)

This document describes the steps taken to install SAM3 dependencies and configure the project to run on macOS with Apple Silicon (M-series chips).

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Installing Dependencies](#installing-dependencies)
4. [Code Modifications for MPS Support](#code-modifications-for-mps-support)
5. [Running the Notebook](#running-the-notebook)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **Python 3.12** or higher
- **Homebrew** (recommended for installing system dependencies)

---

## Environment Setup

### 1. Create a Virtual Environment

```bash
cd /Users/sayandebroy/Developer/zeitview/codes/sam3
python3 -m venv .venv
```

### 2. Activate the Virtual Environment

```bash
source .venv/bin/activate
```

---

## Installing Dependencies

### 1. Install PyTorch with MPS Support

On macOS with Apple Silicon, PyTorch supports MPS (Metal Performance Shaders) for GPU acceleration. Install PyTorch without CUDA (which is not available on Mac):

```bash
pip install torch torchvision torchaudio
```

> **Note:** Unlike the official README which specifies CUDA-based PyTorch (`--index-url https://download.pytorch.org/whl/cu126`), macOS requires the standard PyTorch installation which includes MPS support.

### 2. Install SAM3 Package

Install the SAM3 package in editable mode:

```bash
pip install -e .
```

### 3. Install Notebook Dependencies

To run the example notebooks, install the notebook extras:

```bash
pip install -e ".[notebooks]"
```

This installs:
- `matplotlib`
- `jupyter`
- `notebook`
- `ipywidgets`
- `ipycanvas`
- `ipympl`
- `pycocotools`
- `decord`
- `opencv-python`
- `einops`
- `scikit-image`
- `scikit-learn`

### 4. Install Additional Required Packages (if needed)

```bash
pip install opencv-python matplotlib scikit-learn
```

---

## Code Modifications for MPS Support

### Device Selection in Notebooks

The original notebook code was designed for CUDA GPUs. The following modifications were made to support Apple Silicon MPS devices:

#### Original Code (CUDA-only):

```python
import torch

device = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
```

#### Modified Code (with MPS support):

```python
import torch

# Determine the best available device
if torch.cuda.is_available():
    device = "cuda"
    # turn on tfloat32 for Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # use bfloat16 for CUDA
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
elif torch.backends.mps.is_available():
    device = "mps"
    # MPS doesn't support bfloat16 well, use float32
    print(f"Using MPS (Apple Silicon) device")
else:
    device = "cpu"
    print(f"Using CPU device")

print(f"Device: {device}")
```

### Key Changes:

1. **Added MPS device detection**: `torch.backends.mps.is_available()` checks for Apple Silicon GPU availability
2. **Removed bfloat16 autocast for MPS**: MPS has limited bfloat16 support, so we use float32 instead
3. **Added CPU fallback**: For systems without GPU acceleration
4. **Device printing**: Added informative messages about which device is being used

---

## Running the Notebook

### 1. Activate the Environment

```bash
source /Users/sayandebroy/Developer/zeitview/codes/sam3/.venv/bin/activate
```

### 2. Start Jupyter Notebook

```bash
jupyter notebook
```

Or open the notebook directly in VS Code.

### 3. Open the Example Notebook

Navigate to `examples/sam3_image_predictor_example.ipynb`

### 4. Run the Cells

Execute the notebook cells sequentially. The model will automatically use MPS acceleration on Apple Silicon.

---

## Custom Visualization Function

A custom function was added to display only masks without bounding boxes:

```python
from sam3.visualization_utils import COLORS, plot_mask

def plot_masks_only(img, results):
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    nb_objects = len(results["scores"])
    print(f"found {nb_objects} object(s)")
    for i in range(nb_objects):
        color = COLORS[i % len(COLORS)]
        plot_mask(results["masks"][i].squeeze(0).cpu(), color=color)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
```

---

## Using Custom Images

To use your own images, modify the `image_path` variable:

```python
# Original example image
# image_path = f"{sam3_root}/assets/images/test_image.jpg"

# Custom image path
image_path = "/path/to/your/image.png"
```

---

## Text Prompts

The notebook supports various text prompts for segmentation:

```python
# Reset prompts and set new text prompt
processor.reset_all_prompts(inference_state)
inference_state = processor.set_text_prompt(state=inference_state, prompt="solar panels")

# Visualize results
img0 = Image.open(image_path)
plot_results(img0, inference_state)
```

Example prompts tested:
- `"solar panels"`
- `"Parallel rows of rectangular modules"`
- `"Parallel rows of rectangular modules or solar panels"`

---

## Troubleshooting

### Issue: "MPS doesn't support bfloat16"

**Solution:** The modified code already handles this by not using bfloat16 autocast on MPS devices.

### Issue: PyTorch not detecting MPS

**Solution:** Ensure you have:
- macOS 12.3 or later
- PyTorch 1.12 or later (recommended: 2.0+)

Check MPS availability:
```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
```

### Issue: Import errors for visualization modules

**Solution:** Ensure all notebook dependencies are installed:
```bash
pip install -e ".[notebooks]"
```

### Issue: Model not loading / Checkpoint access

**Solution:** Request access to SAM3 checkpoints on Hugging Face as per the official README instructions.

---

## Summary of Changes

| Change | Description |
|--------|-------------|
| Environment | Created virtual environment at `.venv` |
| PyTorch | Installed without CUDA (MPS-compatible) |
| Device Selection | Added MPS and CPU fallback support |
| Autocast | Disabled bfloat16 for MPS compatibility |
| Custom Function | Added `plot_masks_only()` for mask-only visualization |
| Image Path | Modified to use custom solar panel images |

---

## File Locations

- **Virtual Environment:** `/Users/sayandebroy/Developer/zeitview/codes/sam3/.venv`
- **Example Notebook:** `examples/sam3_image_predictor_example.ipynb`
- **Visualization Utilities:** `sam3/visualization_utils.py`

---

*Document created: January 2026*
