# SAM3 (Segment Anything Model 3) Setup Guide for macOS (Apple Silicon) and AWS Sagemaker

This document describes the steps taken to install SAM3 dependencies and configure the project to run on macOS with Apple Silicon (M-series chips) and AWS Sagemaker.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Clone this repository First](#clone-this-repository-first)
3. [Environment Setup](#environment-setup)
4. [Installing Dependencies](#installing-dependencies)
   - [Install from the requirements.txt files](#1-install-from-the-requirementstxt-files)
   - [Install SAM3 Package](#2-install-sam3-package)
   - [Install Notebook Dependencies](#3-install-notebook-dependencies)
   - [Getting Started](#4-getting-started)
   - [Convert VOC to COCO annotation format](#5-convert-voc-to-coco-annotation-format-to-fine-tune-sam3-on-aws-sagemaker)
   - [Fine-tune SAM3 on AWS-Sagemaker](#6-fine-tune-sam3-on-aws-sagemaker)
   - [Example Notebooks](#7-example-notebooks)

---

## Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **Python 3.10** or higher
- **Homebrew** (recommended for installing system dependencies)

---
## Clone this repository First
```bash
git clone https://github.com/DroneBase/sam3.git
cd sam3
```
## Environment Setup

### 1. Create a Conda or a normal Virtual Environment

```bash
conda create -n sam3 python=3.10
```
or
```bash
python3 -m venv .venv
```

### 2. Activate the Virtual Environment

```bash
conda activate sam3
```
or
```bash
source .venv/bin/activate
```

---

## Installing Dependencies

### 1. Install from the requirements.txt files

```bash
pip install -r requirements.txt
```

```bash
pip install triton  # Linux only, not available on macOS
pip install decord==0.6.0  # Linux only, not available on macOS
```

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

### 4. Getting Started

⚠️ Before using SAM 3, please request access to the checkpoints on the SAM 3
Hugging Face [repo](https://huggingface.co/facebook/sam3). Once accepted, you
need to be authenticated to download the checkpoints. You can do this by running
the following [steps](https://huggingface.co/docs/huggingface_hub/en/quick-start#authentication)
(e.g. `hf auth login` after generating an access token.)

### 5. Convert voc to coco annotation format to fine-tune SAM3 on aws-sagemaker
```bash
Example Dataset Structure:
Assumes the following directory structure:
voc_dataset/
    train/ or test/
        images/
            img1.jpg
            img2.jpg
            ...
        annotations/
            img1.xml
            img2.xml
            ...
            ...
```

```bash
Usage:
python scripts/convert_voc_to_coco.py --voc_dir /voc_dataset/train --output_json /voc_dataset/train/annotations_coco.json
```

### 6. Fine-tune SAM3 on AWS-Sagemaker
```bash
cd sam3/train/
```

#### Create a `.env` file for Hugging Face Authentication
```bash
touch .env
```

Then open the file and add your Hugging Face token:

```
HF_TOKEN=hf_xxxxxxxxxxxx  # Replace `hf_xxxxxxxxxxxx` with your actual Hugging Face access token.
```
Save the file.

#### Run
```bash
aws sso login
```

#### Now run the fine-tune script
```bash
python sam3/train/fine_tune_sam3_sagemaker_det.py --config sam3/train/configs/solar_anomaly/solar_anomaly_finetune_sagemaker.yaml
```

- configs are in sam3/train/configs



### 7. Example Notebooks

The `example_notebooks` directory contains notebooks demonstrating how to use SAM3 with
various types of prompts:

- [`sam3_image_predictor_example.ipynb`](examples/sam3_image_predictor_example.ipynb)
  : Demonstrates how to prompt SAM 3 with text and visual box prompts on images.
- [`sam3_video_predictor_example.ipynb`](examples/sam3_video_predictor_example.ipynb)
  : Demonstrates how to prompt SAM 3 with text prompts on videos, and doing
  further interactive refinements with points.
- [`sam3_image_batched_inference.ipynb`](examples/sam3_image_batched_inference.ipynb)
  : Demonstrates how to run batched inference with SAM 3 on images.
- [`sam3_agent.ipynb`](examples/sam3_agent.ipynb): Demonsterates the use of SAM
  3 Agent to segment complex text prompt on images.
- [`saco_gold_silver_vis_example.ipynb`](examples/saco_gold_silver_vis_example.ipynb)
  : Shows a few examples from SA-Co image evaluation set.
- [`saco_veval_vis_example.ipynb`](examples/saco_veval_vis_example.ipynb) :
  Shows a few examples from SA-Co video evaluation set.

#### TODO:
- Evaluation Script. [local + sagemaker]
- Inference Script. [local + sagemaker]
- Remove unnecessary files and folders.
- Distribution Training.
- .github

---
