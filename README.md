# RQVAE-MotionDiffuse

RQVAE-MotionDiffuse is a text-to-motion research codebase that combines:

- A MotionDiffuse-style transformer diffusion model for motion generation.
- A Residual Vector Quantized VAE (RVQ-VAE) motion tokenizer.
- A latent diffusion pipeline trained on top of RVQ-VAE latents.

This repository also includes dataset preprocessing helpers, evaluation scripts, and notebooks for quick experimentation.

## Project Status

This is a research-oriented codebase and includes some dataset-specific and machine-specific paths in a few scripts. You may need to adjust paths and options for your local environment.

## Features

- Text-to-motion generation with transformer diffusion.
- RVQ-VAE training for motion tokenization.
- Latent diffusion training on RVQ-VAE latent space.
- Support for multiple datasets (including BEAT, HumanML3D, KIT-ML in different scripts).
- Basic evaluation and visualization utilities.

## Repository Structure

```text
.
|-- datasets/                # Dataset loaders, preprocessing, evaluation helpers
|-- models/                  # Diffusion, transformer, and VQ-related models
|-- options/                 # CLI options for training/eval
|-- tools/                   # Train/eval/visualization scripts
|-- trainers/                # Trainer classes
|-- utils/                   # Metrics, plotting, motion processing, utilities
|-- install.md               # Detailed environment and dataset setup notes
|-- inference.ipynb          # Notebook for inference experiments
|-- evaluation.ipynb         # Notebook for evaluation experiments
`-- trainer_visual.ipynb     # Notebook for training visualization
```

## Requirements

Main Python packages listed in `requirements.txt`:

- `tqdm`
- `opencv-python`
- `scipy`
- `matplotlib==3.3.1`
- `spacy`
- `git+https://github.com/openai/CLIP.git`

For full environment setup (PyTorch, CUDA, MMCV), follow [install.md](./install.md).

## Installation

1. Create and activate environment (example with Conda):

```bash
conda create -n motiondiffuse python=3.7 -y
conda activate motiondiffuse
```

2. Install PyTorch for your CUDA version.

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Follow [install.md](./install.md) to prepare datasets and pretrained evaluation assets.

## Data Preparation

The codebase expects preprocessed motion/text data (for example BEAT with `npy/` and `txt/` folders, or HumanML3D/KIT-ML style folders).

For BEAT-related scripts, a typical layout is:

```text
datasets/BEAT_numpy/
|-- npy/
|-- txt/
|-- train.txt
`-- val.txt
```

Notes:

- `tools/train_vq.py` can auto-create `train.txt` and `val.txt` if missing.
- `tools/train.py` and `tools/train_vq_diffusion.py` also include helper logic for creating split files in some modes.

## Training

### 1) Train baseline MotionDiffuse-style model

Use `tools/train.py`:

```bash
python tools/train.py --dataset_name beat --name beat_diffusion_exp
```

Common datasets in this script:

- `t2m`
- `kit`
- `beat`

### 2) Train RVQ-VAE tokenizer

Use `tools/train_vq.py`:

```bash
python tools/train_vq.py \
  --dataset_name beat \
  --data_root ./datasets/BEAT_numpy \
  --name VQVAE_BEAT \
  --max_epoch 100 \
  --batch_size 64
```

Outputs are written under:

```text
checkpoints/<dataset_name>/<name>/
```

### 3) Train latent diffusion on RVQ-VAE

Use `tools/train_vq_diffusion.py`:

```bash
python tools/train_vq_diffusion.py \
  --dataset_name beat \
  --vqvae_name VQVAE_BEAT \
  --name vq_diffusion_beat \
  --max_epoch 100
```

Important:

- This script expects a valid RVQ-VAE checkpoint under your `checkpoints` directory.
- The script currently contains a hardcoded path for loading scaler stats (`global_pipeline.pkl`) and may need manual path updates.

## Inference and Visualization

- Use `inference.ipynb` for notebook-based inference.
- Use `tools/visualization.py` for script-based generation/visualization (primarily configured for `t2m` by default).

Example:

```bash
python tools/visualization.py \
  --opt_path <path_to_opt.txt> \
  --text "a person is jumping" \
  --motion_length 60 \
  --result_path sample.gif
```

## Evaluation

- `evaluation.ipynb` and `tools/evaluation.py` provide evaluation workflows.
- Some evaluation scripts include absolute local paths and should be adapted before running in a new environment.

## Troubleshooting

- If imports fail for `pymo`, ensure `datasets/pymo` is on `PYTHONPATH` or run from project root.
- If training cannot find `mean/std`, verify your preprocessing pipeline output and checkpoint metadata paths.
- If CUDA issues occur, ensure PyTorch/CUDA/MMCV versions are compatible.

## Acknowledgements

This project builds on ideas and components from MotionDiffuse, text-to-motion evaluation pipelines, and RVQ-based motion tokenization work.

## License

No explicit license file is provided in this repository. Add a `LICENSE` file if you plan to distribute or open-source this project.
