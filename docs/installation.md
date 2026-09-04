# Installation

## Tested platform

The released mainline was tested on Linux with Python 3.10, CUDA 12.6,
PyTorch 2.6.0, and a 24 GB NVIDIA GPU. Pixal3D runs in low-VRAM mode at the
frozen 1024 cascade resolution. Registration and Gaussian editing can use CPU
for their numerical work, but Qwen, Pixal3D, MoGe, RMBG, and metric evaluation
require CUDA.

The exact Python package set is defined in `pyproject.toml`. Do not substitute
the Qwen `diffusers==0.36.0` and `transformers==4.57.6` pins with the newer
versions listed in a third-party Pixal3D checkout: this project was validated
with the pins below.

## Create the environment

```bash
conda create -n genpc python=3.10 -y
conda activate genpc

# CUDA-specific PyTorch wheels. Adapt the index if using another CUDA build.
python -m pip install --upgrade pip
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 \
  'torch==2.6.0+cu126' 'torchvision==0.21.0+cu126'

# Mainline runtime, model download clients, and test dependencies.
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 \
  -e '.[download,dev]'
```

For the Pixal stage, install the supplemental Python packages as well:

```bash
python -m pip install '.[pixal]'
```

Then follow the external [Pixal3D installation guide](models.md#pixal3d) for
its TRELLIS.2 / O-Voxel runtime setup. Set `PIXAL3D_SOURCE` to that checkout
before running `scripts/run_pixal3d_gpt_batch.py`.

## Optional offline CD/EMD extensions

Inference does not need the metric extensions. They are required only by
`scripts/evaluate_mainline_redwood.py` and must be compiled against the active
PyTorch/CUDA toolchain:

```bash
python -m pip install '.[metrics]'
python -m pip install -v ./loss_functions/Chamfer3D ./loss_functions/emd
```

Run metric compilation only after confirming that `nvcc`, the active PyTorch
build, and the host CUDA toolkit are compatible. The evaluator is deliberately
offline-only: it must never be called from generation, registration, or
Gaussian editing.

## Verify the codebase

Unit tests exercise deterministic registration, projection, and Gaussian
editing primitives without model weights:

```bash
python -m pytest -q
```

To run the full pipeline, install the external assets described in
[Model assets](models.md), then follow the commands in the root README.
