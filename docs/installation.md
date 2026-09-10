# Installation

The validated setup is Linux, Python 3.10, CUDA 12.6, PyTorch 2.6, and one
24 GB NVIDIA GPU.

```bash
conda create -n genpc python=3.10 -y
conda activate genpc
python -m pip install --upgrade pip
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 \
  'torch==2.6.0+cu126' 'torchvision==0.21.0+cu126'
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu126 \
  -e '.[download,dev]'
python -m pip install -e '.[pixal,diagnostics]'
```

Install the external Pixal3D runtime and checkpoints described in
[models.md](models.md). Direct-GPT semantic completion is an external action:
the repository writes its exact task and audits the returned foreground before
3D generation. The `diffusers` and `transformers` pins remain for the optional
local Qwen baseline.

Offline CD/EMD in `scripts/evaluate_completions.py` requires the CUDA
extensions but inference does not:

```bash
python -m pip install -e '.[metrics]'
python -m pip install -v ./loss_functions/Chamfer3D ./loss_functions/emd
```

Verify deterministic geometry and artifact contracts with:

```bash
python -m pytest -q
```
