# GenPC: Zero-shot Point Cloud Completion via 3D Generative Priors
PyTorch implementation of the CVPR 2025 paper:

> [GenPC: Zero-shot Point Cloud Completion via 3D Generative Priors](https://arxiv.org/abs/2502.19896)

## Overview
GenPC completes real-world partial scans without task-specific training by leveraging strong 3D generative priors. It bridges partial point clouds to image-to-3D models with a depth-prompting module, then aligns generated shapes back to the input via geometric-preserving fusion for scale/pose consistency.

## Status
- [x] Base code released
- [ ] SDS refinement code released

## Requirements
- CUDA >= 12
- Python 3.10
- PyTorch >= 2

> Note
> The current default pipeline uses `qwen-image-edit` for image generation and `Hunyuan3D-2.0` for image-to-3D generation. The environment below focuses on that default path and does not try to keep every optional backend in the repo fully provisioned at the same time.

### Environment setup
```bash
# Recreate the checked environment exactly enough for this project.
conda env create -f environment.yml
conda activate genpc

# Or install manually:
conda create -n genpc python=3.10 -y
conda activate genpc

# IMPORTANT:
# Optional: move cache / temp files to a larger disk

# Core torch stack (CUDA 12.6 build)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu126

# Core GenPC deps
pip install fpsample trimesh open3d opencv-python Pillow scipy matplotlib imageio pytz modelscope \
  iopath munch pyyaml diffusers==0.36.0 bitsandbytes accelerate transformers==4.57.6

# Shared 3D / model deps
pip install pybind11 omegaconf pygltflib xatlas pymeshlab rembg onnxruntime kornia timm
pip install imageio-ffmpeg easydict tensorboard lpips zstandard
pip install git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# Rendering / geometry deps used by GenPC
pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.6.0_cu126.html
pip install warp-lang ipyevents ipycanvas "jupyter_client<8" tornado usd-core
pip install --no-build-isolation git+https://github.com/NVlabs/nvdiffrast.git
pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git

# Nunchaku for Qwen-Image-Edit
# Use the source build directly. On this machine, the official torch2.6 / cp310
# wheel installs but fails at import time with an ABI error such as:
#   undefined symbol: c10::detail::torchInternalAssertFail
pip install --no-build-isolation git+https://github.com/Nunchaku-AI/Nunchaku

# Optional import check:
python -c "import nunchaku; from nunchaku import NunchakuQwenImageTransformer2DModel; print('nunchaku ok')"

# Build CUDA ops for Chamfer/EMD
pip install ninja
cd loss_functions/Chamfer3D/ && python setup.py install && cd ../emd && python setup.py install && cd ../..

# Hunyuan3D-2.0 code path (current default 3D backend)
git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2 models/Hunyuan3D-2
pip install -e models/Hunyuan3D-2

# Optional: TRELLIS.2 code and CUDA extensions
# git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive models/TRELLIS.2
# export CUDA_HOME=/usr/local/cuda
# python -m pip install flash-attn==2.7.3 --no-build-isolation
# python -m pip install git+https://github.com/JeffreyXiang/nvdiffrec.git@renderutils --no-build-isolation
# python -m pip install git+https://github.com/JeffreyXiang/CuMesh.git --no-build-isolation
# python -m pip install git+https://github.com/JeffreyXiang/FlexGEMM.git --no-build-isolation
# python -m pip install models/TRELLIS.2/o-voxel --no-build-isolation
```

### Model downloads
```bash
# Image generator: Qwen-Image-Edit
python - <<'PY'
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download(
    'nunchaku-tech/nunchaku-qwen-image-edit',
    local_dir='models/nunchaku-qwen-image-edit',
    allow_patterns=[
        'svdq-int4_r128-qwen-image-edit-lightningv1.0-8steps.safetensors',
    ],
    max_workers=4,
)
PY

# Qwen pipeline weights (ModelScope)
# The Nunchaku transformer above replaces Qwen's full transformer weights.
# Keep the pipeline components and transformer config, but skip the large
# transformer/*.safetensors shards.
python - <<'PY'
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download(
    'Qwen/Qwen-Image-Edit',
    local_dir='models/Qwen-Image-Edit',
    ignore_patterns=[
        'transformer/*.safetensors',
        'transformer/*.bin',
    ],
    max_workers=4,
)
PY

# Local RMBG-2.0 for background removal
python - <<'PY'
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download(
    'AI-ModelScope/RMBG-2.0',
    local_dir='models/RMBG-2.0-ms-local',
    allow_patterns=[
        'config.json',
        'configuration.json',
        'preprocessor_config.json',
        'birefnet.py',
        'BiRefNet_config.py',
        'model.safetensors',
    ],
    max_workers=4,
)
PY

# Hunyuan3D-2.0 weights (current default shape-only 3D backend)
# The default config has hunyuan_paint: false, so only the shape subfolder is
# required for Stage 2. The loader uses the fp16 variant below. Download
# paint/delight subfolders only if enabling paint.
python - <<'PY'
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download(
    'AI-ModelScope/Hunyuan3D-2',
    local_dir='models/Hunyuan3D-2-ms',
    allow_patterns=[
        'hunyuan3d-dit-v2-0/config.yaml',
        'hunyuan3d-dit-v2-0/model.fp16.safetensors',
    ],
    max_workers=4,
)
PY

# Optional: TRELLIS.2 main weights
# MODELSCOPE_CACHE=/root/autodl-tmp/modelscope-cache python - <<'PY'
# from modelscope.hub.snapshot_download import snapshot_download
# snapshot_download('microsoft/TRELLIS.2-4B', local_dir='models/TRELLIS.2-4B', max_workers=4)
# PY

# Optional: TRELLIS.2 image encoder dependency
# MODELSCOPE_CACHE=/root/autodl-tmp/modelscope-cache python - <<'PY'
# from modelscope.hub.snapshot_download import snapshot_download
# snapshot_download(
#     'facebook/dinov3-vitl16-pretrain-lvd1689m',
#     local_dir='models/dinov3-vitl16-pretrain-lvd1689m',
#     max_workers=4,
# )
# PY

# Current default local paths:
# - Qwen transformer: models/nunchaku-qwen-image-edit/svdq-int4_r128-qwen-image-edit-lightningv1.0-8steps.safetensors
# - Qwen pipeline: models/Qwen-Image-Edit
# - Hunyuan3D-2.0: models/Hunyuan3D-2-ms
# - RMBG-2.0: models/RMBG-2.0-ms-local
```

## Usage
```bash
# 1) Adjust configs/config.yaml as needed
# 2) Run the full pipeline on GPU 6 (Stage 1 + Stage 2 + metric)
CUDA_VISIBLE_DEVICES=6 python main.py

# Checked-in default path:
# Stage 1 uses Qwen-Image-Edit, then Stage 2 uses Hunyuan3D-2.0.
# The default config has run_stage1/run_stage2/run_metric all set to true.
```

Useful runtime config knobs in `configs/config.yaml`:

- `sample_ids: []`
  - empty means run every `.ply` directly under `data/`
  - set `["07136"]` to run a single sample
- `max_samples: null`
  - set an integer to truncate the auto-discovered list
- `run_stage1: true`
- `run_stage2: true`
- `run_metric: true`

External sample example:

```bash
python main.py --config configs/config_kitti_car.yaml
```

`configs/config_kitti_car.yaml` is a ready-to-run example for:

- input point cloud: `/root/autodl-tmp/frame_0_car_0.pcd`
- explicit prompt override: `car`
- dataset mode: `kitti`
- metric disabled because there is no paired GT in this repo

### Current default config
The checked-in default config is:

- `control_model: "qwen"`
- `generative_model: "hunyuan2.0"`
- `rembg_model: "RMBG"`
- `hunyuan_model_path: "models/Hunyuan3D-2-ms"`
- `hunyuan_paint: false`
- `sample_ids: []` (auto-run all `data/*.ply`)
- `run_stage1: true`
- `run_stage2: true`
- `run_metric: true`

This means the default pipeline is:

1. `DepthPrompting` renders depth / mask guidance from the partial point cloud.
2. `Qwen-Image-Edit` generates the completed reference image.
3. `Hunyuan3D-2.0` generates the 3D asset.
4. `ScaleAdapter` aligns and fuses the generated result back to the input scan.

### Verified smoke test
The current checked-in smoke test is sample `07136` with Stage 1 + Stage 2 run
end-to-end using `sample_ids: ["07136"]`.

Previously verified on this machine:

- Stage 1: `Qwen-Image-Edit`
- Stage 2: `Hunyuan3D-2.0` and `TRELLIS.2` were both brought up successfully during integration

The checked-in default has now been switched back to `Hunyuan3D-2.0`.

## Citation
```bibtex
@inproceedings{li2025genpc,
  title={GenPC: Zero-shot Point Cloud Completion via 3D Generative Priors},
  author={Li, An and Zhu, Zhe and Wei, Mingqiang},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={1308--1318},
  year={2025}
}
```

