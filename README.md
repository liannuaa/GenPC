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
> The current default pipeline uses `Qwen-Image-Edit-2511` with a Nunchaku edit transformer for single-stage depth-to-RGB completion, then `Hunyuan3D-2.1` for image-to-3D generation. The environment below focuses on that default path and does not try to keep every optional backend in the repo fully provisioned at the same time.

### Environment setup
```bash
# Create a fresh Python 3.10 environment and install the exported packages
# from the current working genpc environment.
conda create -n genpc python=3.10 -y
conda activate genpc
python -m pip install -r requirements.txt

# Manual install reference:

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
python -c "import nunchaku; from nunchaku.models.transformers.transformer_qwenimage import NunchakuQwenImageTransformer2DModel; print('nunchaku ok')"

# Build CUDA ops for Chamfer/EMD
pip install ninja
cd loss_functions/Chamfer3D/ && python setup.py install && cd ../emd && python setup.py install && cd ../..

# Hunyuan3D-2.1 code path (current default 3D backend)
git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 ../Hunyuan3D-2.1

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
# Image editor transformer: Nunchaku Qwen-Image-Edit 2511
python - <<'PY'
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download(
    'nunchaku-tech/nunchaku-qwen-image',
    local_dir='models/nunchaku-qwen-image-edit',
    allow_patterns=[
        'nunchaku_qwen_image_2511_balance_int4.safetensors',
    ],
    max_workers=4,
)
PY

# Qwen-Image-Edit-2511 pipeline weights (ModelScope)
python - <<'PY'
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download(
    'Qwen/Qwen-Image-Edit-2511',
    local_dir='models/Qwen-Image-Edit-2511',
    max_workers=4,
)
PY

# Local RMBG-2.0 for background removal
python - <<'PY'
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download(
    'AI-ModelScope/RMBG-2.0',
    local_dir='models/RMBG-2.0',
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

# Hunyuan3D-2.1 weights (current default shape-only 3D backend)
python - <<'PY'
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download(
    'AI-ModelScope/Hunyuan3D-2.1',
    local_dir='models/Hunyuan3D-2.1',
    allow_patterns=[
        'hunyuan3d-dit-v2-1/**',
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
# - Qwen edit transformer: models/nunchaku-qwen-image-edit/nunchaku_qwen_image_2511_balance_int4.safetensors
# - Qwen edit pipeline: models/Qwen-Image-Edit-2511
# - Hunyuan3D-2.1: models/Hunyuan3D-2.1
# - RMBG-2.0: models/RMBG-2.0
```

## Usage
```bash
# 1) Adjust configs/config.yaml as needed
# 2) Run the full pipeline on the visible GPU (Stage 1 + Stage 2 + metric)
CUDA_VISIBLE_DEVICES=0 /opt/data/private/cr/miniconda3/envs/genpc/bin/python main.py

# Checked-in default path:
# Stage 1 uses Qwen-Image-Edit-2511, then Stage 2 uses Hunyuan3D-2.1.
# The default config has run_stage1/run_stage2/run_metric all set to true.
```

The main runtime input is a YAML config file:

```bash
CUDA_VISIBLE_DEVICES=0 /opt/data/private/cr/miniconda3/envs/genpc/bin/python main.py \
  --config configs/config.yaml
```

### Core Registration Method

The current reconstruction-registration path is documented in
`docs/core_registration_pipeline.md`. In short, GenPC now uses the saved
partial-depth camera to build a partial-to-MoGe pixel bridge, uses fixed-uv
Sim3 F-FreeReg to align the Hunyuan complete point cloud to the image/DepthPro
frame, estimates a same-pixel DepthPro-to-MoGe Sim3 bridge, then composes:

```text
complete_to_partial =
    moge_to_partial
    @ depthpro_to_moge
    @ complete_to_depthpro
```

The FreeReg source used by this path is vendored under `third_party/FreeReg`;
large checkpoints are not committed and can be configured with
`FREEREG_DEPTHPRO_CKPT`, `FREEREG_FCGF_CKPT`, and `FREEREG_YOHO_CKPT`.
The FreeReg wrapper rejects candidates with too few Kabsch hypotheses instead
of accepting FreeReg's random fallback transform, then retries with adaptive
`ir_3d = 0.10, 0.20` when the original auto threshold fails.

Useful CLI overrides:

```bash
# Run selected samples without editing YAML
CUDA_VISIBLE_DEVICES=0 /opt/data/private/cr/miniconda3/envs/genpc/bin/python main.py \
  --sample_ids 06127 07136 07306

# Override output or model root directories
CUDA_VISIBLE_DEVICES=0 /opt/data/private/cr/miniconda3/envs/genpc/bin/python main.py \
  --workspace /tmp/genpc_workspace \
  --models_dir /opt/models/genpc
```

Key config fields in `configs/config.yaml`:

- `paths.data_dir`: directory for input `.ply` / `.pcd` samples.
- `paths.gt_dir`: directory for GT `.ply` files used by metric.
- `paths.output_dir`: workspace directory for generated outputs.
- `paths.models_dir`: root directory for local model weights.
- `paths.hunyuan_repo_root`: local clone of `Tencent-Hunyuan/Hunyuan3D-2.1`.
- `models.qwen_edit_transformer_path`: Qwen-Image-Edit Nunchaku transformer path, relative to `paths.models_dir` unless absolute.
- `models.qwen_edit_pipeline_path`: Qwen-Image-Edit pipeline directory, relative to `paths.models_dir` unless absolute.
- `models.rmbg_model_path`: RMBG-2.0 directory, relative to `paths.models_dir` unless absolute.
- `models.hunyuan_model_path`: Hunyuan3D-2.1 weights directory, relative to `paths.models_dir` unless absolute.
- `sample_ids`: empty means run every `.ply` directly under `paths.data_dir`; set `["07136"]` for a single sample.
- `input_paths`: optional per-sample explicit input paths for external files.
- `gt_paths`: optional per-sample explicit GT paths for metric.
- `outputs.keep_profile`: default `lean`; keeps only useful pipeline artifacts
  such as core Stage 1 files, Hunyuan PLY, MoGe object/index/transform outputs,
  masked FreeReg outputs, and final fused PLYs.
- `outputs.save_intermediates`: default `false`; set `true` to keep all debug
  and intermediate files for focused experiments.
- `run_stage1`, `run_stage2`, `run_metric`: enable or skip each pipeline stage.

By default, a completed sample directory keeps only:

```text
workspace/<sample_id>/<sample_id>_fused.ply
```

When `outputs.save_intermediates: true`, the pipeline also keeps depth images,
masks, generated RGB images, background-removed images, colorized partial
clouds, generated GLB/PLY files, registered generated point clouds, and
`*_fused_color.ply`.

External sample example:

```bash
CUDA_VISIBLE_DEVICES=0 /opt/data/private/cr/miniconda3/envs/genpc/bin/python main.py \
  --config configs/config_kitti_car.yaml
```

`configs/config_kitti_car.yaml` is a ready-to-run example for:

- input point cloud: `/root/autodl-tmp/frame_0_car_0.pcd`
- explicit prompt override: `car`
- dataset mode: `kitti`
- metric disabled because there is no paired GT in this repo

### Current default config
The checked-in default config is:

- `control_model: "qwen_edit"`
- `generative_model: "hunyuan2.1"`
- `rembg_model: "RMBG"`
- `paths.models_dir: "models"`
- `models.hunyuan_model_path: "Hunyuan3D-2.1"`
- `models.qwen_edit_transformer_path: "nunchaku-qwen-image-edit/nunchaku_qwen_image_2511_balance_int4.safetensors"`
- `models.qwen_edit_pipeline_path: "Qwen-Image-Edit-2511"`
- `qwen_edit_steps: 16`
- `qwen_edit_true_cfg_scale: 4.0`
- `qwen_edit_negative_prompt: " "`
- `qwen_edit_generate_res: 1024`
- `hunyuan_shape_subfolder: "hunyuan3d-dit-v2-1"`
- `hunyuan_shape_steps: 50`
- `hunyuan_seed: null` (random Hunyuan seed; set an integer for reproducibility)
- `sample_ids: []` (auto-run all `data/*.ply`)
- `outputs.save_intermediates: true`
- `run_stage1: true`
- `run_stage2: true`
- `run_metric: true`

This means the default pipeline is:

1. `DepthPrompting` renders depth / mask guidance from the partial point cloud.
2. `Qwen-Image-Edit-2511` directly converts the incomplete depth image into a completed realistic RGB/semantic image in one edit stage. The Plus pipeline output is resized to `generate_res`.
3. `Hunyuan3D-2.1` generates the 3D asset with the regular 50-step shape path; FlashVDM is disabled.
4. `ScaleAdapter` aligns and fuses the generated result back to the input scan.

### Verified smoke test
The current checked-in regression smoke test is three Redwood samples with
Stage 1 + Stage 2 + metric:

```bash
CUDA_VISIBLE_DEVICES=0 /opt/data/private/cr/miniconda3/envs/genpc/bin/python main.py \
  --sample_ids 06127 07136 07306
```

For quick single-sample validation, use `--sample_ids 07136`.

The checked-in default is Qwen Image ControlNet plus `Hunyuan3D-2.1`.

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

