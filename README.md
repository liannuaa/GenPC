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

### Environment setup
```bash
conda create -n genpc python=3.10 -y
conda activate genpc

# Core torch stack (CUDA 12.6 build)
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu126

# Build CUDA ops for Chamfer/EMD
pip install ninja
cd loss_functions/Chamfer3D/ && python setup.py install && cd ../emd && python setup.py install && cd ../..

# Common deps
pip install "rembg[cli]" fpsample trimesh open3d opencv-python Pillow iopath \
  munch diffusers bitsandbytes onnxruntime transformers==4.57.6

# Kaolin (match torch/cu version)
pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.6.0_cu126.html

# Extra geometry / 3D toolkits
pip install --no-build-isolation git+https://github.com/EasternJournalist/utils3d.git@c5daf6f6c244d251f252102d09e9b7bcef791a38
pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git
pip install --no-build-isolation git+https://github.com/Nunchaku-AI/Nunchaku
pip install --no-build-isolation "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
```

### Model downloads
```bash
# 3D generator (recommended: TRELLIS.2; alternative: InstantMesh)
mkdir -p models && cd models
git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive

# Image generator (recommended: Qwen-Image-Edit; alternatives: T2I-Adapter / ControlNet)
mkdir -p nunchaku-qwen-image-edit-2509 && cd nunchaku-qwen-image-edit-2509
wget https://huggingface.co/nunchaku-ai/nunchaku-qwen-image-edit-2509/resolve/main/svdq-int4_r128-qwen-image-edit-2509-lightningv2.0-8steps.safetensors
cd ../..
# (Paths for these weights can be customized in code.)
```

## Usage
```bash
# 1) Adjust configs/config.yaml as needed
# 2) Run pipeline (inference + evaluation)
python main.py

# Note: There may be a memory leak between Step 1 and Step 2; if you hit OOM,
# run those steps separately by commenting out the other stage.
```

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

