# GenPC: Zero-shot Point Cloud Completion via 3D Generative Priors
This repository contains the PyTorch implementation of this paper:

> [**GenPC: Zero-shot Point Cloud Completion via 3D Generative Priors**](https://arxiv.org/abs/2502.19896)             
> **CVPR 2025**

## Abstract
Existing point cloud completion methods, which typically depend on predefined synthetic training datasets, encounter significant challenges when applied to out-of-distribution, real-world scans. To overcome this limitation, we introduce a zero-shot completion framework, termed GenPC, designed to reconstruct high-quality real-world scans by leveraging explicit 3D generative priors. Our key insight is that recent feed-forward 3D generative models, trained on extensive internet-scale data, have demonstrated the ability to perform 3D generation from single-view images in a zero-shot setting. To harness this for completion, we first develop a Depth Prompting module that links partial point clouds with image-to-3D generative models by leveraging depth images as a stepping stone. To retain the original partial structure in the final results, we design the Geometric Preserving Fusion module that aligns the generated shape with input by adaptively adjusting its pose and scale. Extensive experiments on widely used benchmarks validate the superiority and generalizability of our approach, bringing us a step closer to robust real-world scan completion.


## requirements
cuda 12.1/12.4/12.8

```bash
conda create -n genpc python=3.10 -y
conda activate genpc
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126

pip install ninja
cd loss_functions/Chamfer3D/
python setup.py install
cd ../emd
python setup.py install
cd ../..

pip install "rembg[cli]" fpsample trimesh open3d opencv-python Pillow iopath munch diffusers bitsandbytes munch onnxruntime
pip install kaolin==0.18.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.6.0_cu126.html # according to torch version
pip install --no-build-isolation git+https://github.com/EasternJournalist/utils3d.git@c5daf6f6c244d251f252102d09e9b7bcef791a38
pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git
pip install --no-build-isolation git+https://github.com/Nunchaku-AI/Nunchaku
pip install --no-build-isolation "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

# Download models
mkdir models
cd models
git clone https://github.com/microsoft/TRELLIS.2.git
# install trellis2

cd models
mkdir nunchaku-qwen-image-edit-2509
cd nunchaku-qwen-image-edit-2509
wget https://huggingface.co/nunchaku-ai/nunchaku-qwen-image-edit-2509/resolve/main/svdq-int4_r128-qwen-image-edit-2509-lightningv2.0-8steps.safetensors
cd ../..

cd models
mkdir qwen-image-edit-2509
modelscope download --model Qwen/Qwen-Image-Edit-2509 --local_dir ./
# The storage location of these models can be customized in the code.
```

# Example
```
python3 main.py # include inference and evaluation
# There may be a memory leak between Step 1 and Step 2, which may lead to an Out-of-Memory (OOM) error. If you encounter this, please comment out the steps and run Step 1 and Step 2 separately.
```

## Citation
```
@inproceedings{li2025genpc,
  title={GenPC: Zero-shot Point Cloud Completion via 3D Generative Priors},
  author={Li, An and Zhu, Zhe and Wei, Mingqiang},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={1308--1318},
  year={2025}
}
```