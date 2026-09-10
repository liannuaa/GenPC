# Model assets and external source dependencies

No model checkpoint or external source tree is bundled with GenPC+. Download
each item under its upstream terms before running the corresponding stage.
The paths below are the default local layout used by the scripts.

| Component | Used by | Required local path | Upstream source / notes |
| --- | --- | --- | --- |
| Qwen-Image-Edit-2511 | saved-view depth → semantic image | `models/Qwen-Image-Edit-2511` | [Hugging Face](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) or [ModelScope](https://modelscope.cn/models/Qwen/Qwen-Image-Edit-2511); Apache-2.0. |
| Nunchaku Qwen INT4 transformer | Qwen inference on 24 GB GPUs | `models/nunchaku-qwen-image-edit/nunchaku_qwen_image_2511_balance_int4.safetensors` | [nunchaku-tech/nunchaku-qwen-image-edit](https://huggingface.co/nunchaku-tech/nunchaku-qwen-image-edit); use the exact INT4 file named at left. |
| Pixal3D source | image → textured complete prior | `models/Pixal3D` or `$PIXAL3D_SOURCE` | [TencentARC/Pixal3D](https://github.com/TencentARC/Pixal3D), validated at commit `cdbb2bbffbf4e6f298b5f2af3d1d76a8d823d2af`; MIT. |
| Pixal3D weights | Pixal3D inference | `models/Pixal3D-weights` | [TencentARC/Pixal3D](https://huggingface.co/TencentARC/Pixal3D); keep `pipeline.json` and every referenced `ckpts/` file. |
| DINOv3 ViT-L/16 | Pixal image conditioning | `models/dinov3-vitl16-pretrain-lvd1689m` | [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m); access is gated by Meta's licence. |
| MoGe-2 ViT-L | Pixal camera and two-camera registration | `models/moge-2-vitl/model.pt` | [rookie6667/moge-2-vitl](https://modelscope.cn/models/rookie6667/moge-2-vitl). |
| RMBG-2.0 | foreground masks | `models/RMBG-2.0` | [AI-ModelScope/RMBG-2.0](https://modelscope.cn/models/AI-ModelScope/RMBG-2.0). |
| TRELLIS-image-large | corrected multi-view images -> complete mesh/Gaussians | `models/TRELLIS-image-large` | [microsoft/TRELLIS-image-large](https://huggingface.co/microsoft/TRELLIS-image-large); use under its model licence. |
| TRELLIS source | multi-image regeneration runtime | external checkout passed with `--trellis-repo` | [microsoft/TRELLIS](https://github.com/microsoft/TRELLIS), validated at commit `2301d7ba00f6f101695022123bc01363603e4828`. |
| DINOv2 ViT-L/14 reg | finite TRELLIS orientation search | Torch Hub cache | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2); Apache-2.0 code, checkpoint terms apply. |

## Pixal3D

Clone the validated Pixal3D source and expose it to this repository:

```bash
git clone https://github.com/TencentARC/Pixal3D.git models/Pixal3D
git -C models/Pixal3D checkout cdbb2bbffbf4e6f298b5f2af3d1d76a8d823d2af
export PIXAL3D_SOURCE="$PWD/models/Pixal3D"
```

Pixal3D depends on the TRELLIS.2 / O-Voxel stack, `natten`, and `utils3d`.
Install them according to the upstream Pixal3D guide. Keep this project’s Qwen
pins from `pyproject.toml`; do not blindly overwrite them with conflicting
Pixal3D `diffusers` or `transformers` pins.

Download the Pixal checkpoint into `models/Pixal3D-weights`. Its
`pipeline.json` must resolve all sparse-structure, shape, and texture files
under `ckpts/`.

## Qwen and GPT image inputs

The Qwen checkpoint is used locally by `scripts/run_semantic_stage.py`. The
Nunchaku transformer is mandatory for the released 24 GB configuration.

The following GPT clarity edit is intentionally an external artifact, not a
credential-bearing API client in this repository. For each sample, save:

```text
workspace/<run>/inputs/pixal/<sample>/gpt_image.png
workspace/<run>/inputs/pixal/<sample>/prompt.txt
```

The GPT edit must preserve the Qwen image's camera, pose, silhouette, scale,
observed parts, and articulation. It may improve clarity, material, and local
detail only. Store the exact prompt in `prompt.txt`; it is part of a
reproducible experiment record.

The second external checkpoint uses the prompt and evidence files written to
`multiview/<sample>/evidence/`. Save the shared three-view edit and the three
residual-only raw views at the exact paths printed by
`scripts/run_object_mainline.py`. No API token is read from the repository.

## TRELLIS and differentiable registration

TRELLIS is kept in a separate environment because its PyTorch, CUDA, spconv,
and attention stack differs from the Pixal/Qwen environment. Install the
official repository and model according to upstream instructions, then pass
their paths with `--trellis-repo`, `--trellis-model`, and `--trellis-python`.
The validated setup uses Python 3.10, PyTorch 2.4.0/CUDA 12.1, xFormers, native
spconv, and a 24 GB GPU.

TRELLIS generation consumes three images but no extrinsic matrices. GenPC++
therefore uses those views for semantic basin capture and estimates physical
pose/scale afterwards. The main environment supplies `nvdiffrast` and
PyTorch3D through the `differentiable-registration` extra; it does not maintain
a custom differentiable rasterizer.

## Download helpers

`modelscope` and `huggingface-hub` are included in the `download` extra. Their
model commands evolve, so use the upstream repository/model-card instructions
and verify that the resulting local layout matches the table above. The main
scripts fail early when a required directory, weight, or Pixal source checkout
is missing.

## Licences and redistribution

The repository code is MIT-licensed. External models, source checkouts,
datasets, and CUDA extensions retain their own licences and access conditions.
Do not redistribute downloaded weights or gated model files as part of a
GenPC+ release; link to their upstream source instead.
