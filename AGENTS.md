# Workspace Notes

## Default Python Environment
- This project defaults to the conda environment `genpc`.
- Do not assume `python` or `pip` from `PATH` points to that environment.
- Prefer the explicit interpreter and pip paths below when installing deps or running code:
  - `/opt/data/private/cr/miniconda3/envs/genpc/bin/python`
  - `/opt/data/private/cr/miniconda3/envs/genpc/bin/pip`

## Command Convention
- Install packages into `genpc`, not `base`.
- Run project scripts with `/opt/data/private/cr/miniconda3/envs/genpc/bin/python ...`.
- If a command needs `pip`, use `/opt/data/private/cr/miniconda3/envs/genpc/bin/python -m pip ...`.
- For the default full pipeline, run Stage 1 + Stage 2 + metric on the visible GPU with:
  `CUDA_VISIBLE_DEVICES=0 /opt/data/private/cr/miniconda3/envs/genpc/bin/python main.py`.
- The default config already has `run_stage1`, `run_stage2`, and `run_metric` set to `true`.
- Paths are configured through `paths` and `models` in `configs/config.yaml`.
- Default output cleanup keeps only `workspace/<sample_id>/<sample_id>_fused.ply`.
  Set `outputs.save_intermediates: true` to keep depth images, masks, generated
  images, GLB/PLY intermediates, registered point clouds, and debug outputs.
- Current Qwen ControlNet/Nunchaku test pins are `diffusers==0.36.0` and
  `transformers==4.57.6`.

## Model Download Notes
- Nunchaku Qwen transformer weights should come from ModelScope repo
  `nunchaku-tech/nunchaku-qwen-image`, not Hugging Face.
- The current default transformer weight is
  `models/nunchaku-qwen-image/svdq-int4_r128-qwen-image-lightningv1.0-4steps.safetensors`.
- Qwen pipeline files must exist at `models/Qwen-Image`; otherwise
  diffusers will try to resolve `models/Qwen-Image` through Hugging Face
  and fail.
- Because Nunchaku provides the transformer, skip Qwen pipeline full transformer
  shards (`transformer/*.safetensors`) when downloading `Qwen/Qwen-Image`.
- Qwen ControlNet Union files must exist at `models/Qwen-Image-ControlNet-Union`.
- Stage 1 uses a random seed by default. The prompt format is
  `a {flag} on a pure white background`, with ControlNet conditioning scale
  currently set in `tools/qwen_depth.py`.
- The default Hunyuan3D path is shape-only and uses `Hunyuan3D-2.1`.
  Download `AI-ModelScope/Hunyuan3D-2.1` into `models/Hunyuan3D-2.1`,
  including the `hunyuan3d-dit-v2-1` subfolder. The loader uses the `fp16`
  variant for this checkpoint.
- Hunyuan3D uses the regular 50-step path by default, with FlashVDM disabled.
  `hunyuan_seed: null` means the shape generator uses a random seed; set an
  integer only when reproducibility is needed.
- RMBG-2.0 should be downloaded from ModelScope repo `AI-ModelScope/RMBG-2.0`
  into `models/RMBG-2.0`.
