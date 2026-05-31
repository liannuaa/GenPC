# Workspace Notes

## Default Python Environment
- This project defaults to the conda environment `genpc`.
- Do not assume `python` or `pip` from `PATH` points to that environment.
- Prefer the explicit interpreter and pip paths below when installing deps or running code:
  - `/home/chenrui/miniconda3/envs/genpc/bin/python`
  - `/home/chenrui/miniconda3/envs/genpc/bin/pip`

## Command Convention
- Install packages into `genpc`, not `base`.
- Run project scripts with `/home/chenrui/miniconda3/envs/genpc/bin/python ...`.
- If a command needs `pip`, use `/home/chenrui/miniconda3/envs/genpc/bin/python -m pip ...`.
- For the default full pipeline, run Stage 1 + Stage 2 + metric on GPU 6 with:
  `CUDA_VISIBLE_DEVICES=6 /home/chenrui/miniconda3/envs/genpc/bin/python main.py`.
- The default config already has `run_stage1`, `run_stage2`, and `run_metric` set to `true`.
- Current Qwen/Nunchaku test pins are `diffusers==0.36.0` and
  `transformers==4.57.6`.

## Model Download Notes
- Nunchaku Qwen transformer weights should come from ModelScope repo
  `nunchaku-tech/nunchaku-qwen-image-edit`, not Hugging Face.
- The current default transformer weight is
  `models/nunchaku-qwen-image-edit/svdq-int4_r128-qwen-image-edit-lightningv1.0-8steps.safetensors`.
- Qwen pipeline files must exist at `models/Qwen-Image-Edit`; otherwise
  diffusers will try to resolve `models/Qwen-Image-Edit` through Hugging Face
  and fail.
- Because Nunchaku provides the transformer, skip Qwen pipeline full transformer
  shards (`transformer/*.safetensors`) when downloading `Qwen/Qwen-Image-Edit`.
- `Qwen-Image-Edit-2509` / EditPlus produced black or unstable outputs in this
  environment and is not the default path.
- The default Hunyuan3D path is shape-only (`hunyuan_paint: false`). For
  `AI-ModelScope/Hunyuan3D-2`, only download
  `hunyuan3d-dit-v2-0/config.yaml` and
  `hunyuan3d-dit-v2-0/model.fp16.safetensors` unless paint is explicitly enabled.
  The loader uses the `fp16` variant for this checkpoint.
