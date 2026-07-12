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
- Default output cleanup uses `outputs.keep_profile: lean`, which keeps only
  useful pipeline artifacts: `depth.png`, `img.png`, `camera.pth`,
  `point_uv.npy`, `qwen_edit_prompt.txt`, `img_sam.png`, final Hunyuan PLY,
  MoGe object/index/transform metadata, masked FreeReg outputs, and final
  fused PLYs.
- Set `outputs.save_intermediates: true` or `outputs.keep_profile: debug` to
  keep every depth image, mask, generated image, registered point cloud, and
  debug output. Do this only for focused experiments, because it creates many
  files.
- Current Qwen ControlNet/Nunchaku test pins are `diffusers==0.36.0` and
  `transformers==4.57.6`.

## Experiment State Recording
- Maintain `PLAN.md` as the active refactor/task plan. Before making pipeline,
  registration, Qwen, MoGe, Hunyuan, FreeReg, or cleanup changes, read `PLAN.md`
  and check which stage the work belongs to.
- Treat `docs/core_registration_pipeline.md` as the canonical description of
  the project's core registration method. Before changing partial-to-image,
  MoGe, Hunyuan, FreeReg, DepthPro, or complete-to-partial composition logic,
  read that document and update it if the method changes.
- After completing or changing a planned stage, update `PLAN.md` with status,
  output paths, and any new risks or decisions.
- Do not mark a planned stage complete unless the relevant outputs exist and
  have been verified.
- Maintain `PROJECT_STATE.md` as the canonical record for good outputs,
  accepted experiments, and fragile parameter choices.
- When the user says an output is good, correct, should be kept, or uses similar
  language such as "效果很好", "这个是对的", "保留这个", immediately append or
  update an entry in `PROJECT_STATE.md` before continuing with more experiments.
- Each accepted-result entry must include:
  - timestamp and sample id;
  - exact output file path(s);
  - input file path(s);
  - model name/path and checkpoint/transformer path;
  - full prompt and negative prompt, copied verbatim;
  - generation parameters such as resolution, resize policy, seed, steps,
    CFG/guidance scale if used, and any scheduler/backend choices;
  - postprocessing steps such as RMBG, resize, masks, MoGe, Hunyuan, FreeReg,
    ICP, or coordinate flips;
  - what the user approved and what should not be changed without asking.
- If any of those fields are unknown because they were not recorded at the time,
  write `UNKNOWN` explicitly and mark the result as not exactly reproducible.
  Do not later present guesses as facts.
- Before changing prompts, model paths, generation resolution, seeds, or
  registration parameters, check `PROJECT_STATE.md` for accepted baselines and
  preserve or copy them instead of overwriting.
- For one-off experiments, save the prompt next to the output using a descriptive
  `*_prompt.txt` file. For scripted/main-flow experiments, keep prompt builders
  in code and add tests for prompt text when feasible.

## Model Download Notes
- The old Qwen ControlNet path is no longer part of the main pipeline.
- Qwen edit pipeline files must exist at `models/Qwen-Image-Edit-2511`.
- Nunchaku Qwen edit transformer weights should exist at
  `models/nunchaku-qwen-image-edit/nunchaku_qwen_image_2511_balance_int4.safetensors`.
- Stage 1 uses Qwen-Image-Edit-2511 in one edit stage: incomplete depth image
  to completed realistic RGB/semantic image. The input image must be converted
  to RGB before calling the pipeline, and the prompt must use the object
  parameter instead of hard-coding a category.
- The default Hunyuan3D path is shape-only and uses `Hunyuan3D-2.1`.
  Download `AI-ModelScope/Hunyuan3D-2.1` into `models/Hunyuan3D-2.1`,
  including the `hunyuan3d-dit-v2-1` subfolder. The loader uses the `fp16`
  variant for this checkpoint.
- Hunyuan3D uses the regular 50-step path by default, with FlashVDM disabled.
  `hunyuan_seed: null` means the shape generator uses a random seed; set an
  integer only when reproducibility is needed.
- RMBG-2.0 should be downloaded from ModelScope repo `AI-ModelScope/RMBG-2.0`
  into `models/RMBG-2.0`.
