# Workspace Notes

## Research Engineering Scope
- This repository supports an academic paper/research project. Prefer methods
  that are simple, explainable, reproducible, and easy to ablate.
- Avoid over-engineering, model stacking, broad framework rewrites, or
  sample-specific hacks unless the user explicitly asks for a focused
  diagnostic experiment.

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
- The public mainline is Qwen semantic completion → external GPT clarity edit
  → Pixal3D → fixed Pixal--MoGe--partial registration → partial-anchored
  Gaussian edit. Hunyuan, FreeReg, PCA/ICP routing, and historical agent
  branches are not part of this route.
- Qwen edit pipeline files must exist at `models/Qwen-Image-Edit-2511`; the
  Nunchaku transformer must exist at
  `models/nunchaku-qwen-image-edit/nunchaku_qwen_image_2511_balance_int4.safetensors`.
  Stage 1 converts the depth input to RGB and uses the configurable object
  label in its prompt.
- Pixal3D source must be available at `models/Pixal3D` or via
  `PIXAL3D_SOURCE`; the tested source commit and its supplementary TRELLIS.2
  dependencies are recorded in `docs/models.md`. Its model files live under
  `models/Pixal3D-weights`; DINOv3 lives under
  `models/dinov3-vitl16-pretrain-lvd1689m`.
- Native Pixal--MoGe registration uses `models/moge-2-vitl/model.pt` and
  RMBG-2.0 under `models/RMBG-2.0`. The source image, saved MoGe cache, and
  Pixal camera metadata must all describe the same preprocessed Pixal input.
- Model weights are external assets: use the current links and licence notes in
  `docs/models.md`, rather than adding checkpoint binaries to the repository.
