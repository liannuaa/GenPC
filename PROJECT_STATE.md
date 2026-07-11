# Project State

This file records accepted experiment outputs and fragile parameters. Update it
immediately when the user approves a result, says an effect is good/correct, or
asks to preserve a pipeline state.

## Accepted Results

### 2026-07-11 21:07 CST - car__132 Qwen-Image-Edit-2511 Completion

Status: accepted by user as "效果很好"; keep as the current visual quality
baseline for car completion.

Sample:
- `car__132`

Output:
- `workspace/scansalon_zup_side_512/car__132/qwen_edit_2511_car_completion_from_depth.png`

Input:
- `workspace/scansalon_zup_side_512/car__132/depth.png`

Known model/checkpoint:
- Pipeline: `models/Qwen-Image-Edit-2511`
- Nunchaku transformer: `models/nunchaku-qwen-image-edit/nunchaku_qwen_image_2511_balance_int4.safetensors`

Known intent:
- Complete the car/object silhouette and missing contour.
- Preserve the original position and pose.
- Do not output depth-map style.

Unknown / not recorded at generation time:
- Full prompt: `UNKNOWN`
- Negative prompt: `UNKNOWN`
- `true_cfg_scale`: `UNKNOWN`
- `num_inference_steps`: `UNKNOWN`
- Generation resolution and resize policy: `UNKNOWN`
- Random seed: `UNKNOWN`
- Scheduler overrides beyond the local Qwen edit script defaults: `UNKNOWN`

Reproducibility note:
- This result is not exactly reproducible from the current record. Do not claim a
  regenerated image uses the same settings unless the missing fields are
  recovered from logs or scripts.
- Later experiments with explicit `true_cfg_scale=4.0`, `num_inference_steps=40`,
  and 512 or 1024 generation did not reproduce this quality.

Downstream outputs based on this image:
- RMBG image:
  `workspace/scansalon_zup_side_512/car__132/qwen_edit_2511_car_completion_from_depth_rmbg.png`
- MoGe object point cloud:
  `workspace/scansalon_zup_side_512/car__132/moge_object_only.ply`
- MoGe partial-hit visualization:
  `workspace/scansalon_zup_side_512/car__132/moge_object_partial_hits_red.ply`
- MoGe aligned to raw partial:
  `workspace/scansalon_zup_side_512/car__132/moge_aligned_to_raw_partial.ply`
- Raw partial plus aligned MoGe:
  `workspace/scansalon_zup_side_512/car__132/raw_partial_gray_moge_red_aligned.ply`
- MoGe to raw partial transform:
  `workspace/scansalon_zup_side_512/car__132/moge_to_raw_partial_transform.npy`
- Partial to MoGe index:
  `workspace/scansalon_zup_side_512/car__132/partial_to_moge_index.npy`
- Registration info:
  `workspace/scansalon_zup_side_512/car__132/moge_to_raw_partial_info.json`

Do not overwrite without asking:
- `qwen_edit_2511_car_completion_from_depth.png`
- `qwen_edit_2511_car_completion_from_depth_rmbg.png`
- The listed MoGe/raw-partial alignment outputs.

### 2026-07-11 22:01 CST - car__132 MoGe to Raw Partial Alignment

Status: accepted by user as very good partial-to-MoGe alignment.

Code:
- `scripts/run_moge_to_raw_partial_from_camera.py`
- `scripts/run_moge_pixel_index_bridge.py`
- `scripts/run_moge_to_partial_from_index.py`

Committed baseline:
- `a4ed4d5 feat: add MoGe pixel bridge registration probes`

Critical implementation detail:
- `DepthPrompting.paintPixels()` vertically flips `depth.png` before saving.
- Pixel matching must convert projection uv to image uv with `v = 1 - v`.
- The raw partial input is:
  `workspace/scansalon/_inputs_denoised/car/car__132.ply`
- Do not use `depth_view_point_cloud.ply` as the raw partial target; it is in
  depth-view coordinates.

Known stats from accepted run:
- Raw partial points: `7076`
- MoGe object points: `37093`
- Matched partial points: `7022`
- Match ratio: `0.9923685698134539`
- RANSAC inlier ratio: `0.976`
- Median error: `0.010770418731142178`
- P95 error: `0.03548565685376803`

Primary visualization:
- `workspace/scansalon_zup_side_512/car__132/raw_partial_gray_moge_red_aligned.ply`

## Current Main-Flow Qwen Edit Configuration

As of 2026-07-12:
- Config file: `configs/config.yaml`
- Pipeline path: `models/Qwen-Image-Edit-2511`
- Transformer path:
  `models/nunchaku-qwen-image-edit/nunchaku_qwen_image_2511_balance_int4.safetensors`
- `qwen_edit_steps: 40`
- `qwen_edit_true_cfg_scale: 4.0`
- `qwen_edit_generate_res: 1024`
- Return/output size remains `generate_res: 512` unless changed.

Current prompt builder:
- `tools/qwen_image_edit.py::build_completion_prompt`

Important caveat:
- The current main-flow prompt/settings have not reproduced the accepted
  `qwen_edit_2511_car_completion_from_depth.png` quality yet.
