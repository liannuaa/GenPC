# Project State

This file records accepted experiment outputs and fragile parameters. Update it
immediately when the user approves a result, says an effect is good/correct, or
asks to preserve a pipeline state.

## Accepted Results

### 2026-07-14 16:38 CST - Redwood One-Stage Raw-Depth Qwen Image Batch

Status: accepted visual baseline. User said this one-stage raw-depth Qwen
result is "效果非常好". Preserve these one-stage image outputs unless the user
explicitly asks to overwrite them.

Samples:
- `01184`, `05117`, `05452`, `06127`, `06145`, `06830`, `06188`, `07136`,
  `07306`, `09639`

Inputs:
- Raw partial point clouds: `data/<sample>.ply`
- Qwen input images:
  `workspace/redwood_depthfirst_semantic_full_rerun_20260713/<sample>/raw_depth.png`

Outputs:
- Stage 1 Qwen images:
  `workspace/redwood_depthfirst_semantic_full_rerun_20260713/<sample>/qwen_edit_stage1.png`
- Main single-stage images:
  `workspace/redwood_depthfirst_semantic_full_rerun_20260713/<sample>/img.png`
- Stage 1 prompt records:
  `workspace/redwood_depthfirst_semantic_full_rerun_20260713/<sample>/qwen_edit_stage1_prompt.txt`

Model/checkpoint:
- Qwen pipeline: `models/Qwen-Image-Edit-2511`
- Qwen transformer:
  `models/nunchaku-qwen-image-edit/nunchaku_qwen_image_2511_balance_int4.safetensors`
- Hunyuan3D: not run for this accepted image-only result.
- RMBG: not run for this accepted image-only result.
- MoGe: not run for this accepted image-only result.

Prompts:
- `01184`: `生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的rubbish bin，纯白背景`
- `05117`: `生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的red chair，纯白背景`
- `05452`: `生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的armchair，纯白背景`
- `06127`: `生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的terracotta flower pot with leafy plant，纯白背景`
- `06145`: `生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的table，纯白背景`
- `06830`: `生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的tricycle，纯白背景`
- `06188`: `生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的red motorcyle，纯白背景`
- `07136`: `生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的leather sofa，纯白背景`
- `07306`: `生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的red office trash can，纯白背景`
- `09639`: `生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的Ergonomic Chair，纯白背景`
- Negative prompt for all samples: `" "`

Generation parameters:
- Input image size: `512x512`
- Qwen pipeline output size before resize: `1024x1024`
- Saved `qwen_edit_stage1.png` size: `512x512`
- Saved `img.png` size: `512x512`
- Stage 1 steps: `40`
- `true_cfg_scale`: `4.0`
- Seed: `UNKNOWN`, not explicitly fixed by current main pipeline
- Scheduler/backend: Qwen-Image-Edit-2511 Plus pipeline with Nunchaku
  transformer; exact scheduler `UNKNOWN`

Postprocessing:
- `DepthPrompting` generated or reused `raw_depth.png`.
- Qwen output was resized to `512x512`.
- `qwen_edit_stage1.png` and `img.png` are the same one-stage image.
- Qwen stage 2 was not run. Hunyuan, RMBG, MoGe, FreeReg, ICP, coordinate
  flips, and metrics were not run for this accepted image-only batch.

What was approved:
- The one-stage raw-depth Qwen image-generation result for the 10 Redwood
  target samples, with the prompt template above and 40 steps.

Do not overwrite without asking:
- `workspace/redwood_depthfirst_semantic_full_rerun_20260713/<sample>/qwen_edit_stage1.png`
- `workspace/redwood_depthfirst_semantic_full_rerun_20260713/<sample>/img.png`

### 2026-07-14 16:10 CST - 06188 Raw-Depth One-Stage Qwen Diagnostic

Status: accepted diagnostic visual baseline. User said this `06188` one-stage
result is good enough to try on the other samples. Preserve this diagnostic
unless the user explicitly asks to overwrite it.

Sample:
- `06188`

Inputs:
- Raw partial point cloud: `data/06188.ply`
- Qwen input image:
  `workspace/redwood_depthfirst_semantic_full_rerun_20260713/06188/raw_depth.png`

Outputs:
- Stage 1 Qwen image:
  `workspace/redwood_depthfirst_semantic_full_rerun_20260713/06188/qwen_edit_stage1.png`
- Stage 1 prompt record:
  `workspace/redwood_depthfirst_semantic_full_rerun_20260713/06188/qwen_edit_stage1_prompt.txt`

Model/checkpoint:
- Qwen pipeline: `models/Qwen-Image-Edit-2511`
- Qwen transformer:
  `models/nunchaku-qwen-image-edit/nunchaku_qwen_image_2511_balance_int4.safetensors`
- Hunyuan3D: not run for this accepted image-only result.
- RMBG: not run for this accepted image-only result.
- MoGe: not run for this accepted image-only result.

Prompt:
- Stage 1:
  `生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的red motorcyle，纯白背景`
- Negative prompt: `" "`

Generation parameters:
- Input image size: `512x512`
- Qwen pipeline output size before any later resize: `1024x1024`
- Saved stage-1 image size at acceptance time: `1024x1024`
- Stage 1 steps: `40`
- `true_cfg_scale`: `4.0`
- Seed: `UNKNOWN`, not explicitly fixed by current script
- Scheduler/backend: Qwen-Image-Edit-2511 Plus pipeline with Nunchaku
  transformer; exact scheduler `UNKNOWN`

Postprocessing:
- The Qwen input came from the existing projected `raw_depth.png`.
- Qwen output was saved as `qwen_edit_stage1.png`.
- No Qwen stage 2, Hunyuan, RMBG, MoGe, FreeReg, ICP, coordinate flips, or
  metric were run for this accepted image-only diagnostic.

What was approved:
- The `06188` one-stage raw-depth Qwen result with prompt above and 40 steps.
- User requested trying the same one-stage method on the other samples.

Do not overwrite without asking:
- `workspace/redwood_depthfirst_semantic_full_rerun_20260713/06188/qwen_edit_stage1.png`

### 2026-07-14 01:20 CST - 07136 Leather Sofa Depth-First Qwen Image

Status: accepted visual baseline. User said the `07136` result is "非常完美".
Preserve this image result unless the user explicitly asks to overwrite it.

Sample:
- `07136`

Inputs:
- Raw partial point cloud: `data/07136.ply`
- Projected depth image:
  `workspace/redwood_depthfirst_semantic_full_rerun_20260713/07136/depth.png`

Outputs:
- Stage 1 completed depth-like image:
  `workspace/redwood_depthfirst_semantic_full_rerun_20260713/07136/qwen_edit_stage1.png`
- Stage 2 final realistic/semantic image:
  `workspace/redwood_depthfirst_semantic_full_rerun_20260713/07136/img.png`
- Stage 1 prompt record:
  `workspace/redwood_depthfirst_semantic_full_rerun_20260713/07136/qwen_edit_stage1_prompt.txt`
- Stage 2 prompt record:
  `workspace/redwood_depthfirst_semantic_full_rerun_20260713/07136/qwen_edit_stage2_prompt.txt`

Model/checkpoint:
- Qwen pipeline: `models/Qwen-Image-Edit-2511`
- Nunchaku Qwen transformer:
  `models/nunchaku-qwen-image-edit/nunchaku_qwen_image_2511_balance_int4.safetensors`
- Hunyuan3D: not run for this accepted image-only result.
- RMBG: not run for this accepted image-only result.
- MoGe: not run for this accepted image-only result.

Prompt:
- Stage 1:
  `这是一个leather sofa的深度图，补全它。输出仍然是灰度深度图。`
- Stage 2:
  `根据这张完整的leather sofa深度图生成真实leather sofa照片。只保留物体的轮廓、大小、种类、朝向、姿态和相机视角，严格保持输入图中的2D投影轮廓、物体位置和大小，不要旋转、平移、缩放、换视角或重新构图，不需要保留原图的颜色、材质、光照和背景细节；让物体结构、材质和外观更真实自然，背景使用干净的纯白背景，不要生成桌面、地面、石台、墙面、植物丛或其他环境前景。`
- Negative prompt: `" "`

Generation parameters:
- Input depth/image size: `512x512`
- Qwen pipeline output size before resize: `1024x1024`
- Saved final image size: `512x512`
- Stage 1 steps: `16`
- Stage 2 steps: `16`
- `true_cfg_scale`: `4.0`
- Seed: `UNKNOWN`, not explicitly fixed by current main pipeline
- Scheduler/backend: Qwen-Image-Edit-2511 Plus pipeline with Nunchaku
  transformer; exact scheduler `UNKNOWN`

Postprocessing:
- Stage 1 projected partial depth generated by `DepthPrompting`.
- Stage 1 Qwen output saved as completed depth-like image.
- Stage 2 Qwen output resized to `512x512` and saved as `img.png`.
- No Hunyuan, RMBG, MoGe, FreeReg, ICP, coordinate flips, or metric were run
  for this accepted image-only result.

What was approved:
- The regenerated two-stage `07136` visual result with leather sofa category,
  simple depth-completion prompt, pure white background, and 16/16 Qwen steps.

Do not overwrite without asking:
- `workspace/redwood_depthfirst_semantic_full_rerun_20260713/07136/qwen_edit_stage1.png`
- `workspace/redwood_depthfirst_semantic_full_rerun_20260713/07136/img.png`

### 2026-07-14 00:47 CST - 06127 Terracotta Flower Pot Depth-First Qwen Image

Status: accepted visual baseline. User said this result is good after
inspecting the regenerated two-stage image. Preserve this `06127` image result
unless the user explicitly asks to overwrite it.

Sample:
- `06127`

Inputs:
- Raw partial point cloud: `data/06127.ply`
- Projected depth image:
  `workspace/redwood_depthfirst_semantic_full_rerun_20260713/06127/depth.png`

Outputs:
- Stage 1 completed depth-like image:
  `workspace/redwood_depthfirst_semantic_full_rerun_20260713/06127/qwen_edit_stage1.png`
- Stage 2 final realistic/semantic image:
  `workspace/redwood_depthfirst_semantic_full_rerun_20260713/06127/img.png`
- Stage 1 prompt record:
  `workspace/redwood_depthfirst_semantic_full_rerun_20260713/06127/qwen_edit_stage1_prompt.txt`
- Stage 2 prompt record:
  `workspace/redwood_depthfirst_semantic_full_rerun_20260713/06127/qwen_edit_stage2_prompt.txt`

Model/checkpoint:
- Qwen pipeline: `models/Qwen-Image-Edit-2511`
- Nunchaku Qwen transformer:
  `models/nunchaku-qwen-image-edit/nunchaku_qwen_image_2511_balance_int4.safetensors`
- Hunyuan3D: not run for this accepted image-only result.
- RMBG: not run for this accepted image-only result.
- MoGe: not run for this accepted image-only result.

Prompt:
- Stage 1:
  `这是一个terracotta flower pot with leafy plant的深度图，补全它。输出仍然是灰度深度图。`
- Stage 2:
  `根据这张完整的terracotta flower pot with leafy plant深度图生成真实terracotta flower pot with leafy plant照片。只保留物体的轮廓、大小、种类、朝向、姿态和相机视角，严格保持输入图中的2D投影轮廓、物体位置和大小，不要旋转、平移、缩放、换视角或重新构图，不需要保留原图的颜色、材质、光照和背景细节；让物体结构、材质和外观更真实自然，背景使用干净的纯白背景，不要生成桌面、地面、石台、墙面、植物丛或其他环境前景。`
- Negative prompt: `" "`

Generation parameters:
- Input depth/image size: `512x512`
- Qwen pipeline output size before resize: `1024x1024`
- Saved final image size: `512x512`
- Stage 1 steps: `16`
- Stage 2 steps: `16`
- `true_cfg_scale`: `4.0`
- Seed: `UNKNOWN`, not explicitly fixed by current main pipeline
- Scheduler/backend: Qwen-Image-Edit-2511 Plus pipeline with Nunchaku
  transformer; exact scheduler `UNKNOWN`

Postprocessing:
- Stage 1 projected partial depth generated by `DepthPrompting`.
- Stage 1 Qwen output saved as completed depth-like image.
- Stage 2 Qwen output resized to `512x512` and saved as `img.png`.
- No Hunyuan, RMBG, MoGe, FreeReg, ICP, coordinate flips, or metric were run
  for this accepted image-only result.

What was approved:
- The regenerated two-stage `06127` visual result with terracotta flower pot
  category, simple depth-completion prompt, pure white background, and 16/16
  Qwen steps.

Do not overwrite without asking:
- `workspace/redwood_depthfirst_semantic_full_rerun_20260713/06127/qwen_edit_stage1.png`
- `workspace/redwood_depthfirst_semantic_full_rerun_20260713/06127/img.png`

### 2026-07-13 17:08 CST - 01184 Symmetric Partial ICP Comparison Baseline

Status: comparison baseline, not final accepted. User said it is "确实好了一些"
when inspecting `result.png`, but also said it still has a visible gap. Keep it
as the current improved registration reference while searching for lower
CD/EMD.

Sample:
- `01184`

Inputs:
- Raw partial point cloud: `data/01184.ply`
- Stage 1 image: `workspace/redwood_stage1_qwen_refine_preview/01184/img.png`
- Complete point cloud:
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_hunyuan2.1.ply`
- Previous render-to-MoGe registration info:
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_render_to_moge_sim3_info.json`

Outputs:
- Complete-only candidate:
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_sym_partial_icp_complete_aligned_to_raw_partial.ply`
- Gray raw partial plus blue complete visualization:
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_sym_partial_icp_raw_partial_gray_complete_blue_aligned.ply`
- Transform copy:
  `/tmp/genpc_reg_ablation/01184_sym_best_f0.2_i1.0_w4.npy`

Metric:
- `CD-L1 x1e2`: `2.755563`
- `EMD x1e2`: `3.242779`
- User target for next search: lower than `CD-L1 x1e2 = 2.31` and
  `EMD x1e2 = 3.17`.

Model/checkpoint:
- Qwen pipeline: `models/Qwen-Image-Edit-2511`
- Nunchaku Qwen transformer:
  `models/nunchaku-qwen-image-edit/nunchaku_qwen_image_2511_balance_int4.safetensors`
- Hunyuan3D: `models/Hunyuan3D-2.1`
- MoGe: `models/moge-2-vitl`

Prompt:
- `根据这张rubbish bin图生成更贴近真实rubbish bin的照片。只保留物体的轮廓、大小、种类、朝向、姿态和相机视角，不需要保留原图的颜色、材质、光照和背景细节；让物体结构、材质和外观更真实自然，背景为真实场景。`

Negative prompt:
- `" "`

Generation parameters:
- Input depth/image size: `512x512`
- Qwen pipeline output size before resize: `1024x1024`
- Saved image size: `512x512`
- Qwen `num_inference_steps`: `16`
- Qwen refine stage: `True`
- Qwen refine steps: `16`
- `true_cfg_scale`: `4.0`
- Seed: `UNKNOWN`
- Hunyuan shape steps: `50`
- Hunyuan seed: `UNKNOWN`

Postprocessing and registration:
- RMBG: `models/RMBG-2.0`
- MoGe-to-raw-partial bridge from existing 01184 pipeline outputs.
- Initial complete-to-partial came from `moge_to_partial @ complete_to_moge`.
- Final comparison candidate used symmetric partial ICP from that transform with:
  `complete_trim_quantile=0.2`, `partial_trim_quantile=1.0`,
  `partial_weight=4`, `iterations=5`, sampled `6000` complete points and
  `6000` partial points.
- Quick proxy improved partial-to-complete mean distance from `0.0361409` to
  `0.0226324`, and p95 from `0.0853774` to `0.0591858`.

What was approved:
- Only that this candidate is visibly better than the previous result.

Do not treat as final:
- User explicitly said there is still a gap. Continue searching for a lower
  CD/EMD and better visual alignment before replacing `01184_fused.ply`.

### 2026-07-12 21:02 CST - car__132 Single-Stage Qwen-Image-Edit-2511 Plus Completion

Status: accepted by user as "效果非常好"; this is the current main-flow Qwen
completion baseline.

Sample:
- `car__132`

Input:
- `workspace/car__132_depth_qwen_edit_2511_run/depth.png`

Outputs:
- 1024 reference:
  `workspace/car__132_depth_qwen_edit_2511_run/qwen_image_edit_plus_2511_single_stage_16steps_1024.png`
- Main-flow size 512:
  `workspace/car__132_depth_qwen_edit_2511_run/qwen_image_edit_plus_2511_single_stage_16steps_512.png`

Parameter record:
- `workspace/car__132_depth_qwen_edit_2511_run/qwen_image_edit_plus_2511_single_stage_16steps_prompt.txt`

Model/checkpoint:
- Pipeline class: `QwenImageEditPlusPipeline`
- Pipeline path: `models/Qwen-Image-Edit-2511`
- Nunchaku transformer:
  `models/nunchaku-qwen-image-edit/nunchaku_qwen_image_2511_balance_int4.safetensors`

Prompt:
- `根据这张不完整的汽车深度图生成完整的真实汽车照片，保持已有部分的轮廓、姿态、朝向和相机视角不变，合理补全缺失部分。`

Generation parameters:
- Input argument: `image=[depth.convert("RGB")]`
- Input size: `512x512`
- Pipeline output size: `1024x1024`
- Main-flow saved size: `512x512`
- `num_inference_steps`: `16`
- `true_cfg_scale`: `4.0`
- `negative_prompt`: `" "`
- `height`: not passed
- `width`: not passed
- Seed: `None`

What was approved:
- Single-stage Qwen depth-to-realistic-RGB completion quality.
- Main flow should use this method only; no second or third Qwen stage.
- Main flow only needs to keep the final 512x512 image.
- The prompt must be parameterized by object category and must not hard-code
  `car` except when the object parameter is `car`.

Do not overwrite without asking:
- `qwen_image_edit_plus_2511_single_stage_16steps_1024.png`
- `qwen_image_edit_plus_2511_single_stage_16steps_512.png`

### 2026-07-12 19:57 CST - car__132 Qwen-Image-Edit-2511 Plus Depth Completion

Status: accepted by user as "深度补全效果非常好了"; keep this as the current
depth-completion baseline for `car__132`.

Sample:
- `car__132`

Input:
- `workspace/car__132_depth_qwen_edit_2511_run/depth.png`

Output:
- `workspace/car__132_depth_qwen_edit_2511_run/qwen_image_edit_plus_2511_depth_completion.png`

Parameter record:
- `workspace/car__132_depth_qwen_edit_2511_run/qwen_image_edit_plus_2511_depth_completion_prompt.txt`

Model/checkpoint:
- Pipeline class: `QwenImageEditPlusPipeline`
- Pipeline path: `models/Qwen-Image-Edit-2511`
- Nunchaku transformer:
  `models/nunchaku-qwen-image-edit/nunchaku_qwen_image_2511_balance_int4.safetensors`

Prompt:
- `这是一个车的深度图，补全它`

Generation parameters:
- Input argument: `image=[depth.convert("RGB")]`
- Input size: `512x512`
- Output size: `1024x1024`
- `num_inference_steps`: `4`
- `true_cfg_scale`: `4.0`
- `negative_prompt`: `" "`
- `height`: not passed
- `width`: not passed
- Seed: `None`

What was approved:
- The Qwen depth completion quality, not the downstream semantic/RGB stage.

Do not overwrite without asking:
- `qwen_image_edit_plus_2511_depth_completion.png`

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
- `depth_projection: "view_select"`; this is the original max-visible-points
  projection path.
- Semantic view candidate preview/selection is not part of the main flow.
- Pipeline path: `models/Qwen-Image-Edit-2511`
- Transformer path:
  `models/nunchaku-qwen-image-edit/nunchaku_qwen_image_2511_balance_int4.safetensors`
- Pipeline class: `QwenImageEditPlusPipeline`
- `qwen_edit_steps: 16`
- `qwen_edit_generate_res: 1024`
- `qwen_edit_true_cfg_scale: 4.0`
- `qwen_edit_negative_prompt: " "`
- Return/output size remains `generate_res: 512` unless changed.
- Qwen generation is a single stage:
  - Prompt:
    `根据这张不完整的汽车深度图生成完整的真实汽车照片，保持已有部分的轮廓、姿态、朝向和相机视角不变，合理补全缺失部分。`
  - Input is passed as `image=[depth.convert("RGB")]`.
  - No `height` or `width` is passed.
  - Pipeline output is 1024x1024, then resized to 512x512.

Current prompt builder:
- `tools/qwen_image_edit.py::build_completion_prompt`

Important caveat:
- Latest single-stage 16-step test outputs:
  - 1024:
    `workspace/car__132_depth_qwen_edit_2511_run/qwen_image_edit_plus_2511_single_stage_16steps_1024.png`
  - 512:
    `workspace/car__132_depth_qwen_edit_2511_run/qwen_image_edit_plus_2511_single_stage_16steps_512.png`
  - Prompt/parameters:
    `workspace/car__132_depth_qwen_edit_2511_run/qwen_image_edit_plus_2511_single_stage_16steps_prompt.txt`

### 2026-07-12 23:58 CST - 01184 Fixed-UV Sim3 FreeReg Registration

Status: accepted by user as very good complete-to-monocular-depth registration.

Sample:
- `01184`

Primary accepted output:
- Fused visualization:
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_freereg_original_depthpro_fixeduv_sim3_gray_object_depthpro_blue_complete_fused.ply`

Input files:
- Completed/refined semantic image:
  `workspace/redwood_stage1_qwen_refine_preview/01184/img.png`
- Hunyuan complete point cloud:
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_hunyuan2.1.ply`
- RMBG object mask:
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_moge_to_raw_partial_object_mask.png`

Model/checkpoint paths:
- FreeReg root: `/opt/data/private/cr/lab/FreeReg`
- FreeReg DepthPro checkpoint:
  `/opt/data/private/cr/lab/FreeReg/tools/DepthPro/checkpoints/depth_pro.pt`
- FreeReg YOHO FCGF checkpoint:
  `/opt/data/private/cr/lab/FreeReg/tools/YOHO/model/Backbone/best_val_checkpoint.pth`
- FreeReg YOHO checkpoint:
  `/opt/data/private/cr/lab/FreeReg/tools/YOHO/model/PartI_train/model_best.pth`
- Image generation pipeline: `models/Qwen-Image-Edit-2511`
- Qwen edit Nunchaku transformer:
  `models/nunchaku-qwen-image-edit/nunchaku_qwen_image_2511_balance_int4.safetensors`
- Hunyuan3D model: `models/Hunyuan3D-2.1`
- RMBG model: `models/RMBG-2.0`

Qwen prompts and generation parameters for `img.png`:
- Final recorded prompt:
  `根据这张rubbish bin图生成更贴近真实rubbish bin的照片。只保留物体的轮廓、大小、种类、朝向、姿态和相机视角，不需要保留原图的颜色、材质、光照和背景细节；让物体结构、材质和外观更真实自然，背景为真实场景。`
- Negative prompt: `' '`
- `true_cfg_scale`: `4.0`
- `num_inference_steps`: `16`
- `refine_stage`: `True`
- `refine_steps`: `16`
- Stage-1 prompt: `UNKNOWN` in saved prompt file; current code constructs it in
  `tools/qwen_image_edit.py::build_completion_prompt`.
- Generation resolution policy: Qwen pipeline output is 1024x1024 then final
  image is resized to 512x512.
- Seed: `UNKNOWN` / random unless explicitly set elsewhere.

FreeReg parameters:
- Script: `scripts/run_freereg_original_depthpro.py`
- Output prefix: `01184_freereg_original_depthpro_fixeduv_sim3`
- `nkpts`: `5000`
- `w_2d`: `0.5`
- object mask threshold: `128`
- extra erode pixels: `0`
- max DepthPro points: `50000`
- random seed: `1184`
- object DepthPro points: `49578`
- complete points: `100000`
- image keypoints: `5000`
- complete keypoints: `5000`
- YOHO matches: `396`
- FreeReg scale: `1.122206687927246`
- Intrinsic:
  `[[517.53125, 0.0, 255.5], [0.0, 517.53125, 255.5], [0.0, 0.0, 1.0]]`

Critical implementation details:
- Use RMBG object mask before DepthPro point-cloud construction so ground and
  background points are not included as FreeReg targets.
- Use YOHO image keypoint uv coordinates directly for FreeReg 2D scoring.
  Do not index dense DepthPro point uv with YOHO keypoint indices.
- Apply FreeReg's estimated scale as a Sim3 transform. The older objectmask
  result did not record/apply this scale in the final saved transform.

Measured improvement versus older objectmask output:
- Bbox-center z offset decreased from about `0.426` to about `0.018`.
- Complete-to-DepthPro nearest-neighbor mean decreased from about `0.393` to
  about `0.176`.

Do not change without asking:
- Do not overwrite the accepted fused visualization path listed above.
- Do not remove fixed-uv or Sim3 handling from `scripts/run_freereg_original_depthpro.py`.
- Do not switch this accepted baseline back to unmasked DepthPro or rigid-only
  FreeReg output.
# 2026-07-13 Redwood studio-background rerun partial visual approval

- Timestamp: 2026-07-13 21:23 Asia/Shanghai
- Sample ids: `01184`, `05117`, `05452`, `06127`, `06145` completed so far
- Output root: `workspace/redwood_qwen_studio_bg_full_rerun_20260713`
- Input point clouds:
  - `data/01184.ply`
  - `data/05117.ply`
  - `data/05452.ply`
  - `data/06127.ply`
  - `data/06145.ply`
- Current output file paths:
  - `workspace/redwood_qwen_studio_bg_full_rerun_20260713/01184/img.png`
  - `workspace/redwood_qwen_studio_bg_full_rerun_20260713/01184/01184_hunyuan2.1.ply`
  - `workspace/redwood_qwen_studio_bg_full_rerun_20260713/01184/01184_complete_aligned_to_raw_partial.ply`
  - `workspace/redwood_qwen_studio_bg_full_rerun_20260713/01184/01184_fused.ply`
  - `workspace/redwood_qwen_studio_bg_full_rerun_20260713/05117/img.png`
  - `workspace/redwood_qwen_studio_bg_full_rerun_20260713/05117/05117_hunyuan2.1.ply`
  - `workspace/redwood_qwen_studio_bg_full_rerun_20260713/05117/05117_complete_aligned_to_raw_partial.ply`
  - `workspace/redwood_qwen_studio_bg_full_rerun_20260713/05117/05117_fused.ply`
  - `workspace/redwood_qwen_studio_bg_full_rerun_20260713/05452/img.png`
  - `workspace/redwood_qwen_studio_bg_full_rerun_20260713/05452/05452_hunyuan2.1.ply`
  - `workspace/redwood_qwen_studio_bg_full_rerun_20260713/05452/05452_complete_aligned_to_raw_partial.ply`
  - `workspace/redwood_qwen_studio_bg_full_rerun_20260713/05452/05452_fused.ply`
  - `workspace/redwood_qwen_studio_bg_full_rerun_20260713/06127/img.png`
  - `workspace/redwood_qwen_studio_bg_full_rerun_20260713/06127/06127_hunyuan2.1.ply`
  - `workspace/redwood_qwen_studio_bg_full_rerun_20260713/06127/06127_complete_aligned_to_raw_partial.ply`
  - `workspace/redwood_qwen_studio_bg_full_rerun_20260713/06127/06127_fused.ply`
  - `workspace/redwood_qwen_studio_bg_full_rerun_20260713/06145/img.png`
  - `workspace/redwood_qwen_studio_bg_full_rerun_20260713/06145/06145_hunyuan2.1.ply`
  - `workspace/redwood_qwen_studio_bg_full_rerun_20260713/06145/06145_complete_aligned_to_raw_partial.ply`
  - `workspace/redwood_qwen_studio_bg_full_rerun_20260713/06145/06145_fused.ply`
- Model name/path and checkpoint/transformer path:
  - Qwen pipeline: `models/Qwen-Image-Edit-2511`
  - Qwen transformer: `models/nunchaku-qwen-image-edit/nunchaku_qwen_image_2511_balance_int4.safetensors`
  - Hunyuan3D: `models/Hunyuan3D-2.1`, shape subfolder `hunyuan3d-dit-v2-1`, checkpoint variant `fp16`
  - RMBG: `models/RMBG-2.0`
  - MoGe: `models/moge-2-vitl`
- Full prompts:
  - Stage 1 prompt template: `根据这张不完整的{object}深度图生成完整的真实{object}照片，保持已有部分的轮廓、姿态、朝向和相机视角不变，合理补全缺失部分，背景使用干净的普通摄影棚背景。`
  - Stage 2 refinement prompt template: `根据这张{object}图生成更贴近真实{object}的照片。只保留物体的轮廓、大小、种类、朝向、姿态和相机视角，不需要保留原图的颜色、材质、光照和背景细节；让物体结构、材质和外观更真实自然，背景使用干净的普通摄影棚背景，不要生成桌面、地面、石台、墙面、植物丛或其他环境前景。`
  - Object labels from config/dataset: `01184=rubbish bin`, `05117=red chair`, `05452=armchair`, `06127=a vase with leafy plant`, `06145=table`
  - Negative prompt: `" "`
- Generation parameters:
  - Qwen input/output: `depth_image_input_res=512`, `qwen_edit_generate_res=1024`, final `generate_res=512`
  - Qwen steps: `qwen_edit_steps=16`, `qwen_edit_refine_steps=16`
  - Qwen CFG: `qwen_edit_true_cfg_scale=4.0`
  - Qwen seed: UNKNOWN, not explicitly set by current main pipeline
  - Hunyuan steps: `50`
  - Hunyuan seed: `null` in config, random unless Hunyuan internals override it
  - Hunyuan octree resolution: `384`
  - Hunyuan point sample count: `100000`
- Postprocessing:
  - RMBG background removal to `img_sam.png`
  - MoGeV2 on `img.png`
  - RMBG object mask, object mask erosion `object_mask_erode_pixels=2`
  - pixel bridge `partial point index -> MoGe point index`
  - render-to-MoGe Sim3 registration with 2D gate, retry search, bridge-anchor optimization, and anisotropic partial refinement
- Metrics completed so far:
  - `01184`: `CD-L1 x1e2=0.920551`, `EMD x1e2=1.440672`
  - `05117`: `CD-L1 x1e2=1.239928`, `EMD x1e2=1.676881`
  - `05452`: `CD-L1 x1e2=2.557462`, `EMD x1e2=3.513691`
  - `06127`: `CD-L1 x1e2=2.846162`, `EMD x1e2=4.402206`
  - `06145`: `CD-L1 x1e2=1.335033`, `EMD x1e2=2.102884`
- User approval:
  - User said the slight regression is acceptable and the first few samples already look very good: `这点退化我能接受，前几个看上去已经很不错了`.
  - Treat the studio-background prompt direction and this output root as a promising baseline for the first completed samples.
  - Do not overwrite the listed accepted sample directories without asking.
  - The full batch was later interrupted to rerun `06188` by explicit user request.
  - `06188` is not part of this accepted visual baseline.
- Reproducibility note: Qwen and Hunyuan seeds are not fixed, so this is not exactly reproducible.

### 2026-07-13 22:05 CST - 06188 red motorcyle rerun diagnostic

Status: diagnostic only, not accepted. User requested changing `06188` to
`red motorcyle` and rerunning the sample. The resulting metric worsened
slightly and the current 2D/MoGe overlay still shows systematic offset.

Sample:
- `06188`

Input:
- Raw partial point cloud: `data/06188.ply`

Outputs:
- Stage 1 image:
  `workspace/redwood_qwen_studio_bg_full_rerun_20260713/06188/img.png`
- Hunyuan complete point cloud:
  `workspace/redwood_qwen_studio_bg_full_rerun_20260713/06188/06188_hunyuan2.1.ply`
- Final complete aligned to raw partial:
  `workspace/redwood_qwen_studio_bg_full_rerun_20260713/06188/06188_complete_aligned_to_raw_partial.ply`
- Metric output:
  `workspace/redwood_qwen_studio_bg_full_rerun_20260713/06188/06188_fused.ply`
- Gray raw partial plus blue complete visualization:
  `workspace/redwood_qwen_studio_bg_full_rerun_20260713/06188/06188_raw_partial_gray_complete_blue_aligned.ply`
- Gray raw partial plus red MoGe bridge visualization:
  `workspace/redwood_qwen_studio_bg_full_rerun_20260713/06188/06188_moge_to_raw_partial_raw_partial_gray_moge_red_aligned.ply`
- 2D render-to-MoGe overlay:
  `workspace/redwood_qwen_studio_bg_full_rerun_20260713/06188/06188_render_to_moge_overlay.png`

Metric:
- `CD-L1 x1e2`: `5.186385`
- `EMD x1e2`: `6.571867`

Model/checkpoint:
- Qwen pipeline: `models/Qwen-Image-Edit-2511`
- Qwen transformer:
  `models/nunchaku-qwen-image-edit/nunchaku_qwen_image_2511_balance_int4.safetensors`
- Hunyuan3D: `models/Hunyuan3D-2.1`, shape subfolder `hunyuan3d-dit-v2-1`,
  checkpoint variant `fp16`
- RMBG: `models/RMBG-2.0`
- MoGe: `models/moge-2-vitl`

Prompt:
- Stage 1 prompt:
  `根据这张不完整的red motorcyle深度图生成完整的真实red motorcyle照片，保持已有部分的轮廓、姿态、朝向和相机视角不变，合理补全缺失部分，背景使用干净的普通摄影棚背景。`
- Stage 2 refinement prompt:
  `根据这张red motorcyle图生成更贴近真实red motorcyle的照片。只保留物体的轮廓、大小、种类、朝向、姿态和相机视角，不需要保留原图的颜色、材质、光照和背景细节；让物体结构、材质和外观更真实自然，背景使用干净的普通摄影棚背景，不要生成桌面、地面、石台、墙面、植物丛或其他环境前景。`
- Negative prompt: `" "`

Generation parameters:
- Qwen input/output: `depth_image_input_res=512`, `qwen_edit_generate_res=1024`,
  final `generate_res=512`
- Qwen steps: `qwen_edit_steps=16`, `qwen_edit_refine_steps=16`
- Qwen CFG: `qwen_edit_true_cfg_scale=4.0`
- Qwen seed: UNKNOWN, not explicitly set by current main pipeline
- Hunyuan steps: `50`
- Hunyuan seed: `null` in config, random unless Hunyuan internals override it
- Hunyuan octree resolution: `384`
- Hunyuan point sample count: `100000`

Postprocessing and registration:
- RMBG background removal to `img_sam.png`
- MoGeV2 on `img.png`
- RMBG object mask with `object_mask_erode_pixels=2`
- Pixel bridge `partial point index -> MoGe point index`
- Render-to-MoGe Sim3 registration with retry search, visible 3D optimization,
  bridge-anchor optimization, and 2D acceptance gate
- MoGe bridge stats: `match_ratio = 0.987339`, `ransac_inlier_ratio = 0.9885`,
  `ransac_median_error = 0.025176`
- 2D gate failed: `iou = 0.749277`, `coverage = 0.866660`,
  `leakage = 0.153091`; failed checks are `iou` and `leakage`
- Partial-space refinement was skipped because `registration_2d_threshold_not_met`

Do not treat as accepted:
- The old same-root `06188` metric before the prompt override was
  `CD-L1 x1e2 = 5.018309`, `EMD x1e2 = 6.483270`.
- The `red motorcyle` prompt label did not improve this sample.

### 2026-07-14 01:30 CST - 07306 Red Office Trash Can Depth-First Qwen Image

Status: accepted image generation baseline. User said `07306效果也很好`.

Sample:
- `07306`

Input:
- Raw partial point cloud: `data/07306.ply`
- Depth input image:
  `workspace/redwood_depthfirst_semantic_full_rerun_20260713/07306/depth.png`

Accepted outputs:
- Completed depth-like image:
  `workspace/redwood_depthfirst_semantic_full_rerun_20260713/07306/qwen_edit_stage1.png`
- Final semantic/RGB image:
  `workspace/redwood_depthfirst_semantic_full_rerun_20260713/07306/img.png`
- Stage 1 prompt record:
  `workspace/redwood_depthfirst_semantic_full_rerun_20260713/07306/qwen_edit_stage1_prompt.txt`
- Stage 2 prompt record:
  `workspace/redwood_depthfirst_semantic_full_rerun_20260713/07306/qwen_edit_stage2_prompt.txt`

Model/checkpoint:
- Qwen pipeline: `models/Qwen-Image-Edit-2511`
- Qwen transformer:
  `models/nunchaku-qwen-image-edit/nunchaku_qwen_image_2511_balance_int4.safetensors`
- RMBG: `models/RMBG-2.0`
- Hunyuan3D: `models/Hunyuan3D-2.1`, shape subfolder `hunyuan3d-dit-v2-1`,
  checkpoint variant `fp16`
- MoGe: `models/moge-2-vitl`

Full prompts:
- Stage 1 prompt:
  `这是一个red office trash can的深度图，补全它。输出仍然是灰度深度图。`
- Stage 2 prompt:
  `根据这张完整的red office trash can深度图生成真实red office trash can照片。只保留物体的轮廓、大小、种类、朝向、姿态和相机视角，严格保持输入图中的2D投影轮廓、物体位置和大小，不要旋转、平移、缩放、换视角或重新构图，不需要保留原图的颜色、材质、光照和背景细节；让物体结构、材质和外观更真实自然，背景使用干净的纯白背景，不要生成桌面、地面、石台、墙面、植物丛或其他环境前景。`
- Negative prompt: `" "`

Generation parameters:
- Qwen input/output: `depth_image_input_res=512`, `qwen_edit_generate_res=1024`,
  final `generate_res=512`
- Qwen stage 1 steps: `16`
- Qwen stage 2 steps: `16`
- Qwen CFG: `qwen_edit_true_cfg_scale=4.0`
- Qwen seed: UNKNOWN, not explicitly set by current main pipeline
- Backend/scheduler choices: UNKNOWN

Postprocessing:
- This accepted entry covers the image-generation result only.
- Stage 1 used incomplete depth image to completed depth-like image.
- Stage 2 used the completed depth-like image to generate the final realistic
  semantic/RGB image with pure white background.
- Later RMBG, MoGe, Hunyuan, FreeReg, ICP, and metric stages are not part of
  this accepted image baseline unless recorded separately.

Preservation:
- Do not overwrite this `07306` image result without asking.
- The accepted category label is `red office trash can`.
- Reproducibility note: Qwen seed and scheduler/backend details were not fully
  recorded, so this is not exactly reproducible.

### 2026-07-14 01:36 CST - Full Redwood Depth-First Qwen Image Batch Accepted

Status: accepted image-generation baseline for the 10-sample Redwood target
set. User said this batch's image-generation part is very good and should not
be changed further; continue only with later 3D generation, registration, and
metric optimization.

Output root:
- `workspace/redwood_depthfirst_semantic_full_rerun_20260713`

Samples:
- `01184`, `09639`, `05452`, `05117`, `06127`, `07136`, `07306`, `06188`,
  `06145`, `06830`

Inputs:
- Raw partial point clouds: `data/<sample>.ply`
- Depth input images:
  `workspace/redwood_depthfirst_semantic_full_rerun_20260713/<sample>/depth.png`

Accepted image outputs:
- `01184`:
  - `workspace/redwood_depthfirst_semantic_full_rerun_20260713/01184/qwen_edit_stage1.png`
  - `workspace/redwood_depthfirst_semantic_full_rerun_20260713/01184/img.png`
- `09639`:
  - `workspace/redwood_depthfirst_semantic_full_rerun_20260713/09639/qwen_edit_stage1.png`
  - `workspace/redwood_depthfirst_semantic_full_rerun_20260713/09639/img.png`
- `05452`:
  - `workspace/redwood_depthfirst_semantic_full_rerun_20260713/05452/qwen_edit_stage1.png`
  - `workspace/redwood_depthfirst_semantic_full_rerun_20260713/05452/img.png`
- `05117`:
  - `workspace/redwood_depthfirst_semantic_full_rerun_20260713/05117/qwen_edit_stage1.png`
  - `workspace/redwood_depthfirst_semantic_full_rerun_20260713/05117/img.png`
- `06127`:
  - `workspace/redwood_depthfirst_semantic_full_rerun_20260713/06127/qwen_edit_stage1.png`
  - `workspace/redwood_depthfirst_semantic_full_rerun_20260713/06127/img.png`
- `07136`:
  - `workspace/redwood_depthfirst_semantic_full_rerun_20260713/07136/qwen_edit_stage1.png`
  - `workspace/redwood_depthfirst_semantic_full_rerun_20260713/07136/img.png`
- `07306`:
  - `workspace/redwood_depthfirst_semantic_full_rerun_20260713/07306/qwen_edit_stage1.png`
  - `workspace/redwood_depthfirst_semantic_full_rerun_20260713/07306/img.png`
- `06188`:
  - `workspace/redwood_depthfirst_semantic_full_rerun_20260713/06188/qwen_edit_stage1.png`
  - `workspace/redwood_depthfirst_semantic_full_rerun_20260713/06188/img.png`
- `06145`:
  - `workspace/redwood_depthfirst_semantic_full_rerun_20260713/06145/qwen_edit_stage1.png`
  - `workspace/redwood_depthfirst_semantic_full_rerun_20260713/06145/img.png`
- `06830`:
  - `workspace/redwood_depthfirst_semantic_full_rerun_20260713/06830/qwen_edit_stage1.png`
  - `workspace/redwood_depthfirst_semantic_full_rerun_20260713/06830/img.png`

Model/checkpoint:
- Qwen pipeline: `models/Qwen-Image-Edit-2511`
- Qwen transformer:
  `models/nunchaku-qwen-image-edit/nunchaku_qwen_image_2511_balance_int4.safetensors`
- RMBG: `models/RMBG-2.0`
- Hunyuan3D: `models/Hunyuan3D-2.1`, shape subfolder `hunyuan3d-dit-v2-1`,
  checkpoint variant `fp16`
- MoGe: `models/moge-2-vitl`

Full prompts:
- `01184` stage 1:
  `这是一个rubbish bin的深度图，补全它。输出仍然是灰度深度图。`
- `01184` stage 2:
  `根据这张完整的rubbish bin深度图生成真实rubbish bin照片。只保留物体的轮廓、大小、种类、朝向、姿态和相机视角，严格保持输入图中的2D投影轮廓、物体位置和大小，不要旋转、平移、缩放、换视角或重新构图，不需要保留原图的颜色、材质、光照和背景细节；让物体结构、材质和外观更真实自然，背景使用干净的纯白背景，不要生成桌面、地面、石台、墙面、植物丛或其他环境前景。`
- `09639` stage 1:
  `这是一个Ergonomic Chair的深度图，补全它。输出仍然是灰度深度图。`
- `09639` stage 2:
  `根据这张完整的Ergonomic Chair深度图生成真实Ergonomic Chair照片。只保留物体的轮廓、大小、种类、朝向、姿态和相机视角，严格保持输入图中的2D投影轮廓、物体位置和大小，不要旋转、平移、缩放、换视角或重新构图，不需要保留原图的颜色、材质、光照和背景细节；让物体结构、材质和外观更真实自然，背景使用干净的纯白背景，不要生成桌面、地面、石台、墙面、植物丛或其他环境前景。`
- `05452` stage 1:
  `这是一个armchair的深度图，补全它。输出仍然是灰度深度图。`
- `05452` stage 2:
  `根据这张完整的armchair深度图生成真实armchair照片。只保留物体的轮廓、大小、种类、朝向、姿态和相机视角，严格保持输入图中的2D投影轮廓、物体位置和大小，不要旋转、平移、缩放、换视角或重新构图，不需要保留原图的颜色、材质、光照和背景细节；让物体结构、材质和外观更真实自然，背景使用干净的纯白背景，不要生成桌面、地面、石台、墙面、植物丛或其他环境前景。`
- `05117` stage 1:
  `这是一个red chair的深度图，补全它。输出仍然是灰度深度图。`
- `05117` stage 2:
  `根据这张完整的red chair深度图生成真实red chair照片。只保留物体的轮廓、大小、种类、朝向、姿态和相机视角，严格保持输入图中的2D投影轮廓、物体位置和大小，不要旋转、平移、缩放、换视角或重新构图，不需要保留原图的颜色、材质、光照和背景细节；让物体结构、材质和外观更真实自然，背景使用干净的纯白背景，不要生成桌面、地面、石台、墙面、植物丛或其他环境前景。`
- `06127` stage 1:
  `这是一个terracotta flower pot with leafy plant的深度图，补全它。输出仍然是灰度深度图。`
- `06127` stage 2:
  `根据这张完整的terracotta flower pot with leafy plant深度图生成真实terracotta flower pot with leafy plant照片。只保留物体的轮廓、大小、种类、朝向、姿态和相机视角，严格保持输入图中的2D投影轮廓、物体位置和大小，不要旋转、平移、缩放、换视角或重新构图，不需要保留原图的颜色、材质、光照和背景细节；让物体结构、材质和外观更真实自然，背景使用干净的纯白背景，不要生成桌面、地面、石台、墙面、植物丛或其他环境前景。`
- `07136` stage 1:
  `这是一个leather sofa的深度图，补全它。输出仍然是灰度深度图。`
- `07136` stage 2:
  `根据这张完整的leather sofa深度图生成真实leather sofa照片。只保留物体的轮廓、大小、种类、朝向、姿态和相机视角，严格保持输入图中的2D投影轮廓、物体位置和大小，不要旋转、平移、缩放、换视角或重新构图，不需要保留原图的颜色、材质、光照和背景细节；让物体结构、材质和外观更真实自然，背景使用干净的纯白背景，不要生成桌面、地面、石台、墙面、植物丛或其他环境前景。`
- `07306` stage 1:
  `这是一个red office trash can的深度图，补全它。输出仍然是灰度深度图。`
- `07306` stage 2:
  `根据这张完整的red office trash can深度图生成真实red office trash can照片。只保留物体的轮廓、大小、种类、朝向、姿态和相机视角，严格保持输入图中的2D投影轮廓、物体位置和大小，不要旋转、平移、缩放、换视角或重新构图，不需要保留原图的颜色、材质、光照和背景细节；让物体结构、材质和外观更真实自然，背景使用干净的纯白背景，不要生成桌面、地面、石台、墙面、植物丛或其他环境前景。`
- `06188` stage 1:
  `这是一个red motorcyle的深度图，补全它。输出仍然是灰度深度图。`
- `06188` stage 2:
  `根据这张完整的red motorcyle深度图生成真实red motorcyle照片。只保留物体的轮廓、大小、种类、朝向、姿态和相机视角，严格保持输入图中的2D投影轮廓、物体位置和大小，不要旋转、平移、缩放、换视角或重新构图，不需要保留原图的颜色、材质、光照和背景细节；让物体结构、材质和外观更真实自然，背景使用干净的纯白背景，不要生成桌面、地面、石台、墙面、植物丛或其他环境前景。`
- `06145` stage 1:
  `这是一个table的深度图，补全它。输出仍然是灰度深度图。`
- `06145` stage 2:
  `根据这张完整的table深度图生成真实table照片。只保留物体的轮廓、大小、种类、朝向、姿态和相机视角，严格保持输入图中的2D投影轮廓、物体位置和大小，不要旋转、平移、缩放、换视角或重新构图，不需要保留原图的颜色、材质、光照和背景细节；让物体结构、材质和外观更真实自然，背景使用干净的纯白背景，不要生成桌面、地面、石台、墙面、植物丛或其他环境前景。`
- `06830` stage 1:
  `这是一个tricycle的深度图，补全它。输出仍然是灰度深度图。`
- `06830` stage 2:
  `根据这张完整的tricycle深度图生成真实tricycle照片。只保留物体的轮廓、大小、种类、朝向、姿态和相机视角，严格保持输入图中的2D投影轮廓、物体位置和大小，不要旋转、平移、缩放、换视角或重新构图，不需要保留原图的颜色、材质、光照和背景细节；让物体结构、材质和外观更真实自然，背景使用干净的纯白背景，不要生成桌面、地面、石台、墙面、植物丛或其他环境前景。`
- Negative prompt for all samples: `" "`

Generation parameters:
- Qwen input/output: `depth_image_input_res=512`, `qwen_edit_generate_res=1024`,
  final `generate_res=512`
- Qwen stage 1 steps: `16`
- Qwen stage 2 steps: `16`
- Qwen CFG: `qwen_edit_true_cfg_scale=4.0`
- Qwen seed: UNKNOWN, not explicitly set by current main pipeline
- Backend/scheduler choices: UNKNOWN

Postprocessing:
- This accepted entry covers image-generation outputs only.
- Stage 1 used incomplete depth image to completed depth-like image.
- Stage 2 used the completed depth-like image to generate the final
  realistic/semantic RGB image with pure white background.
- Later RMBG, MoGe, Hunyuan, registration, ICP, and metric outputs remain
  optimization targets and may be overwritten.

Preservation:
- Do not change prompts, regenerate images, or overwrite these `depth.png`,
  `qwen_edit_stage1.png`, or `img.png` files without asking.
- Continue method work only from Stage 2 onward.
- Final metric target remains all 10 samples plus average below the requested
  CD/EMD thresholds. `05117` is only a diagnostic sample and must not be used
  as the sole success criterion.
