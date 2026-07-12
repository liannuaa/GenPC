# Project State

This file records accepted experiment outputs and fragile parameters. Update it
immediately when the user approves a result, says an effect is good/correct, or
asks to preserve a pipeline state.

## Accepted Results

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
