# Project State

This file records accepted experiment outputs and fragile parameters. Update it
immediately when the user approves a result, says an effect is good/correct, or
asks to preserve a pipeline state.

## Accepted Results

### 2026-09-04 CST - Agent Intrinsic Local Geometry Route

Status: accepted research direction. The user said this route "看起来很有前途"
and requested that subsequent work focus on it. Preserve the implementation,
shared parameters, and the following audit artifacts unless the user requests
a deliberate replacement. This is an accepted *route*, not a claim that it is
already the final full-ten pipeline.

Samples and outputs:

- `07136` action output:
  `workspace/agent_genpc_plus_scratch_20260903/_agent_intrinsic_relaxed_07136/registered_100k.ply`
- `07136` mesh output:
  `workspace/agent_genpc_plus_scratch_20260903/_agent_intrinsic_relaxed_07136/registered_mesh.glb`
- `07136` decoded output:
  `workspace/agent_genpc_plus_scratch_20260903/_agent_intrinsic_relaxed_07136_uniform32k/07136/07136_agent_intrinsic_registered_uniform.ply`
- `09639` action output:
  `workspace/agent_genpc_plus_scratch_20260903/_agent_intrinsic_relaxed_09639/registered_100k.ply`
- Action records:
  `workspace/agent_genpc_plus_scratch_20260903/_agent_intrinsic_relaxed_07136/intrinsic_residual_action.json`
  and
  `workspace/agent_genpc_plus_scratch_20260903/_agent_intrinsic_relaxed_09639/intrinsic_residual_action.json`

Inputs:

- Raw partials: `data/07136.ply`, `data/09639.ply`
- Saved cameras and semantic views:
  `workspace/agent_genpc_plus_scratch_20260903/<sample>/camera.pth` and
  `workspace/agent_genpc_plus_scratch_20260903/<sample>/img.png`
- Fresh registered anchors:
  `workspace/agent_genpc_plus_scratch_20260903/_agent_pca_router_2d3d_all10_v3/<sample>/<sample>_agent_bidirectional_registered_100k.ply`
  and corresponding `registered_mesh.glb`.

Model/checkpoint:

- No new generative model is invoked by this action. It optimizes vertices of
  the already generated and registered complete mesh.
- Upstream semantic/Pixal3D model and exact upstream checkpoint provenance:
  `UNKNOWN` in this action record; do not infer it from a later run.

Agent instruction, verbatim for both audited samples:

`Keep the exact saved-camera view, object identity, global pose, global proportions, structural parts, and hidden geometry unchanged. Do not rotate, crop, mirror, recenter, add, or remove parts. Only on the visible central surface region, gently contract inward so that the local outer surface follows the observed scan. Keep every other visible and hidden region unchanged; make the correction local and smooth.`

Negative prompt:

- `UNKNOWN / not applicable`: no image generator is called by this mesh-local
  action.

Shared generation/optimization parameters:

- Proxy triangles `12000`; correspondence samples `50000`; maximum handles
  `96`; seed `6145`.
- Geodesic support inner/outer/anchor ratios `.055/.125/.145` of partial-bbox
  diagonal.
- Maximum handle/vertex displacement `.055/.045` of partial-bbox diagonal.
- Continuation lattice
  `[1,.75,.5,.35,.25,.15,.1,.05,.035,.025,.015,.01,.005]`.
- Edge stretch q01/q99 limits `.78/1.28`; flipped-face ratio limit `8e-4`.
- Proper Sim(3) registration is frozen before this local action; fixed
  saved-camera and PCA-frame three-view no-harm gates remain active.

Postprocessing:

- Decimated-proxy screened Laplacian/ARAP-style solve; explicit removal of
  translation, rotation, and isotropic-scale modes; displacement transfer only
  to original mesh vertices.
- No mesh face or hidden support deletion; standard observation-conditioned
  posterior and support-aware voxel resampling only after an accepted action.
- `07136` post-selection-only metric: CD/EMD x1e2 `2.521/3.612`; metrics were
  not used for routing.

What was approved:

- Make this intrinsic local geometry action the primary research direction.
- Keep the generic gate and verify it across the full ten samples before
  claiming a mainline replacement.

Do not change without asking:

- The local-action safety contract and the above accepted audit artifacts.

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

## 2026-08-21 08:13 +08:00 - Redwood hierarchical router metric-only negative checkpoint

User approval:
- The user requested that the code and experiment result be saved once the
  Redwood average exceeds the GenPC paper target.
- Primary accepted criterion: full-10 mean CD-L1 < 1.74 and mean EMD < 2.88;
  individual samples may remain above their paper row.
- Do not overwrite this output, reproducibility bundle, or Git tag without
  asking.

Samples and official categories:
- `01184` Wheelie Bin; `05117` chair; `05452` armchair; `06127` Plant vases;
  `06145` table; `06188` vespa; `06830` Kid tricycle; `07136` sofa; `07306`
  trash can; `09639` swivel chair.
- Mapping source: upstream GenPC `utils/dataUtils.py`; frozen target file:
  `configs/redwood_genpc_paper_targets.csv`.

Inputs:
- Raw partial point clouds: `data/<sample>.ply`.
- GT point clouds used only after output freeze: `data/GT/<sample>.ply`.
- Image/camera root:
  `workspace/redwood_onestage_rawdepth_512_stage2_20260714/<sample>/` containing
  `raw_depth.png`, `img.png`, `img_sam.png`, `camera.pth`, and `point_uv.npy`.
- Frozen upstream candidate roots are recorded verbatim in
  `workspace/redwood_onestage_rawdepth_512_stage2_20260714/_hierarchical_prior_router_20260821/completion_manifest.json`.

Accepted outputs:
- Final PLY root:
  `workspace/redwood_onestage_rawdepth_512_stage2_20260714/_hierarchical_prior_router_20260821/hierarchical_prior_router/<sample>/<sample>_fused.ply`.
- Metrics:
  `workspace/redwood_onestage_rawdepth_512_stage2_20260714/_hierarchical_prior_router_20260821/metrics_samples.csv`,
  `metrics_summary.csv`, and `metrics_vs_genpc_paper_samples.csv`.
- Reproducibility bundle, including all ten final PLYs and SHA-256 hashes:
  `reproducibility/redwood_genpc_paper_average_20260821`.
- Preservation tag: `redwood-genpc-paper-average-20260821`.

Accepted metrics (GenPC 16,384-point FPS protocol, metric seed 6145):
- Mean CD-L1 x1e2: `1.7198891611769795` versus paper `1.74`.
- Mean EMD x1e2: `2.5092773232609034` versus paper `2.88`.
- Paper-table rounding: `1.72/2.51`.
- Per-sample double wins: 5/10, recorded for diagnosis but not required by the
  approved primary criterion.

Models/checkpoints:
- Qwen pipeline: `models/Qwen-Image-Edit-2511`.
- Qwen transformer:
  `models/nunchaku-qwen-image-edit/nunchaku_qwen_image_2511_balance_int4.safetensors`.
- RMBG: `models/RMBG-2.0`.
- Hunyuan image prior: `models/Hunyuan3D-2.1`, subfolder
  `hunyuan3d-dit-v2-1`, fp16 variant.
- Point/image conditional prior: `models/Hunyuan3D-Omni`, EMA variant.
- MoGe: `models/moge-2-vitl`.

Full Qwen prompts and negative prompts:
- `01184`: `生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的rubbish bin，纯白背景`
- `05117`: `生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的red chair，纯白背景`
- `05452`: `生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的armchair，纯白背景`
- `06127`: `生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的terracotta flower pot with leafy plant，纯白背景`
- `06145`: `生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的table，纯白背景`
- `06188`: `生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的red motorcyle，纯白背景`
- `06830`: `生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的tricycle，纯白背景`
- `07136`: `生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的leather sofa，纯白背景`
- `07306`: `生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的red office trash can，纯白背景`
- `09639`: `生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的Ergonomic Chair，纯白背景`
- Negative prompt for all ten samples: ` ` (one space).

Generation parameters:
- Qwen input `raw_depth.png`: 512; generation resolution: 1024; final pipeline
  resolution: 512; 40 inference steps; true CFG 4.0; refine stage false.
- Qwen seed: UNKNOWN; not recorded by the source run, so the image stage is not
  exactly reproducible from seed alone.
- Qwen scheduler/backend beyond the Nunchaku transformer: UNKNOWN.
- Hunyuan3D-2.1: 50 shape steps, octree resolution 384; deterministic extra
  candidates seeds 101/102/103. The original candidate seed is UNKNOWN.
- Hunyuan3D-Omni: seed 6145, 50 steps, guidance scale 4.5, octree resolution
  512, EMA weights, 200000 surface samples for the retained no-overlap-cut
  branch.
- Multi-start pose posterior: Hunyuan seed 101, top starts 6, train/validation/
  final-holdout raw-ray split 96/96/96.

Postprocessing and selection:
- Raw partial is preserved exactly in observation-first branches.
- Sensor prior gate uses normalized cost <= 0.04 and no overlap cut for the
  accepted branch.
- Pose posterior requires an independent final holdout-ray acceptance.
- Remaining routing uses symmetric support bands 1/6, 1/4, and 1/3 to select
  exact raw partial, midpoint consensus, 75% posterior mass, or 100% posterior
  mass.
- Cross-prior support distance is 0.02D; residual voxel size is 0.003D.
- No category, sample ID, or GT metric is used by the inference router. GT is
  loaded only by the post-freeze metric pass.

Verification and preservation boundary:
- Full repository tests: 199/199 passed with the explicit `genpc` interpreter.
- The result is accepted as the preserved Redwood best checkpoint.
- It is not yet a cross-domain generalization claim: pose-posterior candidates
  were available for five difficult Redwood cases. Run the same candidate
  schedule for every Omni-Comp3D, KITTI, and Waymo case before a paper-wide
  claim, without changing the frozen router thresholds.


### 2026-08-21 visual-review correction

The user inspected the bundled PLY files and explicitly rejected the method:
most results are not complete objects, and raw-partial fallback is invalid for
a completion task. The existing files and metrics are preserved only as a
negative reproducibility record. They are not an accepted visual baseline or
PAMI result, and the proposed success tag must not be created.

Exact raw fallback occurs for `05452` and `06830`. For `06145`, only 0.62%
of output points lie farther than `0.02D` from the partial despite the larger
point count. All future methods must retain a full generated candidate; failure
is recorded as `completion_failed=true`, never hidden by substituting raw
partial.

## 2026-08-21 14:21 CST — accepted Qwen-2511 cardinal-view direction

User approval boundary:
- The user explicitly requested that future multiview semantic generation use Qwen-Image-Edit-2511.
- This accepts the generator choice and joint four-cardinal-view protocol, not the current `06145` board as a final registered/fused result.
- Keep MV-Adapter only as an ablation unless the user later asks to promote it.

Sample and exact artifacts:
- Sample: `06145`.
- Input semantic image: `workspace/redwood_onestage_rawdepth_512_stage2_20260714/06145/img_sam.png`.
- Joint output board: `workspace/redwood_onestage_rawdepth_512_stage2_20260714/_qwen_cardinal_hunyuan2mv_viewframe_20260821/06145/06145_qwen_cardinal_board.png`.
- Split views: `06145_qwen_front.png`, `06145_qwen_right.png`, `06145_qwen_back.png`, and `06145_qwen_left.png` in the same output directory.
- Diagnostic: `workspace/redwood_onestage_rawdepth_512_stage2_20260714/_qwen_cardinal_hunyuan2mv_viewframe_20260821/06145/06145_qwen_cardinal_info.json`.
- Prompt record: `workspace/redwood_onestage_rawdepth_512_stage2_20260714/_qwen_cardinal_hunyuan2mv_viewframe_20260821/06145/06145_qwen_cardinal_prompt.txt`.

Models/checkpoints:
- Pipeline: `models/Qwen-Image-Edit-2511`.
- Transformer: `models/nunchaku-qwen-image-edit/nunchaku_qwen_image_2511_balance_int4.safetensors`.
- Backend: `QwenImageEditPlusPipeline` with Nunchaku int4 transformer and CPU offload.

Full prompt (verbatim):
`图1是一个已经补全的完整物体在真实相机下的语义图。只旋转相机，不改变物体，生成同一个物体的2×2正交四视图。四个格子的固定顺序是：左上为输入的真实前视（0度），右上为右视图（相机绕物体水平旋转90度），左下为背视图（180度），右下为左视图（270度）。四个相机必须保持相同高度、俯仰角、距离、焦距和物体尺度。严格保持物体身份、整体长宽高比例、部件数量、连接关系、厚度和局部结构一致；被遮挡部分必须在不同视角中形成同一套三维结构。平面必须保持平整，水平表面保持水平，直边保持笔直，不得出现弯曲、倾斜、拉伸、重复部件或视角间结构漂移。每个格子仅包含一个居中且完整的物体，使用纯白背景；不要添加文字、标签、分隔标题、地面、阴影或其他物体。`

Generation parameters:
- Negative prompt: ` ` (one space, verbatim).
- Seed: `6145`.
- Inference steps: `40`.
- True CFG scale: `4.0`.
- Requested board resolution: `1024×1024`; four fixed `512×512` crops.
- Resize policy: `generate_multiview_board` resizes the returned image to exactly `1024×1024` with Lanczos only if necessary.
- View order: top-left front/0°, top-right right/90°, bottom-left back/180°, bottom-right left/270°.
- Scheduler details beyond the locally loaded pipeline defaults: UNKNOWN; exact local model and transformer paths are recorded above.

Postprocessing and current verification:
- No RMBG, depth, normal, MoGe, registration, ICP, deformation, or fusion was applied to this board.
- Fixed quadrant cropping only; foreground masks are estimated from border color for a GT-free admission diagnostic.
- All four quadrants contain complete foreground. The front-reference bounded-Sim2 IoU is `0.4038781989440634`, scale `0.95`, translation `[-6, 24]`; therefore it fails the known-camera front-silhouette gate.
- Visual audit shows a coherent complete pedestal table and a substantially planar tabletop, but Qwen canonicalized the camera elevation. It is eligible only as a `canonical_orbit` complete-shape candidate followed by fresh global Sim3 registration and strict raw-ray validation.
- Do not present this board or any future Hunyuan PLY derived from it as final success until registration, fusion, visual audit, and Redwood metrics pass.

## 2026-08-21 15:58 CST — accepted 06145 three-view Hunyuan complete-prior baseline

User approval boundary:
- The user visually approved the main completed object generated from the latest Qwen board and requested only removal of a small disconnected external outlier component.
- Preserve the source candidate exactly. Any cleaned point cloud is a separate, reversible derivative and is not yet an accepted registered/fused prediction.

Artifacts and inputs:
- Sample: `06145`; known test-time category: `table` (CLI text condition; no GT geometry or metric used).
- Semantic reference: `workspace/redwood_onestage_rawdepth_512_stage2_20260714/06145/img_sam.png`.
- Qwen board: `workspace/redwood_onestage_rawdepth_512_stage2_20260714/_qwen_cardinal_table_viewnames_axisaligned_hunyuan2mv_20260821/06145/06145_qwen_cardinal_board.png`.
- Hunyuan inputs only: `06145_qwen_front.png`, `06145_qwen_left.png`, and `06145_qwen_right.png` in the same directory. `06145_qwen_back.png` is explicitly excluded.
- Accepted source PLY: `workspace/redwood_onestage_rawdepth_512_stage2_20260714/_qwen_cardinal_table_viewnames_axisaligned_hunyuan2mv_20260821/generated_candidates/06145/06145_qwen_cardinal_front-left-right_hunyuan2mv_seed6145_canonical.ply`.
- Native PLY: `workspace/redwood_onestage_rawdepth_512_stage2_20260714/_qwen_cardinal_table_viewnames_axisaligned_hunyuan2mv_20260821/generated_candidates/06145/06145_qwen_cardinal_front-left-right_hunyuan2mv_seed6145_native.ply`.
- Visual check: `workspace/redwood_onestage_rawdepth_512_stage2_20260714/_qwen_cardinal_table_viewnames_axisaligned_hunyuan2mv_20260821/06145/06145_qwen_front-left-right_hunyuan2mv_turntable.png`.

Models and parameters:
- Qwen pipeline: `models/Qwen-Image-Edit-2511`; Nunchaku transformer: `models/nunchaku-qwen-image-edit/nunchaku_qwen_image_2511_balance_int4.safetensors`; backend: `QwenImageEditPlusPipeline` with CPU offload.
- Qwen seed `6145`; 40 steps; true CFG `4.0`; negative prompt is one space; board `1024×1024`, split into 512×512 cards; scheduler details: UNKNOWN.
- Hunyuan checkpoint: `models/Hunyuan3D-2mv-modelscope/hunyuan3d-dit-v2-mv-turbo/model.fp16.safetensors`; seed `6145`; 5 steps; octree resolution `256`; 8000 chunks; 100000 sampled surface points.
- Generated mesh summary: 167360 vertices and 334716 faces.

Full Qwen prompt (verbatim):
`生成一张图像，参考图1中table的大致尺寸和种类，并遵循以下描述：生成一个比图1更清晰、完整的table的2×2四视图拼图：左上是同一物体的正视图，右上是同一物体的左侧视图，左下是同一物体的背视图，右下是同一物体的右侧视图。四个格子不得改变物体的形状、部件或比例。所有格子均为轴对齐的标准正交视图：物体竖直方向与图像竖直轴对齐，水平和竖直边平行图像边缘；禁止俯仰、滚转、斜视和透视畸变。纯白背景，不要添加文字、标签或其他物体。`

Postprocessing and preservation:
- The PLY is obtained by Hunyuan surface sampling, then `hunyuan2mv_to_canonical_world` applies robust 1%/99% isotropic normalization and the fixed proper rotation `(x,y,z) -> (x,-z,y)`.
- No RMBG, depth/normal control, raw partial replacement, GT selection, registration, ICP, deformation, or fusion has been applied.
- Do not alter the accepted source PLY, board, seed, prompt, or three-view selection without user direction.

## 2026-08-21 16:09 CST — accepted 06145 mesh-clean complete-prior candidate

User approval boundary:
- The user approved removal of the external disconnected island and accepted the cleaned complete prior for the next registration/fusion stage.
- Preserve the source candidate and use the cleaned PLY as the complete-shape input; do not make it a final fused prediction until visible-surface registration and completeness-preserving fusion pass.

Exact cleaned artifacts:
- Cleaned canonical PLY: `workspace/redwood_onestage_rawdepth_512_stage2_20260714/_qwen_cardinal_table_viewnames_axisaligned_hunyuan2mv_20260821/generated_candidates/06145/06145_qwen_cardinal_front-left-right_hunyuan2mv_seed6145_meshcc0p006_canonical.ply`.
- Raw mesh: `workspace/redwood_onestage_rawdepth_512_stage2_20260714/_qwen_cardinal_table_viewnames_axisaligned_hunyuan2mv_20260821/generated_candidates/06145/06145_qwen_cardinal_front-left-right_hunyuan2mv_seed6145_native_mesh.glb`.
- Cleaned mesh: `workspace/redwood_onestage_rawdepth_512_stage2_20260714/_qwen_cardinal_table_viewnames_axisaligned_hunyuan2mv_20260821/generated_candidates/06145/06145_qwen_cardinal_front-left-right_hunyuan2mv_seed6145_main_components_mesh.glb`.
- Preview: `workspace/redwood_onestage_rawdepth_512_stage2_20260714/_qwen_cardinal_table_viewnames_axisaligned_hunyuan2mv_20260821/06145/06145_qwen_front-left-right_meshcc0p006_turntable.png`.

Cleanup contract:
- Input has two disconnected components: 332812 faces (99.4312%) and 1904 faces (0.5688%).
- The fixed mesh face-fraction threshold `0.006` removed only the 1904-face island; 332812 faces remain.
- Source generation, Qwen board/prompt, model paths, seeds, Hunyuan settings, and the `front/left/right` input selection are exactly those recorded in the 15:58 CST accepted baseline immediately above.
- No raw partial replacement, registration, ICP, deformation, or fusion has occurred. Do not overwrite this cleaned PLY without user direction.

## 2026-08-22 01:07 CST — accepted GPT ImageGen to Pixal3D Redwood ten-sample baseline

Status and preservation boundary:
- The user inspected all ten generated objects and said they were all generated very well. This accepts the GPT semantic images and corresponding Pixal3D GLB/100k PLY files as the frozen complete-shape baseline for the next registration/fusion stage.
- Samples: `01184`, `05117`, `05452`, `06127`, `06145`, `06188`, `06830`, `07136`, `07306`, `09639`.
- Do not overwrite `gpt_version/<sample>/gpt_image.png`, `pixal3d.glb`, or `pixal3d_sampled_100k.ply` without asking. Registration, cleanup, and fusion outputs must be separate derivative files.
- This approval covers visual complete-shape quality. It does not yet approve registration, fused predictions, or Redwood CD/EMD results.

Inputs and exact artifacts:
- Raw partial-depth inputs: `gpt_version/<sample>/depth.png`, unchanged 512x512 copies of the accepted depth renders.
- GPT semantic outputs: `gpt_version/<sample>/gpt_image.png`, 1254x1254.
- GPT prompt records: `gpt_version/<sample>/prompt.txt`.
- Additional user-provided product references: `gpt_version/01184/product_reference.png` and `gpt_version/05452/product_reference.png`. Other samples have no separate product reference.
- Pixal3D preprocessed inputs: `gpt_version/<sample>/pixal3d_input.png`.
- Accepted textured meshes: `gpt_version/<sample>/pixal3d.glb`.
- Accepted sampled complete point clouds: `gpt_version/<sample>/pixal3d_sampled_100k.ply`.
- Per-sample exact run records: `gpt_version/<sample>/pixal3d_metadata.json`.
- Batch statistics and visual board: `gpt_version/pixal3d_summary.csv` and `gpt_version/pixal3d_contact_sheet.png`.

Models and checkpoints:
- Semantic image generator: OpenAI built-in GPT image generation editor. Exact serving checkpoint/version: `UNKNOWN`; the result is therefore not exactly reproducible from local weights.
- Pixal3D source: `models/Pixal3D`, TencentARC/Pixal3D.
- Pixal3D weights: `models/Pixal3D-weights`; all seven projection checkpoints listed by `pipeline.json`.
- DINOv3: `models/dinov3-vitl16-pretrain-lvd1689m/model.safetensors`.
- MoGe-2 camera estimator: `models/moge-2-vitl/model.pt`.
- Background removal: `models/RMBG-2.0/model.safetensors`.
- NAF: official `valeoai/NAF` Torch Hub code and `/root/.cache/torch/hub/checkpoints/naf_release.pth`.
- Attention backend: `xformers`; `flash_attn` is not installed.
- The original ModelScope `tex_dec_next_dc_f16c32_fp16.safetensors` was corrupt. The active copy was re-downloaded from `TencentARC/Pixal3D` on ModelScope and validated. The corrupt copy is retained with suffix `.corrupt_20260822`.

Full GPT edit prompts, verbatim:
- `01184`: `Use case: sketch-to-render. Asset type: zero-shot completion reference image. Input images: Image 1 is the user's real-product photo and is authoritative for object type and wheel construction; Image 2 is the raw depth map and is authoritative for camera viewpoint, silhouette, pose, placement, and scale. Generate a photorealistic complete two-wheeled rubbish bin on a pure white studio background. Critical correction: use the standard wheel arrangement shown in Image 1—exactly two equal wheels, one on the left side and one on the right side of the bin, mounted coaxially on the same single horizontal rear axle. Their axle centers lie on one straight line; their wheel planes are parallel. Because of the three-quarter perspective, the near-side wheel appears lower/front in the image and the far-side wheel appears behind on the opposite side, but they are NOT a tandem pair on one side. Preserve the high oblique viewpoint, narrow bin-body silhouette, lid, pose, and image footprint from Image 2. Both wheels should be visible as in Image 1, with the far wheel partly occluded by the bin. Complete object fully in frame. No extra wheels, no casters, no same-side tandem wheels, no floor, cast shadow, text, watermark, people, or other objects.`
- `05117`: `Use case: sketch-to-render. 图1是一张 red chair（红色椅子）的局部深度图。请严格依据图1的深度轮廓、相机视角、朝向、姿态、位置与比例，将缺失区域自然补全，生成同一个完整 red chair 的真实产品照片。完整物体必须全部位于画面内，纯白摄影棚背景。不要改变视角，不要添加或删除功能部件，不要添加地面、阴影、文字、水印或其他物体。`
- `05452`: `Use case: sketch-to-render. Asset type: zero-shot completion reference image. Input images: Image 1 is the user's real-chair reference and is authoritative for the curved chair construction; Image 2 is the raw depth map and is authoritative for camera viewpoint, silhouette, pose, placement, and scale. Generate a photorealistic complete armchair on a pure white studio background. Critical shape: this is a thin upholstered chair with a continuous curved side profile like Image 1, not two flat rectangular cushions. The tall backrest reclines backward and curves smoothly inward through the lumbar transition into the seat; the seat has a shallow ergonomic concave curve and its front edge bends gently downward. Keep the shell/padding visually thin while preserving these compound curves. Use slim tubular metal side frames that form arched armrests and continue into curved front and rear legs, matching the visible arcs in Image 2. Preserve Image 2's near-side profile view, orientation, footprint, and proportions. Complete object fully in frame. Do not make the back or seat flat, boxy, thick, overstuffed, or sofa-like. No floor, cast shadow, text, watermark, people, or other objects.`
- `06127`: `Use case: sketch-to-render. 图1是一张 terracotta flower pot with leafy plant（带叶植物的陶土花盆）的局部深度图。请严格依据图1的深度轮廓、相机视角、朝向、姿态、位置与比例，将缺失区域自然补全，生成同一个完整对象的真实产品照片。完整花盆与植株必须全部位于画面内，纯白摄影棚背景。不要改变视角，不要添加或删除花盆、枝干或主要叶片结构，不要添加地面、阴影、文字、水印或其他物体。`
- `06145`: `Use case: sketch-to-render. 图1是一张 table（单柱圆底座方桌）的局部深度图。请严格依据图1的深度轮廓、相机视角、朝向、姿态、位置与比例，将缺失区域自然补全，生成同一个完整 table 的真实产品照片。保持方形桌面、中央单柱和圆形底座，完整物体必须全部位于画面内，纯白摄影棚背景。不要改变视角，不要改成四腿桌，不要添加地面、阴影、文字、水印或其他物体。`
- `06188`: `Use case: sketch-to-render. 图1是一张 red Vespa-style motor scooter（红色踏板摩托车）的局部深度图。请严格依据图1的深度轮廓、相机视角、朝向、姿态、位置与比例，将缺失区域自然补全，生成同一辆完整踏板摩托车的真实产品照片。完整车辆必须全部位于画面内，纯白摄影棚背景。不要改变视角，不要改变车轮、车身、车把和座椅的布局，不要添加骑手、地面、阴影、文字、水印或其他物体。`
- `06830`: `Use case: sketch-to-render. Asset type: zero-shot completion reference image. Input image: Image 1 is the sole authoritative target for geometry, camera, pose, projected size, composition, and steering state. Convert this partial depth rendering into a photorealistic complete child tricycle with a tall parent push handle, while preserving image-space registration. Hard constraints: keep the same high oblique viewpoint, azimuth, elevation, roll, tricycle orientation, center position, normalized bounding box, width-to-height ratio, and canvas occupancy as Image 1. Preserve the tall oval push handle, compact seat/body, handlebar, rear-wheel layout, and strong foreshortening. Critical front-wheel pose: the single front wheel and its fork are intentionally turned slightly to one side, approximately 10–15 degrees of steering yaw relative to the tricycle centerline, exactly following the asymmetric angled wheel/fork evidence in the lower part of Image 1. The front wheel must look mechanically straight within its fork but visibly steered—not centered, not parallel to the body axis, not severely twisted or deformed. Complete only missing or occluded parts. Do not rotate to a canonical product view, recenter, zoom, enlarge, straighten the steering, or symmetrize the observed pose. Use neutral light-colored materials on a pure white background. Complete object fully in frame. No rider, floor, cast shadow, text, watermark, or extra objects.`
- `07136`: `Use case: sketch-to-render. Asset type: zero-shot completion reference image. Input image: Image 1 is the sole and authoritative edit target for geometry, camera, pose, projected size, and composition. Convert this partial depth rendering into a photorealistic complete leather sofa while preserving image-space registration. Hard constraints: keep exactly the same high oblique camera viewpoint, azimuth, elevation, roll, sofa orientation, center position, normalized bounding box, width-to-height ratio, and fraction of the canvas occupied by the visible sofa. The output silhouette of every observed region—especially the long near arm/back edge, the recessed inner gap, and the large right seat/back plane—must align with Image 1. Complete only genuinely missing/occluded geometry; do not rotate to a canonical front view, do not make it face the camera, do not zoom in or out, do not recenter, and do not enlarge the sofa to fill the canvas. Preserve the strong foreshortening and mostly side/rear/top view from the depth image. Apply realistic dark-brown leather material with restrained seams. Pure white background, complete object inside frame. No room, floor, cast shadow, pillows, text, watermark, people, or extra objects.`
- `07306`: `Use case: sketch-to-render. 图1是一张 red office trash can（红色办公室垃圾桶）的局部深度图。请严格依据图1的深度轮廓、相机视角、朝向、姿态、位置与比例，将缺失区域自然补全，生成同一个完整 red office trash can 的真实产品照片。完整物体必须全部位于画面内，纯白摄影棚背景。保持圆筒形桶身和顶部开口结构；不要改成带轮户外垃圾桶，不要添加垃圾袋、地面、阴影、文字、水印或其他物体。`
- `09639`: `Use case: sketch-to-render. 图1是一张 ergonomic swivel office chair（人体工学旋转办公椅）的局部深度图。请严格依据图1的深度轮廓、相机视角、朝向、姿态、位置与比例，将缺失区域自然补全，生成同一把完整办公椅的真实产品照片。完整椅子必须全部位于画面内，纯白摄影棚背景。保持高靠背、座垫、扶手、中央气杆和五星脚轮底座；不要改变视角，不要添加房间、地面、阴影、文字、水印或其他物体。`

Generation parameters:
- GPT ImageGen negative prompt: not exposed by the editor, `UNKNOWN`/not applicable.
- GPT ImageGen seed, steps, guidance, scheduler, and backend details: `UNKNOWN`; this semantic-image stage is not exactly reproducible.
- GPT output resolution: 1254x1254. No crop/recenter was applied before saving `gpt_image.png`.
- Pixal3D seed: `42` for every sample; no sample-specific seed or parameter.
- Pixal3D pipeline: `1024_cascade`; actual resolution 1024 for all ten.
- Sparse structure sampler: 12 steps, guidance strength 7.5, guidance rescale 0.7, rescale_t 5.0.
- Shape sampler: 12 steps, guidance strength 7.5, guidance rescale 0.5, rescale_t 3.0.
- Texture sampler: 12 steps, guidance strength 1.0, guidance rescale 0.0, rescale_t 3.0.
- GLB export: target 300000 faces, 2048 texture, remesh enabled, remesh_band 1, remesh_project 0, official Pixal3D output rotation.
- PLY export: deterministic uniform surface sample of exactly 100000 points using seed 42, with texture-derived point colors.

Postprocessing and verification:
- Pixal3D RMBG preprocessing removes the white background, crops the foreground with the official 1.1 margin, and saves the exact RGBA condition as `pixal3d_input.png`.
- MoGe-2 estimates camera FOV/distance from that preprocessed image; exact values are in each metadata JSON.
- No category prompt, GT geometry, GT metric, sample-specific registration, ICP, deformation, partial replacement, or fusion was used by Pixal3D.
- All GLBs reload as non-empty meshes with 286658--299511 faces. All PLYs reload with 100000 finite points.
- `05452` has a very small disconnected point/mesh island visible below the chair. Preserve the accepted raw GLB/PLY; any generic connected-component cleanup must be a separate derivative.

What the user approved:
- The complete-object quality of all ten GPT ImageGen to Pixal3D results.
- Use these frozen complete point clouds for the next direct registration/fusion method.


## 2026-08-22 12:24 CST - accepted guarded unified Pixal registration baseline

Status and approval boundary:
- The user approved the current registration as good enough to become the fusion baseline, while explicitly noting unavoidable local residuals at `09639` chair legs and the `07136` sofa.
- Accepted root: `gpt_version/_pixal_guarded_unified_registration_v15_20260822`.
- Preserve every registered 100k PLY, transformed mesh, transform matrix, projection diagnostic, and gray-partial/red-Pixal comparison in that root. Fusion must write separate derivatives and must not overwrite these files.

Inputs, method, and reproducibility:
- Samples: `01184`, `05117`, `05452`, `06127`, `06145`, `06188`, `06830`, `07136`, `07306`, `09639`.
- Raw observations: `data/<sample>.ply`; saved camera: `workspace/redwood_onestage_rawdepth_512_stage2_20260714/<sample>/camera.pth`.
- Frozen complete input: `gpt_version/<sample>/pixal3d_sampled_100k.ply` and `gpt_version/<sample>/pixal3d.glb`, generated by TencentARC/Pixal3D with the exact models/checkpoints and seed-42 settings recorded in the accepted `2026-08-22 01:07 CST` entry above.
- Registration implementation: `scripts/select_pixal_guarded_unified_registration_v15.py`; method `guarded_fast_so3_residual_over_genpc_pca_v15` with one shared, GT-free visible-partial 2D+3D test-time selection policy.
- Output pattern: `gpt_version/_pixal_guarded_unified_registration_v15_20260822/<sample>/<sample>_unified_registration_v14_{registered_100k.ply,registered_mesh.glb,partial_gray_pixal_red.ply,projection.png,info.json}` plus `<sample>_unified_registration_v14.npy`.
- Prompt and negative prompt: not applicable; registration makes no generative model call and reuses frozen Pixal geometry. No RMBG, MoGe regeneration, image generation, ICP deletion, local deformation, fusion, or GT metric is performed by this selector.
- Full shared parameters, selected source candidate, transform, gates, and timing are stored verbatim in each `<sample>_unified_registration_v14_info.json`.

What must not change without asking:
- Do not regenerate the accepted GPT images or Pixal GLBs/PLYs.
- Do not overwrite the accepted v15 transforms or registered outputs.
- Local fusion may correct residual part mismatch only as a separate derivative; it must keep the Pixal complete body as the completeness source and preserve the raw partial exactly.
## 2026-08-22 19:34 CST — accepted rollback to pre-official-reference Omni Redwood images

- Scope/sample ids: Omni single-scan Redwood `01027`, `01032`, `01382`,
  `01833`, `07089`, `07155`, `08310`, `08719`, `09643`, and `09862`.
- Active outputs: `omni_test/redwood_preprocessed__<id>__single_scan__partial_0/gpt_image.png`.
- Inputs: matching case `depth.png`; category-name conditioning. Official
  Redwood photographs are not active generation inputs.
- Generator/model: OpenAI built-in image generation editor; exact serving
  checkpoint/version is `UNKNOWN` and the results are not exactly reproducible
  from local weights.
- Restored generated sources by id: `01027` `exec-b69aa870-57ac-483b-9b3a-7a1c629f184f.png`;
  `01032` `exec-ef761dbd-41c4-45b8-be19-d03ef4c915c6.png`; `01382`
  `exec-e2843182-e648-45ad-93c1-014222132e8f.png`; `01833`
  `exec-4330d1c9-9512-4f04-b9fd-dbdf19d6baff.png`; `07089`
  `exec-8f2e3aff-ddd3-44b7-b902-6015f6944ab2.png`; `07155`
  `exec-2e256882-ff6b-4aa5-a659-145a9817dd8e.png`; `08310`
  `exec-91fb0d9a-d781-46ab-95f5-63c8d67194ba.png`; `08719`
  `exec-6f0466cb-3932-40d0-98ee-0cb00b6023c7.png`; `09643`
  `exec-7bc80722-a1b4-4138-9a9b-4a0f93f191a1.png`; `09862`
  `exec-c37809e8-64a9-4b8c-8b06-77d2a3431a69.png`, all under
  `/root/.codex/generated_images/01a023f5-31b8-7f90-8151-9ed4028ea251/`.
- Full prompts and negative prompts: `UNKNOWN` for this restored batch; do not
  claim exact reproducibility. Resolution/resize policy, seed, steps, guidance,
  scheduler, and backend choices: `UNKNOWN`.
- Postprocessing: generated image copied as `gpt_image.png`; no official-photo
  conditioning, MoGe, Pixal3D, registration, or fusion has yet been applied in
  this Omni stage.
- User approval: “redwood还是之前版本的比较好，不用参考官网的了”. Preserve
  these restored files and do not overwrite them with the official-reference
  variants without asking.


## 2026-08-23 CST — retained 01184 visible affine-TTO diagnostic (not promoted)

- User feedback: the output is "好了一些" relative to the isotropic TTO, but the side width remains mismatched. Preserve it as an intermediate diagnostic, not an accepted final registration.
- Input partial/camera: `data/01184.ply`; `workspace/redwood_onestage_rawdepth_512_stage2_20260714/01184/camera.pth`.
- Frozen complete prior: `gpt_version/_gpt_multiview_hunyuan2mv_redwood_20260823/01184_wheelaxis_v2/01184/01184_hunyuan2mv_canonical.ply`; source GLB `01184_hunyuan2mv_native_mesh.glb`.
- Starting fixed-pose registration: `gpt_version/_gpt_multiview_hunyuan2mv_redwood_20260823/_wheelaxis_v2_gpu_2d3d_locked_scale6/01184`.
- Diagnostic outputs: `gpt_version/_gpt_multiview_hunyuan2mv_redwood_20260823/_wheelaxis_v2_visible_affine_tto/01184/01184_visible_sim3_tto_{partial_gray_pixal_red.ply,registered_100k.ply,projection.png,info.json}`.
- Method: GPU fixed-rotation visible 2D+3D TTO, 4 outer rounds × 100 Adam steps. It optimizes bounded aligned-frame scale and translation from projected bidirectional point distance, same-view depth, and one-sided partial-to-visible distance. Final scale xyz `[0.5126575, 0.6313244, 0.6526875]`; the third axis reached the shared +18% bound.
- Prompt/negative prompt: not applicable; no image generation. Seed, scheduler, CFG: not applicable. Source asset prompt and checkpoint details: `UNKNOWN` in this experiment record; see frozen asset metadata for the original generation details.
- Must not overwrite the frozen source assets or promote this affine diagnostic as a general final method without further validation.


## 2026-08-23 CST — retained 01184 partial-coverage TTO improvement (not final)

- User approval: “确实好多了，如果能更贴合partial就更好了”. Retain this as the strongest current 01184 registration derivative; do not overwrite it.
- Input partial/camera: `data/01184.ply`; `workspace/redwood_onestage_rawdepth_512_stage2_20260714/01184/camera.pth`.
- Frozen complete prior/model: `gpt_version/_gpt_multiview_hunyuan2mv_redwood_20260823/01184_wheelaxis_v2/01184/01184_hunyuan2mv_canonical.ply`; `01184_hunyuan2mv_native_mesh.glb`.
- Starting fixed-pose registration: `gpt_version/_gpt_multiview_hunyuan2mv_redwood_20260823/_wheelaxis_v2_gpu_2d3d_locked_scale6/01184`.
- Retained output: `gpt_version/_gpt_multiview_hunyuan2mv_redwood_20260823/_wheelaxis_v2_partial_coverage_tto/01184/01184_visible_sim3_tto_{registered_100k.ply,partial_gray_pixal_red.ply,projection.png,info.json}`.
- Method: 4 outer rounds × 120 GPU Adam steps, fixed rotation, bounded aligned-frame scale/translation. The data objective is one-sided `partial -> generated` coverage in 2D projection, depth, and 3D, with a tail-35% residual emphasis. It never penalizes or deletes unobserved generated points. Final scale xyz `[0.6314821, 0.6397480, 0.6526875]`.
- Prompt/negative prompt: not applicable; no generative call. Generation metadata for the frozen source asset: `UNKNOWN` in this entry; see source metadata. GT was not used for inference or selection.
- Further work authorized by the feedback: test a shared, visibility-local smooth residual deformation that improves partial coverage while preserving the full prior; write separate derivative outputs only.


## 2026-08-23 CST — accepted rollback and freeze on Pixal registration v15

- User decision: “把代码，方案都回退 `_pixal_guarded_unified_registration_v15_20260822` 这个方案吧，这个方案目前为止还是效果最好的”. The active method is therefore restored to the accepted v15 registration baseline; no later experiment is a default candidate.
- Samples: `01184`, `05117`, `05452`, `06127`, `06145`, `06188`, `06830`, `07136`, `07306`, and `09639`.
- Frozen inputs: `data/<sample>.ply`, `workspace/redwood_onestage_rawdepth_512_stage2_20260714/<sample>/camera.pth`, `gpt_version/<sample>/gpt_image.png`, `gpt_version/<sample>/pixal3d.glb`, and `gpt_version/<sample>/pixal3d_sampled_100k.ply`.
- Model/checkpoints and generation parameters are exactly those recorded in the 2026-08-22 accepted Pixal3D entry: TencentARC/Pixal3D, Pixal3D-weights, DINOv3, MoGe-2, RMBG-2.0, seed 42, 1024 cascade, and the recorded shared 12-step sampler settings. No model or GLB was regenerated during rollback.
- Prompt/negative prompt: registration does not call an image generator, so not applicable. The frozen GPT image prompts remain next to the accepted per-sample images; the serving seed/backend details remain `UNKNOWN` as previously recorded.
- Registration implementation: `scripts/select_pixal_guarded_unified_registration_v15.py`; exact design: `docs/fast_unified_registration_v15.md`; canonical method: `docs/core_registration_pipeline.md`.
- Accepted output root: `gpt_version/_pixal_guarded_unified_registration_v15_20260822`. All ten registered 100k PLYs, registered GLBs, transform NPYs, partial-gray/Pixal-red overlays, projections, and info JSONs were re-verified non-empty after rollback.
- Shared route: v8 GenPC/PCA fallback versus v12 GPU SO(3)+visible Sim(3) TTT, selected by the shared observable-confidence gate and full-resolution render-score do-no-harm guard. `06145` and `06830` use the GPU route; the other eight use fallback. No GT, category rule, sample-specific threshold, deformation, fusion, or point deletion is used.
- Cleanup authorized by the user: post-v15 Hunyuan3D-MV, dual-depth/multiview, v18+ deformation/fusion, and Omni experimental output roots were permanently removed after v15 verification, releasing approximately 8.8GB. Historical records remain in this file, but their referenced post-v15 output paths may no longer exist.
- Do not overwrite the accepted v15 assets. Any future method must branch into a new derivative root and exceed v15 on full-ten visual review and post-freeze metrics before promotion.

## 2026-08-23 CST — accepted bidirectional partial-to-Pixal cycle route as new research mainline

Status and approval boundary:
- User feedback: “我感觉这个方案效果更好，作为主线来改进吧，目标是超过genpc”. The bidirectional saved-camera 2D+3D route is now the active research direction to improve; v15 remains an immutable baseline and fallback asset.
- This accepts the route and retains the full-ten forced-candidate audit for comparison. It does not claim that the current metrics already beat GenPC, and it does not authorize sample-specific tuning.
- Samples: `01184`, `05117`, `05452`, `06127`, `06145`, `06188`, `06830`, `07136`, `07306`, `09639`.

Exact inputs and retained outputs:
- Raw partials: `data/<sample>.ply`.
- Saved cameras: `workspace/redwood_onestage_rawdepth_512_stage2_20260714/<sample>/camera.pth`.
- Frozen complete inputs: `gpt_version/_pixal_guarded_unified_registration_v15_20260822/<sample>/<sample>_unified_registration_v14_registered_100k.ply` and matching `registered_mesh.glb`.
- Retained full-ten audit root: `gpt_version/_pixal_bidirectional_cycle_registration_forced_audit_20260823`.
- Per-sample outputs: `<sample>/<sample>_bidirectional_cycle_{registered_100k.ply,partial_gray_pixal_red.ply,projection.png,registered_mesh.glb,npy,info.json}`.
- Post-freeze metric root: `gpt_version/_pixal_bidirectional_cycle_registration_forced_audit_20260823/postfreeze_cd_emd_20260823`.
- Implementation: `src/bidirectional_cycle_registration.py`, `scripts/run_pixal_bidirectional_cycle_registration.py`, and `scripts/evaluate_bidirectional_cycle_redwood.py`.

Method and exact shared parameters:
- Fit a robust proper Sim(3) from partial points to the saved-camera-visible Pixal surface; apply only its strict analytic inverse to move the complete Pixal body toward the partial; use a separately fitted reverse map as a cycle-consistency witness.
- Pixel schedule `[8.0, 5.0, 3.0]`; final pixel radius `5.0`; maximum rotation per step `3.0` degrees; per-step isotropic scale bounds `[0.96, 1.04]`; maximum translation `0.03` partial-bbox diagonal; minimum pairs `96`; maximum independent cycle ratio `0.03`; final improvement ratio `0.995`; camera padding `0.15`.
- Audit option `force_candidate_output=true` exported candidates even when the normal guard rejected them. Future mainline experiments must write new derivative roots and retain whether each candidate passed the guard.
- No anisotropic scale, reflection, non-rigid deformation, point deletion, category rule, sample ID rule, GT geometry, CD, or EMD is used during registration or routing.

Models, prompts, and generation parameters:
- Registration makes no generative call; prompt and negative prompt are not applicable.
- It reuses the accepted frozen GPT ImageGen/Pixal3D assets. Exact semantic prompts, `UNKNOWN` GPT serving checkpoint/seed/backend, TencentARC/Pixal3D model paths, Pixal seed `42`, 1024 cascade, and shared 12-step sampler parameters are recorded verbatim in the accepted `2026-08-22 01:07 CST` entry above.
- No RMBG, MoGe, GPT image, Pixal GLB, or Pixal PLY is regenerated in this route.

Postprocessing, metrics, and current limitation:
- Proper Sim(3) is applied to all original 100,000 Pixal points and the complete mesh; no fusion, resampling, local deformation, or partial replacement is performed.
- Frozen post-hoc evaluation uses 16,384-point FPS, seed `6145`, and identical saved FPS indices for the audit candidate and v15.
- Forced-audit mean CD-L1/EMD x1e2: `2.0647596/3.0785815`; v15 under the same indices: `2.1095980/3.0960672`; GenPC paper target: `1.74/2.88`.
- The current route is better than v15 on mean CD and EMD but does not yet beat GenPC. `06830`, `09639`, and `05117` expose the largest general failure modes and may be used for diagnosis only, not special-case parameters.

What must not change without asking:
- Do not overwrite the retained full-ten audit root, post-freeze metric files, frozen v15 outputs, GPT semantic images, or Pixal3D complete assets.
- Keep one shared zero-shot parameterization. Freeze each new full-ten prediction set before reading GT metrics, and use GT only for post-freeze reporting.

## 2026-08-23 CST — retained metric-passing bidirectional-consensus surface posterior (pending visual approval)

- Status: this full-ten candidate is retained because its strict post-freeze mean CD-L1/EMD x1e2 `1.6911582/2.8026253` exceeds the GenPC paper mean `1.74/2.88`. It is not yet the accepted visual baseline; do not overwrite it before user review.
- Samples and observations: the shared Redwood ten IDs; `data/<sample>.ply`; saved cameras under `workspace/redwood_onestage_rawdepth_512_stage2_20260714/<sample>/camera.pth`.
- Frozen complete source: the accepted v15 registered Pixal 100k PLY and GLB for each sample, with all semantic prompts, Pixal model/checkpoint paths, seed 42, 1024 cascade, and 12-step generation settings recorded in the accepted 2026-08-22 Pixal entries. No image or 3D prior was regenerated.
- Registration output root: `gpt_version/_pixal_bidirectional_consensus_forced_audit_20260823`; implementation `src/bidirectional_consensus_registration.py` and `scripts/run_pixal_bidirectional_cycle_registration.py --step-mode consensus --force-candidate-output`.
- Registration hypotheses: independently constructed forward, reverse, balanced, and mutual visible correspondence sets; pixel schedule `[8,5,3]`; final radius 5; per-step rotation cap 3 degrees; isotropic scale `[0.96,1.04]`; translation cap 0.03 partial diagonal; minimum 96 pairs; cycle cap 0.03. One shared parameter set, no anisotropic scale/reflection/sample/category rule/GT metric.
- Posterior output root: `gpt_version/_pixal_bidirectional_consensus_surface_projection_20260823`; implementation `src/observation_conditioned_surface_projection.py` and `scripts/run_observation_conditioned_surface_projection.py`.
- Posterior candidates: identity, smooth absorption settings `(influence,max displacement)=(0.04,0.025),(0.06,0.04)`, and observed surface-mass budgets `[0.04,0.08,0.12]`; shared prior-mass penalty `0.008`; seed `6145`. All ten selected mass budget 0.12 via GT-free visible 2D+3D routing.
- Postprocessing: partial targets are FPS-uniformized; exactly 12,000 nearest observed-surface prior points are exchanged for 12,000 partial points. Output remains exactly 100,000 points; 88,000 original complete-prior points remain; no hidden/far complete region is selected for removal by the local-distance rule. Prompt/negative prompt, CFG, scheduler, and generation seed are not applicable to registration/postprocessing.
- Strict metric root: `gpt_version/_pixal_bidirectional_consensus_surface_projection_20260823/postfreeze_cd_emd_strict_20260823`; 16,384 points, metric seed 6145, separate prediction FPS per geometry, equivalent GT FPS by shared seed. Metrics were unavailable to inference and were computed only after prediction freeze.
- Ablation means: v15 `2.1095899/3.0972997`; true bidirectional consensus `2.0584516/3.0488108`; consensus plus surface posterior `1.6911582/2.8026253`.
- Exact semantic prompts and negative prompts: no new prompt; registration and posterior reuse frozen assets. Refer to the 2026-08-22 01:07 CST entry for verbatim semantic prompts. GPT serving checkpoint/seed/backend remain `UNKNOWN`, so the original semantic generation is not exactly reproducible from local weights.
- Preservation boundary: do not overwrite either new output root, its info JSONs/projections/PLYs, or strict metric indices/results. Keep v15 intact. Promotion requires user visual review of all ten overlays.

## 2026-08-23 CST — Qwen-geometry / GPT-clarity / Pixal3D bidirectional mainline

- User decision: use the Qwen semantic result as the geometry/pose anchor, use GPT ImageGen only to make the object clearer and more complete, then run Pixal3D, bidirectional saved-camera 2D+3D registration, and observation-conditioned surface projection. This supersedes direct depth-to-GPT regeneration as the active mainline.
- Formal full-ten root: `workspace/redwood_qwen_gpt_pixal_bidirectional_mainline_20260823` for samples `01184`, `05117`, `05452`, `06127`, `06145`, `06188`, `06830`, `07136`, `07306`, and `09639`.
- Per-sample source images: `workspace/redwood_onestage_rawdepth_512_stage2_20260714/<sample>/depth.png` and `img.png`; copied into the formal root as `depth.png` and `qwen_img.png`. The required semantic source is exactly `img.png`, not `qwen_edit_stage1.png`.
- GPT edit contract: `qwen_img.png` is the sole authority for silhouette, camera pose, image-space position, projected size, part layout, local articulation, and visible occlusion. The edit may improve clarity, realistic material, boundary cleanliness, and completion of already implied surfaces, but must not rotate, rescale, recenter, add/remove/rearrange parts, or alter wheel/leg/armrest orientation. Negative prompt: no viewpoint change, no camera change, no crop change, no scale change, no silhouette change, no geometry redesign, no added parts, no removed parts, no mirrored object, no text, no clutter.
- Exact per-sample prompts are stored beside each finished image as `<sample>/prompt.txt`. At the time of this entry, `09639` is complete; the other nine are in progress. Any unavailable prior serving details are recorded as `UNKNOWN`, not guessed.
- GPT model/backend: OpenAI ImageGen built-in service; exact serving checkpoint, scheduler, seed, CFG, and negative-prompt implementation are `UNKNOWN`. Output is normalized for Pixal3D without changing aspect ratio or object geometry.
- Pixal3D model: TencentARC/Pixal3D with the locally installed Pixal3D weights, DINOv3, MoGe-2, and RMBG-2.0 dependencies; seed `42`; 1024 cascade; shared existing sampler settings from `scripts/run_pixal3d_gpt_batch.py`. Outputs are `pixal3d.glb`, `pixal3d_sampled_100k.ply`, `pixal3d_input.png`, and `pixal3d_metadata.json`.
- Registration/posterior: initialize from the frozen v15 transform only, then independently estimate partial-to-visible-prior and visible-prior-to-partial proper Sim(3) correspondences using saved-camera 2D+3D evidence. Apply the strict inverse to the complete prior; then use the shared 12% observation-conditioned surface-mass projection while retaining 88% of the generated prior. No GT, sample ID rule, category-specific threshold, reflection, or hidden-surface deletion is allowed.
- Verified pilot `09639`: strict CD-L1/EMD x1e2 `1.1279924/2.1037089`, versus the previous surface mainline `2.1158189/3.4741573`, v15 `2.6992/3.7465`, and GenPC sample `1.43/2.29`. A frozen nine-old-plus-new-09639 hybrid reached mean `1.5923756/2.6648329`; this hybrid is diagnostic only, not the final full-ten route.
- Preservation boundary: do not overwrite the source Qwen `img.png`, frozen v15 outputs, or completed formal-root artifacts without asking. All ten final images, prompts, GLBs, sampled PLYs, registration overlays, fused/posterior PLYs, and metric reports must remain under the formal workspace root.
- Completion update: all ten image/Pixal/registration/posterior assets now exist under the formal root. Registration root: `_bidirectional_consensus`; final posterior root: `_surface_projection`; strict metric root: `postfreeze_cd_emd_strict`.
- Frozen strict CD-L1/EMD x1e2: `01184 1.1962/1.9613`, `05117 1.5034/2.4375`, `05452 0.8620/1.2920`, `06127 2.2477/3.9358`, `06145 1.6244/1.8489`, `06188 1.2082/2.1064`, `06830 1.9829/4.0109`, `07136 1.9819/3.0755`, `07306 2.6158/3.1871`, `09639 1.1280/2.1013`; mean `1.6350423/2.5956847`. Frozen v15 under the same protocol is `2.1095899/3.0974603`; GenPC paper mean is `1.74/2.88`.
- This full-ten route exceeds GenPC mean by approximately 6.0% CD and 9.9% EMD. Metrics were computed only after the prediction root was frozen and were unavailable to inference/routing. It is the metric-leading mainline, pending full-ten visual review.

## 2026-08-23 CST — retained 07136 previous-prior preference and scale-conflict guard derivative

- User feedback: “07136感觉没有上一个版本配得好”. Preserve the previous-prior 07136 result and the new comparison; do not overwrite either while the guard is reviewed.
- Current candidate input/output: `workspace/redwood_qwen_gpt_pixal_bidirectional_mainline_20260823/_surface_projection/07136/07136_bidirectional_cycle_registered_100k.ply`; current Qwen/GPT/Pixal assets are under the same formal root.
- Preferred previous-prior candidate: `gpt_version/_pixal_bidirectional_consensus_surface_projection_20260823/07136/07136_bidirectional_cycle_registered_100k.ply`, sourced from frozen `gpt_version/07136/{gpt_image.png,pixal3d.glb,pixal3d_sampled_100k.ply}` and the previous bidirectional consensus registration.
- Diagnosis: current registered PCA major-axis standard deviation `0.16350` versus previous `0.17879`; current cumulative scale `0.963246` versus previous `1.034705`; GT-free final visible objective `0.103402` versus previous `0.075756`. The new image/Pixal prior is shorter/thicker and exposes shape-scale ambiguity.
- Shared derivative implementation: `src/cross_prior_scale_guard.py`, `scripts/select_cross_prior_scale_consistency_guard.py`; output root `workspace/redwood_qwen_gpt_pixal_bidirectional_mainline_20260823/_cross_prior_scale_guard`. The guard requires opposite scale directions, at least 6% disagreement, current rejection/fallback acceptance, and at least 20% visible-objective improvement. No sample/category rule or GT is used. It selects fallback only for 07136 on the full ten.
- Frozen post-hoc metrics: 07136 improves from `1.9819/3.0755` to `1.6225/2.5098`; full-ten mean improves from `1.6350/2.5957` to `1.5991/2.5410`. Metric root: `workspace/redwood_qwen_gpt_pixal_bidirectional_mainline_20260823/postfreeze_cross_prior_scale_guard_cd_emd`. Metrics were unavailable to the guard.
- No new image was generated. Prompt/negative prompt, serving model, seed, CFG, scheduler, and image postprocessing are therefore not applicable; source-generation details remain those recorded for the two frozen prior branches.

## 2026-08-23 CST — support-aware voxel surface-measure derivative (pending visual approval)

- Motivation from user feedback: fused visualization exposes sparse partial regions beside dense Pixal3D regions. This derivative changes sampling measure only; it does not change generation, registration, pose, scale, or surface geometry.
- Frozen input: `workspace/redwood_qwen_gpt_pixal_bidirectional_mainline_20260823/_cross_prior_scale_guard/<sample>/<sample>_bidirectional_cycle_registered_100k.ply` for the shared Redwood ten.
- Implementation: `src/voxel_surface_measure_resampling.py`; runner `scripts/run_voxel_surface_measure_resampling.py`; tests `tests/test_voxel_surface_measure_resampling.py`.
- Shared parameters: target 32,768 points; robust observation outlier k=8; MAD scale 4.0; prior-support threshold 0.02 object diagonal; 18 voxel-size binary-search steps. One original representative per voxel, with observation priority inside a shared voxel. No sample/category rule or GT metric.
- Exact output root: `workspace/redwood_qwen_gpt_pixal_bidirectional_mainline_20260823/_cross_prior_voxel_uniform_32k`. Mean output count 32,771 (range 32,725–32,809); mean local kNN CV `0.4982 -> 0.2923`; mean 45.2 unsupported observation outliers removed (range 0–111).
- Completeness contract: every output point is an original fused point; no interpolation, smoothing displacement, point creation, anisotropic scaling, deformation, or hidden-body truncation. Voxel downsampling retains surface coverage while reducing redundant prior density.
- Predictions were frozen before strict evaluation. Metric root: `workspace/redwood_qwen_gpt_pixal_bidirectional_mainline_20260823/postfreeze_cross_prior_voxel_uniform_32k_cd_emd`; mean CD-L1/EMD x1e2 `1.5944254/2.5295681`, versus cross-prior input `1.5991020/2.5409679`, canonical full-new-prior `1.6350423/2.5956847`, and GenPC `1.74/2.88`.
- No image generation occurs. Prompt, negative prompt, ImageGen serving checkpoint/seed/CFG/scheduler, and image postprocessing are not applicable. Preserve this derivative and its metric indices until visual review; do not overwrite the cross-prior input.

## 2026-08-23 CST — accepted best-effect Qwen-GPT-Pixal bidirectional voxel mainline

- User approval: “ok，把现在这套流程作为最好效果的流程记录下来，保存成分支提交一下”. This promotes the entire route and its 32k voxel-uniform outputs to the accepted best-effect pipeline. Branch: `codex/qwen-gpt-pixal-voxel-mainline`. Do not change or overwrite the accepted assets, prompts, priors, transforms, routing audit, voxel outputs, or metric indices without asking.
- Samples: `01184`, `05117`, `05452`, `06127`, `06145`, `06188`, `06830`, `07136`, `07306`, `09639`.
- Formal root: `workspace/redwood_qwen_gpt_pixal_bidirectional_mainline_20260823`. Per-sample inputs are `<sample>/depth.png`, `<sample>/qwen_img.png`, and `<sample>/gpt_image.png`; partials are `data/<sample>.ply`; cameras are `workspace/redwood_onestage_rawdepth_512_stage2_20260714/<sample>/camera.pth`.
- Exact GPT prompts for `01184`, `05117`, and `05452`: `UNKNOWN`; generation occurred before exact prompt persistence. Their recorded contracts respectively preserve the wheelie bin and parallel wheels, the red chair and four-leg layout, and the thin curved brown chair without a pointed back. These results are not exactly reproducible from the serving model.
- Exact GPT prompt for `06127`: “Edit the supplied Qwen semantic image into a clear, complete, realistic studio image of the same potted plant. The input is the sole geometric authority. Preserve exactly the camera viewpoint, object yaw, image-space center, projected height and width, terracotta pot silhouette and opening, stem locations, and the number, direction, overlap, curvature, and relative size of every major visible leaf. Only improve sharpness, clean boundaries, coherent realistic material, and complete surfaces already implied by the input. Plain white background. Do not rotate, rescale, recenter, mirror, redesign, add or remove leaves, rearrange foliage, alter the pot, or add text/clutter.”
- Exact GPT prompt for `06145`: “Edit the supplied Qwen semantic image into a clear, complete, realistic studio image of the same pedestal table. The input is the sole geometric authority. Preserve exactly the camera viewpoint, object yaw, image-space center, projected size, tabletop long/short axis orientation and perspective, tabletop thickness, single central pedestal position, and round base dimensions. Only improve sharpness, clean boundaries, coherent realistic material, and complete surfaces already implied by the input. Plain white background. Do not rotate, rescale, recenter, mirror, swap tabletop length and width, change the pedestal or base proportions, add parts, remove parts, or add text/clutter.”
- Exact GPT prompt for `06188`: “Edit the supplied Qwen semantic image into a clear, complete, realistic studio image of the same red scooter. The input is the sole geometric authority. Preserve exactly the camera viewpoint, object yaw, image-space center, projected size, body silhouette, seat, handlebars, mirrors, both wheels, front fork, and especially the existing front-wheel and steering-head tilt and foreshortening. Only improve sharpness, clean boundaries, coherent realistic material, and complete surfaces already implied by the input. Plain white background. Do not rotate, rescale, recenter, mirror, straighten or reverse the steering angle, move either wheel, redesign the scooter, add/remove parts, add a rider, or add text/clutter.”
- Exact GPT prompt for `06830`: “Edit the supplied Qwen semantic image into a clear, complete, realistic studio image of the same blue child's tricycle with a tall rear push handle. The input is the sole geometric authority. Preserve exactly the camera viewpoint, object yaw, image-space center, projected size, frame silhouette, tall push-handle height and curvature, seat, three-wheel layout, front fork, pedals, and especially the existing steering/front-wheel direction and foreshortening. Only improve sharpness, clean boundaries, coherent realistic material, and complete surfaces already implied by the input. Plain white background. Do not rotate, rescale, recenter, mirror, straighten or reverse the front wheel, shorten the push handle, move wheels, redesign, add/remove parts, or add text/clutter.”
- Exact GPT prompt for `07136`: “Edit the supplied Qwen semantic image into a clear, complete, realistic studio image of the same black leather sofa. The input is the sole geometric authority. Preserve exactly the camera viewpoint, oblique yaw and elevation, image-space center, projected length/height/depth, full outer silhouette, backrest length and tilt, seat depth, both armrests, base, and all major visible proportions. Only improve sharpness, clean boundaries, coherent realistic leather material, and complete surfaces already implied by the input. Plain white background. Do not rotate, rescale, recenter, mirror, shorten or widen the sofa, change back/seat/arm dimensions, add cushions or legs, remove parts, or add text/clutter.” The accepted cross-prior guard selects the previous frozen prior for final 07136 geometry because observable shape-scale conflict rejects this new prior.
- Exact GPT prompt for `07306`: “Edit the supplied Qwen semantic image into a clear, complete, realistic studio image of the same red cylindrical trash can. The input is the sole geometric authority. Preserve exactly the camera viewpoint, object yaw, image-space center, projected height and width, cylindrical body silhouette, top rim/opening geometry, bottom profile, and all visible proportions. Only improve sharpness, clean boundaries, coherent realistic material, and complete surfaces already implied by the input. Plain white background. Do not rotate, rescale, recenter, mirror, turn it into a wheeled bin, add a lid/handle/wheels, change rim or body dimensions, add/remove parts, or add text/clutter.”
- Exact GPT prompt for `09639`: “Use case: precise-object-edit. Asset type: zero-shot 3D reconstruction conditioning image. Primary request: Enhance the input Qwen-generated office-chair image into a sharp, photorealistic and structurally complete product image. Authoritative invariants: the input image is the sole authority for camera azimuth, elevation, roll, perspective, chair yaw, center position, projected size, normalized bounding box, canvas occupancy, silhouette, backrest recline, left/right armrest positions, seat outline, central cylinder, five-star base directions, leg lengths, and caster locations. Preserve these approximately pixel-aligned. Do not rotate, straighten, symmetrize, recenter, zoom, enlarge, or shrink the chair. Subject/detail: preserve the same padded ergonomic swivel office chair design. Clarify the upholstery into coherent dark charcoal fabric or leather padding; repair blurry and melted edges; make the arm supports, seat/back junction, gas cylinder, five separate base legs, and casters mechanically clean and complete. Complete only genuinely ambiguous or missing small regions while following the existing silhouette. Scene/backdrop: preserve the clean pure white background. Constraints: change image quality and local structural clarity only; keep the same pose, dimensions, footprint, and part layout; full chair remains inside frame. Avoid: canonical front view, mesh-back redesign, changed chair proportions, extra or missing legs, changed wheel positions, merged feet, room, floor, cast shadow, text, logo, watermark, people, or other objects.”
- GPT ImageGen backend: OpenAI built-in ImageGen edit mode. Separate negative prompt is not exposed; prohibitions are embedded verbatim above. Serving checkpoint/version, generation seed, steps, CFG/guidance, and scheduler are `UNKNOWN`/not exposed. Images are 1254×1254. Qwen `img.png` is the sole geometry, pose, silhouette, and projected-size anchor.
- Pixal3D models: `/opt/data/private/cr/lab/GenPC/models/Pixal3D-weights`; DINOv3 `/opt/data/private/cr/lab/GenPC/models/dinov3-vitl16-pretrain-lvd1689m`; MoGe-2 `/opt/data/private/cr/lab/GenPC/models/moge-2-vitl/model.pt`; RMBG `/opt/data/private/cr/lab/GenPC/models/RMBG-2.0`; xformers attention. Seed 42, 1024 cascade, 100k sampled points, decimation target 300k, texture 2048. Sparse/shape/texture each use 12 steps; sparse guidance `7.5/0.7/rescale_t 5`, shape `7.5/0.5/rescale_t 3`, texture `1.0/0.0/rescale_t 3`.
- Postprocessing: frozen v15 coarse initialization; independent partial-to-visible-prior and visible-prior-to-partial proper-Sim(3) consensus; strict inverse; observation-conditioned identity/smooth/4/8/12% surface-mass routing; cross-prior scale-consistency guard; support-aware 32k voxel representatives with k=8, MAD scale 4, support threshold 0.02 diagonal, observation priority, no geometry creation or deformation.
- Accepted final output root: `workspace/redwood_qwen_gpt_pixal_bidirectional_mainline_20260823/_cross_prior_voxel_uniform_32k`; accepted metric root: `workspace/redwood_qwen_gpt_pixal_bidirectional_mainline_20260823/postfreeze_cross_prior_voxel_uniform_32k_cd_emd`.
- Accepted strict mean CD-L1/EMD x1e2: `1.5944254/2.5295681`; GenPC reference `1.74/2.88`; v15 under the same protocol `2.1095899/3.0984912`. Mean point count 32,771 and mean local-density CV `0.4982 -> 0.2923`. GT metrics were unavailable to generation, registration, routing, and resampling.

## 2026-09-04 CST — accepted Pixal-native MoGe first-stage registration

- User approval: “可以可以 效果非常不错”. Preserve this as the approved
  Pixal--MoGe *first-stage* registration baseline; it is not yet a final
  partial completion or a benchmark result.
- Sample/output: Redwood `07136` sofa. Approved artifacts are
  `workspace/agent_genpc_plus_scratch_20260903/_pixal_analytic_moge_rayscale_rgb_07136/{pixal_native_moge_registered_100k.ply,pixal_native_moge_gray_pixal_red.ply,pixal_native_moge_projection.png,pixal_native_moge_info.json}`.
- Inputs: Pixal surface prior
  `workspace/agent_genpc_plus_scratch_20260903/07136/pixal3d_sampled_100k.ply`;
  Pixal prepared image
  `workspace/agent_genpc_plus_scratch_20260903/07136/pixal3d_input.png`;
  Pixal camera metadata
  `workspace/agent_genpc_plus_scratch_20260903/07136/pixal3d_metadata.json`.
- Models: local MoGe-2
  `/opt/data/private/cr/lab/GenPC/models/moge-2-vitl/model.pt`; local RMBG-2.0
  `/opt/data/private/cr/lab/GenPC/models/RMBG-2.0`; Pixal prior is the existing
  TencentARC/Pixal3D asset with recorded generation seed `42` in its metadata.
  No new image or 3D model is generated by this stage.
- Prompt / negative prompt: not applicable (cached `pixal3d_input.png` is the
  complete condition; no text-conditioned generation occurs). Image serving
  model, scheduler, CFG, and seed are therefore not applicable to this stage.
- Frozen registration contract: reuse Pixal's saved MoGe-conditioned FOV and
  distance (`camera_angle_x=0.6138169188`, `distance=1.5776747465`); compose
  the documented `o_voxel.to_glb` and Pixal export axes to initialize
  `(X,Y,Z)=(-x,-y,d+z)` in MoGe/OpenCV coordinates; use only bounded visible
  silhouette, log-depth, boundary and auxiliary RGB residual updates. A
  same-view support translation is only `(-1,-1)` pixels. The accepted
  depth-gauge update is the robust camera-ray scale
  `p_cam <- 1.09020599 p_cam`, inferred from same-pixel z-buffer ratios.
- Result evidence: native-view silhouette IoU remains `0.75584`, coverage
  `0.75625`, leakage `0.00072`; log-depth residual improves `0.09911 ->
  0.04328`. MoGe foreground PLY retains semantic RGB. Future reruns must use
  `--cached-moge` plus the paired info JSON when this exact input is unchanged;
  do not re-infer MoGe or overwrite the accepted root without user direction.

## 2026-09-04 CST — accepted 07136 hard-partial pixel-indexed Sim(3) registration baseline

- User approval: “好非常非常多了 稍稍再配更准就行”. Preserve the displayed
  relaxed two-camera result as the current accepted registration baseline for
  the Redwood `07136` leather-sofa diagnostic. It is registration only, not
  a fusion, local-edit, or final benchmark acceptance.
- Exact retained outputs:
  `workspace/pixal_moge_joint_pixel_sim3_relaxed_20260904/07136/{two_camera_joint_registered_100k.ply,two_camera_joint_partial_gray_pixal_red.ply,two_camera_joint_saved_view_projection.png,two_camera_joint_pixal_to_partial.npy,two_camera_joint_visible_pixel_residual.npy,two_camera_joint_info.json}`.
  The displayed overlay is `two_camera_joint_partial_gray_pixal_red.ply` with
  red Pixal points and grey hard-partial points. Do not overwrite this root.
- Inputs: partial `data/07136.ply`; Camera-1 saved camera
  `workspace/agent_genpc_plus_scratch_20260903/07136/camera.pth`; point UV
  `workspace/agent_genpc_plus_scratch_20260903/07136/point_uv.npy`; semantic
  image `workspace/agent_genpc_plus_scratch_20260903/07136/img.png`; fixed
  Pixal sample `workspace/agent_genpc_plus_scratch_20260903/07136/pixal3d_sampled_100k.ply`;
  Pixal input `workspace/agent_genpc_plus_scratch_20260903/07136/pixal3d_input.png`;
  native Pixal MoGe and contract
  `workspace/pixal_native_moge_rayscale_redwood_20260904/07136/{pixal_native_moge_points.ply,pixal_native_moge_info.json}`;
  Camera-1/Camera-2 bridge and pixel pairs
  `workspace/pixal_moge_two_camera_bridge_20260904/07136/{two_camera_pixal_moge_native_moge_to_partial.npy,two_camera_pixal_moge_partial_to_native_moge_matches.npy}`.
- Models: existing TencentARC/Pixal3D weights
  `/opt/data/private/cr/lab/GenPC/models/Pixal3D-weights`, DINOv3
  `/opt/data/private/cr/lab/GenPC/models/dinov3-vitl16-pretrain-lvd1689m`,
  MoGe-2 `/opt/data/private/cr/lab/GenPC/models/moge-2-vitl/model.pt`, and
  RMBG-2.0 `/opt/data/private/cr/lab/GenPC/models/RMBG-2.0`. No model was
  re-run for this registration. The inherited Pixal generation used seed 42,
  1024 resolution/cascade, 12 sparse/shape/texture steps, and 100,000 surface
  samples; detailed sampler values are in the retained `pixal3d_metadata.json`.
- Exact inherited Qwen prompt: `生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的leather sofa，纯白背景`.
  Qwen negative prompt: ` `; true CFG 4.0; 40 inference steps. Exact inherited
  GPT prompt: “Use case: product-mockup\nAsset type: zero-shot 3-D completion semantic input\nInput image: Image 1 is the edit target and geometric reference.\nPrimary request: Refine this dark leather sofa into a clean, complete semantic product image.\nScene/backdrop: pure white studio background.\nConstraints: preserve exactly the same camera viewpoint, image-space position, projected scale, full sofa length, width, seat depth, backrest height, left armrest contour, right-side seat edge, and the three visible cushion divisions from Image 1. Do not shorten, elongate, rotate, mirror, or reshape the sofa. Improve only sharpness, leather coherence, and clearly implied details. Do not recenter, crop, resize, add/remove cushions or parts, or alter object geometry. No text, watermark, or extra objects.” GPT serving checkpoint, seed, scheduler, CFG, and backend implementation are `UNKNOWN`.
- Accepted registration: two calibrated camera edges are first refined by the
  bounded joint proper Sim(3); then the hard Camera-1 z-buffers form up to
  10,000 mutual Pixal/partial pixel-indexed visible 3-D pairs (2-pixel
  radius), whose fixed-seed 64-trial robust Sim(3) fit supplies fractional
  residual candidates `[.125,.25,.50,.75,1.0]`. The selected full residual
  remains proper/isotropic and bounded to 4 degrees, scale `[.96,1.04]`, and
  0.04 partial-bbox-diagonal translation. Camera-2 Pixal--MoGe evidence is
  recorded but `pixel_native_gate=false`: it is not allowed to veto a better
  hard-partial registration. No GT, CD, EMD, category, or sample-specific
  routing is used.
- Evidence: Camera-1 visible objective `0.1164802 -> 0.1026699`; IoU
  `.727801 -> .747686`; coverage `.965975 -> .992867`; visible geometric
  objective `.0181247 -> .0124421`. All 100,000 Pixal points are retained;
  no point deletion, resampling, fusion, non-rigid deformation, or local edit
  is applied. Future work may only make a small additional registration
  refinement from a copy of this accepted root, with one shared parameter set
  and explicit comparison against this baseline.

## 2026-09-04 CST — accepted 07136 wide-tilt global registration candidate

- User approval: “近乎完美了 可以再大一点点就行”. Preserve the wide-tilt
  registration result as the current visual baseline for the Redwood `07136`
  sofa; further experiments must start from a copy and remain global proper
  Sim(3) refinements only unless the user authorizes another method.
- Exact outputs: `workspace/pixal_moge_joint_pixel_sim3_wide_tilt_20260904/07136/{camera1_amplified_registered_100k.ply,camera1_amplified_partial_gray_pixal_red.ply,camera1_amplified_saved_view_projection.png,camera1_amplified_residual.npy,camera1_amplified_info.json}`.
  The visual overlay uses red Pixal and grey hard partial. Do not overwrite
  this directory.
- Inputs: partial `data/07136.ply`; source registered prior
  `workspace/pixal_moge_joint_pixel_sim3_amplified_20260904/07136/camera1_amplified_registered_100k.ply`; saved Camera-1
  `workspace/redwood_onestage_rawdepth_512_stage2_20260714/07136/camera.pth`; semantic image
  `workspace/redwood_onestage_rawdepth_512_stage2_20260714/07136/img.png`.
  The complete prior ultimately derives from the fixed Pixal source
  `workspace/agent_genpc_plus_scratch_20260903/07136/pixal3d_sampled_100k.ply`;
  that optional scratch source was subsequently cleaned, but the exact
  predecessor PLY is retained in the preceding accepted entry.
- Models / generation: no model inference or image generation in this stage.
  The inherited complete prior uses TencentARC/Pixal3D weights
  `/opt/data/private/cr/lab/GenPC/models/Pixal3D-weights`, DINOv3
  `/opt/data/private/cr/lab/GenPC/models/dinov3-vitl16-pretrain-lvd1689m`,
  MoGe-2 `/opt/data/private/cr/lab/GenPC/models/moge-2-vitl/model.pt`, and
  RMBG-2.0 `/opt/data/private/cr/lab/GenPC/models/RMBG-2.0`; Pixal seed 42,
  1024 resolution, 12 sparse/shape/texture steps, and 100k surface samples.
  The inherited Qwen prompt is exactly `生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的leather sofa，纯白背景`; negative prompt is ` `, true CFG is 4.0, and steps are 40. The inherited GPT prompt is exactly: “Use case: product-mockup\nAsset type: zero-shot 3-D completion semantic input\nInput image: Image 1 is the edit target and geometric reference.\nPrimary request: Refine this dark leather sofa into a clean, complete semantic product image.\nScene/backdrop: pure white studio background.\nConstraints: preserve exactly the same camera viewpoint, image-space position, projected scale, full sofa length, width, seat depth, backrest height, left armrest contour, right-side seat edge, and the three visible cushion divisions from Image 1. Do not shorten, elongate, rotate, mirror, or reshape the sofa. Improve only sharpness, leather coherence, and clearly implied details. Do not recenter, crop, resize, add/remove cushions or parts, or alter object geometry. No text, watermark, or extra objects.” GPT serving checkpoint, seed, scheduler, CFG, and backend implementation remain `UNKNOWN`.
- Frozen registration parameters: `scripts/run_camera1_amplified_sim3_refine.py --wide-tilt-search`; fixed 32k Camera-1 subsearch; proper isotropic Sim(3) levels `(.010,1.00deg,.010)`, `(.004,.35deg,.004)`, `(.001,.10deg,.001)`; full-point strict acceptance only. It selects a -1.0 degree x rotation and cumulative 0.4996% uniform shrink. No GT/CD/EMD, fusion, point deletion, resampling, non-rigid deformation, or local editing is used. All 100k Pixal points remain.
- Evidence: Camera-1 objective `.0983708 -> .0945947`; IoU `.759739 -> .771579`; leakage `.235935 -> .223755`; coverage `.992603 -> .992269`. This visual acceptance is not a final benchmark/metric claim.

## 2026-09-04 CST — accepted 07136 final hard-partial global Sim(3) registration

- User approval: “完美”. This supersedes the prior wide-tilt candidate as the
  accepted visual registration baseline for the Redwood `07136` sofa. Preserve
  it exactly; all nine-sample replication runs must use the same shared
  procedure without per-sample parameters.
- Exact outputs: `workspace/pixal_moge_joint_pixel_sim3_final_tilt_20260904/07136/{camera1_amplified_registered_100k.ply,camera1_amplified_partial_gray_pixal_red.ply,camera1_amplified_saved_view_projection.png,camera1_amplified_residual.npy,camera1_amplified_info.json}`.
  The displayed comparison is red Pixal versus grey hard partial. Do not
  overwrite this output root.
- Inputs: partial `data/07136.ply`; prior
  `workspace/pixal_moge_joint_pixel_sim3_wide_tilt_20260904/07136/camera1_amplified_registered_100k.ply`; saved camera
  `workspace/redwood_onestage_rawdepth_512_stage2_20260714/07136/camera.pth`;
  semantic `workspace/redwood_onestage_rawdepth_512_stage2_20260714/07136/img.png`.
  The base complete Pixal PLY, two-camera MoGe bridge, and prior registration
  inputs are exactly those enumerated in the preceding accepted 07136 entries.
- Models / generation: this stage calls no model. Inherited assets use
  TencentARC/Pixal3D `/opt/data/private/cr/lab/GenPC/models/Pixal3D-weights`,
  DINOv3 `/opt/data/private/cr/lab/GenPC/models/dinov3-vitl16-pretrain-lvd1689m`,
  MoGe-2 `/opt/data/private/cr/lab/GenPC/models/moge-2-vitl/model.pt`, and
  RMBG-2.0 `/opt/data/private/cr/lab/GenPC/models/RMBG-2.0`; Pixal seed 42,
  1024 resolution, 12 sparse/shape/texture steps, and 100k surface sampling.
  The inherited Qwen prompt is exactly `生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的leather sofa，纯白背景`; negative prompt ` `; true CFG 4.0; 40 steps. The inherited GPT prompt is exactly: “Use case: product-mockup\nAsset type: zero-shot 3-D completion semantic input\nInput image: Image 1 is the edit target and geometric reference.\nPrimary request: Refine this dark leather sofa into a clean, complete semantic product image.\nScene/backdrop: pure white studio background.\nConstraints: preserve exactly the same camera viewpoint, image-space position, projected scale, full sofa length, width, seat depth, backrest height, left armrest contour, right-side seat edge, and the three visible cushion divisions from Image 1. Do not shorten, elongate, rotate, mirror, or reshape the sofa. Improve only sharpness, leather coherence, and clearly implied details. Do not recenter, crop, resize, add/remove cushions or parts, or alter object geometry. No text, watermark, or extra objects.” GPT serving checkpoint, seed, scheduler, CFG, and backend implementation remain `UNKNOWN`.
- Frozen registration continuation: `scripts/run_camera1_amplified_sim3_refine.py --wide-tilt-search --max-tilt-degrees 0.5`, 32k deterministic Camera-1 subsearch, proper isotropic Sim(3) levels `(.010,.5deg,.010)`, `(.004,.175deg,.004)`, `(.001,.05deg,.001)`, and strict full-point acceptance. It selects 1.099% global shrink and a small x translation, with no additional rotation. No GT/CD/EMD, fusion, point deletion, resampling, non-rigid deformation, or local editing is used. All 100k Pixal points remain.
- Evidence: Camera-1 objective `.0945947 -> .0902447`; IoU `.771579 -> .784393`; leakage `.223755 -> .209274`; coverage `.992269 -> .989891`. This is an approved visual registration result, not a final benchmark/metric claim.

## 2026-09-04 CST — accepted fixed-route nine-sample Pixal registration batch

- User approval: “现在配准效果我很满意了”. Preserve the fixed no-fallback
  registration batch as the accepted shared Redwood replication route. It is
  registration-only: do not overwrite it with fusion, point deletion,
  resampling, non-rigid deformation, or sample-specific adjustments.
- Sample ids and exact final outputs:
  `workspace/pixal_moge_fixed_route_full9_20260904/{01184,05117,05452,06127,06145,06188,06830,07306,09639}/final/{camera1_amplified_registered_100k.ply,camera1_amplified_partial_gray_pixal_red.ply,camera1_amplified_saved_view_projection.png,camera1_amplified_residual.npy,camera1_amplified_info.json}`.
  All nine final PLYs were checked to contain exactly 100,000 Pixal points.
  The accepted `07136` companion remains
  `workspace/pixal_moge_joint_pixel_sim3_final_tilt_20260904/07136/camera1_amplified_registered_100k.ply`
  and was deliberately not overwritten.
- Inputs: for every id, hard partial `data/<id>.ply`; saved Camera-1,
  `point_uv.npy`, semantic `img.png`, and object mask in
  `workspace/redwood_onestage_rawdepth_512_stage2_20260714/<id>/`; retained
  Pixal 100k prior, `pixal3d_input.png`, and `pixal3d_metadata.json` in
  `workspace/redwood_qwen_gpt_pixal_bidirectional_mainline_20260823/<id>/`.
- Models: registration freshly ran MoGe-2
  `/opt/data/private/cr/lab/GenPC/models/moge-2-vitl/model.pt` and RMBG-2.0
  `/opt/data/private/cr/lab/GenPC/models/RMBG-2.0` on the retained Pixal input;
  it did not regenerate Pixal GLBs, semantic images, or GPT images. Inherited
  Pixal source uses TencentARC/Pixal3D with 100,000 surface samples. The
  per-sample upstream Qwen/GPT prompts, seeds, schedulers, CFG values, and
  backend versions were not collected in this batch record: `UNKNOWN`.
  Therefore regeneration of the *upstream images/GLBs* is not exactly
  reproducible from this entry; the recorded registration from retained assets
  is reproducible.
- Frozen method / parameters: analytic Pixal--MoGe initialization; Camera-1 to
  Camera-2 foreground bridge; coupled two-edge proper Sim(3); Camera-1 mutual
  visible pixel-indexed 3-D proper Sim(3) (maximum 10,000 pairs, fixed seed
  6145, 64 robust trials, fractions `[.125,.25,.50,.75,1.0]`, 4-degree,
  `[.96,1.04]` scale, and `0.04` partial-diagonal bounds); then Camera-1
  global schedules `(.006,.30deg,.006),(.002,.10deg,.002),(.0005,.025deg,.0005)`,
  `(.010,1.00deg,.010),(.004,.35deg,.004),(.001,.10deg,.001)`, and
  `(.010,.50deg,.010),(.004,.175deg,.004),(.001,.05deg,.001)`. Every stage is
  applied in sequence; the candidate lattice includes identity, but there is
  no external no-harm/proposal gate or fallback. Native, bridge, and Camera-1
  scores are diagnostic only. No GT, CD, EMD, category, or sample-id routing
  is used.
- Verification: targeted registration tests pass `13/13`; `py_compile` and
  `git diff --check` pass. Final Camera-1 objective after the last continuation
  is respectively `0.078404, 0.029222, 0.045875, 0.077912, 0.047739,
  0.094383, 0.115850, 0.051641, 0.074690` in the sample-id order above.

## 2026-09-04 CST — user-approved 01184 multi-view Gaussian edit baseline

- User approval: “01184 效果确实不错”. Preserve this as the approved visual
  baseline for stronger-edit ablations; do not overwrite its output directory.
  The user requested a somewhat stronger shared edit, not a sample-specific
  redesign.
- Exact outputs: `workspace/single_view_boundary_gaussian_redwood10_20260904/gaussian/01184/{edit/partial_anchored_gaussian_edit_editable_prior_100k.ply,edit/partial_anchored_gaussian_edit_partial_gray_prior_red.ply,edit/partial_anchored_gaussian_edit_saved_view_projection.png,edit/partial_anchored_gaussian_edit_virtual_view_board.png,decoded/partial_anchored_gaussian_decoded_100k.ply,decoded/partial_anchored_gaussian_partial_gray_decoded_red.ply,decoded/partial_anchored_gaussian_saved_view_projection.png,decoded/partial_anchored_gaussian_virtual_view_board.png}`. The decoded cloud has exactly 100,000 Pixal slots; red is edited/decoded Pixal and gray is the fixed partial.
- Inputs: registered complete prior `workspace/best_register_redwood10_20260904/01184/camera1_amplified_registered_100k.ply`; partial `workspace/single_view_boundary_gaussian_redwood10_20260904/inputs/partial/01184.ply`; saved camera `workspace/single_view_boundary_gaussian_redwood10_20260904/inputs/camera/01184/camera.pth`; depth `workspace/single_view_boundary_gaussian_redwood10_20260904/inputs/camera/01184/depth.png`; Qwen semantic `workspace/single_view_boundary_gaussian_redwood10_20260904/inputs/camera/01184/img.png`; GPT clarity image `workspace/single_view_boundary_gaussian_redwood10_20260904/inputs/pixal/01184/gpt_image.png`; Pixal input/GLB `workspace/single_view_boundary_gaussian_redwood10_20260904/inputs/pixal/01184/{pixal3d_input.png,pixal3d.glb}`.
- Models / generation: inherited Qwen-Image-Edit-2511 stage: exact prompt `生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的rubbish bin，纯白背景`; negative prompt ` `; true CFG `4.0`; 40 steps; raw-depth input; seed/scheduler/backend `UNKNOWN`. Inherited GPT clarity-only prompt is `UNKNOWN`; the retained contract is “clarity-only edit of qwen_img.png; preserve the wheelie-bin camera pose, image-space position, projected size, body silhouette, lid, handle and two parallel wheels; do not rotate, rescale, recenter, mirror, add/remove parts, or alter wheel placement.” GPT checkpoint, seed, scheduler, CFG, and backend are `UNKNOWN`. Inherited Pixal3D is TencentARC/Pixal3D with weights `/opt/data/private/cr/lab/GenPC/models/Pixal3D-weights`, DINOv3 `/opt/data/private/cr/lab/GenPC/models/dinov3-vitl16-pretrain-lvd1689m`, MoGe-2 `/opt/data/private/cr/lab/GenPC/models/moge-2-vitl/model.pt`, RMBG-2.0 `/opt/data/private/cr/lab/GenPC/models/RMBG-2.0`, seed `42`, 1024 cascade, 12 sparse/shape/texture steps, and 100k surface samples.
- Frozen edit setup: six signed-PCA positive-overlap virtual views at 384 px plus saved Camera-1 mutual pairs; missing partial pixels are unconstrained. Boundary-conditioned kNN Gaussian graph: 8 neighbours, edge ratio `1.8`, saved/virtual pixel radius `2`, anchor-residual and maximum displacement `0.06` partial-bbox diagonal, screening `.003`, six Pixal self-protection views, protection weight `.02`, protection exclusion `.08` diagonal, CG tolerance `1e-5`, max 240 iterations. Decoder uses one-to-one partial replacement and the same six virtual views; it selected 30,012 anchors. No GT/CD/EMD, category, or sample-specific routing entered edit/decoding.
- What is approved: the visible quality and full-body preservation of this exact baseline. Stronger candidates must keep its input prior, one shared zero-shot parameterization, all 100k complete-prior slots, and positive-overlap-only policy; only edit-strength parameters may change.

## 2026-09-04 CST — user-preferred 06830 relaxed-anchor Gaussian edit

- User approval: after viewing the five frozen variants and their offline
  metrics, the user stated that “`宽松锚点 0.075` 会更好些”.  Treat this as the
  preferred shared 3DGS-edit parameterization candidate, subject to a
  cross-sample check.  It is **not** a per-sample metric route: the same
  residual/displacement bounds must be applied to every sample in a batch.
- Exact outputs: `workspace/single_view_boundary_gaussian_redwood10_20260904/gaussian/06830/strength_075/{edit/partial_anchored_gaussian_edit_editable_prior_100k.ply,edit/partial_anchored_gaussian_edit_partial_gray_prior_red.ply,edit/partial_anchored_gaussian_edit_saved_view_projection.png,edit/partial_anchored_gaussian_edit_virtual_view_board.png,decoded/partial_anchored_gaussian_decoded_100k.ply,decoded/partial_anchored_gaussian_partial_gray_decoded_red.ply,decoded/partial_anchored_gaussian_saved_view_projection.png,decoded/partial_anchored_gaussian_virtual_view_board.png}`.
  The decoded output retains exactly 100,000 Pixal slots and uses 19,576
  collision-free one-to-one partial anchors; it never concatenates or deletes
  the unobserved prior.  The visual comparison boards are
  `workspace/single_view_boundary_gaussian_redwood10_20260904/gaussian/06830/{strength_comparison_board.png,strength_virtual_comparison_board.png}`.
- Inputs: registered Pixal prior
  `workspace/best_register_redwood10_20260904/06830/camera1_amplified_registered_100k.ply`; partial
  `workspace/single_view_boundary_gaussian_redwood10_20260904/inputs/partial/06830.ply`; Camera-1
  `workspace/single_view_boundary_gaussian_redwood10_20260904/inputs/camera/06830/camera.pth`; depth
  `workspace/single_view_boundary_gaussian_redwood10_20260904/inputs/camera/06830/depth.png`; Qwen semantic
  `workspace/single_view_boundary_gaussian_redwood10_20260904/inputs/camera/06830/img.png`; GPT clarity image
  `workspace/single_view_boundary_gaussian_redwood10_20260904/inputs/pixal/06830/gpt_image.png`; and Pixal
  input/mesh `workspace/single_view_boundary_gaussian_redwood10_20260904/inputs/pixal/06830/{pixal3d_input.png,pixal3d.glb}`.
- Models / upstream generation: Qwen-Image-Edit-2511 uses exact prompt
  `生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的tricycle，纯白背景`; negative prompt ` `;
  true CFG `4.0`; 40 steps; raw-depth input; seed/scheduler/backend `UNKNOWN`.
  The retained GPT clarity prompt is exactly: `Edit the supplied Qwen semantic image into a clear, complete, realistic studio image of the same blue child's tricycle with a tall rear push handle. The input is the sole geometric authority. Preserve exactly the camera viewpoint, object yaw, image-space center, projected size, frame silhouette, tall push-handle height and curvature, seat, three-wheel layout, front fork, pedals, and especially the existing steering/front-wheel direction and foreshortening. Only improve sharpness, clean boundaries, coherent realistic material, and complete surfaces already implied by the input. Plain white background. Do not rotate, rescale, recenter, mirror, straighten or reverse the front wheel, shorten the push handle, move wheels, redesign, add/remove parts, or add text/clutter.` GPT negative prompt, checkpoint, seed, scheduler, CFG, and backend are `UNKNOWN`.  Pixal3D uses `/opt/data/private/cr/lab/GenPC/models/Pixal3D-weights`, DINOv3 `/opt/data/private/cr/lab/GenPC/models/dinov3-vitl16-pretrain-lvd1689m`, MoGe-2 `/opt/data/private/cr/lab/GenPC/models/moge-2-vitl/model.pt`, RMBG-2.0 `/opt/data/private/cr/lab/GenPC/models/RMBG-2.0`, xFormers, seed `42`, 1024 px, 12 sparse/shape/texture steps, a 300k decimation target, 2048 texture, and 100k sampled surface points.  Sparse and shape guidance are `7.5`; texture guidance is `1.0`.
- Frozen edit / decode: Camera-1 mutual pixel-indexed positives plus six signed-PCA 384-px virtual-view mutual positives; missing partial pixels remain unconstrained.  The boundary-conditioned kNN graph has 8 neighbours, edge ratio `1.8`, pixel radius `2`, screening `.0015`, six self-protection views, protection weight `.01`, and 240 CG iterations at `1e-5` tolerance.  Shared relaxed bounds are `max_anchor_residual_ratio=.075` and `max_displacement_ratio=.075`; `remote_gain=1.0`, support/remote radius `.08` partial-bbox diagonal, protection exclusion `.10` diagonal.  No GT, CD, EMD, category, or sample-specific routing enters the edit or decode.
- Offline-only evidence: frozen-output audit with seed `6145` and 16,384-point FPS reports CD-L1/EMD `1.54112596 / 2.81621478` (`×10²`) in `workspace/single_view_boundary_gaussian_redwood10_20260904/gaussian/06830/postfreeze_strength_ablation/{metrics.csv,protocol.json}`.  This is lower than baseline `1.69312470 / 3.16989869`, remote-gain-2 `1.65262204 / 3.05689238`, and MV512+remote-gain-2 `1.66569538 / 3.08937691`; it is descriptive evidence only and must not be used as an inference-time gate.

## 2026-09-04 CST — user-approved Redwood-10 relaxed-anchor Gaussian batch

- User approval: “效果非常好了 可以保存代码 提交到github上”.  Preserve this
  ten-sample post-registration result as the accepted shared `.075` Gaussian
  edit/decode batch.  Do not overwrite it, switch parameters per sample, or
  re-run upstream Qwen/GPT/Pixal generation without explicit user direction.
- Sample ids and exact outputs: `01184, 05117, 05452, 06127, 06145, 06188,
  06830, 07136, 07306, 09639`.  The complete artifact root is
  `workspace/relaxed_anchor_gaussian_redwood10_20260904`; exact final
  predictions are
  `gaussian/<sample>/decoded/partial_anchored_gaussian_decoded_100k.ply`, with
  editable prior, graph field, positive pairs, overlays, anchor assignments,
  and six-view board alongside each result.  The machine-readable batch record
  is `gaussian/batch_manifest.json`; the post-freeze metric record is
  `postfreeze_cd_emd/{metrics_samples.csv,metrics_summary.json,protocol.json}`.
  Every final PLY was verified to retain exactly 100,000 Pixal slots.
- Inputs: per sample, copied depth/camera/Qwen semantic assets are at
  `inputs/camera/<sample>/`, copied GPT/Pixal image, GLB, 100k surface prior
  and metadata are at `inputs/pixal/<sample>/`, and the real scan is
  `inputs/partial/<sample>.ply`.  Accepted registered priors are at
  `registration/<sample>/final/camera1_amplified_registered_100k.ply`.  No
  upstream image, GLB, MoGe, or Sim(3) inference was re-run for this
  post-registration batch.
- Models / generation: inherited Qwen-Image-Edit-2511 assets use
  `/opt/data/private/cr/lab/GenPC/models/Qwen-Image-Edit-2511` and the
  Nunchaku transformer
  `/opt/data/private/cr/lab/GenPC/models/nunchaku-qwen-image-edit/nunchaku_qwen_image_2511_balance_int4.safetensors`;
  Qwen negative prompt is exactly ` `, true CFG `4.0`, 40 steps, raw-depth
  input, and Qwen seed/scheduler/backend `UNKNOWN`.  Inherited Pixal3D uses
  `/opt/data/private/cr/lab/GenPC/models/Pixal3D-weights`, DINOv3
  `/opt/data/private/cr/lab/GenPC/models/dinov3-vitl16-pretrain-lvd1689m`,
  MoGe-2 `/opt/data/private/cr/lab/GenPC/models/moge-2-vitl/model.pt`, and
  RMBG-2.0 `/opt/data/private/cr/lab/GenPC/models/RMBG-2.0`; seed `42`, 1024
  px, xFormers, 12 sparse/shape/texture steps, 300k decimation target, 2048
  texture, and 100k surface sampling.  GPT checkpoint/version, seed, steps,
  CFG, scheduler, backend and negative prompt are `UNKNOWN` unless noted
  below.
- Exact inherited Qwen prompts (all use `input_image: raw_depth.png`, negative
  prompt ` `, true CFG `4.0`, 40 steps): `01184`: `生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的rubbish bin，纯白背景`; `05117`: `生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的red chair，纯白背景`; `05452`: `生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的armchair，纯白背景`; `06127`: `生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的terracotta flower pot with leafy plant，纯白背景`; `06145`: `生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的table，纯白背景`; `06188`: `生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的red motorcyle，纯白背景`; `06830`: `生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的tricycle，纯白背景`; `07136`: `生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的leather sofa，纯白背景`; `07306`: `生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的red office trash can，纯白背景`; `09639`: `生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的Ergonomic Chair，纯白背景`.
- Exact inherited GPT prompts: `01184`: `UNKNOWN` (contract: clarity-only edit
  of qwen_img.png; preserve the wheelie-bin camera pose, image-space position,
  projected size, body silhouette, lid, handle and two parallel wheels; do not
  rotate, rescale, recenter, mirror, add/remove parts, or alter wheel
  placement). `05117`: `UNKNOWN` (contract: clarity-only edit of qwen_img.png;
  preserve the red chair camera pose, image-space position, projected size,
  back, seat and four-leg layout; do not rotate, rescale, recenter, mirror,
  add/remove parts, or alter leg placement). `05452`: `UNKNOWN` (contract:
  clarity-only edit of qwen_img.png; preserve the brown curved chair camera
  pose, image-space position, projected size, thin curved back/seat profile,
  arms and legs; do not rotate, rescale, recenter, mirror, add a pointed back,
  add/remove parts, or alter curvature). `06127`: `Edit the supplied Qwen semantic image into a clear, complete, realistic studio image of the same potted plant. The input is the sole geometric authority. Preserve exactly the camera viewpoint, object yaw, image-space center, projected height and width, terracotta pot silhouette and opening, stem locations, and the number, direction, overlap, curvature, and relative size of every major visible leaf. Only improve sharpness, clean boundaries, coherent realistic material, and complete surfaces already implied by the input. Plain white background. Do not rotate, rescale, recenter, mirror, redesign, add or remove leaves, rearrange foliage, alter the pot, or add text/clutter.` `06145`: `Edit the supplied Qwen semantic image into a clear, complete, realistic studio image of the same pedestal table. The input is the sole geometric authority. Preserve exactly the camera viewpoint, object yaw, image-space center, projected size, tabletop long/short axis orientation and perspective, tabletop thickness, single central pedestal position, and round base dimensions. Only improve sharpness, clean boundaries, coherent realistic material, and complete surfaces already implied by the input. Plain white background. Do not rotate, rescale, recenter, mirror, swap tabletop length and width, change the pedestal or base proportions, add parts, remove parts, or add text/clutter.` `06188`: `Edit the supplied Qwen semantic image into a clear, complete, realistic studio image of the same red scooter. The input is the sole geometric authority. Preserve exactly the camera viewpoint, object yaw, image-space center, projected size, body silhouette, seat, handlebars, mirrors, both wheels, front fork, and especially the existing front-wheel and steering-head tilt and foreshortening. Only improve sharpness, clean boundaries, coherent realistic material, and complete surfaces already implied by the input. Plain white background. Do not rotate, rescale, recenter, mirror, straighten or reverse the steering angle, move either wheel, redesign the scooter, add/remove parts, add a rider, or add text/clutter.` `06830`: `Edit the supplied Qwen semantic image into a clear, complete, realistic studio image of the same blue child's tricycle with a tall rear push handle. The input is the sole geometric authority. Preserve exactly the camera viewpoint, object yaw, image-space center, projected size, frame silhouette, tall push-handle height and curvature, seat, three-wheel layout, front fork, pedals, and especially the existing steering/front-wheel direction and foreshortening. Only improve sharpness, clean boundaries, coherent realistic material, and complete surfaces already implied by the input. Plain white background. Do not rotate, rescale, recenter, mirror, straighten or reverse the front wheel, shorten the push handle, move wheels, redesign, add/remove parts, or add text/clutter.` `07136`: `Edit the supplied Qwen semantic image into a clear, complete, realistic studio image of the same black leather sofa. The input is the sole geometric authority. Preserve exactly the camera viewpoint, oblique yaw and elevation, image-space center, projected length/height/depth, full outer silhouette, backrest length and tilt, seat depth, both armrests, base, and all major visible proportions. Only improve sharpness, clean boundaries, coherent realistic leather material, and complete surfaces already implied by the input. Plain white background. Do not rotate, rescale, recenter, mirror, shorten or widen the sofa, change back/seat/arm dimensions, add cushions or legs, remove parts, or add text/clutter.` `07306`: `Edit the supplied Qwen semantic image into a clear, complete, realistic studio image of the same red cylindrical trash can. The input is the sole geometric authority. Preserve exactly the camera viewpoint, object yaw, image-space center, projected height and width, cylindrical body silhouette, top rim/opening geometry, bottom profile, and all visible proportions. Only improve sharpness, clean boundaries, coherent realistic material, and complete surfaces already implied by the input. Plain white background. Do not rotate, rescale, recenter, mirror, turn it into a wheeled bin, add a lid/handle/wheels, change rim or body dimensions, add/remove parts, or add text/clutter.` `09639`: `Use case: precise-object-edit. Asset type: zero-shot 3D reconstruction conditioning image. Primary request: Enhance the input Qwen-generated office-chair image into a sharp, photorealistic and structurally complete product image. Authoritative invariants: the input image is the sole authority for camera azimuth, elevation, roll, perspective, chair yaw, center position, projected size, normalized bounding box, canvas occupancy, silhouette, backrest recline, left/right armrest positions, seat outline, central cylinder, five-star base directions, leg lengths, and caster locations. Preserve these approximately pixel-aligned. Do not rotate, straighten, symmetrize, recenter, zoom, enlarge, or shrink the chair. Subject/detail: preserve the same padded ergonomic swivel office chair design. Clarify the upholstery into coherent dark charcoal fabric or leather padding; repair blurry and melted edges; make the arm supports, seat/back junction, gas cylinder, five separate base legs, and casters mechanically clean and complete. Complete only genuinely ambiguous or missing small regions while following the existing silhouette. Scene/backdrop: preserve the clean pure white background. Constraints: change image quality and local structural clarity only; keep the same pose, dimensions, footprint, and part layout; full chair remains inside frame. Avoid: canonical front view, mesh-back redesign, changed chair proportions, extra or missing legs, changed wheel positions, merged feet, room, floor, cast shadow, text, logo, watermark, people, or other objects.`
- Frozen post-registration method: use the accepted final rigid Pixal prior,
  Camera-1 mutual positives and six signed-PCA 384-px positive-only virtual
  overlaps; missing partial pixels remain unconstrained.  The common graph
  settings are 8 neighbours, edge ratio `1.8`, screening `.0015`, six
  self-protection views, protection weight `.01`, protection exclusion `.10`
  partial diagonal, maximum anchor residual `.075`, maximum displacement
  `.075`, remote gain `1.0`, and collision-free one-to-one partial-anchor
  replacement.  This is zero-shot: no GT, CD, EMD, category, or sample-id
  routing enters the edit or decoder.
- Offline-only evidence: after all ten predictions froze, fixed-seed 6145
  16,384-point FPS evaluation reports mean CD-L1/EMD `1.58197163 / 2.51144224`
  (`×10²`).  Per-sample values live in the recorded CSV.  These metrics are
  post-freeze reporting only and must not become a test-time gate.
