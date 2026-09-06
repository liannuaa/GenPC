# Accepted project state

## 2026-09-04 CST — Redwood-10 fixed registration + relaxed-anchor Gaussian edit

**User approval.** “效果非常好了 可以保存代码 提交到github上.” This is the
accepted result to preserve. Do not change the shared registration or Gaussian
parameters, regenerate upstream assets, or replace the 100k-prior carrier
without a new explicit experiment.

**Outputs.** The complete result root is
`workspace/relaxed_anchor_gaussian_redwood10_20260904`. For every sample
`01184, 05117, 05452, 06127, 06145, 06188, 06830, 07136, 07306, 09639`, the
registered prior is
`registration/<sample>/final/camera1_amplified_registered_100k.ply` and the
accepted prediction is
`gaussian/<sample>/decoded/partial_anchored_gaussian_decoded_100k.ply`.
Every prediction was checked to retain exactly 100,000 Pixal slots. The batch
manifest is `gaussian/batch_manifest.json`.

**Inputs.** The same root contains `inputs/partial/<sample>.ply`, saved-view
camera/depth/Qwen assets in `inputs/camera/<sample>/`, and GPT image,
Pixal input, GLB, sampled PLY and metadata in `inputs/pixal/<sample>/`.

**Upstream models and prompts.**

- Qwen model: `models/Qwen-Image-Edit-2511`; transformer:
  `models/nunchaku-qwen-image-edit/nunchaku_qwen_image_2511_balance_int4.safetensors`.
  Each full Qwen prompt follows exactly
  `生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的<label>，纯白背景`
  with labels `{rubbish bin, red chair, armchair, terracotta flower pot with
  leafy plant, table, red motorcyle, tricycle, leather sofa, red office trash
  can, Ergonomic Chair}` in the sample order above. Negative prompt is exactly
  ` `; true CFG is `4.0`; steps are `40`; input is `raw_depth.png`; Qwen seed,
  scheduler and backend are `UNKNOWN`.
- GPT clarity image prompt, model/version, seed, scheduler, CFG and negative
  prompt are `UNKNOWN` unless saved beside a retained input as `prompt.txt`.
  The fixed contract is clarity-only: preserve the Qwen camera, pose, scale,
  silhouette and observed part layout; do not rotate, rescale, recenter,
  mirror, add/remove parts, or alter articulated local geometry.
- Pixal3D: `models/Pixal3D-weights`; DINOv3:
  `models/dinov3-vitl16-pretrain-lvd1689m`; MoGe-2:
  `models/moge-2-vitl/model.pt`; RMBG-2.0: `models/RMBG-2.0`. Frozen Pixal
  generation uses seed `42`, 1024 cascade, 12 sparse/shape/texture steps,
  sparse/shape guidance `7.5`, texture guidance `1.0`, 300k decimation target,
  2048 texture and 100k surface samples.

**Frozen inference.** Registration is the fixed native Pixal--MoGe analytic
initialization, two-camera bridge, coupled residual and Camera-1 amplified /
wide-tilt / final-tilt continuation. Gaussian editing uses Camera-1 plus six
positive-overlap signed-PCA virtual views; 8-neighbor graph, edge ratio `1.8`,
screening `.0015`, anchor/displacement caps `.075`, six self-protection views
with weight `.01`, protection exclusion `.10`, remote gain `1.0`, CG tolerance
`1e-5`, and 240 iterations. The decoder performs collision-free one-to-one
partial-anchor replacement. No GT/CD/EMD/category/sample-specific routing is
used by inference.

**Offline-only evidence.** After the batch was frozen, the 16,384-point FPS
audit with seed `6145` reported mean CD-L1/EMD `1.58197163 / 2.51144224`
(×10²). Per-sample results and protocol are under
`postfreeze_cd_emd/{metrics_samples.csv,metrics_summary.json,protocol.json}`.
These values are reporting-only and must not become a test-time selection gate.

## 2026-09-04 CST — `01184` scratch Qwen → GPT → Pixal end-to-end validation

**User approval.** “现在流程已经没问题了.” This confirms the scratch
upstream path is valid. Preserve its saved-view geometry and the fixed
registration/Gaussian parameters; do not replace them with a replay asset or a
metric-selected variant.

**Inputs and outputs.** Partial and GT are respectively
`data/redwood/partial/01184.ply` and `data/redwood/gt/01184.ply`. The complete
run is `workspace/mainline_upstream_smoke_20260904`: Qwen depth/semantic and
Camera-1 assets are in `semantic/01184`; GPT input and prompt are in
`pixal_input/01184`; Pixal GLB/100k prior/metadata are in `pixal/01184`;
registration ends at
`registration/01184/final/camera1_amplified_registered_100k.ply`; and the
approved complete prediction is
`gaussian/01184/decoded/partial_anchored_gaussian_decoded_100k.ply`.

**Models and exact generation record.** Qwen pipeline is
`models/Qwen-Image-Edit-2511`, transformer is
`models/nunchaku-qwen-image-edit/nunchaku_qwen_image_2511_balance_int4.safetensors`;
input is `raw_depth.png`, output is 512 px, true CFG `4.0`, steps `40`, and
negative prompt is exactly ` `. The exact Qwen prompt is
`生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的rubbish bin，纯白背景`.
Qwen seed, scheduler and backend are `UNKNOWN`. The GPT asset is
`pixal_input/01184/gpt_image.png`; its full prompt is recorded verbatim in
`gpt_refinement_prompt.txt`; model/version, seed, scheduler, CFG and negative
prompt are `UNKNOWN`. Pixal uses `models/Pixal3D-weights`, DINOv3
`models/dinov3-vitl16-pretrain-lvd1689m`, MoGe-2
`models/moge-2-vitl/model.pt`, and RMBG-2.0 `models/RMBG-2.0`, with seed `42`,
1024 cascade, 12 sparse/shape/texture steps, guidance `7.5/7.5/1.0`, 300k
decimation, 2048 texture and 100k surface points.

**Post-processing and reporting.** The run uses deterministic 256-view
saved-camera selection, grayscale depth + OpenCV fill, Qwen completion, GPT
clarity edit, Pixal preprocessing, native Pixal--MoGe alignment, two-camera
bridge, Camera-1 continuation, and the fixed partial-anchored Gaussian edit
and decode. No GT is read before the final prediction. The offline 16,384-point
audit (seed `6145`) reported CD-L1×100 `1.16162859` and EMD×100 `1.80500858`;
these are reporting-only.

## 2026-09-05 CST — globally calibrated fixed registration/edit defaults

**Status.** User-directed calibration of the already fixed mainline. The
Qwen → GPT → Pixal assets, registration stage order, Gaussian edit/decode
algorithm, and 100k prior carrier are unchanged. Only two values are updated
globally for every sample: the final Camera-1 Sim(3) trust region is `1.0°`
and the saved/virtual positive-correspondence radii are `1.0 px`. There is no
sample/category branch, learned update, or inference-time GT/CD/EMD access.

**Outputs.** The complete candidate is
`workspace/fixed_mainline_calibration_20260905/combo_final100_pixel100`.
Its registration inputs are the ten final PLYs under
`workspace/fixed_mainline_calibration_20260905/registration_final100_replay/<id>/final/`;
the final predictions are
`combo_final100_pixel100/gaussian/<id>/decoded/partial_anchored_gaussian_decoded_100k.ply`.
All ten (`01184, 05117, 05452, 06127, 06145, 06188, 06830, 07136, 07306,
09639`) were verified to contain exactly 100,000 points. The no-override smoke
run is `workspace/fixed_mainline_default_smoke_20260905`; for `01184` both its
final registration PLY and decoded PLY are byte-identical to the selected
candidate.

**Inputs/models/prompts.** This calibration did not regenerate an image or a
3-D prior. It reuses the exact partial, Camera-1, Qwen/GPT/Pixal inputs from
`workspace/relaxed_anchor_gaussian_redwood10_20260904/inputs`; the complete
Qwen prompts, GPT prompt records, model paths, checkpoints, seeds, and upstream
generation parameters are therefore exactly those recorded in the preceding
2026-09-04 accepted Redwood-10 entry. New Qwen/GPT/Pixal generation prompt:
`NOT RUN`. New image/3-D generation seed, scheduler, CFG and negative prompt:
`NOT APPLICABLE`. Native registration uses the existing Pixal 100k prior plus
MoGe-2 `models/moge-2-vitl/model.pt` and RMBG-2.0 `models/RMBG-2.0` under the
same fixed two-camera route.

**Frozen numerical settings.** Standard/wide Camera-1 continuation remains
unchanged (32,000 points; wide trust region `1.0°`). The final continuation
uses levels `(0.010, 1.0°, 0.010)`, `(0.004, 0.35°, 0.004)`, and
`(0.001, 0.10°, 0.001)`. Gaussian edit/decode uses `1.0 px` for both saved and
virtual positive matches; six virtual views at 384 px; anchor/displacement
caps `.075`; 8-neighbour graph, edge ratio `1.8`, screening `.0015`; six
protection views, weight `.01`, exclusion `.10`; remote gain `1.0`; and CG
`1e-5/240`. Do not alter these values without a new complete global audit.

**Offline-only evidence.** The fixed 16,384-point audit with seed `6145`
reported CD-L1×100 / EMD×100:

| id | CD | EMD |
| --- | ---: | ---: |
| 01184 | 1.0610 | 1.7740 |
| 05117 | 1.6685 | 2.9317 |
| 05452 | 0.9441 | 1.5376 |
| 06127 | 2.3836 | 3.7654 |
| 06145 | 0.6561 | 1.0179 |
| 06188 | 1.2063 | 1.9679 |
| 06830 | 1.5183 | 2.7327 |
| 07136 | 1.2703 | 2.3600 |
| 07306 | 3.1936 | 3.9313 |
| 09639 | 1.7803 | 2.7671 |
| mean | **1.5682** | **2.4786** |

This improves the prior fixed baseline `1.5820 / 2.5114` on both metrics. The
metric file is `combo_final100_pixel100/metrics.{csv,json}` and is reporting
only; it was never read by registration or Gaussian editing.

## 2026-09-05 CST — accepted default broad Camera-1 basin capture

**User approval.** “可以 直接替换冻结主线成为新方案.” The fixed mainline must
therefore run the broad Camera-1 pixel-Sim(3) basin-capture stage by default.
It may be disabled only with `--no-coarse-basin-recovery` for an ablation; do
not restore the former disabled default without a new explicit instruction.

**Outputs.** The accepted Redwood-10 registration audit is
`workspace/mainline_results_20260905/redwood/registration/<id>/final/camera1_amplified_registered_100k.ply`, and the corresponding accepted 100k
predictions are
`workspace/mainline_results_20260905/redwood/gaussian/<id>/decoded/partial_anchored_gaussian_decoded_100k.ply`, for IDs `01184, 05117, 05452,
06127, 06145, 06188, 06830, 07136, 07306, 09639`. Offline-only metrics are
`workspace/mainline_results_20260905/redwood/metrics/coarse_basin.{json,csv}`.

**Inputs/models/prompts.** Registration used `data/redwood/partial/<id>.ply`,
Camera-1 assets under
`/opt/data/private/cr/lab/GenPC/workspace/redwood_onestage_rawdepth_512_stage2_20260714/<id>/`, and GPT/Pixal assets under
`/opt/data/private/cr/lab/GenPC/workspace/redwood_qwen_gpt_pixal_bidirectional_mainline_20260823/<id>/`.
No upstream image or 3-D generation was rerun. The Qwen model is
`models/Qwen-Image-Edit-2511`; transformer is
`models/nunchaku-qwen-image-edit/nunchaku_qwen_image_2511_balance_int4.safetensors`.
The full Qwen prompt is exactly `生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的<label>，纯白背景`, with labels `{rubbish bin, red chair, armchair,
terracotta flower pot with leafy plant, table, red motorcyle, tricycle,
leather sofa, red office trash can, Ergonomic Chair}` in the ID order above.
The Qwen negative prompt is exactly ` `; resolution is 512 px, true CFG is
`4.0`, steps are `40`, input is `raw_depth.png`, and Qwen seed/scheduler/backend
are `UNKNOWN`. The full GPT clarity prompt is stored beside each retained input
as `prompt.txt`; GPT model/version, seed, scheduler, CFG and negative prompt
are `UNKNOWN`. The GPT contract is preserve Qwen camera, pose, scale,
silhouette and observed part layout; do not rotate, rescale, recenter, mirror,
add/remove parts, or alter articulated local geometry. Pixal3D uses
`models/Pixal3D-weights`, DINOv3
`models/dinov3-vitl16-pretrain-lvd1689m`, MoGe-2
`models/moge-2-vitl/model.pt`, and RMBG-2.0 `models/RMBG-2.0`; inherited
Pixal parameters are seed `42`, 1024 cascade, 12 sparse/shape/texture steps,
guidance `7.5/7.5/1.0`, 300k decimation, 2048 texture, and 100k surface samples.

**Frozen route and parameters.** After native Pixal--MoGe alignment and the
two-camera bridge/coupled residual, every sample scores identity and shared
pixel-pair Sim(3) residual fractions `1/8, 1/4, 1/2, 3/4, 1` using a 30-degree,
scale `[0.45,2.40]`, translation `1.25` partial-bbox-diagonal capture region.
The winning transform feeds the unchanged narrow pixel residual, 32k-point
Camera-1 standard/wide/final continuations (wide/final `1.0°`), and the
existing Gaussian edit/decode: 1.0 px saved/virtual matches, six 384-px views,
`.075` anchor/displacement caps, 8-neighbor/1.8-edge graph, `.0015` screening,
six `.01` protection views, `.10` exclusion, remote gain `1.0`, and CG
`1e-5/240`. No GT/CD/EMD/category/sample-specific decision is used in inference.

**Offline-only evidence.** With 16,384 points and seed `6145`, the new route
reports mean CD-L1×100 / EMD×100 `1.5738481 / 2.4813253` compared with the
previous frozen route's `1.5682055 / 2.4785676` (`+0.36% / +0.11%`). This
small bounded regression was accepted to make the mainline robust to the large
wrong-basin failures independently observed on custom scans; offline metrics
remain reporting-only and do not control inference.

## 2026-09-05 CST — accepted scene wrapper single-instance registration POC

**User approval.** The user judged the registered scene mug as “配得确实不错”.
Keep this POC as the verified scene-coordinate-frame reference; do not change
the direct-RGB semantic contract, shared-Pixal-MoGe partial definition, or
textured-mesh export without a new scene audit.

**Sample and outputs.** `coffee_mug_0` from
`data/scene_samples/scene_3.png`. Inputs and all produced assets are under
`workspace/scene3_gpt_mask_poc/run/`: source mask/crop
`masks/coffee_mug_0.png` and `instances/coffee_mug_0/masked_crop.png`, partial
`inputs/partial/coffee_mug_0.ply`, frozen direct semantic
`inputs/camera/coffee_mug_0/img.png` (identical to
`inputs/pixal/coffee_mug_0/gpt_image.png` after 512px aspect-preserving white
padding), Pixal GLB `inputs/pixal/coffee_mug_0/pixal3d.glb`, final registration
`registration/coffee_mug_0/final/camera1_amplified_registered_100k.ply`, and
registered textured-mesh outputs
`scene_meshes/coffee_mug_0_registered_mesh.glb` and
`scene_meshes/completed_scene_registered_meshes.glb`.

**Generation contract.** GPT direct semantic prompt, verbatim:
`Starting only from this isolated RGB scene crop, produce exactly one complete plain white ceramic coffee mug on a pure white square background. Preserve the observed camera viewpoint, silhouette, aspect ratio, material, colour, and every visible structural detail. Complete only genuinely occluded or missing portions; do not rotate, resize disproportionately, or redesign the object. Do not output a depth map or add props, text, people, floor, wall, shadows, or a second object.`
Negative prompt: `NONE`. GPT model/version, seed, scheduler, CFG, and internal
generation resolution: `UNKNOWN`. The mask prompt is saved verbatim in
`gpt_image_actions.json`. No depth-to-semantic/Qwen step was run; the scene
partial is a 1-pixel-eroded GPT-mask subset of the single scene Pixal-MoGe
observation. The final POC predates the later generic stronger pose-lock text,
but preserves the accepted direct-RGB/no-rotation contract.

**Models and fixed post-processing.** Pixal metadata is preserved in
`inputs/pixal/coffee_mug_0/pixal3d_metadata.json`: Pixal3D
`models/Pixal3D-weights`, DINOv3
`models/dinov3-vitl16-pretrain-lvd1689m`, MoGe-2
`models/moge-2-vitl/model.pt`, RMBG-2.0 `models/RMBG-2.0`; seed `42`, 1024px,
12 sparse/shape/texture steps, guidance `7.5/7.5/1.0`, 300k decimation, 2048
texture, and 100k sampled points. Registration is the fixed two-camera
Pixal-MoGe Camera-1 continuation recorded in
`registration/batch_manifest.json`: no proposal gate, broad basin recovery
enabled, 32k Camera-1 points, and wide/final tilt trust regions `1.0°`.
The GLB is transformed only by the cumulative final registration Sim(3);
there is no point fusion, Gaussian edit, point deletion, GT, CD, or EMD use.

## 2026-09-05 CST — accepted five-scene direct-GPT textured mesh delivery

**User approval.** “ok 效果很好”. Preserve the isolated scene pipeline and
its shared scene-MoGe coordinate frame. In particular, do not reintroduce a
depth-shift trust-region cap or alter object-level mainline code without a new
scene audit.

**Samples and final outputs.** The accepted delivery root is
`workspace/final_scene_meshes_20260905/`, with source image, final textured
GLB, collision/placement manifest, and verbatim GPT action record in each of
`scene_1` through `scene_5`. The final files are
`scene_<n>/completed_scene_registered_meshes.glb`. Scene 3 is the previously
accepted seven-instance scene output copied from
`workspace/scene3_all_instances_mesh_erosion5_20260905/run/scene_meshes_scene_moge_collision_exact_20260905/`;
scenes 1, 2, 4, and 5 were run in
`workspace/scene_batch_all_instances_20260905/`. The delivery contains 4, 4,
7, 7, and 5 textured instance meshes for scenes 1--5 respectively. All GLB
nodes have identity transforms because every placement Sim(3) is baked into
the textured mesh vertices. `scene_5` additionally records the user-corrected
full left blue upholstered armchair mask in
`scene_5/scene5_instance_board_corrected.png`.
For reproducibility, the delivery also retains `scene_<n>/run/` for all five
scenes: scene-MoGe/context, masks, crops, frozen GPT RGB, Camera-1/Pixal
inputs, Pixal mesh/100k sample/native-MoGe artifacts, registration/bridge
transforms, and logs. A file-level audit verified this complete chain for
every 4/4/7/7/5-instance scene.

**Inputs, prompts, and models.** Inputs are exactly
`data/scene_samples/scene_<n>.png`. There is no Qwen/depth-to-semantic step:
one shared scene MoGe observation is inferred directly from each RGB input,
then a 5-pixel-eroded GPT instance mask extracts each visible object partial.
The complete per-instance mask and semantic prompts are copied verbatim in
each delivery `gpt_image_actions.json`. In literal template form, for an exact
recorded label `<label>`, the mask prompt is:
`Segment exactly one visible <label> in this scene. Preserve the source image pixel grid and output a binary mask at the exact same resolution: pure white for pixels belonging to this <label>, pure black for everything else. Do not include shadows, floor, wall, adjacent objects, or background. Do not redraw the object.`
The direct completion prompt is:
`Starting only from this isolated RGB scene crop, produce exactly one complete <label> on a pure white square background. POSE LOCK: the crop is the only camera reference. Preserve the exact observed object direction, left/right relation, yaw, pitch, roll, image-plane angle, perspective/foreshortening, silhouette, aspect ratio, material, colour, and visible structural details. Complete only genuinely occluded or missing portions behind the observed view. Never canonicalize it into a frontal, side, top-down, symmetric, catalog, or product-shot viewpoint; do not rotate, mirror, re-pose, resize disproportionately, or redesign it. Do not output a depth map or add props, text, people, floor, wall, shadows, or a second object.`
The exact `<label>` substitutions and asset paths are in those records. GPT
image model/version, seed, scheduler, CFG, negative prompt, and raw output
resolution are `UNKNOWN`; the installed Camera-1 and Pixal image is
aspect-preserving white-padded to 512 px. Pixal uses
`models/Pixal3D-weights`, DINOv3
`models/dinov3-vitl16-pretrain-lvd1689m`, MoGe-2
`models/moge-2-vitl/model.pt`, and RMBG-2.0 `models/RMBG-2.0`; seed `42`,
1024 cascade, 12 sparse/shape/texture steps, guidance `7.5/7.5/1.0`, and
100k-face decimation target. No GT, CD, or EMD is read during scene inference.

**Frozen post-processing.** Each complete textured Pixal mesh undergoes only
native Pixal--MoGe alignment plus the camera-2 → scene-MoGe bridge and a
common table-world transform. No point fusion, Gaussian editing, or mesh-face
pruning occurs. Exact `python-fcl` contact detection then moves only the
scene-MoGe-farther object along the original scene camera's positive depth
direction by the smallest collision-free distance; there is no object-scale
movement cap or lateral/table-normal adjustment. The re-exported scene 2 and
scene 4 reports have zero remaining pairs after shifts of `2.977501` for the
tent and `2.115/.318/.43575` for the coffee table/left side table/right side
table. Scene 1, scene 3, and scene 5 already report zero remaining FCL pairs.
The full project test suite passed `60/60` after the unbounded minimal-depth
solver change.

## 2026-09-07 CST — accepted bounded agentic Redwood-10 probe

**User approval.** The user judged the ten-sample agentic result as “效果非常好”.
Keep this isolated feasibility result, its explicit front/back re-plan for
07306, and the fixed conservative local-edit trust region unchanged unless a
new audit is requested. This is not the frozen object-mainline baseline.

**Samples and outputs.** Inputs are exactly `data/redwood/partial/<id>.ply`
for `01184, 05117, 05452, 06127, 06145, 06188, 06830, 07136, 07306, 09639`.
The accepted ten-sample metric summary is
`workspace/agentic_prior_adaptation_probe_20260906/redwood10_corrected_view/offline_metrics/redwood10_conservative_07306_summary.json`:
offline CD-L1x100 `1.5834`, EMDx100 `2.3504` at 16,384 points, metric seed
6145. GT was read only by this completed offline audit. The nine original
per-sample trace roots are `workspace/agentic_prior_adaptation_probe_20260906/<id>/`.
The user-approved 07306 front-facing re-plan is
`workspace/agentic_prior_adaptation_probe_20260906/07306_front_view_replan/`;
its final conservative decoded 100k PLY is
`gaussian/07306/decoded/partial_anchored_gaussian_decoded_100k.ply`.
The direct trace initially accepted the pre-edit registered carrier, but the
reported ten-sample summary intentionally substitutes this audited conservative
07306 decode. All final clouds contain 100,000 slots.

**Observation and prompt contract.** The planner selected the bounded
partial-only convex-hull candidate `base` for all samples except 07306. The
07306 rear-view failure was retained separately; the accepted retry selected
the explicit antipodal `opposite_180` candidate before any semantic/prior
tool was run. Qwen received `raw_depth.png` and the following prompts,
verbatim (negative prompt: `' '` for every sample):
```
01184: 生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的rubbish bin，纯白背景
05117: 生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的red chair，纯白背景
05452: 生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的armchair，纯白背景
06127: 生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的terracotta flower pot with leafy plant，纯白背景
06145: 生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的table，纯白背景
06188: 生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的red motorcyle，纯白背景
06830: 生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的tricycle，纯白背景
07136: 生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的leather sofa，纯白背景
07306: 生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的red office trash can，纯白背景
09639: 生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的Ergonomic Chair，纯白背景
```
Each exact `qwen_edit_prompt.txt`, depth raster, semantic image, camera,
pixel UV map, Pixal input, registration trace, and state-hashed decision is
stored under its corresponding probe root. Qwen uses
`models/Qwen-Image-Edit-2511` plus
`models/nunchaku-qwen-image-edit/nunchaku_qwen_image_2511_balance_int4.safetensors`,
40 steps, true CFG 4.0, 512px Camera-1 semantic output, and 1024px native
stage-1 generation. Qwen seed/scheduler: `UNKNOWN` (no explicit seed was
passed). No GPT clarity edit was used in this isolated probe.

**Prior, registration, and adaptation.** Pixal uses
`models/Pixal3D-weights`, DINOv3
`models/dinov3-vitl16-pretrain-lvd1689m`, MoGe-2
`models/moge-2-vitl/model.pt`, and RMBG-2.0 `models/RMBG-2.0`; Pixal seed
42, 1024px input, 12 sparse/shape/texture steps, guidance `7.5/7.5/1.0`, and
100k sampled carrier. Every trace uses the bounded native Pixal--MoGe,
pixel-indexed two-camera bridge, joint proper Sim(3), and three-stage
Camera-1 residual continuation; all continuous variables remain in these
deterministic executors. Gaussian adaptation is used only where stated in the
trace: historic 01184 uses the prior `.075` trial, 09639 uses `.020`, and the
accepted 07306 retry uses `.020` anchor/displacement ratios, 1px saved/virtual
pixel support, six virtual views, six prior-protection views, and 100k-slot
collision-free decode. The 07306 editable Gaussian motion is capped at 2% of
the partial diagonal; it anchors 26.427% of carrier slots while retaining the
complete unobserved prior support. No decision used GT, CD, or EMD.

**Approved constraints.** Do not overwrite the failed rear-view 07306 trace;
it is the auditable counterexample for observation re-planning. Do not
silently promote this probe over the frozen object mainline, and do not loosen
the agent-local 2% trust region without a cross-sample wheel/detail audit.
