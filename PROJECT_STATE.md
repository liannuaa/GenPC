# Accepted project state

## 2026-09-11 00:04:06 CST — integrated PosteriorAdapter observation fusion

**User approval.** “确实 效果好特别多.” The approved experiment integrates
physical partial evidence into the smooth PosteriorAdapter deformation instead
of replacing points after deformation. Preserve these two outputs and the
shared configuration while running the remaining samples; do not overwrite
them with hard carrier-slot replacement.

**Outputs and inputs.** Approved predictions are
`workspace/from_scratch_01184_debug_20260910/posterior_integrated_fusion_v2/01184/posterior_prior_100k.ply`
and
`workspace/from_scratch_01184_debug_20260910/posterior_integrated_fusion_v2/07136/posterior_prior_100k.ply`.
Their partial overlays and four-view overlays are beside each prediction as
`partial_gray_integrated_red.ply` and `integrated_four_view_projection.png`.
Inputs are `data/redwood/partial/{01184,07136}.ply`, the registered priors under
`workspace/from_scratch_01184_debug_20260910/registration/<id>/final/`, and the
four-camera manifests under `residuals/<id>/render/render_manifest.json`.

**Upstream model record.** This experiment did not regenerate images or 3-D
priors. It reuses Pixal3D `models/Pixal3D-weights`, DINOv3
`models/dinov3-vitl16-pretrain-lvd1689m`, MoGe-2
`models/moge-2-vitl/model.pt`, and RMBG-2.0 `models/RMBG-2.0`; generation
resolution/seed/steps/guidance remain the saved upstream values in
`inputs/pixal/<id>/pixal3d_metadata.json`. New model call, scheduler, seed, CFG,
negative prompt, and semantic postprocessing: `NOT RUN` / `NOT APPLICABLE`.
The exact already-saved semantic prompt for 01184 is:

> Use case: sketch-to-render
> Asset type: camera-consistent semantic conditioning image for single-image 3D generation
> Primary request: Generate a complete, realistic RGB image by completing the object represented by Image 1, which is an incomplete depth map. The object category is a blue wheeled rubbish bin / outdoor garbage bin.
> Input images: Image 1 is the sole geometric and camera authority: an occluded depth rendering from a partial 3D scan.
> Scene/backdrop: pure white studio background, no floor clutter.
> Subject: one complete blue wheeled rubbish bin with a rectangular tapered bin body, closed hinged lid, rear handle/hinge structure, and exactly two black wheels mounted on the same axle. The two wheels must be front/back parallel wheels as implied by the depth image, not side-by-side decorations.
> Style/medium: clean realistic product photograph, coherent plastic and rubber materials, sharp boundaries, sufficient local detail for image-to-3D reconstruction.
> Composition/framing: preserve exactly Image 1's camera viewpoint, perspective, object orientation, apparent scale, image-plane center, crop, silhouette, body proportions, lid pose, and visible wheel locations. Complete only genuinely missing or occluded surfaces.
> Constraints: follow the depth silhouette and depth ordering closely; keep both wheels parallel and structurally attached; retain the slight three-quarter view; output one complete connected object. Do not rotate, mirror, recrop, rescale, recenter, tilt, shorten, widen, or redesign the bin. Do not add extra wheels, pedals, handles, text, logos, shadows that alter the silhouette, or any secondary object.
> Avoid: stylization, malformed geometry, detached parts, duplicated wheels, changed pose, changed camera, black background, text, watermark.

The exact already-saved semantic prompt for 07136 is:

> Use case: sketch-to-render. Create one complete, realistic RGB product image of a long black leather sofa from the supplied incomplete depth map. Image 1 is the sole geometric, camera, and foreground-mask authority. Treat every non-black depth pixel and its silhouette as a hard ControlNet-like spatial constraint: preserve exactly the oblique end-on camera viewpoint, perspective, image-plane center, apparent width and height, crop, orientation, depth ordering, visible sofa end, seat plane, backrest line, and the strong foreshortening of the long sofa body. Complete only genuinely missing or occluded surfaces along the depth direction. Keep one coherent full-length sofa with a continuous seat, continuous backrest, two end armrests, and short attached feet, in dark leather on a pure white background. Do not turn it into a single chair or sectional sofa, do not shorten or widen the body, and do not rotate, mirror, recenter, rescale, tilt, redesign, or move observed surfaces. Sharp realistic boundaries and upholstery; no cushions detached from the sofa, text, logo, watermark, secondary object, clutter, or silhouette-changing shadow.

**Frozen integrated-fusion configuration.** Four saved cameras; 6144 spatially
sampled observation anchors; 1 px proposal radius; 2 px cross-view radius;
depth and metric anchor limits `.075` of partial diagonal; data weight `2.0`;
screening `.004`; three iterations; displacement cap `.05` of partial
diagonal; hidden coverage at least `.95`. The anchors enter the same embedded
ARAP carrier as soft targets with no hard point replacement, concatenation,
category rule, part rule, GT, CD, or EMD. Both outputs retain exactly 100,000
points and no new connected component. Offline-only CD-L1/EMD x100 are
`1.0650/2.0588` for 01184 and `1.7763/3.4295` for 07136.

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
The consolidated ten-sample metric summary is
`workspace/agentic_redwood10_final_20260907/metrics/redwood10_offline.json`:
offline CD-L1x100 `1.5834`, EMDx100 `2.3454` at 16,384 points, metric seed
6145. GT was read only by this completed offline audit. All complete per-sample
traces are retained under `workspace/agentic_redwood10_final_20260907/samples/<id>/`,
with the selected predictions collected in
`workspace/agentic_redwood10_final_20260907/final_predictions/`. The
user-approved 07306 front-facing re-plan is retained as
`workspace/agentic_redwood10_final_20260907/samples/07306/`; its selected
conservative decoded 100k PLY is `final/agent_selected_100k.ply`.
The direct trace initially accepted the pre-edit registered carrier, but the
reported ten-sample summary intentionally substitutes this audited conservative
07306 decode. All final clouds contain 100,000 slots.

**Observation and prompt contract.** The planner selected the bounded
partial-only convex-hull candidate `base` for all samples except 07306. The
accepted 07306 re-plan selected the explicit antipodal `opposite_180`
candidate before any semantic/prior tool was run. Qwen received `raw_depth.png`
and the following prompts,
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

**Approved constraints.** The old rejected rear-view 07306 trace was retired
during consolidation; do not overwrite the delivered front-facing re-plan.
Do not silently promote this probe over the frozen object mainline, and do not
loosen the agent-local 2% trust region without a cross-sample wheel/detail
audit.

## 2026-09-08 CST — domestic-pig camera-conditioned Gaussian diagnostic

**User feedback.** The user confirmed only that the `domestic_pig` result's
*global scale* is “确实好了不少”. They explicitly found the local front- and
rear-leg alignment still inadequate. Retain this output as a diagnostic
baseline, not as an accepted completion or a promotion candidate.

**Exact outputs.** The approved candidate is
`workspace/custom10_fixed_mainline_20260907/pig_camera_conditioned_gaussian_20260908c/`.
Its complete 100k Gaussian-mean carrier is
`camera_conditioned_gaussian_editable_prior_100k.ply`; the partial-gray / prior-red
overlay is `camera_conditioned_gaussian_partial_gray_prior_red.ply`; the saved
Camera-1 overlay is `camera_conditioned_gaussian_saved_view_projection.png`;
the six-view diagnostic is `adapted_virtual_views.png`; and
`camera_conditioned_gaussian_field.npz` records the means and
`editable/locked/protected/moved` masks.

**Inputs and execution.** The prior was the accepted global extent carrier
`workspace/custom10_fixed_mainline_20260907/pig_global_axis_scale_probe_20260907/principal_then_depth_extent/camera_axis_scaled_prior_100k.ply`.
The observed partial and Camera-1 are respectively
`workspace/custom10_fixed_mainline_20260907/inputs/partial/domestic_pig.ply`
and `workspace/custom10_fixed_mainline_20260907/inputs/camera/domestic_pig/camera.pth`;
the semantic image used only for the overlay is
`workspace/custom10_fixed_mainline_20260907/inputs/camera/domestic_pig/img.png`.
The tool was run with `CUDA_VISIBLE_DEVICES=0`, 512px render, padding `.15`,
Camera-1 residual threshold `.05` of partial diagonal, locked-support threshold
`.018`, and soft projection/depth data weight `.25`. No image or 3D generator
was invoked in this edit stage; model/checkpoint, seed, CFG, scheduler, prompt,
and negative prompt are therefore `N/A`.

**Method and diagnostic boundary.** The edit preserved all 100k prior slots,
used no GT/CD/EMD during inference, selected 562 coherent Camera-1 controls,
locked 2,091 aligned visible means, protected 48,015 unobserved means, and
moved 47,687 means through a local Gaussian graph field. Its no-GT control
projection/depth residual diagnostics fell by `28.84%/29.22%`. Do not overwrite
these artifacts while diagnosing the leg residual. Do not promote this
candidate or claim local alignment is solved; assess a camera-pixel status map
and a stronger partial-depth adaptation before any generalization or mainline
decision.

**Subsequent user feedback.** The user found the three-stage continuation
(`pig_camera_conditioned_gaussian_20260908f`) still insufficient to bridge the
local prior/partial discrepancy at the front and rear legs. Treat every
means-only candidate in `pig_camera_conditioned_gaussian_20260908[a-f]` as
rejected for final completion; retain them only as evidence that scale can
improve while missing observed support cannot be recovered by moving existing
prior slots alone.

## 2026-09-08 12:13:59 CST — accepted generic PosteriorAdapter pig result

**User approval.** The user judged the result “效果很顶级”. Preserve the
category-agnostic Partial-OT plus hierarchical embedded-ARAP posterior and its
fixed shared configuration while running the remaining Custom and Redwood
generalization audit. Do not add pig/animal/limb labels, coordinate-side rules,
manual regions, sample thresholds, or GT-driven selection.

**Exact inputs and outputs.** The fixed registered 100k carrier is
`workspace/custom10_fixed_mainline_20260907/registration/domestic_pig/final/camera1_amplified_registered_100k.ply`;
the observed partial, Camera-1, depth-derived semantic image, and upstream
Pixal assets are respectively
`workspace/custom10_fixed_mainline_20260907/inputs/partial/domestic_pig.ply`,
`workspace/custom10_fixed_mainline_20260907/inputs/camera/domestic_pig/camera.pth`,
`workspace/custom10_fixed_mainline_20260907/inputs/camera/domestic_pig/img.png`,
and `workspace/custom10_fixed_mainline_20260907/inputs/pixal/domestic_pig/`.
The accepted posterior root is
`workspace/custom10_fixed_mainline_20260907/pig_partial_ot_embedded_posterior_20260908h/`.
Its final fixed-cardinality result is `posterior_prior_100k.ply`; the exact
coarse output, OT table, support states, displacement field, saved-camera
overlay, six-view overlay, and complete diagnostic record are
`coarse_prior_100k.ply`, `partial_ot_pairs.npy`,
`posterior_support_status.png`, `posterior_field.npz`,
`posterior_saved_view_projection.png`, `posterior_virtual_views.png`, and
`posterior_info.json`. No prior point is deleted and no partial point is
concatenated. The rebuilt 10-neighbour output surface graph has exactly the
same two connected components and sizes as the input carrier: one 99,958-point
body and its pre-existing 42-point sampling island. Hidden-view occupancy is
`1.23948`, above the required 95% retention.

**Posterior execution parameters.** No learned model, prompt, negative prompt,
seed, scheduler, CFG, or image resize is used by this posterior-only step;
those fields are `N/A`. The full exact configuration is serialized verbatim in
`posterior_info.json`: partial/prior OT samples `3072/6144`, 24-neighbour local
features, 60 unbalanced-Sinkhorn iterations, relaxation `.72`, temperature
floor `.012`, normalized 3D/screen/depth gates `.34/.28/.34`, residual
quantiles `.22/.42`, graph-node fractions `.020/.055` bounded to `192--6144`,
four-node skinning, coarse graph `18` neighbours / `2.45` edge ratio / data
weight `2.5` / screening `.0008` / five iterations / displacement cap `.28`
partial diagonal, fine graph `10` / `1.85` / `4.0` / `.003` / five / `.12`,
attachment ratio `3.2`, minimum observed improvement `.025`, edge compression
`.55`, robust p99 edge-stretch cap `2.8`, hidden coverage `.95`, and coverage
resolution `160`. CUDA is used for batched OT; the sparse graph solve is
deterministic CPU SciPy. All geometric thresholds are dimensionless ratios,
median-spacing scales, residual quantiles, or visibility fractions.

**Upstream reproducibility.** The reused Qwen prompt is, verbatim,
`生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的domestic pig，纯白背景`;
negative prompt is one blank space (`' '`), true CFG `4.0`, 40 steps, 512px
Camera-1 semantic output, and 1024px native generation. Qwen seed and scheduler
are `UNKNOWN`. The GPT clarity prompt is copied verbatim in
`inputs/pixal/domestic_pig/prompt.txt`; GPT model/version, negative prompt,
seed, scheduler, CFG, and raw output resolution are `UNKNOWN`, so the upstream
image generation is not exactly reproducible from metadata alone. The reused
Pixal prior uses `models/Pixal3D-weights`,
`models/dinov3-vitl16-pretrain-lvd1689m`, `models/moge-2-vitl/model.pt`, and
`models/RMBG-2.0`, seed `42`, requested/actual resolution `1024`, xFormers,
12 sparse/shape/texture steps with guidance `7.5/7.5/1.0`, 300k-face target,
and 2048 texture size. The posterior step itself is exactly reproducible from
the four fixed inputs above.

**Offline firewall.** GT is
`data/custom/domestic_pig/gt_data/domestic_pig_gt.ply` and was opened only
after the posterior was frozen. At 16,384 points and seed 6145,
`offline_metrics.json` reports CD-L1x100 `1.47515414` and EMDx100
`1.84705555`; these values were never available to Partial OT, ARAP, line
search, support-state classification, or acceptance.

## 2026-09-08 18:08 CST — fresh Redwood10/Custom10 candidate (pending visual approval)

**Status.** This is a complete from-scratch generalization candidate, not an
accepted replacement for the frozen object mainline.  The user identified the
fresh `07136` sofa and `07306` waste-bin semantic observations as having
unreasonable views/local scale.  The recovery below changes only the generic
bounded view action and reruns the ordinary semantic, prior, registration and
posterior tools.  It contains no category, sample, part, coordinate-side or GT
condition.  Do not promote or delete the rejected attempts until the user has
visually reviewed the selected point clouds.

**Complete artifacts.** The immutable experiment root is
`workspace/posterior_adapter_e2e_redwood_custom_20260908/`.  Redwood final
100k carriers are hard-linked under `redwood/final_selected/<id>/` and Custom
final 100k carriers under `custom/final_selected/<name>/`.  Exact source
selection is in `redwood/final_selected/selection_manifest.json`; aggregate
and per-sample offline metrics are in each dataset's `offline_metrics.json`
and `offline_metrics.csv`.  Every generated image, native Pixal artifact,
registration state, verifier state, rejected candidate and PosteriorAdapter
diagnostic remains below this root.  The run summary is `RUN_SUMMARY.md`.
For convenient review, the selected artifacts are additionally organized by
sample under `workspace/posterior_adapter_selected_redwood_custom_20260908/`.
Each `redwood/samples/<id>/` and `custom/samples/<name>/` directory contains
the selected inputs/intermediates and one canonical `final_100k.ply`. These
are hard links to the immutable source artifacts, not regenerated copies; all
20 final files were verified at exactly 100,000 vertices.

**Generic view recovery.** The bounded candidate family is `base`,
`opposite_180`, and yaw rotations `+/-30`, `+/-60`, `+/-90` degrees around the
partial-derived camera.  A partial-only diagnostic board selects a discrete
camera; semantic/prior branches are accepted using silhouette, depth and
visible-3D verifier evidence, never GT/CD/EMD.  The selected sofa branch is
`redwood/view_retry_alt/07136/` (`opposite_180`), whose final posterior is
`redwood/view_retry_alt/posterior/07136/posterior_prior_100k.ply`.  The selected
waste-bin branch is `redwood/view_retry/07306/` (`opposite_180`), whose final
carrier is `redwood/view_retry/07306/final/agent_selected_100k.ply`.  The
rejected malformed sofa attempt is `redwood/view_retry/07136/`; the rejected
oblique-bin attempt is `redwood/view_retry_alt/07306/`.  A GPT clarity probe at
`redwood/gpt_bin_candidate/` was also rejected by the no-GT verifier and is not
the selected result.

**Semantic and model record.** Qwen prompts are saved verbatim next to each
candidate camera in its `qwen_edit_prompt.txt`; they use the fixed form
`生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的<类别名>，纯白背景`,
negative prompt one blank space (`' '`), true CFG `4.0`, 40 steps, a 512px
camera semantic result and 1024px native generation.  Qwen uses
`models/Qwen-Image-Edit-2511` with
`models/nunchaku-qwen-image-edit/nunchaku_qwen_image_2511_balance_int4.safetensors`;
seed and scheduler are `UNKNOWN`.  Pixal uses `models/Pixal3D-weights`,
`models/dinov3-vitl16-pretrain-lvd1689m`, `models/moge-2-vitl/model.pt`, and
`models/RMBG-2.0`, seed `42`, resolution `1024`, 12 sparse/shape/texture steps,
guidance `7.5/7.5/1.0`, xFormers, 300k-face target, 2048 texture and a 100k
point carrier.  The rejected GPT-bin prompt is copied verbatim in
`redwood/gpt_bin_candidate/inputs/pixal/07306/gpt_image_prompt.txt`; GPT model
version, seed, scheduler, CFG and negative prompt are `UNKNOWN`.

**Geometry and posterior record.** Registration uses the existing
Pixal--MoGe--partial cross-camera Sim(3) plus the ordinary three-stage
broad-to-narrow Camera-1 residual refinement.  The same accepted generic
PosteriorAdapter configuration recorded above is then applied; it preserves
all 100k prior slots and uses no GT.  On `07136` it intentionally returns the
identity because no coherent residual transport passes its structural gates.
On `07306` the existing uniform `.075` partial-anchored Gaussian action is
accepted because no-GT energy improves from `0.1045409` to `0.0948769`, while
both 1% and 2% visible-surface coverage reach `1.0` and all 100k slots remain.

**Offline-only report.** At 16,384 points and seed 6145, selected Redwood10
mean CD-L1x100/EMDx100 is `1.50174597/2.37392348`; selected Custom10 mean is
`2.01179441/2.99937962`.  The recovered `07136` is
`1.11914463/2.02244818` and `07306` is `3.47092301/3.88251394`.  These metrics
were computed only after each no-GT branch decision and were not exposed to
view selection, registration, adaptation or stopping.

**Visual-review update.** The user judged all ten consolidated Redwood outputs
good, but found that some Custom outputs lose too much local detail. Therefore
Redwood is visually acceptable, while the consolidated Custom candidate is
explicitly rejected for final promotion pending a category-independent audit
of OT confidence, stable-region drift and deformation-graph strain. Do not
interpret the improved Custom aggregate metric as approval, and do not tune
the repair using GT/CD/EMD or introduce sample/category branches.

## 2026-09-08T22:17:51+08:00 — accepted Pig cross-view residual image conditions

**User approval and boundary.** The user judged the final FRONT/SIDE/BACK
conditions “完全ok”. Preserve these images, their view order, Camera-1 anchor,
local contours, object identity, and prompt logic. This approval applies to
the image conditions only; the TRELLIS 3-D output and any later registration
remain pending visual/metric review and must not be presented as accepted.

**Exact inputs and accepted images.** The registered Pixal carrier and observed
partial are
`workspace/custom10_fixed_mainline_20260907/registration/domestic_pig/final/camera1_amplified_registered_100k.ply`
and `data/custom/domestic_pig/partial_data/single_scan/domestic_pig_partial.ply`.
The camera/render contract is
`workspace/trellis_pig_multiview_probe_20260908/conditions/original/render_manifest.json`.
The no-GT component evidence, numeric ledger, edit prompt, accepted generated
board, and final Camera-1-frozen TRELLIS inputs are respectively:

- `workspace/trellis_pig_multiview_probe_20260908/conditions/cross_view_ledger_v4/cross_view_residual_ledger_board.png`;
- `workspace/trellis_pig_multiview_probe_20260908/conditions/cross_view_ledger_v4/cross_view_residual_ledger.json`;
- `workspace/trellis_pig_multiview_probe_20260908/conditions/cross_view_ledger_v4/cross_view_residual_refinement_prompt.txt`;
- `workspace/trellis_pig_multiview_probe_20260908/conditions/cross_view_ledger_v4/gpt_cross_view_refined_front_side_back.png`;
- `workspace/trellis_pig_multiview_probe_20260908/conditions/cross_view_ledger_v4_hybrid/trellis_condition_{front,side,back}.png`.

The final front condition is copied byte-for-byte from the unedited registered
Pixal Camera-1 render; only the GPT side/back panels are retained from the
accepted generated board. The object-type argument is `domestic pig`. The
shared algorithm clusters residual partial points in 3-D and projects the same
component colour into every view; it contains no pig/animal/limb labels or
sample-specific direction, crop, threshold, or GT decision.

**Verbatim GPT prompt.** The built-in image generation/editing tool received:

```text
Use case: precise-object-edit
Asset type: cross-view geometry-conditioned input for 3D regeneration

Image 1 is the edit target: one clean FRONT / SIDE / BACK strip of the same complete domestic pig.
Image 2 is a four-row diagnostic board for the same cameras. In each view column the rows are prior RGB, prior depth, partial depth, and shared 3-D residual components. In the final row, dim points are current prior locations, bright points are observed target locations, and arrows point from prior to target. The same colour identifies the same connected 3-D residual component in every view.

Component ledger:
- C1 colour RGB (255, 92, 205), dominant correction: median/p90 residual 0.1565/0.1875 of the observed diagonal.
- C2 colour RGB (108, 238, 94), dominant correction: median/p90 residual 0.1689/0.1832 of the observed diagonal.
- C3 colour RGB (255, 168, 48), fine correction: median/p90 residual 0.0200/0.0355 of the observed diagonal.
- C4 colour RGB (77, 166, 255), fine correction: median/p90 residual 0.0156/0.0211 of the observed diagonal.
- C5 colour RGB (245, 226, 66), fine correction: median/p90 residual 0.0194/0.0247 of the observed diagonal.
- C6 colour RGB (174, 112, 255), fine correction: median/p90 residual 0.0164/0.0211 of the observed diagonal.

Refine Image 1 according to the component ledger. Dominant components require a clearly visible correction in every view where their evidence is visible; fine components permit only conservative contour refinement. Apply a component only where its target is supported by partial depth, and express the same connected 3-D correction consistently in every view where that component is visible. Use the depth rows to distinguish an image-plane contour shift from motion toward or away from the camera. Preserve low-residual regions and geometry without positive partial evidence. Propagate each accepted contour change smoothly through its attachment; never translate a structure as a detached piece.

Keep object identity, topology, material, texture, pose, view cameras, framing, lighting, complete unobserved support, and the FRONT / SIDE / BACK order unchanged. Output exactly one clean three-panel strip on pure white, with no labels, component colours, arrows, depth maps, text, borders, watermark, duplicated structures, missing structures, tears, or intersections.
```

GPT model/version, negative prompt, seed, scheduler, CFG, and exposed resize
policy are `UNKNOWN` because the built-in tool does not return them. Its raw
saved output is
`/root/.codex/generated_images/01a06637-4ce7-7450-a13c-54d3b9789130/exec-86e36c88-52c7-491a-8a73-d8d49b7a7103.png`;
the project copy above is the reproducibility asset.

**TRELLIS execution generated after image approval.** The pending 3-D candidate
is under
`workspace/trellis_pig_multiview_probe_20260908/trellis/cross_view_ledger_v4_hybrid_seed42/`.
It contains `trellis_mesh_raw.glb`, `trellis_mesh_raw.ply`,
`trellis_mesh_sampled_100k.ply`, `trellis_gaussian.ply`, the orbit preview and
`trellis_regeneration_info.json`. Model/checkpoint is
`models/TRELLIS-image-large`, official repository commit
`2301d7ba00f6f101695022123bc01363603e4828`; seed `42`, stochastic multi-image
mode, sparse sampler 12 steps / CFG 7.5, structured-latent sampler 12 steps /
CFG 3.0, 100,000 sampled points, 24 preview frames, xFormers attention and
native spconv. Generation took 10.24 seconds after model load and peaked at
10,868,974,080 bytes GPU memory. No RMBG, MoGe, registration, ICP, Gaussian
edit, GT, CD, or EMD was applied after this TRELLIS generation yet.

**Subsequent offline-only 3-D audit.** After the accepted images and TRELLIS
asset were frozen, the full-surface GT-only Sim(3) oracle produced
`workspace/trellis_pig_multiview_probe_20260908/registration/gt_oracle_cross_view_ledger_v4_trim100/`.
At 16,384 points and seed 6145 it reports CD-L1x100 `1.78322364` and EMDx100
`2.46681198`; the older hybrid under the identical oracle reports
`1.72000229/2.41813362`. The new candidate nevertheless improves observed
partial-to-prior normalized mean/p90 distance from `.013376/.032300` to
`.012526/.030124`, and 1%/2% coverage from `56.1401/75.4761%` to
`57.8918/78.7964%`. These values were unavailable to GPT and TRELLIS. Keep the
image conditions accepted, but keep the resulting 3-D candidate pending: it
has better observed-surface agreement but has not surpassed the older hybrid
in complete-shape metrics or passed no-GT registration.

**User promotion at 2026-09-08 after visual review.** The user judged the v4
generated prior “很顶级” and directed subsequent work exclusively to
registration. Treat the accepted GPT board, the Camera-1-frozen three TRELLIS
inputs, and `trellis/cross_view_ledger_v4_hybrid_seed42/` as immutable inputs
for registration experiments. Do not redraw the images, regenerate TRELLIS,
or substitute another seed without asking. This promotion does not approve
any existing TRELLIS-to-partial transform; registration remains the active
problem.

## 2026-09-09 CST — accepted Pig TRELLIS regeneration and depth-visible Sim(3)

**User approval.** The user judged the final result “非常非常好了”. Preserve
the v7 FRONT/SIDE/BACK images, seed-42 TRELLIS asset, and v37 residual Sim(3)
as the accepted Pig TRELLIS feasibility result. Do not replace it with v38:
that second local pass improved its sampled no-GT objective but regressed the
offline complete-shape result. This approval does not promote TRELLIS into the
frozen GenPC++ mainline or authorize sample-specific logic.

**Exact inputs and accepted outputs.** The observed input is
`data/custom/domestic_pig/partial_data/single_scan/domestic_pig_partial.ply`;
its saved camera, depth/pixel indexing and GT-for-offline-audit are
`camera.pth`, `depth.png`, `point_uv.npy`, and
`data/custom/domestic_pig/gt_data/domestic_pig_gt.ply`. The accepted edited
view strip is
`workspace/trellis_pig_multiview_probe_20260908/conditions/cross_view_shared_local_v7/shared_low_frequency_plus_local_front_side_back.png`;
the exact TRELLIS inputs are `stage2_local/{front,side,back}.png`, and the
camera-framing-normalized registration observations are
`camera_contract/{front,side,back}.png`. The accepted generated assets are
under `trellis/shared_low_frequency_local_v7_seed42/`, including
`trellis_mesh_raw.glb`, `trellis_mesh_raw.ply`,
`trellis_mesh_sampled_100k.ply`, `trellis_gaussian.ply`, and
`trellis_gaussian_orbit_preview.png`. The accepted registered output,
partial/prior coloured overlay, Camera-1 projection, and complete record are:

- `registration/trellis_shared_local_v7_depth_visible_sim3_v37/trellis_registered_100k.ply`;
- `registration/trellis_shared_local_v7_depth_visible_sim3_v37/partial_gray_trellis_red.ply`;
- `registration/trellis_shared_local_v7_depth_visible_sim3_v37/partial_camera1_projection.png`;
- `registration/trellis_shared_local_v7_depth_visible_sim3_v37/registration_info.json`.

All paths above are relative to
`workspace/trellis_pig_multiview_probe_20260908/` unless explicitly rooted
elsewhere.

**Image-edit reproducibility.** Stage 1 uses the full verbatim prompt in
`conditions/cross_view_shared_local_v7/stage1_shared_low_frequency_prompt.txt`;
stage 2 uses the full verbatim per-view prompts in
`stage2_local/{front,side,back}_prompt.txt`. Those files are the canonical
prompt record and must be copied byte-for-byte for replay. The prompts perform
one shared low-frequency update followed by local residual edits after
subtracting that shared motion; they contain the object argument `domestic
pig` but no fixed part, side, world-axis, GT, CD, or EMD rule. GPT image model
name/version, negative prompt, seed, steps, scheduler, CFG, and native resize
policy are `UNKNOWN`, so the image-edit stage is not exactly reproducible from
metadata alone. Final images are square 512px white-background conditions;
camera-contract normalization restores the original foreground centre and
isotropic framing before registration.

**TRELLIS reproducibility.** Model is
`/opt/data/private/cr/lab/GenPC/models/TRELLIS-image-large`; repository is
`/opt/data/private/cr/lab/LaS-Comp` at commit
`2301d7ba00f6f101695022123bc01363603e4828`. Exact settings are seed `42`,
stochastic multi-image mode, sparse sampler 12 steps / CFG `7.5`, structured
latent sampler 12 steps / CFG `3.0`, mesh and Gaussian decode, 100,000 sampled
surface points, 24 preview frames, xFormers, and native spconv. Python is the
`las-comp` environment with Python `3.10.19`, Torch `2.4.0`, CUDA `12.1`.
Generation took `7.4516` seconds after model loading and peaked at
`10,935,504,896` GPU bytes. The official multi-image API receives no camera
extrinsics; its three conditions constrain the asset but are not treated as
three calibrated exported cameras.

**Accepted registration.** Initial no-GT input is
`registration/trellis_shared_local_v7_camera_polish/trellis_registered_100k.ply`.
The accepted v37 method is the category-independent all-partial-support plus
saved-Camera-1 depth/visible-surface proper Sim(3) search implemented by
`src/partial_supported_sim3.py` and
`scripts/run_partial_supported_sim3.py`. Exact configuration is rotation
bound `22` degrees, scale bound `1.14`, translation bound `.18` partial
diagonal, 8,192 partial and 24,000 prior samples, population `5`, 14
iterations, seed `6145`, semantic silhouette weight `0`, and physical
Camera-1 visible score weight `1.2`. It recovers `8.65756` degrees, scale
`1.00291534`, and translation
`[.08896611,.01806257,.00118298]`, preserving all 100k complete-prior points.
No Gaussian/non-rigid edit, point deletion, partial concatenation, semantic
MoGe bridge, GT, CD, or EMD enters this accepted transform.

**Offline firewall and result.** After the transform was frozen, 16,384-point
evaluation with seed 6145 reports CD-L1x100 `2.05028020` and EMDx100
`2.78596357`. The GT-only diagnostic oracle is `1.9392/2.6391` and was used
only to estimate remaining headroom, never to initialize, score, select, or
stop v37. A subsequent v38 local pass reports `2.4164/3.2024` and is rejected.

## 2026-09-09 14:56 CST — promoted fusion-free object mainline

**User decision.** The user directed that the accepted TRELLIS regeneration
and registration be connected to the preceding semantic/Pixal stages and used
as the current mainline, explicitly excluding fusion for now. Preserve the
accepted Pig conditions, TRELLIS asset, final registration parameters, and
100k-point output while validating cross-category transfer. Do not silently
restore Gaussian fusion, PosteriorAdapter, axis-specific deformation, point
deletion, or partial concatenation to the mainline.

**Canonical implementation and output.** The resumable entry point is
`scripts/run_object_mainline.py`; its artifact schema is
`src/object_mainline.py`, and the method contract is
`docs/core_registration_pipeline.md`. The no-regeneration equivalence replay
is
`workspace/mainline_trellis_replay_20260909/final/domestic_pig/complete_100k.ply`
with manifest
`workspace/mainline_trellis_replay_20260909/manifests/domestic_pig.json`. Its
SHA-256 is
`de0efb3dbbbb18369f0395b6173b41e2166ac46e38bcf42f8ccbee5034d3a048`,
byte-identical to the accepted
`workspace/trellis_pig_multiview_probe_20260908/registration/trellis_shared_local_v7_depth_visible_sim3_v37/trellis_registered_100k.ply`.

**Inputs and generation record.** Input partial/camera/semantic/Pixal assets
remain exactly those in the immediately preceding accepted Pig entry. Stage-1
and per-view stage-2 prompts are copied verbatim in
`workspace/trellis_pig_multiview_probe_20260908/conditions/cross_view_shared_local_v7/`;
the accepted FRONT/SIDE/BACK images are `stage2_local/{front,side,back}.png`.
GPT model/version, negative prompt, seed, steps, scheduler, CFG, and native
resize policy remain `UNKNOWN`; the saved images and prompt files are therefore
required replay artifacts. TRELLIS remains
`models/TRELLIS-image-large` with repository commit
`2301d7ba00f6f101695022123bc01363603e4828`, seed 42, stochastic mode, sparse
12/CFG 7.5, structured latent 12/CFG 3.0, xFormers/native-spconv, and 100k
surface samples. Pixal/Qwen/RMBG/MoGe model paths and parameters are unchanged
from the preceding entry.

**Registration and postprocessing.** The fixed mainline performs three-view
semantic orientation capture, 128px/420-step nvdiffrast refinement, one
Camera-1 partial-to-prior inverse capture followed by 256px/400 steps, a
256px/500-step camera polish, and the accepted all-partial-support residual
Sim(3): 22-degree, 1.14-scale and 0.18-diagonal bounds, population 5,
14 iterations, seed 6145, silhouette weight 0, visible Camera-1 weight 1.2.
The replay recovers the same 8.65756-degree rotation, 1.00291534 scale, and
translation `[.08896611,.01806257,.00118298]`. No fusion or post-registration
shape edit is applied. Offline-only CD-L1x100/EMDx100 remains
`2.05028020/2.78596357`; neither quantity was exposed to inference.
## 2026-09-10 14:28:02 CST — accepted Pig four-view diagnosis plus strong continuous posterior

**User approval.** The user judged the stronger Pig result “效果非常好”.
Preserve the category-independent structural-OT plus embedded-ARAP low-frequency
posterior, the four-view diagnostic assets, and the complete 100k carrier. Do
not replace it with the rejected short-range silhouette deformation, and do not
add animal/limb labels, fixed axes, manual regions, or GT-driven decisions.

**Inputs and outputs.** Sample id/type is `domestic_pig`. The fixed registered
prior is
`workspace/custom10_fixed_mainline_20260907/registration/domestic_pig/final/camera1_amplified_registered_100k.ply`;
partial, Camera-1, semantic image, native Pixal GLB, and ordered source carrier
are respectively
`workspace/custom10_fixed_mainline_20260907/inputs/partial/domestic_pig.ply`,
`workspace/custom10_fixed_mainline_20260907/inputs/camera/domestic_pig/camera.pth`,
`workspace/custom10_fixed_mainline_20260907/inputs/camera/domestic_pig/img.png`,
`workspace/custom10_fixed_mainline_20260907/inputs/pixal/domestic_pig/pixal3d.glb`,
and
`workspace/custom10_fixed_mainline_20260907/inputs/pixal/domestic_pig/pixal3d_sampled_100k.ply`.
The accepted experiment root is
`workspace/geometry_first_multiview_proxy_pig_20260910/`. The canonical output
is `final_strong_continuous_posterior_100k.ply`, hard-linked to
`four_view_posterior_low_frequency_v3/posterior_prior_100k.ply`. Exact OT,
field, support, comparison, and diagnostic outputs are
`four_view_posterior_low_frequency_v3/{partial_ot_pairs.npy,posterior_field.npz,posterior_support_status.png,partial_gray_posterior_red.ply,posterior_info.json}`.
The accepted before/after visualization is
`four_view_posterior_low_frequency_v3/registered_before_gray_posterior_after_red_pca.png`.

**Four-view observation.** Four partial-visibility-selected cameras use FOV
`38` degrees and orbit yaws `0`, `180`, `315`, and `135` degrees, recorded with
their full poses in `render/render_manifest.json`. They cover `13,450 / 16,384`
partial points. The clean prior and prior-to-partial boards are
`evidence/prior_rgb_front_side_back_right.png` and
`evidence/partial_correspondence_overlay_front_side_back_right.png`. The
accepted stronger edited condition is `residual_edited_4view_v2_strong.png`,
split without resizing into `edited_conditions_v2/{front,side,back,right}.png`.
Generated resolution is `1254 x 1254`; source boards were supplied at their
native saved resolution. Image-generation model/version, seed, steps, CFG,
scheduler, and negative prompt are `UNKNOWN`; no later guess may be presented
as reproducibility metadata.

**Exact stronger edit prompt.**

```text
Use case: strong camera-consistent structural correction for 3D reconstruction.

Image 1 is the original clean 2x2 four-view RGB target. Image 2 is aligned geometric evidence: cyan is the observed partial-scan target, current prior/RGB-red locations are the source, and yellow arrows point from source to target. Image 3 is a previous edit attempt that was too conservative and failed to express enough low-frequency shape change.

Produce a stronger corrected version of Image 1. Infer one shared continuous three-dimensional deformation from all four evidence panels. Where distant but coherent observed structures lie beyond the current prior, increase the intervening object's low-frequency extent so the connected structures reach the observations. The correction must look like continuous stretching and proportional reshaping of the complete object, not detached endpoint translation. Make the corrected longitudinal proportions and spacing between connected structures visibly different from Image 3 whenever the evidence demands it. Then refine local contours and depth-relative placement consistently in all views.

For this domestic pig, preserve the same individual, anatomy and texture while making its observed body extent and limb placement agree much more strongly with the cyan evidence. All limbs must remain naturally attached; do not change their number.

Freeze the exact row-major 2x2 panel order, camera pose, perspective, crop, object orientation, object centre, white background, identity, topology, material, texture, ears, face and tail. Missing partial regions are UNKNOWN and must not be deleted. Apply one shared 3D correction across views—never independently redesign a panel.

Output only the clean corrected 2x2 RGB board. No diagnostic colours, arrows, text, labels, panel borders or watermark. No mirrored/swapped views, camera changes, detached parts, duplicated/deleted structures, or inconsistent anatomy.
```

**Posterior solver.** No learned model or image prompt is used by the accepted
geometry update itself. It uses `PosteriorAdapterConfig` serialized verbatim in
`posterior_info.json`: partial/prior samples `3072/6144`; 24-neighbour features;
60 unbalanced-OT iterations; coarse/fine graph fractions `.020/.055`; 24
low-frequency modes; five coarse and five fine ARAP iterations; attachment
ratio `3.2`; minimum hidden coverage `.95`. The complete exact numeric config
is the JSON record and must be used instead of a reconstructed summary. Main
axis extent changes from `.71222762` to `.94514656` (about 33%). All `100,000`
carrier slots are retained, significant component count stays one, no connected
component is added, and hidden coverage is `1.23948`. The no-GT verifier accepts
the candidate (`.0523111 -> .0505232`). The weaker broad-nearest-neighbour
result under `four_view_broad_to_narrow_v2/` is diagnostic only.

**Offline firewall.** GT is
`data/custom/domestic_pig/gt_data/domestic_pig_gt.ply` and is never opened by
view selection, image editing, OT, ARAP, or the verifier. Offline-only evaluation
at 16,384 points and seed 6145 gives CD-L1x100 `1.47515414` and EMDx100
`1.84783079` (the CUDA EMD approximation may vary slightly between runs).

## 2026-09-10 — accepted 01184 mask-locked direct-GPT from-scratch flow

**User approval.** The user judged the result “效果很好” and directed that the
same pipeline be run on the remaining Redwood samples. Preserve the saved
Camera-1 depth as the sole camera/framing authority, the direct-GPT semantic,
the mask audit, the Pixal prior, registration, four-view OT evidence, and the
complete 100k carrier. Do not restore the failed Qwen image as the active
semantic input.

**Exact artifacts.** The run root is
`workspace/from_scratch_01184_debug_20260910`. Input partial is
`data/redwood/partial/01184.ply`; depth/camera are
`inputs/camera/01184/{depth.png,camera.pth}`; direct-GPT source and installed
semantic are `inputs/pixal/01184/gpt_image.png` and
`inputs/camera/01184/img.png`. Prompt and framing audit are
`inputs/pixal/01184/gpt_depth_completion_prompt.txt` and
`inputs/camera/01184/semantic_install_manifest.json`. Pixal model outputs are
`inputs/pixal/01184/{pixal3d.glb,pixal3d_sampled_100k.ply,pixal3d_metadata.json,pixal_moge_fp16_observation.npz}`.
Registered and final outputs are
`registration/01184/final/camera1_amplified_registered_100k.ply` and
`final/01184/complete_100k.ply`. Four-view evidence is under
`residuals/01184/`; exact run metadata is `run_manifest.json`.

**Generation record.** The exact direct-GPT prompt is copied verbatim in the
prompt artifact and the semantic-install manifest. It constrains the image to
one blue wheeled rubbish bin, the saved depth camera/framing, exactly two
parallel wheels on one axle, and a white background. GPT image model/version,
seed, steps, CFG, scheduler, and native resize policy are `UNKNOWN`; therefore
the saved source image (SHA-256
`8bc8822cff48ff60b34261ff7c1a2228151639f468e8bf041277259c790052a3`) is required
for exact replay. Installation resizes to 512 square with Lanczos, then uses
`models/RMBG-2.0`. The audit reports center-offset ratio `.005259`, width/height
ratios `1.00909/1.01928`, mask IoU `.83034`, and depth coverage `.97212`.
Pixal3D source/checkpoint and DINO/MoGe paths are those recorded in
`inputs/pixal/01184/pixal3d_metadata.json`; no unrecorded model substitution is
allowed.

**Geometry and evaluation.** Registration uses the shared native
Pixal--MoGe, two-camera bridge, joint, and Camera-1 continuation implementation;
all exact transforms and stage settings are serialized under
`registration/01184`. Four visibility-valid views enter Partial OT through
`residuals/01184/render/render_manifest.json`. PosteriorAdapter preserves all
100,000 points and makes no deformation for this case because reliable
localized residual transport is insufficient; this is not a verifier rollback.
Offline-only CD-L1x100/EMDx100 are `1.34141147/2.33711749` and are not exposed
to generation, registration, OT, or stopping.
