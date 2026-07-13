# GenPC Refactor Plan

This plan tracks the target refactor for partial point cloud reconstruction via
Qwen-Image-Edit, MoGe, Hunyuan3D, and FreeReg.

## Goal

Build a pipeline that reconstructs a complete 3D point cloud from a partial
input point cloud and aligns the complete point cloud back to the original
partial scan.

The target relation is:

```text
partial point index -> MoGe point index -> complete point index / complete aligned to partial
```

## Stage 1 - Partial to Image to MoGe Index Bridge

Status: partially implemented and accepted for `car__132`.

Inputs:
- Raw partial point cloud:
  `workspace/scansalon/_inputs_denoised/car/car__132.ply`

Flow:
1. Project partial point cloud to a depth image.
2. Save camera 1 intrinsics/extrinsics or equivalent projection state.
3. Complete the depth image with Qwen-Image-Edit-2511.
4. Treat the completed image as a semantic/RGB image with the same resolution,
   object pose, and object location as the depth image.
5. Run MoGeV2 on the completed image.
6. Save camera 2 intrinsics/extrinsics or MoGe camera-frame metadata.
7. Use image pixels to establish a point index bridge:
   `partial point index -> MoGe point index`.

Critical details:
- `DepthPrompting.paintPixels()` flips the saved `depth.png` vertically.
- Pixel matching must convert projection uv to saved image uv with `v = 1 - v`.
- Do not use `depth_view_point_cloud.ply` as the raw partial target; it is in
  depth-view coordinates, not raw partial coordinates.

Existing implementation:
- `scripts/run_moge_to_raw_partial_from_camera.py`
- `scripts/run_moge_pixel_index_bridge.py`
- `scripts/run_moge_to_partial_from_index.py`

Accepted `car__132` outputs:
- `workspace/scansalon_zup_side_512/car__132/moge_object_only.ply`
- `workspace/scansalon_zup_side_512/car__132/moge_object_partial_hits_red.ply`
- `workspace/scansalon_zup_side_512/car__132/moge_aligned_to_raw_partial.ply`
- `workspace/scansalon_zup_side_512/car__132/raw_partial_gray_moge_red_aligned.ply`
- `workspace/scansalon_zup_side_512/car__132/moge_to_raw_partial_transform.npy`
- `workspace/scansalon_zup_side_512/car__132/partial_to_moge_index.npy`
- `workspace/scansalon_zup_side_512/car__132/moge_to_raw_partial_info.json`

Open work:
- Integrate this bridge into the main pipeline rather than keeping it as probe
  scripts.
- Preserve accepted `car__132` outputs while refactoring.
- Recover or establish a reproducible Qwen completion prompt/settings baseline.

Current Qwen completion direction:
- Use Qwen-Image-Edit-2511 with `QwenImageEditPlusPipeline`.
- Use one edit stage: input is the projected incomplete depth image, output is a
  complete realistic RGB/semantic image.
- Prompt asks to generate a complete realistic object photo from the incomplete
  depth image while preserving contour, pose, orientation, and camera viewpoint.
- Current default inference steps are `16`.
- Current CFG settings are `true_cfg_scale=4.0` and `negative_prompt=" "`.
- Do not pass `height` or `width`; the Plus pipeline outputs 1024x1024 from the
  512x512 depth input, then the final image is resized to 512x512.
- Qwen now runs an optional second refinement stage from the first completed
  semantic/RGB image. The second stage keeps only object outline, size,
  category, orientation, pose, and camera viewpoint, while making the object
  and background a more realistic scene.
- Default projection is the original `view_select` path, which selects the
  camera with the most visible partial points. The semantic view candidate
  preview/selection experiment has been removed from the default pipeline.

Redwood Stage 1 preview generated with this default projection:
- Output root:
  `workspace/redwood_stage1_qwen_single16_view_select_preview`
- Samples: `01184`, `05117`, `05452`
- Per-sample outputs: `depth.png`, `img.png`, `qwen_edit_prompt.txt`,
  `point_uv.npy`, `camera.pth`, `viewpoint.npy`
- Contact sheet:
  `workspace/redwood_stage1_qwen_single16_view_select_preview/redwood_stage1_depth_vs_qwen_preview.png`
- Status: generated and verified as 512x512 outputs; awaiting user visual
  acceptance.
- Update: `05117` prompt label override changed from `chair` to `red chair`
  through `configs/config.yaml::prompt_overrides`, then `05117` Stage 1 was
  regenerated in the same output root.
- `depth_view_point_cloud.ply` is not needed by the current index bridge and
  should not be saved by default; `save_depth_view_point_cloud` is now `false`.
- Partial-to-MoGe registration was run for `01184`, `05117`, and `05452` using
  raw partial point clouds from `data/<sample>.ply`, MoGeV2 on `img.png`, and
  RMBG-2.0 object masks. Summary:
  `workspace/redwood_stage1_qwen_single16_view_select_preview/redwood_moge_to_raw_partial_summary.csv`
- MoGe object extraction erodes the RMBG object mask by default
  (`object_mask_erode_pixels=2`) before filtering MoGe points. This removes
  edge/background points such as the extra environment points seen in `01184`
  while keeping the partial-to-MoGe registration stable.
- Two-stage Qwen refinement preview was generated for all default Redwood
  samples in:
  `workspace/redwood_stage1_qwen_refine_preview`
- The full contact sheet is:
  `workspace/redwood_stage1_qwen_refine_preview/redwood_qwen_refine_all_preview.png`
- Each sample keeps `qwen_edit_stage1.png` at 1024x1024 and final `img.png` at
  512x512. The `06127` category override experiment was reverted; it uses the
  dataset label `a vase with leafy plant`.

## Stage 2 - Completed Image to Complete 3D Point Cloud

Status: implemented experimentally, needs integration.

Flow:
1. Take the Qwen-completed semantic/RGB image.
2. Remove background before Hunyuan generation.
3. Use Hunyuan3D-2.1 to generate a 3D model.
4. Sample the generated model into a complete point cloud.

Important lessons:
- If Hunyuan sees the background, it can generate extra floor/background
  geometry.
- Use RMBG/alpha image for Hunyuan input.
- Keep the original image with background for MoGe/FreeReg experiments when
  needed; keep the RMBG image for Hunyuan.

Known `car__132` files:
- Good Qwen baseline:
  `workspace/scansalon_zup_side_512/car__132/qwen_edit_2511_car_completion_from_depth.png`
- RMBG input for Hunyuan:
  `workspace/scansalon_zup_side_512/car__132/qwen_edit_2511_car_completion_from_depth_rmbg.png`
- Hunyuan complete point cloud:
  `workspace/scansalon_zup_side_512/car__132/car__132_hunyuan2.1.ply`

Open work:
- Make Qwen completion reproducible.
- Formalize when to keep background and when to remove it.
- Integrate Hunyuan generation and point sampling as a pipeline stage.

Redwood `01184` experiment:
- Input image:
  `workspace/redwood_stage1_qwen_refine_preview/01184/img.png`
- RMBG image for Hunyuan:
  `workspace/redwood_stage1_qwen_refine_preview/01184/img_sam.png`
- Hunyuan complete point cloud:
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_hunyuan2.1.ply`
- Status: generated and verified as 100000 points.
- Decision: Hunyuan3D-2.1 should only keep the final sampled `.ply` in the
  main workflow. Intermediate `_shape.glb` and final `.glb` files are not
  retained by default.
- Output cleanup decision: default runs now use `outputs.keep_profile: lean`
  with `outputs.save_intermediates: false`. The lean profile keeps core Stage 1
  files, `img_sam.png`, final Hunyuan PLY, MoGe object/index/transform metadata,
  masked FreeReg outputs, and final fused/inspection PLYs. Debug previews,
  unmasked FreeReg outputs, raw MoGe projection arrays, hits-only PLYs, stage
  Qwen images, and GLB files are removed unless `outputs.save_intermediates:
  true` or `outputs.keep_profile: debug` is set.

## Stage 3 - MoGe to Complete Registration

Status: integrated into the main pipeline with no-FreeReg render-to-MoGe Sim3.
The fixed-uv/Sim3 F-FreeReg path remains available as a historical experiment,
but `configs/config.yaml` now defaults to `reg_backend: render_to_moge_sim3`.

Intended FreeReg input:
- Image: the same completed image used by MoGe, with background.
- Point cloud: Hunyuan complete point cloud.

Intended output:
- A transform that aligns the complete point cloud to the MoGe/image coordinate
  frame.
- A fused visualization of MoGe point cloud plus registered complete point
  cloud.

Important clarification:
- Original F-FreeReg is image + point cloud.
- Do not replace original F-FreeReg image depth with MoGe inside FreeReg unless
  this is explicitly a separate experiment.

Current default implementation:
- `ScaleAdapter.render_to_moge_sim3_reg` runs the MoGe-to-raw-partial bridge
  and then `scripts/run_render_to_moge_sim3.py`.
- The complete point cloud is aligned directly to MoGe by Sim3 candidate
  search, z-buffer depth/silhouette scoring, coordinate-search refinement, and
  visible trimmed ICP.
- FreeReg and DepthPro are not used by the default main pipeline.

Open work:
- Inspect the no-FreeReg fused visualizations and overlays for high-metric
  samples, especially `06127`, `09639`, `06188`, `07306`, and `07136`.
- Tune render-to-MoGe scoring or per-category candidate parameters if visual
  inspection shows systematic orientation or scale failures.
- Keep the FreeReg adaptive path as a comparison baseline, not the default.
- 2026-07-13 semantic-feature probe: DINOv2-large image patch features can now
  be precomputed and used to re-rank FreeReg image/point-cloud matches for
  visualization without changing the selected transform. This is still an
  inspection experiment, not a main-flow registration criterion.

Redwood `01184` original F-FreeReg experiment:
- Image input:
  `workspace/redwood_stage1_qwen_refine_preview/01184/img.png`
- Complete point cloud input:
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_hunyuan2.1.ply`
- DepthPro image point cloud:
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_freereg_original_depthpro_image_points.ply`
- Complete registered to DepthPro image frame:
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_freereg_original_depthpro_complete_registered_to_image.ply`
- Fused visualization:
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_freereg_original_depthpro_gray_image_blue_complete_fused.ply`
- Metadata:
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_freereg_original_depthpro_info.json`
- Status: generated; original F-FreeReg produced 112 YOHO descriptor matches.
- MoGe object extraction for the same image was also generated with RMBG erode
  2 pixels:
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_moge_to_raw_partial_moge_object_only.ply`
- Issue found: the unmasked DepthPro image point cloud includes ground/background
  points because original F-FreeReg backprojects the whole image.
- Masked rerun: `scripts/run_freereg_original_depthpro.py` now supports an
  RMBG object mask so DepthPro backprojection keeps only object pixels.
- Masked `01184` outputs:
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_freereg_original_depthpro_objectmask_object_depthpro_points.ply`
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_freereg_original_depthpro_objectmask_complete_registered_to_object_depthpro.ply`
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_freereg_original_depthpro_objectmask_gray_object_depthpro_blue_complete_fused.ply`
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_freereg_original_depthpro_objectmask_info.json`
- Masked rerun status: generated; object DepthPro target has 49578 points and
  original F-FreeReg produced 396 YOHO descriptor matches.
- The current `01184` experiment directory was cleaned with the lean profile.
  Retained files are:
  `depth.png`, `img.png`, `camera.pth`, `point_uv.npy`,
  `qwen_edit_prompt.txt`, `img_sam.png`, `01184_hunyuan2.1.ply`,
  `01184_moge_to_raw_partial_moge_object_only.ply`,
  `01184_moge_to_raw_partial_partial_to_moge_index.npy`,
  `01184_moge_to_raw_partial_moge_to_raw_partial_transform.npy`,
  `01184_moge_to_raw_partial_info.json`,
  `01184_moge_to_raw_partial_object_mask.png`,
  `01184_moge_to_raw_partial_raw_partial_gray_moge_red_aligned.ply`,
  `01184_freereg_original_depthpro_objectmask_complete_registered_to_object_depthpro.ply`,
  `01184_freereg_original_depthpro_objectmask_gray_object_depthpro_blue_complete_fused.ply`,
  and `01184_freereg_original_depthpro_objectmask_info.json`.
- `01184` fixed-uv/Sim3 FreeReg probe:
  `scripts/run_freereg_original_depthpro.py` was updated to project YOHO image
  keypoints to uv directly and to apply FreeReg's estimated scale as a Sim3
  transform. This reduced the complete-vs-DepthPro bbox-center z offset from
  about `0.426` to `0.018` and reduced complete-to-DepthPro nearest-neighbor
  mean distance from about `0.393` to `0.176`.
- Fixed-uv/Sim3 outputs:
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_freereg_original_depthpro_fixeduv_sim3_gray_object_depthpro_blue_complete_fused.ply`
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_freereg_original_depthpro_fixeduv_sim3_complete_registered_to_object_depthpro.ply`
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_freereg_original_depthpro_fixeduv_sim3_info.json`
- A bbox-center translation-only probe was also generated:
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_freereg_original_depthpro_fixeduv_sim3_bboxshift_gray_object_depthpro_blue_complete_fused.ply`
  It aligns the bounding-box centers but does not materially improve
  nearest-neighbor distance, so it should remain a visualization/debug probe
  rather than a final registration step.
- FreeReg source used by `scripts/run_freereg_original_depthpro.py` is now
  vendored under `third_party/FreeReg`. The vendored code contains source only;
  checkpoints are intentionally excluded and can be provided through
  `FREEREG_DEPTHPRO_CKPT`, `FREEREG_FCGF_CKPT`, and `FREEREG_YOHO_CKPT`.
- Adaptive `ir_3d` rerun on the Redwood batch:
  - `01184`, `05117`, `05452`, `06127`, `07306`, and `09639` selected the
    original auto threshold.
  - `06145`, `06188`, `06830`, and `07136` selected `fallback_0.1`.
  - Sampled CPU Chamfer x1e2 mean improved from about `19899.46` to `14.00`.
  - Summary:
    `workspace/redwood_stage1_qwen_refine_preview/freereg_adaptive_ir3d_cpu_cd_summary.csv`
- `06145` DINOv2 semantic-match visualization probe:
  - Feature extractor:
    `scripts/extract_dinov2_image_features.py`
  - FreeReg wrapper:
    `scripts/run_freereg_original_depthpro.py`
  - DINOv2-large image feature grid:
    `workspace/redwood_stage1_qwen_refine_preview/06145/06145_dinov2_large_img_features.npz`
  - Original match-line figure:
    `workspace/redwood_stage1_qwen_refine_preview/06145/06145_freereg_adaptive_ir3d_image_pointcloud_match_lines.png`
  - DINO semantic-filtered match-line figure, `min_similarity=0.55`, 8 valid
    matches:
    `workspace/redwood_stage1_qwen_refine_preview/06145/06145_freereg_adaptive_ir3d_dino_semantic_match_lines.png`
  - DINO semantic-filtered match-line figure, `min_similarity=0.35`, 12 valid
    matches:
    `workspace/redwood_stage1_qwen_refine_preview/06145/06145_freereg_adaptive_ir3d_dino035_dino_semantic_match_lines.png`
  - Combined comparison:
    `workspace/redwood_stage1_qwen_refine_preview/06145/06145_freereg_dino_match_comparison.png`
  - Risk/decision: DINO currently re-ranks matches after the selected transform
    and is only a diagnostic visualization. It should not become a registration
    criterion until it is tested as an inlier/candidate scoring term across
    multiple object categories.
- `06145` projected-silhouette candidate-selection probe:
  - FreeReg wrapper:
    `scripts/run_freereg_original_depthpro.py`
  - Candidate mode:
    `--candidate-selection silhouette`
  - Candidate thresholds:
    `auto, 0.08, 0.1, 0.15, 0.2, 0.3`
  - Selected candidate:
    `fallback_0.08`
  - Selected candidate silhouette score:
    `score=0.4198`, `IoU=0.3766`, `coverage=0.7759`,
    `leakage=0.5774`, `edge_chamfer_px=31.01`
  - FreeReg fused visualization:
    `workspace/redwood_stage1_qwen_refine_preview/06145/06145_freereg_silhouette_select_gray_object_depthpro_blue_complete_fused.ply`
  - Complete-to-partial fused visualization:
    `workspace/redwood_stage1_qwen_refine_preview/06145/06145_complete_to_partial_silhouette_select_raw_partial_gray_complete_blue_aligned.ply`
  - Silhouette overlay:
    `workspace/redwood_stage1_qwen_refine_preview/06145/06145_freereg_silhouette_select_silhouette_overlay.png`
  - Adaptive-vs-silhouette overlay comparison:
    `workspace/redwood_stage1_qwen_refine_preview/06145/06145_adaptive_vs_silhouette_overlay_comparison.png`
  - Metric summary:
    `workspace/redwood_stage1_qwen_refine_preview/06145/06145_silhouette_select_metric_summary.json`
  - Same-seed metric comparison against `data/GT/06145.ply`, sampled with
    `metric_num_points=16384`, `metric_seed=1184`:
    adaptive `CD-L1 x1e2=9.3714`, `EMD x1e2=9.0747`; silhouette-select
    `CD-L1 x1e2=5.2366`, `EMD x1e2=6.5815`.
  - Risk/decision: silhouette selection improved `06145`, but leakage remains
    high. Test across more categories before promoting it from probe to default
    candidate selection.

## Stage 4 - Complete Back to Partial

Status: integrated into `main.py` through `ScaleAdapter` when
`reg_backend: render_to_moge_sim3`.

Flow:
1. Use Stage 1 index bridge:
   `partial point index -> MoGe point index`.
2. Use Stage 3 registration:
   `complete point cloud -> MoGe coordinate frame`.
3. Compose transforms/index relations to derive:
   `partial point index -> complete point index`
   or
   `complete point cloud aligned to raw partial`.

Final output:
- Complete point cloud registered to the raw partial point cloud.
- Visualizations showing raw partial plus aligned complete.
- Index mapping from partial points to complete points where possible.

Open work:
- Add broader tests around the `ScaleAdapter.render_to_moge_sim3_reg` adapter
  once heavy MoGe calls can be mocked cleanly.
- Inspect and tune the high-metric samples from the 2026-07-13 no-FreeReg run.

Redwood `01184` composed output:
- Complete aligned to raw partial:
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_complete_to_partial_depthpro_moge_complete_aligned_to_raw_partial.ply`
- Fused visualization:
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_complete_to_partial_depthpro_moge_raw_partial_gray_complete_blue_aligned.ply`
- Transform:
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_complete_to_partial_depthpro_moge_complete_to_raw_partial_transform.npy`
- Metadata:
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_complete_to_partial_depthpro_moge_info.json`
- Metric against `data/GT/01184.ply`, sampled with `metric_num_points=16384`
  and `metric_seed=1184`: `CD-L1 x1e2 = 1.745359`, `EMD x1e2 =
  2.428691`.
- Metric reporting now uses `x1e2` columns and printed labels in `main.py`.

Redwood batch run on 2026-07-13:
- Samples:
  `01184`, `05117`, `05452`, `06127`, `06145`, `06188`, `06830`, `07136`,
  `07306`, `09639`.
- Registration summary:
  `workspace/redwood_stage1_qwen_refine_preview/redwood_complete_to_partial_registration_summary.csv`
- Metric summary:
  `workspace/redwood_stage1_qwen_refine_preview/redwood_complete_to_partial_metrics.csv`
- Normal-scale outputs:
  `01184`, `05117`, `05452`, `07306`, `09639`.
- Suspicious or failed outputs:
  `06127` has low DepthPro-to-MoGe inlier ratio and higher metric;
  `06145`, `06188`, `06830`, and `07136` have large complete-to-partial
  translations caused by unstable FreeReg results.
- Adaptive FreeReg update:
  `06145`, `06188`, `06830`, and `07136` no longer have random hundreds-scale
  complete-to-partial translations. The adaptive outputs use prefix
  `<sample>_complete_to_partial_adaptive_ir3d`. Sampled CPU Chamfer x1e2:
  `06145=7.005`, `06188=10.698`, `06830=15.149`, `07136=18.590`.
  The full comparison is:
  `workspace/redwood_stage1_qwen_refine_preview/freereg_adaptive_ir3d_cpu_cd_summary.csv`.
- Core method documentation:
  `docs/core_registration_pipeline.md`.

Redwood no-FreeReg main-pipeline run on 2026-07-13:
- Command:
  `CUDA_VISIBLE_DEVICES=0 /opt/data/private/cr/miniconda3/envs/genpc/bin/python main.py --workspace workspace/redwood_stage1_qwen_refine_preview --skip_existing --save_intermediates`
- Default backend:
  `render_to_moge_sim3`
- Samples:
  `01184`, `05117`, `05452`, `06127`, `06145`, `06188`, `06830`, `07136`,
  `07306`, `09639`.
- Per-sample outputs:
  `<sample>_complete_registered_to_moge.ply`,
  `<sample>_moge_gray_complete_blue_fused.ply`,
  `<sample>_complete_aligned_to_raw_partial.ply`,
  `<sample>_raw_partial_gray_complete_blue_aligned.ply`,
  `<sample>_complete_to_moge_transform.npy`,
  `<sample>_complete_to_partial_transform.npy`,
  `<sample>_render_to_moge_overlay.png`,
  `<sample>_render_to_moge_sim3_info.json`,
  and metric prediction `<sample>_fused.ply`.
- Metric summary:
  `workspace/redwood_stage1_qwen_refine_preview/metrics_samples.csv`
- Mean metrics:
  `CD-L1 x1e2 = 4.198972`, `EMD x1e2 = 4.746163`.
- Per-sample `CD-L1 x1e2 / EMD x1e2`:
  `01184=2.476381/2.828803`,
  `05117=4.696598/5.814887`,
  `05452=1.503627/1.928873`,
  `06127=8.357799/8.122091`,
  `06145=1.472761/2.026613`,
  `06188=4.973479/6.414759`,
  `06830=2.431914/3.379773`,
  `07136=4.854884/4.702779`,
  `07306=4.865654/5.496590`,
  `09639=6.356620/6.746463`.

## Current Risks

- The best Qwen output for `car__132` was not fully recorded when it was
  generated. See `PROJECT_STATE.md`.
- Prompt, seed, resolution, and scheduler details can materially change Qwen
  quality.
- FreeReg results can look valid in one coordinate frame while being wrong in
  another; always save explicit fused visualizations and transform metadata.
- Coordinate flips and depth-view/raw-partial frame differences are easy to
  reintroduce. Keep y-flip tests active.

## Update Rules

- When a stage is completed, update its `Status` and add concrete output paths.
- When a user accepts an output, also update `PROJECT_STATE.md`.
- When a stage changes direction, add a short note under that stage instead of
  deleting prior context.
