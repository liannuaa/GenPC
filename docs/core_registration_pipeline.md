# Core Registration Pipeline

This document records the current core GenPC method for registering a generated
complete point cloud back to the original partial point cloud.

This is an academic-paper implementation. The core method should stay simple,
explainable, reproducible, and easy to ablate. Prefer concise geometric or
continuous-optimization changes over extra model stacks, broad engineering
frameworks, or sample-specific rules.

## Goal

Given a partial point cloud, reconstruct a complete 3D point cloud and express
that complete cloud in the raw partial point cloud coordinate frame.

The core relation is:

```text
partial raw points
  -> camera-1 depth image
  -> Qwen completed RGB image
  -> MoGe image point cloud
  -> Hunyuan complete point cloud
  -> render-to-MoGe Sim3
  -> complete point cloud aligned to raw partial
```

The method intentionally separates image-space correspondence from 3D
registration. Pixel correspondence is used to align MoGe to the raw partial
scan. The generated complete point cloud is now aligned directly to MoGe with a
no-FreeReg Sim3 search over rendered depth, silhouette overlap, and visible ICP.

## Stage 1: Partial To Image

`DepthPrompting` projects the raw partial point cloud to `raw_depth.png` and
`depth.png`, then saves the projection state:

- `camera.pth`
- `point_uv.npy`
- `raw_depth.png`
- `depth.png`
- `qwen_edit_stage1.png` after one-stage Qwen completion
- `img.png`, copied from the same one-stage Qwen output

The accepted Qwen image-generation flow is now intentionally simple and
one-stage. It uses `raw_depth.png` as the input image and asks Qwen to generate
a complete object image from the occluded depth cue:

```text
生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的{photo_label}，纯白背景
```

The output is resized to `512x512` and saved as both `qwen_edit_stage1.png` and
`img.png`. The second Qwen refinement stage was removed from the main pipeline
after the one-stage raw-depth results were accepted visually; stale
`qwen_edit_stage2_prompt.txt` files are removed during Stage 1 output writing.
Keep this path simple and ablatable unless a future experiment explicitly
reintroduces a second image stage.

The Qwen input depth image is selected by `qwen_edit_depth_input_name`; the
current accepted setting is `raw_depth.png`, not the inpainted/flipped
`depth.png`.

Important coordinate detail:

- `DepthPrompting.paintPixels()` vertically flips the saved `depth.png`.
- Any projection uv used against `img.png` or MoGe pixels must use the saved
  image coordinate convention: `v = 1 - v`.
- Do not use `depth_view_point_cloud.ply` as the raw partial target. It is in
  depth-view coordinates, not the raw partial coordinate frame.

## Stage 2: Image To MoGe And Partial Bridge

Run MoGe on the completed image `img.png`. Remove background with RMBG-2.0 and
erode the object mask before keeping MoGe object points.

Implementation:

- `scripts/run_moge_to_raw_partial_from_camera.py`

Outputs:

- `<sample>_moge_to_raw_partial_moge_object_only.ply`
- `<sample>_moge_to_raw_partial_partial_to_moge_index.npy`
- `<sample>_moge_to_raw_partial_moge_to_raw_partial_transform.npy`
- `<sample>_moge_to_raw_partial_raw_partial_gray_moge_red_aligned.ply`
- `<sample>_moge_to_raw_partial_info.json`
- `<sample>_moge_to_raw_partial_object_mask.png`

This stage estimates `moge_to_raw_partial` using pixel correspondences between
reprojected raw partial points and MoGe object pixels. Good runs usually have
high partial-to-MoGe match ratio and high RANSAC inlier ratio.

## Stage 3: Image To Complete Point Cloud

Before Hunyuan3D generation, remove image background so the generated geometry
does not include floor or scene background.

Outputs:

- `img_sam.png`
- `<sample>_hunyuan2.1.ply`

The current main workflow keeps the final sampled `.ply`; GLB intermediates are
not required for the registration method.

## Stage 4: No-FreeReg Complete-To-MoGe Sim3 Registration

Use render-to-MoGe Sim3 registration:

- image/MoGe input: the same `img.png` used by Stage 2
- MoGe object mask: `<sample>_moge_to_raw_partial_object_mask.png`
- complete point cloud input: `<sample>_hunyuan2.1.ply`
- object mask: `<sample>_moge_to_raw_partial_object_mask.png`

Implementation:

- `scripts/run_render_to_moge_sim3.py`
- main pipeline integration: `ScaleAdapter.render_to_moge_sim3_reg`

The current default variant does not use FreeReg or DepthPro:

- Run MoGe on `img.png` and keep object-mask pixels.
- Build candidate Sim3 transforms from axis-aligned rotations and scale
  multipliers.
- Score candidates by z-buffer rendering the complete point cloud into the MoGe
  camera and comparing silhouette overlap, edge alignment, leakage, and depth
  agreement. The score weights 2D silhouette/edge terms more heavily than
  depth.
- Refine the best candidate with coordinate search over translation, rotation,
  and scale.
- Run a differentiable 2D silhouette optimization from the coordinate-search
  transform. It optimizes a small delta-Sim3 with soft point splatting against
  the object mask, using silhouette Dice, leakage, missing-mask, distance,
  area, center, and transform-regularization terms. The candidate is accepted
  only if the full-resolution hard render score improves.
- Run a differentiable visible-3D refinement from the current transform. It
  builds nearest-neighbor correspondences between the complete cloud's visible
  rendered surface and MoGe object points, optimizes a small delta-Sim3 with a
  3D distance term plus a soft 2D silhouette guard, and accepts the result only
  if the visible 3D distance improves while the full-resolution 2D render score
  stays within the configured drop tolerance.
- Optionally run visible trimmed ICP from rendered complete points to MoGe
  object points. The ICP result is accepted only if it preserves or improves
  the render-to-MoGe 2D score; otherwise the pipeline rolls back to the
  coordinate-search transform.

Outputs:

- `<sample>_complete_registered_to_moge.ply`
- `<sample>_moge_gray_complete_blue_fused.ply`
- `<sample>_complete_to_moge_transform.npy`
- `<sample>_render_to_moge_overlay.png`
- `<sample>_render_to_moge_sim3_info.json`

The resulting transform is `complete_to_moge`.

## Stage 5: Compose Complete To Partial

Before any final partial-space refinement runs, the MoGe-frame 2D render
alignment must pass a hard acceptance gate. By default
`registration_2d_acceptance` requires:

- `final_score.iou >= 0.82`
- `final_score.coverage >= 0.84`
- `final_score.leakage <= 0.10`
- `final_score.edge_iou >= 0.025`
- `final_score.edge_chamfer_px <= 18.0`

The edge terms are hard gates. A high coarse mask overlap is not enough if the
rendered boundary is visibly off. If the 2D gate fails, the pipeline retries a
top-K set of candidates ranked by a 2D-focused objective with wider scale
multipliers. When the
`partial_to_moge_index` bridge exists, retry candidates also record a
MoGe-frame anchor distance to raw partial points and use a joint 2D+anchor
objective for candidate selection.

After retry, the pipeline can run a continuous bridge-anchor delta-Sim3
optimization in the MoGe frame. This uses raw partial points mapped back
through `moge_to_partial^-1` as bridge anchors, optimizes a trimmed anchor
distance plus a soft 2D silhouette guard, and accepts the delta only when
anchor distance improves without dropping the 2D objective beyond the
configured tolerance. If the best result still fails the 2D gate, partial-space
refinement is skipped so a poor image-frame alignment cannot be hidden by a
later partial-distance improvement.

The final transform is now:

```text
complete_to_partial =
    moge_to_partial
    @ complete_to_moge
```

Because the MoGe-to-raw-partial bridge can still have a small residual offset,
the final `complete_to_partial` transform may receive one more conservative
partial-space refinement. This step optimizes a small delta-Sim3 from complete
points to the raw partial point cloud using trimmed nearest-neighbor 3D
distances. It updates only `complete_to_partial` and final fused/metric outputs;
it does not change `complete_to_moge` or the MoGe-frame inspection outputs. The
candidate is accepted only when the partial-space distance improves and the
delta scale, rotation, and translation stay inside configured bounds.

Final outputs:

- `<sample>_complete_aligned_to_raw_partial.ply`
- `<sample>_raw_partial_gray_complete_blue_aligned.ply`
- `<sample>_complete_to_partial_transform.npy`
- `<sample>_fused.ply`

The fused visualization colors the raw partial point cloud gray and the aligned
complete point cloud blue. `<sample>_fused.ply` is the metric prediction path.

## Failure Signals

Do not trust a fused result only because files exist. Check the metadata:

- Low `final_score.iou`, low `final_score.coverage`, or high
  `final_score.leakage` in `<sample>_render_to_moge_sim3_info.json` means the
  complete-to-MoGe render alignment is weak.
- `registration_2d_acceptance.accepted = false` means the render alignment did
  not pass the configured hard 2D gate, so the output should not be treated as a
  good registration even if fused PLY files exist.
- Low `final_score.edge_iou` or high `final_score.edge_chamfer_norm` means the
  2D silhouette boundary is misaligned even if coarse mask overlap is nonzero.
- `silhouette_optimization.accepted = false` means the soft differentiable
  optimization ran but did not improve the full-resolution hard render score.
- `visible_3d_optimization.accepted = false` means the visible 3D refinement
  either could not find stable correspondences, did not reduce visible 3D
  distance enough, or would have hurt the 2D render score too much.
- `bridge_anchor_optimization.accepted = false` means the continuous
  partial-to-MoGe anchor refinement either did not improve anchor distance or
  would have hurt the 2D render objective too much.
- `partial_refinement.accepted = false` means the final complete-to-partial
  correction either did not improve trimmed partial distance enough or proposed
  a delta that exceeded the configured small-motion limits.
- Large visible ICP mean/p95 distances are suspicious even if final files
  exist.
- Very large `complete_to_partial` translation norm is usually a failed
  registration result.
- huge metric values, especially `CD-L1 x1e2` in the thousands, indicate a
  transform-scale or translation failure.

For the original 2026-07-13 Redwood batch, normal-scale samples were:

- `01184`
- `05117`
- `05452`
- `07306`
- `09639`

Problematic or suspicious samples were:

- `06127`: low DepthPro-to-MoGe inlier ratio and higher metric.
- `06145`, `06188`, `06830`, `07136`: large complete-to-partial translation,
  caused by unstable FreeReg results.

The adaptive FreeReg rerun fixed the random-transform failures:

- `01184`, `05117`, `05452`, `06127`, `07306`, and `09639` selected `auto`.
- `06145`, `06188`, `06830`, and `07136` selected `fallback_0.1`.
- Sampled CPU Chamfer x1e2 mean improved from about `19899.46` to `14.00`.
- `06188` improved from about `35297.93` to `10.70` sampled CPU Chamfer x1e2;
  its DepthPro-to-MoGe inlier ratio is still low, so it remains a useful bridge
  quality stress case even though the random FreeReg failure mode is fixed.

Batch summaries:

- `workspace/redwood_stage1_qwen_refine_preview/redwood_complete_to_partial_registration_summary.csv`
- `workspace/redwood_stage1_qwen_refine_preview/redwood_complete_to_partial_metrics.csv`
- `workspace/redwood_stage1_qwen_refine_preview/freereg_adaptive_ir3d_cpu_cd_summary.csv`

The no-FreeReg render-to-MoGe Sim3 main-pipeline run on 2026-07-13 wrote
`<sample>_fused.ply` for all default Redwood samples and saved metrics to:

- `workspace/redwood_stage1_qwen_refine_preview/metrics_samples.csv`
- `workspace/redwood_stage1_qwen_refine_preview/metrics_by_category.csv`

Its mean metrics were `CD-L1 x1e2 = 4.198972` and `EMD x1e2 = 4.746163`.
Higher-error samples were `06127`, `09639`, `06188`, `07306`, and `07136`.

After visual inspection showed weak overlays, the render-to-MoGe score was
changed to emphasize silhouette/edge alignment and to reject visible ICP when
it lowers the 2D render score. A rerun rejected visible ICP on all 10 Redwood
samples. The updated overlays are more conservative with respect to the 2D
score, but the mean metric worsened to `CD-L1 x1e2 = 6.427336` and
`EMD x1e2 = 7.728549`; this is an overlay-protection setting, not yet a better
3D metric setting.

## Runtime Notes

- Use `/opt/data/private/cr/miniconda3/envs/genpc/bin/python` for GenPC, MoGe,
  Hunyuan, Qwen, RMBG, compose, and metric code.
- Use `/opt/data/private/cr/miniconda3/envs/freereg/bin/python` only for
  historical FreeReg experiments, because FreeReg requires MinkowskiEngine. The
  default main pipeline no longer needs FreeReg.
- `third_party/FreeReg` contains source only. Checkpoints are intentionally not
  committed. By default the vendored FreeReg wrapper falls back to the existing
  checkpoint paths under `/opt/data/private/cr/lab/FreeReg`. These can be
  overridden with:
  - `FREEREG_DEPTHPRO_CKPT`
  - `FREEREG_FCGF_CKPT`
  - `FREEREG_YOHO_CKPT`
