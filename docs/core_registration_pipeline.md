# Core Registration Pipeline

This document records the current core GenPC method for registering a generated
complete point cloud back to the original partial point cloud.

## Goal

Given a partial point cloud, reconstruct a complete 3D point cloud and express
that complete cloud in the raw partial point cloud coordinate frame.

The core relation is:

```text
partial raw points
  -> camera-1 depth image
  -> Qwen completed RGB/semantic image
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

`DepthPrompting` projects the raw partial point cloud to `depth.png` and saves
the projection state:

- `camera.pth`
- `point_uv.npy`
- `depth.png`
- `img.png` after Qwen completion

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

The final transform is now:

```text
complete_to_partial =
    moge_to_partial
    @ complete_to_moge
```

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
- Low `final_score.edge_iou` or high `final_score.edge_chamfer_norm` means the
  2D silhouette boundary is misaligned even if coarse mask overlap is nonzero.
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
