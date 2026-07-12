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
  -> DepthPro/FreeReg image frame
  -> Hunyuan complete point cloud
  -> complete point cloud aligned to raw partial
```

The method intentionally separates image-space correspondence from 3D
registration. Pixel correspondence is used where images share the same rendered
view; FreeReg is used where the image must be aligned to the generated complete
point cloud.

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

## Stage 4: FreeReg Image-Complete Registration

Use original F-FreeReg as image + point-cloud registration:

- image input: the same `img.png` used by MoGe
- point cloud input: `<sample>_hunyuan2.1.ply`
- object mask: `<sample>_moge_to_raw_partial_object_mask.png`

Implementation:

- `scripts/run_freereg_original_depthpro.py`
- vendored FreeReg source: `third_party/FreeReg`

The current FreeReg variant is fixed-uv + Sim3:

- DepthPro backprojects only object-mask pixels, not the full image.
- YOHO image keypoint uv coordinates are projected directly from image
  keypoints.
- FreeReg's estimated scale is applied in the saved Sim3 transform.

Outputs:

- `<sample>_freereg_original_depthpro_fixeduv_sim3_object_depthpro_points.ply`
- `<sample>_freereg_original_depthpro_fixeduv_sim3_complete_registered_to_object_depthpro.ply`
- `<sample>_freereg_original_depthpro_fixeduv_sim3_gray_object_depthpro_blue_complete_fused.ply`
- `<sample>_freereg_original_depthpro_fixeduv_sim3_info.json`

The resulting transform is `complete_to_depthpro`.

## Stage 5: DepthPro To MoGe Same-Pixel Bridge

DepthPro and MoGe run on the same `img.png`, so no feature matching is needed
between them. However, they still produce different 3D coordinate frames and
depth scales. The bridge uses same-pixel 3D correspondences:

1. Project object DepthPro points back to image pixels with FreeReg's intrinsic.
2. Match those pixels to MoGe object pixels from the same image.
3. Estimate a robust Sim3 transform `depthpro_to_moge`.

Implementation:

- `scripts/compose_complete_to_partial_via_depthpro_moge.py`

## Stage 6: Compose Complete To Partial

The final transform is:

```text
complete_to_partial =
    moge_to_partial
    @ depthpro_to_moge
    @ complete_to_depthpro
```

Final outputs:

- `<sample>_complete_to_partial_depthpro_moge_complete_aligned_to_raw_partial.ply`
- `<sample>_complete_to_partial_depthpro_moge_raw_partial_gray_complete_blue_aligned.ply`
- `<sample>_complete_to_partial_depthpro_moge_complete_to_raw_partial_transform.npy`
- `<sample>_complete_to_partial_depthpro_moge_info.json`

The fused visualization colors the raw partial point cloud gray and the aligned
complete point cloud blue.

## Failure Signals

Do not trust a fused result only because files exist. Check the metadata:

- `freereg_matches < 100` is weak.
- `depthpro_to_moge_ransac.inlier_ratio < 0.8` is weak.
- very large `complete_to_partial` translation norm is usually a failed FreeReg
  result.
- huge metric values, especially `CD-L1 x1e2` in the thousands, indicate a
  transform-scale or translation failure.

For the 2026-07-13 Redwood batch, normal-scale samples were:

- `01184`
- `05117`
- `05452`
- `07306`
- `09639`

Problematic or suspicious samples were:

- `06127`: low DepthPro-to-MoGe inlier ratio and higher metric.
- `06145`, `06188`, `06830`, `07136`: large complete-to-partial translation,
  caused by unstable FreeReg results.

Batch summaries:

- `workspace/redwood_stage1_qwen_refine_preview/redwood_complete_to_partial_registration_summary.csv`
- `workspace/redwood_stage1_qwen_refine_preview/redwood_complete_to_partial_metrics.csv`

## Runtime Notes

- Use `/opt/data/private/cr/miniconda3/envs/genpc/bin/python` for GenPC, MoGe,
  Hunyuan, Qwen, RMBG, compose, and metric code.
- Use `/opt/data/private/cr/miniconda3/envs/freereg/bin/python` for FreeReg
  because it requires MinkowskiEngine.
- `third_party/FreeReg` contains source only. Checkpoints are intentionally not
  committed. By default the vendored FreeReg wrapper falls back to the existing
  checkpoint paths under `/opt/data/private/cr/lab/FreeReg`. These can be
  overridden with:
  - `FREEREG_DEPTHPRO_CKPT`
  - `FREEREG_FCGF_CKPT`
  - `FREEREG_YOHO_CKPT`
