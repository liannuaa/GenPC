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

## Stage 3 - MoGe to Complete Registration

Status: experimental and currently unstable with original F-FreeReg.

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

Current issue:
- Original F-FreeReg with DepthPro produces too few matches and unstable
  transforms on `car__132`.
- A likely contributor is mismatched image keypoint uv handling in the original
  demo code, plus weak geometric correspondence between DepthPro/MoGe image
  point clouds and Hunyuan complete geometry.

Open work:
- Decide whether to use original F-FreeReg as-is, a fixed-uv FreeReg variant, or
  FreeReg only as one initialization candidate.
- Add robust validation visualizations for image-point correspondences.
- If needed, add fallback refinement such as similarity ICP or projection
  silhouette consistency, but keep that separate from the original FreeReg
  experiment.

## Stage 4 - Complete Back to Partial

Status: design target, not fully integrated.

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
- Implement composition cleanly inside the main pipeline.
- Define saved output names and cleanup behavior.
- Add tests for transform direction and coordinate-frame composition.

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
