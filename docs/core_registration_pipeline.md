# Core Registration Pipeline

## Canonical Method

This is the canonical end-to-end method for the current GenPC project.  It
replaces the old Qwen→external-MoGe→Hunyuan core description.

The active route is:

```text
raw partial point cloud
  -> GenPC saved-camera depth projection
  -> GPT ImageGen complete semantic/RGB image
  -> Pixal3D complete GLB and deterministic 100k surface PLY
  -> saved-camera 2D/depth + visible-partial SO(3)/Sim(3) TTT
  -> registered full Pixal3D model
  -> completeness-preserving observation fusion
  -> frozen prediction
  -> post-freeze GT CD-L1/EMD
```

The method remains recognizably descended from GenPC: it keeps GenPC's raw
partial projection, saved camera, and 2D+3D registration logic; upgrades the
image prior to GPT ImageGen; upgrades the complete-shape prior to Pixal3D; and
turns registration/fusion into visibility-aware test-time posterior fitting.

The paper target is zero-shot Redwood completion with one shared configuration,
no sample-specific tuning, and full-ten mean CD-L1 x1e2 below `1.74` and EMD
x1e2 below `2.88`.

## Current Freeze and Approval State

- Accepted/frozen semantic images:
  `gpt_version/<sample>/gpt_image.png` for all ten Redwood samples.
- Accepted/frozen Pixal assets:
  `gpt_version/<sample>/pixal3d.glb` and
  `gpt_version/<sample>/pixal3d_sampled_100k.ply` for all ten samples.
- Accepted/frozen v10 registrations: `06830` and `06145`, under
  `gpt_version/_pixal_so3_lattice_sim3_v10_20260822/<sample>`.
- No v10 completeness-preserving fusion has yet been approved.  Registration
  and fusion must remain separate outputs and ablations.

Do not regenerate or overwrite an accepted GPT image, original Pixal GLB/PLY,
accepted transform, registered PLY, or registered mesh without asking.

## Stage 0 — Raw Partial, Depth, and Camera

Inputs and camera artifacts:

- raw partial: `data/<sample>.ply`;
- accepted depth: `gpt_version/<sample>/depth.png`, 512×512;
- saved camera root:
  `workspace/redwood_onestage_rawdepth_512_stage2_20260714/<sample>`;
- camera/pixel state: `camera.pth` and `point_uv.npy`.

The depth and semantic images need only be approximately aligned after resize.
The saved camera provides robust silhouette, visible-depth, and same-ray
evidence; the method does not assume exact dense RGB/depth correspondence.
Always use the established `SavedCameraProjector`, including GenPC's saved
vertical image-coordinate convention, instead of rebuilding a new camera.

## Stage 1 — GPT ImageGen Image Completion

Purpose: complete the partial depth rendering into a realistic full-object
image while preserving camera pose, image footprint, orientation, scale, and
all observed geometry.

Per-sample files:

- input: `gpt_version/<sample>/depth.png`;
- exact prompt: `gpt_version/<sample>/prompt.txt`;
- accepted output: `gpt_version/<sample>/gpt_image.png`, 1254×1254;
- optional user references:
  `gpt_version/01184/product_reference.png` and
  `gpt_version/05452/product_reference.png`.

Prompt policy:

1. Use the Redwood ID/category correspondence only to name the object.
2. Treat the depth image as authoritative for pose, viewpoint, projected size,
   silhouette, and observed parts.
3. Complete only missing or occluded geometry.
4. Keep one complete object on a pure white background.
5. Forbid canonical-view rotation, recentering, zoom, extra parts, floor,
   cast shadow, text, and watermark.
6. User-supplied structural references may constrain genuine object structure,
   such as the parallel two-wheel arrangement of `01184` and thin curved chair
   profile of `05452`; they must not become registration case logic.

The generator is OpenAI's built-in GPT image editor.  Exact serving checkpoint,
seed, steps, guidance, scheduler, negative prompt, and backend are not exposed
and are recorded as `UNKNOWN`.  Reproducibility is therefore based on frozen
outputs and verbatim prompt files.  No crop or recenter is applied when saving
`gpt_image.png`.

## Stage 2 — Pixal3D Complete 3D Generation

Implementation: `scripts/run_pixal3d_gpt_batch.py`.

Model assets:

- source: `models/Pixal3D`;
- projection weights: `models/Pixal3D-weights`;
- DINOv3: `models/dinov3-vitl16-pretrain-lvd1689m`;
- internal camera/geometry model: `models/moge-2-vitl/model.pt`;
- background removal: `models/RMBG-2.0`;
- attention backend: `xformers`;
- NAF: official release checkpoint in the Torch Hub cache.

Preprocessing removes the white background, applies Pixal3D's official 1.1
foreground crop, and saves the exact RGBA condition as
`gpt_version/<sample>/pixal3d_input.png`.  Pixal3D then uses its internal MoGe-2
camera estimate.  Because this geometry/camera reasoning already occurs inside
Pixal3D, the old external partial→MoGe→complete transform composition is not
part of the active route.

Shared parameters for all samples:

- seed 42;
- `1024_cascade`, actual resolution 1024;
- sparse structure: 12 steps, guidance 7.5, rescale 0.7, `rescale_t=5.0`;
- shape: 12 steps, guidance 7.5, rescale 0.5, `rescale_t=3.0`;
- texture: 12 steps, guidance 1.0, rescale 0.0, `rescale_t=3.0`;
- remeshed GLB target 300000 faces and 2048 texture;
- deterministic uniform surface sample of 100000 points with seed 42.

Outputs:

- `pixal3d_input.png`;
- accepted `pixal3d.glb`;
- accepted `pixal3d_sampled_100k.ply`;
- exact `pixal3d_metadata.json`.

Registration transforms these frozen outputs.  It must never regenerate a
different GLB to make a difficult case easier.

## Stage 3 — Visibility-Aware SO(3) + Sim(3) TTT

Implementations:

- `scripts/run_pixal_pca_sim3_ttt_v2.py`;
- `scripts/run_pixal_pca_depth_sim3_ttt_v3.py`;
- `scripts/run_pixal_local_sim3_ttt_v4.py`;
- `scripts/run_pixal_so3_lattice_sim3_ttt_v10.py`;
- root launcher: `run_pixal_so3_lattice_sim3_ttt_v10.py`.

The only permitted transform is:

```text
p_registered = s R p_pixal + t
```

`R` must be a proper rotation, `s` one positive isotropic scale, and `t`
translation.  Independent axis scales, affine shear, point deletion, non-rigid
deformation, regenerated shapes, and GT-guided candidate selection are
forbidden.

### Visible partial-to-partial principle

Ordinary complete-to-partial ICP or symmetric Chamfer incorrectly penalizes
valid unseen complete geometry and can shrink long objects.  At each candidate
pose, the method z-buffers the complete Pixal model through the saved camera
and compares only its visible observation-supported subset with the raw
partial.  The optimization behaves as visible-partial-to-raw-partial fitting,
while all hidden Pixal points remain untouched.

### v10 shared hypothesis schedule

Partial-view PCA may miss the correct orientation.  v10 composes 24 proper PCA
rotations with the fixed Euler grid `{-45°, 0°, 45°}^3`, deduplicating to 648
orientations for every category.

Coarse stage:

- render 64;
- 2500 complete and 1500 partial points;
- scales `0.6, 0.8, 1.0, 1.2, 1.4`;
- median-centered translation;
- retain 24 candidates.

Fine stage:

- render 128;
- 10000 complete and 6000 partial points;
- refine isotropic scale and translation;
- locally refine the best eight rotations;
- saved-camera padding 0.15.

The GT-free score combines silhouette IoU, observed coverage, generated
leakage, robust partial-to-complete surface distance, same-camera visible depth,
and bounded transform updates.  Hidden complete points receive no
partial-distance loss.

Each run writes a registered full 100k PLY, transformed original GLB,
gray-partial/red-Pixal comparison PLY, 4×4 Sim(3), projection PNG, and complete
ranking/trace JSON.

Accepted v10 scales:

- `06830`: `1.1975`;
- `06145`: `0.73`.

Both preserve all 100000 source points and the complete model topology.

## Stage 4 — Completeness-Preserving Fusion

Fusion starts only after registration is accepted.  The registered full Pixal
model is the completeness主体; the raw partial is immutable observed evidence.

Required fusion invariants:

1. Keep the registered full Pixal cloud/mesh as an untouched canonical output.
2. Do not delete, crop, deform, or replace any accepted Pixal point.
3. Match raw partial points only to the z-buffer-visible Pixal surface inside
   observed foreground and a robust depth band.
4. Store raw observations as a separate support layer or immutable prefix.
5. Deduplicate only newly added observation points; never deduplicate away the
   Pixal body.
6. Smooth only the added observation/Pixal seam with local MLS or graph-
   Laplacian displacement.  Far-field and unseen Pixal geometry stays fixed.
7. SDS is optional only as a low-weight observation-clamped seam ablation; it
   cannot update global pose, scale, or complete Pixal geometry.
8. Save registered-unfused, direct-union, and seam-smoothed variants separately.

If evaluation requires a fixed point count, create a deterministic stratified
metric derivative.  Do not overwrite or mislabel it as the canonical 100k
Pixal body.

The older GenPC-scaffold detail graft is retained only as an ablation.  It is
not the active fusion route because the current requirement is for Pixal3D,
not the scaffold, to preserve completeness.

No v10 fusion output is accepted yet.  Do not present fusion as complete until
visual approval and frozen full-ten metrics are available.

## Stage 5 — Post-Freeze Evaluation

GT is forbidden during image generation, 3D generation, registration,
candidate selection, fallback routing, and fusion.  Load GT only after outputs
are frozen.

Protocol:

- `utils.loss_util.Completionloss` CD-L1 and EMD;
- deterministic FPS, 16384 points where possible;
- current diagnostic seed 6145;
- report every sample and full-ten mean beside GenPC paper rows;
- never select candidates or per-sample parameters using CD/EMD.

Accepted registration-only diagnostics, x1e2:

| sample | previous CD / EMD | accepted v10 CD / EMD |
| --- | ---: | ---: |
| `06830` | `6.93099 / 11.29695` | `2.72791 / 4.92008` |
| `06145` | `3.22053 / 2.99815` | `1.47280 / 1.75263` |

These are not final fused full-ten paper results.

## Zero-Shot and Generalization Contract

- no training or fine-tuning on Redwood test geometry;
- no GT geometry or metrics during inference/selection;
- one shared orientation lattice, scale range, objective, and gates;
- no sample-ID branches or category-specific pose rules;
- category text may describe the object for image completion but cannot select
  a 3D transform;
- accepted GPT/Pixal inputs are frozen before registration;
- failed cases are reported, not hidden by raw-partial substitution, GT,
  regenerated GLBs, anisotropic scale, or non-rigid deformation.

## Reproduction Order

1. Verify frozen `depth.png`, `prompt.txt`, and `gpt_image.png`.  Regenerate an
   image only after explicit rejection because exact server-side reproduction
   is unavailable.
2. Run `scripts/run_pixal3d_gpt_batch.py` only for missing/unaccepted Pixal
   assets; never overwrite accepted outputs.
3. Run registration from the project root:

   ```bash
   CUDA_VISIBLE_DEVICES=0 /opt/data/private/cr/miniconda3/envs/genpc/bin/python \
     run_pixal_so3_lattice_sim3_ttt_v10.py --samples 06145 06830
   ```

4. Inspect projection PNG and gray-partial/red-Pixal PLY before fusion.
5. Run fusion into a new derivative root only after registration approval.
6. Freeze all outputs, then compute per-sample and full-ten CD-L1/EMD.

## Paper Ablations

1. original GenPC Qwen→MoGe→Hunyuan route;
2. GPT ImageGen + Pixal3D + PCA-24 Sim(3);
3. + shared SO(3) lattice;
4. + same-camera visible depth;
5. + visible-partial-to-partial refinement;
6. ordinary full-cloud ICP failure baseline;
7. direct-union fusion;
8. completeness-preserving observation fusion;
9. + local seam smoothing, with SDS optional;
10. full-ten generalization and category-wise failure analysis.

The intended contribution is not only stronger image or 3D generation.  It is
a zero-shot visibility-aware proper-Sim(3) registration and completeness-
preserving observation-fusion framework that converts a strong single-image
3D prior into a scan-aligned complete object without destroying unseen
geometry.

## Exact Records

- accepted GPT ImageGen/Pixal ten-sample baseline: `PROJECT_STATE.md`, entry
  `2026-08-22 01:07 CST`;
- accepted v10 registration record:
  `reproducibility/ACCEPTED_pixal_so3_lattice_v10_20260822.md`;
- sample diagnostics:
  `reproducibility/06830_so3_lattice_sim3_v10_20260822.md` and
  `reproducibility/06145_so3_lattice_sim3_v10_20260822.md`;
- detailed registration design: `docs/visibility_aware_pixel_sim3_ttt.md`.
