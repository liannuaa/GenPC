# Fusion-free object completion and registration

This document is the canonical specification of the current GenPC++ object
mainline. The scene-level wrapper is documented separately. Historical
Gaussian fusion, PosteriorAdapter, axis-stretch, SDS, and sample-specific
recovery experiments are not part of this route.

## 1. Problem and inference firewall

Given a partial point cloud \(P\), the method predicts a complete 100k-point
cloud \(X^*\). Inference may use the partial-derived camera, depth, semantic
images, generated priors, and their diagnostic residuals. It must not read
ground truth, CD, EMD, or dataset-specific geometry rules.

All transforms are proper isotropic similarities,

\[
T(x)=sRx+t,\qquad R\in SO(3),\;s>0.
\]

The object name is used only as an image-generation prompt. Registration has
one shared configuration across categories and datasets.

## 2. Observation and initial complete prior

The saved-view stage selects a deterministic partial-derived camera, rasterizes
its grayscale depth and stores `camera.pth` plus the point/pixel map. Qwen
completes that observation into a semantic RGB image. An external clarity edit
may improve appearance but must retain the saved camera, foreground location,
silhouette, scale, pose, and articulation.

Pixal3D converts the image into a textured GLB and an ordered 100k-point
carrier. Its native MoGe camera observation is retained. The existing
Pixal--MoGe--partial two-camera procedure estimates an initial Sim(3): native
Pixal-to-MoGe placement, a pixel-indexed bridge to the physical Camera-1
partial, and broad-to-narrow visible 2D+3D refinement. This initial prior is
not the final prediction; it supplies complete topology, texture, and a stable
camera frame for residual reasoning.

## 3. Camera-consistent posterior image conditions

The registered textured prior is rendered in FRONT, SIDE, and BACK orbit views
defined relative to Camera-1. The same cameras project the physical partial.
For every view, the method records prior RGB, the shared low-frequency target,
partial depth, and remaining local residual.

The residual is decomposed once in 3D:

\[
d_i = p_i-x_{\nu(i)} = d_i^{\mathrm{low}}+d_i^{\mathrm{local}},
\]

where \(\nu(i)\) is the current prior support of partial sample \(p_i\).
\(d^{\mathrm{low}}\) is a smooth graph field anchored strongly in already
consistent regions; \(d^{\mathrm{local}}\) is computed only after subtracting
that field. Thresholds are derived from median point spacing and residual
quantiles, not class or part labels.

Image editing is deliberately split into two auditable actions:

1. one joint three-view edit applies only the shared low-frequency change;
2. one residual-only edit per view corrects locally supported contours without
   repeating the global change.

The prompts require connected geometry and protection of unsupported regions.
A deterministic framing operation then restores each edited foreground to the
original view centre and isotropic image scale. These normalized images are
the registration conditions.

## 4. Complete-prior regeneration

TRELLIS-image-large receives the three edited images and generates one new
complete mesh and Gaussian asset. The fixed release settings are:

| Setting | Value |
| --- | --- |
| mode | stochastic multi-image |
| seed | 42 |
| sparse sampler | 12 steps, CFG 7.5 |
| structured-latent sampler | 12 steps, CFG 3.0 |
| exported carrier | 100,000 mesh-surface samples |

The official multi-image API consumes image embeddings but no camera
extrinsics. Consequently, FRONT/SIDE/BACK provide shape evidence rather than
an assumed calibrated coordinate frame in the exported TRELLIS asset. The
subsequent registration explicitly recovers that frame.

## 5. TRELLIS-to-partial registration

Registration uses two complementary evidence sources: the edited views capture
semantic orientation and complete silhouette, while Camera-1 supplies physical
depth and visible 3D support.

### 5.1 Semantic basin capture

For each edited condition, a finite yaw/pitch/roll render set is compared with
the regenerated asset using appearance features, spatial features, silhouette,
aspect ratio, and coarse RGB layout. A joint shortlist selects a proper shared
orientation whose three proposals are mutually consistent.

Starting from that basin, the official NVIDIA `nvdiffrast` rasterizer and
PyTorch3D SO(3) maps optimize one shared object Sim(3). Small side/back orbit
residuals are nuisance variables that absorb image-generation camera drift;
they are not applied to the physical object. Camera-1/front remains the gauge.
The fixed continuation is:

| Pass | Resolution | Steps | Purpose |
| --- | ---: | ---: | --- |
| semantic capture | 128 | 420 | enter the correct complete-shape basin |
| Camera-1 capture | 256 | 400 | one partial-to-prior inverse Sim(3), then rendered refinement |
| camera polish | 256 | 500 | small shared-object and nuisance-orbit refinement |

Per-pass object trust regions use the shared defaults: 3 degrees rotation,
1.025 scale ratio, and 0.02 partial-diagonal translation. The Camera-1 inverse
capture is bounded by 6 degrees, 1.08 scale ratio, and 0.06 diagonal
translation.

### 5.2 Physical Camera-1 residual Sim(3)

The final stage retains all partial support instead of discarding a fixed
high-residual fraction. It searches a bounded residual proper Sim(3) and scores
each candidate with robust partial-to-prior distance plus the saved Camera-1
depth/visible-surface objective. The accepted shared bounds are 22 degrees,
1.14 scale ratio, and 0.18 partial diagonal; population 5, 14 iterations, seed
6145, and visible-objective weight 1.2. Semantic silhouette has zero weight in
this last pass because Camera-1 physical evidence is the authority.

The optimization never uses GT or complete-cloud metrics. It transforms every
one of the 100k regenerated prior samples and outputs
`final/<sample>/complete_100k.ply`.

## 6. What is intentionally absent

The current prediction is the registered regenerated prior. There is no point
concatenation, voxel fusion, point deletion, local Gaussian edit, non-rigid
warp, or metric-selected fallback. This keeps complete support and makes the
method boundary clear while the regeneration and registration stages are
validated across datasets.

## 7. Artifact contract

For a run root and sample `<id>`:

```text
inputs/partial/<id>.ply
inputs/camera/<id>/{depth.png,img.png,camera.pth,point_uv.npy,...}
inputs/pixal/<id>/{gpt_image.png,prompt.txt,pixal3d.glb,pixal3d_sampled_100k.ply,...}
pixal_registration/<id>/final/camera1_amplified_registered_100k.ply
multiview/<id>/render/
multiview/<id>/evidence/
multiview/<id>/edits/
multiview/<id>/conditions/{front,side,back}.png
trellis/<id>/{trellis_mesh_raw.glb,trellis_gaussian.ply,trellis_mesh_sampled_100k.ply}
trellis_registration/<id>/{semantic_capture,camera_capture,camera_polish,final}/
final/<id>/complete_100k.ply
manifests/<id>.json
```

`scripts/run_object_mainline.py` owns this contract and resumes at file-level
checkpoints. Its manifest records available input/output hashes, commands,
`ground_truth_used=false`, and `fusion_used=false`.

## 8. Offline evaluation

`scripts/evaluate_mainline_redwood.py` is a separate process that locates only
published `complete_100k.ply` predictions. It is run after inference is frozen;
its outputs cannot alter view choice, image editing, generation, registration,
or stopping.
