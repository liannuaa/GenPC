# Visibility-Aware Pixel-Sim3 Test-Time Registration

## Status

Active method proposal, adopted on 2026-08-22 for the next Pixal3D
registration and fusion experiments. The first implementation and visual gate
target is Redwood sample `07136`; the same code and parameters must then be
evaluated on all ten target samples.

This document is the method contract. It supersedes anisotropic axis scaling,
non-rigid deformation, and low-density Pixal replacement as the main Pixal3D
route. Earlier results remain available only as ablations and failure analysis.

## Objective

Register a complete Pixal3D mesh to a raw partial Redwood scan while:

1. preserving the accepted complete shape;
2. fitting the observed local surface closely;
3. estimating rotation, translation, and one global isotropic scale;
4. remaining zero-shot and category-agnostic;
5. using no ground truth during inference, candidate selection, or fusion;
6. retaining the full Pixal3D model as the completeness主体 after fusion.

The method is a direct evolution of GenPC's 2D+3D posterior registration:

```text
raw partial + saved camera/pixels
  -> accepted GPT semantic image
  -> frozen accepted Pixal3D mesh
  -> pixel-conditioned visible mesh subset
  -> multi-hypothesis isotropic Sim(3)
  -> visibility-aware test-time refinement
  -> complete-preserving fusion
```

Pixal3D already uses MoGe internally. The old extra
partial-to-MoGe-to-complete composition is therefore not part of this route.

## Frozen Inputs

The following ten accepted assets are immutable inputs:

- samples: `01184`, `05117`, `05452`, `06127`, `06145`, `06188`, `06830`,
  `07136`, `07306`, `09639`;
- textured mesh: `gpt_version/<sample>/pixal3d.glb`;
- deterministic complete sample: `gpt_version/<sample>/pixal3d_sampled_100k.ply`;
- semantic image: `gpt_version/<sample>/gpt_image.png`;
- Pixal input/camera metadata: `gpt_version/<sample>/pixal3d_input.png` and
  `gpt_version/<sample>/pixal3d_metadata.json`;
- raw partial and camera bridge: `data/<sample>.ply`, saved `camera.pth`, and
  `point_uv.npy`.

Do not regenerate or overwrite these GLBs, PLYs, or semantic images. Every
registration and fusion output must be written to a separate derivative root.
Alternate-view or newly generated Pixal GLBs are excluded from this method.

## Why Ordinary Complete-to-Partial ICP Fails

A complete model contains large unobserved regions. A raw partial scan contains
only the camera-visible surface. Symmetric Chamfer, ordinary ICP, and losses
that send every complete point toward a partial nearest neighbor incorrectly
penalize valid hidden geometry. A scale optimizer can then reduce its objective
by shrinking a long object such as the `07136` sofa. This produces a closer
point-set score but an invalid complete shape.

The correct registration target is not the entire complete point cloud. It is
the z-buffer-visible subset of the complete mesh under the current pose, gated
by the observed image support.

## Related Method Principles

The design combines principles from these publication lines without adding
task-specific training:

- FreeZe, ECCV 2024: fuse frozen visual/geometric evidence from rendered model
  views and RGB-D observations for zero-shot model registration;
- PREDATOR, CVPR 2021, and GeoTransformer, CVPR 2022: estimate overlap before
  robust pose recovery in low-overlap point-cloud registration;
- SAM-6D, CVPR 2024: treat model-to-observation pose estimation as matching
  partial visible geometry rather than the full CAD surface;
- robust optimal transport, NeurIPS 2021: use soft, outlier-robust matching
  when generated and observed local geometry are not identical;
- render-and-compare novel-object pose methods: rank and refine pose hypotheses
  with silhouette, depth, and appearance evidence.

References:

- https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/09634.pdf
- https://openaccess.thecvf.com/content/CVPR2021/papers/Huang_Predator_Registration_of_3D_Point_Clouds_With_Low_Overlap_CVPR_2021_paper.pdf
- https://openaccess.thecvf.com/content/CVPR2022/papers/Qin_Geometric_Transformer_for_Fast_and_Robust_Point_Cloud_Registration_CVPR_2022_paper.pdf
- https://openaccess.thecvf.com/content/CVPR2024/papers/Lin_SAM-6D_Segment_Anything_Model_Meets_Zero-Shot_6D_Object_Pose_Estimation_CVPR_2024_paper.pdf
- https://proceedings.neurips.cc/paper/2021/hash/2b0f658cbffd284984fb11d90254081f-Abstract.html

## Transformation Contract

The only permitted geometric transform of the accepted Pixal geometry is:

```text
p_registered = s R p_pixal + t
```

where `R` is a proper rotation, `t` is translation, and `s` is one positive
global scale. The three singular values of the linear transform must be equal
within numerical tolerance, and its determinant must be positive.

Forbidden in the promoted route:

- independent x/y/z or PCA-axis scales;
- affine shear;
- deformation graphs, volumetric warps, or per-point offsets;
- category-specific upright rules or sample-specific ranges;
- ground-truth-guided candidate selection;
- replacing the accepted GLB with a regenerated candidate.

## Stage A: Pixel-Conditioned Correspondences

1. Recover the deterministic semantic-to-Pixal crop transform.
2. Map raw partial points to semantic/Pixal pixels using `point_uv.npy` and the
   saved image-coordinate convention.
3. Render the frozen Pixal mesh with z-buffering to obtain visible triangle
   points, depth, normals, silhouette, and optional texture features.
4. Establish candidate correspondences only where the projected partial and
   rendered visible mesh share foreground support.
5. Weight correspondences by pixel proximity, silhouette-boundary distance,
   normal compatibility, local depth consistency, and optional frozen visual
   feature similarity.

The correspondence stage must record coverage, inlier ratio, pixel residuals,
and spatial coverage. High correspondence count without broad object coverage
is not sufficient.

## Stage B: Multi-Hypothesis Isotropic Sim(3) Initialization

Generate several deterministic hypotheses from high-confidence pixel/3D
correspondences. Estimate scale robustly from pairwise distance ratios among
correspondences, then estimate rotation and translation. Pairwise local scale
is preferred over complete/partial bounding-box scale because a partial scan
does not determine the complete object's global extent.

Retain a small top-K set using only self-supervised evidence. Hypotheses may
include bounded rotational perturbations and symmetry-equivalent poses, but
all samples use the same schedule and limits.

## Stage C: Visibility-Aware Test-Time Refinement

For each candidate, repeatedly re-render the mesh and optimize only its current
visible, observation-supported surface. The common objective is:

```text
L = w_surface * L_robust_point_to_plane
  + w_depth   * L_visible_depth
  + w_mask    * L_silhouette
  + w_edge    * L_boundary
  + w_pixel   * L_pixel_correspondence
  + w_feat    * L_optional_visual_feature
  + w_scale   * L_log_scale_prior
  + w_delta   * L_small_update
```

Critical directionality and masking rules:

- partial points may be matched to the visible mesh;
- rendered visible points may be matched only inside observed foreground and a
  robust depth band;
- hidden/back-side Pixal points receive no partial-distance penalty;
- missing generated geometry in a genuinely observed region is penalized;
- valid generated completion outside observed support is not penalized.

Scale is optimized conservatively in the coarse stage and then frozen. The
final refinement optimizes SE(3) only. This prevents late-stage shrinkage from
trading complete extent for local nearest-neighbor distance.

Robust losses use trimming, Huber/Charbonnier penalties, or unbalanced robust
optimal transport. Ordinary all-point symmetric Chamfer is prohibited as a
registration objective.

## Candidate Selection and Safety Gates

Candidate ranking uses a shared GT-free score combining:

- observed-pixel coverage;
- silhouette IoU, leakage, and edge alignment;
- visible depth residual;
- robust partial-to-visible-mesh surface residual;
- correspondence spatial coverage;
- scale and update regularity.

Reject or fall back when:

- the transform is not an isotropic proper Sim(3);
- scale reaches its search boundary without support;
- visible fit improves only by losing silhouette coverage;
- the candidate decreases object extent through a late scale update;
- pixel/depth evidence is spatially concentrated or contradictory.

GT may be evaluated only after the selected transform and outputs are frozen.

## Stage D: Complete-Preserving Fusion

Registration must be accepted visually and geometrically before fusion starts.
The first diagnostic output is therefore the transformed full Pixal model and
a gray-partial/red-Pixal comparison cloud.

The fused prediction must retain every transformed point from the accepted
100k Pixal sample. The raw partial is exact observed evidence and may be added
as an immutable prefix. Fusion must not delete, crop, deform, or uniformly
downsample the Pixal body.

Seam handling is permitted only on the added observation/support layer:

- merge near-duplicate added partial points without touching Pixal points;
- optionally project only added support points toward the registered mesh;
- apply local MLS/Laplacian smoothing only to added seam points;
- keep far-field and unobserved Pixal completion unchanged.

SDS is not part of registration. It may be tested later only as a low-weight,
observation-clamped seam ablation and must not update the Pixal completeness
body.

## Evaluation Protocol

Development order:

1. `07136`: verify that the sofa length, orientation, and local contact are
   preserved before fusion;
2. `01184`: second visual and geometric diagnostic;
3. all ten samples with one configuration and no per-sample tuning;
4. fuse only after the registration output passes the method gates;
5. report every sample's CD-L1 and EMD beside the GenPC paper values.

Primary research requirements:

- zero-shot and category-agnostic;
- no GT during inference or selection;
- no per-sample parameter tuning;
- exact frozen input provenance;
- reproducible deterministic hypothesis schedule;
- full ten-sample and average reporting;
- target average below GenPC CD-L1 `1.74` and EMD `2.88`, while pursuing the
  stricter goal of beating each GenPC row on both metrics.

Required ablations:

1. 3D-only visible Sim(3);
2. + pixel correspondence;
3. + silhouette/depth render-and-compare;
4. + robust OT or robust correspondence weighting;
5. scale optimized throughout versus coarse-scale-then-freeze;
6. direct union versus observation-layer seam smoothing;
7. frozen full Pixal body versus the rejected low-density/body-replacement
   alternatives.

## Required Outputs

For each sample, save:

- registered full 100k Pixal PLY;
- registered transformed mesh GLB derived from the frozen original GLB;
- isotropic Sim(3) matrix and diagnostics JSON;
- partial/Pixal comparison PLY;
- semantic-camera and raw-depth-camera overlays;
- visible-surface/depth residual diagnostics;
- full-body fused PLY only after registration acceptance;
- post-freeze CD-L1/EMD metrics.
