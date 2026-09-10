# GenPC++ object completion mainline

This document is the canonical specification of the object-level method.
Historical TRELLIS, Pixal3D-MV, axis-stretch, component-motion, point-concatenation,
and metric-routed variants are not part of the released route.

## 1. Inference contract

Given a partial point cloud \(P\), GenPC++ predicts a complete ordered carrier
\(X^*\) without task-specific training. Inference may use only partial-derived
images, saved cameras, generated priors, and no-reference residuals. Ground
truth, CD, EMD, dataset labels, part labels, and sample-specific thresholds are
forbidden.

The method has three conceptual stages:

1. create and register a complete image-conditioned prior;
2. diagnose positive partial evidence in camera-consistent views;
3. infer a structure-preserving posterior over the complete carrier;
4. decode reliable measured observations into that fixed complete carrier.

## 2. Observation and complete prior

A saved partial camera rasterizes a depth observation and stores its camera
parameters. A semantic completion model reconstructs a clean object image
while preserving the observed pose, silhouette, scale, and articulation.
Pixal3D converts that image to a textured GLB and an ordered 100k surface
carrier. Its native MoGe observation is retained as a camera bridge.

Camera-1 has an explicit projection contract. For a pinhole observation, the
intrinsic matrix, world-to-camera extrinsic, raster size, and top-left UV origin
are saved in `camera.json` and reused exactly by every registration and
diagnostic stage. Historical normalized-camera assets retain their original
projection path. When an older asset does not declare its normalized UV origin,
the bridge chooses top-left or bottom-left using only the fraction of projected
partial samples supported by the observed foreground mask. This prevents a
dataset-dependent coordinate convention from being mistaken for geometric
misalignment.

The initial placement composes native Pixal-to-MoGe geometry, pixel-indexed
MoGe-to-partial evidence, and a bounded Camera-1 2D+3D refinement. Every global
transform is a proper similarity,

\[
T(x)=sRx+t, \qquad R\in SO(3),\quad s>0.
\]

This stage solves the global pose, translation, and isotropic scale while
preserving the complete prior.

The residual Camera-1 search may be repeated as a monotonic broad-to-narrow
continuation. Each pass uses the same proper-Sim(3) visible objective; identity
is included in the candidate lattice, and continuation stops when the relative
no-GT objective reduction falls below a fixed threshold. The mechanism changes
neither the objective nor the action space and introduces no category or sample
branch.

## 3. Agentic observation diagnosis

Camera-1 and three auxiliary orbit views form a compact observation state. The
auxiliary views maximize incremental visibility of physical partial points
subject to a yaw-separation constraint. In each selected camera the system
renders the registered textured prior and projects the partial scan with a
z-buffer.

Only visible partial samples are positive evidence. Missing pixels are unknown
and never imply that prior geometry should be removed. Residual arrows point
from the current prior surface toward the supported partial target. These
visibility-valid pixel and depth residuals from every view enter Partial OT.
The physical deformation is still solved in 3D by one shared action rather
than independent per-view edits.

## 4. Structure-aware Partial OT

Nearest-neighbour matching fails when corresponding structures are separated
by a substantial shape or extent error. GenPC++ therefore estimates an
unbalanced Partial OT coupling using extrinsic camera evidence and intrinsic
surface evidence:

\[
C_{ij}=\lambda_{3d}C^{3d}_{ij}
      +\sum_v w_{ij}^{(v)}(\lambda_{xy}C^{xy,(v)}_{ij}+\lambda_zC^{z,(v)}_{ij})
      +\lambda_nC^n_{ij}
      +\lambda_{\Sigma}C^{\Sigma}_{ij}+\lambda_gC^g_{ij}.
\]

The view weight is nonzero only when the partial observation and candidate
prior surfel are both visible in view (v). The terms compare screen
position, camera depth, normals, Euclidean distance,
local covariance spectra, and graph-geodesic signatures. Unbalanced transport
may leave unsupported mass unmatched. Its confidence and robust residual
quantiles induce three states without semantic labels:

- `supported-stable`: reliable and already aligned; kept rigid;
- `supported-residual`: reliable but inconsistent; drives deformation;
- `unsupported`: no positive observation; follows only structural propagation.

All thresholds scale with median spacing, robust quantiles, partial extent, or
visibility ratios.

## 5. Prior-preserving posterior deformation

The registered complete carrier is represented by a multi-resolution surface
graph. Coarse low-frequency modes absorb continuous extent differences; a
fine embedded ARAP field then resolves local supported residuals. Gaussian
means are controlled by neighbouring graph nodes:

\[
x_i'=\sum_k w_{ik}\left[R_k(x_i-g_k)+g_k+t_k\right].
\]

The objective combines transported observation evidence with stable anchors,
ARAP rigidity, Laplacian smoothness, attachment preservation, and unobserved
prior protection:

\[
\min_{\Pi,\Phi}
\langle C(\Phi(X),P),\Pi\rangle
+\lambda_s E_{\mathrm{stable}}
+\lambda_a E_{\mathrm{ARAP}}
+\lambda_l E_{\mathrm{lap}}
+\lambda_c E_{\mathrm{attach}}
+\lambda_p E_{\mathrm{prior}}.
\]

Correspondence and deformation are alternated coarse-to-fine. The solver never
deletes or concatenates points: all 100k prior slots remain the prediction.
Unobserved regions preserve relative structure and coverage, but may move with
their connected support.

## 6. Four-view observation-anchored fusion

The posterior deformation resolves shape disagreement, but an adapted prior is
still an estimate rather than a physical measurement. The final decoder reuses
exactly the four selected cameras from Section 3. In each camera it z-buffers
the posterior and partial, proposes only visibility-valid pixel matches, and
checks both image displacement and camera-depth agreement. The union of these
positive correspondences expands observed coverage, while their cross-view
support resolves competing assignments.

A deterministic collision-free assignment maps at most one partial sample to
each carrier slot and at most one carrier slot to each partial sample. Reliable
assignments replace the corresponding posterior means:

\[
X_i^{\mathrm{final}}=
\begin{cases}
P_j,&(j,i)\in\mathcal A,\\
X_i^*,&\text{otherwise}.
\end{cases}
\]

This is neither concatenation nor another deformation field. It preserves all
100k slots, copies unmatched and unobserved posterior geometry exactly, and
places measured samples directly on the final observed surface. Ground truth
and offline metrics are not available to correspondence construction.

## 7. No-GT posterior diagnostics

The posterior output is not selected or reverted by a separate verifier.
Instead, the runner records the full saved-view objective and these integrity
diagnostics for inspection:

- no new significant connected component;
- bounded robust graph-edge strain;
- stable-support motion near zero;
- hidden-view coverage at least 95% of the input prior.

These values never choose between the registered and adapted prior. GT metrics
are computed later by a separate process and cannot affect inference.

## 8. Artifact contract

For each sample, `scripts/run_posterior_adapter.py` writes:

```text
posterior_prior_100k.ply
coarse_prior_100k.ply
partial_gray_posterior_red.ply
partial_ot_pairs.npy
posterior_field.npz
posterior_support_status.png
posterior_saved_view_projection.png
posterior_virtual_views.png
posterior_info.json
```

`posterior_info.json` records the fixed configuration, support counts,
deformation diagnostics and absolute input/output paths.

The final fusion runner additionally writes:

```text
observation_anchored_fused_100k.ply
partial_gray_fused_red.ply
fused_four_view_projection.png
fusion_anchor_pairs.npy
fusion_positive_pairs.npy
fusion_info.json
```

## 9. Scene-level extension

The scene route remains independent. Scene MoGe and instance masks provide a
shared coordinate system; each complete textured instance mesh is registered
into that frame and composed with minimum camera-depth collision correction.
It does not change the object-level posterior solver.
