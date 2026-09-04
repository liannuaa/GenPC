# Fixed Pixal--MoGe--partial registration and Gaussian completion

This document is the canonical method description for the compact GenPC+
mainline. All parameters below are shared across the Redwood-10 batch. No
ground truth, CD, EMD, category-specific path, or sample-specific router is
available to registration or editing.

## Inputs

For a partial point cloud \(P\) under `data/redwood/partial` (with offline
ground truth under `data/redwood/gt`), Camera-1 \(C_1\) is selected deterministically
from a 256-view Fibonacci sphere: hidden-point-removal coverage is evaluated on
10k FPS points, and a partial-depth front/back tie-break resolves the selected
direction. The visible partial points are rasterised as a normalized **grayscale
depth** image, hole-filled with OpenCV, and completed by Qwen into semantic
image \(I\). The stage also saves \(C_1\) and per-point image coordinates
\(u_P\).

A GPT clarity edit of \(I\) provides the Pixal input image. Its prompt is
recorded with the asset and restricts the edit to material/detail cleanup; the
following Sim(3) route makes no assumption about its absolute image scale.
Pixal3D outputs a textured complete mesh and a 100k-point prior \(X\); its own
input image also gives a MoGe reconstruction \(M_2\) in the Pixal camera frame
\(C_2\).

For an end-to-end run, the FP16 MoGe observation of the already preprocessed
Pixal input is materialized with the Pixal assets and verified against the
input-image hash before native registration.  This is an implementation cache:
it uses the same MoGe tensor contract and does not alter the camera estimate,
registration objective, or any data-dependent decision.

## Registration

The registration is deliberately staged so that global pose and local
partial-scan correction do not compete.

1. **Native Pixal--MoGe alignment.** Pixal export metadata supplies its camera
convention and a deterministic analytic initial transform. A small proper
Sim(3) refinement optimizes rendered silhouette, visible depth, boundary and
3-D agreement between \(X\) and the MoGe cloud \(M_2\). It produces the
native Pixal-frame observation and avoids an unconstrained global PCA search.

2. **Two-camera bridge.** The same semantic image is used to form a MoGe cloud
in the saved partial camera. Pixel-indexed correspondences connect
\(u_P\) to this cloud. The bridge estimates a proper isotropic Sim(3), carries
the native Pixal prior into Camera-1 coordinates, and refines only the
remaining camera-chain error using saved-view silhouette/depth and visible
3-D pairs.

3. **Coupled residual and Camera-1 continuation.** A joint residual aligns
the native MoGe evidence, bridge matches, partial surface, and saved-view
render. The final continuation applies the fixed three-level Camera-1 update,
followed by a 1-degree wide tilt and a final 0.5-degree continuation. Every
stage is applied; diagnostic scores never reject or route samples.

The final registered complete body is
`registration/<sample>/final/camera1_amplified_registered_100k.ply`.

## Partial-anchored Gaussian edit

Registration cannot remove genuine prior/scan shape disagreement. The final
stage interprets the 100k registered Pixal samples as Gaussian means. It does
not concatenate point clouds or discard unobserved Pixal support.

- Saved Camera-1 and six signed-PCA virtual views supply only positive,
  mutual pixel-overlap correspondences. Missing partial pixels are not treated
  as empty space.
- Collision-free partial matches are fixed Dirichlet controls on a kNN graph
  over the Pixal means. A screened harmonic solve propagates their displacement
  along local surface structure.
- Six self-renders of the original Pixal prior softly protect unobserved
  geometry. Components without controls remain unchanged.
- The decoder replaces at most one Pixal slot per partial anchor, so every
  final cloud retains exactly 100,000 slots and the complete prior remains the
  body carrier.

Frozen shared edit parameters:

| Parameter | Value |
| --- | ---: |
| saved/virtual match radius | 2 px |
| virtual views / resolution | 6 / 384 |
| anchor residual cap | 0.075 partial-bbox diagonal |
| displacement cap | 0.075 partial-bbox diagonal |
| graph neighbors / edge ratio | 8 / 1.8 |
| graph screening | 0.0015 |
| self-protection views / weight | 6 / 0.01 |
| protection exclusion radius | 0.10 partial-bbox diagonal |
| remote gain | 1.0 |
| CG tolerance / iterations | 1e-5 / 240 |

The final prediction is
`gaussian/<sample>/decoded/partial_anchored_gaussian_decoded_100k.ply`.

## Evaluation firewall

`scripts/evaluate_mainline_redwood.py` is an offline utility. It may read
ground truth only after predictions are frozen. Its CD-L1/EMD values must not
change registration, edit parameters, candidate selection, or routing.
