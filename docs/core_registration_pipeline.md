# Qwen-GPT-Pixal Bidirectional Completion Pipeline

## Active zero-shot mainline

```text
partial point cloud + saved Redwood camera
  -> Redwood-compatible depth.png
  -> Qwen semantic completion (img.png: geometry/pose anchor)
  -> GPT ImageGen clarity-only edit
  -> Pixal3D complete GLB + 100k surface PLY
  -> frozen v15 coarse proper-Sim(3) initialization
  -> independent partial→visible-prior and visible-prior→partial pairs
  -> saved-camera 2D+3D bidirectional consensus TTO
  -> strict inverse moves the complete Pixal body to the partial frame
  -> observation-conditioned visible surface posterior
  -> complete 100k prediction
```

This remains descended from GenPC: it retains the partial scan, saved camera,
depth projection, and robust v15/GenPC coarse registration, while adding a
Qwen-anchored GPT/Pixal complete prior and bidirectional test-time Sim(3).

## Geometry-preserving image contract

The semantic source is
`workspace/redwood_onestage_rawdepth_512_stage2_20260714/<sample>/img.png`.
Qwen is the sole authority for camera pose, image-space position, projected
size, silhouette, part layout, local articulation, and occlusion. GPT may only
improve sharpness, boundaries, material coherence, and surfaces already
implied by Qwen. It must not rotate, rescale, recenter, mirror, redesign,
add/remove parts, or alter wheel, leg, armrest, leaf, or tabletop orientation.
Each prompt is stored beside its output as `prompt.txt`.

## Pixal3D prior

`scripts/run_pixal3d_gpt_batch.py` consumes `gpt_image.png` with one shared
configuration: TencentARC/Pixal3D, local Pixal3D/DINOv3/MoGe-2/RMBG weights,
seed 42, 1024 cascade, and the shared 12-step sampler. It writes
`pixal3d_input.png`, `pixal3d.glb`, `pixal3d_sampled_100k.ply`, and metadata.

## Bidirectional 2D+3D Sim(3) TTO

`scripts/prepare_pixal_v15_initialized_priors.py` applies the frozen v15
transform only as a coarse initialization. Final registration uses independent
saved-camera correspondence sets in both directions:

```text
partial -> visible complete prior
visible complete prior -> partial
```

Forward-only, reverse-only, balanced, and reciprocal hypotheses are fitted as
proper isotropic Sim(3), bounded by one shared trust region, converted to the
complete-to-partial direction through an analytic strict inverse, and selected
with visible 2D silhouette/depth and 3D surface evidence. Sample IDs,
categories, GT, CD, and EMD are unavailable to inference and routing.

Shared parameters: pixel schedule `[8,5,3]`, final radius 5, per-step rotation
cap 3 degrees, isotropic scale `[0.96,1.04]`, translation cap 0.03 partial-bbox
diagonal, minimum 96 pairs, cycle cap 0.03.

## Observation-conditioned surface posterior

`src/observation_conditioned_surface_projection.py` compares identity, smooth
absorption, and observed surface-mass budgets `[0.04,0.08,0.12]` with a shared
saved-camera objective and prior-mass penalty. Partial points are FPS
uniformized. Surface-mass projection exchanges the same number of nearest
visible-prior points for observed points, preserving 100k total points and at
least 88% of the complete generated body. Hidden/far geometry is not truncated.

## Canonical outputs

All ten inputs and outputs live under:

`workspace/redwood_qwen_gpt_pixal_bidirectional_mainline_20260823`

- `<sample>/`: depth, Qwen/GPT images, prompt, Pixal GLB/PLY/metadata;
- `_v15_transform_initial/`: coarse initialized new priors;
- `_bidirectional_consensus/`: registered PLY/GLB, transform, overlay and info;
- `_surface_projection/`: final complete 100k prediction and diagnostics;
- `postfreeze_cd_emd_strict/`: per-sample metrics, means and protocol.

The frozen v15 baseline remains at
`gpt_version/_pixal_guarded_unified_registration_v15_20260822`.

## Strict full-ten result

Predictions were frozen before GT evaluation. The protocol uses 16,384-point
FPS, seed 6145, separate prediction FPS per geometry, and equivalent GT FPS by
shared seed.

| sample | new CD | new EMD | v15 CD | v15 EMD |
| --- | ---: | ---: | ---: | ---: |
| 01184 | 1.196 | 1.961 | 1.399 | 2.163 |
| 05117 | 1.503 | 2.437 | 2.248 | 3.181 |
| 05452 | 0.862 | 1.292 | 1.163 | 1.587 |
| 06127 | 2.248 | 3.936 | 2.947 | 5.147 |
| 06145 | 1.624 | 1.849 | 1.473 | 1.786 |
| 06188 | 1.208 | 2.106 | 1.373 | 2.227 |
| 06830 | 1.983 | 4.011 | 2.685 | 4.766 |
| 07136 | 1.982 | 3.076 | 2.179 | 2.947 |
| 07306 | 2.616 | 3.187 | 2.929 | 3.429 |
| 09639 | 1.128 | 2.101 | 2.699 | 3.741 |
| **mean** | **1.635** | **2.596** | **2.110** | **3.097** |

The mean exceeds GenPC `1.74/2.88` by about 6.0% CD and 9.9% EMD. One shared
zero-shot parameterization is used. Full-ten visual review remains required.

## Cross-prior scale-consistency guard candidate

Visual review found that the Qwen-GPT 07136 prior was shorter and thicker than
the previous Pixal prior. Its registration shrank by `0.9632`, while the older
prior expanded by `1.0347`; the older candidate also passed the registration
guard and reduced the GT-free visible objective from `0.1034` to `0.0758`.

`src/cross_prior_scale_guard.py` therefore defines a shared conservative
fallback: the two priors must demand opposite scale directions, disagree by at
least 6%, current must fail while fallback passes, and fallback must improve
the visible objective by at least 20%. On all ten samples this selects fallback
only for 07136. It uses no sample ID, category, GT, CD, or EMD.

Guarded intermediate root:
`workspace/redwood_qwen_gpt_pixal_bidirectional_mainline_20260823/_cross_prior_scale_guard`.
After freezing, strict mean CD/EMD is `1.5991/2.5410`, and 07136 improves from
`1.9819/3.0755` to `1.6225/2.5098`. It is an accepted component of the final
voxel-uniform mainline; the full-new-prior result remains an ablation.

## Uniform voxel surface-measure candidate

Direct 88% prior + 12% observed mass has visibly different local sampling
density: across the nine exact-insertion cases, observed-point median nearest-
neighbor spacing is typically about twice the prior spacing. The conservative
uniformization in `src/voxel_surface_measure_resampling.py` performs two steps:

1. remove an observed point only when it is both a robust kNN outlier within
   the partial observation and farther than 2% object diagonal from prior
   support;
2. select one original point per adaptively sized voxel, preferring an observed
   point when a voxel contains one, until approximately 32,768 representatives
   remain.

The output is strictly a subset of the fused input: it creates no points,
interpolates no geometry, changes no pose/scale, and cannot warp the complete
body. On all ten samples, mean point count is 32,771 and mean normalized kNN
density CV drops from `0.498` to `0.292` (41.3% lower). The frozen strict mean
CD/EMD is `1.5944/2.5296`, slightly improving over the cross-prior input
`1.5991/2.5410` while remaining well above GenPC.

Output root:
`workspace/redwood_qwen_gpt_pixal_bidirectional_mainline_20260823/_cross_prior_voxel_uniform_32k`.
Metric root:
`workspace/redwood_qwen_gpt_pixal_bidirectional_mainline_20260823/postfreeze_cross_prior_voxel_uniform_32k_cd_emd`.
The user accepted this complete route as the best-effect pipeline on
2026-08-23. The 32k voxel-uniform root is the canonical final prediction root;
the 100k cross-prior root remains its immutable pre-resampling ablation.

## Reproduction order

1. Generate Pixal assets with `scripts/run_pixal3d_gpt_batch.py`.
2. Prepare coarse priors with `scripts/prepare_pixal_v15_initialized_priors.py`.
3. Run `scripts/run_pixal_bidirectional_cycle_registration.py --step-mode consensus --force-candidate-output`.
4. Run `scripts/run_observation_conditioned_surface_projection.py`.
5. Freeze predictions, then run `scripts/evaluate_bidirectional_cycle_redwood.py`.

Do not read GT metrics before the full prediction root is frozen.
