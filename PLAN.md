# GenPC Refactor Plan

This plan tracks the target refactor for partial point cloud reconstruction via
Qwen-Image-Edit, MoGe, Hunyuan3D, and FreeReg.

Research engineering constraint: this is academic-paper code. Prefer simple,
explainable, reproducible method changes that can be ablated cleanly. Avoid
over-engineering, model stacking, and sample-specific hacks; focused one-off
diagnostic probes are acceptable only when clearly recorded as diagnostics.

## Goal

Build a pipeline that reconstructs a complete 3D point cloud from a partial
input point cloud and aligns the complete point cloud back to the original
partial scan.

The target relation is:

```text
partial point index -> MoGe point index -> complete point index / complete aligned to partial
```

Current Redwood metric target, requested 2026-07-13: optimize the current
method while keeping the implementation academic-paper style, simple,
explainable, and not over-engineered. Avoid sample-specific hacks; improvements
must be evaluated on the full target set below.

| sample | target CD-L1 x1e2 | target EMD x1e2 |
| --- | ---: | ---: |
| `01184` | `< 2.31` | `< 3.17` |
| `09639` | `< 1.43` | `< 2.29` |
| `05452` | `< 1.16` | `< 1.68` |
| `05117` | `< 1.36` | `< 2.20` |
| `06127` | `< 2.86` | `< 4.85` |
| `07136` | `< 1.58` | `< 2.78` |
| `07306` | `< 2.72` | `< 4.36` |
| `06188` | `< 1.36` | `< 2.47` |
| `06145` | `< 1.28` | `< 2.07` |
| `06830` | `< 1.38` | `< 2.97` |
| Average | `< 1.74` | `< 2.88` |

2026-07-14 update: the final optimization target is the full 10-sample table
plus the average above. `05117` is only a diagnostic sample and must not be
treated as the sole objective. A method change is not successful unless the
full set and the average are reported and all rows satisfy the requested
CD/EMD thresholds.

Current accepted image-generation root:
`workspace/redwood_depthfirst_semantic_full_rerun_20260713`. The user accepted
the image-generation results for the full 10-sample set; do not change
`depth.png`, `qwen_edit_stage1.png`, or `img.png` unless explicitly asked.

2026-07-14 image diagnostic: user requested regenerating `09639` and `06188`
images with Qwen stage 2 at 24 inference steps and then asked to preserve the
stage-1 image. The pipeline later converged to a simpler accepted one-stage
Qwen flow: use `raw_depth.png` as the input, prompt
`生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的{photo_label}，纯白背景`,
run 40 Qwen steps, resize the result to `512x512`, save it as both
`qwen_edit_stage1.png` and `img.png`, and do not run a second Qwen refinement
stage. The lean profile keeps `raw_depth.png`, `qwen_edit_stage1.png`, and
`qwen_edit_stage1_prompt.txt`; stale `qwen_edit_stage2_prompt.txt` is removed.

2026-07-14 image diagnostic: user reported the `06188` 40/40-step image result
was still poor, then requested a shorter 8/16-step retry, and then requested
another overwrite with Qwen depth-completion stage 1 set to 4 inference steps
and semantic/RGB stage 2 set to 4 inference steps. This is an image-generation
diagnostic only; registration remains on the rolled-back baseline unless
changed separately.

2026-07-14 prompt update: user noted the stage-1 Qwen output is not reliably a
depth map, so the stage-2 prompt now refers to the input as a complete
reference image (`参考图`) rather than a complete depth map (`深度图`). `06188`
stage 2 was rerun from the existing `qwen_edit_stage1.png` only; stage 1 was
not regenerated.

2026-07-14 image diagnostic: user requested changing the stage-1 Qwen prompt to
`生成一张深度图，补全图1中所勾勒出的{photo_label}的局部形状。`, setting stage-1
steps to 40, and rerunning only `06188` stage 1. The current
`06188/qwen_edit_stage1.png` was regenerated from `raw_depth.png`; the existing
stage-2 `img.png` was intentionally not overwritten in this run.

2026-07-14 image diagnostic: user then requested changing the stage-1 Qwen
prompt to `生成一张图像，符合图1所勾勒出的局部深度图，并遵循以下描述：{photo_label}`,
keeping stage-1 steps at 40, and rerunning only `06188` stage 1. The current
`06188/qwen_edit_stage1.png` was regenerated from `raw_depth.png`; the existing
stage-2 `img.png` was intentionally not overwritten in this run.

2026-07-14 image diagnostic: user then requested adding pure-white background
to the stage-1 Qwen prompt:
`生成一张图像，符合图1所勾勒出的局部深度图，并遵循以下描述：{photo_label}，纯白背景`,
keeping stage-1 steps at 40, and rerunning only `06188` stage 1. The current
`06188/qwen_edit_stage1.png` was regenerated from `raw_depth.png`; the existing
stage-2 `img.png` was intentionally not overwritten in this run.

2026-07-14 image diagnostic: user then requested changing the stage-1 Qwen
background phrase to studio background:
`生成一张图像，符合图1所勾勒出的局部深度图，并遵循以下描述：{photo_label}，背景是摄影棚`,
keeping stage-1 steps at 40, and rerunning only `06188` stage 1. The current
`06188/qwen_edit_stage1.png` was regenerated from `raw_depth.png`; the existing
stage-2 `img.png` was intentionally not overwritten in this run.

2026-07-14 image diagnostic: user then requested switching the stage-1 Qwen
background phrase back to pure-white background:
`生成一张图像，符合图1所勾勒出的局部深度图，并遵循以下描述：{photo_label}，纯白背景`,
keeping stage-1 steps at 40, and rerunning only `06188` stage 1. The current
`06188/qwen_edit_stage1.png` was regenerated from `raw_depth.png`; the existing
stage-2 `img.png` was intentionally not overwritten in this run.

2026-07-14 image diagnostic: user interrupted a 50-step stage-1 retry and
requested changing stage 1 to use `depth.png` with prompt
`生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的{photo_label}，纯白背景`,
stage-1 steps 40, and rerunning only `06188` stage 1. The current
`06188/qwen_edit_stage1.png` was regenerated from `depth.png`; the existing
stage-2 `img.png` was intentionally not overwritten in this run.

2026-07-14 image diagnostic: user then requested switching the stage-1 input
back to `raw_depth.png` while keeping prompt
`生成一张图像，参考图1遮挡情况下的深度图，并遵循以下描述：完整的{photo_label}，纯白背景`
and stage-1 steps 40. The current `06188/qwen_edit_stage1.png` was regenerated
from `raw_depth.png`; the existing stage-2 `img.png` was intentionally not
overwritten in this run.

2026-07-14 rollback note: after the user requested reverting registration to
the pre-target-optimization version, the registration implementation was
restored to the repository baseline for `ScaleAdapter.py`,
`scripts/run_render_to_moge_sim3.py`,
`scripts/run_moge_to_partial_from_index.py`,
`scripts/run_moge_to_raw_partial_from_camera.py`, their registration tests, and
`docs/core_registration_pipeline.md`. The default registration config was also
restored to the pre-target settings (`render_sim3_max_2d_leakage: 0.10`,
silhouette depth/boundary weights `0.0`, no partial-preserving auto source, no
extra target-leakage gate). Prompt/category overrides for accepted image
generation were intentionally kept.

Current frozen-image old-method baseline from this root:

| sample | CD-L1 x1e2 | EMD x1e2 | status |
| --- | ---: | ---: | --- |
| `01184` | 1.847 | 2.650 | pass |
| `05117` | 4.351 | 5.967 | fail |
| `05452` | 1.710 | 2.439 | fail |
| `06127` | 2.355 | 3.948 | pass |
| `06145` | 4.329 | 7.140 | fail |
| `06830` | 9.557 | 10.795 | fail |
| `06188` | 5.004 | 6.484 | fail |
| `07136` | 6.606 | 7.168 | fail |
| `07306` | 10.523 | 9.874 | fail |
| `09639` | 7.327 | 8.435 | fail |
| Average | 5.361 | 6.490 | fail |

Current registration changes under test:
- Enabled depth and boundary terms in differentiable silhouette optimization:
  `render_sim3_silhouette_opt_depth_weight=0.35`,
  `render_sim3_silhouette_opt_boundary_weight=0.75`.
- Changed the 2D gate-aware objective so edge reward saturates once the
  boundary is adequate and edge chamfer is penalized only beyond the threshold.
- Allowed silhouette refinement to accept candidates that improve the 2D
  gate-aware objective even when the legacy composite render score drops.
- 2026-07-14 note: MoGe-frame anisotropic 2D refinement is kept as an optional
  ablation but disabled by default because diagnostics on `05117` and `07306`
  did not improve the final metric or 2D gate.
- 2026-07-14 one-stage raw-depth 512 image rerun: copied accepted image
  artifacts into
  `workspace/redwood_onestage_rawdepth_512_stage2_20260714` and ran Stage 2 +
  metric with `run_stage1=False`, `run_stage2=True`, `run_metric=True`. Final
  predictions and partial comparison point clouds exist for all 10 samples:
  `<sample>/<sample>_fused.ply` and
  `<sample>/<sample>_raw_partial_gray_complete_blue_aligned.ply`.
  Metrics were saved to
  `workspace/redwood_onestage_rawdepth_512_stage2_20260714/metrics_samples.csv`.
  Mean metric was `CD-L1 x1e2 = 4.735808`, `EMD x1e2 = 5.475576`, so this run
  does not meet the full target. Per-sample `CD-L1 x1e2 / EMD x1e2`:
  `01184=1.540649/2.189986`, `05117=4.559220/5.782998`,
  `05452=2.468731/3.322789`, `06127=2.657951/3.998549`,
  `06145=7.291526/7.595840`, `06830=6.588601/7.112984`,
  `06188=3.137147/4.403348`, `07136=4.572681/4.840745`,
  `07306=9.030136/8.985695`, `09639=5.511441/6.522825`.

Diagnostics:
- The final target is the full 10-sample table plus average, not a single
  `05117` target. `05117` remains useful only as a fast diagnostic sample.
- Current strict fused metrics in
  `workspace/redwood_depthfirst_semantic_full_rerun_20260713` after the latest
  registration-only reruns are:
  `01184=1.877/2.663` pass, `09639=2.357/3.482` fail,
  `05452=1.714/2.449` fail, `05117=5.293/7.001` fail,
  `06127=2.354/3.978` pass, `07136=6.613/7.224` fail,
  `07306=9.914/9.319` fail, `06188=5.016/6.507` fail,
  `06145=4.328/7.151` fail, `06830=9.557/10.769` fail, with average
  `4.902/6.054` fail.
- `09639` strict registration-only rerun with the current code and synchronized
  `09639_fused.ply` gives `CD/EMD x1e2 = 2.354/3.453`, still failing the target
  `1.43/2.29` but much better than the stale old-method metric entry.
- Temporarily relaxing only `09639` leakage gate to `0.13` allowed partial
  refinement, but the diagnostic metric on the aligned complete cloud was
  `2.529/3.810`; this did not justify loosening the hard leakage gate.
- Raw-partial and partial+complete diagnostic metrics show that some failures
  are caused by generated complete geometry hurting otherwise good partial
  scans, especially `06145`, but a simple partial+complete output strategy did
  not solve the full 10-sample target.
- Direct raw-partial plus current complete fusion gives average
  `CD/EMD x1e2 = 3.043/4.772`; it improves some samples but remains far above
  the target and can worsen samples with bad generated complete geometry.
- Disabling the 2D gate only to force partial-space refinement is not a valid
  fix. `05117` worsened to `7.180/10.887`; `07136` improved from
  `6.613/7.224` to `4.934/5.621` but still failed by a large margin.
- PCA symmetry completion from raw partial and partial-space PCA Sim3
  enumeration were tested as simple geometric alternatives. They did not meet
  the full target; PCA partial-space enumeration gave, for example,
  `05117=2.191/3.578`, `07136=5.045/5.391`,
  `07306=4.409/5.805`, and `06830=2.920/4.551`.
- Historical `workspace/redwood_qwen_studio_bg_full_rerun_20260713` outputs
  show that some targets are reachable with different generated geometry:
  `01184=0.920/1.432`, `05117=1.239/1.680`,
  `06127=2.849/4.415`, and `06145=1.337/2.104`.
  This suggests the remaining work should focus on simple candidate quality
  selection or better complete-to-MoGe initialization, not just loosening gates.
- Added an optional Hunyuan best-of-seeds path. `hunyuan_candidate_seeds`
  generates extra `<sample>_hunyuan2.1_seed<seed>.ply` candidates from the same
  accepted image. Registration runs each candidate independently and promotes
  the candidate selected by a self-supervised score: if a candidate passes the
  2D gate and partial refinement, rank by partial refinement mean distance;
  otherwise rank by the 2D gate objective. This is simple and ablatable, and
  does not add another model.
- `05117` same-image seeds `101` and `102` did not help:
  current `5.188/6.865`, seed101 `5.162/6.586`, seed102 `5.120/6.714`.
  All failed the 2D gate, so the self-supervised selector kept the current
  candidate.
- `07306` same-image seeds improved but did not solve the sample:
  current `9.927/9.365`, seed101 `6.361/6.156`, seed102 `6.358/6.122`.
  The self-supervised selector chose seed101 and the official
  `07306_fused.ply` was promoted to that candidate.
- `09639` same-image seeds `101` and `102` were generated and registered from
  the accepted image. The self-supervised selector kept the default candidate:
  default objective `2.584`, seed102 `2.494`, seed101 `2.468`. Official metric
  stayed around `1.99/3.29`; seed101 and seed102 were worse (`2.36/4.31` and
  `2.37/4.37` in the verification run).
- `05452` same-image seeds `101` and `102` were generated and registered. The
  selector chose seed102 by 2D objective (`2.524`, versus seed101 `2.506` and
  default `2.410`), but the formal metric stayed essentially unchanged around
  `1.44/2.02`. This suggests same-image Hunyuan seed search alone is not a
  sufficient method-level fix for the remaining near-threshold failures.
- After promoting `07306` seed101, current strict full-set metrics are:
  `01184=1.888/2.681` pass, `09639=2.355/3.475` fail,
  `05452=1.716/2.461` fail, `05117=5.177/6.820` fail,
  `06127=2.354/3.945` pass, `07136=6.607/7.170` fail,
  `07306=6.359/6.131` fail, `06188=4.995/6.440` fail,
  `06145=4.317/7.110` fail, `06830=9.557/10.769` fail, with average
  `4.533/5.700` fail.
- Added partial-preserving final prediction mode. This keeps all raw partial
  points and adds only complete points farther than `0.01` from the observed
  partial. It is a simple geometric guard, not a new model. The aligned
  complete-only cloud remains saved separately for inspection.
- After overwriting the official `<sample>_fused.ply` files with
  partial-preserving predictions, current metrics are:
  `01184=1.695/2.467` pass, `09639=1.991/3.304` fail,
  `05452=1.361/2.319` fail, `05117=3.226/5.530` fail,
  `06127=2.038/3.608` pass, `07136=3.663/5.178` fail,
  `07306=4.307/5.287` fail, `06188=3.031/5.207` fail,
  `06145=3.140/6.048` fail, `06830=3.775/7.441` fail, with average
  `2.823/4.639` fail. This improves the average substantially but still misses
  the final full-set target.
- MoGe-as-depth-geometry diagnostics: partial + MoGe far-point fusion helps
  some samples (`07136` best `2.51/3.12`, `06188` best `2.07/3.42`,
  `06145` best `0.71/1.48`, `06830` best `1.16/2.30`) but hurts or fails
  others (`01184`, `09639`, `07306`). PCA mirroring of the MoGe cloud did not
  solve the hard cases (`07306` best `5.739/5.762`, `07136` best
  `2.723/3.360`), so it should remain a diagnostic, not a default method.
- Added `render_sim3_final_prediction_source` with `complete`, `moge`, and
  `auto`. `auto` is a small self-supervised source selector that uses Hunyuan
  when the 2D gate is reliable and the final extent is modest, or for near-gate
  leakage/edge-only cases where MoGe adds too little new geometry; otherwise it
  uses MoGe as conservative depth geometry. This is still a simple geometric
  rule and uses no GT.
- Uniform MoGe-source partial-preserving output gave average `2.438/3.405` and
  passed `06145` and `06830`, but broke already-good `01184`/`06127`.
- Auto-source partial-preserving output was promoted to the official
  `<sample>_fused.ply` files. The current default uses source-specific
  far-point thresholds: `0.01` for aligned Hunyuan complete geometry and
  `0.02` for MoGe conservative depth geometry. Current metrics are:
  `01184=1.690/2.461` pass, `09639=1.988/3.274` fail,
  `05452=1.445/2.016` fail, `05117=2.123/3.188` fail,
  `06127=2.038/3.616` pass, `07136=2.511/3.031` fail,
  `07306=4.305/5.322` fail, `06188=2.080/3.425` fail,
  `06145=0.713/1.608` pass, `06830=1.337/2.509` pass, with average
  `2.023/3.045` fail. This is the best current all-sample average but still
  misses the final target.
- `07306` extra same-image Hunyuan seeds `103` and `104` were generated and
  registered. The old candidate selector incorrectly promoted seed104 because
  its 2D gate-aware objective was highest (`2.519`), but formal metrics were
  worse: official/seed104 about `6.51/7.11`, seed101 `5.78/6.71`, seed102
  `4.31/5.30`, and seed103 `5.15/6.01`. The selector fallback was changed to
  rank by full render score when no candidate passes the 2D gate, with the
  2D gate-aware objective retained as logging/tie-break information. This
  selected seed102 and promoted it back to the official `07306_fused.ply`.
- After the corrected `07306` promotion, the current official full-set metrics
  are: `01184=1.693/2.483` pass, `09639=1.986/3.293` fail,
  `05452=1.448/1.997` fail, `05117=2.123/3.153` fail,
  `06127=2.034/3.585` pass, `07136=2.547/3.120` fail,
  `07306=4.315/5.256` fail, `06188=2.079/3.427` fail,
  `06145=0.711/1.549` pass, `06830=1.337/2.498` pass, with average
  `2.027/3.036` fail. The remaining method target is still the full
  10-sample table plus the average, not any single diagnostic sample.
- 2026-07-14 clarification: the optimization target must not be narrowed to
  `05117`. The final success criterion is all 10 requested samples plus the
  average CD/EMD thresholds in the table above. `05117` is only a fast
  diagnostic for candidate quality and registration behavior.
- `05117` same-image Hunyuan seeds `103`, `104`, `105`, and `106` were
  generated and registered from the accepted image. Seed103 had the best
  self-supervised full render score and was promoted, but all candidates still
  failed the 2D gate and the final prediction source remained MoGe. Official
  `05117` metric after promotion was `2.123/3.144` in the full-set check
  (`2.124/3.189` in the single-sample run, metric variance from resampling),
  still failing the `1.36/2.20` target. This reinforces that same-image seed
  expansion alone is not sufficient as a method-level fix.
- Current official full-set metrics after the `05117` seed103 promotion are:
  `01184=1.694/2.477` pass, `09639=1.994/3.295` fail,
  `05452=1.443/1.999` fail, `05117=2.123/3.144` fail,
  `06127=2.038/3.614` pass, `07136=2.522/3.078` fail,
  `07306=4.311/5.260` fail, `06188=2.078/3.430` fail,
  `06145=0.708/1.527` pass, `06830=1.337/2.507` pass, with average
  `2.025/3.033` fail against the requested average target `1.74/2.88`.
- 2026-07-14 leakage strictness update: `score_depth_render` now records
  `target_leakage` and `leakage_chamfer_px`, penalizes leakage more strongly
  in both the render score and the 2D gate-aware retry objective, and the hard
  2D gate checks `target_leakage <= 0.08` and
  `leakage_chamfer_px <= 12.0` in addition to the original leakage ratio. This
  is a simple geometric scoring change aimed at rejecting candidates with
  small-area but distant protrusions.
- Focused `01184` rerun after the leakage update selected
  `retry_candidate_22` instead of the previous visibly worse candidate. The
  final 2D score improved to `iou=0.940`, `coverage=0.952`,
  `leakage=0.014`, `target_leakage=0.014`, and
  `leakage_chamfer_px=2.79`; the official metric improved to
  `CD/EMD x1e2 = 0.966/1.489`. Output paths were overwritten in
  `workspace/redwood_depthfirst_semantic_full_rerun_20260713/01184/`.
- Follow-up batch diagnostics showed the new `target_leakage <= 0.08` hard
  gate was too strict for near-boundary overfill: `09639` became
  `2.371/4.333`, and `05452` with seed102 stayed at `1.444/2.005`, both
  failing the gate on leakage/target_leakage and falling back to MoGe despite
  good coarse 2D overlap. Keep the stronger leakage penalties in scoring and
  retry ranking, but relax the hard gate to `leakage <= 0.16` and
  `target_leakage <= 0.18`, while retaining `leakage_chamfer_px <= 12.0` to
  reject distant protrusions such as the visually flipped `01184` failure mode.
- Verification after relaxing only the hard leakage gate: `01184` remains fixed
  and improves to `CD/EMD x1e2 = 0.945/1.490` with gate pass and
  `leakage=0.013`, `target_leakage=0.013`, `leakage_chamfer_px=2.77`.
  `09639` now passes the 2D gate and uses complete geometry, but still fails at
  `2.219/3.815`. Candidate metrics show seed selection is not the bottleneck:
  selected/seed101 are both about `2.22/3.81`, seed102 is worse
  (`2.36/4.31`), complete-only is worse (`2.61/3.92`), and raw partial is
  `2.36/4.28`. The next simple method-level work should therefore target the
  3D partial-space refinement/objective rather than further leakage-gate
  tuning or final fusion thresholds.
- Partial-preserving threshold diagnostics on the six failing samples show that
  changing only the final far-point threshold cannot solve the target. Best
  observed values in the sweep were still failing for the hard samples:
  `09639` complete threshold `0.04` gave `1.862/3.309`, `05117` MoGe threshold
  `0.02` gave `2.124/3.192`, `07136` MoGe threshold `0.02` gave
  `2.531/2.985`, `07306` complete threshold `0.005` gave `4.270/5.273`, and
  `06188` MoGe threshold `0.04` gave `2.027/3.623`. `05452` is near threshold
  but still mixed: complete threshold `0.08` gave `1.122/1.790`, while MoGe
  threshold `0.08` gave `1.286/1.652`.
- Added an optional continuous robust MoGe-to-raw-partial bridge refinement
  after the RANSAC/Umeyama bridge. It optimizes a small delta-Sim3 with a
  trimmed Huber residual on pixel-bridge 3D correspondences and accepts only
  when robust residual improves. Unit coverage was added. A focused `05117`
  run accepted the bridge refinement at the correspondence level, but final
  metric did not improve (`2.125/3.181` versus default rerun
  `2.125/3.192`, both worse/no better than the prior `2.123/3.153` baseline).
  Therefore `moge_bridge_refine_iterations` is `0` by default and the feature
  remains an ablation, not a default method.
- A focused final-MoGe continuous alignment diagnostic was run for `05117` in
  `workspace/redwood_depthfirst_semantic_full_rerun_20260713/_diagnostic_05117_moge_final_icp`.
  It optimized only the final MoGe-source geometry against the raw partial
  with trimmed Sim3 ICP / Open3D point-to-point ICP before partial-preserving
  fusion. The best CD improved only slightly while EMD worsened:
  baseline `2.126/3.175`, trimmed Sim3 best about `2.106/3.190`, and Open3D
  ICP best CD about `2.089/3.237`. This confirms that `05117` is not primarily
  limited by final MoGe-to-partial rigid/Sim3 alignment. Further method work
  should focus on better complete/MoGe candidate geometry or self-supervised
  source/candidate quality, not additional final rigid ICP on this sample.
- A `05117` hybrid final-source diagnostic was run in
  `workspace/redwood_depthfirst_semantic_full_rerun_20260713/_diagnostic_05117_hybrid_source`.
  It kept the current partial+MoGe final source and added Hunyuan complete
  points only when they were far from both partial and MoGe points. This also
  failed: the current default-like MoGe-only result stayed around
  `2.123/3.174`, while adding complete points consistently worsened the metric
  unless almost no complete points were kept. This rules out a simple hybrid
  source fix for `05117`; the current Hunyuan complete geometry is not useful
  for this sample's final metric.
- A `05117` final-MoGe continuous affine diagnostic was run in
  `workspace/redwood_depthfirst_semantic_full_rerun_20260713/_diagnostic_05117_moge_affine`.
  It reused the existing continuous PCA-affine partial objective directly on
  the final MoGe source in raw partial coordinates. The optimized residual to
  the observed partial decreased slightly, but the final CD/EMD did not move
  meaningfully: baseline `2.123/3.176`; tested affine variants were around
  `2.118-2.125 / 3.165-3.201`. This further supports that `05117` is not
  bottlenecked by a residual final Sim3/affine alignment of the current MoGe
  source.
- Historical `workspace/redwood_qwen_studio_bg_full_rerun_20260713/05117`
  remains a useful control: remeasured with the current metric code, its
  `05117_fused.ply` is `1.242/1.690`, and its aligned complete cloud is
  `1.243/1.686`. Its 2D gate was accepted with strong values
  (`iou=0.896`, `coverage=0.975`, `leakage=0.083`,
  `edge_iou=0.083`, `edge_chamfer_px=7.75`). This confirms that `05117` is
  reachable when the generated/candidate geometry and image-space alignment
  are good; the current frozen image batch's `05117` failure is not solved by
  final continuous Sim3/affine/ICP postprocessing.
- Mapping that historical good `05117` complete cloud into the current MoGe
  frame shows that the self-supervised render/depth objective would recognize a
  good candidate if it were present: current complete candidate objective
  `2.101`, historical candidate objective `2.641`, with depth MAE improving
  from about `0.125` to `0.017`. A small current-MoGe coordinate refinement
  using the render-score objective improved the historical candidate to
  `1.100/1.692` under formal metric. This separates the issues: continuous
  image-space refinement can help a near-good candidate, but the current
  frozen-image Hunyuan candidates for `05117` do not contain such a candidate.
- A registration-only loop completed `01184` and `05117`, then was stopped
  during `05452` after the process became too slow. Continue with smaller
  single-sample runs or a faster runner; do not treat the interrupted `05452`
  run as verified.

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

Current Qwen completion direction:
- Use Qwen-Image-Edit-2511 with `QwenImageEditPlusPipeline`.
- Use two edit stages by default. Stage 1 completes the projected incomplete
  depth image as a depth-like image while preserving 2D projection, contour,
  pose, orientation, and camera viewpoint. Stage 2 translates the completed
  depth-like image into a realistic RGB/semantic image with a clean ordinary
  photography-studio background.
- Current default inference steps are `16`.
- Current CFG settings are `true_cfg_scale=4.0` and `negative_prompt=" "`.
- Do not pass `height` or `width`; the Plus pipeline outputs 1024x1024 from the
  512x512 depth input, then the final image is resized to 512x512.
- Qwen now runs an optional second refinement stage from the first completed
  semantic/RGB image. The second stage keeps only object outline, size,
  category, orientation, pose, and camera viewpoint, while making the object
  more realistic and keeping a clean ordinary photography-studio background.
- Default projection is the original `view_select` path, which selects the
  camera with the most visible partial points. The semantic view candidate
  preview/selection experiment has been removed from the default pipeline.

Redwood Stage 1 preview generated with this default projection:
- Output root:
  `workspace/redwood_stage1_qwen_single16_view_select_preview`
- Samples: `01184`, `05117`, `05452`
- Per-sample outputs: `depth.png`, `img.png`, `qwen_edit_prompt.txt`,
  `point_uv.npy`, `camera.pth`, `viewpoint.npy`
- Contact sheet:
  `workspace/redwood_stage1_qwen_single16_view_select_preview/redwood_stage1_depth_vs_qwen_preview.png`
- Status: generated and verified as 512x512 outputs; awaiting user visual
  acceptance.
- Update: `05117` prompt label override changed from `chair` to `red chair`
  through `configs/config.yaml::prompt_overrides`, then `05117` Stage 1 was
  regenerated in the same output root.
- `depth_view_point_cloud.ply` is not needed by the current index bridge and
  should not be saved by default; `save_depth_view_point_cloud` is now `false`.
- Partial-to-MoGe registration was run for `01184`, `05117`, and `05452` using
  raw partial point clouds from `data/<sample>.ply`, MoGeV2 on `img.png`, and
  RMBG-2.0 object masks. Summary:
  `workspace/redwood_stage1_qwen_single16_view_select_preview/redwood_moge_to_raw_partial_summary.csv`
- MoGe object extraction erodes the RMBG object mask by default
  (`object_mask_erode_pixels=2`) before filtering MoGe points. This removes
  edge/background points such as the extra environment points seen in `01184`
  while keeping the partial-to-MoGe registration stable.
- Two-stage Qwen refinement preview was generated for all default Redwood
  samples in:
  `workspace/redwood_stage1_qwen_refine_preview`
- The full contact sheet is:
  `workspace/redwood_stage1_qwen_refine_preview/redwood_qwen_refine_all_preview.png`
- Each sample keeps `qwen_edit_stage1.png` at 1024x1024 and final `img.png` at
  512x512. The `06127` category override experiment was reverted; it uses the
  dataset label `a vase with leafy plant`.

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

Redwood `01184` experiment:
- Input image:
  `workspace/redwood_stage1_qwen_refine_preview/01184/img.png`
- RMBG image for Hunyuan:
  `workspace/redwood_stage1_qwen_refine_preview/01184/img_sam.png`
- Hunyuan complete point cloud:
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_hunyuan2.1.ply`
- Status: generated and verified as 100000 points.
- Decision: Hunyuan3D-2.1 should only keep the final sampled `.ply` in the
  main workflow. Intermediate `_shape.glb` and final `.glb` files are not
  retained by default.
- Output cleanup decision: default runs now use `outputs.keep_profile: lean`
  with `outputs.save_intermediates: false`. The lean profile keeps core Stage 1
  files, `img_sam.png`, final Hunyuan PLY, MoGe object/index/transform metadata,
  masked FreeReg outputs, and final fused/inspection PLYs. Debug previews,
  unmasked FreeReg outputs, raw MoGe projection arrays, hits-only PLYs, stage
  Qwen images, and GLB files are removed unless `outputs.save_intermediates:
  true` or `outputs.keep_profile: debug` is set.

## Stage 3 - MoGe to Complete Registration

Status: integrated into the main pipeline with no-FreeReg render-to-MoGe Sim3.
The fixed-uv/Sim3 F-FreeReg path remains available as a historical experiment,
but `configs/config.yaml` now defaults to `reg_backend: render_to_moge_sim3`.

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

Current default implementation:
- `ScaleAdapter.render_to_moge_sim3_reg` runs the MoGe-to-raw-partial bridge
  and then `scripts/run_render_to_moge_sim3.py`.
- The complete point cloud is aligned directly to MoGe by Sim3 candidate
  search, z-buffer depth/silhouette scoring, coordinate-search refinement, and
  visible trimmed ICP.
- FreeReg and DepthPro are not used by the default main pipeline.
- Update after overlay inspection: render-to-MoGe scoring now emphasizes 2D
  silhouette and edge alignment more strongly, and visible ICP is accepted only
  when it preserves or improves the render-to-MoGe 2D score. If ICP lowers that
  score, the saved transform rolls back to the coordinate-search result.
- Differentiable 2D silhouette optimization is now inserted after
  coordinate-search refinement. It optimizes a small delta-Sim3 with soft point
  splatting against the object mask and accepts the result only if the
  full-resolution hard render score improves.
- Update after off-view inspection of `01184`: pure 2D silhouette alignment can
  look correct from the MoGe camera while still leaving a 3D rotation/depth
  residual from another viewpoint. The render-to-MoGe path now adds a
  differentiable visible-3D delta-Sim3 refinement after 2D silhouette
  optimization. It matches the visible complete surface to MoGe object points,
  keeps a soft 2D silhouette term, and accepts only when visible 3D distance
  improves without exceeding the hard 2D render-score drop tolerance.
- Final complete-to-partial composition now has an optional conservative
  partial-space delta-Sim3 refinement. This handles small residual offsets in
  the MoGe-to-partial bridge by fitting complete points to the raw partial
  point cloud with trimmed nearest-neighbor 3D distances. It updates only the
  final `complete_to_partial`/`fused` outputs and is accepted only when partial
  distance improves and the delta stays within small scale/rotation/translation
  bounds.
- Previous notes about `06127` background contamination are stale; use the
  current artifacts and measured gate/metric values when judging that sample.

Open work:
- Inspect the no-FreeReg fused visualizations and overlays for high-metric
  samples, especially `06127`, `09639`, `06188`, `07306`, and `07136`.
- Default Redwood target set is now `01184`, `09639`, `05452`, `05117`,
  `06127`, `07136`, `07306`, `06188`, `06145`, and `06830`.
- Rerun the Redwood batch with the updated 2D score/ICP rollback and compare
  overlays plus CD/EMD against the previous no-FreeReg run.
- Rerun the Redwood batch with differentiable silhouette optimization enabled
  and inspect which samples accept the optimized delta-Sim3.
- Rerun at least `01184` with visible-3D and partial-space refinement enabled,
  inspect `visible_3d_optimization` and `partial_refinement` metadata, then run
  the full Redwood batch and metric if the smoke result is stable.
- 2026-07-13 requested full Redwood rerun: regenerate Stage 1 images, Hunyuan
  PLYs, render-to-MoGe registration, and metric under
  `workspace/redwood_qwen_studio_bg_full_rerun_20260713` using the current
  pipeline and Qwen prompts constrained to a clean ordinary photography-studio
  background.
- 2026-07-13 `06188` prompt override changed to `red motorcyle` and the sample
  was rerun in `workspace/redwood_qwen_studio_bg_full_rerun_20260713` by user
  request. New metric is `CD-L1 x1e2 = 5.186385` and
  `EMD x1e2 = 6.571867`, slightly worse than the previous same-root `06188`
  result (`5.018309` / `6.483270`). The current 2D registration gate fails on
  `iou = 0.749277 < 0.78` and `leakage = 0.153091 > 0.12`, so partial-space
  refinement is skipped. Inspect/fix MoGe bridge or complete-to-MoGe 2D
  alignment before treating this sample as an accepted result.
- 2026-07-13 `07136` showed that the previous 2D gate was still too loose:
  the run passed with `iou = 0.799961`, `coverage = 0.827981`,
  `leakage = 0.040587`, and `edge_chamfer_px = 24.442658`, then partial-space
  anisotropic refinement was accepted despite visibly poor 2D boundary
  alignment. The gate now makes edge metrics hard thresholds and tightens the
  main thresholds to `iou >= 0.82`, `coverage >= 0.84`, `leakage <= 0.10`,
  `edge_iou >= 0.025`, and `edge_chamfer_px <= 18.0`. A registration-only
  probe in
  `workspace/redwood_qwen_studio_bg_full_rerun_20260713/07136_strict_gate_probe`
  found a better retry candidate (`iou = 0.879749`, `coverage = 0.899454`,
  `leakage = 0.024298`, `edge_iou = 0.028900`) but still rejected it because
  `edge_chamfer_px = 19.595064 > 18.0`; partial refinement was correctly
  skipped with `registration_2d_threshold_not_met`.
- 2026-07-13 retry coordinate search now optimizes `score_2d_gate_objective`
  inside the gate-failure retry branch, while the main coordinate-search path
  still uses the normal render score. This keeps the default path unchanged but
  makes retry candidates refine toward the same 2D gate used for acceptance.
  On `05117`, the strict-gate registration-only probe at
  `workspace/redwood_qwen_studio_bg_full_rerun_20260713/05117_strict_gate_probe`
  passed the stricter gate with `iou = 0.896573`, `coverage = 0.975105`,
  `leakage = 0.082424`, `edge_iou = 0.082445`, and
  `edge_chamfer_px = 7.722045`; partial anisotropic refinement was accepted.
  Its metric is `CD-L1 x1e2 = 1.241699` and `EMD x1e2 = 1.684162`, still below
  the user target `1.36 / 2.20`.
- The same retry-objective probe on `07136` improved the metric from the
  same-root old `07136` run (`CD-L1 x1e2 = 7.000`, `EMD x1e2 = 10.018`) to
  `CD-L1 x1e2 = 5.126419` and `EMD x1e2 = 6.377791` at
  `workspace/redwood_qwen_studio_bg_full_rerun_20260713/07136_retry_gate_objective_probe`.
  It still failed the strict 2D gate only on `edge_chamfer_px`
  (`19.427386 > 18.0`), so partial-space refinement was skipped. A heavier
  top-K/finer-step retry probe was stopped because it was too slow for the
  intended simple academic pipeline; the next useful optimization for `07136`
  is a concise continuous boundary-aware loss, not more search breadth.
- 2026-07-13 `07136` generation diagnostic: overriding the category to
  `leather sofa` and using a depth-first Qwen prompt made the generated
  `img.png` pose closer to the input depth image than the previous direct
  photo prompt, but Hunyuan/registration still failed the strict 2D edge gate
  (`edge_iou = 0.01248 < 0.025`, `edge_chamfer_px = 19.04 > 18.0`), so
  partial refinement was skipped. The metric for the current main `07136`
  output under `workspace/redwood_qwen_studio_bg_full_rerun_20260713/07136`
  is `CD-L1 x1e2 = 5.923551`, `EMD x1e2 = 5.357981`, which is worse than the
  target `1.58 / 2.78` and worse in CD than the previous retry-objective probe.
  Treat this as a generation-side diagnostic, not an accepted result.
- Regenerate `06127` Stage 1 image/mask before trusting its Stage 2/metric
  result.
- Tune per-category candidate parameters if visual inspection still shows
  systematic orientation or scale failures.
- Keep the FreeReg adaptive path as a comparison baseline, not the default.
- 2026-07-13 semantic-feature probe: DINOv2-large image patch features can now
  be precomputed and used to re-rank FreeReg image/point-cloud matches for
  visualization without changing the selected transform. This is still an
  inspection experiment, not a main-flow registration criterion.
- 2026-07-13 anisotropic partial-refinement Redwood rerun: reused existing
  `img.png`, Hunyuan PLYs, and MoGe-to-partial bridge outputs under
  `workspace/redwood_stage1_qwen_refine_preview`, reran render-to-MoGe Sim3
  registration with `partial_refine_mode=pca_anisotropic` for all 10 default
  samples, and wrote final predictions to
  `workspace/redwood_stage1_qwen_refine_preview/<sample>/<sample>_fused.ply`.
  Metrics were saved to
  `workspace/redwood_stage1_qwen_refine_preview/metrics_samples.csv` with mean
  `CD-L1 x1e2 = 4.280061` and `EMD x1e2 = 5.799900`. Highest-error samples are
  still `09639`, `05117`, `06188`, `06127`, and `07136`; inspect overlays and
  Stage 1 masks before treating those as accepted outputs.
- 2026-07-13 2D acceptance gate: render-to-MoGe registration now records
  `registration_2d_acceptance` and requires `final_score.iou >= 0.82`,
  `final_score.coverage >= 0.84`, `final_score.leakage <= 0.10`,
  `final_score.edge_iou >= 0.025`, and
  `final_score.edge_chamfer_px <= 18.0` before running partial-space
  refinement. Edge metrics are hard gates because `07136` showed that coarse
  IoU/coverage/leakage can pass while the 2D boundary is still visibly wrong.
  This explicitly rejects cases like `05117`, where the 2D render alignment is
  poor despite later partial-distance improvement.
- After adding the gate, `05117` was rerun and correctly failed
  `registration_2d_acceptance` on `iou` and `leakage`; partial refinement was
  skipped. Its current metric is `CD-L1 x1e2 = 5.295647` and
  `EMD x1e2 = 7.239970`, and the metrics CSVs were updated for that sample.
- The gate-failure path now retries top-K candidates with a 2D-focused
  objective and wider scale multipliers. On `05117`, this selected
  `retry_candidate_73` and improved `final_score` to `iou = 0.727344`,
  `coverage = 0.841554`, `leakage = 0.157247`, but it still fails the hard 2D
  gate on `iou` and `leakage`. The current `05117` metric is
  `CD-L1 x1e2 = 4.018119` and `EMD x1e2 = 5.561677`. This is better but still
  not a visually accepted 2D registration; likely next work is Stage 1/Hunyuan
  geometry regeneration or non-Sim3 deformation/shape correction, not simply
  more partial-space refinement.
- 2026-07-13 bridge-anchor continuous optimization: render-to-MoGe registration
  now loads `<sample>_moge_to_raw_partial_partial_to_moge_index.npy`, maps valid
  raw partial points back into the MoGe frame with `moge_to_partial^-1`, and
  uses those points as bridge anchors. Retry candidates record anchor distance
  and are ranked with a joint 2D+anchor objective. A continuous delta-Sim3
  optimizer then fits complete points to bridge anchors with a soft 2D
  silhouette guard; it is accepted only if anchor distance improves and the 2D
  objective stays within tolerance. On `05117`, the bridge file had 56,385
  valid matches out of 56,582 partial points (`match_ratio = 0.9965`), but the
  continuous candidate was correctly rejected because it reduced leakage while
  dropping IoU/coverage and worsening anchor mean distance. Final output stayed
  on `retry_candidate_73`; current one-off metric is `CD-L1 x1e2 = 4.007` and
  `EMD x1e2 = 5.533`.

Redwood `01184` original F-FreeReg experiment:
- Image input:
  `workspace/redwood_stage1_qwen_refine_preview/01184/img.png`
- Complete point cloud input:
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_hunyuan2.1.ply`
- DepthPro image point cloud:
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_freereg_original_depthpro_image_points.ply`
- Complete registered to DepthPro image frame:
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_freereg_original_depthpro_complete_registered_to_image.ply`
- Fused visualization:
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_freereg_original_depthpro_gray_image_blue_complete_fused.ply`
- Metadata:
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_freereg_original_depthpro_info.json`
- Status: generated; original F-FreeReg produced 112 YOHO descriptor matches.
- MoGe object extraction for the same image was also generated with RMBG erode
  2 pixels:
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_moge_to_raw_partial_moge_object_only.ply`
- Issue found: the unmasked DepthPro image point cloud includes ground/background
  points because original F-FreeReg backprojects the whole image.
- Masked rerun: `scripts/run_freereg_original_depthpro.py` now supports an
  RMBG object mask so DepthPro backprojection keeps only object pixels.
- Masked `01184` outputs:
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_freereg_original_depthpro_objectmask_object_depthpro_points.ply`
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_freereg_original_depthpro_objectmask_complete_registered_to_object_depthpro.ply`
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_freereg_original_depthpro_objectmask_gray_object_depthpro_blue_complete_fused.ply`
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_freereg_original_depthpro_objectmask_info.json`
- Masked rerun status: generated; object DepthPro target has 49578 points and
  original F-FreeReg produced 396 YOHO descriptor matches.
- The current `01184` experiment directory was cleaned with the lean profile.
  Retained files are:
  `depth.png`, `img.png`, `camera.pth`, `point_uv.npy`,
  `qwen_edit_prompt.txt`, `img_sam.png`, `01184_hunyuan2.1.ply`,
  `01184_moge_to_raw_partial_moge_object_only.ply`,
  `01184_moge_to_raw_partial_partial_to_moge_index.npy`,
  `01184_moge_to_raw_partial_moge_to_raw_partial_transform.npy`,
  `01184_moge_to_raw_partial_info.json`,
  `01184_moge_to_raw_partial_object_mask.png`,
  `01184_moge_to_raw_partial_raw_partial_gray_moge_red_aligned.ply`,
  `01184_freereg_original_depthpro_objectmask_complete_registered_to_object_depthpro.ply`,
  `01184_freereg_original_depthpro_objectmask_gray_object_depthpro_blue_complete_fused.ply`,
  and `01184_freereg_original_depthpro_objectmask_info.json`.
- `01184` fixed-uv/Sim3 FreeReg probe:
  `scripts/run_freereg_original_depthpro.py` was updated to project YOHO image
  keypoints to uv directly and to apply FreeReg's estimated scale as a Sim3
  transform. This reduced the complete-vs-DepthPro bbox-center z offset from
  about `0.426` to `0.018` and reduced complete-to-DepthPro nearest-neighbor
  mean distance from about `0.393` to `0.176`.
- Fixed-uv/Sim3 outputs:
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_freereg_original_depthpro_fixeduv_sim3_gray_object_depthpro_blue_complete_fused.ply`
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_freereg_original_depthpro_fixeduv_sim3_complete_registered_to_object_depthpro.ply`
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_freereg_original_depthpro_fixeduv_sim3_info.json`
- A bbox-center translation-only probe was also generated:
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_freereg_original_depthpro_fixeduv_sim3_bboxshift_gray_object_depthpro_blue_complete_fused.ply`
  It aligns the bounding-box centers but does not materially improve
  nearest-neighbor distance, so it should remain a visualization/debug probe
  rather than a final registration step.
- FreeReg source used by `scripts/run_freereg_original_depthpro.py` is now
  vendored under `third_party/FreeReg`. The vendored code contains source only;
  checkpoints are intentionally excluded and can be provided through
  `FREEREG_DEPTHPRO_CKPT`, `FREEREG_FCGF_CKPT`, and `FREEREG_YOHO_CKPT`.
- Adaptive `ir_3d` rerun on the Redwood batch:
  - `01184`, `05117`, `05452`, `06127`, `07306`, and `09639` selected the
    original auto threshold.
  - `06145`, `06188`, `06830`, and `07136` selected `fallback_0.1`.
  - Sampled CPU Chamfer x1e2 mean improved from about `19899.46` to `14.00`.
  - Summary:
    `workspace/redwood_stage1_qwen_refine_preview/freereg_adaptive_ir3d_cpu_cd_summary.csv`
- `06145` DINOv2 semantic-match visualization probe:
  - Feature extractor:
    `scripts/extract_dinov2_image_features.py`
  - FreeReg wrapper:
    `scripts/run_freereg_original_depthpro.py`
  - DINOv2-large image feature grid:
    `workspace/redwood_stage1_qwen_refine_preview/06145/06145_dinov2_large_img_features.npz`
  - Original match-line figure:
    `workspace/redwood_stage1_qwen_refine_preview/06145/06145_freereg_adaptive_ir3d_image_pointcloud_match_lines.png`
  - DINO semantic-filtered match-line figure, `min_similarity=0.55`, 8 valid
    matches:
    `workspace/redwood_stage1_qwen_refine_preview/06145/06145_freereg_adaptive_ir3d_dino_semantic_match_lines.png`
  - DINO semantic-filtered match-line figure, `min_similarity=0.35`, 12 valid
    matches:
    `workspace/redwood_stage1_qwen_refine_preview/06145/06145_freereg_adaptive_ir3d_dino035_dino_semantic_match_lines.png`
  - Combined comparison:
    `workspace/redwood_stage1_qwen_refine_preview/06145/06145_freereg_dino_match_comparison.png`
  - Risk/decision: DINO currently re-ranks matches after the selected transform
    and is only a diagnostic visualization. It should not become a registration
    criterion until it is tested as an inlier/candidate scoring term across
    multiple object categories.
- `06145` projected-silhouette candidate-selection probe:
  - FreeReg wrapper:
    `scripts/run_freereg_original_depthpro.py`
  - Candidate mode:
    `--candidate-selection silhouette`
  - Candidate thresholds:
    `auto, 0.08, 0.1, 0.15, 0.2, 0.3`
  - Selected candidate:
    `fallback_0.08`
  - Selected candidate silhouette score:
    `score=0.4198`, `IoU=0.3766`, `coverage=0.7759`,
    `leakage=0.5774`, `edge_chamfer_px=31.01`
  - FreeReg fused visualization:
    `workspace/redwood_stage1_qwen_refine_preview/06145/06145_freereg_silhouette_select_gray_object_depthpro_blue_complete_fused.ply`
  - Complete-to-partial fused visualization:
    `workspace/redwood_stage1_qwen_refine_preview/06145/06145_complete_to_partial_silhouette_select_raw_partial_gray_complete_blue_aligned.ply`
  - Silhouette overlay:
    `workspace/redwood_stage1_qwen_refine_preview/06145/06145_freereg_silhouette_select_silhouette_overlay.png`
  - Adaptive-vs-silhouette overlay comparison:
    `workspace/redwood_stage1_qwen_refine_preview/06145/06145_adaptive_vs_silhouette_overlay_comparison.png`
  - Metric summary:
    `workspace/redwood_stage1_qwen_refine_preview/06145/06145_silhouette_select_metric_summary.json`
  - Same-seed metric comparison against `data/GT/06145.ply`, sampled with
    `metric_num_points=16384`, `metric_seed=1184`:
    adaptive `CD-L1 x1e2=9.3714`, `EMD x1e2=9.0747`; silhouette-select
    `CD-L1 x1e2=5.2366`, `EMD x1e2=6.5815`.
  - Risk/decision: silhouette selection improved `06145`, but leakage remains
    high. Test across more categories before promoting it from probe to default
    candidate selection.

## Stage 4 - Complete Back to Partial

Status: integrated into `main.py` through `ScaleAdapter` when
`reg_backend: render_to_moge_sim3`.

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
- Add broader tests around the `ScaleAdapter.render_to_moge_sim3_reg` adapter
  once heavy MoGe calls can be mocked cleanly.
- Inspect and tune the high-metric samples from the 2026-07-13 no-FreeReg run.

Redwood `01184` composed output:
- Complete aligned to raw partial:
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_complete_to_partial_depthpro_moge_complete_aligned_to_raw_partial.ply`
- Fused visualization:
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_complete_to_partial_depthpro_moge_raw_partial_gray_complete_blue_aligned.ply`
- Transform:
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_complete_to_partial_depthpro_moge_complete_to_raw_partial_transform.npy`
- Metadata:
  `workspace/redwood_stage1_qwen_refine_preview/01184/01184_complete_to_partial_depthpro_moge_info.json`
- Metric against `data/GT/01184.ply`, sampled with `metric_num_points=16384`
  and `metric_seed=1184`: `CD-L1 x1e2 = 1.745359`, `EMD x1e2 =
  2.428691`.
- Metric reporting now uses `x1e2` columns and printed labels in `main.py`.

Redwood batch run on 2026-07-13:
- Samples:
  `01184`, `05117`, `05452`, `06127`, `06145`, `06188`, `06830`, `07136`,
  `07306`, `09639`.
- Registration summary:
  `workspace/redwood_stage1_qwen_refine_preview/redwood_complete_to_partial_registration_summary.csv`
- Metric summary:
  `workspace/redwood_stage1_qwen_refine_preview/redwood_complete_to_partial_metrics.csv`
- Normal-scale outputs:
  `01184`, `05117`, `05452`, `07306`, `09639`.
- Suspicious or failed outputs:
  `06127` has low DepthPro-to-MoGe inlier ratio and higher metric;
  `06145`, `06188`, `06830`, and `07136` have large complete-to-partial
  translations caused by unstable FreeReg results.
- Adaptive FreeReg update:
  `06145`, `06188`, `06830`, and `07136` no longer have random hundreds-scale
  complete-to-partial translations. The adaptive outputs use prefix
  `<sample>_complete_to_partial_adaptive_ir3d`. Sampled CPU Chamfer x1e2:
  `06145=7.005`, `06188=10.698`, `06830=15.149`, `07136=18.590`.
  The full comparison is:
  `workspace/redwood_stage1_qwen_refine_preview/freereg_adaptive_ir3d_cpu_cd_summary.csv`.
- Core method documentation:
  `docs/core_registration_pipeline.md`.

Redwood no-FreeReg main-pipeline run on 2026-07-13:
- Command:
  `CUDA_VISIBLE_DEVICES=0 /opt/data/private/cr/miniconda3/envs/genpc/bin/python main.py --workspace workspace/redwood_stage1_qwen_refine_preview --skip_existing --save_intermediates`
- Default backend:
  `render_to_moge_sim3`
- Samples:
  `01184`, `05117`, `05452`, `06127`, `06145`, `06188`, `06830`, `07136`,
  `07306`, `09639`.
- Per-sample outputs:
  `<sample>_complete_registered_to_moge.ply`,
  `<sample>_moge_gray_complete_blue_fused.ply`,
  `<sample>_complete_aligned_to_raw_partial.ply`,
  `<sample>_raw_partial_gray_complete_blue_aligned.ply`,
  `<sample>_complete_to_moge_transform.npy`,
  `<sample>_complete_to_partial_transform.npy`,
  `<sample>_render_to_moge_overlay.png`,
  `<sample>_render_to_moge_sim3_info.json`,
  and metric prediction `<sample>_fused.ply`.
- Metric summary:
  `workspace/redwood_stage1_qwen_refine_preview/metrics_samples.csv`
- Mean metrics:
  `CD-L1 x1e2 = 4.198972`, `EMD x1e2 = 4.746163`.
- Per-sample `CD-L1 x1e2 / EMD x1e2`:
  `01184=2.476381/2.828803`,
  `05117=4.696598/5.814887`,
  `05452=1.503627/1.928873`,
  `06127=8.357799/8.122091`,
  `06145=1.472761/2.026613`,
  `06188=4.973479/6.414759`,
  `06830=2.431914/3.379773`,
  `07136=4.854884/4.702779`,
  `07306=4.865654/5.496590`,
  `09639=6.356620/6.746463`.
- Overlay-protection rerun after strengthening 2D silhouette/edge scoring and
  rolling back ICP score drops:
  all 10 samples rejected visible ICP because it lowered render-to-MoGe 2D
  score. Updated overlays and `<sample>_fused.ply` were written in the same
  sample directories. Mean metric worsened to `CD-L1 x1e2 = 6.427336`,
  `EMD x1e2 = 7.728549`, so this change protects 2D overlay but is not yet a
  better 3D metric setting.
- Overlay-protection per-sample `CD-L1 x1e2 / EMD x1e2`:
  `01184=5.257216/5.661427`,
  `05117=5.300537/7.242264`,
  `05452=2.480401/3.567437`,
  `06127=7.886697/8.750742`,
  `06145=11.054984/13.021341`,
  `06188=8.373745/12.120620`,
  `06830=3.077107/4.447857`,
  `07136=6.110398/5.828304`,
  `07306=4.254048/5.096049`,
  `09639=10.478231/11.549450`.
- Differentiable silhouette smoke test:
  `01184` ran successfully with `render_size=128`, `max_points=12000`,
  `iterations=80`, and wrote `silhouette_optimization` metadata. The soft
  optimizer reduced its own loss, but the full-resolution hard render score did
  not improve (`candidate_score=1.479652` vs `baseline_score=1.541576`), so
  the candidate was rejected as intended.

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
