# Fast Unified Pixal Registration v15

## Status

Implemented and evaluated on all ten target Redwood samples on 2026-08-22.
This is a GT-free registration-routing experiment, not yet a replacement for
the user-accepted v10 artifacts and not yet fused.

## Design

The unified method is a confidence-gated residual upgrade over the existing
GenPC/PCA Sim(3) registration:

```text
frozen Pixal body + raw partial + saved camera
  ├─ GenPC/PCA proper-Sim(3) candidate (robust fallback)
  └─ GPU-batched global SO(3) candidate (residual upgrade)
          └─ visible-depth + partial-surface local TTT
                 └─ observable confidence gate
                        └─ full-resolution do-no-harm guard
                               ├─ pass: global SO(3) upgrade
                               └─ fail: GenPC/PCA fallback
```

Every sample uses the same hypotheses, thresholds, and routing logic.  Sample
IDs, categories, GT geometry, CD, and EMD are not available to the router.

## Speed Changes

`scripts/run_pixal_batched_so3_sim3_ttt_v11.py` replaces 3240 sequential CPU
coarse evaluations (`648 rotations × 5 scales`) with chunked GPU z-buffer
rendering.  The coarse score uses silhouette IoU, coverage, leakage, and robust
visible-depth error.  Expensive KD-tree surface evidence is delayed until the
shortlist.

`scripts/run_pixal_batched_adaptive_ttt_v12.py` adds:

- one best coarse scale per unique rotation;
- 12 fine 3D candidates;
- at most eight local rotation candidates;
- evidence-based early stopping after a locally refined candidate satisfies
  all shared confidence thresholds.

The confidence thresholds are:

- low-resolution IoU `>= 0.85`;
- coverage `>= 0.90`;
- leakage `<= 0.08`;
- normalized visible-depth error `<= 0.10`;
- normalized partial-to-complete trim70 surface error `<= 0.012`.

The 3240-hypothesis coarse stage takes approximately 0.36--0.56 seconds on the
RTX 4090, compared with roughly a minute-scale sequential CPU stage in v10.
The remaining bottleneck is fine/local TTT, not global orientation coverage.

## Full-Ten v12 Timing and Confidence Audit

| sample | seconds | observable gate | full projection IoU |
| --- | ---: | :---: | ---: |
| `01184` | 151.4 | pass | 0.8416 |
| `05117` | 105.0 | pass | 0.9020 |
| `05452` | 113.0 | pass | 0.8945 |
| `06127` | 231.1 | fail | 0.9121 |
| `06145` | 130.8 | pass | 0.9215 |
| `06188` | 235.8 | fail | 0.6354 |
| `06830` | 121.6 | pass | 0.7685 |
| `07136` | 205.1 | pass | 0.8459 |
| `07306` | 122.7 | pass | 0.8673 |
| `09639` | 216.5 | fail | 0.6857 |

Mean runtime is 163.3 seconds/sample; median is 141.1 seconds/sample.  Easy or
well-observed cases stop early.  Low-confidence `06127`, `06188`, and `09639`
correctly run the complete local search.

## Negative Stratification Ablation

`scripts/run_pixal_stratified_adaptive_ttt_v13.py` tested a fixed shortlist of
eight global SO(3) candidates plus four protected unperturbed PCA candidates.
It did not improve `06188`.  The failure is therefore not simply omission of a
PCA orientation; image/depth/local-geometry evidence is internally weak or
conflicting.  Expanding the shortlist further is unlikely to be an efficient
general solution.

## Unified Router and Do-No-Harm Guard

`scripts/select_pixal_unified_registration_v14.py` first introduced the shared
confidence route.  A full-ten audit found that `07136` passed the low-resolution
gate but reduced full-resolution IoU and changed the already accepted scale.

`scripts/select_pixal_guarded_unified_registration_v15.py` therefore adds a
GT-free full-resolution guard.  Define:

```text
render_score = IoU + 0.15 * coverage - 0.45 * leakage
```

The fast candidate is eligible only when it passes the observable confidence
gate and its full-resolution render score is no more than `0.005` below the
GenPC/PCA candidate.  Otherwise the method falls back.  This is a shared
do-no-harm rule, not a sample-specific selector.

Full-ten v15 routing:

| sample | selected route |
| --- | --- |
| `01184` | GenPC/PCA fallback |
| `05117` | GenPC/PCA fallback |
| `05452` | GenPC/PCA fallback |
| `06127` | GenPC/PCA fallback |
| `06145` | fast global SO(3) TTT |
| `06188` | GenPC/PCA fallback |
| `06830` | fast global SO(3) TTT |
| `07136` | GenPC/PCA fallback |
| `07306` | GenPC/PCA fallback |
| `09639` | GenPC/PCA fallback |

The router automatically upgrades exactly the two cases whose prior
orientation was visually rejected (`06145` tabletop axes and `06830` global
pose), while preserving the other accepted poses and the `07136` scale.  The
router has no access to those user judgments; this correspondence is an
outcome of the shared evidence gates.

Outputs:

- v12 candidates:
  `gpt_version/_pixal_batched_adaptive_ttt_v12_20260822`;
- v15 routed outputs:
  `gpt_version/_pixal_guarded_unified_registration_v15_20260822`;
- v15 summary:
  `redwood_10_guarded_registration_summary.json` in the v15 root.

## Generalization Contract

- one hypothesis lattice, scale set, loss, confidence gate, and render guard;
- no category-upright assumptions or sample-ID conditions;
- no GT or final metrics in candidate generation or routing;
- proper rotation, one isotropic scale, and translation only;
- all 100000 frozen Pixal points retained;
- low confidence causes fallback or blocks fusion, never non-rigid correction,
  anisotropic scale, raw-partial substitution, or regenerated GLBs.

## Next Optimization

Do not reduce the 648-orientation coverage; its GPU cost is already negligible
and it is necessary for `06830`.  Optimize the fine/local stage instead:

1. batch fine silhouette/depth proposals on GPU;
2. cache camera-space rotated source points across scale/translation trials;
3. use GPU kNN only for the 12 shortlisted candidates;
4. evaluate candidate diversity before local TTT to remove near-duplicate
   rotations;
5. preserve the current evidence gates and fallback unchanged;
6. run post-freeze full-ten CD/EMD before promoting v15 to fusion.

Fusion remains blocked for any low-confidence registration.  For selected
outputs, fusion must follow the complete-Pixal-body invariants in
`docs/core_registration_pipeline.md`.
