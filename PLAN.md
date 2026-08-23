# GenPC Active Plan

## Frozen baseline: Pixal guarded unified registration v15

Status: active and accepted as of 2026-08-23.  All later Hunyuan3D-MV,
dual-anchor, non-rigid deformation, posterior-fusion, and Omni generalization
routes are retired from the default pipeline.

The active zero-shot route is:

```text
Redwood partial point cloud
  -> saved-camera depth image
  -> frozen GPT ImageGen semantic/RGB completion
  -> frozen Pixal3D complete GLB + 100k surface PLY
  -> v8 GenPC/PCA proper-Sim(3) fallback candidate
  -> v12 GPU global SO(3) + visible Sim(3) TTT candidate
  -> v15 shared confidence gate + full-resolution do-no-harm guard
  -> registered complete Pixal3D body
```

Canonical output root:

`gpt_version/_pixal_guarded_unified_registration_v15_20260822`

Canonical implementation:

`scripts/select_pixal_guarded_unified_registration_v15.py`

Detailed method record:

`docs/fast_unified_registration_v15.md`

## Frozen inputs and dependencies

- Keep `gpt_version/<sample>/gpt_image.png`, `pixal3d.glb`, and
  `pixal3d_sampled_100k.ply` unchanged for all ten samples.
- Keep `_pixal_scale_ttt_v8_20260822` and
  `_pixal_batched_adaptive_ttt_v12_20260822`; v15 routes between them.
- Keep v10/v11 roots as reproducibility diagnostics for the GPU search lineage.
- Never overwrite v15 transforms, registered PLYs, registered meshes,
  projections, or info JSON files.

## Shared method contract

- One hypothesis lattice, scale set, objective, confidence gate, and render
  guard for all samples.
- No sample ID, category-specific registration rule, GT geometry, CD, or EMD
  during inference or routing.
- Proper rotation, one isotropic scale, and translation only.
- Retain all 100,000 Pixal points; no non-rigid deformation and no generated
  geometry deletion.
- Low confidence falls back to the v8 GenPC/PCA candidate.

## Verification

- Verify the ten accepted v15 registered PLY and info files exist.
- Run syntax/unit checks for the v11/v12/v15 implementation after rollback.
- Do not regenerate outputs unless explicitly requested; existing v15 outputs
  are the accepted prediction freeze.

## Paper target

The research target remains full-ten Redwood mean CD-L1 x1e2 below `1.74`
and EMD x1e2 below `2.88`, with zero-shot shared parameters.  v15 is the
current visual-quality baseline; later work must branch from it and cannot
replace it without a full-ten visual and post-freeze metric improvement.

## Experimental branch: guarded hierarchical residual registration

Status: two-case GT-free pilot complete on 2026-08-23; not promoted to the
canonical pipeline. The two cases are diagnostics only and do not define any
parameters, thresholds, or routing rules.

Branch: `codex/v15-hierarchical-residual-registration`

Implementation:

- `src/hierarchical_residual_registration.py`
- `scripts/run_pixal_hierarchical_residual_registration.py`
- `tests/test_hierarchical_residual_registration.py`

Pilot output:

`gpt_version/_pixal_hierarchical_residual_registration_multiscale_pilot_20260823`

The frozen v15 body is refined by a small proper-Sim(3) trust region and, only
when supported by a compact visible residual, an intrinsic mesh deformation.
The local displacement is projected out of the seven-dimensional similarity
nullspace and guarded by coverage, edge-stretch, face-flip, and visible 2D+3D
objectives. A pure global residual candidate is always retained; the lowest
passing GT-free candidate is selected, with exact v15 fallback.

Pilot routing and visible objective:

- `09639`: global residual Sim(3), `0.037437 -> 0.032842`;
- `07136`: hierarchical residual, `0.030215 -> 0.022343`, grid coverage
  `0.9615 -> 1.0000`.

All shared thresholds are identical across the two tests. No category, sample
ID, GT, CD, or EMD is available to inference or selection. Before promotion,
freeze the method and run all ten Redwood cases, visual review, timing, and
post-freeze CD/EMD.

## Side pilot: bidirectional cycle-consistent 2D+3D Sim(3)

Status: independent two-case pilot complete; not integrated or promoted.

Implementation:

- `src/bidirectional_cycle_registration.py`
- `scripts/run_pixal_bidirectional_cycle_registration.py`
- `tests/test_bidirectional_cycle_registration.py`

Pilot output:

`gpt_version/_pixal_bidirectional_cycle_registration_pilot_20260823`

This candidate fits partial-to-visible-Pixal, applies its strict inverse to the
complete body, and uses an independently estimated reverse map only as a cycle
witness. It improved the shared visible objective by 1.18% on `09639` and
3.71% on `07136`, while preserving proper isotropic Sim(3) and all 100k points.
It is currently weaker than the hierarchical residual route and should be
treated as a fast initialization/candidate rather than a replacement.

Full-ten forced-candidate audit requested on 2026-08-23:

`gpt_version/_pixal_bidirectional_cycle_registration_forced_audit_20260823`

This audit intentionally exports the best bidirectional candidate even when
the normal do-no-harm gate rejects it. It is for visual diagnosis only and
must not be used as the paper prediction root. With all parameters frozen, 5
of 10 candidates pass the normal gate. Mean visible-objective improvement is
1.25% and average runtime is 15.23 seconds per sample. The forced candidates
for `01184`, `06127`, `06145`, and `06188` worsen the shared objective; `05452`
improves the aggregate objective but still fails another safety condition.

Post-freeze official metric audit:

`gpt_version/_pixal_bidirectional_cycle_registration_forced_audit_20260823/postfreeze_cd_emd_20260823`

The ten predictions were frozen before GT evaluation. Both the forced
bidirectional candidates and v15 use the same 16,384-point FPS indices with
metric seed 6145. Mean CD-L1/EMD x1e2 is `2.0648/3.0786` for the forced
candidate and `2.1096/3.0961` for v15. Thus the audit candidate improves over
v15 slightly but does not beat the GenPC paper mean `1.74/2.88`. GT metrics
remain unavailable to inference and routing.
