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
