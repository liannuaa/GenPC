# GenPC Active Plan

## Active mainline

Status: accepted best-effect full-ten pipeline as of 2026-08-23. Predictions,
density-uniform outputs, and strict metric indices are frozen.

Canonical root:
`workspace/redwood_qwen_gpt_pixal_bidirectional_mainline_20260823`

Canonical method: `docs/core_registration_pipeline.md`.

## Completed stages

- [x] Copy Redwood-compatible `depth.png` and Qwen `img.png` for all ten cases.
- [x] GPT clarity-only edits with Qwen pose/scale/silhouette as geometry anchor.
- [x] Pixal3D GLB and 100k PLY generation using one seed/configuration.
- [x] v15 coarse initialization for each new complete prior.
- [x] Shared bidirectional saved-camera 2D+3D proper-Sim(3) TTO.
- [x] Shared observation-conditioned surface posterior.
- [x] Freeze predictions before strict 16,384-point CD/EMD evaluation.
- [x] Mean `CD 1.635 / EMD 2.596`, exceeding GenPC `1.74 / 2.88`.

## Remaining paper gates

- [ ] Full-ten visual review of overlays and final posterior PLYs.
- [x] Review the GT-free cross-prior scale-consistency guard derivative. It
  detects the 07136 shape-scale conflict without a sample rule and improves
  frozen mean CD/EMD from `1.635/2.596` to `1.599/2.541`.
- [x] Accept the 32k support-aware voxel surface-measure derivative. It lowers
  mean local-density CV from `0.498` to `0.292` and improves frozen mean
  CD/EMD further to `1.594/2.530`, without creating or deforming geometry.
- [ ] Runtime profiling and GPU acceleration of correspondence construction.
- [ ] Ablations: Qwen only vs Qwen+GPT; v15 vs consensus; identity vs smooth
  absorption vs 4/8/12% surface mass.
- [ ] Generalization evaluation beyond the accepted Redwood ten.
- [x] Commit the reproducible code/documentation snapshot on the dedicated
  `codex/qwen-gpt-pixal-voxel-mainline` branch.

## Method contract

- One shared zero-shot parameter set; no sample/category rules.
- No GT, CD, or EMD during generation, registration, fusion, or routing.
- Proper rotation, isotropic scale, translation; strict inverse composition.
- Keep the complete Pixal prior as the geometry majority; no hidden-body
  truncation or unrestricted non-rigid deformation.
- Keep accepted v15 immutable as the visual and ablation baseline.
