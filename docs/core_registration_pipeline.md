# Core Registration Pipeline

## Canonical method

The project is rolled back to the accepted Pixal guarded unified registration
v15 baseline.  Hunyuan3D-MV, dual-depth semantic generation, non-rigid
deformation, and post-v15 fusion are not part of the active pipeline.

```text
partial point cloud + saved Redwood camera
  -> accepted saved-camera depth projection
  -> frozen GPT ImageGen complete semantic image
  -> frozen Pixal3D complete GLB and 100k surface PLY
  -> two shared proper-Sim(3) candidates
       (v8 GenPC/PCA fallback, v12 GPU SO(3)+visible TTT)
  -> v15 observable-confidence gate
  -> v15 full-resolution do-no-harm guard
  -> registered full Pixal3D body
```

This remains recognizably descended from GenPC: it keeps the saved-camera
depth projection and robust PCA registration fallback, while adding a complete
Pixal3D prior and a GPU visibility-aware test-time registration candidate.

## Frozen assets

For each of `01184`, `05117`, `05452`, `06127`, `06145`, `06188`,
`06830`, `07136`, `07306`, and `09639`:

- raw partial: `data/<sample>.ply`;
- saved camera: `workspace/redwood_onestage_rawdepth_512_stage2_20260714/<sample>/camera.pth`;
- depth: `gpt_version/<sample>/depth.png`;
- semantic image: `gpt_version/<sample>/gpt_image.png`;
- complete mesh: `gpt_version/<sample>/pixal3d.glb`;
- complete points: `gpt_version/<sample>/pixal3d_sampled_100k.ply`.

These inputs and the accepted v15 outputs are immutable unless the user
explicitly asks to regenerate them.

## Candidate A: GenPC/PCA fallback

The fallback root is `gpt_version/_pixal_scale_ttt_v8_20260822`.  It provides
the robust GenPC/PCA-oriented proper-Sim(3) candidate used whenever visible
evidence is weak or the GPU candidate fails the do-no-harm guard.

## Candidate B: GPU SO(3) + visible Sim(3) TTT

The fast candidate root is
`gpt_version/_pixal_batched_adaptive_ttt_v12_20260822`.  Its coarse stage
evaluates the shared 648 proper rotations and five isotropic scales with
batched GPU z-buffer rendering.  Silhouette IoU, coverage, leakage, and robust
visible-depth error rank candidates; expensive 3D surface evidence and local
proper-Sim(3) TTT are applied only to a shortlist.

The shared observable-confidence thresholds are:

- low-resolution IoU at least `0.85`;
- coverage at least `0.90`;
- leakage at most `0.08`;
- normalized visible-depth error at most `0.10`;
- normalized trim-70 partial-to-complete surface error at most `0.012`.

## v15 guarded routing

Implementation: `scripts/select_pixal_guarded_unified_registration_v15.py`.

For full-resolution projection metrics, define:

```text
render_score = IoU + 0.15 * coverage - 0.45 * leakage
```

The GPU candidate is selected only if it passes the observable-confidence
gate and its full-resolution render score is no more than `0.005` below the
fallback score.  Otherwise v15 selects the GenPC/PCA fallback.  The router has
no access to sample IDs, categories, GT geometry, CD, or EMD.

Accepted full-ten routing:

| sample | v15 route |
| --- | --- |
| 01184 | GenPC/PCA fallback |
| 05117 | GenPC/PCA fallback |
| 05452 | GenPC/PCA fallback |
| 06127 | GenPC/PCA fallback |
| 06145 | GPU global SO(3) TTT |
| 06188 | GenPC/PCA fallback |
| 06830 | GPU global SO(3) TTT |
| 07136 | GenPC/PCA fallback |
| 07306 | GenPC/PCA fallback |
| 09639 | GenPC/PCA fallback |

Accepted output root:

`gpt_version/_pixal_guarded_unified_registration_v15_20260822`

The historical output stem contains `unified_registration_v14`; the outer
router, method field, and root identify the accepted v15 method.

## Completeness and geometry contract

- Preserve all 100,000 frozen Pixal points.
- Use only proper rotation, one isotropic global scale, and translation.
- Do not use anisotropic scaling, non-rigid deformation, partial replacement,
  generated-point deletion, or Hunyuan regeneration.
- The registered Pixal body is the active complete prediction.  Registration
  and any future fusion must remain separate ablations.
- Low confidence invokes the robust fallback; it never authorizes shape edits.

## Reproduction

With v8 and v12 candidates already present:

```bash
/opt/data/private/cr/miniconda3/envs/genpc/bin/python \
  -m scripts.select_pixal_guarded_unified_registration_v15
```

Do not run this command against the accepted output root unless overwrite is
explicitly intended.  The frozen output files already exist and are the
canonical predictions.

For the complete design, timing audit, gates, and routing rationale, see
`docs/fast_unified_registration_v15.md`.  Reproducibility and approval details
are recorded in `PROJECT_STATE.md`.

## Generalization and evaluation contract

- Use one shared parameter set for all ten Redwood samples.
- Freeze predictions before reading GT metrics.
- Report every sample and the mean CD-L1/EMD.
- The paper target remains mean CD-L1 x1e2 below `1.74` and EMD x1e2 below
  `2.88`.
- A future method may replace v15 only after full-ten visual review and
  post-freeze metric improvement with no sample-specific tuning.

## Post-v15 research candidate: guarded hierarchical residual TTO

This section records an experimental derivative and does not change the
canonical v15 pipeline above.

The candidate alternates two levels at test time:

1. a small, proper and isotropic residual Sim(3) update estimated from the
   saved-camera visible correspondences;
2. an optional intrinsic mesh residual solve on a compact screen-space error
   component.

The local field is propagated by mesh geodesic support so it cannot jump
between nearby disconnected surfaces. Before application, translation,
rotation, and isotropic-scale modes are explicitly projected from the field.
This separates global pose/scale from local shape. Continuation line search,
visible 2D+3D improvement, coverage preservation, edge-stretch bounds, and
face-flip limits guard every local update. Original Pixal point identities and
mesh topology remain; no generated region is deleted or replaced.

For generalization, every sample receives the same geometry-derived candidate
set: surface/curve handle budgets crossed with residual-mass/line-priority
component routing. A pure global residual trajectory is evaluated in parallel.
The lowest passing visible objective is selected, otherwise output is exactly
v15. Sample IDs, semantic categories, GT geometry, CD, and EMD are unavailable
to this router.

The 2026-08-23 pilot uses `09639` and `07136` only as diagnostics. Its output
is in
`gpt_version/_pixal_hierarchical_residual_registration_multiscale_pilot_20260823`.
It remains an ablation until frozen and validated on all ten samples.

An independent bidirectional candidate in
`src/bidirectional_cycle_registration.py` estimates partial-to-visible-prior
Sim(3), applies the strict inverse to the complete prior, and checks an
independently fitted reverse transform as a cycle witness. Its two-case pilot
is in `gpt_version/_pixal_bidirectional_cycle_registration_pilot_20260823`.
Current gains are smaller, so it is not yet part of the hierarchical method.
The runner also provides `--force-candidate-output` solely for visual auditing:
it records failed gates but exports the candidate instead of restoring v15.
Outputs from this mode are not eligible for benchmark reporting or automatic
promotion.

The frozen full-ten forced audit was evaluated post hoc with shared 16,384
point FPS indices and seed 6145. It obtained mean CD-L1/EMD x1e2
`2.0648/3.0786`, versus `2.1096/3.0961` for v15 under exactly the same sampled
points. This is a small baseline improvement but remains behind the GenPC
paper mean `1.74/2.88`; therefore the bidirectional route is not promoted.

## Metric-passing PAMI extension candidate

The next bidirectional version fixes a conceptual weakness in the side pilot:
the old reverse witness reused forward point pairs and therefore understated
cycle disagreement. The new implementation builds two independent saved-camera
visible correspondence sets:

```text
partial -> visible complete prior
visible complete prior -> partial
```

It evaluates forward-only, reverse-only, balanced bidirectional, and reciprocal
pair proper-Sim(3) hypotheses. Every hypothesis is converted to the complete-
to-partial direction by an analytic strict inverse, bounded by the shared trust
region, and selected with the same GT-free visible 2D+3D objective. This is the
global registration contribution of the proposed GenPC extension.

Because a generated prior cannot exactly reproduce observed local geometry,
the posterior stage treats the partial scan as observed surface measure rather
than concatenating arbitrary-density points. It FPS-samples the partial under a
maximum 12% mass budget and removes exactly the same amount of generated mass
nearest the observation. Thus total point count stays 100k, partial scan-line
density is normalized, and the complete Pixal prior remains an 88% majority.
Identity, smooth absorption, and 4/8/12% mass hypotheses are selected with a
shared saved-camera objective and explicit prior-mass penalty. No GT or semantic
category enters this selection.

The strict post-freeze ten-sample result is CD-L1/EMD x1e2
`1.6912/2.8026`, improving over both v15 (`2.1096/3.0973`) and the GenPC paper
mean (`1.74/2.88`). Predictions are in
`gpt_version/_pixal_bidirectional_consensus_surface_projection_20260823`.
This is a metric-passing research candidate pending full visual acceptance; it
does not overwrite the frozen v15 canonical assets.
