# AgentGenPC+: Intrinsic Local Geometry Action

## Goal

Image-to-3D priors provide a complete object but cannot perfectly match an
observed partial scan.  Re-decoding a whole prior to fix one armrest, chair
leg, or sofa side is unreliable: it can improve the observed view while
destroying correct hidden geometry.  This action therefore treats the complete
prior as the immutable global shape hypothesis and edits only a compact,
observation-supported mesh region.

The method is zero-shot.  It never uses ground truth, CD, EMD, category rules,
or sample IDs to create, select, or reject an action.  Metrics are strictly
post-freeze evaluation.

## Agent state and action

The agent state contains only

```text
registered complete mesh and its 100k surface sample
+ raw partial scan
+ saved depth-view camera and semantic image
```

After the existing GPU-batched 2-D+3-D proper Sim(3) registration, the agent
measures saved-view residuals and finds compact connected residual components.
For a component, it may select one action:

```text
intrinsic_local_geometry_edit
```

This is an alternative to semantic regeneration or full 3-D re-decoding.  The
action always has an exact-anchor fallback.

## Local solve

1. Sample a decimated proxy of the *registered current mesh*.
2. Build robust camera-visible partial-to-prior correspondences inside one
   connected residual component.
3. Attach their displacement vectors to proxy vertices (handles).
4. Compute intrinsic geodesic distance from the handles.  Vertices outside a
   compact support are anchors and remain fixed.
5. Solve a screened normalized-Laplacian system for the displacement field:

   \[
   \min_{d}
   \lambda_h\|d_H-(y_H-x_H)\|^2+
   \lambda_s\|Ld\|^2+
   \lambda_a\|d_A\|^2+\lambda_i\|d\|^2.
   \]

6. Remove the translation, rotation, and isotropic-scale components from the
   field.  Global Sim(3) and local geometry consequently cannot explain the
   same residual.
7. Transfer the compact displacement from the proxy only to vertices of the
   original mesh.  No faces, hidden components, or prior samples are deleted.

The action is mesh-local but the final output still uses the standard
observation-conditioned surface posterior and support-aware uniform resampler.

## Shared safety policy

The solver uses one continuation lattice for every object:

```text
[1, .75, .5, .35, .25, .15, .1, .05, .035, .025, .015, .01, .005]
```

A proposal must strictly improve the local geometric residual, satisfy

```text
q01 edge stretch >= 0.78
q99 edge stretch <= 1.28
flipped-face ratio <= 8e-4
```

and then pass both existing global checks:

- saved-camera 2-D+3-D no-harm gate;
- fixed PCA-frame three-view no-harm gate.

The small flipped-face allowance is not a general topology relaxation.  It is
only to permit a sub-0.08% near-contact artifact when the global visual and
geometric evidence improves; larger folds, significant stretch, silhouette
loss, extra leakage, or out-of-canvas growth are rejected.  The action returns
the unmodified complete prior whenever any condition fails.

## Current generic audits

Both audits start from the fresh agent registration, use the identical shared
parameters above, and make no metric-based choice.

| Sample | Local objective | Selected step | Three-view IoU / coverage / leakage | Result |
| --- | --- | --- | --- | --- |
| `07136` sofa | `0.060629 -> 0.060555` | `0.10` | `0.65349/0.82331/0.24191 -> 0.65390/0.82355/0.24148` | accepted |
| `09639` chair | `0.024242 -> 0.024051` | `0.05` | `0.64104/0.94435/0.33328 -> 0.64145/0.94452/0.33292` | accepted |

For `07136`, post-selection-only evaluation of the standard decoded output is
CD/EMD x1e2 `2.521 / 3.612`, compared with `2.524 / 3.621` for the rigid fresh
state.  This is encouraging but intentionally not a full-ten claim.

## Frozen ten-sample audit

The shared action was then run once on all ten Redwood samples. Nine actions
were accepted and `01184` automatically retained its complete-prior anchor.
The common posterior and uniform resampler were run after the action decisions
were frozen; CD/EMD were evaluated only afterwards.

| Sample | Route | CD x1e2 | EMD x1e2 |
| --- | --- | ---: | ---: |
| 01184 | anchor | 1.1444 | 1.7999 |
| 05117 | local edit | 1.0874 | 1.5527 |
| 05452 | local edit | 1.3050 | 1.9650 |
| 06127 | local edit | 1.7550 | 3.1915 |
| 06145 | local edit | 1.0398 | 1.7655 |
| 06188 | local edit | 1.3469 | 2.3464 |
| 06830 | local edit | 2.5534 | 4.7068 |
| 07136 | local edit | 2.5214 | 3.6104 |
| 07306 | local edit | 3.0602 | 3.5948 |
| 09639 | local edit | 1.2310 | 2.1962 |

The mean is `1.7044 / 2.6729`, improving the frozen fresh-rigid agent baseline
`1.7076 / 2.6793` under the same metric protocol. It also exceeds v15
(`2.1096 / 3.0963`), but remains below the historical frozen v22 result; this
is a no-regression local-action result, not a claim of final superiority.

## Implementation and artifacts

- Solver: `src/hierarchical_residual_registration.py`
- Action runner: `scripts/run_agent_intrinsic_residual_edit.py`
- `07136` action: `workspace/agent_genpc_plus_scratch_20260903/_agent_intrinsic_relaxed_07136`
- `07136` decoded output: `workspace/agent_genpc_plus_scratch_20260903/_agent_intrinsic_relaxed_07136_uniform32k/07136/07136_agent_intrinsic_registered_uniform.ply`
- `09639` action: `workspace/agent_genpc_plus_scratch_20260903/_agent_intrinsic_relaxed_09639`
- Full ten action audit: `workspace/agent_genpc_plus_scratch_20260903/_agent_intrinsic_relaxed_all10_20260904`
- Full ten decoded predictions:
  `workspace/agent_genpc_plus_scratch_20260903/_agent_intrinsic_relaxed_all10_20260904_uniform32k`
- Frozen metric report:
  `workspace/agent_genpc_plus_scratch_20260903/postfreeze_agent_intrinsic_relaxed_all10_20260904_cd_emd/metrics_samples.csv`

## Controller integration

`scripts/run_agent_qwen_nano_round.py` now executes this action first in every
state transition. An accepted local successor ends the current round with
`next_step=reobserve_successor_before_any_further_action`; the next invocation
measures the new residual before choosing another local or generative action.
If the local action rejects, the controller continues to its existing guarded
Qwen/Nano branch. Thus a global 3-D re-decode cannot overwrite a compact local
correction in the same causal step.

Smoke evidence: `01184` safely returns `preserve_prior` under `--skip-nano`;
`07136` returns `accepted_intrinsic_local_geometry` with local objective
`0.060629 -> 0.060555`. Its next invocation consumes that exact successor and
again selects a local action (`0.060555 -> 0.060373`), proving stateful
re-observation rather than a fixed replay.

`scripts/run_agent_iterative_loop.py` provides the bounded outer-loop policy.
Every successor is re-evaluated against the immutable round-zero saved-view
and PCA-frame reference, not merely against its immediate predecessor. A
state must pass both no-harm gates; among safe states, the one with the lowest
observed geometric residual is retained. Iteration stops at a fixed round
budget, when relative geometric gain is below `1e-4`, when an action returns
its anchor, or when the cross-round gate fails. A two-round 07136 local-only
audit keeps both successors and selects round two, lowering saved-view
objective `0.119793 -> 0.119656`. This validates stateful bounded selection;
post-selection evaluation remains separate. Under the standard decoder, its
post-selection-only 07136 result is CD/EMD x1e2 `2.5205/3.5950`: EMD improves
over the one-round local state (`3.6104`), but it remains below v15/v22. Thus
additional rounds are not promoted as a path to large gains; their principal
role is a controlled state interface before a higher-level text/3-D action.

### Local-to-generative hand-off

`--generative-final-probe` makes that hand-off executable.  After at least one
accepted, re-observed local action and only after the bounded local phase has
stopped, the controller may invoke exactly one camera-locked
residual-text → Qwen image → Nano3D mesh-edit proposal.  It cannot replace the
current best merely because its own local action gate accepts it: it is
re-measured against the immutable round-zero saved and PCA-frame evidence and
must also lower the observed geometric residual of the retained best state.
Consequently a text/3-D proposal is a genuine distinct agent action, while
the complete prior remains the safe fallback if it changes hidden geometry or
does not improve observed support.

The Qwen guidance strength and Nano3D editing start are explicit shared action
parameters (`--qwen-true-cfg-scale`, `--qwen-steps`, and `--nano-st-step`).
They change proposal generation only; neither relaxes the 2-D, 3-D, or
cross-round acceptance policy. This permits a fixed generic action lattice
when a residual text instruction is visually under-expressed, without using a
sample-specific route.

An optional residual-warp control is also available before Qwen.  It projects
partial-to-prior residual correspondences into the saved image and warps only
their compact support.  It is used only when the measured median image motion
is at least two pixels, and both the pre-warp and Qwen target must pass the
immutable saved-camera gate.  This prevents a depth-only residual (whose
projection is unchanged) from masquerading as useful 2-D guidance.

The first complete audit of this hand-off used `07136` with two accepted local
states, then one Qwen→Nano3D probe.  The global edited mesh was rejected by
the complete-prior gate (`0.124232 -> 0.114726` was still insufficient versus
the retained state), but its compact `0.025`-diagonal support-locked successor
passed the immutable-reference gate.  Across the whole action chain, saved
objective fell `0.119793 -> 0.113441`, and observed geometric residual fell
`0.060629 -> 0.054656`.  The frozen posterior plus uniform decoder produced
CD/EMD x1e2 `2.4567/3.4941`.  This is worse than v15
(`2.1795/2.9443`) and frozen v22 (`1.5264/2.3702`), so it is recorded as a
working closed-loop result—not a promoted quality claim.  The key outcome is
that the controller selected the safe local replacement rather than the
global mesh and retained a complete prior fallback automatically.

Artifacts: `workspace/agent_genpc_plus_scratch_20260903/_agent_iterative_local_generative_07136_v2`, its final uniform PLY is
`.../_agent_iterative_local_generative_07136_v2_uniform32k/07136/07136_agent_loop_registered_uniform.ply`,
and its post-freeze comparison is
`workspace/agent_genpc_plus_scratch_20260903/postfreeze_agent_iterative_local_generative_07136_v2_cd_emd`.
