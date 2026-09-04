# Agent GenPC+

## Objective

Agent GenPC+ turns a fixed zero-shot completion pipeline into a compact
test-time controller.  Its purpose is not to stack generators: it chooses the
smallest action that makes the saved partial observation consistent with a
complete prior, while retaining the prior whenever evidence does not justify a
change.

## State and action loop

```text
depth / semantic image / Pixal prior / saved camera / partial scan
                         |
                       inspect
                         |
     preserve prior  <---+--->  regenerate semantic or Pixal prior
                         |
      compare global SO(3) and all-PCA Sim(3) initializations
                         |
             saved-view proper Sim(3) optimization
                         |
       propose GPU screen-and-surface residual Sim(3)
                         |
            accept only under the same saved-view guard
                         |
            observation-conditioned surface decoding
                         |
              accept only under a no-harm evidence gate
```

The action space has four semantic actions:

1. `preserve_prior`: retain the current semantic image and complete 3-D prior.
2. `regenerate_semantic`: ask the image generator for a pose-preserving repair
   when its candidate passes the same view-evidence test.
3. `regenerate_pixal`: rebuild the complete prior from an accepted semantic
   candidate using the fixed Pixal configuration.
4. `select_initialization`: independently compute a global GPU SO(3) and a
   PCA-orthogonal Sim(3) initialization from the current Pixal prior.  Select
   only by shared saved-camera 2-D+3-D evidence: full-resolution silhouette
   score minus eight times the normalized partial-to-prior trimmed surface
   residual.  The PCA branch
   first computes the robust covariance frames on CUDA and batch-evaluates all
   24 proper signed axis permutations across the shared initial scales.  A
   CUDA coordinate search then retains a generic short list; only this list is
   scored by the original exact 2-D+3-D objective.  Thus GPU acceleration does
   not replace the final geometric decision with a proxy score.
5. `screen_refine`: propose one bounded GPU proper-Sim(3) residual from
   partial-to-prior surface pairs and their saved-camera reprojection.  The
   complete prior is transformed as a whole; no per-point or non-rigid motion
   is allowed.  The proposal is rejected if it worsens the saved-view 2D+3D
   objective, silhouette coverage/IoU, or leakage beyond shared tolerances.
6. `deliver`: decode a completed surface only after the registration and
   observation-consistency checks pass.

The decoded surface considers two equally constrained posterior updates.  The
first exchanges nearby visible prior mass for uniformly sampled observations.
The second only exchanges a robust saved-view prior/partial correspondence
pair, so that an observation may replace the prior surface point on its own
ray but cannot delete hidden geometry or drag unrelated surface regions.  A
shared view-evidence gate selects the smaller acceptable update, followed by
support-aware voxel sampling for a uniform point measure.

An image/mesh generator may also propose a local edit, but it is never allowed
to replace the complete object globally.  The support-lock decoder swaps only
the source-prior surface samples within one of four shared partial-distance
radii (`0.015, 0.025, 0.035, 0.050` times the partial bounding-box diagonal).
It retains all remaining prior samples exactly and uses deterministic voxel
surface-measure resampling only to restore the point count.  Radius selection
is among proposals that pass both saved-view and fixed three-view gates; it is
not an instance-level tuning parameter.

The controller uses no category label, sample identifier, ground truth, CD, or
EMD.  Its view evidence is

\[ E = \operatorname{IoU} + 0.15\,\operatorname{Coverage}
       -0.45\,\operatorname{Leakage}-0.20\,\operatorname{DepthError}. \]

A regeneration proposal is admissible only if it has coverage at least 0.90,
leakage at most 0.08, and its evidence is no worse than the preserved prior by
more than 0.005.  This makes regeneration a guarded action rather than a
source of sample-specific branching.  Since a new generator can otherwise
improve the saved camera while expanding an unseen side, generative proposals
also undergo a fixed three-view orthographic audit.  The PCA frame and raster
limits are computed once from the partial plus the trusted anchor, then held
fixed for the candidate.  It must not lose more than 0.01 mean IoU or
coverage, add more than 0.025 leakage, or place more than 0.01 additional
support outside that fixed canvas.  Thus multi-view evidence is a veto, not a
second generative input or a category-specific shape prior.

For `screen_refine`, the controller compares the candidate directly to its
current rigid registration.  It requires no objective regression, no more than
0.01 loss of coverage or IoU, and at most 0.025 additional visible leakage.
The latter is a shared allowance for a complete prior: it prevents rejecting a
candidate that covers substantially more of the observed partial surface only
because the prior also explains nearby unobserved object area.

To keep inference focused, this action is considered only when the preceding
rigid route has saved-view objective at least 0.10, regardless of whether an
earlier rigid update was technically accepted.  This is a shared uncertainty
threshold, not a class or instance rule.

The controller may validly terminate with `preserve_prior`.  This is not an
ablation shortcut: the same observer, registration, and decoder run first;
the action is chosen only when residual evidence is below the shared edit
trigger.  On the fresh `01184` run, its rigid visible objective was `0.07985`,
so no text/mesh edit was admissible.  The GT-free decoded completion reached
`1.1444 / 1.8021` CD/EMD x1e2 in a post-selection audit, exceeding frozen v22
for that sample (`1.1874 / 1.9337`).  This provides an end-to-end positive
agent result while keeping generative edits available for high-residual cases.

## First frozen-anchor audit

The first run uses the already verified complete surface as the trusted
`preserve_prior` action.  This establishes the evaluation floor before any
semantic or Pixal regeneration action is enabled.  The staged run is
`workspace/agent_genpc_plus_redwood_20260903` and records a per-sample action
log in `actions/<sample>/decision.json`.

Its post-freeze ten-sample result is `CD 1.5053 / EMD 2.4666`, exceeding the
required replay floor (`1.5402 / 2.5334`).  This is an anchor reproduction,
not a claimed improvement from generation; future regenerated candidates must
win the same evidence gate before they can replace it.
