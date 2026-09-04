# Text-Conditioned Prior-Editing Agent

## Purpose

This is the proposed TPAMI-facing extension of Agent GenPC+.  It makes the
complete-shape prior editable without using ground truth or any
sample/category-specific rule.  The prior is still obtained from the semantic
image by an image-to-3D model; a text-conditioned 3D *variant* model is an
optional, evidence-gated action rather than an unconditional second generator.

## One agent iteration

1. Register the current complete prior to the partial scan with the shared
   saved-view 2D+3D proper-Sim(3) solver.
2. Observe both the saved camera view and three deterministic PCA-orthographic
   geometry views.  The latter are agent diagnostics only, not generated image
   conditioning.  Measure partial-to-prior residuals only in the observed
   region.  Compute the
   70/90-percentile normalized distance and compare observed and matched-prior
   5--95% extents in the partial's PCA frame.
3. Convert this numeric diagnosis into one bounded text instruction, such as
   a local extension, contraction, or conformance correction along the long,
   middle, or short observed principal axis.  The prompt explicitly preserves
   identity, global pose, global proportions, structure, and hidden geometry.
   The Codex visual observer receives a fixed three-panel saved-view board
   (partial, prior, overlay) and can only approve or veto this proposed edit
   through a strict JSON schema. For a mesh-variant action it cannot introduce
   a category, a new part, a free-form direction, or a global rewrite. For a
   from-scratch constrained action, it may instead write a conservative object
   description grounded in the semantic input, then append the same bounded
   local correction. An optional OpenAI Responses adapter exists only for
   unattended replication; it is not a method dependency.
4. Select one backend action. The executable action set offers: (a) a
   camera-locked semantic-image edit followed by an image-to-3D proposal;
   (b) SPAR3D with the edited semantic image and the observed partial as a
   point-cloud condition; (c) Arbor text generation constrained by a
   watertight local hull built from the observed partial; or (d) Nano3D
   FlowEdit over a source mesh render and an agent-produced target render.
   The controller only chooses from this finite action set and never applies
   an unconstrained point deformation. A generated proposal may additionally
   be support-locked: it can replace the current complete prior only within a
   shared-radius 3-D tube around the observed partial, while unobserved prior
   support is preserved verbatim.
5. Map the candidate from its recorded native frame, run the same saved-view
   2D+3D proper-Sim(3) re-registration, and accept it only if its GT-free
   objective, coverage, IoU, leakage, and complete-shape support pass shared
   no-harm gates.

The loop stops when the multi-view observer reports no material visible
discrepancy, when no action passes the saved-view gate, or after three accepted
iterations. The fixed cap is a shared compute/safety limit rather than a
sample-specific schedule; the best accepted prior remains the safe fallback.

## Executed two-round trace

The fresh `07136` audit exercises the loop without replay semantic or 3-D
assets.  In round one, Codex inspected the saved-view and three PCA views,
created a camera-locked target render, and Nano3D edited the registered source
mesh.  The complete edited mesh did not itself pass the global-shape gate, but
its partial-supported `0.050`-diagonal support lock did: saved-view objective
changed from `0.11979` to `0.09397`, IoU from `0.84832` to `0.85490`, and
coverage from `0.92094` to `0.95405`.  The selected result was then decoded
through the standard observation-conditioned posterior and uniform surface
resampling.  CD/EMD is measured only afterwards (`2.2792 / 3.1913` x1e2), not
used to select the action.

Round two rendered new evidence from that accepted round-one prior and asked
for only a further local correction.  Its Nano proposal reduced its own score
from `0.12620` to `0.11831`, but failed against the current trusted state
(`0.09397`), so the no-harm gate returned the first-round prior.  This is the
intended agent termination behavior: a generative edit can be attempted but
cannot overwrite an already accepted complete prior merely because it looks
plausible in one view.

## Qwen image-action to Nano3D audit

The text action is now executable through a local image editor rather than a
hand-authored target image. The controller turns the largest partial-supported
saved-view residual component into a bounded phrase containing only its camera
region and a local `expand` or `contract` operation. Qwen-Image-Edit receives
that instruction together with the camera-locked source render. Before any
3-D backend is invoked, a binary foreground gate checks source/target IoU,
source coverage, target leakage, bounding-box scale and centroid shift. This
prevents a fluent-looking edit from silently changing camera pose or global
scale.

On the independent 07136 audit, the Qwen target passed this gate (IoU 0.9934,
scale 1.0051, centroid shift 0.00069). Nano3D FlowEdit then proposed a new
complete mesh from the cached source encoding and this target. Its global
proposal failed the same multi-view re-registration gate, so only a shared
0.035-diagonal partial-support tube was allowed to borrow geometry from it.
That restricted candidate improved the no-GT saved-view objective from 0.11979
to 0.11187 and coverage from 0.92094 to 0.94755. However, the standard frozen
post-selection metric, which was not used anywhere in this decision, was CD
2.4461 and EMD 3.4312 (x1e2), below both v15 and frozen v22. Thus the audit
establishes the full text-to-image-to-3D agent path and its safety controls,
but does not promote Nano re-decoding as a mainline improvement: local
saved-view conformance alone cannot compensate for hidden-surface drift caused
by a global mesh decoder.

## Generated proposal as local advice

Two conservative alternatives prevent an edited 3-D proposal from becoming a
replacement complete shape.  The first transfers only camera-visible residual
vectors from the generated proposal to compact, geodesically supported handles
of the original mesh; all vertices outside that intrinsic support and the mesh
topology stay unchanged.  The second is a surfel-domain version: every output
point is an original prior point displaced by a compact RBF field, never a
generated point appended to or substituted into the completion.  Both remove
the local similarity modes before updating, and an agent state transition must
strictly improve its own saved-view objective as well as pass the common
multi-view no-harm gate.

The fresh 07136 Nano proposal was informative in the visible image but not
safe enough for either route.  Mesh-handle transfer produced folded triangles
because nearest-proxy correspondences crossed a thin surface.  The surfel
variant retained all points but found no strictly improving step in a fixed
tight/local/medium action lattice.  Both routes correctly preserved the
trusted anchor.  They remain useful guarded action definitions, not claimed
metric improvements.

## Model contract

The standard shared TRELLIS checkout is not used as a mesh-text edit backend,
but the pinned public TRELLIS implementation shipped with Arbor exposes the
official `run_variant(base_mesh, prompt)` interface. It voxelizes the supplied
base mesh into fixed structure coordinates, then samples only a
text-conditioned SLAT geometry latent; this is a reproducible mesh-plus-text
proposal, not a claimed hidden API. The complete local
`TRELLIS-text-xlarge` checkpoint is required (the small `-local` overlay is
not a complete default). On the 07136 audit, the shared residual controller
asked for a slight visible short-axis extension; the proposal ran but was
rejected after common re-registration (`0.32273` objective versus `0.11979`
anchor), and its best support-locked descendant was `0.16528`. It therefore
remains an executable negative action, not a promoted route. The initial prior
may be generated by Pixal3D or an image-conditioned TRELLIS model and is
scored with the same saved-view evidence. TRELLIS.2 is not an edit action
because its available interface is image-to-3D/texturing only.

SPAR3D provides a complementary point-aware image-to-3D action: its semantic
image receives the text-conditioned 2-D edit, while a deterministically
subsampled, bbox-normalized partial point cloud is supplied as the model's
native geometry condition. Its known mesh-output rotation and conditioning
normalization are serialized; neither becomes a hidden method-frame axis flip.

Arbor provides a text-and-geometry constrained alternative. The partial is
represented by a union of small watertight octahedral hull primitives rather
than a guessed watertight completion, so the text model is asked to produce a
complete object that includes observed local support without forcing its hidden
surface. The visual agent creates the text description and can add only a
bounded local correction phrase after a residual diagnosis.

Nano3D is a direct complete-prior editing action rather than another
from-scratch generator. The controller keeps Nano3D's canonical 150-view
mesh encoding, but renders the *editing source condition* in the deterministic
saved-view camera crop. It copies the semantic target only inside the current
prior silhouette, so source and target differ only where the observed view
asks for a geometric correction. The source/target pair then drives Nano3D's
training-free FlowEdit. Canonical source encodings and voxel latents are
cached across conservative edit trials. The resulting native mesh is exported
even when texturing is unavailable, because the shared registration and
surface reasoning use geometry only.

This isolates an important failure mode. On 07136, camera locking improved
the candidate geometric objective from `0.06063` to `0.05146` and coverage
from `0.92094` to `0.94317`, but caused unobserved-region over-expansion:
IoU fell from `0.84832` to `0.82442` and leakage rose from `0.08505` to
`0.13249`. A globally conservative schedule (`st_step=18`) also failed the
same gate (`0.12275` objective, `0.82041` IoU, `0.13259` leakage). The gate
   therefore selects the anchor in both cases. Nano3D remains an executable
   ablation, not a promoted route: the next viable action must explicitly lock
   the observed 3-D support rather than simply weaken global image editing.

The support-locked variant implements precisely that constraint without a
non-rigid warp. After common re-registration, it keeps anchor samples beyond a
shared tube around the partial and admits generated samples only inside the
tube. Fixed candidate radii of `0.015, 0.025, 0.035, 0.050` times the partial
bounding-box diagonal compete under the same saved-view gate. Thus a local edit
may improve observed conformance but cannot erase an unobserved chair back,
sofa arm, or other complete-shape support. On the high-residual 07136
diagnostic, the accepted radius improved objective `0.11979 -> 0.11247`, IoU
`0.84832 -> 0.85287`, and coverage `0.92094 -> 0.94826`; 06188 did not pass,
so the controller retained its anchor. These are action diagnostics, not
ground-truth-based selection or a sample-specific rule.

## Intrinsic local geometry action

For a compact camera-visible residual, the agent may edit the current mesh
instead of decoding a replacement prior. It attaches robust partial-to-prior
correspondences to vertices of a decimated copy of the current complete mesh,
locks vertices outside a geodesic support region, and solves a screened
Laplacian/ARAP-style displacement. Similarity modes are removed from that
displacement, so global pose and local shape do not explain the same error.
The displacement is transferred only to vertices of the original mesh; its
topology, unobserved support, and complete-prior point count are retained.

The shared continuation lattice is
`[1,.75,.5,.35,.25,.15,.1,.05,.035,.025,.015,.01,.005]`. A candidate must
strictly lower the local geometric residual, have q01/q99 edge stretch within
`[.78,1.28]`, and have at most `8e-4` flipped faces. This last tolerance only
permits tiny near-contact inversions; saved-camera and fixed three-view
no-harm gates remain mandatory. On `07136`, the 10% continuation lowers its
local objective `0.060629 -> 0.060555`, with q99 stretch `1.084` and `0.0677%`
flipped faces; its three-view IoU/coverage/leakage improves
`0.65349/0.82331/0.24191 -> 0.65390/0.82355/0.24148`. The identical action
also accepts on `09639`, so this is an executable generic action rather than
a sofa-specific rule. It remains subject to a full-ten no-regression audit.

Hunyuan3D-Omni is also available as an image-plus-observed-point executor. It
records its normalized partial control frame and is subject to the same common
proper-Sim(3) solver. Its installed public inference path does not consume the
exposed `prompt` argument, so it is not described as a text-to-3-D edit: text
must instead conservatively alter the semantic image upstream. Its frozen
DINOv2 image encoder is deliberately loaded separately from the Omni
checkpoint. In the current 06188 audit the generated prior fails the
saved-view gate (IoU `0.4549`, coverage `0.7720`, leakage `0.4746`), therefore
Omni remains an unpromoted executor ablation.

The text editing action never sees GT, CD, EMD, a replay asset, or a category
label.  It does not use non-rigid point deformation.  The final decoded point
set remains the view-conditioned surfel posterior of the accepted complete
prior and partial observation.

## Executable round controller

`scripts/run_agent_qwen_nano_round.py` combines the complete action sequence
into one state transition. Given only the current registered prior/mesh,
partial scan, saved camera and semantic image, it writes the residual-derived
instruction and first evaluates the topology-preserving intrinsic local
geometry action. An accepted local successor terminates the round and is
re-observed on its next invocation. Only after local rejection does it create
a camera-locked Qwen target, invoke Nano3D, adapt its native frame with common
PCA proper-Sim(3), and evaluate global then support-locked successors against
the fixed multi-view reference. It returns an exact copy of the anchor on
every failed gate. A shared objective trigger of 0.10 prevents a low-residual
state from spending generation compute: fresh 01184 terminates before
Qwen/Nano at 0.07985.

On fresh-state 07136, the controller rejected the full Nano proposal and
accepted only its 0.015-diagonal support-locked descendant, improving visible
objective 0.11979 to 0.11368, IoU 0.84832 to 0.85535, and coverage 0.92094 to
0.93817. The post-selection score was nevertheless CD 2.4834 / EMD 3.5488
(x1e2), below v15 and frozen v22. This is a negative quality result, not a
promotion criterion; it shows why an agent state gate must be followed by
frozen global evaluation before claiming an improvement in completion.

## Backend coordinate contract

The loop is backend-independent.  Every backend exports an explicit
`native_to_method` proper-Sim(3), tagged with its native frame.  It may not
reuse Pixal axis flips, allow per-axis scale, or silently assume a canonical
front direction.  The generic PCA/global initialization and saved-camera
2D+3D re-registration then refine this recorded adapter transform against the
partial scan.  A candidate is invalid if its adapter contains a reflection,
anisotropic scale, or shear.  This lets Pixal3D, TRELLIS, Hunyuan3D-1, and any
future instruction-editing backend compete under the same coordinate and
evidence contract.
