# Active plan

## Camera-1 plus visible-partial-ranked view probe — 2026-09-09

- [ ] In the isolated 06830 probe only, replace the fixed orthogonal auxiliary
  views with Camera-1 plus three category-free views ranked by the total number
  of z-buffer-visible physical partial points. Keep a minimum yaw separation
  to suppress redundant nearby cameras; save every candidate count, incremental
  coverage diagnostic, and selected yaw.
- [x] Regenerate one camera-locked four-view image from the adaptive evidence,
  then run both TRELLIS-image-large and Pixal3D-MV from exactly that image and
  seed. Pixal3D-MV must consume the selected physical yaw matrices rather than
  relabeling them as 90-degree canonical views. No GT or offline metric may
  affect view selection, image generation, or model choice.
  The no-GT agent loop selected its per-view best state at round 4 (round 5
  stalled and was rolled back). Regeneration now uses that accepted board for
  both generators, with an identical deterministic camera-framing pass and
  seed 42, under `workspace/agent_loop_regeneration_06830_20260910/`.
  Both 100k carriers and GLBs were generated successfully on the RTX 4090.
  TRELLIS used about 9.85 GiB peak memory and 40.4 s load+generation;
  Pixal3D-MV used 16.41 GiB and 70.0 s after its longer initialization.
  For visual diagnosis, the MV carrier was also placed with the original
  Pixal-native-MoGe to Camera-1 partial bridge only: native residual refinement
  and the final partial residual refinement were both disabled. The analytic
  result is under `registration_analytic_only/`; it uses no PCA, ICP, GT, or
  complete-to-partial search.
- [x] Verify every edited four-view board before 3-D regeneration by projecting
  the physical partial into the same saved cameras. Report positive-support
  coverage and residual overlays per view; missing partial pixels remain
  unknown and are never treated as negative silhouette evidence.
  The reusable no-GT check is
  `scripts/compare_edited_multiview_partial.py`. For the 06830 strong-v2 edit,
  3-pixel positive-support coverage changes relative to the registered prior
  by `+0.97/+3.61/+7.74/+1.55` percentage points for the four selected views.
  Artifacts and per-view statistics are under
  `workspace/ranked_visible_partial_views_06830_20260909/edit/v2_partial_comparison/`.
  This verifier now drives a no-GT agent loop. It uses a visibility-weighted
  one-sided positive-support residual (3-pixel coverage, mean outside distance,
  and p90 outside distance), not a Front-relative coverage target. Low-support
  views are marked uncertain rather than forced to overfit. Each round accepts
  or rolls back each view independently, keeps a per-view best-so-far state,
  and stops early or after at most six rounds. The 06830 loop accepted round 4
  and rejected the stalled round 5. Its accepted board and trace are under
  `workspace/ranked_visible_partial_views_06830_20260909/edit/agent_loop/`.

## Calibrated Pixal3D-MV object-mainline replacement — 2026-09-09

- [x] Verify the newly released official Pixal3D multi-view entry point. It
  consumes posed views through NeRF/Blender camera-to-world matrices and a
  horizontal FOV; frame 0 is the canonical main/front gauge. This removes the
  uncalibrated asset-frame ambiguity of the TRELLIS multi-image interface.
- [x] Fast-forward the isolated local Pixal3D checkout from `cdbb2bb` to the
  official MV release commit `f7cf384` without overwriting local changes.
- [x] Download the new `pipeline_mv.json` and four `ckpts/*_mv` denoisers
  from ModelScope. Validate multi-view inference on the 24 GB RTX 4090 using
  low-VRAM mode at 1024 resolution.
- [x] Build a dataset-independent Camera-1-relative FRONT/SIDE/BACK/RIGHT
  `transforms.json` and prepare the isolated 06830 pilot from the accepted
  edited conditions.
- [x] Run the isolated 06830 four-view pilot and verify official camera
  loading, GLB export, native coordinate provenance, and 24 GB feasibility.
  The 1024 low-VRAM run uses 16.41 GiB and 50.2 s after model loading.
- [ ] Replace TRELLIS generation and unconstrained camera-bank registration in
  the object mainline with calibrated Pixal3D-MV generation followed by the
  existing Pixal--MoGe--partial bridge and a small bounded 2D+3D residual
  Sim(3). Keep old artifacts only until equivalence is verified, then remove
  obsolete TRELLIS code and rewrite the canonical documentation.
- [ ] Validate the same frozen MV camera/configuration on Pig and 06830 before
  broader dataset claims. GT/CD/EMD remain post-hoc-only and cannot select a
  camera, generation, or registration candidate.
- [x] Test a camera-locked view-edit contract on 06830 without changing the
  frozen object mainline. FRONT remains Camera-1; SIDE/BACK/RIGHT use the same
  elevation, FOV, radius, and horizontal orbit. GPT may alter only
  partial-supported contour/local shape, after which deterministic isotropic
  framing restores the source camera footprint. The corrected RIGHT view
  reduces principal-axis drift from about 59 degrees to 3.35 degrees and
  foreground-centre drift to about 5 pixels. Under the identical no-GT
  Pixal--MoGe--partial bridge and Camera-1 refinement, the visible objective
  improves from 0.35814 to 0.35627. Post-hoc-only CD-L1/EMD x100 are
  4.3185/5.0329 versus 4.6839/5.6943 for the three-view MV pilot; this is a
  useful recovery but remains below the single-view Pixal baseline and is not

## Fusion-free TRELLIS object-mainline promotion — 2026-09-09

- [x] Promote the accepted residual-guided TRELLIS regeneration route from an
  isolated feasibility probe to the current object-level mainline. The route
  now composes saved-view depth/semantic preparation, clarity editing, initial
  Pixal generation and Pixal--MoGe--partial registration, shared-low-frequency
  plus residual-only three-view editing, TRELLIS regeneration, and physical
  Camera-1 2D+3D Sim(3) registration.
- [x] Add the resumable entry point `scripts/run_object_mainline.py` and the
  dataset-independent artifact contract `src/object_mainline.py`. External GPT
  edits remain explicit checkpoints with saved prompts; inference contains no
  GT/CD/EMD input and publishes `final/<sample>/complete_100k.ply`.
- [x] Remove fusion and Gaussian/non-rigid adaptation from the canonical route.
  Historical implementations may remain in the research worktree but are not
  called by the current object mainline.
- [x] Rewrite the root README, canonical method specification, model list,
  installation notes, documentation index, and offline evaluator for the new
  final-prediction contract.
- [x] Re-run the accepted Pig final residual Sim(3) with the frozen shared
  parameters. The replay at
  `workspace/mainline_trellis_replay_20260909/final/domestic_pig/complete_100k.ply`
  is byte-identical to the accepted output (SHA-256
  `de0efb3dbbbb18369f0395b6173b41e2166ac46e38bcf42f8ccbee5034d3a048`).
  This is a no-regeneration equivalence check, not a new metric-selected run.
- [ ] Run the complete promoted route on cross-category Custom and Redwood
  samples before claiming dataset-level performance. Pig validates exact
  orchestration and registration replay only; it does not establish
  generalization.

## Isolated TRELLIS multi-view prior regeneration probe — 2026-09-08

- [x] Verify the official tuning-free `run_multi_image` interface and the
  local 24 GB execution environment.  Reuse the existing official TRELLIS
  checkout and `TRELLIS-image-large` checkpoint; do not alter the frozen
  GenPC++ mainline or the accepted Pig PosteriorAdapter output.
- [x] Render a camera-consistent front/side/back board from the registered
  Pixal mesh, edit all views jointly under a single identity/attachment
  constraint, and split the result into deterministic TRELLIS inputs.
- [x] Generate one mesh and Gaussian asset with TRELLIS-image-large, align the
  regenerated complete prior back to the fixed partial observation, and let a
  no-GT verifier choose between the regenerated and accepted prior.
- [x] Only after the no-GT decision is frozen, report Pig CD/EMD offline and
  record exact inputs, prompts, model paths, seeds, and artifacts.
  - A category-parameterized residual board now exports one evidence image per
    view in addition to the joint board.  A failed joint edit exposed that an
    auxiliary view can ignore sparse depth evidence, so the probe uses the
    generic action `REFINE_VIEW(view_id, object_type)`: edit only the selected
    view from its depth/residual evidence, retain another accepted view as a
    3-D consistency reference, then deterministically restore the original
    foreground centre and isotropic image scale.  No category part name,
    coordinate-side rule, GT, CD, or EMD enters this action.
  - Pig artifacts for this test are under
    `workspace/trellis_pig_multiview_probe_20260908/conditions/residual_guided_v2/`;
    the corrected three-view board is
    `gpt_local_contour_refined_front_side_back_v3.png`.  TRELLIS outputs are
    `trellis/local_contour_v3_seed42/` and the Camera-1-frozen hybrid is
    `trellis/residual_guided_v3_hybrid_seed42/`.
  - The original GT oracle used 75% trimmed ICP, which could ignore complete
    end structures.  The diagnostic default now fits all complete surfaces.
    Under the same full-surface oracle, the retained older hybrid measures
    CD-L1/EMD x100 `1.7200/2.4181`; the generic residual hybrid measures
    `1.8788/2.6050`.  The latter's partial-to-prior p90 distance is marginally
    lower (`.03224` vs `.03230` of the partial diagonal), but its aggregate
    quality is not yet better, so it remains a feasibility result and cannot
    replace the existing candidate.
  - The next generic refinement replaces independent per-view arrows with a
    **cross-view 3-D residual component ledger**. Residual partial samples are
    clustered in 3-D using only median spacing and partial-diagonal scales;
    the same component ID/colour, source-to-target arrows, and depth change are
    rendered in every camera. Components are automatically split into
    dominant and fine corrections by normalized residual magnitude. GPT edits
    the retained hybrid board from this shared ledger, while the exact
    Camera-1 image is restored before TRELLIS. The user approved the resulting
    three-view conditions at
    `conditions/cross_view_ledger_v4/gpt_cross_view_refined_front_side_back.png`.
    The pending 3-D candidate is
    `trellis/cross_view_ledger_v4_hybrid_seed42/`.
  - With the full-surface GT oracle used only after generation, v4 measures
    CD-L1/EMD x100 `1.7832/2.4668`, close to but not better than the older
    hybrid's `1.7200/2.4181`.  Against the observed partial under the same
    oracle pose, v4 improves normalized mean/p90 surface distance from
    `.01338/.03230` to `.01253/.03012`, and coverage at 1%/2% from
    `56.14/75.48%` to `57.89/78.80%`.  Thus the generic GPT action improves
    partial-supported local geometry but is not yet an accepted replacement
    for the older complete-shape candidate.
  - The TRELLIS registration audit exposed a separate **condition-camera
    drift**: GPT preserved the front frame but recentered the side and back
    conditions by approximately `(-23.18, -3.21)` and `(-11.09, -11.74)`
    pixels at 512 resolution. Treating those shifts as object translation
    improves auxiliary overlays while damaging the trusted front/Camera-1
    alignment. The isolated registration now estimates this per-view
    principal-point nuisance motion from the original-render/edited-condition
    mask pair, then fits only the shared residual Sim(3). Per-view zoom is
    deliberately not absorbed because it can erase genuine shape/scale
    evidence; a trial that did so made the side view visibly too small.
  - The shift-only self-calibrated probe is under
    `workspace/trellis_pig_multiview_probe_20260908/registration/self_calibrated_shift_only_sim3_v15/`.
    Relative to the previous constrained pose it recovers a `4.7880` degree
    residual rotation, `1.0148` scale ratio, and bounded translation without
    using GT. Its post-hoc-only CD-L1/EMD x100 changes from
    `4.2122/4.8369` to `3.8361/4.4850`. This remains an isolated feasibility
    result pending visual approval and cross-category validation.
  - A direct official-rasterizer audit shows that a shared Sim(3) alone cannot
    explain all three edited views: even a full global affine fit trades the
    side mask against the back mask.  The clean registration model therefore
    keeps Camera-1/front fixed as the gauge and jointly estimates one shared
    object Sim(3) plus small side/back orbit residuals.  Rasterization and
    antialias gradients use NVIDIA `nvdiffrast==0.4.0`; SO(3) maps use
    `pytorch3d==0.7.9`; no custom differentiable rasterizer is maintained.
    The best no-GT candidate is
    `registration/joint_camera_object_nvdiffrast_v16/`, with 128px soft mask
    IoUs `0.8273/0.7489/0.7621` (front/side/back) and post-hoc-only CD-L1/EMD
    x100 `3.4884/4.3317`.  A 256px residual refinement changed these to
    `3.4945/4.3490`, so it is rejected; the v16 transform remains the probe
    candidate.  The reusable implementation and true mask board are in
    `src/nvdiffrast_multiview_registration.py` and
    `scripts/run_nvdiffrast_multiview_registration.py`.  This remains isolated
    from the frozen mainline and is not accepted without user review and
    cross-category validation.
  - A provenance audit confirmed that the earlier v4 TRELLIS candidate was a
    hybrid: its front condition was the retained Pixal render, while only the
    side/back conditions came from the accepted GPT residual edit.  A fair
    all-new replay now splits all three panels from
    `conditions/cross_view_ledger_v4/gpt_cross_view_refined_front_side_back.png`
    into `conditions/cross_view_ledger_v4_all_new/` and regenerates TRELLIS at
    `trellis/cross_view_ledger_v4_all_new_seed42/`.  Its metadata records all
    three new input files.  Applying the previous transform only as a post-hoc
    coordinate diagnostic already improves CD-L1/EMD x100 from the hybrid
    candidate's `3.4884/4.3317` to `3.3642/4.2079`, showing that regeneration
    improved the geometry and that automatic basin capture was the remaining
    error.
  - The all-new automatic alignment now uses one Camera-1 partial-to-prior
    inverse Sim(3) capture with rotation frozen, followed by official
    nvdiffrast shared-object refinement.  Clearing inherited auxiliary-camera
    rotations is essential: otherwise side/back nuisance motion absorbs the
    shared object rotation.  The current no-GT candidate is
    `registration/trellis_all_new_reset_aux_nvdiffrast_v28/`, with 512px hard
    mask IoUs `0.8215/0.7595/0.7021` and post-hoc-only CD-L1/EMD x100
    `3.3948/4.3265`.  A subsequent 512px refinement changed this to
    `3.3991/4.3416` and is rejected.  This remains an isolated Pig probe and
    does not modify the frozen mainline.
  - The shared-low-frequency/local-residual v7 TRELLIS geometry was audited
    separately from its pose.  The official TRELLIS `run_multi_image` API
    consumes an unordered set of image embeddings and no camera extrinsics;
    therefore the edited front/side/back panels constrain generated shape but
    are not three calibrated cameras in the exported asset frame.  A GT-only
    diagnostic (never used by inference) nevertheless confirms a substantially
    better proper Sim(3) basin exists: CD-L1/EMD x100 `1.9392/2.6391`.
  - A new isolated, category-independent residual capture keeps all partial
    support instead of dropping the fixed high-residual quartile and performs
    a broad proper-Sim(3) search using the saved Camera-1 silhouette, physical
    depth, visible 3-D surface, and bounded robust partial-to-prior distance.
    It is implemented in `src/partial_supported_sim3.py` and
    `scripts/run_partial_supported_sim3.py`.  The retained no-GT result is
    `registration/trellis_shared_local_v7_depth_visible_sim3_v37/`: relative
    to the previous pose it recovers `8.66` degrees, scale `1.0029`, and the
    missing translation; CD-L1/EMD x100 improves from approximately
    `3.62/4.72` to `2.0503/2.7860`.  The user subsequently approved this
    visual result as the retained TRELLIS feasibility output. Its full
    Camera-1 visible objective is
    `.11725`, better than the diagnostic oracle's `.12975`, so the remaining
    approximately `4.1` degree oracle difference is not identifiable from the
    trusted partial view alone.  A second local pass at v38 improved its
    sampled training objective but regressed to `2.4164/3.2024`; it is
    rejected and v37 remains the isolated candidate.
  - Running MoGe on the edited front image and pixel-bridging it to partial
    was also tested at `registration/semantic_front_moge_bridge_v35/`.  The
    semantic-image depth is not the physical partial depth (only 45.9% robust
    bridge inliers), so this observation is retained only as a diagnostic and
    is not used in the selected registration.

## Generic posterior adaptation and clean end-to-end replay — 2026-09-08

- [x] Implement a category- and axis-independent `PosteriorAdapter`: camera-aware
  unbalanced Partial OT labels stable/residual/unsupported support, and one
  coarse-to-fine embedded ARAP field updates all complete Gaussian carrier
  slots without concatenating the partial or deleting prior support.
- [x] Add structural acceptance checks.  Stable support remains fixed, hidden
  orthographic coverage must remain at least 95%, all 100k carrier slots are
  retained, accidental new small fragments are projected onto neighbouring
  rest-graph motion, and a strict no-threshold Pareto verifier rejects any
  candidate that improves selected OT anchors while worsening the complete
  Camera-1 visible 2D+3D score.
- [x] Freeze one shared configuration and validate it without GT-driven
  routing.  Custom-10 improves from CD-L1/EMD x100 `2.59310/3.28957` to
  `2.24959/3.17492`; Redwood-10 preserves CD `1.46930` and measures EMD
  `2.27689` (the baseline audit is `2.27595`, within approximate EMD solver
  variation).  Both suites pass carrier/integrity checks for all 10 samples.
  Artifacts and post-hoc audits are under
  `workspace/posterior_adapter_generalization_20260908g/`.
- [~] With the generic configuration frozen, rebuild independent Custom-10 and
  Redwood-10 runs from partial-derived depth through fresh semantic images,
  fresh 3D priors, registration, Gaussian carrier construction, and posterior
  adaptation.  Do not reuse semantic or 3D prior files from accepted replay
  directories; GT remains inaccessible until final offline evaluation.
- [ ] After the clean replay, update the canonical method document and project
  state with exact prompts, model/checkpoint paths, seeds, outputs, timing,
  qualitative boards, and final offline metrics.

## MVP-32 agentic-probe transfer — 2026-09-07

## Custom-10 fixed-mainline replay — 2026-09-07

- [x] Compare the fresh Custom-10 replay against the accepted
  `agent_genpc_plus` outputs before changing any geometry.  All ten fixed
  registration carriers and all ten Gaussian decodes are byte-identical to
  their accepted coarse-basin counterparts.  The observed Custom weakness is
  therefore a prior/observation generalization limit, not a mainline replay
  regression.  The audit is
  `workspace/custom10_fixed_mainline_20260907/metrics/custom10_fixed_mainline.{csv,json}`.

- [~] Upgrade the isolated agentic probe into a generic, no-GT closed-loop
  controller without modifying the frozen object mainline.  The controller
  now preserves a supplied partial acquisition camera, records a fixed
  no-GT diagnostic recommendation, distinguishes global evidence failure
  from local residuals, compares edited and unedited 100k carriers before
  acceptance, and makes the external clarity image a selectable candidate
  rather than an unconditional overwrite.  It also makes interrupted Qwen
  completion idempotent.  The in-progress validation root is
  `workspace/agentic_custom_pig_camera_probe_r2_20260907/`; all decisions
  use only partial-derived observations, never labels beyond prompt text,
  complete clouds, CD, or EMD.

- [~] Diagnose the pig's large hind-leg residual without weakening the
  ordinary local-anchor trust region. On the fixed Custom-10 `domestic_pig`
  registered carrier, a hard coherent-component experiment could pull the
  rear leg toward the scan, but the user rejected it after visual inspection:
  direct local anchors damaged the complete prior's structural coherence.
  The experiment and its code were removed from the active path. The retained
  candidate is intentionally simpler: fit a partial-supported *relative
  Camera-1 axis scale* after proper Sim(3), apply it coherently to all 100k
  complete-prior slots, then run the unchanged bounded local Gaussian edit.
  The candidate never overwrites the ordinary output; its use is decided only
  by a no-GT comparison in the same saved camera. On pig the scale estimate
  is active (`[.985,.982,1.033]`), and its offline CD/EMD improves from
  `4.5790/3.2211` to `3.8722/3.0915`, while the current Camera-1 verifier
  worsens from `.02988` to `.03343`. This exposes a real score conflict, so
  the ongoing Custom-10 and Redwood-10 validation will measure both evidence
  sources before any verifier rule is changed. Offline metrics remain strictly
  post-hoc and cannot be used in selection.
  - The active code path was refactored to remove the rejected coherent-hard-
    anchor experiment and its CLI plumbing. `scripts/run_camera_axis_scale.py`
    now produces the coherent scale carrier, and
    `scripts/run_mainline_gaussian.py --camera-axis-scale` composes it before
    the unchanged local Gaussian edit. Relevant unit coverage passes (`18
    passed`).
  - On Custom-10, the separate ordinary and active-scale candidates are at
    `workspace/custom10_fixed_mainline_20260907/{gaussian,gaussian_axis_global_candidate_20260907}/`.
    The post-hoc audit is
    `gaussian_axis_global_candidate_20260907/offline_custom10_axis_scale.json`.
    Selecting all six candidates whose scale fitter is active (`cartoon_fox`,
    `domestic_pig`, `handheld_power_drill`, `light_helicopter`,
    `single_engine_airplane`, `tyrannosaurus_rex_skeleton`) changes mean
    CD/EMD×100 from `2.59310/3.28957` to `2.47356/3.23329`.
  - On Redwood-10, the fair ordinary candidate and the same scale-first
    candidate are respectively under
    `workspace/agentic_redwood10_final_20260907/samples/<id>/{gaussian_baseline_candidate_20260907,gaussian_axis_global_candidate_20260907}/`.
    Offline-only audits are
    `workspace/agentic_redwood10_final_20260907/offline_gaussian_{baseline_candidate,axis_global_candidate}_20260907.json`.
    The direct Camera-1 scale evidence is active for `05452`, `06127`,
    `06145`, `06188`, `06830`, and `09639`; selecting those six reduces mean
    CD/EMD×100 from `1.50393/2.36005` to `1.46930/2.27671`.
  - The earlier scalar rendered Camera-1 objective would reject pig and
    `06830` despite their scale fitter being supported and their post-hoc
    quality improving. The agent policy therefore uses the scale fitter's
    own no-GT evidence contract (pixel-indexed visible support, independent
    axes, anisotropy, and trimmed 3-D residual reduction) to select the
    whole-carrier scale before local Gaussian editing. Rendered Camera-1
    scores remain logged for diagnosis but do not veto this distinct shape-
    extent action.

- [~] Diagnose a residual end-structure mismatch in the accepted
  `domestic_pig` axis-scale candidate.  The fixed global trimming step removes
  a small but geometrically coherent set of high-residual visible endpoint
  correspondences: its Camera-1 depth lower decile needs approximately 20%
  extension, whereas the retained bulk fit applies only 3.3% relative depth
  scale.  Test a generic endpoint-preserving whole-carrier scale candidate
  that retains already collision-free, cap-bounded correspondences through
  the scale fit, then runs the unchanged bounded Gaussian edit.  It must be
  evaluated first on pig, then on the full Custom-10 and Redwood-10 suites;
  all selection logic remains no-GT.
  - The pig audit showed this is primarily a **global registration-scale**
    error, not a Gaussian-anchor failure: Camera-1 physical-depth 1--99%
    extent is `.64096` for the partial versus `.52098` for the registered
    carrier.  The existing matched-pair scale cannot observe this because it
    is dominated by the overlapping central body.  A separate candidate now
    estimates a two-sided visible-depth extent scale and matching depth-axis
    translation directly from the partial and the registered carrier's
    z-buffered visible surface.  It expands only, so incomplete scans cannot
    shrink hidden prior support.  On pig it applies `1.23030` depth scale and
    `-.09148` depth translation before the unchanged Gaussian stage; its
    post-hoc CD/EMD×100 is `2.58708/2.98959`.  This is an isolated candidate
    at `workspace/custom10_fixed_mainline_20260907/registration_visible_depth_extent_candidate_20260907/` pending visual review and Custom/Redwood regression.
  - A subsequent residual audit showed that the apparent "short body" is not
    supported by a uniform whole-body scale: the middle 80% of the partial and
    registered prior have nearly equal principal-axis extent, whereas two
    screen-connected posterior regions share a highly coherent Camera-1 depth
    residual. The isolated structural-registration candidate therefore
    translates those two complete-prior surface components together before the
    Gaussian stage, with a graph-geodesic seam back to the untouched carrier.
    This is deliberately a registration candidate, not a direct Gaussian
    anchor edit. On `domestic_pig` it finds 1,187 visible residual pairs in
    two components, applies camera-frame translation
    `[-.03383, -.04309, -.25057]`, and moves only 18.09% of the 100k carrier
    (all 100k slots remain). The posterior-decile visible residual median is
    reduced from `.2527` to `.0249` before Gaussian editing and `.0059` after
    decoding. The post-hoc-only CD/EMD×100 is `2.64822/2.62632`; artifacts
    are under
    `workspace/custom10_fixed_mainline_20260907/registration_component_residual_candidate_20260907/domestic_pig/`.
    The proposal is unapproved and must be tested on Custom/Redwood before it
    can change the frozen public route.
  - The initial `.020/.060` core/seam field did place the legs, but visual
    inspection exposed a short transition seam at their attachment to the
    torso. The retained isolated variant is a *structural continuation*, not
    a detached component shift: it uses a `.040` core radius and `.250`
    graph-geodesic seam radius (at most 60% active carrier) so the posterior
    translation decays through its supporting torso surface. On pig this
    changes 54.57% of the carrier smoothly, retains all 100k slots, and reaches
    offline-only CD/EMD×100 `2.38834/2.52545`. Its complete artifacts are
    `workspace/custom10_fixed_mainline_20260907/registration_component_continuation_candidate_20260907/domestic_pig/`.
    This remains a candidate until a no-GT acceptance contract and
    Custom/Redwood generalization audit are complete.
  - Visual review of the later fixed-support component translation exposed a
    structural failure: it can detach the hind-leg surface from the body even
    while reducing the visible residual.  A new, isolated
    **attachment-aware axial Gaussian deformation** probe instead infers the
    residual core's graph attachment, fixes aligned partial/MoGe support, and
    applies a bounded log-scale field along a data-selected local axis.  On
    pig it reduces the selected visible-component median residual by 52.0%
    under the conservative scale bound, without translating the leg away from
    its attachment.  This is not accepted and is not part of the frozen route.
  - The pig audit further shows that the saved Camera-1 is a primary source of
    the ambiguity.  The provided `0°` camera has maximal point coverage but
    is nearly frontal and hides the hind-leg structure.  A partial-only
    oblique `-45°` reprojection exposes a recognisable side profile with the
    torso and posterior limbs simultaneously visible.  The atlas is
    `workspace/agentic_pig_view_review_20260907/diagnostics/oblique_view_candidates.png`.
    The next isolated experiment is therefore a bounded view/semantic/prior
    replan before accepting any local deformation; it must be selected by
    partial-only visibility, separability, and no-GT downstream evidence.
  - Do **not** turn the pig observation into a limb, tail, posterior, or
    category rule.  The next mesh-attached Gaussian prototype is admissible
    only for a connected residual surface patch with: (i) an inferred mesh
    attachment boundary, (ii) hard partial/MoGe support outside that boundary,
    (iii) consistent positive residual evidence in the saved and auxiliary
    partial-only projections, and (iv) a no-GT reduction of visible residual
    without support drift.  Otherwise it is the identity.  Pig is a focused
    diagnostic only; acceptance requires a registration-level Custom-10 and
    Redwood-10 audit before any public-route change.

- [x] Run the current frozen object mainline on the ten canonical
  `data/custom/<id>/partial_data/single_scan/<id>_partial.ply` scans. Reuse
  the accepted Qwen/GPT/Pixal assets from the prior Custom-10 run only after
  verifying exact partial-file hashes; rerun the current fixed registration
  and partial-anchored Gaussian stages into a fresh worktree-local output.
  Ground truth remains offline-only for the final metric audit.
  - All ten fixed registrations and Gaussian decodes completed with exactly
    100,000 points each under
    `workspace/custom10_fixed_mainline_20260907/{registration,gaussian}/`.
    The offline-only 16,384-point audit (seed 6145) is recorded in
    `workspace/custom10_fixed_mainline_20260907/metrics/custom10_fixed_mainline.{csv,json}`:
    mean CD-L1x100 `2.59309743`, mean EMDx100 `3.28957170`. The temporary
    `offline_gt/` link directory was created only for that completed audit.

- [x] Establish the paper-facing MVP completion protocol independently from
  all agent decisions: use all 41,600 `incomplete_pcds` in
  `/opt/data/private/cr/lab/GenPC/data/MVP/MVP_Test_CP.h5`, require a
  16,384-point prediction, and report CD-L2×10⁴ plus F-score@1%.  The
  currently installed H5 stores 2,048-point complete targets; the evaluator
  will therefore preserve that official target cardinality rather than
  synthetically upsample it.  Add a resumable manifest/shard contract and
  validate the metrics on the existing 32 accepted predictions before any
  full inference is scheduled.
  - `scripts/prepare_mvp_full_protocol.py` now materializes inference-only
    contiguous shards and `scripts/evaluate_mvp_protocol.py` evaluates either
    a shard manifest or an official H5 index range.  The evaluator
    deterministically FPS-samples only predictions to 16,384 points, keeps the
    official H5 complete target at 2,048 points, and reports CD-L2×10⁴ and
    F-score@1%.  `docs/mvp_evaluation_protocol.md` records the exact contract.
  - The protocol was exercised on all 32 accepted agentic-policy predictions:
    `workspace/mvp32_agentic_policy_v3_20260907/offline_mvp32_standard_16k.{json,csv}`
    reports mean CD-L2×10⁴ `152.00524014` and F-score@1% `0.22648689`.
    Complete H5 clouds were read only by this post-hoc evaluator.

- [~] Probe one verifier-justified upstream observation replan for high-error
  MVP attempts whose observed-surface support falls outside the fixed local
  edit trust region.  The isolated targets `mvp_test_14176` and
  `mvp_test_24194` begin from the accepted v3 carrier, test only the saved
  partial-only antipodal camera, and compare its completed/prior branch with
  the archived base using no-GT verifier evidence.  No category rule, metric,
  or frozen Redwood artifact may enter the decision.
  - The completed `mvp_test_14176` alternate-view branch improved its no-GT
    observed-surface coverage@1% from `.2500` to `.5259` and verifier energy
    from `.3908` to `.3655` after the fixed residual Sim(3) pass. It was
    accepted on that evidence alone; the post-hoc standard MVP audit improved
    CD-L2×10⁴ from `424.5530` to `54.9447` and F-score@1% from `.0330` to
    `.0916`.
  - `mvp_test_24194` reached zero positive pairs and infinite verifier energy,
    so the action restored its archived carrier byte-for-byte. The paired audit
    is recorded in
    `workspace/mvp32_agentic_upstream_replan_probe_20260907/offline_replan_audit_standard_16k.json`.

- [x] Add a bounded verifier-triggered upstream replan to the isolated MVP
  probe. When visible geometry is unsupported, the planner may select one
  untried partial-only saved view; the semantic/prior branch restarts and the
  rejected state is archived without touching the frozen mainline. First
  diagnostic targets: `mvp_test_26167`, `mvp_test_21341`, and
  `mvp_test_35832`, whose initial bridge had zero positive visible pairs.
  - The state machine now archives a rejected attempt, restarts only the
    upstream semantic/prior branch from an untried partial-only view, and may
    restore the archived carrier when the retry has weaker no-GT verifier
    evidence. The implementation is isolated to the probe and its Pixal cache
    is namespaced per attempt.
  - On `mvp_test_26167`, base had zero visible pairs; the antipodal replan
    reached 732 pairs after fixed residual Sim(3), and the post-hoc-only audit
    of the accepted carrier is CD-L1×100 `3.49078663`, EMD×100 `4.68531214`.
    The original MVP batch result was CD-L1×100 `78.79465818`, EMD×100
    `81.74112439`. No metric entered the replan decision.
  - On `mvp_test_35832`, the replan degraded from the archived base's 282
    positive pairs / finite verifier energy `.2511` to zero pairs / infinity;
    it therefore restored the archived base carrier. The resulting offline
    audit is CD-L1×100 `13.40282261`, EMD×100 `20.08180618`.
  - `mvp_test_21341` could not be repaired by its antipodal retry alone, so it
    motivated the recovery action below rather than a sample-specific view
    rule.

- [x] Add a verifier-triggered global recovery policy for sparse/out-of-
  distribution partials, isolated from the frozen mainline.  It replays a
  finished agentic attempt, leaves a candidate untouched whenever its verifier
  has at least six positive visible Camera-1 correspondences and finite energy,
  and otherwise invokes the fixed rendered 2-D center/scale Sim(3) rescue
  followed by the ordinary residual solver.  The final action compares the
  rescue and refinement using only the no-GT verifier.  It neither routes by
  category nor reads complete clouds, CD, or EMD.
  - The three MVP zero-evidence cases (`mvp_test_21341`, `mvp_test_26167`,
    `mvp_test_35832`) respectively recover to 1218, 1195, and 1196 visible
    pairs.  Their post-acceptance audit is CD-L1×100 `5.4522/2.7460/9.4612`
    and EMD×100 `9.2697/3.7787/12.7903`, compared with the static attempts'
    `42.6299/78.7947/35.3629` CD-L1×100 and
    `49.2621/81.7411/34.0909` EMD×100.  Offline metrics were computed only
    after the action traces had reached `accepted`.
  - A second, still generic severity tier treats high dimensionless verifier
    energy together with a poor silhouette or coverage as a **candidate
    proposal**, never as a forced replacement.  On the two extra MVP probes,
    `mvp_test_21480` wins the verifier comparison and improves from
    CD/EMD×100 `16.2882/20.4582` to `14.2718/12.9174`; `mvp_test_13625` loses
    the comparison and its baseline PLY is restored byte-for-byte.
  - The current full 32-sample replay is
    `workspace/mvp32_agentic_policy_v3_20260907/`.  Five samples opened a
    recovery proposal; one was rejected by the verifier, so 28 final PLYs are
    byte-identical to the prior agentic outputs and only four changed.  Its
    offline-only audit at 16,384 points / seed 6145 is CD-L1×100
    `5.15166616`, EMD×100 `7.14666393`, versus the prior probe's
    `9.56243024/11.73469804`.
  - A read-only audit of the accepted Redwood-10 agentic states finds finite
    evidence for every sample (minimum 11,942 positive pairs), so this branch
    triggers `0/10` times and leaves the frozen Redwood result untouched.

- [~] Materialize a class-balanced MVP Completion transfer set: two random
  partial views for each of the 16 official H5 categories, with canonical
  partial coordinates preserved and complete clouds isolated for offline-only
  evaluation. The inference manifest is
  `workspace/mvp32_agentic_probe_20260907/inference_manifest.json`.
- [x] Execute the bounded agentic observation → native prior → cross-camera
  alignment → residual-refinement → diagnostic local-adaptation loop without
  GT access. All 32 state traces reached `accepted`; each final output has
  exactly 100,000 points under
  `workspace/mvp32_agentic_probe_20260907/samples/<case>/final/`.
  The offline-only 16,384-point audit (seed 6145) is
  `workspace/mvp32_agentic_probe_20260907/offline_mvp32_metrics.{csv,json}`
  with mean CD-L1×100 `9.56243024` and EMD×100 `11.73469804`.
- [x] Make the shared saved-camera z-buffer robust to an all-out-of-frame
  complete-prior proposal. It now emits an empty depth raster so the proposal
  is scored as unsupported instead of aborting a cross-camera candidate sweep.
  The change is sample-agnostic; a direct empty-projection smoke and
  `tests/test_bidirectional_cycle_registration.py` (4 passed) verify it.
  Resume the halted second MVP alignment from its unchanged
  `global_alignment_pending` state.
- [x] Make visible pixel-residual stages deterministic under sparse evidence:
  if fewer than six positive Camera-1 pairs exist, their residual lattice is
  the identity proposal and the native/bridge transform remains active. This
  preserves the same proper-Sim(3) route without synthetic correspondences.
  `tests/test_bidirectional_cycle_registration.py` plus
  `tests/test_visible_pixel_sim3_refinement.py` pass (9 total).

## Scene-level instance completion — 2026-09-05

- [x] Keep the frozen single-object GenPC+ path unchanged and add an isolated
  scene wrapper under `src/scene_completion/` plus one public scene runner.
- [x] Use the same Pixal MoGe-2 inference contract on an RGB scene image to
  establish one shared camera-coordinate visible scene cloud.  Consume
  auditable GPT instance masks to extract object **MoGe partials** in that
  frame. Verified on the coffee-mug preparation smoke at
  `workspace/scene3_gpt_mask_poc/run`: 107,203 mask-indexed scene-MoGe points.
  - The scene default uses a 5-pixel mask erosion before this extraction so
    unstable MoGe edge points cannot dominate downstream registration.
- [x] Use direct GPT RGB semantic completion from every masked scene crop—no
  depth-to-semantic stage—then run unchanged Pixal per instance. Camera-1
  metadata is materialised only to estimate the camera-2 Pixal-input MoGe →
  camera-1 scene-MoGe bridge; the two MoGe observations are never treated as
  the same camera frame.
  - The direct-GPT → Pixal smoke completed for `coffee_mug_0` under
    `workspace/scene3_gpt_mask_poc/run/`; Camera-1 and Pixal consume the same
    pose-locked direct GPT semantic image.
- [x] Stop scene registration after native Pixal--MoGe alignment and the
  pixel-indexed camera-2 → camera-1 bridge. Do not run joint, amplified,
  wide-tilt, final Camera-1 residual, Gaussian fusion, or point decoding.
  Compose each textured `pixal3d.glb` using the bridge's original-Pixal →
  scene-partial Sim(3), which preserves the scene partial's original position
  and scale rather than leaving meshes in object-local frames.
  - Verified all seven `scene_3` instances at
    `workspace/scene3_all_instances_mesh_erosion5_20260905/run/registration_bridge_only_20260905`.
    The separate scene output is
    `workspace/scene3_all_instances_mesh_erosion5_20260905/run/scene_meshes_bridge_only_20260905/`.
    Every bridge-placed complete bbox centre is near its corresponding
    scene-partial centre; the merged GLB is 60 MB and no duplicate transformed
    instance GLBs are written by default. Scene Pixal generation now exposes a
    conservative explicit 100k-face target. `pytest -q` passed 55 tests.
  - Final GLB composition now follows the reference `3D-Reconstruction`
    convention exactly: apply the full scene transform to each mesh's vertices
    first, then construct a `trimesh.Scene` from those baked textured meshes.
    This preserves independent PBR materials without leaving nested GLTF node
    transforms for a downstream viewer to miss. Verified output:
    `workspace/scene3_all_instances_mesh_erosion5_20260905/run/scene_meshes_baked_scene_nodes_20260905/`;
    all seven final scene nodes have identity transforms, while their vertices
    retain distinct shared-scene positions. `pytest -q` passed 57 tests.
  - Add an exact, scene-only mesh collision refinement after all meshes are in
    the common table-world frame. It retains Sim(3), texture, and image-plane
    placement; for each actual FCL collision it moves only the instance that
    is farther according to its source scene-MoGe mask-anchor depth, along the
    original camera's positive depth direction by the minimum collision-free
    amount. The rule therefore does not depend on category or generated-prior
    mesh centre. On `scene_3`, the only initial pair
    was mouse--keyboard; `black_mouse_0` moved `0.070359375` while all six
    other instances stayed fixed. The final textured GLB at
    `workspace/scene3_all_instances_mesh_erosion5_20260905/run/scene_meshes_scene_moge_collision_exact_20260905/`
    has zero exact FCL collision pairs. `pytest -q` passed 59 tests.
  - [x] Run the same scene-image → textured-scene-mesh route on the remaining
    four source scenes at `workspace/scene_batch_all_instances_20260905/`.
    The frozen direct-GPT inputs and scene-MoGe partials produced 20 Pixal
    instances, then native bridge placements and baked final GLBs:
    `scene_{1,2,4,5}/run/scene_meshes/completed_scene_registered_meshes.glb`.
    Verification confirms 4/4/7/5 textured geometries respectively and
    identity GLB nodes, so all object coordinates are baked in the shared
    scene frame. The corrected `scene_5/left_armchair_0` uses the user's
    revised full-chair mask.
    - Exact collision refinement now clears every detected pair in all four
      scenes. The previous `.5` object-scale cap was removed: the solver
      brackets and bisects the smallest collision-free translation along the
      original scene camera's positive depth axis, without arbitrary lateral
      movement. On the re-exported outputs, `scene_2/camping_tent_0` moves
      `2.977501`; `scene_4` moves `coffee_table_0`/`left_side_table_0`/
      `right_side_table_0` by `2.115`/`.318`/`.43575`. All four final GLBs
      have zero remaining exact FCL collision pairs.
  - [x] Publish the scene method alongside the object mainline. The root
    README, method specification, documentation index, and dedicated
    `docs/scene_completion.md` now define the scene-MoGe/GPT/Pixal route,
    the external saved-GPT asset contract, bridge-only registration boundary,
    textured GLB assembly, unbounded minimum camera-depth collision resolution,
    and the complete reproducibility bundle layout.

## Mainline consolidation — 2026-09-04

- [x] Preserve the accepted implementation before pruning.
  - Backup ref: `best_register-pre-mainline-prune-20260904` at `ce9fdeb`.
- [x] Keep only the fixed Qwen/GPT/Pixal input path, native Pixal--MoGe
  registration, Camera-1 continuation, partial-anchored Gaussian edit/decode,
  and offline evaluation.
- [x] Remove historical agent loops, PCA/ICP/FreeReg probes, Hunyuan/TRELLIS
  branches, old registration variants, and their tests/docs.
- [x] Run a one-sample end-to-end downstream smoke test on `01184` using the
  accepted retained Qwen/GPT/Pixal inputs. Verify final registration and
  100k-slot decoded output before committing the consolidation.
  - Verified at `workspace/mainline_prune_smoke_20260904`: fixed registration
    completed with no GT/metric input, and the frozen `.075/.0015/.01/.10`
    Gaussian route emitted a complete 100k-point prediction. The compact test
    suite passes `38/38`.
- [x] Commit the compact mainline files as `3070253` (`Prune repository to
  fixed Pixal MoGe mainline`). Existing unrelated `third_party/FreeReg`
  deletions and untracked `data/custom/` remain outside the commit.

## Upstream semantic/Pixal integrity correction — 2026-09-04

- [x] Restore the actual frozen saved-view selection used by the accepted
  Redwood assets: deterministic 256-view Fibonacci coverage, partial-only
  front/back depth tie-break, and visible-point rasterisation. The accidental
  single reference camera was removed.
- [x] Correct raw-depth export so Qwen receives the normalized grayscale depth
  raster rather than the RGB sparse point rendering.
- [x] Make worktree stage runners resolve shared model/source directories under
  `/opt/data/private/cr/lab/GenPC/models`.
- [x] Verify `01184` from scratch at
  `workspace/mainline_upstream_smoke_20260904`: Qwen -> GPT clarity edit ->
  Pixal3D -> fixed registration -> partial-anchored Gaussian decode -> offline
  metric. The saved-view `point_uv.npy` exactly matches the accepted camera
  asset; 38 tests pass. Final offline result: CD-L1×100 `1.1616`, EMD×100
  `1.8050` (16,384 points, seed 6145). The metric was evaluated only after the
  prediction was frozen.

## Mainline performance preservation — 2026-09-04

- [x] Move Redwood defaults to `data/redwood/partial` and `data/redwood/gt`
  across semantic generation, fixed registration, input materialisation, and
  offline evaluation. The four runtime entry points now share canonical path
  helpers, preventing a future split between partial and GT defaults.
- [x] Replace point-by-point Python z-buffer loops with a vectorised
  deterministic rasteriser that preserves the former depth and offset-order
  tie rule exactly. Cache immutable partial-camera projections during each
  local Sim(3) candidate sweep.
- [x] Run the six fixed registration stages in one Python process by default,
  retaining `--no-in-process` as a diagnostic subprocess fallback. A full
  `01184` registration produced a byte-identical final PLY.
- [x] Make Pixal resume check completed GLB/PLY pairs before initialising
  Pixal/DINO/MoGe. A completed `01184` now resumes in 5.8 seconds.
- [x] Profile and regression-check `01184`: the final Camera-1 stage decreased
  from 26.78 s to 13.48 s, while its 100k PLY remained byte-identical. The
  complete registration and decoded Gaussian PLY also remained byte-identical;
  final offline CD-L1×100 was unchanged at `1.1616` (EMD is CUDA-nondeterministic
  at the fourth decimal on an identical PLY). Test suite: 40 passed.
- [x] Materialize the existing FP16 Pixal-input MoGe observation during a fresh
  Pixal run and reuse it in native registration after an SHA-256 input check.
  This removes a later model reload/inference without changing the method. On
  `01184`, direct and cached native target/registered PLYs were byte-identical;
  the full fixed registration was also byte-identical when both paths consumed
  the current `data/redwood/partial` input. Test suite: 40 passed.
- [x] Parallelize only the independent Camera-1 candidate scores (default:
  eight workers), keeping proposal order and the sequential minimum tie rule.
  On the current `01184` Redwood input the full fixed registration decreased
  from `103.50 s` to `94.54 s`; every emitted registration PLY was
  byte-identical. Test suite: 41 passed.

## Open-source mainline documentation and packaging — 2026-09-05

- [x] Replace public documentation with one concise description of the fixed
  Qwen/GPT/Pixal → registration → Gaussian mainline and its reproducible run
  contract. The retained public set is root README plus `docs/{README,
  installation,models,core_registration_pipeline}.md`.
- [x] Add an installable `pyproject.toml` with the tested Python/CUDA runtime
  constraints and a development test extra; remove superseded packaging files.
- [x] Document external model/source dependencies, required local layout,
  checkpoint provenance, licences, and the separate model download step.
  The Pixal source commit is fixed in `docs/models.md`; checkpoints remain
  external and are never redistributed with the code.
- [x] Validate package metadata and documentation commands without changing
  inference outputs; retain `PROJECT_STATE.md` only as the internal record of
  accepted mainline runs. `pip install --no-deps -e .`, local-link validation,
  py-compile, and the 41-test suite all pass.

## Mainline code-structure stabilization — 2026-09-05

- [x] Centralize the fixed Redwood sample set and the registration/Gaussian
  artifact contract in one importable module, while retaining the current
  legacy-path fallbacks for existing experiment folders.
- [x] Make `src` an explicit package and replace duplicated constants in the
  six public stage runners without changing any CLI defaults or frozen
  numerical parameters.
- [x] Add path-contract unit tests and run a one-sample byte-level registration
  regression against the accepted `01184` output before committing.
  - `workspace/structure_cleanup_regression_20260905/01184/final/`
    reproduced the accepted final PLY byte-for-byte (SHA-256
    `8625684013a23b8266872a4a5dd6813fbb63ae6ba800b0a9fa4525e677c6c4e6`).
    `pytest -q` passed `44/44` after adding the path-contract coverage.

## Fixed-mainline global parameter calibration — 2026-09-05

- [x] Freeze the Qwen → GPT → Pixal assets, the staged two-camera Sim(3)
  route, and the partial-anchored Gaussian algorithm. The only experimental
  degrees of freedom are globally shared registration/edit numerical
  parameters; no sample/category routing or inference-time GT access is
  permitted.
- [x] Record the accepted ten-sample offline baseline from
  `workspace/relaxed_anchor_gaussian_redwood10_20260904`: CD-L1×100
  `1.58197163`, EMD×100 `2.51144224` (16,384 points, seed 6145).
- [x] Run a pre-registered Redwood-10 Gaussian edit sweep in fresh output
  roots, evaluate every complete candidate offline, and identify globally
  robust settings from mean and per-sample deltas.
  - The accepted edit caps remain `.075`; `.08` and `.09` both regressed.
    A shared saved/virtual correspondence radius of `1.5 px` improved the
    baseline to CD-L1×100 `1.58079308`, EMD×100 `2.50663875`.
- [x] Starting from the fixed accepted wide-tilt registrations, test only
  global Camera-1 final-continuation trust regions; run the selected setting
  through the selected Gaussian edit and compare the complete ten-sample
  output to the baseline.
  - With the same `wide_tilt` input and `1.5 px` edit, final trust regions
    `.5/.75/1.0/1.25` degrees gave, respectively,
    `1.5723614/2.4918336`, `1.5700483/2.4854301`,
    `1.5691580/2.4829483`, and `1.5728989/2.4932225` (CD-L1×100 / EMD×100).
    The `1.0` degree setting is therefore the strongest tested global value.
  - Holding that `1.0°` registration fixed, the shared saved/virtual radii
    `.5/1.0/1.5 px` gave `1.5686855/2.4826558`, `1.5682055/2.4785676`, and
    `1.5691580/2.4829483`. Thus the central `1.0 px` radius is selected on
    both metrics, without any sample-specific route.
- [x] Freeze the strongest reproducible global parameter set in the public
  defaults/docs, run tests plus a clean ten-sample audit, and record the final
  accepted output without changing upstream generation assets or method route.
  - Defaults are final Camera-1 trust region `1.0°` and saved/virtual positive
    correspondence radii `1.0 px`; every other fixed numerical setting remains
    at the accepted `.075/.0015/.01/.10` configuration. The full frozen audit
    is `workspace/fixed_mainline_calibration_20260905/combo_final100_pixel100`
    with CD-L1×100 `1.5682054963`, EMD×100 `2.4785676040`. All ten decoded
    outputs contain 100,000 points. A no-override 01184 rerun at
    `workspace/fixed_mainline_default_smoke_20260905` reproduced the selected
    registration and decoded PLY byte-for-byte; `pytest -q` passed `44/44`.

## Custom non-orthogonal evaluation — 2026-09-05

- [x] Run the fixed Qwen → GPT → Pixal → registration → Gaussian mainline on
  the ten renamed custom scans using their regenerated non-orthogonal partial
  views. Keep all method parameters fixed; read custom GT only after final
  predictions are frozen for offline reporting.
  - The initial semantic batch was stopped after an input-camera audit:
    reselecting a 256-view camera from a sparse partial punctured the depth
    raster and changed the observation view. The custom batch will use the
    recorded partial-camera `viewpoint.npy` for Camera-1 depth rasterisation;
    this is camera metadata only, with no complete-geometry access.
  - Hydra's original `59° / 27°` partial camera occluded one head. A
    pre-generation oblique-view diagnostic finally selected `289° / 18°`,
    which separates all three heads and necks while retaining the complete
    oblique body; regenerate only this partial and dependent semantic assets
    before resuming the batch.
  - The custom category-only prompts were refined before Qwen generation:
    hydra explicitly requires exactly three distinct heads, while mirrorless
    camera requires one central lens and a complete camera body. These are
    category descriptions, not evaluation-time routing or GT supervision.
  - Pixal startup required a runtime-only namespace isolation because NAF's
    Torch-Hub checkout imports its own top-level `src` package. This avoids a
    collision with GenPC+'s `src` package without changing Pixal checkpoints,
    sampler settings, or exported priors.
  - Qwen/GPT/Pixal input preparation is complete in
    `workspace/mainline_results_20260905/custom/full_pipeline/inputs/`. All ten samples now
    have the recorded partial-view depth/camera assets, semantic image,
    geometry-preserving GPT image, Pixal GLB, 100k-point prior, and cached
    Pixal-input MoGe observation. No custom GT was read during those stages.
  - Fixed registration and partial-anchored Gaussian decoding completed for
    all ten samples. Every final prediction contains 100,000 points under
    `workspace/mainline_results_20260905/custom/full_pipeline/gaussian/<sample>/decoded/`.
    The subsequent offline-only 16,384-point audit (seed 6145) is recorded at
    `workspace/mainline_results_20260905/custom/full_pipeline/metrics/offline.{json,csv}`:
    mean CD-L1×100 `13.6373`, mean EMD×100 `12.8285`. GT was first accessed
    only by this final audit.

## Camera-1 coarse basin recovery diagnostic — 2026-09-05

- [x] Add an optional, globally shared coarse visible-surface Sim(3) stage
  before the fixed residual continuation. It must use no GT or category route:
  pixel-indexed visible 3-D pairs propose broad candidates, while a
  pixel-residual plus saved-view projection objective selects from identity and
  those candidates. Test it in an isolated output root on the four custom
  cases whose current Camera-1 score has zero or near-zero 3-D pairs:
  `single_engine_airplane`, `light_helicopter`,
  `tyrannosaurus_rex_skeleton`, and `wolf`.
  - The isolated registration root is
    `workspace/mainline_results_20260905/custom/full_pipeline/registration_coarse_basin_20260905`.
    Each case selected the full shared pixel-residual candidate and restored
    valid final Camera-1 3-D pairs: `7093`, `4778`, `5287`, and `7057` in the
    order listed above (the old route had `0`, `0`, `0`, and `221`).
  - With the unchanged Gaussian edit/decode and the same offline 16,384-point
    metric seed, CD-L1×100 / EMD×100 changed from
    `41.0854/38.2824`, `40.8114/29.2885`, `21.2220/22.8760`, and
    `13.8432/15.5566` to `0.7406/1.5297`, `1.3764/2.5790`,
    `0.9152/1.9992`, and `3.1437/4.0370`. The four-sample mean is
    `29.2405/26.5009 → 1.5440/2.5363`. Outputs and the offline comparison
    are retained under `workspace/mainline_results_20260905/custom/coarse_basin_hard4/`.
  - The flag remains opt-in pending visual and broader-distribution validation;
    the frozen default mainline is not silently changed.
  - [x] Run the same opt-in coarse stage on the remaining six custom samples
    (`handheld_power_drill`, `mirrorless_camera`, `dragon_character`,
    `hydra_creature`, `domestic_pig`, `cartoon_fox`) and compare their frozen
    downstream predictions offline before deciding whether to promote it.
    - The shared selection chose a non-identity coarse candidate for five
      samples and identity for `hydra_creature`. With unchanged downstream
      Gaussian parameters, the six-sample mean changed from CD-L1×100 / EMD×100
      `3.2352/3.7246` to `3.2925/3.7962`; the largest regressions were still
      small (`+0.2179/+0.2755` on the handheld drill). Combining this audit
      with the four diagnosed failures gives `13.6373/12.8351 → 2.5931/3.2922`
      over all ten custom samples. Results are in
      `workspace/mainline_results_20260905/custom/coarse_basin_remaining6/`.
  - [x] Run the same opt-in stage on the frozen Redwood-10 Qwen/GPT/Pixal
    inputs and compare its complete offline audit with the accepted fixed
    baseline before changing the default.
    - The isolated result is
      `workspace/mainline_results_20260905/redwood/`. The unchanged
      16,384-point offline audit (seed 6145) is CD-L1×100 / EMD×100
      `1.5738481/2.4813253`, versus the frozen `1.5682055/2.4785676`:
      `+0.0056426/+0.0027576` (`+0.36%/+0.11%`). Three samples selected
      identity; the other seven selected shared pixel-pair fractions. The
      largest per-sample change is 06188 CD `+0.0365`; no catastrophic
      regression occurred. Keep the stage opt-in until a further global
      decision, rather than silently replacing the accepted mainline.

## Default coarse-basin registration promotion — 2026-09-05

- [x] Promote the globally shared broad Camera-1 pixel-Sim(3) basin-capture
  step to the default registration route at the user's direction. Its
  identity-inclusive fixed candidate lattice remains the only selection rule;
  no category/sample route, proposal rejection, or GT/CD/EMD input was added.
- [x] Update the canonical method documentation and public run instructions
  to describe the new stage and retain `--no-coarse-basin-recovery` solely as
  an ablation switch.
- [x] Run a fresh one-sample no-override registration plus Gaussian smoke
  path, then verify the manifest reports coarse capture enabled and the final
  decoded cloud has 100,000 points.
  - `workspace/default_coarse_mainline_smoke_20260905` ran 01184 without an
    explicit coarse flag. The registration manifest records `true`, the joint
    record selected `pixel_pair_fraction_1.000`, and the decoded cloud has
    100,000 points. Its SHA-256 is
    `e7cdbd3d256b044f592d495c7feb175d4628cb1e89f395f1c67e7776219cec09`,
    byte-identical to the corresponding accepted Redwood coarse-ablation
    output. `pytest -q` passed `45/45`.

## GPT-visual unknown-camera selection diagnostic — 2026-09-05

- [x] For an uncalibrated ScanSalon partial, render an orbit atlas solely for
  GPT-5.6 Terra visual inspection. The agent must choose the final continuous
  observation direction from semantic recognisability, not from a
  geometry-scored/PCA candidate selector. Save the selected Camera-1, readable
  extrinsics, depth raster, per-point UVs, and rationale; first validate on
  `data/custom/wolf/partial_data/single_scan/wolf_partial.ply`.
  - GPT selected the semantic side view `(azimuth, elevation) = (0°, 10°)`.
    The output root is `workspace/gpt_visual_view_diagnostic/wolf/`; it retains
    the 24-view inspection atlas, full target-aware `camera.pth`, readable
    `semantic_view_selection.json`, raw/filled depth, foreground mask, and
    per-point UVs. All 16,384 points are forward-facing in the stored camera;
    the mainline-identical sparse-depth raster has 6,482 observed pixels and
    only infills its local support ring. The original ScanSalon car
    probe was intentionally abandoned because it has only 796 points, below
    the new 1,000-point ScanSalon quality floor.

## GPT-visual saved-depth Redwood/Custom diagnostic — 2026-09-05

- [x] Apply the same GPT-visual, unscored view selection to Redwood-10 and
  Custom-10 partials in an isolated output root. Preserve the frozen mainline
  depth rasterisation exactly after the continuous direction is selected; do
  not overwrite accepted Camera-1, semantic, Pixal, registration, or Gaussian
  outputs.
  - All 20 outputs are in
    `workspace/gpt_visual_depth_redwood_custom_20260905/{redwood,custom}/<sample>/`.
    Each directory contains the visual-only orbit atlas, GPT selection record,
    full target-aware camera, raw/final depth, inpaint mask, and per-point UVs.
    The selection source is explicitly logged as GPT-5.6 Terra visual
    inspection with no geometry-score ranking. Dataset-level selected-depth
    boards and CSV summaries are at the output root for manual review.

## Scene registration throughput — 2026-09-05

- [x] Add an opt-in, sample-independent multi-process scheduler to the fixed
  registration runner. It must preserve each sample's frozen stage order and
  numerical route; only independent samples may execute concurrently. Start
  with a conservative two-worker default for the scene wrapper on the 24GB
  GPU, retain one worker as the exact serial ablation, and validate equality
  on completed/resumed scene outputs before promoting the option.
  - `--sample-workers 2` uses two isolated child processes. Object mode keeps
    its native-MoGe → bridge → joint → amplified → wide-tilt → final sequence;
    scene mode intentionally stops after its native-MoGe → camera-2-to-camera-1
    bridge. Single-object mode remains serial by default.
  - Resume scheduling over the seven completed `scene_3` instances preserved
    every final PLY SHA-256. A full two-object concurrent smoke run at
    `workspace/scene3_all_instances_mesh_erosion5_20260905/parallel_registration_smoke`
    used 12.7GB on the 24GB GPU, completed both independent registrations in
    about 60.3 seconds (per-object artifact span), and reproduced the serial
    final PLYs byte-for-byte for both the mug and mouse.

## Scene-3 complete textured-mesh reconstruction — 2026-09-05

- [x] Run the direct-RGB scene wrapper across all seven visible `scene_3`
  instances using shared Pixal-MoGe partials, direct pose-locked GPT semantic
  assets, fixed registration, and no point/Gaussian fusion.
  - The retained output is
    `workspace/scene3_all_instances_mesh_erosion5_20260905/run/`. It contains
    all scene masks/partials/cameras/direct semantic assets, seven Pixal GLBs,
    seven final registered 100k PLYs, seven individually transformed textured
    GLBs, and `scene_meshes/completed_scene_registered_meshes.glb`.
  - For every instance, applying the cumulative saved Sim(3) to its Pixal
    100k samples reproduced its final registered PLY with maximum absolute
    coordinate error at most `8.882e-16`. This verifies that the merged scene
    is precisely the registered mesh composition, not an additional pose
    estimate or a lossy fusion stage.

## Isolated agentic prior-adaptation validation — 2026-09-06

- [x] Create a clean `codex/agentic-01184-probe` worktree from the accepted
  `best_register` baseline. Implement a state-hashed, bounded decision loop
  that lets a high-level planner choose only discrete tools while retaining
  the frozen geometric executors.
- [x] Complete an end-to-end, no-GT pilot for Redwood `01184`. The planner
  selected the partial-only base view, depth-conditioned semantic completion,
  native prior generation, cross-camera global alignment, residual Sim(3),
  one prior-protected local adaptation, then acceptance. The accepted output
  is `workspace/agentic_redwood10_final_20260907/samples/01184/final/agent_selected_100k.ply`.
  Its verifier record is stored with the state trace; CD/EMD was not exposed
  to any decision.
- [x] Expand the same isolated controller to the remaining Redwood-10 samples
  (`05117`, `05452`, `06127`, `06145`, `06188`, `06830`, `07136`, `07306`,
  and `09639`). All tool decisions are state-hashed and no GT/CD/EMD was
  exposed before `ACCEPT`.
  - The first `07306` trace exposed a real agent failure: the legacy
    convex-hull saved-view tie-break selected the semantic rear. The retained
    front-facing re-plan at
    `workspace/agentic_redwood10_final_20260907/samples/07306/` selected the
    antipodal `opposite_180` view as an explicit agent action.
    Its no-GT verifier energy fell from `0.18598` (old rear trace) to
    `0.10856` after the identical global/residual Sim(3) executors. Its
    offline-only accepted result is CD-L1x100 `3.1593`, EMDx100 `3.3834`,
    compared with `22.9567/26.8012` for the rejected rear interpretation.
  - The ten accepted traces, with this corrected `07306` substituted and no
    post-acceptance metric feedback, are consolidated under
    `workspace/agentic_redwood10_final_20260907/`. A fresh offline audit at
    16,384 points reports CD-L1x100 `1.58338`, EMDx100 `2.34539`.
- [x] Diagnose why the corrected `07306` still looked less surface-tight than
  the retained static mainline result. The front-view registered prior is not
  the problem: its partial-to-prior 1% coverage is `88.35%`, higher than the
  retained mainline registration's `85.17%`. The controller accepted too
  early because its original verifier used only rendered overlap. The static
  Gaussian decode instead raises its own coverage to `99.25%` through
  collision-free one-to-one partial anchors.
  - A conservative fixed 2% agent-side edit on the corrected `07306` trace
    increases coverage to `97.83%`, yields CD-L1x100 `3.0876` and EMDx100
    `3.3050`, retains all 100k prior slots, and caps editable-Gaussian motion
    at 2% of the partial diagonal. These metrics are offline-only and were
    not used to select the action.
- [ ] Re-run the bounded planner with the new general observed-surface support
  diagnostic. It records partial-to-prior 1%/2% coverage and p90/p95 distance
  ratios, then *recommends* (but does not force) `ADAPT_LOCAL` only when the
  residual lies inside the globally fixed 2% trust region; otherwise it
  recommends upstream re-planning. This closes the early-acceptance gap
  without a `07306`-specific route. Validate it first on `01184` (thin wheel
  protection) and corrected `07306`, then reassess all ten traces.
- Risk: this is a bounded feasibility study, not yet evidence that an MLLM
  planner improves a fixed pipeline. Any eventual paper claim must report a
  static-executor control and must not relabel static outputs as agent outputs.

## Partial-supported camera-axis scale probe — 2026-09-07

- [x] Add an opt-in local-scale executor for the isolated agent probe.  It
  keeps global pose and isotropic scale in proper Sim(3), estimates only a
  relative residual in the saved Camera-1 horizontal/vertical/depth frame,
  and injects it as a weak target into the existing partial-anchored Gaussian
  graph.  Unsupported axes remain fixed.  Direct scale targets are restricted
  to Camera-1 z-buffer-visible prior means close to real partial anchors;
  disconnected/occluded prior components retain existing protection.
  - Activation is data-independent and no-GT: >=512 collision-free visible
    anchors, >=2 independent visible directions, >=8% axis spread, relative
    log stretch >=.025, and >=3% trimmed residual reduction over its uniform
    counterpart.  The fixed cap is .16 log scale and the weak graph weight is
    .006.
  - On `domestic_pig`, the camera-axis probe first established an observed
    relative depth scale of `1.03294`.  The unrestricted connected-component
    variant gave a large offline gain but also over-pulled `cartoon_fox`, so
    it was rejected as a general method.  The retained visible-support variant
    is at `workspace/agentic_camera_axis_visible_support_probe_20260907/`:
    it improves pig's no-GT Camera-1 energy from `0.02987722` to `0.02970595`
    and its offline-only CD-L1x100 / EMDx100 from `4.579000 / 3.222942` to
    `4.572345 / 3.206997`, while preserving the unobserved carrier.
  - A Custom-10 no-GT audit activated the camera-scale candidate for fox, pig,
    drill, helicopter, airplane, and T-rex; strict Camera-1 candidate
    comparison retained fox, pig, drill, and helicopter and rejected airplane
    and T-rex.  The visible-support changes are small and their offline
    aggregate is not yet convincingly better, so this remains an agentic
    probe rather than a promoted mainline stage.
- [ ] Design a stronger *local structural support* test (rather than an
  object-wide connected-component axis stretch) before evaluating this tool
  on Redwood-10.  Do not promote it based on the current Custom-10 metrics;
  any new criterion must remain partial-only and must preserve the ordinary
  Gaussian candidate as a no-GT fallback.

## Topology-preserving pig scale feasibility probe — 2026-09-07

- [x] Verify a category-free remedy for the `domestic_pig` end-structure
  mismatch without changing the frozen mainline.  The partial-visible
  principal-extent estimator is active: the registered carrier's visible
  extent `.78012` is shorter than the partial extent `.97011`; it applies a
  `1.24354` expansion about the already aligned low endpoint (endpoint error
  `.00015`).  This is a whole-carrier affine scale, selected from partial and
  Camera-1 evidence only; no GT/CD/EMD is consulted.
- [x] Transfer the slot-preserving carrier adaptation exactly to the original
  textured Pixal mesh, rather than moving independently sampled points.  The
  fitted affine has max carrier residual `6.14e-16`, determinant `1.24354`,
  and singular values `[1.24354, 1, 1]`.  The output keeps all `240106`
  vertices and `283115` faces, UVs, and PBR material unchanged:
  `workspace/custom10_fixed_mainline_20260907/pig_global_axis_scale_probe_20260907/textured_mesh_principal_extent/`.
  The sampled diagnostic is `pig_mesh_principal_extent_overlay.png`.
- [~] A UV-seam-welded mesh/3DGS-compatible local ARAP executor was added at
  `src/mesh_attached_gaussian_deformation.py`; coincident UV-chart vertices
  are joined only in the deformation graph, never in the exported textured
  mesh.  Its strict no-fold gate correctly rejects the current pig residual
  as too large for a *local* edit before the global scale correction.  Keep
  local ARAP disabled for this candidate.  A later generic test may run it
  only after global extent calibration, and must retain the global-only output
  when its no-GT support or topology checks fail.

## Camera-conditioned Gaussian adaptation probe — 2026-09-08

- [x] Replace the rejected direct textured-mesh local edit with a
  slot-preserving Gaussian-mean field.  `src/camera_conditioned_gaussian_adaptation.py`
  partitions Camera-1-visible prior means into coherent residual controls,
  low-residual locked support, and unobserved protected support.  It solves a
  screened vector field from pixel-indexed camera-plane **and depth** residuals
  on a locally pruned Gaussian graph.  The decision uses partial, the saved
  camera and the registered prior only; it never reads GT/CD/EMD or class
  names.
- [x] Verify the basic invariants with
  `tests/test_camera_conditioned_gaussian_adaptation.py`: depth-supported
  controls move, selected aligned support is fixed exactly, and Gaussian
  slots are preserved.  The focused suite (13 tests) passes.
- [x] Run the candidate on `domestic_pig` after its accepted global
  principal/depth extent correction.  The stronger, still category-free
  residual trigger (`5%` of the partial diagonal) produces
  `workspace/custom10_fixed_mainline_20260907/pig_camera_conditioned_gaussian_20260908c/`.
  It selects 562 coherent controls, locks 2,091 low-residual visible means,
  protects 48,015 unobserved means, and moves 47,687 means through a
  continuous graph field.  Its held-out-from-decision Camera-1 control
  residual is reduced by `28.84%` and its physical depth residual by `29.22%`;
  local edge-stretch percentiles are `[0.894, 1.175, 1.745]` at 1/99/99.9%.
  These are feasibility diagnostics only, not GT-selected results.
- [~] Extend the diagnostic to a three-stage, lock-preserving coarse-to-fine
  pass.  Stages 2--3 inherit every previously locked Gaussian slot, while the
  matching radius expands only from 1px to 3px then 5px and the residual
  support is tightened to independently coherent screen components.  The
  latest third-stage candidate is
  `workspace/custom10_fixed_mainline_20260907/pig_camera_conditioned_gaussian_20260908f/`:
  it uses three components (294 controls), inherits 2,100 locked slots,
  protects 61,354 unsupported slots, and reduces its stage-local no-GT
  projection/depth residual by `39.83%/39.81%` with 12%-diagonal movement cap.
  It is deliberately still unaccepted: inspect its leg overlay with the user
  before investigating a distinct ray-extension action for partial pixels
  outside the rendered prior silhouette.
- [ ] Do not promote the Gaussian adaptation to the frozen mainline until it
  is shown to preserve both appearance and offline metrics across Custom and
  Redwood samples.

## Structure-aware partial ARAP pig feasibility probe — 2026-09-08

- [x] Diagnose why Camera-1-only Gaussian mean fields cannot close the
  `domestic_pig` leg gap.  The saved-view residual selector reaches only an
  ear and one silhouette leg, while a normal-compatible 3-D partial/prior
  audit exposes four spatially separated, coherent residual components on
  the legs.  The audit uses no category, GT, CD, or EMD:
  `workspace/custom10_fixed_mainline_20260907/pig_structure_correspondence_diagnostic_20260908a/`.
- [x] Add an isolated fixed-slot 3DGS mean deformation probe at
  `src/structure_aware_arap_gaussian.py`.  It estimates local normals from
  the registered partial and complete carrier, accepts a partial-to-prior
  anchor only under normal and distance compatibility, keeps low-residual
  support fixed, and transfers each coherent residual component through a
  geodesically bounded ARAP graph.  The 100k carrier count is preserved and
  the edit never concatenates a partial point cloud.
- [~] Pig candidate completed at
  `workspace/custom10_fixed_mainline_20260907/pig_structure_aware_arap_20260908b/`.
  It anchors 174 carrier means across four structure-compatible components,
  moves 27,527 means inside a 28.8% geodesic influence band, and hits those
  anchors exactly.  Its no-GT structural residual audit changes from 1,124
  high-residual compatible partial points before the edit to 121 after; the
  compatible distance median/p95 changes from `.02198/.07402` to
  `.01621/.04788`.  ARAP continuity stays bounded at edge stretch
  p01/p99/p999 = `.853/1.376/2.353`.  These are diagnostics only; inspect
  the generated overlays and virtual views with the user before promoting or
  evaluating any other sample.
- [x] Test whether a deficient image-derived bridge can be handled explicitly
  rather than implicitly by the direct partial field.  The isolated
  `moge_unsupported_partial_mask` marks a partial sample only when the
  transformed native MoGe has neither a nearby compatible surface nor a
  locally compatible normal.  On pig this marks `1,819 / 16,384` partial
  samples (11.1%) and exposes one 36-point, fully bridge-unknown residual
  component.  The continuation at
  `workspace/custom10_fixed_mainline_20260907/pig_structure_aware_arap_20260908d_bridge_unknown/`
  adds 9 direct anchors while preserving the 3,102 first-stage locked slots;
  its ARAP edge-stretch p01/p99.9 is `.884/1.888`.  However, a no-GT direct
  surface audit leaves 18 high-residual partial samples—the same count as the
  ordinary fine continuation—and the bridge-unknown p90 distance changes
  slightly adversely (`.04098 -> .04324`).  Retain this as a negative
  feasibility result / diagnostic only; do not promote the relaxed-bridge
  route or use it to select a final result.

## Generic PosteriorAdapter and fresh generalization run — 2026-09-08

- [x] Replace pig-specific diagnostic branches with one category-independent
  `PosteriorAdapter`. Partial OT assigns stable, residual, and unsupported
  support from camera projection, depth, normals, local covariance and 3-D
  distance. One coarse-to-fine deformation graph uses screened ARAP and
  attachment preservation; all thresholds are normalized by spacing,
  residual quantiles, visibility, or partial diagonal. The operator preserves
  every 100k prior slot and never reads GT/CD/EMD.
- [x] Freeze that implementation after the accepted pig pressure test and run
  the identical code/configuration over the remaining Custom and Redwood
  samples. The fixed-input generalization audit is retained at
  `workspace/posterior_adapter_generalization_20260908g/{custom,redwood}`.
  It improves the Custom aggregate from `2.5931/3.2896` to `2.2496/3.1749`
  and preserves Redwood at `1.4693/2.2769` versus `1.4693/2.2760`.
- [x] Regenerate semantic observations, Pixal priors, registration and
  posterior outputs from scratch for Custom10 and Redwood10 under
  `workspace/posterior_adapter_e2e_redwood_custom_20260908/`. The first fresh
  Redwood pass exposed two upstream observation failures (`07136`, `07306`),
  not posterior failures.
- [x] Generalize the bounded view action from base/antipode/orthogonal views
  to include +/-30 and +/-60 degree yaw candidates. Replan only from
  partial-derived diagnostic boards. `07136` and `07306` select the antipodal
  camera before regeneration; no sample-specific geometry rule or numerical
  parameter is introduced. Their fresh offline results recover from
  `7.4015/10.5119` and `20.2171/24.5511` to `1.1191/2.0224` and
  `3.4709/3.8825`, respectively.
- [x] Consolidate the selected results and preserve every intermediate/rejected
  attempt. Redwood final predictions and the no-GT selection audit are under
  `redwood/final_selected/`; Custom final predictions are under
  `custom/final_selected/`. The final offline-only aggregates are Redwood10
  `1.5017/2.3739` and Custom10 `2.0118/2.9994` at 16,384 points, seed 6145.
- [ ] Pending user visual approval: do not promote this from-scratch run over
  a previously accepted mainline result solely from the offline metrics. The
  run establishes category-independent recovery and full reproducibility; a
  later promotion must preserve the no-GT candidate-selection firewall.
- [x] Consolidate the selected Redwood10 and Custom10 artifacts into the
  sample-centric hard-linked collection
  `workspace/posterior_adapter_selected_redwood_custom_20260908/`. Redwood
  `07136` and `07306` point to their selected antipodal-view reruns; the other
  eight Redwood samples and all ten Custom samples point to their selected
  fresh runs. All 20 `final_100k.ply` files were header-verified at exactly
  100,000 vertices. The source experiment remains untouched.

## TRELLIS regenerated-prior registration probe — 2026-09-08

- [x] Keep the probe isolated from the object mainline.  Render the registered
  Pixal asset from Camera-1/side/back, edit only the hidden-view structural
  evidence, and generate a complete mesh plus Gaussian carrier with
  TRELLIS-image-large on a 24 GB GPU.
- [~] Replace complete-to-complete PCA capture with a Camera-1-first bridge.
  The current implementation uses a bounded yaw/pitch/roll view set, spatial
  DINO, low-frequency RGB layout and silhouette evidence to propose canonical
  TRELLIS views; partial silhouette, depth and visible 3-D evidence then fit
  and select proper Sim(3) basins.  This remains a feasibility implementation,
  not a promoted mainline.
- [x] Build an explicitly offline GT-only Sim(3) oracle to measure the
  registration/shape upper bound.  The pig oracle output is under
  `workspace/trellis_pig_multiview_probe_20260908/registration/gt_oracle/`.
  `gt_gray_trellis_red.ply` contains GT in gray and oracle-aligned TRELLIS in
  red; `partial_gray_trellis_red.ply` provides the corresponding observed-view
  comparison.  The corrected offline values are CD-L1 x100 `1.8519` and EMD
  x100 `2.6904`.  This oracle is marked `ground_truth_used=true` and is never
  eligible for inference or candidate selection.
- [ ] Use the oracle transform only to decompose the remaining no-GT error
  into rotation, isotropic scale and translation.  Do not remove 3DGS or
  PosteriorAdapter from the mainline until a frozen, category-independent
  TRELLIS registration reaches a competitive no-GT result on Pig and then
  passes Custom/Redwood generalization.
- [x] Replace the sparse top-component edit ledger with dense, continuous,
  per-view residual evidence.  Each FRONT/SIDE/BACK card retains prior RGB,
  prior depth, partial depth, and all partial-supported residual samples;
  connected components are explanatory annotations rather than a gate that
  drops moderate evidence.  The retained evidence covers `7,373 / 16,384`
  partial points instead of the previous `3,486 / 16,384`.
- [~] Validate three independent, auditable `REFINE_VIEW(view_id)` actions.
  FRONT, SIDE, and BACK are each edited from their own residual/depth card and
  the same three-view identity reference, then deterministically normalized
  and composed.  The current unaccepted visual candidate and exact prompts
  are under
  `workspace/trellis_pig_multiview_probe_20260908/conditions/cross_view_dense_v5/per_view_refine/`;
  inspect cross-view limb/layout consistency before invoking TRELLIS.
- [~] Replace that independent raw-residual edit with a strict three-view,
  two-stage decomposition.  A single smooth 3-D graph field first explains
  the shared low-frequency deformation for FRONT/SIDE/BACK; each view then
  receives only the residual remaining after subtracting that shared field.
  The decomposition is category-independent and uses only partial-to-prior
  observations, spacing-normalized graph scales, and stable-region anchors.
  On Pig, residual q95 decreases from `.17239` to `.04532`, while shared
  motion over stable support remains `.00534` at q90.  The generated but not
  yet user-approved three-view candidate, exact prompts, graph carrier and
  diagnostic ledger are retained under
  `workspace/trellis_pig_multiview_probe_20260908/conditions/cross_view_shared_local_v7/`.
  Do not invoke TRELLIS or promote the probe until cross-view consistency is
  visually accepted.
  - The v7 three-view conditions were then used directly by
    `TRELLIS-image-large` (seed 42, stochastic multi-image mode) and retained
    at `trellis/shared_low_frequency_local_v7_seed42/`.  The no-GT alignment
    performs a three-view canonical shortlist, low-resolution joint capture,
    one Camera-1 partial-to-prior inverse update, and a final object-frozen
    auxiliary-camera polish.  The Pareto-improving final candidate is at
    `registration/trellis_shared_local_v7_camera_polish/`, with 512px hard-mask
    IoUs `.8485/.7327/.6135` for FRONT/SIDE/BACK.  Only after this selection,
    offline evaluation reports CD-L1/EMD x100 `3.6223/4.7198`.  This historical
    intermediate was not competitive and was not promoted; the later accepted
    all-support Camera-1 result documented at the top supersedes it.

## Minimal four-view partial-evidence probe — 2026-09-09

- [~] In the isolated TRELLIS probe only, replace the three-view residual-edit
  contract with one fixed-orbit FRONT/SIDE/BACK/RIGHT observation.  Qwen receives
  exactly two aligned 2x2 boards: the registered prior RGB views and a
  visibility-aware partial projection.  Valid partial pixels are positive
  geometric evidence; missing pixels are explicitly unknown and never deletion
  evidence.
- [ ] Run the category-independent probe on Redwood 06830, save the four edited
  views and their camera manifest, and verify cross-view structure before
  invoking TRELLIS.  Do not add part detectors, sample rules, extra optimizers,
  GT selection, or modify the frozen object mainline.

## Geometry-first shared-view proxy probe — 2026-09-10

- [x] Keep the accepted registered single-view Pixal asset as one textured,
  connected identity carrier.  Project its ordered 100k carrier and the
  partial through the four saved informative cameras, then solve one
  category-free screened low-frequency 3-D displacement field.  Stable
  visible support is softly anchored; all remaining mesh motion is propagated
  through the welded triangle graph rather than independent image edits or
  spatial KNN across thin sheets.
- [x] Run the isolated 06830 probe at
  `workspace/geometry_first_multiview_proxy_06830_20260910/`.  One conservative
  accepted update improves the no-GT four-view visible median/P90 residual
  from `.03172/.18843` to `.03063/.18088` and mean partial coverage from
  `.97779` to `.97843`.  The mesh remains one connected indexed surface with
  zero resolved triangle flips; median edge scale is `.99741` and
  edge-stretch P01/P99 are `.93094/1.04981`.
  Offline-only CD-L1/EMD x100 improve from `2.4825/4.4202` for the input
  registered carrier to `2.3765/4.1987` for the strictly topology-preserving
  deformed carrier.  A less conservative step reached `2.2954/4.0318` but
  flipped `0.0046%` of resolved triangles and is retained only as a rejected
  diagnostic, not the exported result.
- [x] Re-render that single deformed mesh with the exact saved camera poses.
  The four views are therefore cross-view consistent by construction and need
  no GPT geometry repaint.  Their board and residual audit are under
  `render/` and `evidence/` in the same experiment directory.
- [x] Generate a TRELLIS comparison from those clean views.  TRELLIS produces
  a plausible complete tricycle, but its subsequent condition registration is
  substantially worse (`6.0127/7.9827` offline-only) and has only
  `.516-.622` condition-mask IoU.  This shows that regenerating an already
  connected deformed proxy reintroduces canonical-frame and shape error.  Do
  not promote the TRELLIS output; the geometry-first Pixal mesh is the useful
  result of this probe.
- [x] Replace TRELLIS with the updated Pixal3D-MV implementation for a direct
  controlled comparison, using the same exact-camera renders and no image
  repaint.  The camera-lock audit passes for all four views; 1024-cascade MV
  inference completes in `58.4 s` at `16.41 GiB` peak VRAM and exports a valid
  textured mesh (`180,173` vertices, `280,692` faces).  Nevertheless, its
  native-MoGe/two-camera placement reaches only offline CD-L1/EMD x100
  `5.4749/6.4770` analytically and `4.8415/5.9042` after the standard no-GT
  Camera-1 2D+3D micro-refinement.  This is worse than the directly deformed
  connected Pixal carrier (`2.3765/4.1987`): MV regeneration changes hidden
  geometry and its main-view mask contract has low IoU (`.3111`) even though
  the Camera-1 overlay looks plausible.  Retain the comparison under
  `workspace/geometry_first_multiview_proxy_06830_20260910/pixal3d_mv*`, but do
  not promote Pixal3D-MV regeneration over the geometry-first mesh result.
- [x] Strengthen the same category-independent proxy deformation without
  relaxing it into unconstrained point motion.  The mesh surface operator is
  built once in the initial topology, six incremental updates recompute
  partial correspondences, and a cumulative log-edge-scale budget permits
  local contraction/expansion while each accepted step remains
  orientation-preserving.  The isolated result is under
  `workspace/geometry_first_multiview_proxy_06830_20260910/strong_local_scale_v3/`.
  No-GT visible median/P90 residual improves from `.03172/.18843` to
  `.02733/.15993`, with coverage `.97779 -> .98132`; cumulative edge-scale
  P01/P50/P99 is `.76960/.99078/1.20685`.  Offline-only CD-L1/EMD x100 is
  `2.1131/3.6103`, versus `2.3765/4.2055` for the conservative deformation and
  `2.4825/4.4184` before deformation.  Keep this as an isolated 06830 probe
  pending user visual approval; it does not modify the frozen mainline.
- [x] Evaluate the corrected six-view ordering on 06830: Camera-1 plus five
  partial-visible, yaw-separated cameras are rendered first; a 3x2 residual
  board then drives one cross-view image edit, whose silhouettes provide only
  image-plane controls while physical partial depth/3-D correspondences remain
  the acceptance signal.  Six views raise unique partial visibility from
  `14,581` to `16,962` points, but do not improve the completion: offline-only
  CD-L1/EMD x100 is `2.1654/3.8168` without the edit and `2.2069/3.8413` with
  the edit, versus `2.1131/3.6163` for the four-view strong deformation.
  Retain the auditable six-view artifacts under `six_view_input/`,
  `six_view_no_edit_deformation/`, and
  `six_view_residual_edited_deformation/` as an ablation; do not replace the
  four-view probe or frozen mainline.  The likely failure mode is conflicting
  incomplete evidence from the two lower-information auxiliary views, not
  insufficient deformation capacity.
- [x] Run the same four-view residual-edit probe on the held-out Custom Pig
  stress case without any category/part rule.  Artifacts are under
  `workspace/geometry_first_multiview_proxy_pig_20260910/`.  Four informative
  views cover `13,450 / 16,384` partial points.  The residual-guided update
  improves its no-GT visible median/P90 residual from `.04545/.35895` to
  `.03896/.31656`, retains `99.9996%` resolved triangles, and passes the final
  topology gate.  However, offline-only CD-L1/EMD x100 regresses from
  `4.7063/3.3237` for the registered input to `5.1995/3.6937`; the no-edit
  deformation is `5.2244/3.6347`.  Therefore this is a diagnostic probe, not a
  promoted mainline result.  It exposes a verifier-calibration issue: visible
  partial agreement can improve while complete-shape error worsens.
- [x] Diagnose the insufficient Pig amplitude.  The original local solver's
  5-pixel capture radius excluded the distant observed structure; merely
  widening it increases the main-axis extent by only about 3% and remains a
  rejected diagnostic (`four_view_broad_to_narrow_v2/`).  Use the existing
  category-free structural OT plus embedded-ARAP posterior as the shared
  low-frequency layer instead: it increases the corresponding bounding-box
  extent from `.7122` to `.9451` (about 33%), retains all 100k carrier slots,
  creates no new connected component, and passes its no-GT visible verifier.
  The rerun is under `four_view_posterior_low_frequency_v3/`, with convenient
  output `final_strong_continuous_posterior_100k.ply`; offline-only CD-L1/EMD
  x100 is `1.4752/1.8478`.  The stronger four-view edit condition is saved as
  `residual_edited_4view_v2_strong.png`; it is not used to select the offline
  result.
