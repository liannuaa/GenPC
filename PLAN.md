# Active plan

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
  is `workspace/agentic_prior_adaptation_probe_20260906/01184/final/agent_selected_100k.ply`.
  Its verifier record is stored with the state trace; CD/EMD was not exposed
  to any decision.
- [x] Expand the same isolated controller to the remaining Redwood-10 samples
  (`05117`, `05452`, `06127`, `06145`, `06188`, `06830`, `07136`, `07306`,
  and `09639`). All tool decisions are state-hashed and no GT/CD/EMD was
  exposed before `ACCEPT`.
  - The first `07306` trace exposed a real agent failure: the legacy
    convex-hull saved-view tie-break selected the semantic rear. A separate,
    immutable retry at
    `workspace/agentic_prior_adaptation_probe_20260906/07306_front_view_replan/`
    retained the antipodal `opposite_180` view as an explicit agent action.
    Its no-GT verifier energy fell from `0.18598` (old rear trace) to
    `0.10856` after the identical global/residual Sim(3) executors. Its
    offline-only accepted result is CD-L1x100 `3.1593`, EMDx100 `3.3834`,
    compared with `22.9567/26.8012` for the rejected rear interpretation.
  - The ten accepted traces, with this corrected `07306` substituted and no
    post-acceptance metric feedback, are recorded under
    `workspace/agentic_prior_adaptation_probe_20260906/redwood10_corrected_view/`.
    Their offline audit is CD-L1x100 `1.59054`, EMDx100 `2.35776`.
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
