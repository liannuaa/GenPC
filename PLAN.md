# Active plan

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
