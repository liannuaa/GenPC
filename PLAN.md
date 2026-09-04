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
