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
