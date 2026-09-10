# Documentation index

GenPC++ has one fusion-free, zero-shot object mainline plus an independent
scene-level route.

- [Installation](installation.md) explains the tested Python/CUDA setup,
  optional CUDA metric extensions, and test command.
- [Model assets](models.md) lists every external source checkout and checkpoint
  required for a full run. Weights and upstream source trees are not included
  in this repository.
- [Method](core_registration_pipeline.md) defines the observation/Pixal,
  residual-guided TRELLIS regeneration, and visible 2D+3D Sim(3) procedure.
- [Scene-level completion](scene_completion.md) documents the separate GPT-mask
  wrapper that uses scene-MoGe partials and exports registered textured meshes.

The resumable runner is `scripts/run_object_mainline.py`; its per-sample
artifact contract is implemented in `src/object_mainline.py`. The shared path
helpers in `src/mainline_paths.py` define Redwood-10 data and offline prediction
lookup. The stage scripts may still target arbitrary IDs.

The root [README](../README.md) gives the end-to-end artifact contract and
commands. `PROJECT_STATE.md` is an internal record of accepted runs, not an
additional inference route.
