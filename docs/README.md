# Documentation index

GenPC+ has one fixed, zero-shot mainline. It does not contain historical
registration routes, per-sample selectors, or training code.

- [Installation](installation.md) explains the tested Python/CUDA setup,
  optional CUDA metric extensions, and test command.
- [Model assets](models.md) lists every external source checkout and checkpoint
  required for a full run. Weights and upstream source trees are not included
  in this repository.
- [Method](core_registration_pipeline.md) defines the fixed Qwen/GPT/Pixal,
  two-camera Sim(3), and partial-anchored Gaussian procedure.
- [Scene-level completion](scene_completion.md) documents the separate GPT-mask
  wrapper that uses scene-MoGe partials and exports registered textured meshes.

The shared runner contract lives in `src/mainline_paths.py`: it defines the
Redwood-10 default IDs, the canonical `data/redwood/{partial,gt}` locations,
and the final registration/decode artifact names. The stage scripts retain
their documented command-line interfaces and may still target arbitrary IDs.

The root [README](../README.md) gives the end-to-end artifact contract and
commands. `PROJECT_STATE.md` is an internal record of accepted runs, not an
additional inference route.
