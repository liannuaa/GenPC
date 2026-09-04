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

The root [README](../README.md) gives the end-to-end artifact contract and
commands. `PROJECT_STATE.md` is an internal record of accepted runs, not an
additional inference route.
