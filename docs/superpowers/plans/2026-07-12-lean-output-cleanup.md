# Lean Output Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce default intermediate artifacts while preserving files needed to inspect and continue the GenPC refactor pipeline.

**Architecture:** Add a named output keep profile in `utils/runtime.py` so cleanup is centralized. Keep debug-heavy files only when `outputs.save_intermediates: true` or `outputs.keep_profile: debug/full`; default to `lean` in config.

**Tech Stack:** Python, pathlib cleanup, existing GenPC YAML config, unittest/py_compile verification.

## Global Constraints

- Use `/opt/data/private/cr/miniconda3/envs/genpc/bin/python` for project checks.
- Do not delete accepted baseline outputs outside the current requested experiment directory.
- Keep final PLYs and transform/index metadata needed for downstream stages.

---

### Task 1: Centralize Lean Keep Profiles

**Files:**
- Modify: `utils/runtime.py`
- Test: add `tests/test_output_cleanup_profiles.py`

**Interfaces:**
- Produces: `cleanup_intermediates(cfg, flag)` honoring `outputs.keep_profile`.
- Produces: `cleanup_stage1_intermediates(cfg, flag)` honoring the same profile.

- [ ] Add tests for lean cleanup preserving expected files and deleting debug files.
- [ ] Implement `output_keep_profile()` and profile whitelist helpers.
- [ ] Run the new tests.

### Task 2: Set Default Config To Lean

**Files:**
- Modify: `configs/config.yaml`
- Modify: `AGENTS.md`

**Interfaces:**
- Produces: default `outputs.keep_profile: lean` with `save_intermediates: false` for normal runs.

- [ ] Update config defaults.
- [ ] Update workspace notes to describe `lean`, `debug`, and `full` behavior.

### Task 3: Clean Current 01184 Experiment Outputs

**Files:**
- Modify generated files under `workspace/redwood_stage1_qwen_refine_preview/01184` only.

**Interfaces:**
- Keeps useful Stage 1, Hunyuan, MoGe bridge, masked FreeReg, and final inspection files.

- [ ] Remove unmasked FreeReg outputs and debug MoGe/Qwen files from current 01184 directory.
- [ ] Verify remaining files include the useful PLYs and metadata.

### Task 4: Verification And Plan Update

**Files:**
- Modify: `PLAN.md`

**Interfaces:**
- Produces: documented cleanup policy and current 01184 retained file list.

- [ ] Run `python -m unittest tests.test_output_cleanup_profiles -v`.
- [ ] Run `python -m py_compile utils/runtime.py scripts/run_freereg_original_depthpro.py tools/hunyuan3d_2.py`.
- [ ] Update `PLAN.md` with the lean cleanup policy.
