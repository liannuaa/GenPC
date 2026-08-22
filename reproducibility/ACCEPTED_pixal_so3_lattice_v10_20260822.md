# Accepted Pixal3D SO(3)-lattice registration baselines (2026-08-22)

The user inspected the v10 repairs for `06830` and `06145` and approved them
with: `okok，效果很好`.  This freezes both v10 registrations as accepted
pose/scale baselines for subsequent completeness-preserving fusion.  Fusion is
not approved yet.

Do not regenerate or overwrite the source `pixal3d.glb`, source
`pixal3d_sampled_100k.ply`, accepted v10 transform, registered PLY, or
registered mesh without asking.

## 06830

- Inputs: `gpt_version/06830/pixal3d.glb`,
  `gpt_version/06830/pixal3d_sampled_100k.ply`, and `data/06830.ply`.
- Accepted root: `gpt_version/_pixal_so3_lattice_sim3_v10_20260822/06830`.
- Accepted transform: `06830_so3_lattice_sim3_ttt_v10.npy`.
- Accepted complete PLY: `06830_so3_lattice_sim3_ttt_v10_registered_100k.ply`.
- Accepted mesh: `06830_so3_lattice_sim3_ttt_v10_registered_mesh.glb`.
- Accepted comparison: `06830_so3_lattice_sim3_ttt_v10_partial_gray_pixal_red.ply`.
- Isotropic scale: `1.1975`.
- Post-freeze CD-L1/EMD x1e2, 16384 FPS points, seed 6145:
  `2.72791 / 4.92008`.

## 06145

- Inputs: `gpt_version/06145/pixal3d.glb`,
  `gpt_version/06145/pixal3d_sampled_100k.ply`, and `data/06145.ply`.
- Accepted root: `gpt_version/_pixal_so3_lattice_sim3_v10_20260822/06145`.
- Accepted transform: `06145_so3_lattice_sim3_ttt_v10.npy`.
- Accepted complete PLY: `06145_so3_lattice_sim3_ttt_v10_registered_100k.ply`.
- Accepted mesh: `06145_so3_lattice_sim3_ttt_v10_registered_mesh.glb`.
- Accepted comparison: `06145_so3_lattice_sim3_ttt_v10_partial_gray_pixal_red.ply`.
- Isotropic scale: `0.73`.
- Post-freeze CD-L1/EMD x1e2, 16384 FPS points, seed 6145:
  `1.47280 / 1.75263`.

## Reproducibility contract

- Implementation: `scripts/run_pixal_so3_lattice_sim3_ttt_v10.py` with
  project-root launcher `run_pixal_so3_lattice_sim3_ttt_v10.py`.
- Frozen semantic-image prompts, Pixal3D model/checkpoints, seed 42, and 100k
  sampling settings are inherited verbatim from the accepted ten-sample Pixal
  baseline in `PROJECT_STATE.md`.  Registration itself has no text or negative
  prompt (`not applicable`).
- Fixed category-agnostic 648-candidate initialization: 24 proper PCA
  rotations composed with shared Euler offsets `{-45, 0, 45}^3`.
- Shared coarse scales: `0.6, 0.8, 1.0, 1.2, 1.4`; coarse render 64 with
  2500/1500 complete/partial points; fine render 128 with 10000/6000 points;
  24 fine candidates; eight local rotation candidates; camera padding 0.15.
- Selection uses only saved-camera partial silhouette, visible depth, and
  partial-to-complete surface consistency.  GT is not loaded for inference or
  selection; reported GT metrics are post-freeze diagnostics only.
- Proper rotation, isotropic scale, and translation only.  No fusion, point
  deletion, non-rigid deformation, anisotropic scale, or source regeneration.
  All 100000 frozen Pixal points are retained.

Detailed diagnostics are in:

- `reproducibility/06830_so3_lattice_sim3_v10_20260822.md`
- `reproducibility/06145_so3_lattice_sim3_v10_20260822.md`

`PROJECT_STATE.md` should link or absorb this record when the environment's
existing-file patch mount is available again.
