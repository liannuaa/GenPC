# Pixal v18 Full-Ten Fusion Record

Status: frozen full-ten candidate; visual approval is pending.

## Protocol

- Input registration: `gpt_version/_pixal_guarded_unified_registration_v15_20260822`.
- Output root: `gpt_version/_pixal_strict_compact_local_edit_v18_20260822`.
- Shared zero-shot parameters for every sample; no sample-specific tuning.
- High-residual same-camera handles, translation-only sparse graph, C1 compact
  support, exact zero displacement beyond 7% of the partial diagonal.
- The visible-correspondence gate is mandatory. A failed correspondence gate,
  excessive active-node ratio, projection regression, inadequate improvement,
  or far-field motion triggers a direct-union fallback.
- All 100k Pixal points are retained and the raw partial is appended exactly.
- GT is loaded only after all predictions are frozen, using deterministic
  16384-point FPS with metric seed 6145.

## Full-Ten Results

| sample | route | CD-L1 x1e2 | EMD x1e2 | GenPC CD / EMD |
| --- | --- | ---: | ---: | ---: |
| 01184 | local edit | 1.188995 | 1.924919 | 2.31 / 3.17 |
| 05117 | safe union fallback | 1.771338 | 3.032858 | 1.36 / 2.20 |
| 05452 | safe union fallback | 0.934114 | 1.467725 | 1.16 / 1.68 |
| 06127 | safe union fallback | 2.315015 | 4.420082 | 2.86 / 4.85 |
| 06145 | safe union fallback | 0.856748 | 1.469414 | 1.28 / 2.07 |
| 06188 | local edit | 1.141484 | 1.944241 | 1.36 / 2.47 |
| 06830 | safe union fallback | 1.964918 | 3.903052 | 1.38 / 2.97 |
| 07136 | local edit | 1.750700 | 2.535489 | 1.58 / 2.78 |
| 07306 | local edit | 2.819940 | 3.409398 | 2.72 / 4.36 |
| 09639 | local edit | 2.013308 | 3.161812 | 1.43 / 2.29 |
| **mean** | **5 edit / 5 fallback** | **1.675656** | **2.726899** | **1.74 / 2.88** |

The full-ten mean beats the GenPC paper mean on both metrics. Five samples beat
their GenPC rows on both metrics. `06830`, `07306`, and `09639` remain the main
risks. The method is not accepted as the final fusion route until the full-ten
visual contact sheet and individual PLYs are approved.

## Artifacts

- Per-sample metrics: `metrics_samples.csv`.
- Fusion diagnostics: `fusion_summary.csv` and `fusion_summary.json`.
- Visual board: `fusion_contact_sheet.png`.
- Per-sample canonical metric link: `<sample>/<sample>_fused.ply`.
- Per-sample full body, fused PLY, compare PLY, deformed mesh, and exact JSON
  diagnostics are stored in each sample directory.
