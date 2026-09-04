# GenPC+: zero-shot complete point clouds from a partial scan

This repository contains one compact, fixed mainline.  It retains GenPC's
saved-view depth construction, replaces its 3-D prior with Pixal3D, and uses
the partial only for test-time registration and a bounded Gaussian edit.  It
does not train, use GT during inference, or route samples through alternative
methods.

```text
partial scan
  -> deterministic saved-view grayscale depth + Qwen semantic image
  -> GPT clarity-only image edit (external; prompt recorded with the asset)
  -> Pixal3D GLB + 100k sampled complete prior
  -> native Pixal--MoGe alignment -> two-camera bridge -> small partial refinement
  -> fixed complete-to-partial Sim(3)
  -> partial-anchored multiview Gaussian edit -> 100k complete point cloud
```

The canonical method and frozen parameters are in
[`docs/core_registration_pipeline.md`](docs/core_registration_pipeline.md).
The accepted Redwood-10 output record is in `PROJECT_STATE.md`.

## Environment and model paths

Run commands with:

```bash
PY=/opt/data/private/cr/miniconda3/envs/genpc/bin/python
```

The following local assets are required:

- `models/Qwen-Image-Edit-2511`
- `models/nunchaku-qwen-image-edit/nunchaku_qwen_image_2511_balance_int4.safetensors`
- `models/Pixal3D-weights`
- `models/dinov3-vitl16-pretrain-lvd1689m`
- `models/moge-2-vitl/model.pt`
- `models/RMBG-2.0`

`PIXAL3D_SOURCE` should point to the local Pixal3D source tree if it is not
under `models/Pixal3D`.

## Run the mainline

1. Generate Qwen semantic images and saved-camera files from partial scans:

```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/run_semantic_stage.py \
  --output-root workspace/new_run/semantic \
  --partial-root data --samples 01184
```

2. Apply the GPT clarity-only edit outside this repository. Save each result
as `gpt_image.png` under `workspace/new_run/pixal/<sample>/` and save the exact
prompt beside it. The Qwen image is the geometry authority: preserve camera,
pose, silhouette and local part layout. The fixed downstream Sim(3) does not
assume the edit retained an absolute image scale.

3. Create Pixal3D priors:

```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/run_pixal3d_gpt_batch.py \
  --input-root workspace/new_run/pixal \
  --output-root workspace/new_run/pixal --ids 01184
```

4. Put `camera.pth`, `point_uv.npy`, `img.png`, and
`<sample>_moge_to_raw_partial_object_mask.png` from Stage 1 under
`workspace/new_run/inputs/camera/<sample>/`; put partial scans under
`workspace/new_run/inputs/partial/`. Then run the fixed registration:

```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/run_fixed_pixal_moge_registration.py \
  --samples 01184 \
  --pixal-root workspace/new_run/pixal \
  --camera-root workspace/new_run/inputs/camera \
  --partial-root workspace/new_run/inputs/partial \
  --output-root workspace/new_run/registration
```

5. Use the same `workspace/new_run` root for the fixed Gaussian edit and
decode.  Its input contract is `inputs/{camera,pixal,partial}`; use
`scripts/materialize_mainline_inputs.py` to reproduce the accepted Redwood
layout, or arrange an equivalent layout yourself.

```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mainline_gaussian.py \
  --root workspace/new_run --registration-root workspace/new_run/registration \
  --samples 01184 \
  --max-anchor-residual-ratio .075 --max-displacement-ratio .075 \
  --graph-screening .0015 --prior-protection-weight .01 \
  --protection-exclusion-ratio .10
```

The final cloud is
`gaussian/<sample>/decoded/partial_anchored_gaussian_decoded_100k.ply`.

## Offline evaluation

Metrics are explicitly separate from inference:

```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/evaluate_mainline_redwood.py \
  --prediction-root workspace/new_run/gaussian \
  --output workspace/new_run/metrics
```
