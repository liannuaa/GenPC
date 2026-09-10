# GenPC++: zero-shot point-cloud completion with regenerated 3D priors

GenPC++ completes an object from a partial point cloud without task-specific
training or test-time ground truth. The current object mainline first obtains
a complete image-conditioned prior, exposes its discrepancy with the observed
scan in three consistent views, regenerates a corrected complete 3D asset, and
registers that asset back to the physical partial observation.

```text
partial point cloud
  -> saved-view depth and semantic completion
  -> clarity-preserving image edit
  -> initial textured Pixal prior
  -> Pixal--MoGe--partial camera-aware Sim(3)
  -> front/side/back residual evidence
  -> shared low-frequency edit, then residual-only local edits
  -> TRELLIS multi-image complete prior
  -> semantic basin capture + Camera-1 visible 2D+3D Sim(3)
  -> complete 100k-point prediction
```

The released route currently ends at registration. It does not fuse partial
points, delete prior points, or run Gaussian/non-rigid post-processing. Ground
truth, CD, and EMD are available only to the separate offline evaluator.

## Documentation

- [Installation](docs/installation.md)
- [External model assets](docs/models.md)
- [Object mainline and fixed geometry](docs/core_registration_pipeline.md)
- [Scene-level textured reconstruction](docs/scene_completion.md)
- [Documentation index](docs/README.md)

## Repository layout

```text
configs/mainline_redwood.yaml          saved-view and semantic configuration
data/redwood/{partial,gt}/             inference scans / offline-only GT
scripts/run_object_mainline.py         resumable object-level entry point
scripts/run_semantic_stage.py          partial -> depth and semantic image
scripts/run_pixal3d_gpt_batch.py       image -> initial textured 3D prior
scripts/run_fixed_pixal_moge_registration.py
                                       camera-aware initial-prior registration
scripts/build_shared_local_residual_cards.py
                                       three-view geometric edit evidence
scripts/run_trellis_multiview_regeneration.py
                                       edited views -> complete TRELLIS asset
scripts/run_nvdiffrast_multiview_registration.py
                                       rendered semantic basin capture
scripts/run_partial_supported_sim3.py  final Camera-1 visible 2D+3D Sim(3)
scripts/evaluate_mainline_redwood.py    offline CD-L1/EMD only
scripts/run_scene_completion.py        independent scene-level wrapper
```

## Object mainline

The runner accepts any dataset path and an object-name prompt; its geometry is
category independent. For example:

```bash
PY=/opt/data/private/cr/miniconda3/envs/genpc/bin/python
RUN=workspace/example_01184

CUDA_VISIBLE_DEVICES=0 $PY scripts/run_object_mainline.py \
  --sample 01184 \
  --object-type "blue wheeled trash bin" \
  --partial data/redwood/partial/01184.ply \
  --run-root "$RUN" \
  --stage all
```

`--stage all` resumes completed artifacts and pauses at either external image
checkpoint. Calling the same command again continues from the newly supplied
images. `--stage status` reports every checkpoint without loading a model.

### External image checkpoint 1: clarity

After the semantic stage, edit
`$RUN/inputs/camera/01184/img.png` while preserving its camera, silhouette,
pose, scale, crop, and articulated state. Save the result and exact prompt as:

```text
$RUN/inputs/pixal/01184/gpt_image.png
$RUN/inputs/pixal/01184/prompt.txt
```

The next run generates and registers the initial Pixal prior, renders its
front/side/back views, and writes the residual evidence and prompts under:

```text
$RUN/multiview/01184/render/
$RUN/multiview/01184/evidence/
```

### External image checkpoint 2: prior regeneration

The second checkpoint is a two-stage edit so that global extent is not applied
three times independently:

1. Use the original three-view strip, residual board, and
   `stage1_shared_low_frequency_prompt.txt` to create one shared edit at
   `$RUN/multiview/01184/edits/shared_low_frequency_front_side_back.png`.
2. For each view, use its split stage-1 image, residual card, full stage-1
   strip, and per-view prompt. Save raw outputs as
   `$RUN/multiview/01184/edits/local_raw/{front,side,back}.png`.

The runner restores foreground centre and isotropic framing, generates the
TRELLIS asset, and performs the fixed registration chain. The prediction and
complete hash/provenance manifest are:

```text
$RUN/final/01184/complete_100k.ply
$RUN/manifests/01184.json
```

Individual stages can be resumed with `upstream`, `prepare-edits`,
`materialize-edits`, `trellis`, and `register`. No stage accepts a ground-truth
path.

## Offline evaluation

Only after predictions are frozen:

```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/evaluate_mainline_redwood.py \
  --prediction-root "$RUN" \
  --ground-truth-root data/redwood/gt \
  --samples 01184 \
  --output "$RUN/metrics/redwood"
```

The evaluator is not imported or called by the mainline runner.

## Scene-level route

The scene wrapper remains independent. It segments one scene observation,
generates and camera-places one textured instance mesh at a time in the shared
scene-MoGe frame, and resolves collisions with minimum depth-axis motion. It
does not invoke the object-level TRELLIS regeneration or modify its outputs.
See [scene completion](docs/scene_completion.md).

## Reproducibility boundary

- Qwen, Pixal, TRELLIS, and every geometric solver use the shared settings
  documented in the method specification.
- The official TRELLIS multi-image interface does not receive camera
  extrinsics. The three images constrain regenerated shape; physical pose and
  scale are recovered afterwards from the saved partial camera and 3D scan.
- External GPT images and their exact prompts are first-class artifacts. A run
  is not exactly reproducible if they are missing.
- `PROJECT_STATE.md` records accepted internal experiments but is never read by
  inference.

## Licence

GenPC++ code is released under the [MIT License](LICENSE). External models,
checkpoints, CUDA extensions, and datasets retain their own licences.
