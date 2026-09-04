# GenPC+: zero-shot complete point clouds from partial scans

GenPC+ recovers a complete 100k-point object from a single partial point
cloud, without training or test-time ground truth. It keeps GenPC's
deterministic saved-view depth construction, uses a Pixal3D image-conditioned
prior, aligns it through a fixed two-camera 2D+3D Sim(3) procedure, and uses
the partial only as anchored evidence for a smooth Gaussian edit.

```text
partial point cloud
  → saved-view grayscale depth + Qwen semantic image
  → external GPT clarity edit
  → Pixal3D textured GLB + complete 100k-point prior
  → native Pixal–MoGe alignment + two-camera bridge + Camera-1 Sim(3)
  → partial-anchored multiview Gaussian edit/decode
  → complete 100k-point prediction
```

The method is one fixed route: it has no training stage, GT/CD/EMD-guided
selection, category-specific parameter sets, or per-sample fallback branch.

## Documentation

- [Installation](docs/installation.md)
- [External model assets](docs/models.md)
- [Fixed method and frozen parameters](docs/core_registration_pipeline.md)
- [Documentation index](docs/README.md)

## Repository layout

```text
configs/mainline_redwood.yaml       fixed saved-view and Qwen configuration
data/redwood/partial/               inference partial scans
data/redwood/gt/                    offline-only ground truth
scripts/run_semantic_stage.py       partial → depth/Qwen/camera assets
scripts/run_pixal3d_gpt_batch.py    GPT image → Pixal GLB/100k prior
scripts/run_fixed_pixal_moge_registration.py
                                    fixed Pixal–MoGe–partial registration
scripts/run_mainline_gaussian.py    partial-anchored edit and 100k decode
scripts/evaluate_mainline_redwood.py offline CD-L1/EMD only
```

## Data contract

For a Redwood sample ID `<id>`, inference reads:

```text
data/redwood/partial/<id>.ply
```

Ground truth has the parallel path below, but is read exclusively by the
offline evaluator after a prediction is frozen:

```text
data/redwood/gt/<id>.ply
```

The released fixed batch contains:

```text
01184  05117  05452  06127  06145
06188  06830  07136  07306  09639
```

Other datasets may be run by arranging one partial PLY plus the same workspace
artifact contract. Copy `configs/mainline_redwood.yaml`, add a semantic
`prompt_overrides` label for each new sample ID, and pass that config to the
semantic stage. The registration and Gaussian parameters remain shared; the
pipeline never reads a category label during those stages.

## End-to-end run

Set the interpreter after following [Installation](docs/installation.md):

```bash
PY=python
RUN=workspace/example_01184
ID=01184
```

### 1. Saved-view depth and Qwen semantic image

```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/run_semantic_stage.py \
  --output-root "$RUN/inputs/camera" \
  --partial-root data/redwood/partial \
  --samples "$ID"
```

This saves `depth.png`, `raw_depth.png`, `img.png`, `camera.pth`,
`point_uv.npy`, and the Camera-1 foreground mask under
`$RUN/inputs/camera/$ID/`.

### 2. External GPT clarity edit

Create the Pixal input directory and save the GPT result there:

```bash
mkdir -p "$RUN/inputs/pixal/$ID"
# Save the geometry-preserving GPT edit as:
#   $RUN/inputs/pixal/$ID/gpt_image.png
# Save its exact prompt as:
#   $RUN/inputs/pixal/$ID/prompt.txt
```

`img.png` is the geometry authority. GPT may clarify material/detail but must
not rotate, mirror, recrop, rescale, recenter, add/remove parts, or change an
articulated local pose. See [Model assets](docs/models.md#qwen-and-gpt-image-inputs).

### 3. Pixal3D complete prior

```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/run_pixal3d_gpt_batch.py \
  --input-root "$RUN/inputs/pixal" \
  --output-root "$RUN/inputs/pixal" \
  --ids "$ID"
```

Outputs include `pixal3d.glb`, `pixal3d_sampled_100k.ply`, camera metadata,
the preprocessed input, and a hash-checked native FP16 MoGe observation.

### 4. Fixed two-camera registration

The Gaussian stage keeps a self-contained copy of the partial scan:

```bash
mkdir -p "$RUN/inputs/partial"
cp "data/redwood/partial/$ID.ply" "$RUN/inputs/partial/$ID.ply"

CUDA_VISIBLE_DEVICES=0 $PY scripts/run_fixed_pixal_moge_registration.py \
  --samples "$ID" \
  --pixal-root "$RUN/inputs/pixal" \
  --camera-root "$RUN/inputs/camera" \
  --partial-root "$RUN/inputs/partial" \
  --output-root "$RUN/registration"
```

The registered complete prior is:

```text
$RUN/registration/<id>/final/camera1_amplified_registered_100k.ply
```

### 5. Partial-anchored Gaussian edit and decode

```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/run_mainline_gaussian.py \
  --root "$RUN" \
  --registration-root "$RUN/registration" \
  --samples "$ID"
```

The final prediction is:

```text
$RUN/gaussian/<id>/decoded/partial_anchored_gaussian_decoded_100k.ply
```

It always contains 100,000 slots. The unobserved Pixal body is retained rather
than discarded or naively concatenated with the partial scan.

## Offline evaluation

Compile the optional CUDA metric extensions first, as described in
[Installation](docs/installation.md#optional-offline-cdemd-extensions). Then
evaluate only after predictions are frozen:

```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/evaluate_mainline_redwood.py \
  --prediction-root "$RUN/gaussian" \
  --ground-truth-root data/redwood/gt \
  --samples "$ID" \
  --output "$RUN/metrics"
```

The evaluator is never called by inference and cannot affect image generation,
Pixal3D, registration, or Gaussian-edit parameters.

## Reproducibility and scope

- The Qwen stage uses the frozen configuration in
  `configs/mainline_redwood.yaml`.
- Pixal3D uses seed 42, the 1024 cascade, 12 sampling steps per stage, and a
  100k surface sampling target.
- Registration uses only proper isotropic Sim(3); its Camera-1 candidate set
  and continuation schedule are fixed for every sample.
- Gaussian edit uses the shared parameters documented in
  [the method specification](docs/core_registration_pipeline.md).
- `PROJECT_STATE.md` records accepted internal experiment assets. It is not an
  inference branch and should not be used for test-time selection.

## Licence

GenPC+ code is released under the [MIT License](LICENSE). External models,
checkpoints, CUDA extensions, and datasets retain their own licences; see
[Model assets](docs/models.md#licences-and-redistribution).
