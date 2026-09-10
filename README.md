# GenPC++

GenPC++ is a zero-shot point-cloud completion system built around complete 3D
generative priors. It keeps generation, geometric inference, and offline
evaluation separate: neither ground truth nor CD/EMD is available to the
inference path.

The object pipeline is:

```text
partial point cloud
  -> saved-camera depth
  -> mask-locked direct-GPT semantic completion
  -> textured Pixal3D prior and native MoGe observation
  -> camera-aware Pixal--MoGe--partial Sim(3) registration
  -> four informative prior/partial residual views
  -> structure-aware Partial OT
  -> coarse-to-fine prior-preserving posterior deformation
  -> four-view observation-anchored carrier fusion
  -> complete 100k-point prediction
```

The depth raster is the camera and foreground-layout authority for semantic
completion. Direct GPT is instructed to preserve its mask, center, apparent
size, crop, orientation, and visible part locations; an automatic mask audit
rejects images that drift before any 3D generation is run.
The shared audit allows small boundary uncertainty on thin structures but has
no category- or sample-specific bypass.

Prepare the external GPT task and install the returned image with:

```bash
PY=/opt/data/private/cr/miniconda3/envs/genpc/bin/python

$PY scripts/run_semantic_stage.py \
  --output-root workspace/run/inputs/camera \
  --samples 01184 --depth-only

$PY scripts/prepare_depth_gpt_semantic_task.py \
  --depth workspace/run/inputs/camera/01184/depth.png \
  --object "a blue wheeled rubbish bin" \
  --constraint "Keep exactly two wheels parallel and attached to the same axle." \
  --output workspace/run/inputs/pixal/01184/gpt_depth_completion_prompt.txt

# Run GPT image generation with the saved depth and exact prompt, then:
$PY scripts/install_depth_conditioned_semantic.py \
  --sample 01184 \
  --semantic workspace/run/inputs/pixal/01184/gpt_image.png \
  --depth workspace/run/inputs/camera/01184/depth.png \
  --camera-dir workspace/run/inputs/camera/01184 \
  --rmbg-model models/RMBG-2.0
```

The four-view observation exposes where the registered prior disagrees with
positive partial evidence and contributes visibility-valid pixel/depth costs
directly to Partial OT. One category-independent `PosteriorAdapter` performs
the physical update. It preserves every prior
carrier slot, anchors already aligned surface, propagates reliable residuals
through a deformation graph, and protects unobserved structure. Missing
partial pixels are unknown rather than deletion evidence.

The final fusion is not point concatenation and does not run a second shape
deformation. The same four saved cameras establish visibility-valid 2-D/depth
correspondences; a collision-free one-to-one decoder writes reliable measured
partial samples into an equal number of posterior carrier slots. All unmatched
slots remain unchanged, so observed geometry becomes exact while complete and
unobserved prior support is retained at a fixed 100k-point density.

## Quick start from a registered prior

Use the project interpreter explicitly:

```bash
PY=/opt/data/private/cr/miniconda3/envs/genpc/bin/python

CUDA_VISIBLE_DEVICES=0 $PY scripts/run_posterior_adapter.py \
  --prior workspace/run/registration/01184/final/camera1_amplified_registered_100k.ply \
  --partial data/redwood/partial/01184.ply \
  --camera workspace/run/inputs/camera/01184/camera.pth \
  --semantic workspace/run/inputs/camera/01184/img.png \
  --multiview-manifest workspace/run/residuals/01184/render/render_manifest.json \
  --output-dir workspace/posterior_run/01184

$PY scripts/run_observation_anchored_fusion.py \
  --posterior workspace/posterior_run/01184/posterior_prior_100k.ply \
  --partial data/redwood/partial/01184.ply \
  --multiview-manifest workspace/run/residuals/01184/render/render_manifest.json \
  --output-dir workspace/final_run/01184
```

For heterogeneous datasets, write a JSON case manifest:

```json
[
  {
    "sample_id": "01184",
    "prior": "/absolute/path/to/registered_prior.ply",
    "partial": "/absolute/path/to/partial.ply",
    "camera": "/absolute/path/to/camera.pth",
    "semantic": "/absolute/path/to/img.png",
    "multiview_manifest": "/absolute/path/to/render_manifest.json"
  }
]
```

Then run the same frozen configuration over all cases:

```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/run_posterior_adapter_batch.py \
  --case-manifest workspace/cases.json \
  --output-root workspace/posterior_run \
  --sample-workers 2
```

The pre-fusion posterior is
`workspace/posterior_run/<sample>/posterior_prior_100k.ply`; the final fused
prediction is
`workspace/final_run/<sample>/observation_anchored_fused_100k.ply`. Each sample also
contains OT pairs, support states, the deformation field, camera overlays, and
no-GT residual and integrity diagnostics.

## Optional four-view diagnosis

When a textured GLB and its ordered pre/post-registration carriers are
available:

```bash
$PY scripts/run_multiview_diagnostics.py \
  --glb workspace/run/inputs/pixal/01184/pixal3d.glb \
  --source-carrier workspace/run/inputs/pixal/01184/pixal3d_sampled_100k.ply \
  --registered-prior workspace/run/registration/01184/final/camera1_amplified_registered_100k.ply \
  --partial data/redwood/partial/01184.ply \
  --camera workspace/run/inputs/camera/01184/camera.pth \
  --output-dir workspace/posterior_run/01184/multiview
```

Camera-1 is retained and three auxiliary views are selected by incremental
visibility of physical partial points with a minimum yaw separation.

## Offline evaluation

Run metrics only after predictions are frozen:

```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/evaluate_completions.py \
  --prediction-root workspace/posterior_run \
  --ground-truth-root data/redwood/gt \
  --output workspace/posterior_run/metrics/redwood
```

## Repository layout

```text
src/posterior_adapter.py              structure OT and posterior deformation
src/deformation_graph.py              reusable graph and strain primitives
src/multiview_diagnostics.py          informative camera selection/rendering
src/multiview_partial_evidence.py     prior-to-partial residual visualization
scripts/run_posterior_adapter.py      one-sample inference
scripts/run_posterior_adapter_batch.py dataset-independent batch inference
scripts/run_multiview_diagnostics.py  optional four-view audit
src/observation_anchored_fusion.py     fixed-cardinality observation fusion
scripts/run_observation_anchored_fusion.py final one-sample fusion
scripts/run_observation_anchored_fusion_batch.py parallel final fusion
scripts/run_scene_completion.py       independent scene-level extension
```

See [installation](docs/installation.md), [models](docs/models.md), the
[canonical object method](docs/core_registration_pipeline.md), and the
[scene-level route](docs/scene_completion.md).

## License

Project code is released under the [MIT License](LICENSE). External models,
datasets, and CUDA extensions retain their own licenses.
