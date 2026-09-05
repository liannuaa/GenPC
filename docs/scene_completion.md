# Scene-level completion wrapper

The fixed GenPC+ mainline reconstructs one object from one partial point cloud.
The scene wrapper is a separate instance-level route: each object's partial is
the visible subset of one scene-wide Pixal MoGe observation, and the final
output is a set of complete, textured, registered Pixal meshes. It does not
modify the object-level Pixal or registration code or its frozen parameters.

```text
RGB scene image
  → Pixal MoGe-2 visible scene cloud (one shared camera coordinate system)
  → Codex GPT image instance masks
  → one mask-indexed MoGe partial cloud per visible object
  → direct GPT semantic completion per isolated RGB crop (no depth raster)
  → Pixal3D textured GLB + native Pixal--MoGe alignment (camera 2)
  → pixel-indexed camera-2 → camera-1 bridge into the scene partial frame
  → bridge Sim(3) baked directly into each GLB's mesh vertices
  → merged complete textured-mesh scene
```

## Why the shared scene frame matters

MoGe-2 is run once on the original RGB scene with the same Pixal contract used
by native registration: `PIL RGB → float/255 CHW → MoGeModel.infer`, using the
same local Pixal MoGe-2 checkpoint. A GPT mask selects points by their exact
source-image pixels. The resulting PLY is the object's **partial**—there is no
separate depth-derived partial. Object partials are never centered, normalized,
or transformed after extraction. Before extraction, every binary instance mask
uses a fixed 5-pixel erosion to suppress unreliable MoGe boundary points; a
thin mask that would become too small safely falls back to its un-eroded mask.

The wrapper saves perspective Camera-1 metadata (`camera.pth` and `point_uv.npy`)
for this scene-MoGe frame. It does not save or complete a depth image. Pixal's
input is a separate square semantic image, so its native MoGe observation is
**camera 2**, not the scene-MoGe camera. The scene route therefore retains the
minimal two-camera bridge: foreground-support transfer provides indexed
camera-1/camera-2 pairs, robust Sim(3) maps native MoGe to the scene partial,
and that transform composes with native Pixal--MoGe alignment. Its resulting
`pixal_to_partial` matrix is already in the original shared scene coordinate
system, including the instance's scene translation and scale. Scene mode stops
there—there is no joint, amplified, wide-tilt, or final Camera-1 residual.

For final composition, the scene mask supplies a robust scene-MoGe **position
anchor** (a fixed 2–98% coordinate trim rejects boundary-depth outliers), while
the two-camera bridge retains its own rotation and isotropic scale. Finally the
dominant plane in the unmasked scene-MoGe context is rotated once into a
table-aligned `Y-up` world frame. This common rigid transform changes neither
an object's registration nor any inter-object relative pose; it prevents a
free-view GLB renderer from presenting the camera-frame tabletop edge-on.

## GPT image assets

Codex's GPT image tool is an interactive agent capability rather than a hidden
runtime Python dependency.  The wrapper therefore consumes saved, auditable
assets:

1. A `scene_instances.json` manifest lists each instance ID, free-text object
   label, optional layer, and binary mask.  Every mask must have the exact RGB
   scene resolution; white is the visible object, black is everything else.
   Greyscale, monochrome RGB, and RGBA-alpha masks are accepted; coloured
   cutouts and checkerboard previews are rejected.
2. The wrapper writes `gpt_image_actions.json`. Its `completion_task` takes
   only `masked_scene_crop.png` and asks GPT Image to directly produce one
   complete semantic RGB object on white. Its mandatory **pose lock** treats
   that crop as the only camera reference: object direction, yaw/pitch/roll,
   image-plane angle, perspective/foreshortening, silhouette, scale and
   visible structure must be preserved. It may complete only unseen portions;
   frontal/top-down/catalog canonicalisation and a depth image are forbidden.
3. Save each direct result as `<gpt-image-source>/<instance-id>.png`. The
   wrapper aspect-preserving-pads it to a 512px square and installs the exact
   same image as both `inputs/camera/<instance-id>/img.png` and
   `inputs/pixal/<instance-id>/gpt_image.png`. RMBG makes the corresponding
   Camera-1 foreground mask. The wrapper refuses to silently replace a frozen
   GPT asset.

A minimal manifest is:

```json
{
  "schema_version": 1,
  "source_image": "/absolute/path/to/scene.png",
  "instances": [
    {
      "id": "sofa_0",
      "label": "blue three-seat sofa",
      "mask": "masks/sofa_0.png",
      "layer": 1
    }
  ]
}
```

Layers define only source-image occlusion ownership: a larger layer is closer
to the camera, so it keeps an overlapping pixel before its scene-MoGe partial
is extracted. Ties follow manifest order. They never select a registration
branch or use GT.

## Commands

Set `PY` to the project interpreter and choose a new output root:

```bash
PY=/opt/data/private/cr/miniconda3/envs/genpc/bin/python
SCENE=data/scene_samples/scene_3.png
MANIFEST=workspace/scene_3_gpt/scene_instances.json
RUN=workspace/scene_3_genpc_plus
```

Use `prepare` after producing the GPT masks.  It makes the shared scene MoGe
cache, per-instance partial PLYs, masked RGB crops, and unmasked scene context:

```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/run_scene_completion.py \
  --scene-image "$SCENE" --instances "$MANIFEST" --output-root "$RUN" \
  --stage prepare
```

Write the exact direct-GPT semantic tasks. There is intentionally no Qwen or
depth-to-semantic stage in the scene route:

```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/run_scene_completion.py \
  --scene-image "$SCENE" --instances "$MANIFEST" --output-root "$RUN" \
  --stage gpt-actions
```

After saving the declared GPT results under `$GPT/<instance-id>.png`, dispatch
Pixal and the fixed registration route, then export registered meshes:

```bash
GPT=workspace/scene_3_gpt/clarity_images
CUDA_VISIBLE_DEVICES=0 $PY scripts/run_scene_completion.py \
  --scene-image "$SCENE" --instances "$MANIFEST" --output-root "$RUN" \
  --gpt-image-source "$GPT" --stage pixal

CUDA_VISIBLE_DEVICES=0 $PY scripts/run_scene_completion.py \
  --scene-image "$SCENE" --instances "$MANIFEST" --output-root "$RUN" --stage registration

CUDA_VISIBLE_DEVICES=0 $PY scripts/run_scene_completion.py \
  --scene-image "$SCENE" --instances "$MANIFEST" --output-root "$RUN" --stage meshes
```

Registration is independent across instances. Scene mode therefore uses two
worker processes by default on the 24GB reference GPU; each worker runs the
same native-plus-bridge route for one object. This changes throughput, not any
objective, correspondence, or selected transform. Use `--registration-workers
1` for the byte-comparable serial ablation, or lower the count when another GPU
job is resident.

The final files are:

```text
$RUN/scene_meshes/completed_scene_registered_meshes.glb
$RUN/scene_meshes/mesh_scene_manifest.json
$RUN/registration/<instance-id>/scene_bridge_placement/partial_gray_pixal_red.ply
```

## Reproducible scene bundle

To reproduce an exported scene rather than merely view its final GLB, retain
the complete `$RUN/` directory together with the source RGB image and
`scene_instances.json`. In particular, a self-contained audit bundle must
preserve:

```text
$RUN/scene_moge/                         shared RGB-scene MoGe observation
$RUN/masks/ and $RUN/instances/          pixel-grid masks and isolated crops
$RUN/gpt_image_actions.json              exact mask/completion prompts
$RUN/gpt_outputs/                        frozen direct GPT semantic images
$RUN/inputs/{partial,camera,pixal}/      installed assets, Pixal GLBs and MoGe caches
$RUN/registration/                       native alignment and bridge transforms
$RUN/scene_meshes/mesh_scene_manifest.json
                                         scene placement and FCL collision record
```

The direct GPT images are external interactive assets, so they must be saved
and reused exactly; the runner intentionally never regenerates or silently
replaces them. The batch delivery at
`workspace/final_scene_meshes_20260905/` follows this layout for the five
accepted scene outputs.

Every complete object mesh—including its texture and unobserved surfaces—is
retained. The final scene mesh is not a point-cloud fusion or Gaussian decode:
it is a GLB composition of the exact meshes transformed by the explicit
camera-2-to-camera-1 bridge matrices. Per-instance transformed GLBs are not
written by default, avoiding duplicate storage; use `--write-instance-meshes`
only for debugging. Scene Pixal generation uses a 100k-face target by default;
change it with `--pixal-decimation-target` when a smaller final asset is needed.
The merged GLB is expressed in the table-aligned scene world frame, and its
exact global transform is saved in `mesh_scene_manifest.json`. It remains a
`trimesh.Scene` to retain one textured material per object, but every
instance's Sim(3) is baked into mesh vertices before composition. Thus the
final GLB has identity instance nodes and displays correctly even in viewers
that do not recursively apply nested GLTF node transforms.

Before export, scene mode also runs exact mesh collision detection through
`python-fcl`. For each colliding pair it retains every Pixal rotation and
scale, selects the object already farther according to the original
scene-MoGe mask-anchor depth (not the generated mesh's centre),
and finds the smallest extra displacement along that same camera's positive
optical axis. The default `--collision-clearance .0015` adds a small gap and
does not move an object laterally or along the table normal, so the original
scene-image projection remains stable. The collision search has no
object-scale trust-region cap: it brackets the first collision-free point
along the depth ray and returns the smallest feasible depth translation. Set
`--collision-clearance 0` to disable this optional final assembly refinement.
