# MVP completion evaluation protocol

This protocol is **offline-only**: `complete_pcds` are never materialized for
generation, registration, agent decisions, or adaptation.  They are opened
only by `scripts/evaluate_mvp_protocol.py` after a prediction collection has
finished.

## Test split

Use `data/MVP/MVP_Test_CP.h5` in its supplied canonical coordinate system.
It contains 41,600 `incomplete_pcds`, 1,600 `complete_pcds`, and 16 class
labels.  Each complete target has 26 partial views.  The test runner therefore
uses the H5 mapping `complete_index = partial_index // 26`.

`scripts/prepare_mvp_full_protocol.py` writes an inference-only contiguous
shard and manifest, for example:

```bash
/opt/data/private/cr/miniconda3/envs/genpc/bin/python scripts/prepare_mvp_full_protocol.py \
  --h5 /opt/data/private/cr/lab/GenPC/data/MVP/MVP_Test_CP.h5 \
  --output-root workspace/mvp_full/shard_00000_00064 --start 0 --stop 64
```

The manifest deliberately contains partial paths and semantic labels only; it
does not contain an offline complete-cloud path.

## Paper-facing metrics

For every final prediction:

1. deterministically farthest-point sample it to **16,384** points using
   `start_idx=0`;
2. retain the official H5 complete target at its native **2,048** points (do
   not synthetically upsample it);
3. report symmetric squared Chamfer distance as **CD-L2 x 10^4**;
4. report **F-score@1%**, with a Euclidean threshold of `0.01` (equivalently
   squared nearest-neighbour threshold `1e-4`).

Run the offline evaluator on one finished shard or a full collection:

```bash
CUDA_VISIBLE_DEVICES=0 /opt/data/private/cr/miniconda3/envs/genpc/bin/python \
  scripts/evaluate_mvp_protocol.py \
  --h5 /opt/data/private/cr/lab/GenPC/data/MVP/MVP_Test_CP.h5 \
  --collection-root workspace/mvp_full/shard_00000_00064 \
  --manifest workspace/mvp_full/shard_00000_00064/inference_manifest_00000_00064.json \
  --output workspace/mvp_full/shard_00000_00064/mvp_16k_metrics.json
```

The JSON contains per-sample values, arithmetic means, and a missing-case
list.  `--allow-missing` supports resuming a partially completed shard without
silently treating missing predictions as zeros.
