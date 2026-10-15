# MAGMA-SLAM

**Multi-View Gaussian Map Alignment for Collaborative Multi-Agent SLAM**

MAGMA-SLAM extends the MAGiC-SLAM codebase while retaining its local Gaussian
mapping, DINO loop-candidate retrieval, and GTSAM pose-graph backend. The new
registration path assigns view-aggregated DINO descriptors to Gaussian
landmarks, matches Gaussian-to-Gaussian, applies geometric consistency and
Kabsch/RANSAC, then refines every viable loop with point-to-plane ICP before
loop admission and pose-graph optimization.

The original output names remain available, including `merged_coarse.ply`,
`merged_refined.ply`, loop pickle files, rendering metrics, and wall-time
statistics. Old configs without a `registration` section continue through the
legacy path.

## Registration modes

`registration.descriptor_mode` accepts:

- `baseline`: inherited MAGiC-SLAM registration behavior;
- `single`: best valid observation per selected Gaussian;
- `multiview`: depth-weighted aggregation across at least two valid views.

Opacity contributes to landmark reliability/selection, not to each view's
descriptor weight. No ground-truth pose is used by matching, RANSAC, ICP, or
loop admission.

## Office-0 runs

Replace the input and output paths with local paths. The input directory must
have the same per-agent layout used by the existing Replica loader.

Baseline MAGiC-SLAM:

```bash
python run_slam.py configs/magma/baseline_magic.yaml \
  --input_path /path/to/ReplicaMultiagent/Office-0 \
  --output_path output/magma/office0/baseline/seed_0 --seed 0
```

Single-view Gaussian-to-Gaussian:

```bash
python run_slam.py configs/magma/magma_single.yaml \
  --input_path /path/to/ReplicaMultiagent/Office-0 \
  --output_path output/magma/office0/single/seed_0 --seed 0
```

Multi-view MAGMA-SLAM:

```bash
python run_slam.py configs/magma/magma_multiview.yaml \
  --descriptor_mode multiview \
  --input_path /path/to/ReplicaMultiagent/Office-0 \
  --output_path output/magma/office0/multiview/seed_0 --seed 0
```

Point-to-plane ICP is enabled with
`registration.refinement: point_to_plane_icp`; its threshold, iteration limit,
and admission gate are in `configs/magma/magma_multiview.yaml`.

Enable post-PGO Gaussian compaction without modifying the YAML:

```bash
python run_slam.py configs/magma/magma_multiview.yaml \
  --descriptor_mode multiview --enable_global_compaction \
  --input_path /path/to/ReplicaMultiagent/Office-0 \
  --output_path output/magma/office0/multiview_compact/seed_0 --seed 0
```

## Ablations

The landmark budgets requested by the benchmark are exposed directly:

```bash
python run_slam.py configs/magma/magma_multiview.yaml \
  --descriptor_mode multiview --max_landmarks 256 \
  --ambiguity_margin 0.0 --geometric_consistency distance \
  --input_path /path/to/ReplicaMultiagent/Office-0 \
  --output_path output/magma/office0/ablation_k256/seed_0 --seed 0
```

Use `--max_landmarks 256`, `512`, or `1024`; compare `single` and `multiview`;
and select `none`, `distance`, `distance_scale`, or
`distance_scale_relative_axis` for geometric consistency.

For a paired offline benchmark on the exact retrieved loops and finalized
submaps from one completed run:

```bash
python tools/benchmark/benchmark_single_multi.py \
  --run-dir output/magma/office0/multiview/seed_0 \
  --k 256 512 1024 --seeds 0 1 2 \
  --ambiguity-margins 0.0 0.01 \
  --geometries distance distance_scale_relative_axis
```

This command does not rerun loop retrieval. Descriptor/dense cache-build
times are logged separately from matcher and ICP time; multi-seed benchmark
time is never added to full-pipeline runtime.

## Colored TSDF export

TSDF is a post-SLAM export and does not replace the Gaussian map:

```bash
python tools/export_tsdf_mesh.py \
  --slam-output output/magma/office0/multiview/seed_0 \
  --voxel-size 0.01 --sdf-trunc 0.04 --depth-trunc 5.0 \
  --frame-stride 2 --min-component-triangles 1000 --zip
```

The exporter reconstructs every global camera pose from optimized submap-start
poses and local submap trajectories. It never reads a dataset pose for fusion.
Set `export.tsdf.enabled: true` to invoke the same exporter only after the full
pipeline has saved optimized poses and final maps.

## Tests

```bash
python -m pytest -q tests/test_magma_registration.py \
  tests/test_magma_backend_and_tsdf.py tests/test_version_migration.py
```

See [method changes](docs/METHOD_CHANGES.md) and
[version migration](docs/VERSION_MIGRATION.md) for the notebook audit, and
[output formats](docs/OUTPUT_FORMAT.md) for implementation details.
