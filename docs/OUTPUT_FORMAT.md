# MAGMA-SLAM output format

## Loop registration

- `detected_loops.pkl`: candidate identities before registration mutates them.
- `loops.pkl`: every attempted registration, accepted or rejected.
- `filtered_loops.pkl`: only admitted pose-graph constraints.
- `registration_metrics.csv`: per-loop transforms, counts, timing, ICP metrics,
  admission result, and failure reason. `T_init` and `T_refined` are JSON
  4-by-4 arrays stored in CSV fields.
- `registration_summary.json`: aggregate registration and evaluation metrics.
- `loop_pipeline_summary.json`: retrieval/registration mode and candidate counts.

Ground truth appears only in post-registration error/evaluation columns. It is
not an input to matching, RANSAC, ICP, or the admission gate.

## Gaussian maps

- `merged_coarse.ply`: transformed and merged map after PGO and optional
  one-shot compaction, before global refinement.
- `merged_refined.ply`: final Gaussian map.
- `gaussian_compaction_stats.json`: counts before/after merge, compaction and
  refinement; runtimes; model/file sizes; and peak CUDA allocation when CUDA is
  available.
- `coarse_render_metrics.json`, `fine_render_metrics.json`, and optionally
  `novel_render_metrics.json`: inherited render evaluation outputs.
- `walltime_stats.json`: registration, merge and refinement wall times.

No synthetic metric values are emitted. Files that require a full dataset run
are created only by that run.

## Colored TSDF export

`tools/export_tsdf_mesh.py` creates:

```text
tsdf_mesh/
├── <scene>_tsdf_mesh.ply
├── <scene>_tsdf_mesh_preview.ply
├── tsdf_stats.json
├── tsdf_integrated_frames.csv
├── tsdf_params.json
└── <method>_<scene>_tsdf_mesh.zip   # only with --zip
```

`tsdf_stats.json` records dataset, scene, SLAM source, optimized-pose source,
frame count, vertex/triangle counts, bounds, voxel/truncation values, runtime,
and mesh size. `tsdf_integrated_frames.csv` contains one row per fused agent,
submap and frame.
