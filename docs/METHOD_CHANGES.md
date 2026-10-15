# MAGMA-SLAM method changes

## Baseline architecture

The inherited pipeline has one process per agent for RGB-D tracking, Gaussian
local mapping, keyframing, and submap finalization. The main process retrieves
loop candidates with DINO/FAISS, registers candidate submaps, builds a g2o or
GTSAM pose graph, applies optimized pose corrections to submaps, merges their
Gaussians, and globally refines the merged map.

MAGMA-SLAM does not replace local mapping, DINO candidate retrieval, or GTSAM.
Its main changes are between candidate retrieval and PGO, and after global map
merge.

## Notebook-to-module mapping

| Notebook cell/function | Repository destination | Production status |
|---|---|---|
| Cell 22, embedded original cells 46–48: projection, bilinear DINO sampling, view accumulation, best-view descriptor | `src/registration/multiview_descriptor.py` | Production; paths and notebook globals removed |
| Cell 22: `mv_spatial_topk`, stability/observation/residual/opacity score | `src/registration/landmark_selection.py` | Production; opacity moved to reliability only |
| Cell 22, embedded cells 49–50: `mv_match_descriptors` / `a_match_descriptors` | `src/registration/gaussian_matching.py` | Production MNN, cosine and optional ambiguity gate |
| Cell 22: `mv_geometric_compatibility`, `mv_topk_rigidity_support` | `src/registration/geometric_consistency.py` | Production distance consistency; scale/axis terms are optional ablations |
| Cell 22: `mv_ransac_baseline`, `a_ransac`, weighted Kabsch concepts | `src/registration/coarse_registration.py` | Production three-point RANSAC with rigid Kabsch refit |
| Cell 22, embedded cell 69: `exact_production_icp` and loop/start-frame conversions | `src/registration/icp_refinement.py`, production registration integration | Production ICP and admission; pose convention is explicit source-to-target |
| Cells 28–39: post-PGO opacity/voxel/cap compaction and periodic pruning | `src/mapping/gaussian_compaction.py`, `magic_slam_utils.py` | Production, config-gated and disabled by default |
| Cell 45: colored scalable TSDF fusion with optimized-pose correction | `tools/export_tsdf_mesh.py` | Production post-processing tool |
| Cells 14–20: Kaggle clone/build/full-run/handoff orchestration | Not migrated | Environment-specific runner only |
| Cell 22: V3 coverage, confidence scoring and V3.2 hard candidates | `src/registration/landmark_selection.py`, `gaussian_matching.py`, `coarse_registration.py` | Production coverage/PROSAC/hard recovery; exact local IRLS is not ported |
| Cell 22: V3.3.1 holdout/cycle-validation and paper figures | `tools/benchmark/` for protocol/reporting; exact notebook holdout remains notebook-only | Production keeps a smaller guarded hard-recovery path |
| Cells 35–43: notebook comparison frames, archives, Gaussian Poisson-style mesh | Not migrated | Benchmark/presentation utility; TSDF is the supported colored mesh export |

## Registration flow

```text
DINO candidate retrieval (baseline)
  -> Gaussian projection into every submap keyframe
  -> visibility and RGB-D depth consistency
  -> single-best or depth-weighted multi-view descriptor
  -> reliability + voxel-balanced landmark selection
  -> cosine mutual nearest neighbors
  -> optional Top-1/Top-2 ambiguity filter
  -> pairwise rigidity consistency
  -> source/target voxel coverage
  -> confidence-guided PROSAC + composite scoring + all-inlier Kabsch refit
  -> optional mutual-Top-K hard-case recovery
  -> point-to-plane ICP on dense submap geometry
  -> fitness/RMSE admission gate
  -> GTSAM/g2o pose-graph optimization
```

For a Gaussian `g` in view `v`, the descriptor weight is
`beta(g,v) = exp(-abs(z_projected - z_depth) / tau_depth)`. Multi-view
descriptors are the L2-normalized weighted sum. Opacity does not multiply each
observation; it enters the landmark reliability score with descriptor
stability, observation count, and mean depth residual.

All registration transforms map source camera coordinates to target camera
coordinates in metres. Rotation validity is checked before and after ICP.
Failures are recorded per candidate and do not terminate the full run.

## Global-map flow

```text
local mapping -> submap finalization -> loop registration -> PGO
-> transform submaps -> merge -> optional global compaction
-> global refinement + optional periodic pruning -> final Gaussian map
```

When `global_compaction.enabled: false`, merge behavior and the original
MAGiC-SLAM refinement pruning condition are preserved. Compaction never runs
before descriptor construction or loop registration.

## Pose correction used by TSDF

For each submap:

```text
T_correction = T_optimized,start @ inverse(T_local,start)
T_global,i   = T_correction @ T_local,i
T_open3d     = inverse(T_global,i)
```

The exporter reads original RGB-D through the repository loader but ignores
the pose returned by that loader.
