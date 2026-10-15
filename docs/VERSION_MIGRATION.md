# V3.2 / V3.3.1 migration audit

## Phạm vi nguồn

Workspace chỉ có một notebook đính kèm:
`single_multi_replica_office0.ipynb` (46 cell), tiêu đề V3.3.1. Không có
notebook V3.2 độc lập. Mã V3.2 được notebook này nhúng nguyên văn trong cell
22, biến `_SB_HELPERS`, với chú thích các cell lịch sử 60–67. Vì vậy bảng dưới
đây chỉ dựa trên các hàm thực sự xuất hiện trong notebook, không suy luận từ
tên phiên bản.

Các cell 4–12 và phần điều phối trong cell 14 là cài đặt/đóng gói môi trường
Kaggle. Chúng không được đưa vào `src/registration/`.

## Bảng đối chiếu

| Version | Notebook source | Feature/fix | Code location | Test | Status |
|---|---|---|---|---|---|
| V3.3.1 | Cell 22, embedded cells 46–48: `mv_build_cache` | Chiếu Gaussian vào mọi keyframe, kiểm tra visibility/depth | `src/registration/multiview_descriptor.py`: `depth_consistent_projection`, `build_gaussian_landmark_descriptors` | `test_projection_and_depth_weights` | Đã tích hợp từ lượt trước |
| V3.3.1 | Cell 22, `mv_build_cache` | Trọng số depth-consistency và descriptor Multi-view chuẩn hóa | `src/registration/multiview_descriptor.py`: `depth_consistency_weights`, `aggregate_multiview_descriptors` | `test_multiview_descriptor_normalization_and_minimum_observations` | Đã tích hợp từ lượt trước; opacity chỉ dùng ở reliability thay vì nhân lặp theo view |
| V3.3.1 | Cell 22, `mv_build_cache`: “ONE shared selected landmark set” | Single-best-view và Multi-view dùng đúng cùng Gaussian source | `src/registration/multiview_descriptor.py`, `src/registration/coarse_registration.py`: cache không còn phụ thuộc mode | `test_single_and_multiview_share_landmark_source` | Đã tích hợp trong lượt này |
| V3.3.1 | Cell 22, `valid = obs_count >= MV_MIN_OBSERVATIONS` | Minimum multi-view observations áp dụng cho tập source chung | `src/registration/multiview_descriptor.py` | `test_single_and_multiview_share_landmark_source` | Đã tích hợp trong lượt này |
| V3.3.1 | Cell 22, `stability`, `score` trong `mv_build_cache` | Descriptor stability, observation count, residual và opacity reliability | `src/registration/landmark_selection.py`: `landmark_reliability` | `test_multiview_descriptor_normalization_and_minimum_observations` | Đã tích hợp từ lượt trước |
| V3.3.1 | Cell 22, `mv_spatial_topk` | Spatial/voxel Gaussian landmark selection | `src/registration/landmark_selection.py`: `select_spatially_balanced_landmarks` | `test_spatial_selection_honors_k_and_voxels` | Đã tích hợp từ lượt trước |
| V3.3.1 | Cell 22, worker `sb_slice_cache`; specs K | K = 256, 512, 1024 trên cùng cache/source | CLI `run_slam.py`; `tools/benchmark/benchmark_single_multi.py`; `tools/benchmark/protocol.py:landmark_view` | `test_single_and_multiview_share_landmark_source`; CLI smoke | Đã tích hợp trong lượt này |
| V3.3.1 | Cell 22, embedded cell 49: `mv_match_descriptors` | Mutual nearest-neighbor | `src/registration/gaussian_matching.py`: `mutual_nearest_neighbors` | `test_mutual_nn_ambiguity_margin` | Đã tích hợp từ lượt trước |
| V3.3.1 | Cell 22, `mv_match_descriptors` | Optional bidirectional ambiguity margin | `src/registration/gaussian_matching.py`; benchmark `--ambiguity-margins` | `test_mutual_nn_ambiguity_margin`; CLI smoke | Đã tích hợp |
| V3.3.1 | Cell 22, `mv_geometric_compatibility`, `mv_topk_rigidity_support` | Distance-only consistency | `src/registration/geometric_consistency.py` | `test_pairwise_rigidity_rejects_inconsistent_match` | Đã tích hợp từ lượt trước |
| V3.3.1 | Cell 22, worker `sb_shape_support` ablation | Optional distance + scale + relative rotation-axis consistency | `src/registration/geometric_consistency.py`; benchmark `--geometries distance_scale_relative_axis` | production unit path + CLI smoke | Đã tích hợp từ lượt trước |
| V2/V3 | Cell 22, embedded cells 63/65: `v2_weighted_kabsch`, `v3_prosac_ransac` | Kabsch/RANSAC, confidence-guided PROSAC | `src/registration/coarse_registration.py`: `estimate_rigid_transform`, `ransac_rigid_transform` | `test_kabsch_and_ransac_with_outliers` | Đã tích hợp đầy đủ trong lượt này |
| V3 | Cell 22, embedded cell 65: `v3_consensus_stats` | RANSAC scoring cải tiến: weighted ratio, ratios 5/3 cm, residual quality | `src/registration/coarse_registration.py`: composite scoring | `test_hard_recovery_and_rigid_validation` | Đã tích hợp trong lượt này |
| V2/V3 | Cell 22, embedded cells 63/65 | Refit trên toàn bộ consensus inlier, sau đó đánh giá lại | `src/registration/coarse_registration.py`: `ransac_rigid_transform` | `test_kabsch_and_ransac_with_outliers` | Đã tích hợp; mặc định bật |
| V3 | Cell 22, embedded cell 64: `v3_spatial_coverage_select` | Coverage fix: luôn chạy voxel pass kể cả khi số pair dưới cap | `src/registration/landmark_selection.py`: `select_spatially_covered_correspondences` | `test_coverage_fix_runs_below_pair_cap` | Đã tích hợp trong lượt này |
| V3.2 | Cell 22, embedded cells 66–67: `v32_build_hard_candidates`, `v32_registration_once` | Hard-case trigger, mutual Top-K pool, quality, unique pair, spatial coverage và recovery | `src/registration/gaussian_matching.py`: `mutual_topk_candidates`; `src/registration/coarse_registration.py`: guarded hard recovery | `test_hard_recovery_and_rigid_validation` | Đã tích hợp chức năng; production dùng hard RANSAC recovery thay cho local IRLS/line-search nguyên bản |
| V3.2 | Cell 22, embedded cell 66: `v32_fractional_rotation`, `v32_select_local_candidate` | Local full/rotation/translation IRLS với trust-region line search | Không port nguyên xi vào production; hard recovery hiện dùng rigid Kabsch/RANSAC và SE(3) gate | Không có | Còn thiếu; đây là nhánh nghiên cứu riêng, không nằm trong checklist V3.3.1 bắt buộc |
| V3.3.1 | Cell 22, embedded cells 79/81/83: `v331_registration_once` | Spatial holdout, observability-gated rotation, cycle validation, giữ nguyên V3.2 nếu gain chưa đủ | Không port nguyên xi; production giữ giải pháp fast nếu hard candidate không có score tốt hơn | `test_hard_recovery_and_rigid_validation` chỉ kiểm tra safety cơ bản | Tích hợp một phần |
| V3.2/V3.3.1 | Cell 22 worker: `sb_pose` | Rigid-rotation validation, homogeneous row và finite values | `src/registration/coarse_registration.py:is_valid_rigid_transform`; `src/registration/icp_refinement.py:is_rigid_transform`; production frame gates | `test_hard_recovery_and_rigid_validation`; `test_icp_failure_is_reported_not_raised` | Đã tích hợp trong lượt này |
| Shared | Cell 22, embedded cell 69: `exact_production_icp` | Point-to-plane ICP, cùng threshold/iteration config, initial/final fitness/RMSE | `src/registration/icp_refinement.py`; `src/utils/magic_slam_utils.py` | `test_icp_failure_is_reported_not_raised`; `test_loop_and_submap_start_transform_roundtrip` | Đã tích hợp; notebook không có hàm riêng tên “ICP-Plus” |
| Shared | Cell 22 worker `sb_attempt` | Không trả transformation không hợp lệ; lưu failure reason | `src/registration/coarse_registration.py`, `src/registration/icp_refinement.py`, `src/utils/magic_slam_utils.py`, `src/entities/logger.py` | `test_failure_never_returns_a_transform` | Đã tích hợp trong lượt này |
| Shared | Cell 22, `loop_to_submap_start_transform`, `submap_start_to_loop_transform` | Chiều source-loop → target-loop và đổi sang source-start → target-start | `src/registration/icp_refinement.py` | `test_loop_and_submap_start_transform_roundtrip` | Đã tích hợp từ lượt trước |
| Benchmark | Cell 22 worker `SBContext`, `sb_prepare_caches`, `sb_attempt` | Fixed retrieved loops, finalized submaps, cache chung, cùng seed/ICP | `tools/benchmark/benchmark_single_multi.py` | protocol/unit tests + `--help` smoke | Đã tích hợp trong lượt này |
| Benchmark | Cell 22 worker `sb_summaries`, `sb_pairwise` | Pass rate trên mọi trial; conditional error chỉ trên valid estimate; failure reason | `tools/benchmark/protocol.py:summarize_trials` | `test_benchmark_summary_denominators_and_conditional_errors` | Đã tích hợp trong lượt này |
| Benchmark | Cell 22 worker `sb_prepare_caches` | Tách `cache_build_time_s_measured` khỏi matcher runtime, không trùng cột | `tools/benchmark/benchmark_single_multi.py`: `descriptor_cache_runtime.csv` | source/CLI smoke | Đã tích hợp trong lượt này |
| Benchmark | Cell 22 worker `sb_fpfh_register` | FPFH reference baseline trên cùng loop và ICP config | `tools/benchmark/benchmark_single_multi.py`, bật mặc định bằng `--include-fpfh` | CLI smoke; cần dataset để chạy số liệu | Đã tích hợp trong lượt này |
| Benchmark | Cell 22 worker specs | Single/Multi, K, margin và geometry ablations | `tools/benchmark/benchmark_single_multi.py` | CLI smoke | Đã tích hợp trong lượt này |
| Boundary fix | `mv_find_submap_index` dùng `start <= frame <= end` | Ánh xạ loop/submap và biên đầu/cuối inclusive | `src/utils/submap_utils.py`; `tools/benchmark/protocol.py` | `test_inclusive_submap_boundaries_and_unique_mapping` | Đã tích hợp trong lượt này |
| Pose fix | Worker kiểm tra input/fingerprint; TSDF cell 45 | Optimized poses phải khớp chính xác keyframe IDs | `src/utils/pose_utils.py`; `src/utils/magic_slam_utils.py`; `tools/export_tsdf_mesh.py`; benchmark loader | `test_optimized_pose_ids_are_not_positionally_guessed` | Đã tích hợp trong lượt này |
| Dataset fix | Cell 22 `SBContext`; cell 45 dataset loader | Agent-folder resolution theo config agent, tên folder hoặc vị trí trong `agent_ids` | `tools/export_tsdf_mesh.py`; `tools/benchmark/protocol.py` | `test_agent_folder_resolution_and_gpu_shortage` | Đã tích hợp trong lượt này |
| Runtime fix | Notebook full-run multi-agent setup | Báo lỗi sớm khi GPU ít hơn agent, không tạo device ID vượt số GPU | `src/utils/gpu_utils.py:build_agent_gpu_map` | `test_agent_folder_resolution_and_gpu_shortage` | Đã có từ lượt trước, đã kiểm chứng lại |
| Compaction | Cells 28–39 | Post-PGO Gaussian compaction/pruning | `src/mapping/gaussian_compaction.py`, backend merge/refine | `test_compaction_opacity_voxel_and_cap` | Đã tích hợp từ lượt trước; không thay đổi lượt này |
| TSDF | Cell 45 | Colored TSDF bằng optimized SLAM poses, không dùng GT pose | `tools/export_tsdf_mesh.py` | `test_tiny_colored_tsdf_fusion`, pose-ID tests | Đã tích hợp từ lượt trước; validation ID bổ sung lượt này |
| Kaggle env | Cells 4–12, phần setup của cell 14 | Clone, venv, pip/apt, CUDA extension build, dataset download | Không đưa vào thuật toán | `rg` audit | Không port theo chủ đích |

## Ranh giới production và benchmark

- `src/registration/` chỉ chứa descriptor construction, landmark selection,
  matching, geometric consistency, Kabsch/RANSAC, hard recovery và ICP dùng
  trực tiếp bởi full SLAM.
- `tools/benchmark/` chỉ đọc output đã hoàn tất. Nó không gọi
  `LoopDetector.detect_loops`, không sửa `walltime_stats.json`, và không cộng
  tổng thời gian nhiều seed vào runtime full pipeline.
- `cache_build_time_s_measured` chỉ xuất hiện một lần trong
  `descriptor_cache_runtime.csv`; trial rows chỉ chứa matcher/ICP/online time.
- Ground truth chỉ được đọc sau khi coarse/ICP/admission đã quyết định xong.
- Các cell cài Kaggle không xuất hiện trong production hoặc benchmark
  algorithm.

## Cấu hình

- Baseline: `configs/magma/baseline_magic.yaml`
- Single: `configs/magma/magma_single.yaml`
- Multi-view: `configs/magma/magma_multiview.yaml`

Ba mode đều đi qua `validate_registration_config`. Single và Multi-view dùng
cùng builder/cache landmark; chỉ trường descriptor được chọn khác nhau.

## Chạy benchmark offline

```bash
python tools/benchmark/benchmark_single_multi.py \
  --run-dir /path/to/completed_slam_output \
  --k 256 512 1024 --seeds 0 1 2 \
  --ambiguity-margins 0.0 0.01 \
  --geometries distance distance_scale_relative_axis
```

Output gồm `protocol.json`, `descriptor_cache_runtime.csv`,
`dense_cache_runtime.csv`, `trials.csv`, `summary.csv` và `summary.json`.
Mặc định benchmark bắt buộc `detected_loops.pkl`. `loops.pkl` chỉ được phép
với cờ `--allow-legacy-loops`, dành cho run cũ và được ghi rõ trong protocol.
