# MVGS-SLAM

Production repository for multi-agent Gaussian SLAM with MAGiC-SLAM
baseline, Single-view and Multi-view registration, point-to-plane ICP,
optional Gaussian compaction, and optional colored TSDF export.

## Installation

The environment targets Python 3.11, PyTorch 2.5.1, torchvision 0.20.1 and
NumPy 1.26.4.

```bash
python -m pip install -r requirements.txt
python scripts/install_cuda_extensions.py --check-only
python scripts/install_cuda_extensions.py
```

The CUDA installer checks `torch.version.cuda` against `nvcc` before building
the pinned `gaussian_rasterizer` and `simple_knn` extensions.

## Run SLAM

Each supported scene under `configs/experiments/` provides `baseline.yaml`,
`single.yaml` and `multiview.yaml`.

```bash
python run_slam.py configs/experiments/replica_office_0/baseline.yaml \
  --input_path /data/ReplicaMultiagent/Office-0 \
  --output_path /outputs/office0/baseline --seed 0

python run_slam.py configs/experiments/replica_office_0/single.yaml \
  --input_path /data/ReplicaMultiagent/Office-0 \
  --output_path /outputs/office0/single --seed 0

python run_slam.py configs/experiments/replica_office_0/multiview.yaml \
  --input_path /data/ReplicaMultiagent/Office-0 \
  --output_path /outputs/office0/multiview --seed 0
```

Available scene directories are `replica_office_0`, `replica_apart_0`,
`replica_apart_1`, `replica_apart_2`, `tum_fr1_desk`, `aria_room_0` and
`aria_room_1`. Add `--multi_gpu` for multi-process GPU execution. Explicit GPU
assignment can be configured as:

```yaml
runtime:
  server_gpu: 0
  agent_gpu_map: {0: 0, 1: 1, 2: 1}
  strict_gpu_mapping: false
```

## Optional production outputs

Enable Gaussian compaction:

```bash
python run_slam.py <config.yaml> --enable_global_compaction
```

Enable post-SLAM TSDF export and ZIP packaging:

```bash
python run_slam.py <config.yaml> --enable_tsdf_export --tsdf_zip
```

Export TSDF from an existing completed run:

```bash
python tools/export_tsdf_mesh.py --slam-output /outputs/completed_run --zip
```

Use `python run_slam.py --help` for all runtime overrides.
