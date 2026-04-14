# PoseDetect Rebuild (A/B Dual Track)

This repository now follows two model tracks:

- A-line: `MediaPipe Pose + engineered temporal features + XGBoost/RandomForest + smoothing + repetition counting`
- B-line: `OpenTAD/ActionFormer/VideoMAE target interface`, with a local `minimal` backend for 8GB smoke training.

## Data Entry

Use manifests under `data/rebuild_dataset`:

- `ucf_binary/group/{train,val,test}.csv` (recommended)
- `ucf_binary/random/{train,val,test}.csv`
- `labels/class_map.csv`

Pose files:

- UCF: `data/pose/...`
- LLSP: `data/pose_llsp/...`

## A-line Commands

```powershell
python scripts/train_a_line.py --config configs/a_line.yaml
python scripts/export_pseudo_timestamps.py --config configs/a_line.yaml --split-variant group
```

Outputs:

- model: `outputs/checkpoints/a_line/a_line_model.joblib`
- metrics: `outputs/metrics/a_line_metrics.json`
- pseudo TAL labels: `outputs/pseudo_labels/group/{train,val,test}_tal.csv`

## B-line Commands

### Local smoke (8GB)

```powershell
python scripts/train_b_line.py --config configs/b_line.yaml --backend minimal
```

### Formal training target (4x5090)

Set `configs/b_line.yaml`:

- `backend: opentad`
- `opentad_train_cmd: python tools/train.py configs/your_actionformer_videomae.py`

Then run:

```powershell
python scripts/train_b_line.py --config configs/b_line.yaml --backend opentad
```

## Unified Inference Output

A-line inference:

```powershell
python scripts/infer_segments.py --line a --config configs/a_line.yaml --pose-path data/pose/PushUps/v_PushUps_g01_c01.npz --video-path dataset/PushUps/v_PushUps_g01_c01.avi --out-json outputs/predictions/a_line_example.json
```

B-line minimal inference:

```powershell
python scripts/infer_segments.py --line b_minimal --config configs/b_line.yaml --pose-path data/pose/PushUps/v_PushUps_g01_c01.npz --video-path dataset/PushUps/v_PushUps_g01_c01.avi --out-json outputs/predictions/b_line_example.json
```

Schema contains:

- `segments`: detected action intervals
- `instances`: rep-level intervals
- `summary`: count and duration statistics by label
