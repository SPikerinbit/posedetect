# PoseDetect Rebuild (A/B Dual Track)

This repository now follows two model tracks:

- A-line: `MediaPipe Pose + engineered temporal features + XGBoost/RandomForest + smoothing + repetition counting`
- B-line: `complete temporal localization model (conv+transformer+boundary regression+NMS)`, and `OpenTAD/ActionFormer/VideoMAE` integration interface.

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

### Local full training

```powershell
python scripts/train_b_line.py --config configs/b_line.yaml --backend full
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

B-line full inference:

```powershell
python scripts/infer_segments.py --line b_full --config configs/b_line.yaml --pose-path data/pose/PushUps/v_PushUps_g01_c01.npz --video-path dataset/PushUps/v_PushUps_g01_c01.avi --out-json outputs/predictions/b_line_example.json
```

Schema contains:

- `segments`: detected action intervals
- `instances`: rep-level intervals
- `summary`: count and duration statistics by label

## Penn Action Pipeline (pushup + jumping_jacks)

Dataset root assumed at `/data1/shiyuqi/Penn_Action`.

### 1) Prepare official split + local val split

```powershell
python scripts/prepare_penn_action.py --penn-root /data1/shiyuqi/Penn_Action --out-root data/penn_action/rebuild_dataset --val-ratio 0.1
```

### 2) Extract MediaPipe pose from frame folders

```powershell
python scripts/extract_pose_from_frames.py --split-dir data/penn_action/rebuild_dataset/ucf_binary/penn --pose-dir data/penn_action/pose --frames-root /data1/shiyuqi/Penn_Action/frames --assume-fps 30
```

### 3) Train A-line and export pseudo timestamps

```powershell
python scripts/train_a_line.py --config configs/penn_a_line.yaml
python scripts/export_pseudo_timestamps.py --config configs/penn_a_line.yaml --split-variant penn --out-root outputs/pseudo_labels
```

### 4) Clean pseudo labels for B-line training

```powershell
python scripts/filter_pseudo_annotations.py --input-csv outputs/pseudo_labels/penn/train_tal.csv --output-csv outputs/pseudo_labels/penn/train_tal_clean.csv
python scripts/filter_pseudo_annotations.py --input-csv outputs/pseudo_labels/penn/val_tal.csv --output-csv outputs/pseudo_labels/penn/val_tal_clean.csv
python scripts/filter_pseudo_annotations.py --input-csv outputs/pseudo_labels/penn/test_tal.csv --output-csv outputs/pseudo_labels/penn/test_tal_clean.csv
```

### 5) Train B-line full model

```powershell
python scripts/train_b_line.py --config configs/penn_b_line.yaml --backend full
```

### 6) Run split inference + evaluate

```powershell
python scripts/infer_split_b_line.py --config configs/penn_b_line.yaml --split-csv data/penn_action/rebuild_dataset/ucf_binary/penn/test.csv --out-jsonl outputs/predictions/penn_b_test.jsonl
python scripts/evaluate_penn_pipeline.py --pred-input outputs/predictions/penn_b_test.jsonl --gt-csv outputs/pseudo_labels/penn/test_tal_clean.csv --out-json outputs/metrics/penn_eval.json --out-md outputs/metrics/penn_eval.md
```

### 7) Optional OpenTAD bridge export (interface only)

```powershell
python scripts/export_opentad_json.py --annotation-csv outputs/pseudo_labels/penn/train_tal_clean.csv --output-json outputs/pseudo_labels/penn/opentad_train.json --subset train
```

## Penn Action OpenTAD (Manual-Refined Only)

OpenTAD is used as B-line training backend, not as auto-labeling tool.

### 1) Sample 15% manual-refine subset from official train split

```powershell
python scripts/sample_manual_subset.py --input-split-csv data/penn_action/rebuild_dataset/ucf_binary/penn/train.csv --manual-ratio 0.15 --seed 42 --out-manual-csv data/penn_action/manual_refined/manual_subset_videos.csv --out-rest-csv data/penn_action/manual_refined/train_rest_videos.csv
```

### 2) Build manual template from pseudo labels

```powershell
python scripts/build_manual_refined_template.py --manual-videos-csv data/penn_action/manual_refined/manual_subset_videos.csv --pseudo-annotation-csv outputs/pseudo_labels/penn/train_tal_clean.csv --output-template-csv data/penn_action/manual_refined/train_manual_refined_template.csv
```

Annotate `train_manual_refined_template.csv` and save as:
- `data/penn_action/manual_refined/train_manual_refined.csv`
- `data/penn_action/manual_refined/val_manual_refined.csv`
- `data/penn_action/manual_refined/test_manual_refined.csv`

### 3) Validate manual refined annotations

```powershell
python scripts/validate_manual_annotations.py --input-csv data/penn_action/manual_refined/train_manual_refined.csv --strict-source-type --expected-source-type manual_refined
python scripts/validate_manual_annotations.py --input-csv data/penn_action/manual_refined/val_manual_refined.csv --strict-source-type --expected-source-type manual_refined
```

### 4) Export OpenTAD JSON (manual only)

```powershell
python scripts/export_opentad_json.py --annotation-csv data/penn_action/manual_refined/train_manual_refined.csv --keep-source-types manual_refined --subset train --version-name penn_action_manual_v1 --output-json data/penn_action/opentad/train_manual.json
python scripts/export_opentad_json.py --annotation-csv data/penn_action/manual_refined/val_manual_refined.csv --keep-source-types manual_refined --subset val --version-name penn_action_manual_v1 --output-json data/penn_action/opentad/val_manual.json
```

### 5) Launch OpenTAD manual training

```powershell
python scripts/train_b_line.py --config configs/penn_opentad_manual.yaml --backend opentad_manual
```

Run metadata is written to `opentad_work_dir/run_meta.json` for reproducibility.

## GPU Acceleration Notes

- MediaPipe extraction:
  - Use `--use-gpu-delegate` and `--model-path models/pose_landmarker_lite.task`
  - Example:
    - `python scripts/extract_pose_from_frames.py ... --use-gpu-delegate --model-path models/pose_landmarker_lite.task`
- A-line:
  - `configs/a_line.yaml` and `configs/penn_a_line.yaml` support:
    - `use_gpu: true`
    - `gpu_id: 0`
  - Effective only when XGBoost backend is available.
- B-line full:
  - `configs/b_line.yaml` and `configs/penn_b_line.yaml` support:
    - `use_data_parallel: true`
    - `gpu_ids: "0,1,2,3"`
  - Uses PyTorch `DataParallel` for multi-GPU.
- OpenTAD:
  - Multi-GPU distributed training should be launched by OpenTAD's distributed scripts/`torchrun`.
