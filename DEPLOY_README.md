# Deploy README (Code-only repo + server-side dataset)

## 1) Goal
This repository is uploaded without local datasets/artifacts. On server, download high-quality dataset, prepare manifests/pose, then train.

## 2) What is included in GitHub
- src/
- scripts/
- configs/
- README.md
- requirements.txt
- environment.yml
- DEPLOY_README.md

Excluded by `.gitignore`: `dataset/`, `data/`, `outputs/`, `reports/`, local env/cache folders.

## 3) Server setup
```bash
# clone
git clone <YOUR_REPO_URL>
cd posedetect

# conda env (choose one)
conda env create -f environment.yml
conda activate poseact
# or install by pip
# pip install -r requirements.txt
```

## 4) Prepare high-quality dataset on server
Create this structure:

```text
posedetect/
  dataset/
    PushUps/*.avi|*.mp4
    JumpingJack/*.avi|*.mp4
    LLSP/
      push_up_data/
      jump_jack_data/
```

If you have timestamp annotations, store them as CSV (recommended):

```text
video_path,start_time,end_time,label_name,label_id
```

## 5) Build splits / extract pose
```bash
python scripts/make_group_splits.py --dataset-root dataset --out-dir data/splits_group --seed 42
python scripts/extract_pose.py --split-dir data/splits_group --pose-dir data/pose --model-complexity 1
```

## 6) Train A-line and export pseudo timestamps
```bash
python scripts/train_a_line.py --config configs/a_line.yaml
python scripts/export_pseudo_timestamps.py --config configs/a_line.yaml --split-variant group
```

Outputs:
- `outputs/checkpoints/a_line/a_line_model.joblib`
- `outputs/pseudo_labels/group/*_tal.csv`

## 7) Train B-line (minimal local backend)
```bash
python scripts/train_b_line.py --config configs/b_line.yaml --backend minimal
```

## 8) Train B-line with OpenTAD/ActionFormer/VideoMAE (formal)
Set `configs/b_line.yaml`:
- `backend: opentad`
- `opentad_train_cmd: python tools/train.py configs/your_actionformer_videomae.py`

Then run:
```bash
python scripts/train_b_line.py --config configs/b_line.yaml --backend opentad
```

## 9) Unified inference output
```bash
python scripts/infer_segments.py --line a --config configs/a_line.yaml \
  --pose-path data/pose/PushUps/v_PushUps_g01_c01.npz \
  --video-path dataset/PushUps/v_PushUps_g01_c01.avi \
  --out-json outputs/predictions/a_line_example.json
```

Schema contains:
- `segments`
- `instances`
- `summary`

## 10) Notes
- Current local manifests in `data/rebuild_dataset` are not uploaded by design.
- On server, regenerate splits/manifests from your high-quality dataset.
- For multi-GPU DDP (e.g., 4x5090), keep backend/config aligned with OpenTAD launcher.
