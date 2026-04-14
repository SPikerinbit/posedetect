from pathlib import Path
import yaml, numpy as np, sys
sys.path.append(str(Path('.').resolve()))
from src.a_line.pipeline import ALinePipeline

cfg=yaml.safe_load(Path('configs/penn_a_line_calibrated.yaml').read_text())
pipe=ALinePipeline(
    window_size=int(cfg.get('window_size',48)),
    stride=int(cfg.get('stride',8)),
    background_quantile=float(cfg.get('background_quantile',0.2)),
    median_kernel=int(cfg.get('median_kernel',9)),
    min_segment_frames=int(cfg.get('min_segment_frames',10)),
    max_gap_frames=int(cfg.get('max_gap_frames',6)),
    peak_min_distance_sec=float(cfg.get('peak_min_distance_sec',0.35)),
    peak_prominence_ratio=float(cfg.get('peak_prominence_ratio',0.15)),
    seed=int(cfg.get('seed',42)),
    use_gpu=bool(cfg.get('use_gpu',False)),
    gpu_id=int(cfg.get('gpu_id',0)),
)
split_root=Path(cfg['split_root'])/'ucf_binary'/'penn'
pose_root=Path(cfg['pose_root_ucf'])
xtr,ytr=pipe._build_split_xy(split_root/'train.csv',pose_root,pose_root)
xv,yv=pipe._build_split_xy(split_root/'val.csv',pose_root,pose_root)
print('train samples',xtr.shape, 'label_counts', {int(k):int(v) for k,v in zip(*np.unique(ytr,return_counts=True))}, flush=True)
print('val samples',xv.shape, 'label_counts', {int(k):int(v) for k,v in zip(*np.unique(yv,return_counts=True))}, flush=True)

pipe=ALinePipeline.load(Path(cfg['model_out']))
ptr=pipe.model.predict(xtr)
pv=pipe.model.predict(xv)
if getattr(ptr,'ndim',1)==2: ptr=np.argmax(ptr,axis=1)
if getattr(pv,'ndim',1)==2: pv=np.argmax(pv,axis=1)
print('pred train counts', {int(k):int(v) for k,v in zip(*np.unique(ptr,return_counts=True))}, flush=True)
print('pred val counts', {int(k):int(v) for k,v in zip(*np.unique(pv,return_counts=True))}, flush=True)
