import pandas as pd, numpy as np
from pathlib import Path
import sys, yaml
sys.path.append(str(Path('.').resolve()))
from src.a_line.pipeline import LABEL_TO_ID
from src.features.window_features import build_video_windows

cfg=yaml.safe_load(Path('configs/penn_a_line_calibrated.yaml').read_text())
pose_root=Path(cfg['pose_root_ucf'])
df=pd.read_csv(Path(cfg['split_root'])/'ucf_binary'/'penn'/'train.csv')

stats=[]
for _,row in df.iterrows():
    lab=str(row['label_name'])
    aid=LABEL_TO_ID.get(lab,0)
    p=pose_root/Path(str(row['pose_path']))
    d=np.load(p,allow_pickle=True)
    k=d['keypoints'].astype(np.float32)
    fps=float(d['fps'][0]) if 'fps' in d else 25.0
    b=build_video_windows(k,fps,window_size=int(cfg['window_size']),stride=int(cfg['stride']),action_label=aid,background_quantile=float(cfg['background_quantile']))
    if b.labels.size==0:
        continue
    stats.append((lab,int((b.labels==aid).sum()),int((b.labels==0).sum()),int(b.labels.size)))

out=pd.DataFrame(stats,columns=['label','n_action','n_bg','n_total'])
print(out.groupby('label')[['n_action','n_bg','n_total']].sum())
print('videos_with_any_action_by_label')
print(out.groupby('label').apply(lambda x: int((x['n_action']>0).sum())))
