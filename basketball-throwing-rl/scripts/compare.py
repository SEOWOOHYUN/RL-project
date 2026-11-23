# 두 결과 비교

# python scripts/compare.py --results [RESULT/PATH/ONE] [RESULT/PATH/TWo]
# python scripts/compare.py --results results/ppo_seed42.csv results/sac_seed42.csv

import argparse,csv
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,default=Path('results/comparison.csv'));a=p.parse_args()
summary=[]
for path in a.results:
    with open(path,newline='') as f: rows=list(csv.DictReader(f))
    distances=np.array([float(r['best_distance']) for r in rows]);distances=distances[np.isfinite(distances)]
    summary.append(dict(model=path.stem,episodes=len(rows),success_rate=np.mean([int(r['success']) for r in rows]),mean_best_distance=np.mean(distances) if len(distances) else float('nan'),mean_episode_reward=np.mean([float(r['episode_reward']) for r in rows])))
a.out.parent.mkdir(parents=True,exist_ok=True)
with open(a.out,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
for row in summary: print(row)
print('Saved:',a.out)