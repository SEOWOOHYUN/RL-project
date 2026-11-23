# 학습된 모델로 평가 수행 

# python scripts/evaluate.py --algo [ppo,sac] --model [path_to_parameter] --task [hoop,target] --episodes 100 --seed 10000 --out [path_to_output]

# python scripts/evaluate.py --algo ppo --model runs/hoop_ppo/best_model.zip --task hoop --episodes 100 --seed 10000 --out results/hoop_ppo/hoop_ppo.csv
# python scripts/evaluate.py --algo sac --model runs/hoop_sac/best_model.zip --task hoop --episodes 100 --seed 10000 --out results/hoop_sac/hoop_sac.csv

import argparse, csv, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from stable_baselines3 import PPO,SAC
from envs import BasketballThrowEnv, BasketballHoopEnv

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--algo',choices=['ppo','sac'],required=True)
    p.add_argument("--task", choices=["target", "hoop"], default="hoop")
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--episodes',type=int,default=100)
    p.add_argument('--seed',type=int,default=10000)
    p.add_argument('--out',type=Path,default=None)
    a=p.parse_args()
    model=(PPO if a.algo=='ppo' else SAC).load(str(a.model),device='cpu')
    env=(BasketballHoopEnv if a.task == "hoop" else BasketballThrowEnv)();rows=[]
    try:
        for ep in range(a.episodes):
            obs,_=env.reset(seed=a.seed+ep);ret=0.;steps=0
            while True:
                action,_=model.predict(obs,deterministic=True)
                obs,reward,terminated,truncated,info=env.step(action)
                ret+=float(reward);steps+=1
                if terminated or truncated:break
            rows.append(dict(episode=ep+1,seed=a.seed+ep,success=int(info['is_success']),best_distance=float(info['best_distance']),episode_reward=ret,episode_length=steps,hoop_crossed=int(info.get('hoop_crossed',False)),min_hoop_error=float(info.get('min_hoop_error',float('inf')))))
    finally:env.close()
    valid=[r['best_distance'] for r in rows if np.isfinite(r['best_distance'])]
    summary=dict(task=a.task,algo=a.algo,model=str(a.model),episodes=len(rows),success_rate=float(np.mean([r['success'] for r in rows])),mean_best_distance=float(np.mean(valid)) if valid else None,mean_episode_reward=float(np.mean([r['episode_reward'] for r in rows])),mean_episode_length=float(np.mean([r['episode_length'] for r in rows])))
    out=a.out or ROOT/'results'/f'{a.task}_{a.algo}_evaluation.csv';out.parent.mkdir(parents=True,exist_ok=True)
    with open(out,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    out.with_suffix('.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2));print('Saved:',out)