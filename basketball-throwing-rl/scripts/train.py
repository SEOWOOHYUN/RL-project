# 학습 위한 코드

# python ./scripts/train.py --algo [ppo,sac] --task [hoop,target] --timesteps [1_000_000]

# python scripts/train.py --algo ppo --task hoop --timesteps 300_000 --seed 42
# python scripts/train.py --algo sac --task hoop --timesteps 300_000 --seed 42

import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from envs import BasketballThrowEnv, BasketballHoopEnv

class MetricsCallback(BaseCallback):
    def __init__(self, path):
        super().__init__(); self.path=path; self.records=[]
    def _on_step(self):
        infos=self.locals.get('infos', [])
        for info in infos:
            if 'episode' in info:
                self.records.append({'timesteps':self.num_timesteps, 'reward':float(info['episode']['r']), 'length':int(info['episode']['l']), 'success':bool(info.get('is_success',False)), 'best_distance':float(info.get('best_distance',float('nan')))})
        return True
    def _on_training_end(self):
        import csv
        if self.records:
            with open(self.path,'w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(self.records[0]));writer.writeheader();writer.writerows(self.records)

p=argparse.ArgumentParser()
p.add_argument('--algo',choices=['ppo','sac'],required=True)
p.add_argument("--task", choices=["target", "hoop"], default="hoop")
p.add_argument('--timesteps',type=int,default=300_000)
p.add_argument('--seed',type=int,default=42)
p.add_argument('--out',type=Path,default=ROOT/'runs')
p.add_argument('--eval-freq',type=int,default=10_000)
p.add_argument('--eval-episodes',type=int,default=20)
a=p.parse_args()
run=a.out/(f'{a.algo}' if a.task == 'target' else f'hoop_{a.algo}_seed{a.seed}');run.mkdir(parents=True,exist_ok=True)
env_cls = BasketballHoopEnv if a.task == "hoop" else BasketballThrowEnv
train_env=Monitor(env_cls());eval_env=Monitor(env_cls())
train_env.reset(seed=a.seed);eval_env.reset(seed=10_000+a.seed)
common=dict(policy='MlpPolicy',env=train_env,learning_rate=3e-4,gamma=0.99,seed=a.seed,verbose=1,tensorboard_log=str(run/'tb'),device='cpu')
if a.algo=='ppo':
    model=PPO(n_steps=1024,batch_size=256,n_epochs=10,gae_lambda=0.95,clip_range=0.2,policy_kwargs=dict(net_arch=[128,128]),**common)
else:
    model=SAC(buffer_size=100_000,learning_starts=5_000,batch_size=256,train_freq=1,gradient_steps=1,tau=0.005,ent_coef='auto',policy_kwargs=dict(net_arch=[128,128]),**common)
callbacks=[EvalCallback(eval_env,best_model_save_path=str(run),log_path=str(run/'eval'),eval_freq=a.eval_freq,n_eval_episodes=a.eval_episodes,deterministic=True,verbose=1),CheckpointCallback(save_freq=50_000,save_path=str(run/'checkpoints')),MetricsCallback(run/'train_episodes.csv')]
(run/'config.json').write_text(json.dumps(vars(a),indent=2,default=str))
try:
    model.learn(total_timesteps=a.timesteps,callback=callbacks,progress_bar=True)
finally:
    model.save(str(run/'final_model'));train_env.close();eval_env.close()
print('Saved:',run/'final_model.zip')
