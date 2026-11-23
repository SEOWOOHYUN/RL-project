# 학습된 모델 실행하여 gif로 저장


# MUJOCO_GL=egl python scripts/play.py --algo [ppo,sac] --model [path_to_parameter] --task [hoop,target] --episodes 5 --out [path_to_output]

# MUJOCO_GL=egl python scripts/play.py --algo ppo --model runs/hoop_ppo/best_model.zip --task hoop --episodes 5 --out results/hoop_ppo/hoop_ppo.gif
# MUJOCO_GL=egl python scripts/play.py --algo sac --model runs/hoop_sac/best_model.zip --task hoop --episodes 5 --out results/hoop_sac/hoop_sac.gif

import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import imageio.v2 as imageio
from stable_baselines3 import PPO, SAC
from envs import BasketballThrowEnv, BasketballHoopEnv

p = argparse.ArgumentParser()
p.add_argument("--algo", choices=["ppo", "sac"], default="ppo")
p.add_argument("--task", choices=["target", "hoop"], default="hoop")
p.add_argument("--model", type=Path, required=True)
p.add_argument("--out", type=Path, default=ROOT / "results" / "rollout.gif")
p.add_argument("--episodes", type=int, default=3)
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()
model = (PPO if args.algo == "ppo" else SAC).load(str(args.model), device="cpu")
env = (BasketballHoopEnv if args.task == "hoop" else BasketballThrowEnv)(render_mode="rgb_array")
frames = []
try:
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        frames.append(env.render())
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            frames.append(env.render())
            if terminated or truncated:
                print(f"episode={ep+1} success={info['is_success']} best_distance={info['best_distance']:.3f}")
                break
    args.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, frames, duration=env.dt, loop=0)
    print("Saved:", args.out)
finally:
    env.close()