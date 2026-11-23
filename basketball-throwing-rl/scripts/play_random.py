
# Random / Smooth Random Policy Visualization
#
# MUJOCO_GL=egl python scripts/play_random.py --episodes 5 --policy smooth --out results/random_policy.gif

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import imageio.v2 as imageio
from envs import BasketballThrowEnv, BasketballHoopEnv


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        choices=["target", "hoop"],
        default="hoop",
    )
    parser.add_argument(
        "--policy",
        choices=["uniform", "smooth"],
        default="smooth",
    )
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-std", type=float, default=0.12)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "random_throw.gif",
    )

    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be >= 1")

    if not 0 <= args.momentum < 1:
        parser.error("--momentum must be in [0, 1)")

    env_class = (
        BasketballHoopEnv
        if args.task == "hoop"
        else BasketballThrowEnv
    )

    env = env_class(render_mode="rgb_array")
    frames = []

    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep)
            rng = np.random.default_rng(args.seed + ep)

            # Start from the vertical reference pose.
            action = np.zeros(
                env.action_space.shape,
                dtype=np.float32,
            )

            frames.append(env.render())
            total_reward = 0.0
            step_count = 0

            while True:
                if args.policy == "uniform":
                    # Independent random action at every step.
                    action = rng.uniform(
                        -1.0,
                        1.0,
                        size=env.action_space.shape,
                    ).astype(np.float32)

                else:
                    # Temporally correlated random action.
                    noise = rng.normal(
                        loc=0.0,
                        scale=args.noise_std,
                        size=env.action_space.shape,
                    )

                    action = np.clip(
                        args.momentum * action + noise,
                        env.action_space.low,
                        env.action_space.high,
                    ).astype(np.float32)

                obs, reward, terminated, truncated, info = env.step(
                    action
                )

                total_reward += float(reward)
                step_count += 1

                frame = env.render()
                frames.append(frame)

                if terminated or truncated:
                    print(
                        f"Episode {ep + 1} | "
                        f"Task: {args.task} | "
                        f"Policy: {args.policy} | "
                        f"Steps: {step_count} | "
                        f"Reward: {total_reward:.3f} | "
                        f"Success: {info.get('is_success', False)} | "
                        f"Best distance: "
                        f"{info.get('best_distance', float('nan')):.3f} m"
                    )
                    break

        args.out.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        imageio.mimsave(
            args.out,
            frames,
            duration=env.dt,
            loop=0,
        )

        print(f"GIF saved: {args.out}")

    finally:
        env.close()


if __name__ == "__main__":
    main()
