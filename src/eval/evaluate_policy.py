from __future__ import annotations

import argparse
import numpy as np
from stable_baselines3 import PPO

from src.envs.attack_graph_env import AttackGraphEnv, EnvConfig


def run_eval(model, env, n_episodes: int = 300, deterministic: bool = True):
    outcomes = {"goal_reached": 0, "detected_on_decoy": 0, "horizon": 0, "other": 0}
    returns = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        total = 0.0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, r, term, trunc, info = env.step(int(action))
            total += float(r)
            done = term or trunc

        evt = info.get("event", "other")
        outcomes[evt] = outcomes.get(evt, 0) + 1
        returns.append(total)

    returns = np.asarray(returns, dtype=np.float64)
    return outcomes, float(returns.mean()), float(returns.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=500)
    args = ap.parse_args()

    env = AttackGraphEnv(EnvConfig(n_nodes=12, horizon=30, seed=123))
    model = PPO.load(args.model)

    outcomes, mean_ret, std_ret = run_eval(model, env, n_episodes=args.episodes)
    total = sum(outcomes.values())

    print("Episodes:", total)
    print("Mean return:", mean_ret)
    print("Std return:", std_ret)
    print("Outcome rates:")
    for k, v in outcomes.items():
        print(f"  {k:18s}: {v:4d} ({v/total:.2%})")


if __name__ == "__main__":
    main()


