from __future__ import annotations

import numpy as np
from src.envs.attack_graph_env import AttackGraphEnv, EnvConfig
from src.defenders.baselines import defender_random_action, defender_degree_greedy


def eval_defender(policy_name: str, n_episodes: int = 1000, seed: int = 123):
    env = AttackGraphEnv(EnvConfig(n_nodes=12, horizon=30, seed=seed))
    rng = np.random.default_rng(seed)

    outcomes = {"goal_reached": 0, "detected_on_decoy": 0, "horizon": 0, "other": 0}
    returns = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        total = 0.0
        done = False

        while not done:
            if policy_name == "random":
                action = defender_random_action(env, rng)
            elif policy_name == "degree":
                action = defender_degree_greedy(env)
            else:
                raise ValueError(f"Unknown policy_name: {policy_name}")

            obs, r, term, trunc, info = env.step(int(action))
            total += float(r)
            done = term or trunc

        evt = info.get("event", "other")
        outcomes[evt] = outcomes.get(evt, 0) + 1
        returns.append(total)

    returns = np.asarray(returns, dtype=np.float64)
    total_eps = int(sum(outcomes.values()))

    print(f"\n=== Baseline: {policy_name} ===")
    print("Episodes:", total_eps)
    print("Mean return:", float(returns.mean()))
    print("Std return:", float(returns.std()))
    print("Outcome rates:")
    for k, v in outcomes.items():
        print(f"  {k:18s}: {v:4d} ({v/total_eps:.2%})")


def main():
    print("Running baseline evaluation...")
    eval_defender("random", n_episodes=1000)
    eval_defender("degree", n_episodes=1000)


if __name__ == "__main__":
    main()
