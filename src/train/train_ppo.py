from __future__ import annotations

import os
from dataclasses import asdict

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

from src.envs.attack_graph_env import AttackGraphEnv, EnvConfig


def make_env_fn(cfg: EnvConfig):
    def _thunk():
        return AttackGraphEnv(cfg)
    return _thunk


def main():
    # Small, fast config for first training run
    cfg = EnvConfig(
        n_nodes=12,
        horizon=30,
        decoy_cost=0.2,
        compromise_penalty=1.0,
        detect_reward=5.0,
        step_alive_reward=0.05,
        seed=42,
    )

    run_dir = os.path.join("runs", "ppo_mvp")
    os.makedirs(run_dir, exist_ok=True)

    # Training env: vectorized copies for speed/stability
    train_env = make_vec_env(make_env_fn(cfg), n_envs=8, seed=cfg.seed)

    # Eval env: single env, deterministic eval
    eval_env = AttackGraphEnv(cfg)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(run_dir, "best"),
        log_path=os.path.join(run_dir, "logs"),
        eval_freq=5_000,
        n_eval_episodes=50,
        deterministic=True,
        render=False,
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log=os.path.join(run_dir, "tb"),
        n_steps=512,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
    )

    total_timesteps = 100_000  # quick first run (a few minutes)
    model.learn(total_timesteps=total_timesteps, callback=eval_cb)

    model_path = os.path.join(run_dir, "final_model.zip")
    model.save(model_path)

    with open(os.path.join(run_dir, "config.txt"), "w", encoding="utf-8") as f:
        f.write(str(asdict(cfg)))

    print("Saved model to:", model_path)


if __name__ == "__main__":
    main()