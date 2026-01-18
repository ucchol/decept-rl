from __future__ import annotations

import os
from dataclasses import asdict

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

from src.envs.attack_graph_env import AttackGraphEnv, EnvConfig
from src.envs.wrappers import AttackerMixtureWrapper


def make_env_fn(cfg: EnvConfig, p_random: float, seed: int):
    def _thunk():
        env = AttackGraphEnv(cfg)
        env = AttackerMixtureWrapper(env, p_random=p_random, seed=seed)
        return env
    return _thunk


def main():
    print(">>> Starting PPO mixture training...")

    cfg = EnvConfig(
        n_nodes=12,
        horizon=30,
        decoy_cost=0.2,
        compromise_penalty=1.0,
        detect_reward=5.0,
        step_alive_reward=0.05,
        seed=42,
        attacker_policy="shortest",
    )

    p_random = 0.5
    run_dir = os.path.join("runs", "ppo_mixture_50_50")
    os.makedirs(run_dir, exist_ok=True)
    print(">>> Run dir:", run_dir)

    train_env = make_vec_env(
        make_env_fn(cfg, p_random=p_random, seed=cfg.seed),
        n_envs=8,
        seed=cfg.seed,
        monitor_dir=os.path.join(run_dir, "monitor"),
    )

    eval_short = Monitor(AttackGraphEnv(EnvConfig(**{**asdict(cfg), "attacker_policy": "shortest"})))
    eval_rand  = Monitor(AttackGraphEnv(EnvConfig(**{**asdict(cfg), "attacker_policy": "random"})))

    cb = CallbackList([
        EvalCallback(
            eval_short,
            best_model_save_path=os.path.join(run_dir, "best_shortest"),
            log_path=os.path.join(run_dir, "logs_shortest"),
            eval_freq=5_000,
            n_eval_episodes=200,
            deterministic=True,
        ),
        EvalCallback(
            eval_rand,
            best_model_save_path=os.path.join(run_dir, "best_random"),
            log_path=os.path.join(run_dir, "logs_random"),
            eval_freq=5_000,
            n_eval_episodes=200,
            deterministic=True,
        ),
    ])

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

    model.learn(total_timesteps=150_000, callback=cb)

    model_path = os.path.join(run_dir, "final_model.zip")
    model.save(model_path)

    with open(os.path.join(run_dir, "config.txt"), "w", encoding="utf-8") as f:
        f.write(str(asdict(cfg)) + f"\nmixture_p_random={p_random}\n")

    print(">>> Saved model to:", model_path)


if __name__ == "__main__":
    main()
