from __future__ import annotations

from pathlib import Path
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from src.envs.attack_graph_env import AttackGraphEnv, EnvConfig


def sample_attacker(rng: np.random.Generator) -> str:
    # Hard-focused mixture:
    # shortest = hard, stealthy = hard (decoy-avoid), random = easy
    return rng.choice(["shortest", "stealthy", "random"], p=[0.25, 0.65, 0.10]).item()


class MixtureEnv(gym.Env):
    """
    Gymnasium Env that re-creates the underlying AttackGraphEnv each episode
    with a sampled attacker policy.
    """
    metadata = {"render_modes": []}

    def __init__(self, base_cfg: EnvConfig, seed: int = 123):
        super().__init__()
        self.base_cfg = base_cfg
        self.rng = np.random.default_rng(seed)

        self.env: AttackGraphEnv | None = None
        self._last_attacker: str | None = None

        # Create once to expose spaces
        self._make_env("shortest")
        assert self.env is not None
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def _make_env(self, attacker_policy: str):
        cfg = EnvConfig(
            n_nodes=self.base_cfg.n_nodes,
            horizon=self.base_cfg.horizon,
            decoy_cost=self.base_cfg.decoy_cost,
            compromise_penalty=self.base_cfg.compromise_penalty,
            detect_reward=self.base_cfg.detect_reward,
            step_alive_reward=self.base_cfg.step_alive_reward,
            seed=int(self.rng.integers(0, 1_000_000)),
            attacker_policy=attacker_policy,
        )
        self.env = AttackGraphEnv(cfg)
        self._last_attacker = attacker_policy

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        attacker = sample_attacker(self.rng)
        self._make_env(attacker)

        obs, info = self.env.reset(seed=seed, options=options)
        info = dict(info)
        info["attacker_policy"] = attacker
        return obs, info

    def step(self, action):
        assert self.env is not None
        return self.env.step(action)

    def render(self):
        if self.env is not None and hasattr(self.env, "render"):
            return self.env.render()
        return None

    def close(self):
        if self.env is not None and hasattr(self.env, "close"):
            self.env.close()


def main():
    run_dir = Path("runs") / "ppo_mixture_hard"
    tb_dir = run_dir / "tb"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(">>> Starting PPO hard-mixture training...")
    print(f">>> Run dir: {run_dir}")

    base_cfg = EnvConfig(n_nodes=12, horizon=30, seed=0)

    train_env = Monitor(MixtureEnv(base_cfg, seed=123))

    eval_env_shortest = Monitor(AttackGraphEnv(EnvConfig(n_nodes=12, horizon=30, seed=999, attacker_policy="shortest")))
    eval_env_stealthy  = Monitor(AttackGraphEnv(EnvConfig(n_nodes=12, horizon=30, seed=999, attacker_policy="stealthy")))
    eval_env_random    = Monitor(AttackGraphEnv(EnvConfig(n_nodes=12, horizon=30, seed=999, attacker_policy="random")))

    # Evaluate on hardest ones
    eval_cb_shortest = EvalCallback(
        eval_env_shortest,
        best_model_save_path=str(run_dir / "best_shortest"),
        log_path=str(run_dir / "eval_shortest"),
        eval_freq=40_000,
        n_eval_episodes=200,
        deterministic=True,
        verbose=1,
    )
    eval_cb_stealthy = EvalCallback(
        eval_env_stealthy,
        best_model_save_path=str(run_dir / "best_stealthy"),
        log_path=str(run_dir / "eval_stealthy"),
        eval_freq=40_000,
        n_eval_episodes=200,
        deterministic=True,
        verbose=1,
    )

    callbacks = CallbackList([eval_cb_shortest, eval_cb_stealthy])

    # PPO: longer + more exploration + bigger net
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=8192,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.02,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=1,
        tensorboard_log=str(tb_dir),
    )

    total_timesteps = 500_000
    model.learn(total_timesteps=total_timesteps, callback=callbacks)

    out_path = run_dir / "final_model.zip"
    model.save(str(out_path))
    print(f">>> Saved model to: {out_path}")

    # quick final eval on all three attackers
    from stable_baselines3.common.evaluation import evaluate_policy
    for name, env in [("shortest", eval_env_shortest), ("stealthy", eval_env_stealthy), ("random", eval_env_random)]:
        mean_r, std_r = evaluate_policy(model, env, n_eval_episodes=300, deterministic=True)
        print(f"[Final eval] attacker={name:8s} mean={mean_r:.3f} std={std_r:.3f}")


if __name__ == "__main__":
    main()

