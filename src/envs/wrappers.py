from __future__ import annotations
import gymnasium as gym
import numpy as np

class AttackerMixtureWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, p_random: float = 0.5, seed: int = 0):
        super().__init__(env)
        assert 0.0 <= p_random <= 1.0
        self.p_random = p_random
        self.rng = np.random.default_rng(seed)

    def reset(self, **kwargs):
        policy = "random" if self.rng.random() < self.p_random else "shortest"
        options = kwargs.get("options") or {}
        options = dict(options)
        options["attacker_policy"] = policy
        kwargs["options"] = options
        return self.env.reset(**kwargs)
