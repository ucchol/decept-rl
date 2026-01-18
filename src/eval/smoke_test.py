from src.envs.attack_graph_env import AttackGraphEnv, EnvConfig

def main():
    env = AttackGraphEnv(EnvConfig(n_nodes=12, horizon=25, seed=42))
    obs, info = env.reset()
    print("Reset info:", info)
    total = 0.0

    for _ in range(10):
        action = env.action_space.sample()  # random defender
        obs, r, term, trunc, info = env.step(action)
        total += r
        env.render()
        print("  action:", action, "reward:", r, "info:", info)
        if term or trunc:
            break

    print("Total reward:", total)

if __name__ == "__main__":
    main()