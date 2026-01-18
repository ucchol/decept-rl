from __future__ import annotations
import numpy as np

def defender_random_action(env, rng: np.random.Generator) -> int:
    # same action space as env: 0..N
    return int(rng.integers(0, env.action_space.n))

def defender_degree_greedy(env) -> int:
    """
    Place a decoy on the highest-degree node that is not:
      - goal
      - already a decoy
    If none available -> noop.
    """
    best_node = None
    best_deg = -1

    for node in env.G.nodes():
        if env.is_goal[node] == 1:
            continue
        if env.is_decoy[node] == 1:
            continue
        deg = env.G.degree[node]
        if deg > best_deg:
            best_deg = deg
            best_node = node

    if best_node is None:
        return 0
    return int(best_node + 1)  # action encoding
