# Deceptive RL on Attack Graphs

This project trains a defender agent to place decoys on a small attack graph to detect an attacker before they reach a goal node. It includes a custom Gymnasium environment, reinforcement learning training with PPO, and baseline defender strategies for comparison.

## Overview

### Environment (AttackGraphEnv)
- The environment generates an attack graph with `n_nodes` and a designated **goal node**.
- The **attacker** starts at some node and moves each step according to a chosen attacker policy.
- The **defender** places decoys (honeypots) on nodes to increase the chance of detection.
- An episode ends when:
  - the attacker is **detected on a decoy** (defender success), or
  - the attacker **reaches the goal** (attacker success), or
  - the episode hits the time horizon.

### Attacker policies
- `shortest`: moves along a shortest path toward the goal.
- `random`: chooses a random neighbor at each step.

### Defender actions
At each step, the defender selects a node index to place a decoy. If the attacker steps onto a decoy, the episode ends with a detection event.

## Algorithms Implemented

### 1) PPO Defender (Stable-Baselines3)
**Proximal Policy Optimization (PPO)** trains a neural policy to choose decoy placements that maximize long-term reward.  
PPO is an on-policy actor-critic algorithm that:
- collects rollouts from the current policy,
- optimizes a clipped surrogate objective to keep policy updates stable,
- learns a value function baseline to reduce variance.

This project supports:
- PPO training against a fixed attacker type (e.g., `shortest`)
- PPO training with a **mixture attacker wrapper** that randomizes the attacker type per episode

### 2) Baseline Defenders
Two simple baselines are provided:

- **Random baseline**: places decoys uniformly at random.
- **Degree-greedy baseline**: places decoys on high-degree nodes (nodes with many edges), a common heuristic to cover high-traffic parts of the graph.

## Results (high level)

All results below are from evaluations with **1000 episodes**.

### Baselines
- **Random baseline**
  - Detected on decoy: **34.70%**
  - Attacker reached goal: **65.30%**
  - Mean return: **0.7753**

- **Degree-greedy baseline**
  - Detected on decoy: **66.50%**
  - Attacker reached goal: **33.50%**
  - Mean return: **2.6663**

### PPO Defender (best highlights)
- PPO trained/evaluated on **shortest-path attacker**
  - Detected on decoy: **78.80%**
  - Attacker reached goal: **21.20%**
  - Mean return: **3.5301**

- PPO evaluated on **random attacker**
  - Detected on decoy: **94.00%**
  - Attacker reached goal: **5.90%**
  - Mean return: **4.4636**

**Best part:** PPO achieves the highest decoy-detection rate overall, and is especially strong against the random attacker (**94–95% detection**), clearly outperforming both baselines.

### PPO trained with mixture attacker (50% shortest, 50% random)
- Evaluated on `shortest`
  - Detected on decoy: **79.30%**
  - Attacker reached goal: **20.70%**
  - Mean return: **3.5423**

- Evaluated on `random`
  - Detected on decoy: **95.10%**
  - Attacker reached goal: **4.90%**
  - Mean return: **4.4905**

This mixture-trained model maintains strong performance on `shortest` while slightly improving detection on `random`.

## Project layout

- `src/envs/attack_graph_env.py` — Gymnasium environment (attack graph, rewards, termination)
- `src/envs/wrappers.py` — attacker-mixture wrapper (randomizes attacker type per episode)
- `src/train/train_ppo.py` — PPO training (single attacker setting)
- `src/train/train_ppo_mixture.py` — PPO training with attacker mixture
- `src/eval/smoke_test.py` — quick environment sanity test
- `src/eval/evaluate_policy.py` — evaluate a saved PPO model
- `src/eval/evaluate_baselines.py` — evaluate baseline defenders
- `src/eval/evaluate_cross_attacker.py` — evaluate one PPO model against multiple attacker policies

## Setup (Windows)

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
