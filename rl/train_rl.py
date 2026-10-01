"""
rl/train_rl.py — Train Deep Q-Network (DQN) Traffic Controller
==============================================================
Trains the DQNAgent across urban scenarios, tracks episode returns and
training loss, updates target networks, and saves model checkpoints and plots.

Usage
-----
    python -m rl.train_rl
    python -m rl.train_rl --episodes 150 --max-steps 40 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from simulation.scenario_generator import ScenarioType
from rl.environment import CityTrafficEnv
from rl.agent import DQNAgent
from rl.replay_buffer import ReplayBuffer

OUTPUT_DIR = _ROOT / "outputs" / "rl"
MODEL_DIR = _ROOT / "models"


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _save_training_plot(
    rewards: List[float],
    avg_rewards: List[float],
    output_path: Path,
) -> None:
    """Plot training returns and rolling average."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot skipped] matplotlib not installed.")
        return

    fig, ax = plt.subplots(figsize=(10, 4.5))

    episodes = range(1, len(rewards) + 1)
    ax.plot(episodes, rewards, color="#90CAF9", alpha=0.6, label="Episode Return")
    ax.plot(episodes, avg_rewards, color="#1565C0", linewidth=2.0, label="10-Episode Moving Avg")

    ax.set_title("Unit IV — DQN Traffic Controller Training Progress", fontsize=12, fontweight="bold")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training plot saved to: {output_path}")


def train_dqn(
    episodes: int = 150,
    max_steps: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    epsilon_decay: float = 0.97,
    target_update_steps: int = 50,
    seed: int = 42,
    save_dir: Path = MODEL_DIR,
    output_dir: Path = OUTPUT_DIR,
    verbose: bool = True,
) -> Dict:
    """
    Main DQN training loop.
    """
    _set_seed(seed)
    save_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        ScenarioType.NORMAL,
        ScenarioType.RUSH_HOUR,
        ScenarioType.ACCIDENT,
        ScenarioType.HEAVY_RAIN,
        ScenarioType.ACCIDENT_RAIN,
        ScenarioType.ROAD_BLOCKAGE,
        ScenarioType.EMERGENCY,
    ]

    env = CityTrafficEnv(scenario=ScenarioType.NORMAL, max_steps=max_steps, seed=seed)
    agent = DQNAgent(lr=lr, gamma=gamma)
    replay_buffer = ReplayBuffer(capacity=10000)

    epsilon = epsilon_start
    total_steps = 0
    episode_rewards: List[float] = []
    moving_avg_rewards: List[float] = []
    losses: List[float] = []

    best_avg_reward = float("-inf")
    model_save_path = save_dir / "rl_agent.pt"

    t0 = time.time()
    if verbose:
        print("\n" + "=" * 65)
        print("  CityTwin AI — Training DQN Traffic Controller (Stage 4)")
        print("=" * 65)
        print(f"  Episodes: {episodes} | Steps/Ep: {max_steps} | Batch: {batch_size} | Gamma: {gamma}")
        print("  " + "-" * 60)
        print(f"  {'Ep':>4}  {'Scenario':>14}  {'Ep Reward':>10}  {'10-Ep Avg':>10}  {'Epsilon':>8}")
        print("  " + "-" * 60)

    for ep in range(1, episodes + 1):
        # Rotate scenarios to expose agent to diverse traffic conditions
        current_scenario = scenarios[(ep - 1) % len(scenarios)]
        state, _ = env.reset(options={"scenario": current_scenario}, seed=seed + ep)

        ep_reward = 0.0

        for step in range(max_steps):
            total_steps += 1

            # Select action
            action = agent.select_action(state, epsilon=epsilon)

            # Environment step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Push transition
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward

            # Minibatch gradient step
            loss = agent.update(replay_buffer, batch_size=batch_size)
            if loss is not None:
                losses.append(loss)

            # Target network update
            if total_steps % target_update_steps == 0:
                agent.update_target_network()

            if done:
                break

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        episode_rewards.append(ep_reward)
        avg_10 = float(np.mean(episode_rewards[-10:]))
        moving_avg_rewards.append(avg_10)

        # Checkpoint saving
        if avg_10 > best_avg_reward and ep >= 10:
            best_avg_reward = avg_10
            agent.save(model_save_path)

        if verbose and (ep % 10 == 0 or ep == 1):
            print(f"  {ep:>4}  {current_scenario.name:>14}  {ep_reward:>10.2f}  {avg_10:>10.2f}  {epsilon:>8.3f}")

    training_time = time.time() - t0

    # Ensure final model is saved
    if not model_save_path.exists():
        agent.save(model_save_path)

    if verbose:
        print("  " + "-" * 60)
        print(f"  Training completed in {training_time:.1f}s. Final 10-Ep Avg: {moving_avg_rewards[-1]:.2f}")
        print(f"  Model saved to: {model_save_path}")

    # Save metrics JSON
    metrics = {
        "episodes": episodes,
        "max_steps": max_steps,
        "total_steps": total_steps,
        "training_time_seconds": round(training_time, 2),
        "final_avg_reward": round(moving_avg_rewards[-1], 2),
        "best_avg_reward": round(best_avg_reward if best_avg_reward != float("-inf") else moving_avg_rewards[-1], 2),
        "episode_rewards": [round(r, 2) for r in episode_rewards],
        "moving_avg_rewards": [round(r, 2) for r in moving_avg_rewards],
    }
    metrics_path = output_dir / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save training curve
    plot_path = output_dir / "training_curve.png"
    _save_training_plot(episode_rewards, moving_avg_rewards, plot_path)

    return {
        "model_path": str(model_save_path),
        "metrics": metrics,
        "training_time": training_time,
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CityTwin AI DQN Traffic Controller")
    p.add_argument("--episodes", type=int, default=150)
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train_dqn(
        episodes=args.episodes,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        seed=args.seed,
        verbose=not args.quiet,
    )
