"""
rl/evaluate_rl.py — Comparative Policy Evaluation Suite
========================================================
Compares DQN, NoAction, and Heuristic policies under identical deterministic
seeds across all standard urban traffic scenarios.

Metrics Tracked
---------------
  - Cumulative Episode Reward
  - Average Traffic Density [0, 1]
  - Average Speed (km/h)
  - Average Queue Length (metres)
  - Average Emergency Vehicles
  - Active Intervention Count (Actions != 0)
  - Final Urban Disruption Risk Score (0–100)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np

from simulation.scenario_generator import ScenarioType
from rl.environment import CityTrafficEnv, TrafficAction
from rl.agent import DQNAgent, NoActionPolicy, HeuristicPolicy

OUTPUT_DIR = _ROOT / "outputs" / "rl"
MODEL_DIR = _ROOT / "models"


def evaluate_policy_on_scenario(
    env: CityTrafficEnv,
    policy: Any,
    scenario: ScenarioType,
    steps: int = 40,
    seed: int = 100,
) -> Dict[str, float]:
    """
    Run a single deterministic episode for a policy on a scenario.
    """
    state, info = env.reset(options={"scenario": scenario}, seed=seed)

    rewards: List[float] = []
    densities: List[float] = []
    speeds: List[float] = []
    queues: List[float] = []
    ev_counts: List[float] = []
    action_counts = 0

    for _ in range(steps):
        action = policy.select_action(state, epsilon=0.0)
        if action != TrafficAction.NO_ACTION:
            action_counts += 1

        next_state, reward, terminated, truncated, step_info = env.step(action)
        rewards.append(reward)
        densities.append(step_info["traffic_density"])
        speeds.append(step_info["average_speed"])
        queues.append(step_info["queue_length"])
        ev_counts.append(step_info["emergency_vehicle_count"])

        state = next_state
        if terminated or truncated:
            break

    # Final risk score from encoded state (feature index 16 = risk / 100.0)
    final_risk = float(state[16] * 100.0)

    return {
        "cumulative_reward": float(np.sum(rewards)),
        "avg_density": float(np.mean(densities)),
        "avg_speed": float(np.mean(speeds)),
        "avg_queue": float(np.mean(queues)),
        "avg_ev_count": float(np.mean(ev_counts)),
        "interventions": int(action_counts),
        "final_risk": final_risk,
    }


def run_comparative_evaluation(
    steps_per_episode: int = 40,
    seed: int = 100,
    model_path: Path = MODEL_DIR / "rl_agent.pt",
    output_dir: Path = OUTPUT_DIR,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run side-by-side evaluation across all policies and scenarios.
    """
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

    # Initialize policies
    policies: Dict[str, Any] = {
        "NO_ACTION": NoActionPolicy(),
        "HEURISTIC": HeuristicPolicy(),
    }

    if Path(model_path).exists():
        dqn = DQNAgent()
        dqn.load(model_path)
        policies["DQN"] = dqn
    else:
        if verbose:
            print(f"  [Notice] DQN model not found at {model_path}. Evaluating baselines only.")

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    aggregated: Dict[str, Dict[str, float]] = {}

    for pol_name in policies.keys():
        results[pol_name] = {}

    env = CityTrafficEnv(max_steps=steps_per_episode, seed=seed)

    for scenario in scenarios:
        for pol_name, policy in policies.items():
            metrics = evaluate_policy_on_scenario(
                env=env,
                policy=policy,
                scenario=scenario,
                steps=steps_per_episode,
                seed=seed,
            )
            results[pol_name][scenario.name] = metrics

    # Compute aggregated averages
    for pol_name in policies.keys():
        sc_metrics = results[pol_name]
        aggregated[pol_name] = {
            "mean_reward": float(np.mean([m["cumulative_reward"] for m in sc_metrics.values()])),
            "mean_density": float(np.mean([m["avg_density"] for m in sc_metrics.values()])),
            "mean_speed": float(np.mean([m["avg_speed"] for m in sc_metrics.values()])),
            "mean_queue": float(np.mean([m["avg_queue"] for m in sc_metrics.values()])),
            "mean_interventions": float(np.mean([m["interventions"] for m in sc_metrics.values()])),
            "mean_final_risk": float(np.mean([m["final_risk"] for m in sc_metrics.values()])),
        }

    # Print summary tables
    if verbose:
        print("\n" + "=" * 80)
        print("  CityTwin AI — Policy Comparative Evaluation (Stage 4)")
        print("=" * 80)

        for scenario in scenarios:
            print(f"\n  Scenario: {scenario.name}")
            print(f"  {'Policy':>12} | {'Reward':>8} | {'Density':>8} | {'Speed(km/h)':>11} | {'Queue(m)':>9} | {'Actions':>8} | {'Risk':>6}")
            print("  " + "-" * 76)
            for pol_name in policies.keys():
                m = results[pol_name][scenario.name]
                print(
                    f"  {pol_name:>12} | "
                    f"{m['cumulative_reward']:>8.2f} | "
                    f"{m['avg_density']:>8.3f} | "
                    f"{m['avg_speed']:>11.1f} | "
                    f"{m['avg_queue']:>9.1f} | "
                    f"{m['interventions']:>8d} | "
                    f"{m['final_risk']:>6.1f}"
                )

        print("\n" + "=" * 80)
        print("  AGGREGATED PERFORMANCE ACROSS ALL SCENARIOS")
        print("=" * 80)
        print(f"  {'Policy':>12} | {'Mean Reward':>12} | {'Mean Density':>12} | {'Mean Speed':>11} | {'Mean Queue':>11} | {'Mean Actions':>12}")
        print("  " + "-" * 82)
        for pol_name, agg in aggregated.items():
            print(
                f"  {pol_name:>12} | "
                f"{agg['mean_reward']:>12.2f} | "
                f"{agg['mean_density']:>12.3f} | "
                f"{agg['mean_speed']:>11.1f} | "
                f"{agg['mean_queue']:>11.1f} | "
                f"{agg['mean_interventions']:>12.1f}"
            )
        print("=" * 80 + "\n")

    # Save to JSON
    eval_payload = {
        "scenario_results": results,
        "aggregated_results": aggregated,
    }
    json_path = output_dir / "evaluation_results.json"
    with open(json_path, "w") as f:
        json.dump(eval_payload, f, indent=2)
    if verbose:
        print(f"  Results saved to: {json_path}")

    return eval_payload


if __name__ == "__main__":
    run_comparative_evaluation()
