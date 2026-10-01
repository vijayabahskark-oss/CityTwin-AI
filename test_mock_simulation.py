"""
test_mock_simulation.py
========================
Smoke-test and demonstration for CityTwin AI — Mock Simulation Layer.

Runs every scenario, validates each generated state, and prints a
per-scenario summary table.  Exits with code 0 on success, 1 on failure.

Usage
-----
    python test_mock_simulation.py
    python test_mock_simulation.py --seed 123 --steps 10
"""

from __future__ import annotations

import argparse
import sys
import traceback
from typing import List

# ---------------------------------------------------------------------------
# Adjust sys.path so the script can be run from the project root without
# installing the package (important for early-stage dev with no setup.py).
# ---------------------------------------------------------------------------
import os

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from simulation import MockCitySimulator, ScenarioGenerator, CityState, ScenarioType


# ---------------------------------------------------------------------------
# ANSI colours (gracefully disabled if the terminal does not support them)
# ---------------------------------------------------------------------------

def _supports_color() -> bool:
    # Disable color/ANSI on Windows terminals that use cp1252 encoding
    # to avoid UnicodeEncodeError from ANSI escape sequences.
    if sys.platform == "win32":
        enc = getattr(sys.stdout, "encoding", "") or ""
        if enc.lower().replace("-", "") in ("cp1252", "cp850", "cp437"):
            return False
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


_C = {
    "reset":  "\033[0m"  if _supports_color() else "",
    "bold":   "\033[1m"  if _supports_color() else "",
    "green":  "\033[92m" if _supports_color() else "",
    "yellow": "\033[93m" if _supports_color() else "",
    "red":    "\033[91m" if _supports_color() else "",
    "cyan":   "\033[96m" if _supports_color() else "",
    "grey":   "\033[90m" if _supports_color() else "",
}


def _ok(msg: str) -> str:
    return f"{_C['green']}[PASS]{_C['reset']} {msg}"


def _fail(msg: str) -> str:
    return f"{_C['red']}[FAIL]{_C['reset']} {msg}"


def _header(msg: str) -> str:
    line = "-" * 80
    return f"\n{_C['bold']}{_C['cyan']}{line}\n  {msg}\n{line}{_C['reset']}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scenario_stats(states: List[CityState]) -> dict:
    """Compute mean values across a list of states."""
    n = len(states)
    if n == 0:
        return {}
    return {
        "n":                    n,
        "avg_vehicles":         sum(s.vehicle_count for s in states) / n,
        "avg_speed_kmh":        sum(s.average_speed for s in states) / n,
        "avg_density":          sum(s.traffic_density for s in states) / n,
        "avg_queue_m":          sum(s.queue_length for s in states) / n,
        "avg_rain":             sum(s.rain_intensity for s in states) / n,
        "avg_fog":              sum(s.fog_intensity for s in states) / n,
        "accidents":            sum(1 for s in states if s.accident),
        "blockages":            sum(1 for s in states if s.road_blockage),
        "ev_any":               sum(s.emergency_vehicle_count for s in states),
    }


def _print_stats(stats: dict) -> None:
    print(f"    Steps        : {stats['n']}")
    print(f"    Avg vehicles : {stats['avg_vehicles']:.1f}")
    print(f"    Avg speed    : {stats['avg_speed_kmh']:.2f} km/h")
    print(f"    Avg density  : {stats['avg_density']:.3f}")
    print(f"    Avg queue    : {stats['avg_queue_m']:.1f} m")
    print(f"    Avg rain     : {stats['avg_rain']:.3f}")
    print(f"    Avg fog      : {stats['avg_fog']:.3f}")
    print(f"    Accident tks : {stats['accidents']}")
    print(f"    Blockage tks : {stats['blockages']}")
    print(f"    Total EVs    : {stats['ev_any']}")


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_reproducibility(seed: int = 42, steps: int = 5) -> bool:
    """Two simulators with the same seed must produce identical sequences."""
    print(_header("TEST: Reproducibility"))
    sim_a = MockCitySimulator(seed=seed)
    sim_b = MockCitySimulator(seed=seed)
    gen = ScenarioGenerator()

    overlay = gen.get_overlay(ScenarioType.RUSH_HOUR)
    states_a = sim_a.run(steps=steps, overlay=overlay)
    states_b = sim_b.run(steps=steps, overlay=overlay)

    for i, (a, b) in enumerate(zip(states_a, states_b)):
        if a.to_dict() != b.to_dict():
            print(_fail(f"Step {i} mismatch:\n  A={a}\n  B={b}"))
            return False

    print(_ok(f"All {steps} steps are identical across two simulators with seed={seed}"))
    return True


def test_validation() -> bool:
    """Validation should pass for all valid states and fail for bad ones."""
    print(_header("TEST: State Validation"))

    # Valid state (should not raise)
    sim = MockCitySimulator(seed=0)
    state = sim.step()
    try:
        state.validate()
        print(_ok("Valid state passed validation"))
    except ValueError as exc:
        print(_fail(f"Valid state failed validation: {exc}"))
        return False

    # Manually corrupt a state to confirm validation fires
    state.vehicle_count = 9999
    try:
        state.validate()
        print(_fail("Corrupted state did NOT raise ValueError"))
        return False
    except ValueError:
        print(_ok("Corrupted state correctly raised ValueError"))

    return True


def test_all_scenarios(seed: int = 42, steps: int = 20, verbose: bool = True) -> bool:
    """Run every scenario, validate all states, and print stats."""
    print(_header("TEST: All Scenarios"))
    gen = ScenarioGenerator()
    all_passed = True

    for scenario_type in gen.all_scenarios():
        print(
            f"\n  {_C['bold']}{scenario_type.name}{_C['reset']}"
            f"  {_C['grey']}— {gen.describe(scenario_type)}{_C['reset']}"
        )
        try:
            sim = MockCitySimulator(seed=seed)
            overlay = gen.get_overlay(scenario_type)
            states = sim.run(steps=steps, overlay=overlay)

            # Validate every state
            for state in states:
                state.validate()

            stats = _scenario_stats(states)
            if verbose:
                _print_stats(stats)

            # Print the first state as a sample
            print(f"    {_C['grey']}Sample:{_C['reset']} {states[0]}")
            print(f"  {_ok(f'{steps} states generated and validated')}")

        except Exception as exc:
            print(f"  {_fail(scenario_type.name)}: {exc}")
            traceback.print_exc()
            all_passed = False

    return all_passed


def test_reset(seed: int = 42) -> bool:
    """Reset should restart the tick counter and restore determinism."""
    print(_header("TEST: Simulator Reset"))
    sim = MockCitySimulator(seed=seed)
    first_run = sim.run(steps=3)
    sim.reset()
    second_run = sim.run(steps=3)

    for i, (a, b) in enumerate(zip(first_run, second_run)):
        if a.to_dict() != b.to_dict():
            print(_fail(f"Step {i} mismatch after reset"))
            return False

    print(_ok("Simulator produces identical output after reset()"))
    return True


def test_mixed_scenario_sequence(seed: int = 42) -> bool:
    """Simulate a realistic day: normal → rush hour → accident → normal."""
    print(_header("TEST: Mixed Scenario Sequence (simulated day)"))
    gen = ScenarioGenerator()
    sim = MockCitySimulator(seed=seed)
    all_states: list = []

    timeline = [
        (ScenarioType.NORMAL,       10, "Morning quiet"),
        (ScenarioType.RUSH_HOUR,    15, "Morning rush"),
        (ScenarioType.ACCIDENT,      5, "Accident clears slowly"),
        (ScenarioType.NORMAL,        5, "Post-accident recovery"),
        (ScenarioType.HEAVY_RAIN,    8, "Afternoon storm"),
        (ScenarioType.HEAVY_RAIN_FOG,4, "Storm + fog"),
        (ScenarioType.EMERGENCY,     3, "Emergency response"),
        (ScenarioType.NORMAL,        5, "Evening quiet"),
    ]

    for scenario_type, steps, label in timeline:
        overlay = gen.get_overlay(scenario_type)
        phase_states = [sim.step(overlay=overlay) for _ in range(steps)]
        all_states.extend(phase_states)
        total_v = sum(s.vehicle_count for s in phase_states) / len(phase_states)
        avg_spd = sum(s.average_speed for s in phase_states) / len(phase_states)
        print(
            f"    {label:30s} | steps={steps:2d}"
            f" | avg_vehicles={total_v:5.1f}"
            f" | avg_speed={avg_spd:5.1f} km/h"
        )

    print(f"\n  {_ok(f'Full day simulated: {len(all_states)} total states')}")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="CityTwin AI — Mock Simulation Test Suite"
    )
    parser.add_argument("--seed",    type=int, default=42,  help="Random seed")
    parser.add_argument("--steps",   type=int, default=20,  help="Steps per scenario")
    parser.add_argument("--quiet",   action="store_true",   help="Suppress per-scenario stats")
    args = parser.parse_args()

    print(f"\n{_C['bold']}{_C['cyan']}CityTwin AI — Mock Simulation Test Suite{_C['reset']}")
    print(f"seed={args.seed}  steps_per_scenario={args.steps}\n")

    results = {
        "Reproducibility":     test_reproducibility(seed=args.seed),
        "Validation":          test_validation(),
        "All Scenarios":       test_all_scenarios(
            seed=args.seed, steps=args.steps, verbose=not args.quiet
        ),
        "Reset":               test_reset(seed=args.seed),
        "Mixed Day Sequence":  test_mixed_scenario_sequence(seed=args.seed),
    }

    # Summary
    print(_header("SUMMARY"))
    all_passed = True
    for name, passed in results.items():
        status = _ok(name) if passed else _fail(name)
        print(f"  {status}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print(f"{_C['green']}{_C['bold']}All tests passed.{_C['reset']}")
        return 0
    else:
        print(f"{_C['red']}{_C['bold']}Some tests FAILED.{_C['reset']}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
