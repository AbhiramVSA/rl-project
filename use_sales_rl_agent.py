from __future__ import annotations

import argparse
import json
from pathlib import Path

from sales_rl_core import (
    ACTION_NAMES,
    STATE_FEATURE_NAMES,
    build_observation_from_state,
    load_policy_from_checkpoint,
    predict_action,
    resolve_device,
    rollout_single_episode,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load, test, and use a trained Sales RL agent.")
    parser.add_argument("--checkpoint", required=True, help="Path to a saved checkpoint.")
    parser.add_argument("--device", default="auto", help="Device to use: auto, cpu, or cuda.")
    parser.add_argument(
        "--state-file",
        help="Path to a JSON file containing a single sales state to score.",
    )
    parser.add_argument(
        "--state-json",
        help="Inline JSON string containing a single sales state to score.",
    )
    parser.add_argument(
        "--simulate-episodes",
        type=int,
        default=0,
        help="Run greedy simulator rollouts with the loaded policy for testing.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Seed used for simulator testing.")
    return parser.parse_args()


def load_state_from_args(args: argparse.Namespace) -> dict[str, float | int | str] | None:
    if args.state_file and args.state_json:
        raise ValueError("Provide only one of --state-file or --state-json.")
    if args.state_file:
        return json.loads(Path(args.state_file).read_text(encoding="utf-8"))
    if args.state_json:
        return json.loads(args.state_json)
    return None


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model, algorithm, config, checkpoint_metrics = load_policy_from_checkpoint(args.checkpoint, device=device)
    print(f"Loaded {algorithm} on {device}")
    print(json.dumps({"checkpoint_metrics": checkpoint_metrics, "config": config.__dict__}, indent=2))

    state = load_state_from_args(args)
    if state is not None:
        observation = build_observation_from_state(state)
        prediction = predict_action(model, observation, algorithm=algorithm, device=device)
        print("\nRecommended action:")
        print(json.dumps(prediction, indent=2))
    else:
        print("\nNo manual state provided.")
        print("Expected state keys:", ", ".join(STATE_FEATURE_NAMES[:-1] + ["last_action"]))
        print("Valid last_action values:", ", ".join(ACTION_NAMES + ["none"]))

    if args.simulate_episodes > 0:
        print(f"\nRunning {args.simulate_episodes} greedy simulator rollouts...")
        rollouts = [
            rollout_single_episode(model, algorithm=algorithm, seed=args.seed + idx, device=device)
            for idx in range(args.simulate_episodes)
        ]
        summary = {
            "mean_episode_return": sum(run["episode_return"] for run in rollouts) / len(rollouts),
            "conversion_rate": sum(1.0 for run in rollouts if run["converted"]) / len(rollouts),
            "mean_num_steps": sum(run["num_steps"] for run in rollouts) / len(rollouts),
        }
        print(json.dumps(summary, indent=2))
        print("\nFirst rollout:")
        print(json.dumps(rollouts[0], indent=2))


if __name__ == "__main__":
    main()
