from __future__ import annotations

import argparse
import json
from pathlib import Path

from sales_rl_core import (
    TrainingConfig,
    evaluate_policy,
    plot_tail_risk_curve,
    plot_training_curves,
    resolve_device,
    save_checkpoint,
    save_history,
    save_metrics,
    train_distributional_a2c,
    train_scalar_a2c,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Sales RL agent and save checkpoints.")
    parser.add_argument(
        "--algorithm",
        choices=["distributional_a2c", "scalar_a2c", "both"],
        default="distributional_a2c",
        help="Which controller to train.",
    )
    parser.add_argument("--device", default="auto", help="Device to use: auto, cpu, or cuda.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--rollout-steps", type=int, default=16)
    parser.add_argument("--total-updates", type=int, default=480)
    parser.add_argument("--batch-envs", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=0.6)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--n-quantiles", type=int, default=31)
    parser.add_argument("--cvar-alpha", type=float, default=0.3)
    parser.add_argument("--evaluate-episodes", type=int, default=512)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--figures-dir", default="figures")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace, algorithm: str, device: str) -> TrainingConfig:
    default_lr = 3e-4 if algorithm == "scalar_a2c" else args.lr
    default_value_coef = 0.5 if algorithm == "scalar_a2c" else args.value_coef
    default_updates = 400 if algorithm == "scalar_a2c" else args.total_updates
    return TrainingConfig(
        rollout_steps=args.rollout_steps,
        total_updates=default_updates,
        batch_envs=args.batch_envs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        lr=default_lr,
        entropy_coef=args.entropy_coef,
        value_coef=default_value_coef,
        grad_clip=args.grad_clip,
        hidden_dim=args.hidden_dim,
        n_quantiles=args.n_quantiles,
        cvar_alpha=args.cvar_alpha,
        device=device,
    )


def train_one(
    algorithm: str,
    args: argparse.Namespace,
    device: str,
) -> tuple[dict[str, float], dict[str, list[float]], str]:
    config = config_from_args(args, algorithm=algorithm, device=device)
    output_dir = Path(args.output_dir)
    checkpoint_dir = Path(args.checkpoint_dir)

    if algorithm == "scalar_a2c":
        model, history = train_scalar_a2c(seed=args.seed, config=config)
        metrics = evaluate_policy(
            model,
            distributional=False,
            episodes=args.evaluate_episodes,
            device=device,
            seed=1000 + args.seed,
        )
        checkpoint_name = "scalar_a2c.pt"
    else:
        model, history = train_distributional_a2c(seed=args.seed, config=config)
        metrics = evaluate_policy(
            model,
            distributional=True,
            episodes=args.evaluate_episodes,
            device=device,
            seed=2000 + args.seed,
        )
        checkpoint_name = "distributional_a2c.pt"

    save_history(history, output_dir / f"{algorithm}_history.json")
    save_checkpoint(
        model,
        algorithm=algorithm,
        config=config,
        path=checkpoint_dir / checkpoint_name,
        metrics=metrics,
    )
    save_metrics(
        {"algorithm": algorithm, "metrics": metrics, "config": config.__dict__},
        output_dir / f"{algorithm}_metrics.json",
    )
    return metrics, history, checkpoint_name


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    trained_metrics: dict[str, dict[str, float]] = {}
    histories: dict[str, dict[str, list[float]]] = {}

    algorithms = (
        ["scalar_a2c", "distributional_a2c"]
        if args.algorithm == "both"
        else [args.algorithm]
    )

    for algorithm in algorithms:
        metrics, history, checkpoint_name = train_one(algorithm, args, device)
        trained_metrics[algorithm] = metrics
        histories[algorithm] = history
        print(f"Saved {algorithm} checkpoint to {Path(args.checkpoint_dir) / checkpoint_name}")
        print(json.dumps(metrics, indent=2))

    if args.algorithm == "both":
        figures_dir = Path(args.figures_dir)
        plot_training_curves(histories["scalar_a2c"], histories["distributional_a2c"], figures_dir)
        plot_tail_risk_curve(histories["distributional_a2c"], figures_dir)
        save_metrics(
            {
                "scalar_a2c": trained_metrics["scalar_a2c"],
                "distributional_a2c": trained_metrics["distributional_a2c"],
            },
            Path(args.output_dir) / "comparison_metrics.json",
        )
        print(f"Saved comparison figures to {figures_dir}")


if __name__ == "__main__":
    main()
