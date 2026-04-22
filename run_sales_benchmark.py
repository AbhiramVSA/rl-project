from __future__ import annotations

from pathlib import Path

from sales_rl_core import (
    TrainingConfig,
    evaluate_policy,
    plot_tail_risk_curve,
    plot_training_curves,
    save_history,
    save_metrics,
    train_distributional_a2c,
    train_scalar_a2c,
)


def main() -> None:
    output_dir = Path("artifacts")
    figures_dir = Path("figures")

    scalar_config = TrainingConfig(
        rollout_steps=16,
        total_updates=220,
        batch_envs=64,
        gamma=0.98,
        gae_lambda=0.95,
        lr=3e-4,
        entropy_coef=0.01,
        value_coef=0.5,
        grad_clip=1.0,
        hidden_dim=128,
        n_quantiles=31,
        cvar_alpha=0.2,
        device="cpu",
    )
    dist_config = TrainingConfig(
        rollout_steps=16,
        total_updates=260,
        batch_envs=64,
        gamma=0.98,
        gae_lambda=0.95,
        lr=2e-4,
        entropy_coef=0.01,
        value_coef=0.6,
        grad_clip=1.0,
        hidden_dim=128,
        n_quantiles=31,
        cvar_alpha=0.3,
        device="cpu",
    )

    scalar_model, scalar_history = train_scalar_a2c(seed=7, config=scalar_config)
    dist_model, dist_history = train_distributional_a2c(seed=7, config=dist_config)

    scalar_metrics = evaluate_policy(scalar_model, distributional=False, episodes=512, device=scalar_config.device)
    dist_metrics = evaluate_policy(dist_model, distributional=True, episodes=512, device=dist_config.device)

    save_history(scalar_history, output_dir / "scalar_history.json")
    save_history(dist_history, output_dir / "distributional_history.json")
    save_metrics(
        {
            "scalar_a2c": scalar_metrics,
            "distributional_a2c": dist_metrics,
            "scalar_config": scalar_config.__dict__,
            "distributional_config": dist_config.__dict__,
        },
        output_dir / "metrics.json",
    )
    plot_training_curves(scalar_history, dist_history, figures_dir)
    plot_tail_risk_curve(dist_history, figures_dir)

    print("Scalar A2C:", scalar_metrics)
    print("Distributional A2C:", dist_metrics)


if __name__ == "__main__":
    main()
