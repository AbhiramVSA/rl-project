from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


ACTION_NAMES = [
    "discover",
    "qualify_budget",
    "present_value",
    "social_proof",
    "offer_discount",
    "schedule_followup",
    "close",
]

STATE_FEATURE_NAMES = [
    "fit",
    "budget",
    "urgency",
    "trust",
    "price_sensitivity",
    "objections",
    "fatigue",
    "stage",
    "info_coverage",
    "time_left",
    "last_action_scaled",
]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SalesDialogueBatchEnv:
    """Vectorized synthetic sales simulator with partial observability."""

    def __init__(self, batch_size: int, seed: int = 0, max_steps: int = 8) -> None:
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)

        self.obs_dim = 11
        self.action_dim = len(ACTION_NAMES)

        self.fit = np.zeros(batch_size, dtype=np.float32)
        self.budget = np.zeros(batch_size, dtype=np.float32)
        self.urgency = np.zeros(batch_size, dtype=np.float32)
        self.trust = np.zeros(batch_size, dtype=np.float32)
        self.price_sensitivity = np.zeros(batch_size, dtype=np.float32)
        self.fatigue = np.zeros(batch_size, dtype=np.float32)
        self.objections = np.zeros(batch_size, dtype=np.float32)
        self.stage = np.zeros(batch_size, dtype=np.float32)
        self.info_coverage = np.zeros(batch_size, dtype=np.float32)
        self.discount_offered = np.zeros(batch_size, dtype=np.float32)
        self.steps = np.zeros(batch_size, dtype=np.int32)
        self.last_action = np.full(batch_size, -1, dtype=np.int32)
        self.episode_return = np.zeros(batch_size, dtype=np.float32)

        self.reset()

    def _sample_hidden_state(self, n: int) -> dict[str, np.ndarray]:
        return {
            "fit": self.rng.beta(2.5, 1.8, size=n).astype(np.float32),
            "budget": self.rng.beta(2.2, 2.0, size=n).astype(np.float32),
            "urgency": self.rng.beta(2.0, 2.0, size=n).astype(np.float32),
            "trust": self.rng.beta(1.8, 2.4, size=n).astype(np.float32),
            "price_sensitivity": self.rng.beta(2.0, 2.0, size=n).astype(np.float32),
            "fatigue": self.rng.uniform(0.02, 0.12, size=n).astype(np.float32),
            "objections": self.rng.uniform(0.1, 0.5, size=n).astype(np.float32),
            "stage": self.rng.uniform(0.0, 0.12, size=n).astype(np.float32),
        }

    def reset(self, indices: np.ndarray | None = None) -> np.ndarray:
        if indices is None:
            indices = np.arange(self.batch_size)
        n = len(indices)
        sampled = self._sample_hidden_state(n)

        self.fit[indices] = sampled["fit"]
        self.budget[indices] = sampled["budget"]
        self.urgency[indices] = sampled["urgency"]
        self.trust[indices] = sampled["trust"]
        self.price_sensitivity[indices] = sampled["price_sensitivity"]
        self.fatigue[indices] = sampled["fatigue"]
        self.objections[indices] = sampled["objections"]
        self.stage[indices] = sampled["stage"]
        self.info_coverage[indices] = 0.0
        self.discount_offered[indices] = 0.0
        self.steps[indices] = 0
        self.last_action[indices] = -1
        self.episode_return[indices] = 0.0

        return self._observe()

    def _observe(self) -> np.ndarray:
        noise = self.rng.normal(
            loc=0.0,
            scale=np.array(
                [0.04, 0.05, 0.05, 0.04, 0.05, 0.03, 0.02, 0.02, 0.01],
                dtype=np.float32,
            ),
            size=(self.batch_size, 9),
        ).astype(np.float32)
        base = np.stack(
            [
                self.fit,
                self.budget,
                self.urgency,
                self.trust,
                self.price_sensitivity,
                self.objections,
                self.fatigue,
                self.stage,
                self.info_coverage,
            ],
            axis=1,
        ).astype(np.float32)
        obs = np.clip(base + noise, 0.0, 1.0)
        time_left = 1.0 - self.steps.astype(np.float32) / float(self.max_steps)
        last_action_scaled = np.where(
            self.last_action < 0,
            0.0,
            self.last_action.astype(np.float32) / float(self.action_dim - 1),
        )
        obs = np.concatenate(
            [
                obs,
                time_left[:, None],
                last_action_scaled[:, None],
            ],
            axis=1,
        )
        return obs.astype(np.float32)

    def _conversion_probability(self) -> np.ndarray:
        logits = (
            -1.00
            + 1.55 * self.fit
            + 1.05 * self.budget
            + 0.90 * self.urgency
            + 1.35 * self.trust
            + 0.70 * self.stage
            + 0.30 * self.info_coverage
            + 0.25 * self.discount_offered * self.price_sensitivity
            - 1.15 * self.price_sensitivity
            - 0.95 * self.fatigue
            - 0.90 * self.objections
        )
        p_conv = 1.0 / (1.0 + np.exp(-logits))
        p_conv *= 0.55 + 0.45 * self.stage
        p_conv *= 0.60 + 0.40 * self.info_coverage
        return np.clip(p_conv, 0.01, 0.99)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        actions = actions.astype(np.int32)
        rewards = np.zeros(self.batch_size, dtype=np.float32)
        done = np.zeros(self.batch_size, dtype=bool)
        success = np.zeros(self.batch_size, dtype=np.float32)

        discover = actions == 0
        rewards[discover] += 0.05 + 0.08 * (1.0 - self.info_coverage[discover]) - 0.03 * self.fatigue[discover]
        self.info_coverage[discover] += 0.18
        self.trust[discover] += 0.03 * (1.0 - self.trust[discover])
        self.stage[discover] += 0.06

        qualify = actions == 1
        rewards[qualify] += (
            0.08 * self.budget[qualify]
            + 0.03 * self.info_coverage[qualify]
            - 0.04 * self.objections[qualify]
        )
        self.info_coverage[qualify] += 0.15
        self.trust[qualify] += 0.02 * self.fit[qualify] - 0.01 * self.price_sensitivity[qualify]
        self.stage[qualify] += 0.07

        value = actions == 2
        rewards[value] += 0.08 + 0.14 * self.fit[value] + 0.05 * self.urgency[value] - 0.05 * self.objections[value]
        self.trust[value] += 0.08 * self.fit[value]
        self.stage[value] += 0.12

        proof = actions == 3
        rewards[proof] += 0.04 + 0.11 * (1.0 - self.trust[proof]) + 0.03 * self.objections[proof]
        self.trust[proof] += 0.10 * (1.0 - self.trust[proof])
        self.objections[proof] *= 0.92
        self.stage[proof] += 0.08

        discount = actions == 4
        rewards[discount] += (
            -0.06
            + 0.14 * self.price_sensitivity[discount]
            + 0.05 * self.stage[discount]
            - 0.04 * (1.0 - self.urgency[discount])
        )
        self.discount_offered[discount] = 1.0
        self.trust[discount] -= 0.03 * (1.0 - self.price_sensitivity[discount])
        self.stage[discount] += 0.05

        followup = actions == 5
        rewards[followup] += (
            0.03
            + 0.09 * (1.0 - self.urgency[followup])
            + 0.06 * self.trust[followup]
            - 0.04 * self.fatigue[followup]
        )
        self.stage[followup] += 0.05
        self.trust[followup] += 0.04

        close = actions == 6
        p_conv = np.clip(self._conversion_probability(), 0.02, 0.98)
        win = self.rng.random(self.batch_size) < p_conv
        close_success = close & win
        close_fail = close & ~win
        premature_close = close & ((self.stage < 0.45) | (self.info_coverage < 0.35))
        deal_value = 1.8 + 1.8 * self.budget + 1.2 * self.fit
        rewards[close_success] += 2.5 + deal_value[close_success]
        rewards[close_fail] -= 0.65 + 0.25 * (1.0 - self.trust[close_fail])
        rewards[premature_close] -= (
            1.10
            + 0.40 * np.maximum(0.0, 0.45 - self.stage[premature_close])
            + 0.40 * np.maximum(0.0, 0.35 - self.info_coverage[premature_close])
        )
        done[close] = True
        success[close_success] = 1.0

        # Global dialogue dynamics.
        non_close = ~close
        self.fatigue[non_close] += 0.05 + 0.03 * (self.steps[non_close] / max(self.max_steps - 1, 1))
        self.objections[non_close] += 0.02 * self.price_sensitivity[non_close] - 0.03 * self.trust[non_close]
        self.objections = np.clip(self.objections, 0.0, 1.0)
        self.fatigue = np.clip(self.fatigue, 0.0, 1.0)
        self.trust = np.clip(self.trust, 0.0, 1.0)
        self.stage = np.clip(self.stage, 0.0, 1.0)
        self.info_coverage = np.clip(self.info_coverage, 0.0, 1.0)

        self.steps += 1
        self.last_action = actions
        timeout = self.steps >= self.max_steps
        timeout_only = timeout & ~done
        rewards[timeout_only] += -0.55 + 0.10 * self.trust[timeout_only] + 0.05 * self.stage[timeout_only]
        done[timeout_only] = True

        self.episode_return += rewards
        episode_returns = self.episode_return.copy()
        episode_lengths = self.steps.copy()

        done_indices = np.flatnonzero(done)
        if len(done_indices) > 0:
            self.reset(done_indices)

        info = {
            "done_mask": done,
            "episode_returns": episode_returns[done],
            "episode_lengths": episode_lengths[done],
            "successes": success[done],
            "conversion_prob": p_conv[done],
        }
        return self._observe(), rewards.astype(np.float32), done.astype(np.float32), info


class ScalarActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        return self.actor(h), self.critic(h).squeeze(-1)


class DistributionalActorCritic(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        n_quantiles: int = 31,
    ) -> None:
        super().__init__()
        self.n_quantiles = n_quantiles
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.quantile_head = nn.Linear(hidden_dim, n_quantiles)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        logits = self.actor(h)
        quantiles = torch.sort(self.quantile_head(h), dim=-1).values
        return logits, quantiles


@dataclass
class TrainingConfig:
    rollout_steps: int = 16
    total_updates: int = 220
    batch_envs: int = 64
    gamma: float = 0.98
    gae_lambda: float = 0.95
    lr: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    grad_clip: float = 1.0
    hidden_dim: int = 128
    n_quantiles: int = 31
    cvar_alpha: float = 0.2
    device: str = "cpu"


def resolve_device(device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def build_observation_from_state(state: dict[str, float | int | str]) -> np.ndarray:
    action_to_scale = {name: idx / float(len(ACTION_NAMES) - 1) for idx, name in enumerate(ACTION_NAMES)}
    observation = np.array(
        [
            float(state["fit"]),
            float(state["budget"]),
            float(state["urgency"]),
            float(state["trust"]),
            float(state["price_sensitivity"]),
            float(state["objections"]),
            float(state["fatigue"]),
            float(state["stage"]),
            float(state["info_coverage"]),
            float(state["time_left"]),
            _normalize_last_action(state["last_action"], action_to_scale),
        ],
        dtype=np.float32,
    )
    return np.clip(observation, 0.0, 1.0)


def _normalize_last_action(last_action: float | int | str, action_to_scale: dict[str, float]) -> float:
    if isinstance(last_action, str):
        if last_action == "none":
            return 0.0
        if last_action not in action_to_scale:
            raise ValueError(f"Unknown last_action '{last_action}'. Valid actions: {ACTION_NAMES} or 'none'.")
        return action_to_scale[last_action]
    if isinstance(last_action, (int, float)):
        action_idx = int(last_action)
        if action_idx < 0:
            return 0.0
        if action_idx >= len(ACTION_NAMES):
            raise ValueError(f"last_action index {action_idx} is out of range for {len(ACTION_NAMES)} actions.")
        return action_idx / float(len(ACTION_NAMES) - 1)
    raise TypeError("last_action must be a string action name, an integer action index, or 'none'.")


def _gae_returns(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(rewards.shape[1], device=rewards.device)
    for t in reversed(range(rewards.shape[0])):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values * mask - values[t]
        last_adv = delta + gamma * gae_lambda * mask * last_adv
        advantages[t] = last_adv
        next_values = values[t]
    returns = advantages + values
    return advantages, returns


def _quantile_huber_loss(
    pred_quantiles: torch.Tensor,
    target_quantiles: torch.Tensor,
    taus: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    diff = target_quantiles.unsqueeze(1) - pred_quantiles.unsqueeze(2)
    abs_diff = diff.abs()
    huber = torch.where(abs_diff <= kappa, 0.5 * diff.pow(2), kappa * (abs_diff - 0.5 * kappa))
    tau = taus.view(1, -1, 1)
    weight = (tau - (diff.detach() < 0).float()).abs()
    return (weight * huber / kappa).mean()


def _rolling_mean(values: list[float], window: int = 25) -> list[float]:
    if not values:
        return []
    result = []
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        result.append(float(np.mean(values[start : idx + 1])))
    return result


def train_scalar_a2c(seed: int, config: TrainingConfig) -> tuple[ScalarActorCritic, dict[str, list[float]]]:
    set_global_seed(seed)
    device = torch.device(config.device)
    env = SalesDialogueBatchEnv(batch_size=config.batch_envs, seed=seed)
    model = ScalarActorCritic(env.obs_dim, env.action_dim, hidden_dim=config.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    obs = torch.tensor(env.reset(), dtype=torch.float32, device=device)

    history: dict[str, list[float]] = {
        "episode_return": [],
        "success_rate": [],
        "value_loss": [],
        "policy_loss": [],
        "entropy": [],
    }

    recent_successes: list[float] = []

    for _ in range(config.total_updates):
        rollout_obs = []
        rollout_logprob = []
        rollout_reward = []
        rollout_done = []
        rollout_value = []
        rollout_entropy = []

        for _ in range(config.rollout_steps):
            logits, value = model(obs)
            dist = Categorical(logits=logits)
            action = dist.sample()

            next_obs, reward, done, info = env.step(action.cpu().numpy())

            rollout_obs.append(obs)
            rollout_logprob.append(dist.log_prob(action))
            rollout_reward.append(torch.tensor(reward, dtype=torch.float32, device=device))
            rollout_done.append(torch.tensor(done, dtype=torch.float32, device=device))
            rollout_value.append(value)
            rollout_entropy.append(dist.entropy())

            if len(info["episode_returns"]) > 0:
                history["episode_return"].extend(info["episode_returns"].astype(float).tolist())
                recent_successes.extend(info["successes"].astype(float).tolist())
                history["success_rate"].append(float(np.mean(recent_successes[-100:])))

            obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

        with torch.no_grad():
            _, next_value = model(obs)

        rewards = torch.stack(rollout_reward)
        dones = torch.stack(rollout_done)
        values = torch.stack(rollout_value)
        log_probs = torch.stack(rollout_logprob)
        entropies = torch.stack(rollout_entropy)

        advantages, returns = _gae_returns(
            rewards=rewards,
            dones=dones,
            values=values,
            next_values=next_value,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        policy_loss = -(log_probs * advantages.detach()).mean()
        value_loss = F.mse_loss(values, returns.detach())
        entropy_bonus = entropies.mean()
        loss = policy_loss + config.value_coef * value_loss - config.entropy_coef * entropy_bonus

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        history["value_loss"].append(float(value_loss.item()))
        history["policy_loss"].append(float(policy_loss.item()))
        history["entropy"].append(float(entropy_bonus.item()))

    history["episode_return_smoothed"] = _rolling_mean(history["episode_return"])
    history["success_rate_smoothed"] = _rolling_mean(history["success_rate"])
    return model, history


def train_distributional_a2c(
    seed: int, config: TrainingConfig
) -> tuple[DistributionalActorCritic, dict[str, list[float]]]:
    set_global_seed(seed)
    device = torch.device(config.device)
    env = SalesDialogueBatchEnv(batch_size=config.batch_envs, seed=seed)
    model = DistributionalActorCritic(
        env.obs_dim,
        env.action_dim,
        hidden_dim=config.hidden_dim,
        n_quantiles=config.n_quantiles,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    obs = torch.tensor(env.reset(), dtype=torch.float32, device=device)
    taus = torch.linspace(
        0.5 / config.n_quantiles,
        1.0 - 0.5 / config.n_quantiles,
        config.n_quantiles,
        device=device,
    )
    k_cvar = max(1, int(math.ceil(config.cvar_alpha * config.n_quantiles)))

    history: dict[str, list[float]] = {
        "episode_return": [],
        "success_rate": [],
        "critic_loss": [],
        "policy_loss": [],
        "entropy": [],
        "tail_value": [],
    }
    recent_successes: list[float] = []

    for _ in range(config.total_updates):
        rollout_logprob = []
        rollout_reward = []
        rollout_done = []
        rollout_quantiles = []
        rollout_entropy = []
        next_quantile_inputs = []

        for _ in range(config.rollout_steps):
            logits, quantiles = model(obs)
            dist = Categorical(logits=logits)
            action = dist.sample()

            next_obs, reward, done, info = env.step(action.cpu().numpy())

            rollout_logprob.append(dist.log_prob(action))
            rollout_reward.append(torch.tensor(reward, dtype=torch.float32, device=device))
            rollout_done.append(torch.tensor(done, dtype=torch.float32, device=device))
            rollout_quantiles.append(quantiles)
            rollout_entropy.append(dist.entropy())
            next_quantile_inputs.append(torch.tensor(next_obs, dtype=torch.float32, device=device))

            if len(info["episode_returns"]) > 0:
                history["episode_return"].extend(info["episode_returns"].astype(float).tolist())
                recent_successes.extend(info["successes"].astype(float).tolist())
                history["success_rate"].append(float(np.mean(recent_successes[-100:])))

            obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

        rewards = torch.stack(rollout_reward)
        dones = torch.stack(rollout_done)
        pred_quantiles = torch.stack(rollout_quantiles)
        log_probs = torch.stack(rollout_logprob)
        entropies = torch.stack(rollout_entropy)

        with torch.no_grad():
            next_quantiles = []
            for next_obs_tensor in next_quantile_inputs:
                _, q_next = model(next_obs_tensor)
                next_quantiles.append(q_next)
            next_quantiles = torch.stack(next_quantiles)

        target_quantiles = rewards.unsqueeze(-1) + config.gamma * (1.0 - dones.unsqueeze(-1)) * next_quantiles

        pred_flat = pred_quantiles.view(-1, config.n_quantiles)
        target_flat = target_quantiles.view(-1, config.n_quantiles).detach()
        critic_loss = _quantile_huber_loss(pred_flat, target_flat, taus)

        pred_mean = pred_quantiles.mean(dim=-1)
        target_mean = target_quantiles.mean(dim=-1)
        pred_tail = pred_quantiles[:, :, :k_cvar].mean(dim=-1)
        target_tail = target_quantiles[:, :, :k_cvar].mean(dim=-1)
        uncertainty_penalty = 0.10 * pred_quantiles.std(dim=-1)
        advantages = target_mean - pred_mean.detach() - uncertainty_penalty.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

        policy_loss = -(log_probs * advantages.detach()).mean()
        entropy_bonus = entropies.mean()
        loss = policy_loss + config.value_coef * critic_loss - config.entropy_coef * entropy_bonus

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        history["critic_loss"].append(float(critic_loss.item()))
        history["policy_loss"].append(float(policy_loss.item()))
        history["entropy"].append(float(entropy_bonus.item()))
        history["tail_value"].append(float(target_tail.mean().item()))

    history["episode_return_smoothed"] = _rolling_mean(history["episode_return"])
    history["success_rate_smoothed"] = _rolling_mean(history["success_rate"])
    history["tail_value_smoothed"] = _rolling_mean(history["tail_value"])
    return model, history


def evaluate_policy(
    model: nn.Module,
    distributional: bool,
    episodes: int = 512,
    seed: int = 1234,
    device: str = "cpu",
) -> dict[str, float]:
    env = SalesDialogueBatchEnv(batch_size=64, seed=seed)
    obs = torch.tensor(env.reset(), dtype=torch.float32, device=device)
    finished_returns: list[float] = []
    finished_success: list[float] = []
    finished_lengths: list[float] = []

    with torch.no_grad():
        while len(finished_returns) < episodes:
            if distributional:
                logits, _ = model(obs)
            else:
                logits, _ = model(obs)
            action = torch.argmax(logits, dim=-1)
            next_obs, _, _, info = env.step(action.cpu().numpy())
            if len(info["episode_returns"]) > 0:
                finished_returns.extend(info["episode_returns"].astype(float).tolist())
                finished_success.extend(info["successes"].astype(float).tolist())
                finished_lengths.extend(info["episode_lengths"].astype(float).tolist())
            obs = torch.tensor(next_obs, dtype=torch.float32, device=device)

    returns = np.array(finished_returns[:episodes], dtype=np.float32)
    successes = np.array(finished_success[:episodes], dtype=np.float32)
    lengths = np.array(finished_lengths[:episodes], dtype=np.float32)
    sorted_returns = np.sort(returns)
    cvar_count = max(1, int(0.1 * len(sorted_returns)))
    return {
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "success_rate": float(np.mean(successes)),
        "mean_length": float(np.mean(lengths)),
        "cvar10": float(np.mean(sorted_returns[:cvar_count])),
    }


def save_history(history: dict[str, list[float]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2), encoding="utf-8")


def save_metrics(metrics: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def save_checkpoint(
    model: nn.Module,
    algorithm: str,
    config: TrainingConfig,
    path: str | Path,
    metrics: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "algorithm": algorithm,
            "config": config.__dict__,
            "model_state_dict": model.state_dict(),
            "metrics": metrics or {},
            "action_names": ACTION_NAMES,
            "state_feature_names": STATE_FEATURE_NAMES,
        },
        path,
    )


def load_checkpoint(path: str | Path, device: str = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=device, weights_only=False)


def load_policy_from_checkpoint(
    path: str | Path,
    device: str = "cpu",
) -> tuple[nn.Module, str, TrainingConfig, dict[str, Any]]:
    checkpoint = load_checkpoint(path, device=device)
    config = TrainingConfig(**checkpoint["config"])
    algorithm = checkpoint["algorithm"]
    obs_dim = len(STATE_FEATURE_NAMES)
    action_dim = len(ACTION_NAMES)

    if algorithm == "scalar_a2c":
        model = ScalarActorCritic(obs_dim, action_dim, hidden_dim=config.hidden_dim)
    elif algorithm == "distributional_a2c":
        model = DistributionalActorCritic(
            obs_dim,
            action_dim,
            hidden_dim=config.hidden_dim,
            n_quantiles=config.n_quantiles,
        )
    else:
        raise ValueError(f"Unsupported algorithm '{algorithm}' in checkpoint.")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, algorithm, config, checkpoint.get("metrics", {})


def predict_action(
    model: nn.Module,
    observation: np.ndarray,
    algorithm: str,
    device: str = "cpu",
) -> dict[str, Any]:
    obs_tensor = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        if algorithm == "distributional_a2c":
            logits, quantiles = model(obs_tensor)
            quantiles_np = quantiles.squeeze(0).cpu().numpy()
            value_mean = float(np.mean(quantiles_np))
            tail_count = max(1, int(math.ceil(0.1 * len(quantiles_np))))
            value_cvar10 = float(np.mean(quantiles_np[:tail_count]))
            uncertainty = float(np.std(quantiles_np))
        else:
            logits, value = model(obs_tensor)
            quantiles_np = None
            value_mean = float(value.squeeze(0).item())
            value_cvar10 = value_mean
            uncertainty = 0.0

        probs = torch.softmax(logits.squeeze(0), dim=-1).cpu().numpy()
        action_idx = int(np.argmax(probs))

    ranked = sorted(
        [
            {"action": ACTION_NAMES[idx], "probability": float(prob)}
            for idx, prob in enumerate(probs.tolist())
        ],
        key=lambda item: item["probability"],
        reverse=True,
    )
    return {
        "recommended_action": ACTION_NAMES[action_idx],
        "recommended_action_index": action_idx,
        "action_probabilities": ranked,
        "value_mean": value_mean,
        "value_cvar10": value_cvar10,
        "uncertainty": uncertainty,
        "quantiles": quantiles_np.tolist() if quantiles_np is not None else None,
    }


def rollout_single_episode(
    model: nn.Module,
    algorithm: str,
    seed: int = 1234,
    device: str = "cpu",
) -> dict[str, Any]:
    env = SalesDialogueBatchEnv(batch_size=1, seed=seed)
    obs = env.reset()[0]
    steps: list[dict[str, Any]] = []
    total_reward = 0.0
    converted = False

    while True:
        prediction = predict_action(model, obs, algorithm=algorithm, device=device)
        action_idx = prediction["recommended_action_index"]
        next_obs, reward, done, info = env.step(np.array([action_idx], dtype=np.int32))
        total_reward += float(reward[0])
        step_record = {
            "observation": obs.tolist(),
            "action": ACTION_NAMES[action_idx],
            "reward": float(reward[0]),
            "value_mean": prediction["value_mean"],
            "value_cvar10": prediction["value_cvar10"],
            "uncertainty": prediction["uncertainty"],
        }
        steps.append(step_record)
        obs = next_obs[0]
        if float(done[0]) > 0.5:
            converted = bool(info["successes"][0]) if len(info["successes"]) else False
            break

    return {
        "episode_return": total_reward,
        "converted": converted,
        "num_steps": len(steps),
        "steps": steps,
    }


def plot_training_curves(
    scalar_history: dict[str, list[float]],
    distributional_history: dict[str, list[float]],
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    axes[0].plot(scalar_history["episode_return_smoothed"], label="A2C", linewidth=2.2, color="#1f77b4")
    axes[0].plot(
        distributional_history["episode_return_smoothed"],
        label="Distributional A2C",
        linewidth=2.2,
        color="#d62728",
    )
    axes[0].set_title("Episode Return")
    axes[0].set_xlabel("Completed episodes")
    axes[0].set_ylabel("Smoothed return")
    axes[0].grid(alpha=0.2)
    axes[0].legend(frameon=False)

    axes[1].plot(scalar_history["success_rate_smoothed"], label="A2C", linewidth=2.2, color="#1f77b4")
    axes[1].plot(
        distributional_history["success_rate_smoothed"],
        label="Distributional A2C",
        linewidth=2.2,
        color="#d62728",
    )
    axes[1].set_title("Success Rate")
    axes[1].set_xlabel("Update index")
    axes[1].set_ylabel("Rolling success")
    axes[1].grid(alpha=0.2)
    axes[1].legend(frameon=False)
    fig.savefig(output_dir / "training_curves.png", dpi=220)
    fig.savefig(output_dir / "training_curves.pdf")
    plt.close(fig)


def plot_tail_risk_curve(
    distributional_history: dict[str, list[float]],
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6.2, 4.4), constrained_layout=True)
    ax.plot(
        distributional_history["tail_value_smoothed"],
        linewidth=2.2,
        color="#2ca02c",
    )
    ax.set_title("Distributional Tail Value")
    ax.set_xlabel("Update index")
    ax.set_ylabel("Smoothed CVaR proxy")
    ax.grid(alpha=0.2)
    fig.savefig(output_dir / "tail_risk_curve.png", dpi=220)
    fig.savefig(output_dir / "tail_risk_curve.pdf")
    plt.close(fig)
