"""Train a neural-network policy on ToggleDuelEnv with REINFORCE + baseline.

Run:
    python scripts/train_toggle_duel.py --episodes 200

This is intentionally small and dependency-light — no stable-baselines, just
pure PyTorch. REINFORCE-with-baseline isn't the most sample-efficient algo
but it's the simplest one that actually learns, and it makes the gradient
flow obvious. Output is saved to ai/toggle_duel_policy.pt by default.

The trained policy can be loaded into the live duel via `MLPPolicy.load`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make sibling packages importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from ai.nn_policy import MLPPolicy
from envs import ToggleDuelEnv


def run_episode(env: ToggleDuelEnv, net: MLPPolicy,
                gamma: float = 0.99, train: bool = True):
    """Roll out one episode. Returns (total_return, policy_loss, value_loss).
    When `train` is False, no gradients are accumulated (eval rollouts)."""
    obs = env.reset()
    log_probs: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    rewards: list[float] = []
    entropies: list[torch.Tensor] = []

    done = False
    while not done:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits, value = net(obs_t)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        if train:
            action = dist.sample()
        else:
            action = logits.argmax(dim=-1)
        obs, reward, done, _ = env.step(int(action.item()))
        if train:
            log_probs.append(dist.log_prob(action).squeeze(0))
            values.append(value.squeeze(0))
            entropies.append(dist.entropy().squeeze(0))
        rewards.append(float(reward))

    total_return = float(sum(rewards))

    if not train:
        return total_return, 0.0, 0.0

    # Discounted returns
    returns: list[float] = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    returns_t = torch.tensor(returns, dtype=torch.float32)

    log_probs_t = torch.stack(log_probs)
    values_t    = torch.stack(values)
    entropies_t = torch.stack(entropies)

    # Advantages = returns - baseline value
    advantages = returns_t - values_t.detach()
    # Normalize advantages for stability (REINFORCE w/ baseline trick)
    if advantages.numel() > 1 and advantages.std() > 1e-6:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    policy_loss = -(log_probs_t * advantages).mean()
    value_loss  = F.mse_loss(values_t, returns_t)
    entropy_bonus = entropies_t.mean()
    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus
    return total_return, loss, value_loss.detach().item()


def train(episodes: int = 200,
          episode_steps: int = 300,
          lr: float = 1e-3,
          save_path: str = "ai/toggle_duel_policy.pt",
          log_every: int = 10,
          curriculum_frac: float = 0.3) -> MLPPolicy:
    """Two-phase curriculum:
      Phase 1 (first `curriculum_frac` of episodes): opponent disabled
              → policy learns to approach the toggle and back into it.
      Phase 2 (remaining episodes): opponent enabled (fighting scripted bot)
              → policy learns to fight for ownership."""
    env = ToggleDuelEnv(episode_steps=episode_steps, seed=42,
                        opponent_enabled=False)
    net = MLPPolicy(obs_dim=env.OBS_DIM, action_dim=env.ACTION_DIM, hidden=64)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    phase1_eps = int(episodes * curriculum_frac)
    print(f"Training {episodes} episodes × {episode_steps} steps "
          f"(~{episodes * episode_steps * env.DT:.0f} sim seconds)")
    print(f"  curriculum: {phase1_eps} eps vs frozen opponent, then "
          f"{episodes - phase1_eps} eps vs active ScriptedBot")
    print(f"Network: 14 → 64 → 64 → {{policy:9, value:1}}  "
          f"({sum(p.numel() for p in net.parameters())} params)")

    returns_window: list[float] = []
    t_start = time.time()
    for ep in range(episodes):
        # Curriculum transition
        if ep == phase1_eps:
            env.opponent_enabled = True
            print(f"  --- curriculum: opponent ENABLED at episode {ep+1} ---")
        ret, loss, vloss = run_episode(env, net, train=True)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        opt.step()
        returns_window.append(ret)
        if len(returns_window) > 25:
            returns_window.pop(0)
        if (ep + 1) % log_every == 0 or ep == 0:
            avg = sum(returns_window) / len(returns_window)
            elapsed = time.time() - t_start
            phase = "P1" if ep < phase1_eps else "P2"
            print(f"  [{phase}] ep {ep+1:4d}/{episodes}  return={ret:7.1f}  "
                  f"avg25={avg:7.1f}  vloss={vloss:.2f}  "
                  f"({elapsed:.0f}s)")

    save_full = Path(save_path)
    save_full.parent.mkdir(parents=True, exist_ok=True)
    net.save(save_full)
    print(f"Saved policy to {save_full}")
    return net


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--steps",    type=int, default=300,
                    help="Steps per episode (1 step = 0.05 s). "
                         "Default 300 = 15 sim seconds.")
    ap.add_argument("--lr",       type=float, default=1e-3)
    ap.add_argument("--save",     type=str,   default="ai/toggle_duel_policy.pt")
    args = ap.parse_args()
    train(episodes=args.episodes, episode_steps=args.steps,
          lr=args.lr, save_path=args.save)
    return 0


if __name__ == "__main__":
    sys.exit(main())
