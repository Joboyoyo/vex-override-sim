"""PPO fine-tuner — actual reinforcement learning on top of DAGGER.

DAGGER teaches the network to imitate the scripted bot, so the policy
ceiling is "as good as the teacher." PPO (Proximal Policy Optimization)
instead optimizes the network for *winning the reward* directly. Starting
from the DAGGER weights as a warm start, PPO can keep what's good about
the imitation policy and start exceeding the teacher where it can.

Usage:
    python scripts/ppo_finetune.py --updates 100

That runs ~100 update steps × 2048 env steps each ≈ 200K total timesteps
≈ 20-30 min on CPU. Each update collects a rollout, computes advantages
with GAE, then does several minibatched PPO updates.

Loads:  ai/toggle_duel_policy.pt   (DAGGER-trained init)
Saves:  ai/toggle_duel_policy.pt   (PPO-fine-tuned)

This is a single-file CleanRL-style PPO — no stable-baselines dependency.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make sibling packages importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from ai.nn_policy import MLPPolicy
from envs import ToggleDuelEnv


# ---- PPO hyperparams (CleanRL-style, tuned for fast CPU training) ----
N_STEPS         = 2048      # rollout length per update
PPO_EPOCHS      = 4         # passes over each rollout
MINIBATCH_SIZE  = 256
GAMMA           = 0.99      # discount
GAE_LAMBDA      = 0.95      # GAE smoothing
CLIP_EPS        = 0.2       # PPO clip range
LR              = 1e-4      # lower than BC since we're fine-tuning
VALUE_COEF      = 0.5
ENTROPY_COEF    = 0.01
MAX_GRAD_NORM   = 0.5


def compute_gae(rewards: list[float], values: list[float], dones: list[bool],
                last_value: float,
                gamma: float = GAMMA,
                lam: float = GAE_LAMBDA) -> list[float]:
    """Generalized Advantage Estimation. Returns the advantage per timestep."""
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = last_value if t == len(rewards) - 1 else values[t + 1]
        mask = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_v * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        advantages[t] = gae
    return advantages


def collect_rollout(env: ToggleDuelEnv, net: MLPPolicy, n_steps: int = N_STEPS):
    """Run the env for n_steps with the current policy. Returns tensors for
    the PPO update + bookkeeping (episode returns, ownership stats)."""
    obs_buf, act_buf, rew_buf = [], [], []
    val_buf, logp_buf, done_buf = [], [], []
    ep_returns: list[float] = []
    ep_ownership: list[float] = []
    cur_return = 0.0
    cur_ours_steps = 0
    cur_steps = 0

    obs = env.reset()
    for _ in range(n_steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits, value = net(obs_t)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        next_obs, reward, done, info = env.step(int(action.item()))
        obs_buf.append(obs)
        act_buf.append(int(action.item()))
        rew_buf.append(float(reward))
        val_buf.append(float(value.item()))
        logp_buf.append(float(log_prob.item()))
        done_buf.append(bool(done))

        cur_return += reward
        cur_ours_steps += info.get("ours_count", 0)
        cur_steps += 1
        if done:
            ep_returns.append(cur_return)
            ep_ownership.append(cur_ours_steps / max(1, cur_steps))
            cur_return, cur_ours_steps, cur_steps = 0.0, 0, 0
            obs = env.reset()
        else:
            obs = next_obs

    # Bootstrap last value for incomplete episode
    with torch.no_grad():
        last_obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        _, last_value = net(last_obs_t)
        last_value = float(last_value.item())

    advantages = compute_gae(rew_buf, val_buf, done_buf, last_value)
    returns = [a + v for a, v in zip(advantages, val_buf)]

    return {
        "obs":         torch.tensor(np.asarray(obs_buf), dtype=torch.float32),
        "actions":     torch.tensor(act_buf, dtype=torch.long),
        "old_log_probs": torch.tensor(logp_buf, dtype=torch.float32),
        "advantages":  torch.tensor(advantages, dtype=torch.float32),
        "returns":     torch.tensor(returns, dtype=torch.float32),
        "ep_returns":  ep_returns,
        "ep_ownership": ep_ownership,
    }


def ppo_update(net: MLPPolicy, opt: torch.optim.Optimizer, rollout: dict,
               epochs: int = PPO_EPOCHS,
               batch_size: int = MINIBATCH_SIZE,
               clip_eps: float = CLIP_EPS) -> dict:
    """Multiple minibatch PPO updates over the rollout."""
    n = rollout["obs"].shape[0]
    advs = rollout["advantages"]
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)

    policy_losses, value_losses, entropies, kls = [], [], [], []

    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            new_logits, new_values = net(rollout["obs"][idx])
            new_probs = F.softmax(new_logits, dim=-1)
            dist = torch.distributions.Categorical(probs=new_probs)
            new_log_probs = dist.log_prob(rollout["actions"][idx])
            entropy = dist.entropy().mean()

            ratio = torch.exp(new_log_probs - rollout["old_log_probs"][idx])
            adv_batch = advs[idx]
            surr1 = ratio * adv_batch
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_batch
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(new_values, rollout["returns"][idx])
            loss = (policy_loss
                    + VALUE_COEF * value_loss
                    - ENTROPY_COEF * entropy)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)
            opt.step()

            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropies.append(float(entropy.item()))
            with torch.no_grad():
                approx_kl = ((rollout["old_log_probs"][idx] - new_log_probs)
                              .mean().item())
            kls.append(approx_kl)
    return {
        "policy_loss":  sum(policy_losses) / max(1, len(policy_losses)),
        "value_loss":   sum(value_losses)  / max(1, len(value_losses)),
        "entropy":      sum(entropies)     / max(1, len(entropies)),
        "approx_kl":    sum(kls)           / max(1, len(kls)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--updates",   type=int,   default=100,
                    help="Number of PPO update steps to run.")
    ap.add_argument("--n-steps",   type=int,   default=N_STEPS,
                    help="Env steps per rollout per update.")
    ap.add_argument("--lr",        type=float, default=LR)
    ap.add_argument("--load",      type=str,
                    default="ai/toggle_duel_policy.pt",
                    help="Path to the DAGGER policy to warm-start from.")
    ap.add_argument("--save",      type=str,
                    default="ai/toggle_duel_policy.pt",
                    help="Where to save the PPO-fine-tuned policy.")
    args = ap.parse_args()

    env = ToggleDuelEnv(episode_steps=300, seed=0)
    # Warm start from DAGGER if available; otherwise random init.
    load_path = Path(args.load)
    if load_path.exists():
        net = MLPPolicy.load(load_path)
        print(f"Warm-started from {load_path} "
              f"({sum(p.numel() for p in net.parameters())} params)")
    else:
        net = MLPPolicy(obs_dim=env.OBS_DIM, action_dim=env.ACTION_DIM,
                          hidden=128)
        print(f"No checkpoint — random init "
              f"({sum(p.numel() for p in net.parameters())} params)")

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    total_steps = 0
    t_start = time.time()
    best_ownership = -1.0
    for update in range(args.updates):
        rollout = collect_rollout(env, net, n_steps=args.n_steps)
        stats = ppo_update(net, opt, rollout)
        total_steps += args.n_steps

        ep_returns = rollout["ep_returns"]
        ep_own     = rollout["ep_ownership"]
        avg_return = (sum(ep_returns) / len(ep_returns)) if ep_returns else 0.0
        avg_own    = (sum(ep_own)     / len(ep_own))     if ep_own     else 0.0
        elapsed = time.time() - t_start
        sps = int(total_steps / max(0.01, elapsed))
        if update == 0 or (update + 1) % 5 == 0:
            print(f"upd {update+1:3d}/{args.updates}  "
                  f"steps={total_steps:>7d}  "
                  f"ep_ret={avg_return:7.1f}  "
                  f"own={avg_own:.2f}  "
                  f"π_loss={stats['policy_loss']:+.3f}  "
                  f"v_loss={stats['value_loss']:.2f}  "
                  f"H={stats['entropy']:.2f}  "
                  f"kl={stats['approx_kl']:+.3f}  "
                  f"({sps} sps)")
        # Save best-so-far based on ownership
        if avg_own > best_ownership:
            best_ownership = avg_own
            save_path = Path(args.save)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            net.save(save_path)

    print(f"\nDone. Best avg ownership across updates: {best_ownership:.3f}")
    print(f"Final policy saved to {args.save}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
