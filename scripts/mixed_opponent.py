"""Mixed-opponent PPO — train against ScriptedBot AND frozen NN snapshots.

Self-play sharpened the policy against NN-style opponents but cost us some
of the scripted-bot exploitation we had from DAGGER+PPO. The fix is to stop
specializing: each rollout, randomly pick the opponent from a pool of
{ScriptedBot, frozen-NN-snapshot}. The trainee has to handle both styles,
so it can't overfit to either.

Usage:
    python scripts/mixed_opponent.py --updates 60 --nn-prob 0.5

    --nn-prob 0.5  →  50/50 split between scripted and frozen-NN opponents
    --nn-prob 0.0  →  100% scripted (i.e. plain ppo_finetune)
    --nn-prob 1.0  →  100% NN (i.e. self-play)

Every N updates the frozen snapshot is refreshed from the current trainee
(if trainee has improved), so the NN-opponent track gets harder over time.

Loads:  ai/toggle_duel_policy.pt
Saves:  ai/toggle_duel_policy.pt   (only when avg ownership improves)
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

# Make sibling packages importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from ai.nn_bot import NNBot
from ai.nn_policy import MLPPolicy
from envs import ToggleDuelEnv
from scripts.ppo_finetune import (
    collect_rollout, ppo_update, N_STEPS, LR,
)
from scripts.self_play import make_nn_opponent_factory, evaluate_against


def evaluate_vs_scripted(policy: MLPPolicy, episodes: int = 5) -> dict:
    """Win-rate of policy vs the default ScriptedBot opponent."""
    total_return = 0.0
    total_ours = 0
    total_steps = 0
    for ep in range(episodes):
        env = ToggleDuelEnv(episode_steps=300, seed=7000 + ep)
        obs = env.reset()
        done = False
        while not done:
            a = policy.act(obs, greedy=True)
            obs, r, done, info = env.step(a)
            total_return += r
            total_ours += info.get("ours_count", 0)
            total_steps += 1
    return {
        "avg_return": total_return / episodes,
        "ours_avg":   total_ours / max(1, total_steps),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--updates",   type=int,   default=60)
    ap.add_argument("--n-steps",   type=int,   default=N_STEPS)
    ap.add_argument("--lr",        type=float, default=LR)
    ap.add_argument("--nn-prob",   type=float, default=0.5,
                    help="Probability of picking the frozen-NN opponent per "
                         "rollout. Default 0.5 = half scripted, half NN.")
    ap.add_argument("--refresh-every", type=int, default=15,
                    help="Refresh the frozen NN snapshot from the trainee "
                         "every N updates (if trainee has improved).")
    ap.add_argument("--seed",      type=int,   default=0)
    ap.add_argument("--load",  type=str, default="ai/toggle_duel_policy.pt")
    ap.add_argument("--save",  type=str, default="ai/toggle_duel_policy.pt")
    args = ap.parse_args()

    load_path = Path(args.load)
    if not load_path.exists():
        print(f"ERROR: warm-start checkpoint not found at {load_path}.")
        print(f"Train a PPO policy first: python scripts/ppo_finetune.py")
        return 1
    training = MLPPolicy.load(load_path)
    frozen   = MLPPolicy.load(load_path)   # snapshot for the NN-opponent track
    print(f"Warm-started from {load_path} "
          f"({sum(p.numel() for p in training.parameters())} params)")
    print(f"Mixed opponent: nn_prob={args.nn_prob:.2f}  "
          f"refresh_every={args.refresh_every} updates  "
          f"updates={args.updates} × n_steps={args.n_steps}")

    rng = random.Random(args.seed)
    opt = torch.optim.Adam(training.parameters(), lr=args.lr)

    # Track the best-so-far across both opponent types, weighted equally,
    # to decide when to checkpoint and when to refresh the frozen snapshot.
    best_combined = -1.0
    last_refresh_combined = -1.0   # so we only refresh on real improvement
    n_scripted = 0
    n_nn       = 0
    t_start = time.time()

    for upd in range(args.updates):
        # Pick opponent for this rollout.
        use_nn = (rng.random() < args.nn_prob)
        if use_nn:
            factory = make_nn_opponent_factory(frozen)
            n_nn += 1
            opp_label = "NN  "
        else:
            factory = None
            n_scripted += 1
            opp_label = "scr "

        env = ToggleDuelEnv(episode_steps=300,
                              seed=rng.randint(0, 10_000),
                              opponent_factory=factory)
        rollout = collect_rollout(env, training, n_steps=args.n_steps)
        stats   = ppo_update(training, opt, rollout)
        own_list = rollout["ep_ownership"]
        avg_own  = (sum(own_list) / len(own_list)) if own_list else 0.0

        if upd == 0 or (upd + 1) % 5 == 0:
            print(f"upd {upd+1:3d}/{args.updates}  opp={opp_label}  "
                  f"own={avg_own:.2f}  "
                  f"pi_loss={stats['policy_loss']:+.3f}  "
                  f"kl={stats['approx_kl']:+.3f}")

        # Periodic combined eval: a fair score across BOTH opponents.
        if (upd + 1) % args.refresh_every == 0 or upd == args.updates - 1:
            m_scr = evaluate_vs_scripted(training, episodes=3)
            m_nn  = evaluate_against(training, frozen, episodes=3)
            combined = 0.5 * (m_scr["ours_avg"] + m_nn["ours_avg"])
            print(f"  >>> eval @upd {upd+1}: "
                  f"vs-scripted own={m_scr['ours_avg']:.2f}  "
                  f"vs-frozen   own={m_nn['ours_avg']:.2f}  "
                  f"combined={combined:.2f}")

            # Checkpoint only when combined score improves
            if combined > best_combined:
                best_combined = combined
                save_path = Path(args.save)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                training.save(save_path)
                print(f"      [BEST] new best combined={combined:.2f} -- saved to {save_path}")

            # Refresh the frozen NN snapshot if the trainee is clearly better
            # than the current frozen on the NN-track. Avoids making the NN
            # opponent harder before we're ready.
            if m_nn["ours_avg"] > 0.55 and combined > last_refresh_combined:
                frozen.load_state_dict(training.state_dict())
                last_refresh_combined = combined
                print(f"      [REFRESH] frozen NN snapshot refreshed "
                      f"(beat it {m_nn['ours_avg']:.2f}).")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed/60:.1f} min. "
          f"({n_scripted} scripted rollouts, {n_nn} NN rollouts)")
    print(f"Best combined ownership: {best_combined:.3f}")
    print(f"Final policy at {args.save}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
