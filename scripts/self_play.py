"""Self-play PPO — train the network against a frozen copy of itself.

The standard PPO trainer optimizes against a ScriptedBot opponent. Once the
network plateaus at "as good as the scripted opponent," there's no signal
to keep improving. Self-play breaks the ceiling: we freeze the current
policy as the opponent, train a fresh copy against it, and when the trainee
beats the snapshot we promote it to be the new snapshot. Round by round
the bot keeps having to outdo its own prior self, so it keeps finding new
tactics — the same mechanism that drove AlphaGo / OpenAI Five.

Usage:
    python scripts/self_play.py --rounds 4 --updates-per-round 30

That's 4 promotions × ~30 PPO updates each (~2k env steps per update) ≈
240K timesteps ≈ 6-10 min CPU. Each round, the training policy plays a
fresh frozen snapshot of the previous round's winner.

Loads:  ai/toggle_duel_policy.pt   (initial PPO/DAGGER warm start)
Saves:  ai/toggle_duel_policy.pt   (every promotion overwrites; final
                                      = the best self-play graduate)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make sibling packages importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import copy
import torch

from ai.nn_bot import NNBot
from ai.nn_policy import MLPPolicy
from envs import ToggleDuelEnv
from scripts.ppo_finetune import (
    collect_rollout, ppo_update, N_STEPS, PPO_EPOCHS, MINIBATCH_SIZE,
    LR, CLIP_EPS,
)


def make_nn_opponent_factory(frozen_policy: MLPPolicy):
    """Returns a callable that builds an NNBot driven by the given (frozen)
    policy. Used as ToggleDuelEnv.opponent_factory."""
    def factory(robot_id: int, world):
        bot = NNBot(robot_id=robot_id, world=world, policy=frozen_policy,
                     greedy=False)   # mild stochasticity so duels vary
        return bot
    return factory


def evaluate_against(policy: MLPPolicy, opponent: MLPPolicy,
                      episodes: int = 5) -> dict:
    """Win-rate of `policy` (red) against a frozen `opponent` (blue) over
    several short matches. Returns avg return + average toggle ownership."""
    factory = make_nn_opponent_factory(opponent)
    total_return = 0.0
    total_ours = 0
    total_steps = 0
    for ep in range(episodes):
        env = ToggleDuelEnv(episode_steps=300, seed=5000 + ep,
                              opponent_factory=factory)
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


def train_one_round(training_policy: MLPPolicy,
                     frozen_opponent: MLPPolicy,
                     updates: int,
                     n_steps: int,
                     lr: float) -> tuple[MLPPolicy, dict]:
    """Run `updates` PPO update steps with `training_policy` driving the
    agent and a fresh NNBot wrapping `frozen_opponent` driving the opp.
    Returns the trained policy and the last round's stats."""
    factory = make_nn_opponent_factory(frozen_opponent)
    env = ToggleDuelEnv(episode_steps=300, seed=0,
                          opponent_factory=factory)
    opt = torch.optim.Adam(training_policy.parameters(), lr=lr)
    best = -1e9
    last_stats = {}
    for upd in range(updates):
        rollout = collect_rollout(env, training_policy, n_steps=n_steps)
        stats = ppo_update(training_policy, opt, rollout)
        avg_own = (sum(rollout["ep_ownership"]) / max(1, len(rollout["ep_ownership"]))
                    if rollout["ep_ownership"] else 0.0)
        last_stats = {**stats, "ep_own": avg_own}
        if avg_own > best:
            best = avg_own
        if upd == 0 or (upd + 1) % 5 == 0:
            print(f"    upd {upd+1:3d}/{updates}  own={avg_own:.2f}  "
                  f"π_loss={stats['policy_loss']:+.3f}  "
                  f"kl={stats['approx_kl']:+.3f}")
    last_stats["best_round_own"] = best
    return training_policy, last_stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds",            type=int, default=4)
    ap.add_argument("--updates-per-round", type=int, default=30)
    ap.add_argument("--n-steps",           type=int, default=N_STEPS)
    ap.add_argument("--lr",                type=float, default=LR)
    ap.add_argument("--load",  type=str, default="ai/toggle_duel_policy.pt")
    ap.add_argument("--save",  type=str, default="ai/toggle_duel_policy.pt")
    args = ap.parse_args()

    load_path = Path(args.load)
    if not load_path.exists():
        print(f"ERROR: warm-start checkpoint not found at {load_path}.")
        print(f"Train a PPO policy first: python scripts/ppo_finetune.py")
        return 1
    training = MLPPolicy.load(load_path)
    # Frozen snapshot copy — opponent for round 1
    frozen = MLPPolicy.load(load_path)
    print(f"Warm-started from {load_path} "
          f"({sum(p.numel() for p in training.parameters())} params)")
    print(f"Self-play: {args.rounds} rounds × {args.updates_per_round} "
          f"PPO updates × {args.n_steps} env steps")

    t_start = time.time()
    for rnd in range(args.rounds):
        print(f"\n=== Round {rnd + 1}/{args.rounds} "
              f"(training vs frozen-snapshot opponent) ===")
        training, stats = train_one_round(
            training, frozen, args.updates_per_round, args.n_steps, args.lr,
        )
        # Eval vs the frozen opponent we just trained against
        m = evaluate_against(training, frozen)
        print(f"  graduate vs snapshot:  return={m['avg_return']:.1f}  "
              f"own={m['ours_avg']:.2f}  "
              f"(best during round: {stats['best_round_own']:.2f})")

        # If we clearly beat the snapshot, promote — it becomes the new
        # opponent for the next round. (Threshold avoids regressing.)
        if m["ours_avg"] > 0.55:
            print(f"  ✓ trainee clearly beats snapshot — promoting.")
            frozen.load_state_dict(training.state_dict())
        else:
            print(f"  trainee did not clearly beat snapshot — "
                  f"keeping old frozen, continuing training.")

        # Save the current trainee after each round (so progress isn't lost)
        save_path = Path(args.save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        training.save(save_path)
        print(f"  saved trainee → {save_path}")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed/60:.1f} min. Final policy at {args.save}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
