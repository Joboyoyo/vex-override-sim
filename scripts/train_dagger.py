"""DAGGER trainer: behavior-cloning that fixes its own mistakes.

Pure behavior cloning has a famous failure mode: the network learns to
imitate the teacher on the teacher's trajectories, but at test time tiny
prediction errors drift the bot into states the teacher never visited,
where the network has no idea what to do. The bot wanders.

DAGGER ("Dataset Aggregation") fixes this by having the NETWORK drive the
robot during data collection. At every state it visits — including the
"wrong" ones — we ask the rule-based expert "what would YOU do here?",
record (state, expert action) into a growing dataset, and retrain. After
a few iterations, the network has seen its own failure modes annotated
with the right answer.

This script:
  1. Loads or trains an initial behavior-cloned policy.
  2. Runs the policy in the env, with a shadow ScriptedBot observing the
     same world. At each step we capture the shadow's intended action.
  3. Adds those (obs, action) pairs to the dataset and retrains.
  4. Repeats for the configured number of DAGGER iterations.

Run:
    python scripts/train_dagger.py --iterations 3 --episodes-per-iter 30

Output:  ai/toggle_duel_policy.pt   (same path the live game loads)
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

# Make sibling packages importable when this script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F

from ai import ScriptedBot
from ai.nn_bot import NNBot
from ai.nn_policy import MLPPolicy
from core.state import Alliance
from envs.toggle_duel_env import (
    DISCRETE_ACTIONS, ToggleDuelEnv, _make_duel_world, make_observation,
)
from scripts.train_bc import (
    _nearest_discrete_action, _resolve_overlaps, _update_toggle,
    collect_mixed_dataset, train_bc,
)


def _shadow_expert_action(shadow: ScriptedBot, robot, world, dt: float,
                           nn_wl: float, nn_wr: float) -> int:
    """Query the scripted bot for "what would YOU do at this state?".

    We snapshot the robot pose, sync the shadow's wheel state to the NN
    bot's so its planning is grounded in the actual current motion, run
    one shadow.update(dt), capture the resulting wheel command as a
    discrete action, then restore the robot pose so the shadow's pose
    update doesn't override the NN bot's drive."""
    saved_x, saved_y, saved_theta = robot.x, robot.y, robot.theta
    shadow._v_left = nn_wl
    shadow._v_right = nn_wr
    prev_vl, prev_vr = nn_wl, nn_wr
    shadow.update(dt)
    label = _nearest_discrete_action(
        shadow._v_left, shadow._v_right, prev_vl, prev_vr,
    )
    # Restore robot pose — NN bot will do the actual drive
    robot.x, robot.y, robot.theta = saved_x, saved_y, saved_theta
    return label


def collect_dagger_dataset(policy: MLPPolicy,
                             episodes: int,
                             steps_per_episode: int,
                             seed: int = 0,
                             ) -> tuple[torch.Tensor, torch.Tensor]:
    """Roll out the policy in the duel env. The shadow ScriptedBot driver
    for R0 produces expert labels for each state the policy visits."""
    rng = random.Random(seed)
    all_obs: list[np.ndarray] = []
    all_acts: list[int]      = []
    dt = ToggleDuelEnv.DT
    t_start = time.time()
    for ep in range(episodes):
        world = _make_duel_world()
        # Wide random start so we cover lots of starting positions
        world.robots[0].x = rng.uniform(-1.0, 4.5)
        world.robots[0].y = rng.uniform(-2.7, 2.7)
        world.robots[0].theta = rng.uniform(-math.pi, math.pi)
        world.robots[1].x = rng.uniform(-1.0, 4.5)
        world.robots[1].y = rng.uniform(-2.7, 2.7)
        world.robots[1].theta = rng.uniform(-math.pi, math.pi)

        nn_bot = NNBot(robot_id=0, world=world, policy=policy)
        nn_bot.greedy = True   # deterministic-ish for clean data
        opp_bot = ScriptedBot(robot_id=2, world=world)
        opp_bot.phase = "POST_LOADS"
        opp_bot.enabled = True
        # Shadow expert: drives R0 in plan-only mode
        shadow = ScriptedBot(robot_id=0, world=world)
        shadow.phase = "POST_LOADS"
        shadow.enabled = True

        r0 = world.robots[0]
        for _ in range(steps_per_episode):
            obs = make_observation(world, r0, Alliance.RED, nn_bot._wl, nn_bot._wr)
            label = _shadow_expert_action(shadow, r0, world, dt,
                                            nn_bot._wl, nn_bot._wr)
            all_obs.append(obs)
            all_acts.append(label)

            # Now actually step the world (NN drives R0, scripted drives R2)
            nn_bot.update(dt)
            opp_bot.update(dt)
            _resolve_overlaps(world)
            _update_toggle(world)
    elapsed = time.time() - t_start
    print(f"  collected {len(all_obs)} DAGGER samples in {elapsed:.1f}s")
    return (torch.tensor(np.asarray(all_obs), dtype=torch.float32),
            torch.tensor(all_acts, dtype=torch.long))


def evaluate(policy: MLPPolicy, episodes: int = 5) -> dict:
    """Win-rate check on the real-field env. Returns avg return and the
    average per-step count of toggles we own (0 to 4). With 4 toggles in
    play, "ours_avg" close to 4 means total domination, ~2 is split."""
    total_return = 0.0
    total_ours_count = 0.0
    total_steps = 0
    for ep in range(episodes):
        env = ToggleDuelEnv(episode_steps=300, seed=1000 + ep,
                              opponent_enabled=True)
        obs = env.reset()
        done = False
        while not done:
            a = policy.act(obs, greedy=True)
            obs, r, done, info = env.step(a)
            total_return += r
            total_ours_count += info["ours_count"]
            total_steps += 1
    return {
        "avg_return":  total_return / episodes,
        "ours_avg":    total_ours_count / max(1, total_steps),
        "n_toggles":   info["n_toggles"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iterations", type=int, default=3,
                    help="Number of DAGGER iterations (collect+retrain).")
    ap.add_argument("--episodes-per-iter", type=int, default=30,
                    help="New DAGGER rollouts collected per iteration.")
    ap.add_argument("--bc-epochs", type=int, default=80,
                    help="Epochs for the initial behavior-cloning train.")
    ap.add_argument("--dagger-epochs", type=int, default=30,
                    help="Epochs retrained after adding each DAGGER batch.")
    ap.add_argument("--save", type=str, default="ai/toggle_duel_policy.pt")
    args = ap.parse_args()

    # ---- step 1: initial BC ----
    print("=== Phase 1: initial behavior cloning ===")
    base_obs, base_act = collect_mixed_dataset(seed=0)
    print(f"BC dataset: {base_obs.shape[0]} samples, obs_dim={base_obs.shape[1]}")
    policy = train_bc(base_obs, base_act, epochs=args.bc_epochs, hidden=128)
    metrics = evaluate(policy)
    print(f"  after BC:  avg_return={metrics['avg_return']:.1f}  "
          f"ours_avg={metrics['ours_avg']:.2f}/{metrics['n_toggles']} toggles")

    obs_buf, act_buf = base_obs, base_act

    # ---- step 2: DAGGER iterations ----
    for it in range(args.iterations):
        print(f"\n=== Phase 2.{it+1}: DAGGER collection + retrain ===")
        new_obs, new_act = collect_dagger_dataset(
            policy, episodes=args.episodes_per_iter,
            steps_per_episode=300, seed=100 + it,
        )
        # Append to the cumulative dataset
        obs_buf = torch.cat([obs_buf, new_obs], dim=0)
        act_buf = torch.cat([act_buf, new_act], dim=0)
        print(f"  cumulative dataset: {obs_buf.shape[0]} samples")
        policy = train_bc(obs_buf, act_buf, epochs=args.dagger_epochs,
                            hidden=128)
        metrics = evaluate(policy)
        print(f"  after DAGGER#{it+1}:  avg_return={metrics['avg_return']:.1f}  "
              f"ours_avg={metrics['ours_avg']:.2f}/{metrics['n_toggles']} toggles")

    # ---- step 3: save ----
    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    policy.save(out)
    print(f"\nSaved final policy to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
