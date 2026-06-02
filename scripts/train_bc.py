"""Behavior-cloning trainer for the toggle-duel policy.

Records (observation, action) pairs from a ScriptedBot driving R0 in the
duel environment, then supervised-trains MLPPolicy on cross-entropy to
predict the same actions. Much faster to converge than REINFORCE because
every sample carries a clean target signal.

Run:
    python scripts/train_bc.py --episodes 50 --epochs 40

Output: ai/toggle_duel_policy.pt (same path as the REINFORCE trainer, so
the live game's N-key can load whichever you trained last).
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
from ai.nn_policy import MLPPolicy
from ai.scripted_bot import WHEEL_MAX_SPEED
from core.state import Alliance, Phase, Robot, Toggle, ToggleState, World
from envs.toggle_duel_env import (
    DISCRETE_ACTIONS, ToggleDuelEnv, _back_sensor_overlaps,
    _make_duel_world, _toggle_outward_normal,
)


# Precomputed once — used by `_nearest_discrete_action`.
_ACTIONS_T = torch.tensor(DISCRETE_ACTIONS, dtype=torch.float32)


def _nearest_discrete_action(v_left: float, v_right: float,
                              prev_vl: float = 0.0,
                              prev_vr: float = 0.0) -> int:
    """Map the scripted bot's wheel-velocity DELTA (change this step) into
    the nearest discrete (throttle, turn) action.

    Reading the absolute wheel velocity is wrong because the bot's wheels
    ramp gradually toward their target — the first frame of "drive forward"
    looks like "barely moving" which the snap-to-nearest classifier maps to
    "stop", flooding the dataset with mislabeled stops.

    Instead we look at *how the wheels are changing* — that reflects the
    bot's intended throttle/turn even when the actuation hasn't caught up."""
    # The bot is COMMANDING a velocity; observed change ≈ acceleration toward
    # that command. Project the delta onto the (throttle, turn) basis.
    dvl = v_left  - prev_vl
    dvr = v_right - prev_vr
    delta_throttle = (dvl + dvr) * 0.5
    delta_turn     = (dvr - dvl) * 0.5
    # Also consider the current absolute velocity (for steady-state cases).
    abs_throttle   = (v_left + v_right) * 0.5 / WHEEL_MAX_SPEED
    abs_turn       = (v_right - v_left) * 0.5 / WHEEL_MAX_SPEED
    # If the wheels are actively changing this step (|delta| meaningful),
    # the command direction is more informative than the absolute speed.
    delta_mag = math.hypot(delta_throttle, delta_turn)
    abs_mag   = math.hypot(abs_throttle, abs_turn)
    if delta_mag > 0.05 and delta_mag > abs_mag * 0.5:
        # Use the delta direction, but scaled to typical action magnitudes
        scale = 1.0 / max(abs(delta_throttle), abs(delta_turn), 1e-6)
        throttle = delta_throttle * scale
        turn     = delta_turn * scale
        # Clamp so we don't extrapolate beyond the action grid
        throttle = max(-1.0, min(1.0, throttle))
        turn     = max(-1.0, min(1.0, turn))
    else:
        throttle = abs_throttle
        turn     = abs_turn
    pair = torch.tensor([throttle, turn], dtype=torch.float32)
    dists = ((_ACTIONS_T - pair) ** 2).sum(dim=-1)
    return int(dists.argmin().item())


def _make_obs(world: World, agent_robot: Robot, agent_alliance: Alliance,
              wl: float, wr: float) -> np.ndarray:
    """Same 14-d obs that ToggleDuelEnv._observation produces."""
    opp = next(r for r in world.robots if r.id != agent_robot.id)
    t = world.toggles[0]
    my_color = (ToggleState.RED if agent_alliance == Alliance.RED
                 else ToggleState.BLUE)
    opp_color = (ToggleState.BLUE if my_color == ToggleState.RED
                  else ToggleState.RED)
    if t.resting_state == my_color:
        rs = 1.0
    elif t.resting_state == opp_color:
        rs = -1.0
    else:
        rs = 0.0
    live_unset = 1.0 if t.state == ToggleState.UNSET else 0.0
    return np.array([
        agent_robot.x / 6.0, agent_robot.y / 6.0,
        math.cos(agent_robot.theta), math.sin(agent_robot.theta),
        wl / WHEEL_MAX_SPEED, wr / WHEEL_MAX_SPEED,
        opp.x / 6.0, opp.y / 6.0,
        math.cos(opp.theta), math.sin(opp.theta),
        t.x / 6.0, t.y / 6.0,
        rs, live_unset,
    ], dtype=np.float32)


def _resolve_overlaps(world: World) -> None:
    from ai.scripted_bot import ROBOT_COLL_R_FT
    r = ROBOT_COLL_R_FT
    min_d = 2.0 * r
    min_d_sq = min_d * min_d
    bound = 6.0 - r
    robots = world.robots
    for i in range(len(robots)):
        a = robots[i]
        for j in range(i + 1, len(robots)):
            b = robots[j]
            dx = b.x - a.x
            dy = b.y - a.y
            d2 = dx * dx + dy * dy
            if d2 >= min_d_sq:
                continue
            if d2 < 1e-9:
                ux, uy = 1.0, 0.0
                overlap = min_d
            else:
                d = math.sqrt(d2)
                overlap = min_d - d
                ux = dx / d
                uy = dy / d
            half = overlap * 0.5
            a.x = max(-bound, min(bound, a.x - ux * half))
            a.y = max(-bound, min(bound, a.y - uy * half))
            b.x = max(-bound, min(bound, b.x + ux * half))
            b.y = max(-bound, min(bound, b.y + uy * half))


def _update_toggle(world: World) -> None:
    """SC4 toggle contact resolution (mirror of ToggleDuelEnv)."""
    for t in world.toggles:
        any_contact = any(_back_sensor_overlaps(r, t) for r in world.robots)
        t.state = ToggleState.UNSET if any_contact else t.resting_state


def collect_bc_dataset(num_episodes: int, steps_per_episode: int,
                       seed: int = 0,
                       opponent_enabled: bool = True,
                       randomize_start: bool = True,
                       wide_start: bool = False,
                       ) -> tuple[torch.Tensor, torch.Tensor]:
    """Run `num_episodes` rollouts with one or two ScriptedBots active,
    record (obs, action_index) from R0's perspective every step.

    `opponent_enabled=False` produces a navigation-heavy dataset (R0 just
    drives to the toggle and flips it repeatedly). Mix navigation and
    fighting datasets when training to avoid the action distribution
    collapsing to "stop" (which dominates the defending phase)."""
    rng = random.Random(seed)
    all_obs: list[np.ndarray] = []
    all_actions: list[int] = []
    dt = ToggleDuelEnv.DT
    t_start = time.time()
    for ep in range(num_episodes):
        world = _make_duel_world()
        if wide_start:
            # Place R0 anywhere on the right half of the field facing a
            # random direction. Lots of diverse "drive to the toggle" data.
            world.robots[0].x = rng.uniform(-1.0, 4.5)
            world.robots[0].y = rng.uniform(-4.0, 4.0)
            world.robots[0].theta = rng.uniform(-math.pi, math.pi)
            world.robots[1].x = rng.uniform(-1.0, 4.5)
            world.robots[1].y = rng.uniform(-4.0, 4.0)
            world.robots[1].theta = rng.uniform(-math.pi, math.pi)
        elif randomize_start:
            for r in world.robots:
                r.x += rng.uniform(-0.4, 0.4)
                r.y += rng.uniform(-0.4, 0.4)
                r.theta += rng.uniform(-0.3, 0.3)
        red_bot  = ScriptedBot(robot_id=0, world=world)
        blue_bot = ScriptedBot(robot_id=2, world=world)
        red_bot.phase  = "POST_LOADS"
        blue_bot.phase = "POST_LOADS"
        red_bot.enabled  = True
        blue_bot.enabled = opponent_enabled
        for _ in range(steps_per_episode):
            obs = _make_obs(world, red_bot.robot, Alliance.RED,
                            red_bot._v_left, red_bot._v_right)
            prev_vl, prev_vr = red_bot._v_left, red_bot._v_right
            red_bot.update(dt)
            label = _nearest_discrete_action(
                red_bot._v_left, red_bot._v_right, prev_vl, prev_vr,
            )
            all_obs.append(obs)
            all_actions.append(label)
            blue_bot.update(dt)
            _resolve_overlaps(world)
            _update_toggle(world)
    elapsed = time.time() - t_start
    print(f"Collected {len(all_obs)} samples ({num_episodes} eps, "
          f"opp={'on' if opponent_enabled else 'off'}) in {elapsed:.1f}s")
    obs_t = torch.tensor(np.asarray(all_obs), dtype=torch.float32)
    act_t = torch.tensor(all_actions, dtype=torch.long)
    return obs_t, act_t


def collect_mixed_dataset(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Combine three regimes so the trained policy sees a representative slice
    of the scripted bot's behavior, weighted heavily toward NAVIGATION so
    the trained policy actually pursues the toggle rather than wandering."""
    # Lots of short wide-start nav rollouts — every kind of "drive to toggle
    # from random starting pose" the policy will see at inference time.
    wide_obs, wide_act = collect_bc_dataset(
        num_episodes=250, steps_per_episode=100, seed=seed,
        opponent_enabled=False, wide_start=True)
    # A few longer single-bot rollouts capturing the back-into-toggle motif
    nav_obs, nav_act = collect_bc_dataset(
        num_episodes=40, steps_per_episode=200, seed=seed + 1,
        opponent_enabled=False)
    # Some fight rollouts so the policy also sees defending positions
    fight_obs, fight_act = collect_bc_dataset(
        num_episodes=30, steps_per_episode=300, seed=seed + 2,
        opponent_enabled=True)
    return (torch.cat([wide_obs, nav_obs, fight_obs], dim=0),
            torch.cat([wide_act, nav_act, fight_act], dim=0))


def train_bc(obs_t: torch.Tensor, act_t: torch.Tensor,
             *, epochs: int = 40, batch_size: int = 128, lr: float = 1e-3,
             hidden: int = 128) -> MLPPolicy:
    n_actions = int(DISCRETE_ACTIONS.shape[0])
    net = MLPPolicy(obs_dim=obs_t.shape[1], action_dim=n_actions, hidden=hidden)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = obs_t.shape[0]
    # Per-class loss weights to counter the "stop dominates" imbalance —
    # frequent classes get less weight, rare classes get more, so a constant-
    # output policy doesn't trivially win the loss.
    counts = torch.bincount(act_t, minlength=n_actions).float()
    # avoid div-by-zero
    counts = counts.clamp_min(1.0)
    weights = (n / (n_actions * counts))
    weights = weights / weights.mean()        # normalize
    print(f"BC training: {n} samples, {epochs} epochs, "
          f"net params={sum(p.numel() for p in net.parameters())}, "
          f"class weights={weights.tolist()}")
    for epoch in range(epochs):
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0
        n_correct = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logits, _ = net(obs_t[idx])
            loss = F.cross_entropy(logits, act_t[idx], weight=weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += float(loss.item())
            n_batches += 1
            n_correct += int((logits.argmax(dim=-1) == act_t[idx]).sum().item())
        avg_loss = epoch_loss / max(1, n_batches)
        acc = n_correct / n
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch+1:2d}/{epochs}  loss={avg_loss:.3f}  "
                  f"top-1 acc={acc*100:.1f}%")
    return net


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--steps",    type=int, default=300)
    ap.add_argument("--epochs",   type=int, default=40)
    ap.add_argument("--save",     type=str, default="ai/toggle_duel_policy.pt")
    ap.add_argument("--mixed",    action="store_true", default=True,
                    help="Use a mixed nav+fight dataset for better coverage.")
    args = ap.parse_args()
    if args.mixed:
        obs_t, act_t = collect_mixed_dataset(seed=0)
    else:
        obs_t, act_t = collect_bc_dataset(args.episodes, args.steps, seed=0)
    counts = torch.bincount(act_t, minlength=DISCRETE_ACTIONS.shape[0]).tolist()
    print(f"raw action distribution:        {counts}")

    # Subsample "stop" — without this, the trainer collapses to predicting
    # stop on every input. Real navigation needs the moving actions to be a
    # meaningful fraction of the training set, not 86%.
    stop_idx = (act_t == 0).nonzero(as_tuple=True)[0]
    move_idx = (act_t != 0).nonzero(as_tuple=True)[0]
    n_moves = len(move_idx)
    # Keep at most n_moves stop samples (so stop is at most 50% of the data).
    n_keep_stops = min(len(stop_idx), n_moves)
    perm = torch.randperm(len(stop_idx))[:n_keep_stops]
    keep_stop_idx = stop_idx[perm]
    keep_idx = torch.cat([move_idx, keep_stop_idx])
    keep_idx = keep_idx[torch.randperm(len(keep_idx))]
    obs_t = obs_t[keep_idx]
    act_t = act_t[keep_idx]
    balanced_counts = torch.bincount(act_t, minlength=DISCRETE_ACTIONS.shape[0]).tolist()
    print(f"balanced action distribution:   {balanced_counts}")

    net = train_bc(obs_t, act_t, epochs=args.epochs)
    out = Path(args.save)
    out.parent.mkdir(parents=True, exist_ok=True)
    net.save(out)
    print(f"Saved policy to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
