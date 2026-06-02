"""Adapter that drives a robot using an MLPPolicy network.

Mirrors the public interface of `ScriptedBot` enough that App.run() can hold
a list of mixed bots and call `.update(dt)` and `.enabled` on each.

The bot:
  - reads the same observation the env would produce
  - asks the network for a discrete action (greedy by default)
  - applies the action through the same tank-physics pipeline the env uses
  - resolves collisions / wall bounds the same way

Toggle-flipping is handled outside this class (in App._update_toggle_contact),
since SC4 needs all-robots visibility — the NN bot just produces motion.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Optional

from core.state import Alliance, Phase, Robot, ToggleState, World
from ai.nn_policy import MLPPolicy
from ai.scripted_bot import (
    BACK_SENSOR_OFFSET_FT, BACK_SENSOR_RADIUS_FT,
    TOGGLE_ALIGNMENT_DOT_MIN, TOGGLE_LONG_HALF_FT, TOGGLE_SHORT_HALF_FT,
    WHEEL_ACCEL, WHEEL_BASE_FT, WHEEL_DECEL, WHEEL_MAX_SPEED, ROBOT_COLL_R_FT,
    _approach, _wrap_pi,
)
from envs.toggle_duel_env import DISCRETE_ACTIONS, make_observation


class NNBot:
    """Network-driven robot. Stateless besides wheel velocities + the policy."""

    def __init__(self, robot_id: int, world: World, policy: MLPPolicy,
                 collides_at: Optional[Callable[[float, float], bool]] = None,
                 greedy: bool = True):
        """Default is GREEDY. Greedy + stuck-detection-with-goal-bias is the
        sweet spot — the network drives most of the time, but if it freezes
        for ~1 second we override with a "head toward the toggle" maneuver
        until the bot's situation changes enough that the network can take
        over again. This gives goal-directed behavior even when the policy
        is weakly trained."""
        self.robot_id = robot_id
        self.world = world
        self.policy = policy
        self.collides_at = collides_at or (lambda x, y: False)
        self.greedy = greedy
        self._wl = 0.0
        self._wr = 0.0
        self.enabled: bool = True
        # Stuck detection + escalating recovery.
        # The bot can be stuck two ways:
        #   1) NN keeps picking "stop" or similar → translation freezes.
        #   2) NN keeps picking "turn in place" → bot spins but doesn't translate
        #      (heuristic-toward-toggle also picks turn-in-place when toggle is
        #      to the side, so escalation is needed to break this loop).
        # We track CONSECUTIVE stuck firings — each subsequent one escalates
        # the recovery action: heuristic → reverse → random.
        self._stuck_dt: float = 0.0
        self._unstuck_dt: float = 0.0
        self._prev_pos: tuple[float, float] = (0.0, 0.0)
        self._unstuck_action: int = 0
        self._consecutive_stuck: int = 0
        import random as _random
        self._rng = _random.Random(robot_id)
        # For interop with App._rebind_bots_to_world — these are read but
        # never used by the NN logic.
        self.state: str = "NN"
        self.phase: str = "POST_LOADS"
        self.target = None
        self.target_goal_id = None
        self.target_toggle_id = None
        self.match_loads_delivered: int = 0
        self.status_msg: str = "NN bot ready"

    # --- properties ---

    @property
    def robot(self) -> Robot:
        return next(r for r in self.world.robots if r.id == self.robot_id)

    @property
    def alliance(self) -> Alliance:
        return self.robot.alliance

    # --- per-frame ---

    def update(self, dt: float) -> None:
        if not self.enabled or dt <= 0.0:
            return
        if self.world.phase == Phase.ENDED:
            self._wl = 0.0
            self._wr = 0.0
            return

        # Pick an action. Escalating un-stuck if the bot's been frozen >1s:
        #   1st trigger → drive toward toggle (heuristic)
        #   2nd trigger → reverse straight back (escape walls/clamps)
        #   3rd+ trigger → random non-stop action (last resort)
        # Each override lasts ~0.8s before handing back to the network. If
        # the bot actually moved (>0.05 ft accumulated) the consecutive
        # counter resets so we start over from the heuristic next time.
        r = self.robot
        moved = math.hypot(r.x - self._prev_pos[0], r.y - self._prev_pos[1])
        if moved < 0.003:
            self._stuck_dt += dt
        else:
            self._stuck_dt = 0.0
            if moved > 0.05:
                # Significant progress — bot is escaping cleanly
                self._consecutive_stuck = 0
        self._prev_pos = (r.x, r.y)

        if self._unstuck_dt > 0.0:
            self._unstuck_dt -= dt
            action = self._unstuck_action
        elif self._stuck_dt > 1.0:
            self._consecutive_stuck += 1
            if self._consecutive_stuck == 1:
                action = self._heuristic_action_toward_toggle()
                tag = "heuristic→toggle"
            elif self._consecutive_stuck == 2:
                action = 2          # straight reverse
                tag = "reverse-out"
            else:
                action = self._rng.randint(1, len(DISCRETE_ACTIONS) - 1)
                tag = "random"
            self._unstuck_action = action
            self._unstuck_dt = 0.8
            self._stuck_dt = 0.0
            self.status_msg = (f"R{self.robot_id} NN un-stuck "
                                f"[{self._consecutive_stuck}: {tag}] a={action}")
        else:
            obs = self._observation()
            action = self.policy.act(obs, greedy=self.greedy)
        throttle, turn = DISCRETE_ACTIONS[int(action)]
        # Arcade → tank
        raw_l = float(throttle) - float(turn)
        raw_r = float(throttle) + float(turn)
        peak = max(abs(raw_l), abs(raw_r))
        if peak > 1.0:
            raw_l /= peak
            raw_r /= peak
        target_wl = raw_l * WHEEL_MAX_SPEED
        target_wr = raw_r * WHEEL_MAX_SPEED
        self._wl = _approach(self._wl, target_wl, WHEEL_ACCEL, WHEEL_DECEL, dt)
        self._wr = _approach(self._wr, target_wr, WHEEL_ACCEL, WHEEL_DECEL, dt)
        v_lin = 0.5 * (self._wl + self._wr)
        v_ang = (self._wr - self._wl) / WHEEL_BASE_FT
        r = self.robot
        r.theta = _wrap_pi(r.theta + v_ang * dt)
        bound = 6.0 - ROBOT_COLL_R_FT
        move_x = math.cos(r.theta) * v_lin * dt
        move_y = math.sin(r.theta) * v_lin * dt
        try_x = max(-bound, min(bound, r.x + move_x))
        if not self.collides_at(try_x, r.y):
            r.x = try_x
        try_y = max(-bound, min(bound, r.y + move_y))
        if not self.collides_at(r.x, try_y):
            r.y = try_y
        # Flip the toggle if back sensor engages
        self._maybe_flip_toggle()
        self.status_msg = f"R{self.robot_id} NN  a={action}"

    # --- helpers (mirror env) ---

    def _heuristic_action_toward_toggle(self) -> int:
        """Pick the discrete action that best points the chassis at the
        nearest toggle. Used only as a stuck-recovery override — the rest
        of the time the network drives. Maps the bearing-to-toggle into
        one of the 9 discrete actions:
          ahead → forward
          ahead-left / ahead-right → arc forward
          behind / behind-left / behind-right → reverse arcs
          perpendicular → turn in place
        """
        r = self.robot
        if not self.world.toggles:
            return 1   # default to forward
        t = self.world.toggles[0]
        dx = t.x - r.x
        dy = t.y - r.y
        bearing = math.atan2(dy, dx)
        # Heading-relative bearing — positive = toggle to my left, negative = right
        rel = (bearing - r.theta + math.pi) % (2 * math.pi) - math.pi
        # Convert to one of: forward / forward-arc / turn-in-place / reverse
        # action ids (matching DISCRETE_ACTIONS in env):
        #   1=fwd  3=L  4=R  5=fwd-arc-L  6=fwd-arc-R  7=back-arc-L  8=back-arc-R
        abs_rel = abs(rel)
        if abs_rel < math.radians(20):       # toggle is roughly ahead
            return 1
        if abs_rel < math.radians(60):       # ahead-but-off-angle → forward arc
            return 5 if rel > 0 else 6
        if abs_rel < math.radians(120):      # to the side → turn in place
            return 3 if rel > 0 else 4
        # Behind — reverse with the appropriate arc
        return 7 if rel > 0 else 8

    def _observation(self):
        """Defer to the shared make_observation so we always see exactly
        the layout the env produced during training."""
        return make_observation(
            self.world, self.robot, self.alliance, self._wl, self._wr,
        )

    def _maybe_flip_toggle(self) -> None:
        """If the NN-driven robot's back sensor is on a toggle (with proper
        alignment), set its resting_state to the bot's alliance color.
        SC4 contact handling stays with App._update_toggle_contact."""
        r = self.robot
        for t in self.world.toggles:
            if abs(t.x) > abs(t.y):
                nx = 1.0 if t.x > 0 else -1.0
                ny = 0.0
            else:
                nx = 0.0
                ny = 1.0 if t.y > 0 else -1.0
            bx = -math.cos(r.theta)
            by = -math.sin(r.theta)
            if (bx * nx + by * ny) < TOGGLE_ALIGNMENT_DOT_MIN:
                continue
            sx = r.x - math.cos(r.theta) * BACK_SENSOR_OFFSET_FT
            sy = r.y - math.sin(r.theta) * BACK_SENSOR_OFFSET_FT
            if abs(t.x) > abs(t.y):
                xmin = t.x - TOGGLE_SHORT_HALF_FT
                xmax = t.x + TOGGLE_SHORT_HALF_FT
                ymin = t.y - TOGGLE_LONG_HALF_FT
                ymax = t.y + TOGGLE_LONG_HALF_FT
            else:
                xmin = t.x - TOGGLE_LONG_HALF_FT
                xmax = t.x + TOGGLE_LONG_HALF_FT
                ymin = t.y - TOGGLE_SHORT_HALF_FT
                ymax = t.y + TOGGLE_SHORT_HALF_FT
            cx = max(xmin, min(sx, xmax))
            cy = max(ymin, min(sy, ymax))
            if (sx - cx) ** 2 + (sy - cy) ** 2 < BACK_SENSOR_RADIUS_FT ** 2:
                color = (ToggleState.RED if self.alliance == Alliance.RED
                          else ToggleState.BLUE)
                t.resting_state = color


__all__ = ["NNBot"]
