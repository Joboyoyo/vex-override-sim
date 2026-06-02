"""Toggle-Duel — a minimal RL environment for the 1v1 toggle fight.

The agent drives one robot; the opposing robot is the existing rule-based
`ScriptedBot`. Each second the toggle's resting_state matches the agent's
alliance, the agent earns +1; otherwise 0. The episode is 30 seconds at
20 Hz (600 steps).

This is a pure-physics environment — no pygame import. Mirrors the duel-mode
update loop in `render.pygame_view.App` but skips rendering, joystick polling,
and event handling.

API mimics OpenAI Gym (no dependency on gymnasium yet):
    env = ToggleDuelEnv()
    obs = env.reset()
    obs, reward, done, info = env.step(action)
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Optional

# Make sibling packages importable when this file is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from core.state import (
    Alliance, Goal, GoalType, Phase, Robot, Toggle, ToggleState, World,
)
from ai import ScriptedBot
from ai.scripted_bot import (
    BACK_SENSOR_OFFSET_FT, BACK_SENSOR_RADIUS_FT,
    TOGGLE_ALIGNMENT_DOT_MIN, TOGGLE_LONG_HALF_FT, TOGGLE_SHORT_HALF_FT,
    WHEEL_ACCEL, WHEEL_BASE_FT, WHEEL_DECEL, WHEEL_MAX_SPEED, ROBOT_COLL_R_FT,
    _approach, _wrap_pi,
)


# Same nine-action grid PPO/discrete-policy networks tend to like — covers
# stop, forward/back, in-place turns, and diagonal arcs.
DISCRETE_ACTIONS = np.array([
    [ 0.0,  0.0],   # 0  stop
    [ 1.0,  0.0],   # 1  forward
    [-1.0,  0.0],   # 2  backward
    [ 0.0, -1.0],   # 3  turn left in place
    [ 0.0,  1.0],   # 4  turn right in place
    [ 0.7, -0.5],   # 5  forward + arc left
    [ 0.7,  0.5],   # 6  forward + arc right
    [-0.7, -0.5],   # 7  backward + arc left
    [-0.7,  0.5],   # 8  backward + arc right
], dtype=np.float32)


def _make_duel_world() -> World:
    """Pygame-free clone of `render.pygame_view.make_toggle_duel_world`."""
    goals = [
        Goal(0, GoalType.ALLIANCE, Alliance.RED,  x=-4.0, y=-2.0, quadrant=3, awp_side=Alliance.RED),
        Goal(1, GoalType.ALLIANCE, Alliance.RED,  x=-2.0, y=-4.0, quadrant=2, awp_side=Alliance.RED),
        Goal(2, GoalType.ALLIANCE, Alliance.BLUE, x=2.0,  y=4.0,  quadrant=0, awp_side=Alliance.BLUE),
        Goal(3, GoalType.ALLIANCE, Alliance.BLUE, x=4.0,  y=2.0,  quadrant=1, awp_side=Alliance.BLUE),
        Goal(4, GoalType.SHORT, Alliance.NEUTRAL, x=-2.0, y=4.0,  quadrant=0, awp_side=Alliance.BLUE),
        Goal(5, GoalType.SHORT, Alliance.NEUTRAL, x=4.0,  y=-2.0, quadrant=1, awp_side=Alliance.BLUE),
        Goal(6, GoalType.SHORT, Alliance.NEUTRAL, x=2.0,  y=-4.0, quadrant=2, awp_side=Alliance.RED),
        Goal(7, GoalType.SHORT, Alliance.NEUTRAL, x=-4.0, y=2.0,  quadrant=3, awp_side=Alliance.RED),
        Goal(8, GoalType.TALL,  Alliance.NEUTRAL, x=0.0,  y=0.0,  quadrant=0,
             in_midfield=True, awp_side=Alliance.NEUTRAL),
    ]
    toggles = [
        Toggle(1, quadrant=1, state=ToggleState.YELLOW, x=6.0, y=0.0),
    ]
    robots = [
        Robot(id=0, alliance=Alliance.RED,  x=2.0, y= 2.0, theta=0.0),
        Robot(id=2, alliance=Alliance.BLUE, x=2.0, y=-2.0, theta=0.0),
    ]
    return World(goals=goals, toggles=toggles, robots=robots, phase=Phase.DRIVER)


def _toggle_aabb(t: Toggle) -> tuple[float, float, float, float]:
    if abs(t.x) > abs(t.y):
        return (t.x - TOGGLE_SHORT_HALF_FT, t.x + TOGGLE_SHORT_HALF_FT,
                t.y - TOGGLE_LONG_HALF_FT,  t.y + TOGGLE_LONG_HALF_FT)
    return (t.x - TOGGLE_LONG_HALF_FT,  t.x + TOGGLE_LONG_HALF_FT,
            t.y - TOGGLE_SHORT_HALF_FT, t.y + TOGGLE_SHORT_HALF_FT)


def _toggle_outward_normal(t: Toggle) -> tuple[float, float]:
    if abs(t.x) > abs(t.y):
        return (1.0 if t.x > 0 else -1.0, 0.0)
    return (0.0, 1.0 if t.y > 0 else -1.0)


def _back_sensor_overlaps(robot: Robot, t: Toggle) -> bool:
    """Same alignment-gated back-sensor / AABB test the game uses."""
    nx, ny = _toggle_outward_normal(t)
    bx = -math.cos(robot.theta)
    by = -math.sin(robot.theta)
    if (bx * nx + by * ny) < TOGGLE_ALIGNMENT_DOT_MIN:
        return False
    x_min, x_max, y_min, y_max = _toggle_aabb(t)
    sx = robot.x - math.cos(robot.theta) * BACK_SENSOR_OFFSET_FT
    sy = robot.y - math.sin(robot.theta) * BACK_SENSOR_OFFSET_FT
    cx = max(x_min, min(sx, x_max))
    cy = max(y_min, min(sy, y_max))
    ddx = sx - cx
    ddy = sy - cy
    return ddx * ddx + ddy * ddy < BACK_SENSOR_RADIUS_FT * BACK_SENSOR_RADIUS_FT


class ToggleDuelEnv:
    """1v1 toggle-fight env, Gym-style.

    Observation (14-d float vector, all normalized roughly to ~[-1, 1]):
        0-1   own.x, own.y            (÷6)
        2-3   own.cosθ, own.sinθ
        4-5   own wheel velocities    (÷WHEEL_MAX_SPEED)
        6-7   opp.x, opp.y            (÷6)
        8-9   opp.cosθ, opp.sinθ
       10-11  toggle.x, toggle.y       (÷6) — fixed, but helps the net localize
       12    resting_state == own color ? 1 : (== opp color ? -1 : 0)
       13    live state == UNSET ? 1 : 0

    Action: discrete 0..8 → (throttle, turn) per DISCRETE_ACTIONS.

    Reward: +1 per step the toggle's resting_state matches the agent's
            alliance, 0 otherwise. Light shaping bonus (+0.05) on the step
            we actually flip it (resting_state transitions to our color).
    """

    OBS_DIM = 14
    ACTION_DIM = 9
    DT = 0.05
    DEFAULT_EPISODE_STEPS = 600

    def __init__(self,
                 agent_alliance: Alliance = Alliance.RED,
                 episode_steps: int = DEFAULT_EPISODE_STEPS,
                 seed: Optional[int] = None,
                 use_shaping: bool = True,
                 opponent_enabled: bool = True):
        """
        use_shaping       — add tiny dense bonuses for proximity to the toggle
                            and heading alignment with the wall normal. Without
                            this, REINFORCE struggles to ever stumble onto the
                            sparse +1 reward signal in a reasonable budget.
        opponent_enabled  — if False, the opposing ScriptedBot sits motionless
                            in IDLE. Useful as a curriculum step: first learn
                            to reach the toggle, then learn to fight for it.
        """
        self.agent_alliance = agent_alliance
        self.episode_steps = episode_steps
        self.use_shaping = use_shaping
        self.opponent_enabled = opponent_enabled
        self._rng = random.Random(seed)
        self.world: Optional[World] = None
        self.agent_robot: Optional[Robot] = None
        self.opp_robot: Optional[Robot] = None
        self.opp_bot: Optional[ScriptedBot] = None
        self.toggle: Optional[Toggle] = None
        self._wl = 0.0      # agent's left wheel velocity
        self._wr = 0.0      # agent's right wheel velocity
        self._step = 0
        self._prev_resting: Optional[ToggleState] = None

    # ---------- core API ----------

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        if seed is not None:
            self._rng = random.Random(seed)
        self.world = _make_duel_world()
        # Randomize starting positions a touch so the agent doesn't overfit.
        for r in self.world.robots:
            r.x += self._rng.uniform(-0.3, 0.3)
            r.y += self._rng.uniform(-0.3, 0.3)
            r.theta += self._rng.uniform(-0.2, 0.2)
        # Pick which robot the agent drives
        if self.agent_alliance == Alliance.RED:
            self.agent_robot = self.world.robots[0]
            self.opp_robot   = self.world.robots[1]
        else:
            self.agent_robot = self.world.robots[1]
            self.opp_robot   = self.world.robots[0]
        # Opponent: existing scripted bot, immediately in POST_LOADS phase
        self.opp_bot = ScriptedBot(robot_id=self.opp_robot.id, world=self.world)
        self.opp_bot.phase = "POST_LOADS"
        self.opp_bot.enabled = self.opponent_enabled
        # Toggle reference
        self.toggle = self.world.toggles[0]
        self._wl = 0.0
        self._wr = 0.0
        self._step = 0
        self._prev_resting = self.toggle.resting_state
        return self._observation()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        if not (0 <= action < self.ACTION_DIM):
            raise ValueError(f"action must be in 0..{self.ACTION_DIM - 1}, got {action}")
        throttle, turn = DISCRETE_ACTIONS[int(action)]

        # ----- agent kinematics (mirrors render.pygame_view._update_robot_kinematics)
        raw_l = float(throttle) - float(turn)
        raw_r = float(throttle) + float(turn)
        peak = max(abs(raw_l), abs(raw_r))
        if peak > 1.0:
            raw_l /= peak
            raw_r /= peak
        target_wl = raw_l * WHEEL_MAX_SPEED
        target_wr = raw_r * WHEEL_MAX_SPEED
        self._wl = _approach(self._wl, target_wl, WHEEL_ACCEL, WHEEL_DECEL, self.DT)
        self._wr = _approach(self._wr, target_wr, WHEEL_ACCEL, WHEEL_DECEL, self.DT)
        v_lin = 0.5 * (self._wl + self._wr)
        v_ang = (self._wr - self._wl) / WHEEL_BASE_FT
        r = self.agent_robot
        r.theta = _wrap_pi(r.theta + v_ang * self.DT)
        bound = 6.0 - ROBOT_COLL_R_FT
        r.x = max(-bound, min(bound, r.x + math.cos(r.theta) * v_lin * self.DT))
        r.y = max(-bound, min(bound, r.y + math.sin(r.theta) * v_lin * self.DT))

        # ----- opponent step
        self.opp_bot.update(self.DT)

        # ----- robot-robot overlap resolve (50/50 push)
        self._resolve_robot_overlaps()

        # ----- SC4 toggle contact / agent's own back-sensor flip check
        self._update_toggle_state()

        # ----- reward
        my_color = (ToggleState.RED if self.agent_alliance == Alliance.RED
                     else ToggleState.BLUE)
        opp_color = (ToggleState.BLUE if my_color == ToggleState.RED
                      else ToggleState.RED)
        reward = 0.0
        if self.toggle.resting_state == my_color:
            reward += 1.0
        elif self.toggle.resting_state == opp_color:
            reward -= 0.2          # small penalty when opponent holds it
        # Big bonus on a fresh flip into our color
        if (self.toggle.resting_state == my_color
                and self._prev_resting != my_color):
            reward += 5.0
        self._prev_resting = self.toggle.resting_state

        # Optional shaping — dense gradient for "approach toggle, square up".
        # Small magnitudes so the +1 ownership signal still dominates.
        if self.use_shaping:
            t = self.toggle
            r = self.agent_robot
            # 1. Closeness — peaks at ~0.05/step when the bot is 1 ft away
            dx = t.x - r.x
            dy = t.y - r.y
            dist = math.hypot(dx, dy)
            reward += 0.05 * math.exp(-((dist - 1.2) ** 2) / 0.6)
            # 2. Back-alignment bonus — peaks at ~0.05/step when back direction
            # matches the wall's outward normal.
            nx, ny = _toggle_outward_normal(t)
            bx = -math.cos(r.theta)
            by = -math.sin(r.theta)
            alignment = bx * nx + by * ny
            reward += 0.05 * max(0.0, alignment)

        self._step += 1
        done = self._step >= self.episode_steps
        info = {"resting_state": self.toggle.resting_state.value,
                "live_state":    self.toggle.state.value}
        return self._observation(), reward, done, info

    # ---------- helpers ----------

    def _observation(self) -> np.ndarray:
        own = self.agent_robot
        opp = self.opp_robot
        t = self.toggle
        my_color = (ToggleState.RED if self.agent_alliance == Alliance.RED
                     else ToggleState.BLUE)
        opp_color = (ToggleState.BLUE if my_color == ToggleState.RED
                      else ToggleState.RED)
        if t.resting_state == my_color:
            rs_signed = 1.0
        elif t.resting_state == opp_color:
            rs_signed = -1.0
        else:
            rs_signed = 0.0
        live_unset = 1.0 if t.state == ToggleState.UNSET else 0.0
        obs = np.array([
            own.x / 6.0, own.y / 6.0,
            math.cos(own.theta), math.sin(own.theta),
            self._wl / WHEEL_MAX_SPEED, self._wr / WHEEL_MAX_SPEED,
            opp.x / 6.0, opp.y / 6.0,
            math.cos(opp.theta), math.sin(opp.theta),
            t.x / 6.0, t.y / 6.0,
            rs_signed, live_unset,
        ], dtype=np.float32)
        return obs

    def _resolve_robot_overlaps(self) -> None:
        r = ROBOT_COLL_R_FT
        min_d = 2.0 * r
        min_d_sq = min_d * min_d
        bound = 6.0 - r
        robots = self.world.robots
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

    def _update_toggle_state(self) -> None:
        """Agent-side toggle flip + SC4 contact resolution.

        The agent doesn't run the scripted bot logic — it learns. So if the
        agent's back sensor is on the toggle this step, we set the toggle's
        resting_state to the agent's alliance color (matching what the bot
        would do). Then SC4: while ANY robot is in contact, state=UNSET."""
        t = self.toggle
        my_color = (ToggleState.RED if self.agent_alliance == Alliance.RED
                     else ToggleState.BLUE)
        # Agent flip (scripted bots set their own toggles in their update step).
        if _back_sensor_overlaps(self.agent_robot, t):
            t.resting_state = my_color
        # SC4: state = UNSET if any robot's back is on the toggle; else resting.
        any_contact = any(_back_sensor_overlaps(r, t) for r in self.world.robots)
        t.state = ToggleState.UNSET if any_contact else t.resting_state


__all__ = ["ToggleDuelEnv", "DISCRETE_ACTIONS"]
