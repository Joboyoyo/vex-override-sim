"""Main pygame app — whole-field top-down view.

Run:
    python -m render.pygame_view
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pygame

from core.scoring import score
from core.state import (
    Alliance, Cup, Goal, GoalType, Phase, Pin, PinColor, Robot, Toggle,
    ToggleState, World, build_stack,
)

from ai import ScriptedBot
from ai.scripted_bot import MATCH_LOAD_BUDGET

from . import colors as C
from . import draw as D
from . import hud as H
from .coords import WindowLayout


# -- Constants ----------------------------------------------------------------

INITIAL_W = 1400
INITIAL_H = 900

# Match timing (per Override game manual section 2): Driver Controlled period
# is 1 minute 45 seconds. We simulate just the driver phase by default.
DRIVER_DURATION_S = 105.0

# Object selection — kind + style. Click decides what to place; user-selected style is used.
OBJECT_TYPES = [
    ("R / Y pin",      "pin", PinColor.RED,    PinColor.YELLOW, None),
    ("B / Y pin",      "pin", PinColor.BLUE,   PinColor.YELLOW, None),
    ("R / B pin",      "pin", PinColor.RED,    PinColor.BLUE,   None),
    ("Y / Y pin",      "pin", PinColor.YELLOW, PinColor.YELLOW, None),
    ("Cup (clear up)", "cup", None,            None,            True),
    ("Cup (gray up)",  "cup", None,            None,            False),
]


# -- World factories ----------------------------------------------------------


# Field collision constants (feet)
GOAL_HALF_FT = 0.35      # ~36 px goal -> half size in ft
ROBOT_COLLISION_RADIUS_FT = 0.55   # conservative robot circle

# Toggle interaction
TOGGLE_LONG_HALF_FT  = 1.0    # toggle is ~2 ft along the wall
TOGGLE_SHORT_HALF_FT = 0.30   # ~0.5 ft thick perpendicular to the wall
# Only the BACK of the robot interacts with toggles. The "back sensor" is a
# small disc mounted half-the-robot-length behind the chassis center along
# -heading. Driving over a toggle nose-first does NOT trigger SC4 or the
# cycle/set actions — you have to back into it.
BACK_SENSOR_OFFSET_FT = (14.0 / 12.0) / 2.0   # half robot length (14")
BACK_SENSOR_RADIUS_FT = 0.10                  # ~1.2" sensing radius (tight)
# The robot must be roughly perpendicular to the wall — its back direction
# has to point along the toggle's outward wall normal within this tolerance.
# Otherwise the back-mounted mechanism can't engage the toggle properly.
TOGGLE_ALIGNMENT_TOL_RAD = math.radians(25.0)
TOGGLE_ALIGNMENT_DOT_MIN = math.cos(TOGGLE_ALIGNMENT_TOL_RAD)   # ≈ 0.906


# Loader positions in ft — mounted on the field wall at x = ±6.
# Robots can drive right up to them (no separate collision hitbox; the wall
# itself bounds the chassis).
_LOADER_POSITIONS_FT = [(-6.0, 5.0), (-6.0, -5.0), (6.0, 5.0), (6.0, -5.0)]
LOADER_INTERACT_RANGE_FT = 1.4

# Physical footprints for loose scoring objects — pins and cups can be shoved
# around the field by any robot that drives into them. Conservative radii so
# the robot has to actually contact them.
PIN_PUSH_RADIUS_FT = 0.20    # ~2.4" — pins are small upright cylinders
CUP_PUSH_RADIUS_FT = 0.25    # ~3.0" — cups are slightly wider

# Optional FRONT WING for R0 (toggled with controller button 2 / X key).
# When extended, a thin rectangular push zone sits at the front bumper:
# 24" wide perpendicular to the heading, only ~3" thick along the heading.
# Pushes pins/cups within reach of any point on this rectangle. Only R0.
WING_HALF_WIDTH_FT       = (24.0 / 12.0) / 2.0   # 1.0 ft (24" total span / 2)
WING_LENGTH_FT           = 3.0 / 12.0            # 0.25 ft thick along heading
# Back edge of the wing sits exactly at the front bumper of the chassis.
WING_BACK_OFFSET_FT      = (14.0 / 12.0) / 2.0   # 0.583 ft — front bumper


from core.worlds import make_empty_world, make_demo_world      # noqa: E402


def make_toggle_duel_world() -> World:
    """Duel scenario — the REAL field (pins, cups, goals all intact) BUT
    only the Q1 (right-wall) toggle exists. Both bots are forced to fight
    for the same single objective, so the duel produces actual tactical
    combat (pushing, blocking, contested back-ins) instead of degenerating
    into "race to the closest toggle, never interact."

    Also reduced to 1v1 (R0 red, R2 blue). Partner bots in a 4-toggle
    world tended to spread out and cycle; with one toggle and two robots,
    every encounter matters."""
    w = make_empty_world()
    w.phase = Phase.DRIVER
    # Keep only Q1
    w.toggles = [t for t in w.toggles if t.quadrant == 1]
    # 1v1
    w.robots = [r for r in w.robots if r.id in (0, 2)]
    return w


# -- App ----------------------------------------------------------------------


class App:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Override Sim — Field View")
        self.win_w, self.win_h = INITIAL_W, INITIAL_H
        self.is_fullscreen = False
        self.screen = pygame.display.set_mode(
            (self.win_w, self.win_h), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()

        self.layout = WindowLayout.compute(self.win_w, self.win_h)

        # World + UI state
        self.world = make_empty_world()
        self.current_object_idx = 0
        self.auto_red: int | None = None
        self.auto_blue: int | None = None

        # Joystick / controller (VEX V5 controller, Xbox, PS, etc.)
        pygame.joystick.init()
        self.joysticks: list[pygame.joystick.JoystickType] = []
        for i in range(pygame.joystick.get_count()):
            j = pygame.joystick.Joystick(i)
            j.init()
            self.joysticks.append(j)
        if self.joysticks:
            j0 = self.joysticks[0]
            self.status = (f"Controller: {j0.get_name()}  "
                           f"({j0.get_numaxes()} axes, "
                           f"{j0.get_numbuttons()} buttons). "
                           f"Left stick = drive, right stick X = turn, "
                           f"button {self.JOY_PICKUP_BUTTON} = pickup.")
        else:
            self.status = ("No controller detected. WASD/arrows to drive R0, "
                           "E to pickup/drop.")

        # Hover tracking
        self.hovered_goal_id: int | None = None
        self.hovered_toggle_id: int | None = None
        self.hovered_pin_id: int | None = None
        self.hovered_cup_id: int | None = None
        self.hovered_robot_id: int | None = None

        # Hit-test rects from last render (for click handling)
        self._stack_hits: list[tuple[int, str, int, pygame.Rect]] = []

        # Per-side wheel velocity state — persists between frames so accel/decel
        # work, and so linear + angular velocity fall out of the same physics.
        self._wheel_v_left: float = 0.0    # ft/s, signed (+ forward)
        self._wheel_v_right: float = 0.0   # ft/s, signed (+ forward)

        # Joystick curve t — adjustable at runtime via [ ] (D-pad ↑/↓)
        self.joy_curve_t: float = self.JOY_CURVE_T
        # Turn sensitivity scaler — adjustable via = / − (D-pad ← / →)
        self.turn_sensitivity: float = self.TURN_SENSITIVITY

        # AI bots — full 2v2:
        #   R0 = YOU (red, human-driven)
        #   R1 = your RED PARTNER bot (helps you load + flips toggles to RED)
        #   R2 = opposing BLUE bot
        #   R3 = opposing BLUE bot
        # All three bots follow the same ScriptedBot logic but with their own
        # alliance. Toggle all bots on/off with B.
        self.bots: list[ScriptedBot] = [
            ScriptedBot(robot_id=1, world=self.world,
                         collides_at=self._make_collides_for(1)),
            ScriptedBot(robot_id=2, world=self.world,
                         collides_at=self._make_collides_for(2)),
            ScriptedBot(robot_id=3, world=self.world,
                         collides_at=self._make_collides_for(3)),
        ]

        # Trigger-as-axis edge detection (so analog R2 only fires pickup once
        # when crossing the threshold, not every frame it's held).
        self._trigger_held: bool = False

        # R0 front-wing state — when True, the player robot has a 24"-wide
        # push zone at its front bumper (lets you sweep multiple pins/cups in
        # a single drive-by). Toggled with controller button 2 / X key.
        self.wings_extended: bool = False

        # AI-vs-AI toggle-duel mode. When True: world is replaced by a minimal
        # one-toggle world, the field renders zoomed in on the right quadrant,
        # both visible robots are bots, and player input is ignored.
        self.duel_mode: bool = False

    # -- Layout recompute on resize / fullscreen toggle --

    def _recompute_layout(self) -> None:
        self.win_w, self.win_h = self.screen.get_size()
        self.layout = WindowLayout.compute(self.win_w, self.win_h)
        # Duel mode no longer zooms — the full field is the game, and the
        # bots fight for ALL four toggles on it, not just one. Matches the
        # training environment exactly.

    # -- Main loop --

    def run(self) -> None:
        running = True
        last_t = pygame.time.get_ticks() / 1000.0
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    self.screen = pygame.display.set_mode(
                        (event.w, event.h), pygame.RESIZABLE)
                    self._recompute_layout()
                elif event.type == pygame.KEYDOWN:
                    if not self._handle_key(event):
                        running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_click(event)
                elif event.type == pygame.JOYBUTTONDOWN:
                    # Always echo the index so the user can re-bind if needed
                    if event.button == self.JOY_PICKUP_BUTTON:    role = " (PICKUP)"
                    elif event.button == self.JOY_WING_BUTTON:    role = " (WING)"
                    elif event.button == self.JOY_TOGGLE_BUTTON:  role = " (TOGGLE)"
                    else: role = (f"  (pickup=#{self.JOY_PICKUP_BUTTON} "
                                   f"wing=#{self.JOY_WING_BUTTON} "
                                   f"toggle=#{self.JOY_TOGGLE_BUTTON})")
                    self.status = f"Joystick button #{event.button} pressed{role}"
                    if event.button == self.JOY_PICKUP_BUTTON:
                        self._try_interact()
                    elif event.button == self.JOY_WING_BUTTON:
                        self._toggle_wings()
                    elif event.button == self.JOY_TOGGLE_BUTTON:
                        self._cycle_nearest_toggle()
                elif event.type == pygame.JOYHATMOTION:
                    # D-pad ↑↓ adjusts curve t. D-pad ←→ adjusts turn sens.
                    if event.value[1] > 0:
                        self._adjust_curve_t(+1.0)
                    elif event.value[1] < 0:
                        self._adjust_curve_t(-1.0)
                    if event.value[0] > 0:
                        self._adjust_turn_sensitivity(+0.05)
                    elif event.value[0] < 0:
                        self._adjust_turn_sensitivity(-0.05)
                elif event.type == pygame.JOYDEVICEADDED:
                    j = pygame.joystick.Joystick(event.device_index)
                    j.init()
                    self.joysticks.append(j)
                    self.status = f"Controller connected: {j.get_name()}"
                elif event.type == pygame.JOYDEVICEREMOVED:
                    self.joysticks = [j for j in self.joysticks
                                      if j.get_instance_id() != event.instance_id]
                    self.status = "Controller disconnected."

            # Tick robot kinematics from held keys
            now_t = pygame.time.get_ticks() / 1000.0
            dt = max(0.0, min(0.05, now_t - last_t))
            last_t = now_t
            self._poll_analog_trigger()
            # Player drives R0 — except in duel mode, where R0 is also an AI.
            if not self.duel_mode:
                self._update_robot_kinematics(dt)
            # Tick AI bots
            for bot in self.bots:
                bot.update(dt)
            # After all motion: push apart overlapping robots (player shoves bots)
            self._resolve_robot_overlaps()
            # Then displace any loose pins/cups that the robots are touching
            self._push_loose_objects()
            # Per-frame: enforce SC4 (toggle UNSET while any robot touches it)
            self._update_toggle_contact()
            # Per-frame: tick the match clock during the driver period
            self._tick_match_clock(dt)

            self._update_hover(pygame.mouse.get_pos())
            self._render()
            self.clock.tick(60)

        pygame.quit()

    # -- Robot control -------------------------------------------------------

    # Arcade-drive input → tank-drive physics:
    #   left_cmd  = throttle − turn
    #   right_cmd = throttle + turn
    # Both clamped together (scaled down if either |side| > 1) so a move + turn
    # never exceeds 100% on either wheel. Then per-side targets * WHEEL_MAX_SPEED
    # become the wheel velocity targets, ramped via accel/decel, and the body's
    # v_lin / v_ang fall out of differential-drive kinematics.
    WHEEL_MAX_SPEED   = 6.0
    WHEEL_ACCEL       = 22.0
    WHEEL_DECEL       = 32.0
    ROBOT_WHEEL_BASE_FT = 10.0 / 12.0

    # Joystick input curve (VEX-style expo). Adjust at runtime with [ / ].
    JOY_CURVE_T       = 2.0

    # Linear scaling on the turn axis AFTER the expo curve.
    # Adjust at runtime with = / − or D-pad ← / →.
    TURN_SENSITIVITY  = 0.7

    # Joystick axes for arcade drive
    JOY_AXIS_THROTTLE = 1     # left stick Y (we invert: up = forward)
    JOY_AXIS_TURN     = 2     # right stick X
    JOY_DEADZONE      = 0.08

    # Button bindings — change if your controller maps differently
    # (every press prints its index to the status bar so you can verify).
    JOY_PICKUP_BUTTON = 7    # R2: pickup / drop / place / loader-spawn
    JOY_TOGGLE_BUTTON = 0    # A:  cycle nearest toggle (Y -> R -> B -> Y)
    JOY_WING_BUTTON   = 2    # X:  extend / retract front wing (24" push zone)

    # Optional fallback: many HID controllers report L2/R2 as ANALOG TRIGGERS
    # on an axis instead of as a button. If JOY_TRIGGER_AXIS is set, crossing
    # JOY_TRIGGER_THRESHOLD upward fires pickup as well.
    JOY_TRIGGER_AXIS      = 5     # common Xbox-style right-trigger axis
    JOY_TRIGGER_THRESHOLD = 0.4

    def _poll_analog_trigger(self) -> None:
        """If R2 is reported as an analog axis instead of a button, fire the
        pickup on the upward threshold crossing only (debounced via
        self._trigger_held)."""
        if not self.joysticks or self.JOY_TRIGGER_AXIS is None:
            return
        j = self.joysticks[0]
        if j.get_numaxes() <= self.JOY_TRIGGER_AXIS:
            return
        try:
            v = j.get_axis(self.JOY_TRIGGER_AXIS)
        except pygame.error:
            return
        # Xbox/SDL2 trigger idle = -1, fully pressed = +1.
        # Some other layouts use 0..1. Both work with a +0.4 threshold.
        pressed = v > self.JOY_TRIGGER_THRESHOLD
        if pressed and not self._trigger_held:
            self.status = (f"Trigger axis #{self.JOY_TRIGGER_AXIS} crossed "
                           f"{self.JOY_TRIGGER_THRESHOLD:+.2f} (value {v:+.2f}) → pickup")
            self._try_interact()
        self._trigger_held = pressed

    @staticmethod
    def _expo_curve(input_norm: float, t: float) -> float:
        """VEX-style expo joystick curve, parameterized by `t`.

        Original formula (input in [-100, 100], output in [-100, 100]):
            ( exp(-t/10) + exp((|in|-100)/10) * (1 - exp(-t/10)) ) * in

        Properties (for sensible t):
          - f(0)   = 0
          - f(±100) = ±100         (saturates exactly at max)
          - Small inputs are heavily compressed (dead-feel near zero)
          - Large inputs reach full output

        We accept and return values in [-1, 1] (pygame's scale). The math
        is identical since the scale factor cancels out.
        """
        if input_norm == 0.0:
            return 0.0
        abs_scaled = abs(input_norm) * 100.0
        w_lin = math.exp(-t / 10.0)
        w_expo = math.exp((abs_scaled - 100.0) / 10.0) * (1.0 - w_lin)
        return (w_lin + w_expo) * input_norm

    def _read_drive_input(self) -> tuple[float, float]:
        """Return ARCADE-DRIVE input (throttle, turn), each in [-1, 1].
        Joystick passes through `_expo_curve(t=JOY_CURVE_T)` after deadzone.
        Keyboard (WASD/arrows) overrides joystick when held; keyboard inputs
        are binary so the curve doesn't apply meaningfully (full-strength
        equivalent to curve(±1) = ±1)."""
        throttle = 0.0
        turn = 0.0

        if self.joysticks:
            j = self.joysticks[0]
            try:
                if j.get_numaxes() > self.JOY_AXIS_THROTTLE:
                    raw_t = -j.get_axis(self.JOY_AXIS_THROTTLE)  # up = forward
                    if abs(raw_t) > self.JOY_DEADZONE:
                        # Throttle stays LINEAR (no curve) per user preference.
                        throttle = raw_t
                if j.get_numaxes() > self.JOY_AXIS_TURN:
                    raw_q = j.get_axis(self.JOY_AXIS_TURN)
                    if abs(raw_q) > self.JOY_DEADZONE:
                        # Curve only the turn axis.
                        turn = self._expo_curve(-raw_q, self.joy_curve_t)
            except pygame.error:
                pass

        keys = pygame.key.get_pressed()
        kb_t = 0
        kb_q = 0
        if keys[pygame.K_w] or keys[pygame.K_UP]:    kb_t += 1
        if keys[pygame.K_s] or keys[pygame.K_DOWN]:  kb_t -= 1
        if keys[pygame.K_a] or keys[pygame.K_LEFT]:  kb_q += 1
        if keys[pygame.K_d] or keys[pygame.K_RIGHT]: kb_q -= 1
        if kb_t != 0:
            throttle = float(kb_t)
        if kb_q != 0:
            turn = float(kb_q)

        return throttle, turn

    @staticmethod
    def _approach(current: float, target: float,
                  accel: float, decel: float, dt: float) -> float:
        """Ramp `current` toward `target`. Uses `accel` when the magnitude is
        growing in the same direction, `decel` when it's shrinking or
        reversing — gives a firmer brake than acceleration."""
        diff = target - current
        if diff == 0.0:
            return current
        # Accelerating only if signs match AND we're moving away from zero.
        if target * current >= 0.0 and abs(target) > abs(current):
            rate = accel
        else:
            rate = decel
        max_change = rate * dt
        if abs(diff) <= max_change:
            return target
        return current + (max_change if diff > 0 else -max_change)

    def _update_robot_kinematics(self, dt: float) -> None:
        """Arcade-drive input → tank-physics. Pipeline:
          1. Read (throttle, turn) from joystick/keyboard.
          2. Convert arcade to per-side targets:
               left  = throttle − turn
               right = throttle + turn
          3. CAP TO 100%: if either |side| > 1, scale both down so the bigger
             one is exactly 1. (Prevents the 200% problem when move + turn
             together would exceed the chassis max.)
          4. Multiply by WHEEL_MAX_SPEED → target wheel velocities (ft/s).
          5. Ramp current wheel velocities toward those targets (accel/decel).
          6. Differential-drive output: v_lin = (vR + vL) / 2,
             v_ang = (vR − vL) / wheel_base."""
        if dt <= 0.0 or not self.world.robots:
            return

        throttle, turn = self._read_drive_input()
        turn = turn * self.turn_sensitivity     # scale down per-axis sensitivity

        raw_left  = throttle - turn
        raw_right = throttle + turn

        # Cap to ±1 by scaling both proportionally
        peak = max(abs(raw_left), abs(raw_right))
        if peak > 1.0:
            raw_left  /= peak
            raw_right /= peak

        target_wl = raw_left  * self.WHEEL_MAX_SPEED
        target_wr = raw_right * self.WHEEL_MAX_SPEED

        self._wheel_v_left = self._approach(
            self._wheel_v_left, target_wl,
            self.WHEEL_ACCEL, self.WHEEL_DECEL, dt,
        )
        self._wheel_v_right = self._approach(
            self._wheel_v_right, target_wr,
            self.WHEEL_ACCEL, self.WHEEL_DECEL, dt,
        )

        if abs(self._wheel_v_left) < 1e-4 and abs(self._wheel_v_right) < 1e-4:
            self._wheel_v_left = 0.0
            self._wheel_v_right = 0.0
            return

        # Differential-drive kinematics
        v_lin = 0.5 * (self._wheel_v_left + self._wheel_v_right)
        v_ang = (self._wheel_v_right - self._wheel_v_left) / self.ROBOT_WHEEL_BASE_FT

        robot = self.world.robots[0]

        # Apply angular velocity (no collision check)
        robot.theta += v_ang * dt
        robot.theta = (robot.theta + math.pi) % (2 * math.pi) - math.pi

        # Apply linear velocity along the new heading
        dx = math.cos(robot.theta) * v_lin * dt
        dy = math.sin(robot.theta) * v_lin * dt

        bound = 6.0 - ROBOT_COLLISION_RADIUS_FT
        blocked_x = False
        blocked_y = False

        try_x = max(-bound, min(bound, robot.x + dx))
        # Player drives THROUGH other robots (pushes them aside afterward);
        # only goal hitboxes block player motion.
        if not self._collides_at(robot.id, try_x, robot.y, include_robots=False):
            robot.x = try_x
        else:
            blocked_x = True
        try_y = max(-bound, min(bound, robot.y + dy))
        if not self._collides_at(robot.id, robot.x, try_y, include_robots=False):
            robot.y = try_y
        else:
            blocked_y = True

        # If both axes blocked (crashed into a corner), kill both wheels so
        # the chassis doesn't keep "stalling" against the wall.
        if blocked_x and blocked_y:
            self._wheel_v_left = 0.0
            self._wheel_v_right = 0.0

    def _collides_at(self, robot_id: int, x: float, y: float,
                      include_robots: bool = True) -> bool:
        """Would robot `robot_id` at center (x, y) overlap any obstacle?
        Checked obstacles: every goal (AABB), and OPTIONALLY every other robot
        (circle). The player drives through bots (include_robots=False) and
        bots get shoved aside afterward in `_resolve_robot_overlaps`.

        NO-WORSE-THAN-CURRENT escape rule: if the robot is currently overlapping
        an obstacle (e.g. it got shoved onto a goal by another robot in the
        previous frame's resolve step), a candidate move is allowed as long as
        it doesn't deepen the overlap with that specific obstacle. Without this,
        a shoved-into-goal robot would be blocked on both axes, kill its wheels,
        and sit there forever."""
        r = ROBOT_COLLISION_RADIUS_FT
        r2 = r * r

        # Locate this robot's current position (for the "is the new pose worse
        # than where we already are?" comparison).
        cur = next((rr for rr in self.world.robots if rr.id == robot_id), None)

        for goal in self.world.goals:
            cxn = max(goal.x - GOAL_HALF_FT, min(x, goal.x + GOAL_HALF_FT))
            cyn = max(goal.y - GOAL_HALF_FT, min(y, goal.y + GOAL_HALF_FT))
            new_d2 = (x - cxn) * (x - cxn) + (y - cyn) * (y - cyn)
            if new_d2 >= r2:
                continue  # clear of this goal at the candidate position
            # Candidate position overlaps this goal. Allow the move only if
            # the robot is ALREADY inside this goal and the new pose is at
            # least as far from the goal as the current pose.
            if cur is not None:
                cxc = max(goal.x - GOAL_HALF_FT, min(cur.x, goal.x + GOAL_HALF_FT))
                cyc = max(goal.y - GOAL_HALF_FT, min(cur.y, goal.y + GOAL_HALF_FT))
                cur_d2 = (cur.x - cxc) * (cur.x - cxc) + (cur.y - cyc) * (cur.y - cyc)
                if cur_d2 < r2 and new_d2 >= cur_d2:
                    continue  # escape mode — no deeper than where we are
            return True

        if include_robots:
            rr_sum_sq = (2 * r) * (2 * r)
            for other in self.world.robots:
                if other.id == robot_id:
                    continue
                ddx = other.x - x
                ddy = other.y - y
                new_d2 = ddx * ddx + ddy * ddy
                if new_d2 >= rr_sum_sq:
                    continue
                if cur is not None:
                    cdx = other.x - cur.x
                    cdy = other.y - cur.y
                    cur_d2 = cdx * cdx + cdy * cdy
                    if cur_d2 < rr_sum_sq and new_d2 >= cur_d2:
                        continue
                return True
        return False

    def _make_collides_for(self, robot_id: int):
        """Return a (x, y) -> bool closure bound to a specific robot_id.
        Robot-vs-robot collision is handled in the post-motion resolution
        step (which splits the push 50/50), so the bot's per-step check
        only blocks on STATIC obstacles (goals)."""
        return lambda x, y: self._collides_at(robot_id, x, y, include_robots=False)

    def _resolve_robot_overlaps(self) -> None:
        """After all motion this frame, push any overlapping robots apart
        along the line connecting their centers. Split is 50/50 — either
        robot can push the other. Static obstacles (goals) are NOT included
        here (they're enforced per-step via `_collides_at`)."""
        r = ROBOT_COLLISION_RADIUS_FT
        min_dist = 2.0 * r
        min_dist_sq = min_dist * min_dist
        bound = 6.0 - r
        robots = self.world.robots
        for i in range(len(robots)):
            a = robots[i]
            for j in range(i + 1, len(robots)):
                b = robots[j]
                dx = b.x - a.x
                dy = b.y - a.y
                d2 = dx * dx + dy * dy
                if d2 >= min_dist_sq:
                    continue
                if d2 < 1e-9:
                    ux, uy = 1.0, 0.0
                    overlap = min_dist
                else:
                    d = math.sqrt(d2)
                    overlap = min_dist - d
                    ux = dx / d
                    uy = dy / d
                # 50/50 split — either robot can shove the other
                half = overlap * 0.5
                a.x = max(-bound, min(bound, a.x - ux * half))
                a.y = max(-bound, min(bound, a.y - uy * half))
                b.x = max(-bound, min(bound, b.x + ux * half))
                b.y = max(-bound, min(bound, b.y + uy * half))

    def _push_loose_objects(self) -> None:
        """Robots can shove loose pins/cups around the field. "Loose" means
        the object is not nested in a goal/cup and not held by a robot.
        For each robot×object pair that overlaps, the object snaps to the
        edge of the robot's circle along the line connecting their centers.

        Snapping (rather than impulse integration) gives a sticky "you can't
        drive through a pin" feel that scales with frame rate without tuning.

        When R0 has its FRONT WING deployed, an extra RECTANGULAR push zone
        is attached to its front bumper: 24" wide perpendicular to heading,
        ~3" thick along heading. Pins/cups touching the rectangle get snapped
        to just outside it (closest-point + object_radius along the outward
        normal). The thin rectangle hugs the bumper so it doesn't push from
        unexpected distances."""
        held_pin_ids = {r.holding_pin_id for r in self.world.robots
                        if r.holding_pin_id is not None}
        held_cup_ids = {r.holding_cup_id for r in self.world.robots
                        if r.holding_cup_id is not None}
        R = ROBOT_COLLISION_RADIUS_FT
        bound_pin = 6.0 - PIN_PUSH_RADIUS_FT - 0.02
        bound_cup = 6.0 - CUP_PUSH_RADIUS_FT - 0.02

        # Pre-compute R0's wing rectangle once per frame (if deployed).
        # Stored as (cos, sin, robot center, rect bounds in body frame).
        wing_ctx = None
        if self.wings_extended and self.world.robots:
            r0 = self.world.robots[0]
            wing_ctx = (
                math.cos(r0.theta), math.sin(r0.theta),
                r0.x, r0.y,
                WING_BACK_OFFSET_FT,                          # xmin (body frame)
                WING_BACK_OFFSET_FT + WING_LENGTH_FT,         # xmax
                -WING_HALF_WIDTH_FT,                          # ymin
                +WING_HALF_WIDTH_FT,                          # ymax
            )

        def _push_against_circle(cx: float, cy: float, push_r: float,
                                  obj, obj_radius: float, bound: float) -> None:
            min_d = push_r + obj_radius
            dx = obj.x - cx
            dy = obj.y - cy
            d2 = dx * dx + dy * dy
            if d2 >= min_d * min_d:
                return
            if d2 < 1e-9:
                ux, uy = 1.0, 0.0
            else:
                d = math.sqrt(d2)
                ux = dx / d
                uy = dy / d
            new_x = cx + ux * min_d
            new_y = cy + uy * min_d
            obj.x = max(-bound, min(bound, new_x))
            obj.y = max(-bound, min(bound, new_y))

        def _push_against_wing(obj, obj_radius: float, bound: float) -> None:
            if wing_ctx is None:
                return
            cos_t, sin_t, rx, ry, xmin, xmax, ymin, ymax = wing_ctx
            # Transform object center into the robot's body frame
            dx = obj.x - rx
            dy = obj.y - ry
            bx =  dx * cos_t + dy * sin_t
            by = -dx * sin_t + dy * cos_t
            # Front-only gate: ignore objects whose CENTER is behind the
            # wing's back edge. Those are alongside or behind the chassis
            # and belong to the body-circle's job, not the wing's.
            if bx < xmin:
                return
            # Closest point on the wing rect (in body frame)
            qx = max(xmin, min(bx, xmax))
            qy = max(ymin, min(by, ymax))
            ndx = bx - qx
            ndy = by - qy
            d2 = ndx * ndx + ndy * ndy
            if d2 >= obj_radius * obj_radius:
                return                  # object is clear of the wing
            if d2 < 1e-9:
                # Object is INSIDE the rect — push out the closest edge.
                # Pick whichever edge (top/bottom/front/back) we're nearest to.
                gaps = [
                    ("front", xmax - bx, ( 1.0,  0.0)),
                    ("back",  bx - xmin, (-1.0,  0.0)),
                    ("top",   ymax - by, ( 0.0,  1.0)),
                    ("bot",   by - ymin, ( 0.0, -1.0)),
                ]
                _, _, (ux, uy) = min(gaps, key=lambda g: g[1])
                # Snap the object to that edge plus obj_radius along the normal
                if ux != 0.0:    # front/back edge
                    new_bx = (xmax if ux > 0 else xmin) + ux * obj_radius
                    new_by = by
                else:            # top/bottom edge
                    new_bx = bx
                    new_by = (ymax if uy > 0 else ymin) + uy * obj_radius
            else:
                d = math.sqrt(d2)
                ux = ndx / d
                uy = ndy / d
                new_bx = qx + ux * obj_radius
                new_by = qy + uy * obj_radius
            # Body frame → world frame
            new_dx = new_bx * cos_t - new_by * sin_t
            new_dy = new_bx * sin_t + new_by * cos_t
            obj.x = max(-bound, min(bound, rx + new_dx))
            obj.y = max(-bound, min(bound, ry + new_dy))

        def _push(obj, obj_radius: float, bound: float) -> None:
            for robot in self.world.robots:
                _push_against_circle(robot.x, robot.y, R,
                                      obj, obj_radius, bound)
            _push_against_wing(obj, obj_radius, bound)

        for pin in self.world.pins:
            if pin.in_goal is not None or pin.in_cup is not None:
                continue
            if pin.id in held_pin_ids:
                continue
            _push(pin, PIN_PUSH_RADIUS_FT, bound_pin)
        for cup in self.world.cups:
            if cup.in_goal is not None or cup.on_pin is not None:
                continue
            if cup.id in held_cup_ids:
                continue
            _push(cup, CUP_PUSH_RADIUS_FT, bound_cup)

    @staticmethod
    def _toggle_aabb(toggle) -> tuple[float, float, float, float]:
        """Return (x_min, x_max, y_min, y_max) for the toggle's physical bbox.
        Long axis runs along the wall the toggle is mounted on."""
        if abs(toggle.x) > abs(toggle.y):
            # Left or right wall -> long axis vertical
            return (toggle.x - TOGGLE_SHORT_HALF_FT, toggle.x + TOGGLE_SHORT_HALF_FT,
                    toggle.y - TOGGLE_LONG_HALF_FT,  toggle.y + TOGGLE_LONG_HALF_FT)
        else:
            # Top or bottom wall -> long axis horizontal
            return (toggle.x - TOGGLE_LONG_HALF_FT,  toggle.x + TOGGLE_LONG_HALF_FT,
                    toggle.y - TOGGLE_SHORT_HALF_FT, toggle.y + TOGGLE_SHORT_HALF_FT)

    @staticmethod
    def _back_sensor_point(robot: Robot) -> tuple[float, float]:
        """Position of the robot's back-mounted toggle sensor — a small disc
        centered BACK_SENSOR_OFFSET_FT behind the chassis center along the
        -heading axis."""
        return (
            robot.x - math.cos(robot.theta) * BACK_SENSOR_OFFSET_FT,
            robot.y - math.sin(robot.theta) * BACK_SENSOR_OFFSET_FT,
        )

    @staticmethod
    def _toggle_outward_normal(toggle) -> tuple[float, float]:
        """Unit vector pointing OUTWARD from the wall this toggle is mounted
        on (away from field center). The robot's BACK direction must align
        with this to engage the toggle."""
        if abs(toggle.x) > abs(toggle.y):
            return (1.0 if toggle.x > 0 else -1.0, 0.0)
        return (0.0, 1.0 if toggle.y > 0 else -1.0)

    def _aligned_to_toggle(self, robot: Robot, toggle) -> bool:
        """Robot's back direction (-heading) must point along the toggle's
        outward wall normal within TOGGLE_ALIGNMENT_TOL_RAD."""
        nx, ny = self._toggle_outward_normal(toggle)
        bx = -math.cos(robot.theta)
        by = -math.sin(robot.theta)
        return (bx * nx + by * ny) >= TOGGLE_ALIGNMENT_DOT_MIN

    def _back_sensor_overlaps(self, robot: Robot, toggle) -> bool:
        """Does this robot's back sensor disc overlap the toggle AABB AND
        the robot's heading point roughly perpendicular to the wall? Both
        conditions are required — sitting on the toggle at an angle does
        NOT count as engagement."""
        if not self._aligned_to_toggle(robot, toggle):
            return False
        x_min, x_max, y_min, y_max = self._toggle_aabb(toggle)
        sx, sy = self._back_sensor_point(robot)
        cx_ = max(x_min, min(sx, x_max))
        cy_ = max(y_min, min(sy, y_max))
        ddx = sx - cx_
        ddy = sy - cy_
        return ddx * ddx + ddy * ddy < BACK_SENSOR_RADIUS_FT * BACK_SENSOR_RADIUS_FT

    def _any_robot_touching(self, toggle) -> bool:
        """Disc-vs-AABB between any robot's BACK sensor and the toggle bbox.
        Only the back of the robot interacts with toggles (per the player's
        request to model a back-mounted mechanism)."""
        for robot in self.world.robots:
            if self._back_sensor_overlaps(robot, toggle):
                return True
        return False

    def _update_toggle_contact(self) -> None:
        """SC4: while any robot is in contact, toggle.state = UNSET. When the
        last robot leaves, the toggle settles back into its resting_state."""
        for tog in self.world.toggles:
            if self._any_robot_touching(tog):
                tog.state = ToggleState.UNSET
            else:
                tog.state = tog.resting_state

    # Distance from robot front to a goal center within which "drop" becomes
    # "place in stack" (SC2-compliant).
    GOAL_PLACE_RANGE_FT = 1.5

    def _nearest_loader_within(self, x: float, y: float,
                                range_ft: float) -> tuple[float, float] | None:
        """Closest loader (x, y) within range_ft of (x, y), or None."""
        best = None
        best_d2 = range_ft * range_ft
        for lx, ly in _LOADER_POSITIONS_FT:
            ddx = lx - x
            ddy = ly - y
            d2 = ddx * ddx + ddy * ddy
            if d2 < best_d2:
                best_d2 = d2
                best = (lx, ly)
        return best

    def _spawn_from_loader(self, loader_xy: tuple[float, float]) -> None:
        """Spawn the pre-loaded configuration:
          - cup with opaque side DOWN (clear_face_up=True)
          - pin with RED side DOWN (half_a=YELLOW on top, half_b=RED on bottom)
        Counts against the RED alliance's match-load budget (shared with R1
        partner bot). Refuses to spawn if budget is exhausted."""
        if self.world.match_loads_red >= MATCH_LOAD_BUDGET:
            self.status = (f"RED alliance match-load budget "
                           f"({MATCH_LOAD_BUDGET}) exhausted.")
            return

        robot = self.world.robots[0]
        next_pid = max((p.id for p in self.world.pins), default=-1) + 1
        next_cid = max((c.id for c in self.world.cups), default=-1) + 1
        spawn_x, spawn_y = loader_xy

        new_cup = Cup(
            id=next_cid, x=spawn_x, y=spawn_y,
            in_goal=None, on_pin=None,
            clear_face_up=True,
        )
        new_pin = Pin(
            id=next_pid,
            half_a_color=PinColor.YELLOW,
            half_b_color=PinColor.RED,
            x=spawn_x, y=spawn_y,
            in_goal=None, in_cup=None,
        )
        self.world.cups.append(new_cup)
        self.world.pins.append(new_pin)
        robot.holding_pin_id = new_pin.id
        robot.holding_cup_id = new_cup.id
        self.world.match_loads_red += 1
        self.status = (f"Loaded cup#{new_cup.id} + pin#{new_pin.id}  "
                       f"(RED match loads: {self.world.match_loads_red}/"
                       f"{MATCH_LOAD_BUDGET})")

    def _try_interact(self) -> None:
        """E key / R2:
          - If holding items: place into nearby goal (SC2) or drop on floor.
          - Else if near a loader: spawn a pre-stacked cup+pin and pick up.
          - Else if near a loose item: pick it up."""
        robot = self.world.robots[0]
        front_x = robot.x + math.cos(robot.theta) * 0.6
        front_y = robot.y + math.sin(robot.theta) * 0.6

        # If holding anything, place / drop
        if robot.holding_pin_id is not None or robot.holding_cup_id is not None:
            placed_msgs: list[str] = []
            goal = self._nearest_goal_within(front_x, front_y,
                                              self.GOAL_PLACE_RANGE_FT)
            if goal is not None:
                placed_msgs = self._place_held_into_goal(goal.id)

            # Anything still held? drop it on the floor.
            dropped: list[str] = []
            if robot.holding_pin_id is not None:
                pin = next(p for p in self.world.pins if p.id == robot.holding_pin_id)
                pin.x, pin.y = front_x, front_y
                dropped.append(f"pin#{pin.id}")
                robot.holding_pin_id = None
            if robot.holding_cup_id is not None:
                cup = next(c for c in self.world.cups if c.id == robot.holding_cup_id)
                cup.x, cup.y = front_x, front_y
                dropped.append(f"cup#{cup.id}")
                robot.holding_cup_id = None

            if placed_msgs and dropped:
                self.status = (f"Placed {', '.join(placed_msgs)}. "
                                f"Dropped {', '.join(dropped)} on floor.")
            elif placed_msgs:
                self.status = f"Placed {', '.join(placed_msgs)}."
            elif dropped:
                where = (f" — G{goal.id} stack needed something else"
                         if goal is not None else "")
                self.status = f"Dropped {', '.join(dropped)}{where}."
            return

        # Holding nothing → try loader first, then loose items
        loader_xy = self._nearest_loader_within(front_x, front_y,
                                                  LOADER_INTERACT_RANGE_FT)
        if loader_xy is not None:
            self._spawn_from_loader(loader_xy)
            return

        # Find the nearest loose pin AND the nearest loose cup independently.
        # If they share a position (pin-on-cup "set"), grab both at once just
        # like a loader spawns a pin+cup pair.
        pickup_range_sq = 0.7 * 0.7
        nearest_pin = None
        nearest_pin_d2 = pickup_range_sq
        for pin in self.world.pins:
            if pin.in_goal is not None or pin.in_cup is not None:
                continue
            ddx = pin.x - front_x
            ddy = pin.y - front_y
            d2 = ddx * ddx + ddy * ddy
            if d2 < nearest_pin_d2:
                nearest_pin_d2 = d2
                nearest_pin = pin
        nearest_cup = None
        nearest_cup_d2 = pickup_range_sq
        for cup in self.world.cups:
            if cup.in_goal is not None or cup.on_pin is not None:
                continue
            ddx = cup.x - front_x
            ddy = cup.y - front_y
            d2 = ddx * ddx + ddy * ddy
            if d2 < nearest_cup_d2:
                nearest_cup_d2 = d2
                nearest_cup = cup

        # If the nearest pin and nearest cup share a position (within ~1 inch),
        # treat them as a pin-on-cup SET and grab both — same semantics as a
        # match-load from a loader.
        if nearest_pin is not None and nearest_cup is not None:
            sep_d2 = ((nearest_pin.x - nearest_cup.x) ** 2 +
                       (nearest_pin.y - nearest_cup.y) ** 2)
            if sep_d2 < 0.01:    # < 0.1 ft — visually nested
                robot.holding_pin_id = nearest_pin.id
                robot.holding_cup_id = nearest_cup.id
                self.status = (f"Picked up pin#{nearest_pin.id} + "
                                f"cup#{nearest_cup.id} (pin-on-cup set).")
                return

        if nearest_pin is None and nearest_cup is None:
            self.status = "Nothing in pickup range (drive closer to a loose pin/cup)."
            return

        # Otherwise grab whichever one is closer (single-object pickup)
        if nearest_pin is not None and (
                nearest_cup is None or nearest_pin_d2 <= nearest_cup_d2):
            robot.holding_pin_id = nearest_pin.id
            self.status = f"Picked up pin#{nearest_pin.id}"
            return
        if nearest_cup is not None:
            robot.holding_cup_id = nearest_cup.id
            self.status = f"Picked up cup#{nearest_cup.id}"

    def _nearest_goal_within(self, x: float, y: float,
                              range_ft: float) -> Goal | None:
        """Closest goal to (x, y) within `range_ft` (or None)."""
        best = None
        best_d2 = range_ft * range_ft
        for goal in self.world.goals:
            ddx = goal.x - x
            ddy = goal.y - y
            d2 = ddx * ddx + ddy * ddy
            if d2 < best_d2:
                best_d2 = d2
                best = goal
        return best

    def _place_held_into_goal(self, goal_id: int) -> list[str]:
        """Place currently-held items into the goal's stack per SC2.
        Up to two iterations so a held pin + held cup can both be placed
        (pin → forced cup → pin sequence, etc.). Returns list of label strings
        for status output."""
        robot = self.world.robots[0]
        placed: list[str] = []
        for _ in range(2):
            stack = self.world.stack_in_goal(goal_id)
            next_kind = self.world.next_placeable_kind(goal_id)
            if next_kind == "pin" and robot.holding_pin_id is not None:
                pin = next(p for p in self.world.pins
                            if p.id == robot.holding_pin_id)
                pin.in_goal = goal_id
                pin.in_cup = stack[-1][1].id if stack else None
                placed.append(f"pin#{pin.id} → G{goal_id}")
                robot.holding_pin_id = None
            elif next_kind == "cup" and robot.holding_cup_id is not None:
                cup = next(c for c in self.world.cups
                            if c.id == robot.holding_cup_id)
                cup.in_goal = goal_id
                cup.on_pin = stack[-1][1].id   # always a pin on top here
                placed.append(f"cup#{cup.id} → G{goal_id}")
                robot.holding_cup_id = None
            else:
                break
        return placed

    # -- Input --

    def _handle_key(self, event) -> bool:
        if event.key == pygame.K_ESCAPE:
            return False
        if event.key == pygame.K_F11:
            self._toggle_fullscreen()
            return True

        if pygame.K_1 <= event.key <= pygame.K_6:
            idx = event.key - pygame.K_1
            if idx < len(OBJECT_TYPES):
                self.current_object_idx = idx
                self.status = f"Selected style: {OBJECT_TYPES[idx][0]}"
            return True

        if event.key == pygame.K_t:
            if self.hovered_toggle_id is not None:
                tog = next(t for t in self.world.toggles if t.id == self.hovered_toggle_id)
                tog.state = _next_toggle_state(tog.state)
                self.status = f"Toggle Q{tog.quadrant} → {tog.state.value.upper()}"

        elif event.key == pygame.K_m:
            if self.hovered_robot_id is not None:
                r = next(rb for rb in self.world.robots if rb.id == self.hovered_robot_id)
                r.in_midfield = not r.in_midfield
                self.status = f"Robot#{r.id} midfield = {r.in_midfield}"

        elif event.key == pygame.K_SPACE:
            self._advance_phase()

        elif event.key == pygame.K_d:
            self.world = make_demo_world()
            self.auto_red, self.auto_blue = 35, 0
            self._rebind_bots_to_world()
            self.status = "Loaded demo (strategy floor 113 pts)."

        elif event.key == pygame.K_r:
            self.world = make_empty_world()
            self.auto_red = self.auto_blue = None
            self._rebind_bots_to_world()
            self.status = "Field reset."

        elif event.key == pygame.K_BACKSPACE:
            if self.hovered_goal_id is not None:
                self._remove_top_of_stack(self.hovered_goal_id)

        elif event.key == pygame.K_e:
            self._try_interact()

        elif event.key == pygame.K_x:
            # Toggle R0's front wing (mirror of controller button 2)
            self._toggle_wings()

        elif event.key == pygame.K_f:
            # Toggle AI-vs-AI duel mode (zoomed Q1 toggle fight, no scoring objects)
            if self.duel_mode:
                self._exit_duel_mode()
            else:
                self._enter_duel_mode()

        elif event.key == pygame.K_n:
            # Swap red duel bot ↔ neural-network policy (loads ai/toggle_duel_policy.pt)
            self._swap_red_bot_to_nn()

        elif event.key == pygame.K_b:
            # Toggle all bots on/off
            if self.bots:
                new_enabled = not self.bots[0].enabled
                for b in self.bots:
                    b.enabled = new_enabled
                self.status = (f"AI bots {'ENABLED' if new_enabled else 'disabled'} "
                               f"({len(self.bots)} bot(s))")
            else:
                self.status = "No AI bots configured."

        # Curve t (controls shape of partial-stick response)
        elif event.key in (pygame.K_PAGEUP, pygame.K_RIGHTBRACKET):
            self._adjust_curve_t(+1.0)
        elif event.key in (pygame.K_PAGEDOWN, pygame.K_LEFTBRACKET):
            self._adjust_curve_t(-1.0)
        # Turn sensitivity (caps the MAX angular speed)
        elif event.key in (pygame.K_EQUALS, pygame.K_PLUS):
            self._adjust_turn_sensitivity(+0.05)
        elif event.key == pygame.K_MINUS:
            self._adjust_turn_sensitivity(-0.05)

        return True

    def _adjust_curve_t(self, delta: float) -> None:
        """Bump curve t up or down. Higher t = more linear; lower = more curved."""
        new_t = max(2.0, min(100.0, self.joy_curve_t + delta))
        if new_t == self.joy_curve_t:
            return
        self.joy_curve_t = new_t
        sample = self._expo_curve(0.5, new_t)
        self.status = (f"curve t = {self.joy_curve_t:.1f}   "
                        f"sens = {self.turn_sensitivity:.2f}   "
                        f"(half-stick → {sample * self.turn_sensitivity:.2f})")

    def _adjust_turn_sensitivity(self, delta: float) -> None:
        """Bump turn-sensitivity scaler. Caps max angular speed."""
        new_s = max(0.10, min(1.00, self.turn_sensitivity + delta))
        # Snap to nearest 0.05 to keep status readable
        new_s = round(new_s * 20) / 20
        if new_s == self.turn_sensitivity:
            return
        self.turn_sensitivity = new_s
        # ω_max = 2 · S · WHEEL_MAX_SPEED / wheel_base
        w_max = 2 * new_s * self.WHEEL_MAX_SPEED / self.ROBOT_WHEEL_BASE_FT
        self.status = (f"turn sens = {self.turn_sensitivity:.2f}   "
                        f"curve t = {self.joy_curve_t:.1f}   "
                        f"max ω ≈ {w_max:.1f} rad/s "
                        f"({math.degrees(w_max):.0f}°/s)")

    def _toggle_under_back_sensor(self) -> Toggle | None:
        """The toggle (if any) currently engaged by R0's back sensor (overlap
        AND alignment). Returns the first match — toggles are far enough
        apart that at most one can be engaged at any time."""
        if not self.world.robots:
            return None
        robot = self.world.robots[0]
        for tog in self.world.toggles:
            if self._back_sensor_overlaps(robot, tog):
                return tog
        return None

    def _back_sensor_overlaps_geom_only(self, robot: Robot, toggle) -> bool:
        """Pure geometric back-sensor / AABB overlap with NO alignment gate.
        Used to detect the 'touching but misaligned' case so we can give the
        player a specific hint instead of the generic 'back into a toggle'."""
        x_min, x_max, y_min, y_max = self._toggle_aabb(toggle)
        sx, sy = self._back_sensor_point(robot)
        cx_ = max(x_min, min(sx, x_max))
        cy_ = max(y_min, min(sy, y_max))
        ddx = sx - cx_
        ddy = sy - cy_
        return ddx * ddx + ddy * ddy < BACK_SENSOR_RADIUS_FT * BACK_SENSOR_RADIUS_FT

    def _toggle_misalignment_hint(self) -> str:
        """If R0's back sensor is geometrically over a toggle but heading is
        off, return a 'square up' hint. Else generic 'back into a toggle'."""
        if not self.world.robots:
            return "Back into a toggle to cycle it."
        robot = self.world.robots[0]
        for tog in self.world.toggles:
            if self._back_sensor_overlaps_geom_only(robot, tog):
                return (f"Q{tog.quadrant}: square up — your back must face the "
                        f"wall (within ±25°) to engage the toggle.")
        return "Back into a toggle to cycle it."

    def _toggle_wings(self) -> None:
        """A button / X key: extend or retract R0's front wing. When extended,
        a 24"-wide push zone hangs off the front bumper so you can sweep
        several pins/cups in one pass."""
        self.wings_extended = not self.wings_extended
        if self.wings_extended:
            self.status = "WINGS DEPLOYED — 24\" front push zone active"
        else:
            self.status = "Wings retracted — back to 10\" chassis width"

    def _cycle_nearest_toggle(self) -> None:
        """A button / T key: cycle the toggle the robot's BACK is engaging.
        Requires both the sensor disc to overlap the toggle AABB AND the
        heading to point along the wall's outward normal (within ±25°)."""
        tog = self._toggle_under_back_sensor()
        if tog is None:
            self.status = self._toggle_misalignment_hint()
            return
        tog.resting_state = _next_toggle_state(tog.resting_state, tog.quadrant)
        self.status = (f"Toggle Q{tog.quadrant} → "
                        f"{tog.resting_state.value.upper()} "
                        f"(still touching — back off to lock it in)")

    def _tick_match_clock(self, dt: float) -> None:
        """Drive the 1:45 timer. Only ticks during DRIVER phase. Auto-transitions
        to ENDED when time runs out."""
        if self.world.phase != Phase.DRIVER:
            return
        self.world.time_elapsed += dt
        if self.world.time_elapsed >= DRIVER_DURATION_S:
            self.world.time_elapsed = DRIVER_DURATION_S
            self.world.phase = Phase.ENDED
            final = score(self.world,
                           auto_red=self.auto_red, auto_blue=self.auto_blue)
            self.status = (f"1:45 OVER — match ended. "
                           f"FINAL  RED {final.red}  BLUE {final.blue}")

    def _set_nearest_toggle_to_alliance(self) -> None:
        """Y key: set the toggle the robot's BACK is engaging to the player's
        alliance color. Same engagement rules as _cycle_nearest_toggle."""
        tog = self._toggle_under_back_sensor()
        if tog is None:
            self.status = self._toggle_misalignment_hint()
            return
        robot = self.world.robots[0]
        color = (ToggleState.RED if robot.alliance == Alliance.RED
                 else ToggleState.BLUE)
        tog.resting_state = color
        # Always still touching at this point (we just confirmed back-contact);
        # the player needs to break contact for SC4 to lock the resting_state in.
        self.status = (f"Q{tog.quadrant} resting → {color.value.upper()} "
                        f"(still touching — back off to lock it in)")

    def _rebind_bots_to_world(self) -> None:
        """When the world is replaced (R / D keys), rewire each bot's world
        reference and reset its state machine to IDLE."""
        for b in self.bots:
            b.world = self.world
            b.collides_at = self._make_collides_for(b.robot_id)
            b.state = "IDLE"
            b.phase = "MATCH_LOADS"
            b.match_loads_delivered = 0
            b.target = None
            b.target_goal_id = None
            b.target_toggle_id = None
            b._v_left = 0.0
            b._v_right = 0.0

    def _enter_duel_mode(self) -> None:
        """Switch to the AI-vs-AI toggle-duel scenario:
          - Replace the world with one toggle + 2 robots (red R0, blue R2).
          - Replace bots with one driving each robot in POST_LOADS phase.
          - Zoom the field layout to the Q1 quadrant viewport.
          - Player input on R0 is suspended (the bot drives R0)."""
        self.duel_mode = True
        self.world = make_toggle_duel_world()
        self.auto_red = self.auto_blue = None
        self._recompute_layout()
        # Re-create the bot roster — one bot per robot in this world.
        self.bots = [
            ScriptedBot(robot_id=r.id, world=self.world,
                         collides_at=self._make_collides_for(r.id))
            for r in self.world.robots
        ]
        for b in self.bots:
            b.phase = "POST_LOADS"     # skip match-loading; straight to toggle fight
            b.enabled = True
        self.status = ("TOGGLE DUEL — red vs blue race for the Q1 toggle. "
                        "Press R to reset, F again to leave duel mode.")

    def _exit_duel_mode(self) -> None:
        """Return to normal play: standard world, player drives R0, three
        bots run R1/R2/R3."""
        self.duel_mode = False
        self.world = make_empty_world()
        self.auto_red = self.auto_blue = None
        self._recompute_layout()
        self.bots = [
            ScriptedBot(robot_id=1, world=self.world,
                         collides_at=self._make_collides_for(1)),
            ScriptedBot(robot_id=2, world=self.world,
                         collides_at=self._make_collides_for(2)),
            ScriptedBot(robot_id=3, world=self.world,
                         collides_at=self._make_collides_for(3)),
        ]
        self.status = "Duel mode off — back to normal play."

    def _swap_red_bot_to_nn(self) -> None:
        """In duel mode, replace the red (R0) bot with a neural-network policy
        loaded from ai/toggle_duel_policy.pt. Press N again to swap back to
        the rule-based ScriptedBot. No-op if the model file isn't there."""
        if not self.duel_mode:
            self.status = "NN bot is only available in duel mode (press F)."
            return
        from pathlib import Path
        model_path = (Path(__file__).resolve().parent.parent
                      / "ai" / "toggle_duel_policy.pt")
        if not model_path.exists():
            self.status = (f"No trained policy found at {model_path.name}. "
                            f"Run: python scripts/train_toggle_duel.py")
            return
        # Find R0's bot in self.bots
        for i, b in enumerate(self.bots):
            if b.robot_id == 0:
                if isinstance(b, ScriptedBot):
                    # Swap to NN
                    from ai.nn_policy import MLPPolicy
                    from ai.nn_bot import NNBot
                    try:
                        net = MLPPolicy.load(model_path)
                    except Exception as e:
                        self.status = f"Failed to load policy: {e}"
                        return
                    self.bots[i] = NNBot(
                        robot_id=0, world=self.world, policy=net,
                        collides_at=self._make_collides_for(0),
                    )
                    self.status = "Red bot = NEURAL NET (greedy). Press N to revert."
                else:
                    # Swap back to ScriptedBot
                    self.bots[i] = ScriptedBot(
                        robot_id=0, world=self.world,
                        collides_at=self._make_collides_for(0),
                    )
                    self.bots[i].phase = "POST_LOADS"
                    self.bots[i].enabled = True
                    self.status = "Red bot reverted to ScriptedBot."
                return
        self.status = "No red (R0) bot in current roster."

    def _toggle_fullscreen(self) -> None:
        self.is_fullscreen = not self.is_fullscreen
        if self.is_fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode(
                (INITIAL_W, INITIAL_H), pygame.RESIZABLE)
        self._recompute_layout()
        self.status = "Fullscreen " + ("on" if self.is_fullscreen else "off")

    def _advance_phase(self) -> None:
        if self.world.phase == Phase.AUTO:
            # START the 1:45 driver-period clock.
            self.world.phase = Phase.DRIVER
            self.world.time_elapsed = 0.0
            snap = score(self.world)
            self.auto_red, self.auto_blue = snap.red, snap.blue
            self.status = (f"MATCH START — driver phase (1:45). "
                           f"Auto snapshot {self.auto_red}-{self.auto_blue}.")
        elif self.world.phase == Phase.DRIVER:
            # Manually end early (skips the timer)
            self.world.phase = Phase.ENDED
            self.status = "Driver ended manually. Midfield park bonus applies."
        else:
            # ENDED → AUTO (reset world)
            self.world = make_empty_world()
            self.auto_red = self.auto_blue = None
            self._rebind_bots_to_world()
            self.status = "Field reset to AUTO phase. Press SPACE to start the 1:45 match."

    def _handle_click(self, event) -> None:
        mx, my = event.pos
        if event.button == 1:
            # Stack item click → flip orientation
            for goal_id, kind, obj_id, rect in self._stack_hits:
                if rect.collidepoint(mx, my):
                    self._flip_object(kind, obj_id)
                    return
            # Goal click → place next valid object per SC2
            if self.hovered_goal_id is not None:
                self._place_next_in_goal(self.hovered_goal_id)
            elif self.hovered_toggle_id is not None:
                tog = next(t for t in self.world.toggles if t.id == self.hovered_toggle_id)
                tog.resting_state = _next_toggle_state(tog.resting_state, tog.quadrant)
                self.status = (f"Toggle Q{tog.quadrant} resting state → "
                                f"{tog.resting_state.value.upper()}")
            elif self.hovered_robot_id is not None:
                r = next(rb for rb in self.world.robots if rb.id == self.hovered_robot_id)
                r.in_midfield = not r.in_midfield
                self.status = f"Robot#{r.id} midfield = {r.in_midfield}"

        elif event.button == 3:  # right-click → remove top of stack
            # Find which goal we're hovering
            if self.hovered_goal_id is not None:
                self._remove_top_of_stack(self.hovered_goal_id)
            else:
                # Check stack hits — remove top of containing goal
                for goal_id, kind, obj_id, rect in self._stack_hits:
                    if rect.collidepoint(mx, my):
                        self._remove_top_of_stack(goal_id)
                        return

    def _flip_object(self, kind: str, obj_id: int) -> None:
        if kind == "pin":
            pin = next(p for p in self.world.pins if p.id == obj_id)
            pin.half_a_color, pin.half_b_color = pin.half_b_color, pin.half_a_color
            self.status = (f"Pin#{pin.id} flipped — top now {pin.half_a_color.value}")
        else:
            cup = next(c for c in self.world.cups if c.id == obj_id)
            cup.clear_face_up = not cup.clear_face_up
            self.status = (f"Cup#{cup.id} flipped — clear face "
                           f"{'UP' if cup.clear_face_up else 'DOWN'}")

    def _place_next_in_goal(self, gid: int) -> None:
        stack = self.world.stack_in_goal(gid)
        next_kind = self.world.next_placeable_kind(gid)
        cur = OBJECT_TYPES[self.current_object_idx]
        pref_pin = cur if cur[1] == "pin" else OBJECT_TYPES[0]
        pref_cup = cur if cur[1] == "cup" else OBJECT_TYPES[4]

        if next_kind == "pin":
            _, _, ha, hb, _ = pref_pin
            in_cup_id = stack[-1][1].id if stack else None
            self.world.pins.append(Pin(
                id=_next_pin_id(self.world),
                half_a_color=ha, half_b_color=hb,
                in_goal=gid, in_cup=in_cup_id,
            ))
            forced = "" if cur[1] == "pin" else " (forced pin)"
            self.status = f"Added {pref_pin[0]} to G{gid}{forced}"
        else:
            _, _, _, _, clear_up = pref_cup
            on_pin_id = stack[-1][1].id
            self.world.cups.append(Cup(
                id=_next_cup_id(self.world),
                in_goal=gid, on_pin=on_pin_id,
                clear_face_up=bool(clear_up),
            ))
            forced = "" if cur[1] == "cup" else " (forced cup)"
            self.status = f"Added {pref_cup[0]} to G{gid}{forced}"

    def _remove_top_of_stack(self, gid: int) -> None:
        stack = self.world.stack_in_goal(gid)
        if not stack:
            self.status = f"G{gid} is already empty."
            return
        kind, obj = stack[-1]
        if kind == "pin":
            self.world.pins = [p for p in self.world.pins if p.id != obj.id]
            self.status = f"Removed pin#{obj.id} from G{gid}"
        else:
            self.world.cups = [c for c in self.world.cups if c.id != obj.id]
            self.status = f"Removed cup#{obj.id} from G{gid}"

    # -- Hover detection --

    def _update_hover(self, mouse_pos: tuple[int, int]) -> None:
        mx, my = mouse_pos
        self.hovered_goal_id = None
        self.hovered_toggle_id = None
        self.hovered_pin_id = None
        self.hovered_cup_id = None
        self.hovered_robot_id = None

        fl = self.layout.field
        if not fl.contains_px(mx, my):
            return

        # Stack items first (top items in stack)
        for goal_id, kind, obj_id, rect in self._stack_hits:
            if rect.collidepoint(mx, my):
                if kind == "pin":
                    self.hovered_pin_id = obj_id
                else:
                    self.hovered_cup_id = obj_id
                self.hovered_goal_id = goal_id
                return

        # Robots
        for r in self.world.robots:
            rx, ry = fl.ft_to_px(r.x, r.y)
            if abs(mx - rx) <= 20 and abs(my - ry) <= 20:
                self.hovered_robot_id = r.id
                return

        # Goals — all same size (28 px half-extent for the 56px goal)
        for goal in self.world.goals:
            gx, gy = fl.ft_to_px(goal.x, goal.y)
            if abs(mx - gx) <= 28 and abs(my - gy) <= 28:
                self.hovered_goal_id = goal.id
                return

        # Toggles
        for tog in self.world.toggles:
            tx, ty = fl.ft_to_px(tog.x, tog.y)
            if abs(mx - tx) <= 24 and abs(my - ty) <= 24:
                self.hovered_toggle_id = tog.id
                return

    # -- Render --

    def _render(self) -> None:
        self.screen.fill(C.BACKGROUND)
        s = score(self.world, auto_red=self.auto_red, auto_blue=self.auto_blue)

        # Top bar
        H.draw_top_bar(self.screen, pygame.Rect(*self.layout.topbar_rect), self.world, s)

        # Field + elements
        result = D.draw_world(
            self.screen, self.world, self.layout.field,
            hovered_goal=self.hovered_goal_id,
            hovered_toggle=self.hovered_toggle_id,
            hovered_pin=self.hovered_pin_id,
            hovered_cup=self.hovered_cup_id,
            wings_extended_robot_id=(0 if self.wings_extended else None),
        )
        self._stack_hits = result["stack_hits"]

        # Side panels
        margin = self.layout.margin
        side_y = self.layout.topbar_h + margin
        side_h_total = self.layout.win_h - side_y - self.layout.statusbar_h - margin

        bd_h = 290
        tog_h = 220
        help_h = max(120, side_h_total - bd_h - tog_h - 24)

        bd_rect = pygame.Rect(self.layout.side_x, side_y, self.layout.side_w, bd_h)
        H.draw_score_breakdown(self.screen, bd_rect, s)

        tog_rect = pygame.Rect(self.layout.side_x, bd_rect.bottom + 12,
                               self.layout.side_w, tog_h)
        H.draw_toggle_panel(self.screen, tog_rect, self.world)

        help_rect = pygame.Rect(self.layout.side_x, tog_rect.bottom + 12,
                                self.layout.side_w, help_h)
        H.draw_help_panel(self.screen, help_rect,
                          OBJECT_TYPES[self.current_object_idx][0])

        # Status bar
        H.draw_status_bar(self.screen,
                          pygame.Rect(*self.layout.statusbar_rect),
                          self.status)

        pygame.display.flip()


# -- Helpers ------------------------------------------------------------------


def _next_pin_id(world: World) -> int:
    return max((p.id for p in world.pins), default=-1) + 1


def _next_cup_id(world: World) -> int:
    return max((c.id for c in world.cups), default=-1) + 1


def _next_toggle_state(s: ToggleState, quadrant: int = 0) -> ToggleState:
    """Cycle the toggle's resting state. Per-quadrant order:
      Q0 (TOP) + Q1 (RIGHT)   : Y → R → B → Y     ("blue side" YRB)
      Q2 (BOTTOM) + Q3 (LEFT) : Y → B → R → Y     ("red side"  YBR)
    UNSET is a contact-driven state, not part of the user cycle."""
    if quadrant in (2, 3):
        order = [ToggleState.YELLOW, ToggleState.BLUE, ToggleState.RED]
    else:
        order = [ToggleState.YELLOW, ToggleState.RED, ToggleState.BLUE]
    if s in order:
        return order[(order.index(s) + 1) % len(order)]
    return ToggleState.YELLOW


def main() -> None:
    App().run()


if __name__ == "__main__":
    main()
