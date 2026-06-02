"""ScriptedBot — match-load runner state machine.

The simplest possible "interesting" opponent: loops loader → goal → loader → goal.

State machine:
    IDLE
      └─→ if hands empty: target = nearest alliance loader   → DRIVING_TO_LOADER
      └─→ if hands full:  target = nearest alliance goal      → DRIVING_TO_GOAL

    DRIVING_TO_LOADER
      └─→ on arrival (within 0.8 ft): spawn pin+cup, hands full → IDLE

    DRIVING_TO_GOAL
      └─→ on arrival (within 1.3 ft): place per SC2, hands empty → IDLE

Driving uses a simple "point-and-shoot" controller: turn toward target until
heading error is small, then drive forward; reduce throttle when close so the
bot doesn't overshoot its target.

Tank physics + collision is duplicated here so the bot and the player feel
identical from a kinematics standpoint. (Constants mirror App's defaults.)
"""

from __future__ import annotations

import math
from typing import Callable, Optional

from core.state import Alliance, Cup, Phase, Pin, PinColor, Robot, ToggleState, World


# Mirrors render.pygame_view tank-physics constants.
WHEEL_MAX_SPEED   = 6.0
WHEEL_ACCEL       = 22.0
WHEEL_DECEL       = 32.0
WHEEL_BASE_FT     = 10.0 / 12.0
ROBOT_COLL_R_FT   = 0.55

# Loader positions in field ft. Must mirror render.draw.LOADER_POSITIONS_FT.
_LOADERS = [(-6.0, 5.0), (-6.0, -5.0), (6.0, 5.0), (6.0, -5.0)]

# Arrival thresholds
_AT_LOADER_FT  = 0.9
_AT_GOAL_FT    = 1.3
_AT_TOGGLE_FT  = 0.4    # arrival range at toggle staging point
_AT_MIDFIELD_FT = 0.4   # arrival range at midfield parking spot

# Game rule: 10 pin match loads per alliance per match (SG11 / bill of materials).
MATCH_LOAD_BUDGET = 10

# Back-mounted toggle sensor — must mirror render.pygame_view constants.
# Toggles can ONLY be flipped by the back of the robot, not the front.
BACK_SENSOR_OFFSET_FT = (14.0 / 12.0) / 2.0   # half robot length (14")
BACK_SENSOR_RADIUS_FT = 0.10                  # tight ~1.2" sensor (mirror)
# Heading must be roughly perpendicular to the wall — back direction within
# this angle of the wall's outward normal.
TOGGLE_ALIGNMENT_TOL_RAD = math.radians(25.0)
TOGGLE_ALIGNMENT_DOT_MIN = math.cos(TOGGLE_ALIGNMENT_TOL_RAD)   # ≈ 0.906
# Toggle bbox half-dimensions (must match render.pygame_view).
TOGGLE_LONG_HALF_FT  = 1.0
TOGGLE_SHORT_HALF_FT = 0.30
# Reverse maneuver tuning. The bot's "OK to reverse" tolerance is TIGHTER
# than the toggle-engage alignment tolerance so by the time the back touches
# the toggle, alignment is comfortably inside the engagement cone.
_BACK_ALIGN_TOL = 0.18    # rad ≈ 10.3° — well inside the 25° engagement cone
_REVERSE_THROTTLE = 0.40  # how fast to back up into the toggle


def _approach(current: float, target: float,
              accel: float, decel: float, dt: float) -> float:
    """Same approach helper as the player kinematics: ramp toward target
    with separate accel/decel rates depending on whether magnitude grows."""
    diff = target - current
    if diff == 0.0:
        return current
    if target * current >= 0.0 and abs(target) > abs(current):
        rate = accel
    else:
        rate = decel
    max_change = rate * dt
    if abs(diff) <= max_change:
        return target
    return current + (max_change if diff > 0 else -max_change)


def _wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class ScriptedBot:
    """Match-load runner. Drives a single robot autonomously."""

    def __init__(self, robot_id: int, world: World,
                 collides_at: Optional[Callable[[float, float], bool]] = None):
        self.robot_id = robot_id
        self.world = world
        # Allow injecting the player-app's collision check so the bot respects
        # the same goal hitboxes the player does. Default to "no obstacles"
        # which still respects wall clamping internally.
        self.collides_at = collides_at or (lambda x, y: False)

        self.state: str = "IDLE"
        self.target: Optional[tuple[float, float]] = None
        self.target_goal_id: Optional[int] = None
        self.target_toggle_id: Optional[int] = None

        self._v_left: float = 0.0
        self._v_right: float = 0.0

        self.status_msg: str = "bot idle"
        self.enabled: bool = True

        # High-level phase. After MATCH_LOAD_BUDGET deliveries we transition
        # from grinding match-loads to setting alliance toggles + parking
        # midfield (the "endgame" of an Override match).
        self.phase: str = "MATCH_LOADS"
        self.match_loads_delivered: int = 0

        # Stuck detection — if wheels say "we should be moving" but the chassis
        # hasn't moved for ~0.4 s, trigger an "unstuck" maneuver (back up + turn).
        self._prev_pos: tuple[float, float] = (0.0, 0.0)
        self._prev_dist_to_target: Optional[float] = None
        self._stuck_dt: float = 0.0
        self._unstuck_dt: float = 0.0    # remaining seconds of unstuck routine
        self._unstuck_dir: float = 1.0   # +1 or -1 — which way to swing while reversing
        # Timeout for BACK_INTO_TOGGLE — if we can't engage the toggle within
        # this many seconds (e.g. opponent is blocking us), abandon and re-plan.
        self._back_into_dt: float = 0.0
        # Post-flip cooldown — after flipping a toggle, refuse to immediately
        # re-engage it. This stops two bots from spamming flips at each other
        # in a tight burst zone.
        self._flip_cooldown_dt: float = 0.0
        self._just_flipped_id: Optional[int] = None   # which toggle was last flipped

    # -- Accessors ------------------------------------------------------------

    @property
    def robot(self) -> Robot:
        return next(r for r in self.world.robots if r.id == self.robot_id)

    @property
    def alliance(self) -> Alliance:
        return self.robot.alliance

    # -- Tick -----------------------------------------------------------------

    def update(self, dt: float) -> None:
        if not self.enabled or dt <= 0.0:
            return
        # Bot only acts while the match is live (AUTO or DRIVER).
        # Once the 1:45 clock runs out, it sits still.
        if self.world.phase == Phase.ENDED:
            self._v_left = 0.0
            self._v_right = 0.0
            return

        # Tick cooldowns
        if self._flip_cooldown_dt > 0.0:
            self._flip_cooldown_dt = max(0.0, self._flip_cooldown_dt - dt)
            if self._flip_cooldown_dt == 0.0:
                self._just_flipped_id = None

        # If we're recovering from being stuck, override normal logic.
        if self._unstuck_dt > 0.0:
            self._tick_unstuck(dt)
            self._unstuck_dt -= dt
            self._prev_pos = (self.robot.x, self.robot.y)
            return

        if self.phase == "MATCH_LOADS":
            self._tick_match_loads(dt)
        elif self.phase == "POST_LOADS":
            self._tick_post_loads(dt)

        self._update_stuck_detection(dt)

    def _update_stuck_detection(self, dt: float) -> None:
        """If the bot isn't making meaningful progress toward its target,
        trigger an unstuck maneuver. Two fail modes are caught:

        1. Wheels spinning but body not moving (hit a wall/goal cleanly)
        2. Wheels spinning, body moving slowly, but PROGRESS toward the
           target is negligible (sliding along an obstacle's edge or being
           pinned by another robot).

        Both fire at 0.25 s — faster than before, because long lock-ups
        are visually obvious and the duel can't wait."""
        driving_states = {
            "DRIVING_TO_LOADER", "DRIVING_TO_GOAL",
            "DRIVING_TO_TOGGLE", "DRIVING_TO_MIDFIELD",
            "BACK_INTO_TOGGLE",     # also catch stuck-during-reverse
        }
        r = self.robot

        if self.state not in driving_states or self.target is None:
            self._stuck_dt = 0.0
            self._prev_pos = (r.x, r.y)
            self._prev_dist_to_target = None
            return

        dist_to_target = math.hypot(self.target[0] - r.x,
                                       self.target[1] - r.y)
        # Don't trigger near the target — slowdowns are legitimate there
        if dist_to_target < 0.8:
            self._stuck_dt = 0.0
            self._prev_pos = (r.x, r.y)
            self._prev_dist_to_target = dist_to_target
            return

        expected = abs(self._v_left) + abs(self._v_right) > 1.0
        actual_move = math.hypot(r.x - self._prev_pos[0],
                                    r.y - self._prev_pos[1])
        moved_at_all = actual_move > 0.002

        # Progress toward target — positive means closing the gap
        if self._prev_dist_to_target is not None:
            progress = self._prev_dist_to_target - dist_to_target
        else:
            progress = 0.0

        # Stuck if wheels are spinning AND (not moving OR not making progress)
        not_progressing = progress < 0.005     # ~3 cm/frame of closing
        if expected and (not moved_at_all or not_progressing):
            self._stuck_dt += dt
            if self._stuck_dt > 0.25:
                # Hard escape — full-speed reverse for 0.7 s with a turn.
                # Alternate the swing each trigger so we don't loop.
                self._unstuck_dt = 0.7
                self._unstuck_dir = (-self._unstuck_dir
                                       if self._unstuck_dir else 1.0)
                self._stuck_dt = 0.0
                self.status_msg = (f"R{self.robot_id} no progress — "
                                    f"backing off")
        else:
            self._stuck_dt = 0.0

        self._prev_pos = (r.x, r.y)
        self._prev_dist_to_target = dist_to_target

    def _tick_unstuck(self, dt: float) -> None:
        """Drive backward HARD with an asymmetric wheel split so we also
        rotate away from whatever we hit. Full speed (was 60%) so the bot
        actually breaks free of pin situations (e.g., another robot pushing
        it into a goal)."""
        bias = 0.35 * self._unstuck_dir
        target_wl = -WHEEL_MAX_SPEED * (1.0 - bias)
        target_wr = -WHEEL_MAX_SPEED * (1.0 + bias)

        self._v_left = _approach(self._v_left, target_wl,
                                  WHEEL_ACCEL, WHEEL_DECEL, dt)
        self._v_right = _approach(self._v_right, target_wr,
                                   WHEEL_ACCEL, WHEEL_DECEL, dt)

        v_lin = 0.5 * (self._v_left + self._v_right)
        v_ang = (self._v_right - self._v_left) / WHEEL_BASE_FT

        r = self.robot
        r.theta = _wrap_pi(r.theta + v_ang * dt)
        move_x = math.cos(r.theta) * v_lin * dt
        move_y = math.sin(r.theta) * v_lin * dt

        bound = 6.0 - ROBOT_COLL_R_FT
        try_x = max(-bound, min(bound, r.x + move_x))
        if not self.collides_at(try_x, r.y):
            r.x = try_x
        try_y = max(-bound, min(bound, r.y + move_y))
        if not self.collides_at(r.x, try_y):
            r.y = try_y

    # -- MATCH_LOADS phase ----------------------------------------------------

    def _alliance_loads_used(self) -> int:
        """Per-ALLIANCE match-load count (shared between any robots on the
        same alliance, including the player)."""
        if self.alliance == Alliance.RED:
            return self.world.match_loads_red
        return self.world.match_loads_blue

    def _record_alliance_load(self) -> None:
        """Increment the alliance counter when this bot spawns from a loader."""
        if self.alliance == Alliance.RED:
            self.world.match_loads_red += 1
        else:
            self.world.match_loads_blue += 1

    def _tick_match_loads(self, dt: float) -> None:
        # Once the alliance budget is gone AND we've delivered everything we
        # already loaded, transition to the post-loads phase. The empty-hands
        # check ensures we finish the current delivery cycle (otherwise the
        # 10th spawned pin+cup would be abandoned on the bot's chassis).
        r = self.robot
        hands_empty = (r.holding_pin_id is None and r.holding_cup_id is None)
        if (self.state == "IDLE"
                and self._alliance_loads_used() >= MATCH_LOAD_BUDGET
                and hands_empty):
            self.phase = "POST_LOADS"
            self.state = "IDLE"
            self.target = None
            self.target_goal_id = None
            self._v_left = 0.0
            self._v_right = 0.0
            self.status_msg = (f"R{self.robot_id} done — alliance match-load "
                                f"budget ({MATCH_LOAD_BUDGET}) exhausted; endgame")
            return

        if self.state == "IDLE":
            self._plan_next_action()

        if self.state == "DRIVING_TO_LOADER":
            if self._at_target(_AT_LOADER_FT):
                self._spawn_from_loader()
            else:
                self._drive_step(dt)
        elif self.state == "DRIVING_TO_GOAL":
            if self._at_target(_AT_GOAL_FT):
                self._place_in_goal()
                self.match_loads_delivered += 1
            else:
                self._drive_step(dt)

    # -- POST_LOADS phase: set alliance toggles → park midfield ---------------

    def _tick_post_loads(self, dt: float) -> None:
        # Whenever we're not in the middle of executing a sub-goal, re-plan.
        if self.state == "IDLE" or self.state in (
                "DRIVING_TO_LOADER", "DRIVING_TO_GOAL"):
            self.state = "IDLE"
            self._plan_post_loads()
            return

        if self.state == "DRIVING_TO_TOGGLE":
            # Drive forward (nose-first) to the staging point in front of the
            # toggle. On arrival, switch to BACK_INTO_TOGGLE which rotates the
            # chassis and reverses until the rear bumper sensor touches.
            if self._at_target(_AT_TOGGLE_FT):
                self.state = "BACK_INTO_TOGGLE"
                self._back_into_dt = 0.0          # reset engagement timer
            else:
                self._drive_step(dt)
        elif self.state == "BACK_INTO_TOGGLE":
            self._tick_back_into_toggle(dt)
        elif self.state == "LEAVING_TOGGLE":
            self._tick_leaving_toggle(dt)
        elif self.state == "DEFENDING":
            # Re-plan EACH frame: if any toggle has drifted off our color
            # (player flipped it), go fix it before continuing to defend.
            color = (ToggleState.RED if self.alliance == Alliance.RED
                     else ToggleState.BLUE)
            for tog in self.world.toggles:
                if tog.resting_state != color:
                    self.state = "IDLE"
                    self._plan_post_loads()
                    return
            self._tick_defend(dt)
        elif self.state == "DRIVING_TO_MIDFIELD":
            # Legacy state — replanning will route us to DEFENDING after one tick
            self.state = "IDLE"

    def _plan_post_loads(self) -> None:
        """Priority order:
          1. Drive to ANY toggle that isn't our color and flip it. (Even
             enemy-side toggles — owning them flips yellow ownership.)
          2. If all four toggles are ours, chase the player to disrupt them.
        Re-evaluated every IDLE tick, so if the player flips a toggle back
        we go fix it again."""
        color = (ToggleState.RED if self.alliance == Alliance.RED
                 else ToggleState.BLUE)

        # 1. Any toggle not our color — go set it (pick nearest first).
        # Skip the toggle we JUST flipped during the post-flip cooldown so
        # we don't get sucked into a flip-flop burst with the opponent.
        best_tog = None
        best_d2 = float("inf")
        for tog in self.world.toggles:
            if tog.resting_state == color:
                continue
            if (self._flip_cooldown_dt > 0.0
                    and self._just_flipped_id == tog.id):
                continue   # let things settle before re-engaging
            ddx = tog.x - self.robot.x
            ddy = tog.y - self.robot.y
            d2 = ddx * ddx + ddy * ddy
            if d2 < best_d2:
                best_d2 = d2
                best_tog = tog
        if best_tog is not None:
            self.target = self._toggle_approach_point(best_tog)
            self.target_toggle_id = best_tog.id
            self.state = "DRIVING_TO_TOGGLE"
            self.status_msg = (f"R{self.robot_id} → flip Q{best_tog.quadrant} "
                                f"to {color.value.upper()}")
            return

        # 2. No flippable toggle (all ours, or all on cooldown).
        # Pick the toggle most likely to be attacked (closest to the opponent)
        # and park between them and that toggle — smart defense.
        opp = self._nearest_opponent()
        if opp is None:
            return
        self.target_toggle_id = None
        self.state = "DEFENDING"
        # Initial target — _tick_defend recomputes each frame
        self.target = (opp.x, opp.y)
        self.status_msg = f"R{self.robot_id} → defend (block R{opp.id})"

    def _nearest_opponent(self):
        """Closest robot on the opposing alliance, or None."""
        opps = [r for r in self.world.robots if r.alliance != self.alliance]
        if not opps:
            return None
        return min(opps, key=lambda r: math.hypot(r.x - self.robot.x,
                                                    r.y - self.robot.y))

    def _set_alliance_toggle(self) -> None:
        if self.target_toggle_id is None:
            self.state = "IDLE"
            return
        tog = next((t for t in self.world.toggles
                    if t.id == self.target_toggle_id), None)
        if tog is None:
            self.state = "IDLE"
            return
        color = (ToggleState.RED if self.alliance == Alliance.RED
                 else ToggleState.BLUE)
        tog.resting_state = color
        self.status_msg = (f"R{self.robot_id} set Q{tog.quadrant} → "
                            f"{color.value.upper()}")
        self.state = "IDLE"
        self.target_toggle_id = None

    def _park_midfield(self) -> None:
        self.robot.in_midfield = True
        self.state = "PARKED"
        self._v_left = 0.0
        self._v_right = 0.0
        self.status_msg = f"R{self.robot_id} PARKED in midfield (+8 pts)"

    def _toggle_approach_point(self, tog) -> tuple[float, float]:
        """A staging point ~1.2 ft inward from the toggle along the wall
        normal, plus a small ALLIANCE-DEPENDENT offset along the wall's long
        axis so red and blue bots target slightly different spots and don't
        deadlock at the same point when racing for the same toggle."""
        norm = math.hypot(tog.x, tog.y)
        if norm < 0.01:
            ux, uy = 0.0, -1.0
        else:
            ux = -tog.x / norm
            uy = -tog.y / norm
        # Offset along the wall's long axis (perpendicular to the wall normal).
        # For side walls (|x|>|y|) the long axis is y; for top/bottom it's x.
        offset = 0.7 if self.alliance == Alliance.RED else -0.7
        if abs(tog.x) > abs(tog.y):
            return (tog.x + ux * 1.2,         tog.y + uy * 1.2 + offset)
        return     (tog.x + ux * 1.2 + offset, tog.y + uy * 1.2)

    # -- Planning -------------------------------------------------------------

    def _plan_next_action(self) -> None:
        r = self.robot
        # If hands have anything, deliver. Otherwise go load.
        if r.holding_pin_id is not None or r.holding_cup_id is not None:
            goal = self._nearest_alliance_goal()
            if goal is None:
                return
            self.target = (goal.x, goal.y)
            self.target_goal_id = goal.id
            self.state = "DRIVING_TO_GOAL"
            self.status_msg = f"R{self.robot_id} → deliver to G{goal.id}"
        else:
            loader = self._nearest_alliance_loader()
            if loader is None:
                return
            self.target = loader
            self.state = "DRIVING_TO_LOADER"
            self.status_msg = f"R{self.robot_id} → load @ {loader}"

    # -- Target lookups -------------------------------------------------------

    def _nearest_alliance_goal(self):
        r = self.robot
        best = None
        best_d2 = float("inf")
        for g in self.world.goals:
            if g.alliance != self.alliance:
                continue
            dx = g.x - r.x
            dy = g.y - r.y
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best = g
        return best

    def _nearest_alliance_loader(self) -> Optional[tuple[float, float]]:
        # Red on the LEFT wall (x < 0), blue on the RIGHT wall (x > 0).
        if self.alliance == Alliance.RED:
            candidates = [(lx, ly) for lx, ly in _LOADERS if lx < 0]
        else:
            candidates = [(lx, ly) for lx, ly in _LOADERS if lx > 0]
        r = self.robot
        best = None
        best_d2 = float("inf")
        for lx, ly in candidates:
            dx = lx - r.x
            dy = ly - r.y
            d2 = dx * dx + dy * dy
            if d2 < best_d2:
                best_d2 = d2
                best = (lx, ly)
        return best

    def _at_target(self, radius_ft: float) -> bool:
        if self.target is None:
            return False
        r = self.robot
        dx = self.target[0] - r.x
        dy = self.target[1] - r.y
        return (dx * dx + dy * dy) <= radius_ft * radius_ft

    # -- Actions --------------------------------------------------------------

    def _spawn_from_loader(self) -> None:
        # Bail if the alliance has already used all its match loads.
        if self._alliance_loads_used() >= MATCH_LOAD_BUDGET:
            self.phase = "POST_LOADS"
            self.state = "IDLE"
            self.status_msg = (f"R{self.robot_id} no match loads left "
                                f"({MATCH_LOAD_BUDGET}/{MATCH_LOAD_BUDGET} used)")
            return

        r = self.robot
        next_pid = max((p.id for p in self.world.pins), default=-1) + 1
        next_cid = max((c.id for c in self.world.cups), default=-1) + 1
        bot_color = (PinColor.RED if self.alliance == Alliance.RED
                     else PinColor.BLUE)
        self.world.cups.append(Cup(
            id=next_cid, x=r.x, y=r.y,
            in_goal=None, on_pin=None,
            clear_face_up=True,
        ))
        self.world.pins.append(Pin(
            id=next_pid,
            half_a_color=PinColor.YELLOW, half_b_color=bot_color,
            x=r.x, y=r.y, in_goal=None, in_cup=None,
        ))
        r.holding_pin_id = next_pid
        r.holding_cup_id = next_cid
        self._record_alliance_load()
        self.state = "IDLE"
        self.status_msg = (f"R{self.robot_id} loaded pin#{next_pid} + cup#{next_cid} "
                            f"({self._alliance_loads_used()}/{MATCH_LOAD_BUDGET})")

    def _place_in_goal(self) -> None:
        if self.target_goal_id is None:
            return
        r = self.robot
        placed = 0
        # Up to two iterations: pin → forced cup → pin → ...
        for _ in range(2):
            stack = self.world.stack_in_goal(self.target_goal_id)
            kind = self.world.next_placeable_kind(self.target_goal_id)
            if kind == "pin" and r.holding_pin_id is not None:
                pin = next(p for p in self.world.pins if p.id == r.holding_pin_id)
                pin.in_goal = self.target_goal_id
                pin.in_cup = stack[-1][1].id if stack else None
                r.holding_pin_id = None
                placed += 1
            elif kind == "cup" and r.holding_cup_id is not None:
                cup = next(c for c in self.world.cups if c.id == r.holding_cup_id)
                cup.in_goal = self.target_goal_id
                cup.on_pin = stack[-1][1].id
                r.holding_cup_id = None
                placed += 1
            else:
                break

        # Anything leftover gets dropped at robot's front (rare with the
        # pin+cup pair from the loader).
        if r.holding_pin_id is not None:
            pin = next(p for p in self.world.pins if p.id == r.holding_pin_id)
            pin.x = r.x + math.cos(r.theta) * 0.6
            pin.y = r.y + math.sin(r.theta) * 0.6
            r.holding_pin_id = None
        if r.holding_cup_id is not None:
            cup = next(c for c in self.world.cups if c.id == r.holding_cup_id)
            cup.x = r.x + math.cos(r.theta) * 0.6
            cup.y = r.y + math.sin(r.theta) * 0.6
            r.holding_cup_id = None

        self.state = "IDLE"
        self.target_goal_id = None
        self.status_msg = f"R{self.robot_id} placed {placed} in G{self.target_goal_id}"

    # -- Back-sensor helpers --------------------------------------------------

    def _back_sensor_point(self) -> tuple[float, float]:
        """Position of the rear-mounted toggle sensor (mirror of pygame_view)."""
        r = self.robot
        return (
            r.x - math.cos(r.theta) * BACK_SENSOR_OFFSET_FT,
            r.y - math.sin(r.theta) * BACK_SENSOR_OFFSET_FT,
        )

    @staticmethod
    def _toggle_aabb(tog) -> tuple[float, float, float, float]:
        """Toggle bbox — mirror of render.pygame_view._toggle_aabb."""
        if abs(tog.x) > abs(tog.y):
            # Mounted on left/right wall — long axis vertical
            return (tog.x - TOGGLE_SHORT_HALF_FT, tog.x + TOGGLE_SHORT_HALF_FT,
                    tog.y - TOGGLE_LONG_HALF_FT,  tog.y + TOGGLE_LONG_HALF_FT)
        # Mounted on top/bottom wall — long axis horizontal
        return (tog.x - TOGGLE_LONG_HALF_FT,  tog.x + TOGGLE_LONG_HALF_FT,
                tog.y - TOGGLE_SHORT_HALF_FT, tog.y + TOGGLE_SHORT_HALF_FT)

    def _back_overlaps_toggle(self, tog) -> bool:
        """Does this bot's rear sensor disc overlap the toggle's AABB AND
        the bot's heading point along the toggle's outward wall normal
        (within TOGGLE_ALIGNMENT_TOL_RAD)? Both are required to engage —
        mirror of the player-side rule in render.pygame_view."""
        # Alignment check first (cheap)
        if abs(tog.x) > abs(tog.y):
            nx = 1.0 if tog.x > 0 else -1.0
            ny = 0.0
        else:
            nx = 0.0
            ny = 1.0 if tog.y > 0 else -1.0
        r = self.robot
        bx = -math.cos(r.theta)
        by = -math.sin(r.theta)
        if (bx * nx + by * ny) < TOGGLE_ALIGNMENT_DOT_MIN:
            return False
        # Then the sensor disc / AABB overlap
        x_min, x_max, y_min, y_max = self._toggle_aabb(tog)
        sx, sy = self._back_sensor_point()
        cx_ = max(x_min, min(sx, x_max))
        cy_ = max(y_min, min(sy, y_max))
        ddx = sx - cx_
        ddy = sy - cy_
        return ddx * ddx + ddy * ddy < BACK_SENSOR_RADIUS_FT * BACK_SENSOR_RADIUS_FT

    def _apply_tank_kinematics(self, dt: float) -> None:
        """Integrate v_left/v_right into chassis pose + clamp to field/goal
        collisions. Shared between forward driving (`_drive_step`) and the
        back-into-toggle reverse maneuver.

        If both axes get blocked (cornered against a goal), trigger the
        unstuck reverse routine — same trick as in `_drive_step`."""
        r = self.robot
        v_lin = 0.5 * (self._v_left + self._v_right)
        v_ang = (self._v_right - self._v_left) / WHEEL_BASE_FT
        r.theta = _wrap_pi(r.theta + v_ang * dt)
        move_x = math.cos(r.theta) * v_lin * dt
        move_y = math.sin(r.theta) * v_lin * dt
        bound = 6.0 - ROBOT_COLL_R_FT
        blocked_x = False
        blocked_y = False
        try_x = max(-bound, min(bound, r.x + move_x))
        if not self.collides_at(try_x, r.y):
            r.x = try_x
        elif abs(move_x) > 1e-4:
            blocked_x = True
        try_y = max(-bound, min(bound, r.y + move_y))
        if not self.collides_at(r.x, try_y):
            r.y = try_y
        elif abs(move_y) > 1e-4:
            blocked_y = True
        # Cornered? Fire the unstuck routine.
        if blocked_x and blocked_y and self._unstuck_dt <= 0.0:
            self._unstuck_dt = 0.7
            self._unstuck_dir = (-self._unstuck_dir
                                   if self._unstuck_dir else 1.0)
            self._v_left = 0.0
            self._v_right = 0.0

    # -- Back-into-toggle maneuver --------------------------------------------

    def _tick_back_into_toggle(self, dt: float) -> None:
        """Rotate so the chassis back faces the target toggle, then reverse
        until the back sensor disc overlaps the toggle AABB. On overlap, flip
        the resting_state and transition to LEAVING_TOGGLE.

        Gives up after ~4 seconds of trying — if we can't engage in that
        time we're probably blocked by an opponent. Re-planning lets us try
        a different angle or fall back to defending elsewhere."""
        tog = next((t for t in self.world.toggles
                    if t.id == self.target_toggle_id), None)
        if tog is None:
            self.state = "IDLE"
            self.target_toggle_id = None
            return

        # If the rear bumper is already on the toggle, flip and leave.
        if self._back_overlaps_toggle(tog):
            color = (ToggleState.RED if self.alliance == Alliance.RED
                     else ToggleState.BLUE)
            tog.resting_state = color
            self.status_msg = (f"R{self.robot_id} backed into Q{tog.quadrant} "
                                f"→ {color.value.upper()}")
            self.state = "LEAVING_TOGGLE"
            self._back_into_dt = 0.0
            # Refuse to immediately re-flip the same toggle for a short
            # cooldown — gives the opponent time to react and the position
            # battle time to play out instead of devolving into a flip burst.
            self._flip_cooldown_dt = 1.2
            self._just_flipped_id = tog.id
            return

        # Engagement timeout — abandon if we've been trying for too long.
        self._back_into_dt += dt
        if self._back_into_dt > 4.0:
            self.status_msg = (f"R{self.robot_id} abandoning Q{tog.quadrant} "
                                f"(opponent is blocking) — re-planning")
            self.state = "IDLE"
            self.target_toggle_id = None
            self._back_into_dt = 0.0
            self._v_left = 0.0
            self._v_right = 0.0
            return

        # Desired heading: PERPENDICULAR to the wall the toggle sits on, so
        # the back direction is exactly the wall's outward normal. (Aiming
        # the back at the toggle center fails when the bot is offset along
        # the wall — the heading ends up diagonal and trips the 25° toggle
        # alignment gate.)
        if abs(tog.x) > abs(tog.y):
            nx = 1.0 if tog.x > 0 else -1.0
            ny = 0.0
        else:
            nx = 0.0
            ny = 1.0 if tog.y > 0 else -1.0
        # Heading is opposite of the outward normal (so back = +normal).
        desired_theta = math.atan2(-ny, -nx)
        r = self.robot
        herr = _wrap_pi(desired_theta - r.theta)

        if abs(herr) > _BACK_ALIGN_TOL:
            # Rotate in place — no translation while aligning
            turn_mag = 0.55
            target_wl = -math.copysign(turn_mag, herr) * WHEEL_MAX_SPEED
            target_wr = +math.copysign(turn_mag, herr) * WHEEL_MAX_SPEED
        else:
            # Heading aligned — reverse at a gentle pace, with light steering
            throttle = -_REVERSE_THROTTLE
            turn = max(-0.15, min(0.15, herr * 0.5))
            raw_l = throttle - turn
            raw_r = throttle + turn
            peak = max(abs(raw_l), abs(raw_r))
            if peak > 1.0:
                raw_l /= peak
                raw_r /= peak
            target_wl = raw_l * WHEEL_MAX_SPEED
            target_wr = raw_r * WHEEL_MAX_SPEED

        self._v_left = _approach(self._v_left, target_wl,
                                  WHEEL_ACCEL, WHEEL_DECEL, dt)
        self._v_right = _approach(self._v_right, target_wr,
                                   WHEEL_ACCEL, WHEEL_DECEL, dt)
        self._apply_tank_kinematics(dt)

    def _tick_defend(self, dt: float) -> None:
        """Smart defense: pick the toggle the opponent is closest to (the
        most-threatened one) and park ON the line between the opponent and
        that toggle, ~1.8 ft from the toggle. From there we physically block
        the opponent's approach — they have to push us out to reach the wall.

        Once we're at the block point we stop and TURN to face the toggle so
        our heavy back end is toward the opponent (better push physics) and
        the front nose points along the wall normal — ready to either re-flip
        if needed or hold position.

        If the toggle's actual state has drifted to UNSET (opponent currently
        touching), we treat that as if it's drifted off our color and race
        back to re-engage."""
        # First — if SC4 has the toggle currently showing UNSET (opponent in
        # contact), don't wait. Race in to flip it back.
        color = (ToggleState.RED if self.alliance == Alliance.RED
                 else ToggleState.BLUE)
        for tog in self.world.toggles:
            if tog.state == ToggleState.UNSET:
                # Treat this as if it's been lost — go re-engage
                self.target_toggle_id = tog.id
                self.target = self._toggle_approach_point(tog)
                self.state = "DRIVING_TO_TOGGLE"
                self.status_msg = (f"R{self.robot_id} re-engaging Q{tog.quadrant} "
                                    f"(opponent is touching it)")
                return

        opp = self._nearest_opponent()
        if opp is None:
            # No opponent on the field — just coast to a stop wherever we are
            self._v_left = _approach(self._v_left, 0.0,
                                      WHEEL_ACCEL, WHEEL_DECEL, dt)
            self._v_right = _approach(self._v_right, 0.0,
                                       WHEEL_ACCEL, WHEEL_DECEL, dt)
            return

        # Threatened toggle = the one closest to the opponent
        threatened = min(self.world.toggles,
                          key=lambda t: math.hypot(t.x - opp.x, t.y - opp.y))

        # Block point: 1.8 ft inward from the threatened toggle along the
        # line toward the opponent. Fall back to the wall normal if opponent
        # is essentially at the toggle.
        bdx = opp.x - threatened.x
        bdy = opp.y - threatened.y
        bd  = math.hypot(bdx, bdy)
        if bd < 0.2:
            # Opponent is right at the toggle — back off along the wall normal
            if abs(threatened.x) > abs(threatened.y):
                ux = -1.0 if threatened.x > 0 else 1.0
                uy = 0.0
            else:
                ux = 0.0
                uy = -1.0 if threatened.y > 0 else 1.0
        else:
            ux = bdx / bd
            uy = bdy / bd
        block_x = threatened.x + ux * 1.8
        block_y = threatened.y + uy * 1.8
        self.target = (block_x, block_y)
        self.target_toggle_id = threatened.id

        dist = math.hypot(block_x - self.robot.x, block_y - self.robot.y)
        if dist > 0.4:
            # Still moving into position
            self._drive_step(dt)
            return

        # AT block position — coast to a stop, but keep aiming the chassis
        # so the rear is toward the toggle (ready to back into it instantly
        # if the opponent slips around).
        if abs(threatened.x) > abs(threatened.y):
            wall_nx = 1.0 if threatened.x > 0 else -1.0
            wall_ny = 0.0
        else:
            wall_nx = 0.0
            wall_ny = 1.0 if threatened.y > 0 else -1.0
        desired_theta = math.atan2(-wall_ny, -wall_nx)
        herr = _wrap_pi(desired_theta - self.robot.theta)
        if abs(herr) > 0.25:
            # Spin in place to face the right way
            turn_mag = 0.4
            target_wl = -math.copysign(turn_mag, herr) * WHEEL_MAX_SPEED
            target_wr = +math.copysign(turn_mag, herr) * WHEEL_MAX_SPEED
            self._v_left = _approach(self._v_left, target_wl,
                                      WHEEL_ACCEL, WHEEL_DECEL, dt)
            self._v_right = _approach(self._v_right, target_wr,
                                       WHEEL_ACCEL, WHEEL_DECEL, dt)
            self._apply_tank_kinematics(dt)
        else:
            # Aligned and at block point — full stop
            self._v_left = _approach(self._v_left, 0.0,
                                      WHEEL_ACCEL, WHEEL_DECEL, dt)
            self._v_right = _approach(self._v_right, 0.0,
                                       WHEEL_ACCEL, WHEEL_DECEL, dt)
            self.status_msg = (f"R{self.robot_id} blocking Q{threatened.quadrant} "
                                f"({color.value.upper()})")

    def _tick_leaving_toggle(self, dt: float) -> None:
        """After flipping, drive forward away from the wall to a strategic
        distance (~1.6 ft from the toggle center). Driving just far enough
        to break SC4 contact isn't enough — at that range, the opponent can
        immediately push us back into contact and trigger a flurry of flips.
        Putting real distance between us and the toggle gives the toggle
        time to actually settle to our color, and lets us re-plan cleanly."""
        tog = next((t for t in self.world.toggles
                    if t.id == self.target_toggle_id), None)
        if tog is None:
            self.state = "IDLE"
            self.target_toggle_id = None
            self._v_left = 0.0
            self._v_right = 0.0
            return
        dist_to_tog = math.hypot(tog.x - self.robot.x, tog.y - self.robot.y)
        if dist_to_tog >= 1.6 and not self._back_overlaps_toggle(tog):
            # Strategically clear — let the planner decide what's next.
            self.state = "IDLE"
            self.target_toggle_id = None
            self._v_left = 0.0
            self._v_right = 0.0
            return
        # Still too close — drive forward straight ahead at moderate throttle.
        target_w = 0.55 * WHEEL_MAX_SPEED
        self._v_left = _approach(self._v_left, target_w,
                                  WHEEL_ACCEL, WHEEL_DECEL, dt)
        self._v_right = _approach(self._v_right, target_w,
                                   WHEEL_ACCEL, WHEEL_DECEL, dt)
        self._apply_tank_kinematics(dt)

    # -- Driving (point-and-shoot controller) ---------------------------------

    def _drive_step(self, dt: float) -> None:
        r = self.robot
        tx, ty = self.target  # type: ignore[misc]
        dx = tx - r.x
        dy = ty - r.y
        dist = math.hypot(dx, dy)
        target_angle = math.atan2(dy, dx)
        herr = _wrap_pi(target_angle - r.theta)

        # Coarse policy: big heading error → spin in place; medium → arc;
        # small → drive straight with light correction. Slow down near target.
        if abs(herr) > math.pi / 3:
            throttle = 0.0
            turn = math.copysign(0.7, herr)
        elif abs(herr) > 0.15:
            throttle = 0.6
            turn = max(-0.5, min(0.5, herr * 1.5))
        else:
            throttle = 0.85
            turn = max(-0.3, min(0.3, herr * 0.7))

        if dist < 1.5:
            throttle *= max(0.25, dist / 1.5)

        # Arcade → tank → wheel-speed targets (matches player physics)
        raw_left = throttle - turn
        raw_right = throttle + turn
        peak = max(abs(raw_left), abs(raw_right))
        if peak > 1.0:
            raw_left /= peak
            raw_right /= peak
        target_wl = raw_left * WHEEL_MAX_SPEED
        target_wr = raw_right * WHEEL_MAX_SPEED

        self._v_left = _approach(self._v_left, target_wl,
                                  WHEEL_ACCEL, WHEEL_DECEL, dt)
        self._v_right = _approach(self._v_right, target_wr,
                                   WHEEL_ACCEL, WHEEL_DECEL, dt)

        v_lin = 0.5 * (self._v_left + self._v_right)
        v_ang = (self._v_right - self._v_left) / WHEEL_BASE_FT

        r.theta = _wrap_pi(r.theta + v_ang * dt)
        move_x = math.cos(r.theta) * v_lin * dt
        move_y = math.sin(r.theta) * v_lin * dt

        bound = 6.0 - ROBOT_COLL_R_FT
        blocked_x = False
        blocked_y = False
        try_x = max(-bound, min(bound, r.x + move_x))
        if not self.collides_at(try_x, r.y):
            r.x = try_x
        else:
            blocked_x = True
        try_y = max(-bound, min(bound, r.y + move_y))
        if not self.collides_at(r.x, try_y):
            r.y = try_y
        else:
            blocked_y = True

        if blocked_x and blocked_y:
            self._v_left = 0.0
            self._v_right = 0.0
            # CORNERED on a goal — fire the unstuck routine immediately.
            # The 0.4s-of-trying-to-move stuck detector won't catch this
            # because we just zeroed the wheels (so "expected motion" is 0,
            # and the detector resets). Without this hop the bot can sit
            # against a goal forever, waiting to be rescued.
            if self._unstuck_dt <= 0.0:
                self._unstuck_dt = 0.7
                self._unstuck_dir = (-self._unstuck_dir
                                       if self._unstuck_dir else 1.0)
                self.status_msg = (f"R{self.robot_id} cornered on a goal — "
                                    f"backing off")
