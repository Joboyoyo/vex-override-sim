"""World state dataclasses for VEX Override 2026-2027 sim.

Design rules:
- Pure data. No behavior, no physics, no rendering.
- All numeric IDs so JSON round-trips are trivial.
- Coordinates: field centered at (0, 0), x and y in feet, range [-6, +6].
- All enums are str-valued for human-readable JSON.

Pin model (post-v0.1 manual review):
- Each pin has TWO colored halves (half_a_color, half_b_color).
- Pin "identity" is the unordered pair of half colors; the four legal pin types
  are red/blue, red/yellow, blue/yellow, yellow/yellow.
- Pin orientation is committed at stack time via `scored_half`:
    None = not placed (or both halves covered)
    0    = half_a is the visible/scored half
    1    = half_b is the visible/scored half
- Each pin contributes at most ONE visible half per the conservation law
  (a stack of N pins produces exactly N visible halves), so we model the
  scoring decision as a single int per pin, not two booleans.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Optional


# -- Enums ---------------------------------------------------------------------


class Alliance(str, Enum):
    RED = "red"
    BLUE = "blue"
    NEUTRAL = "neutral"      # for goals + toggles in yellow state
    NONE = "none"            # toggle unset/contested; midfield ownership tied


class PinColor(str, Enum):
    RED = "red"
    BLUE = "blue"
    YELLOW = "yellow"


class GoalType(str, Enum):
    SHORT = "short"
    TALL = "tall"
    ALLIANCE = "alliance"


class ToggleState(str, Enum):
    RED = "red"
    BLUE = "blue"
    YELLOW = "yellow"        # SC4 — third valid set state (yellow = no yellow ownership)
    UNSET = "unset"          # not seated, or robot in contact


class Phase(str, Enum):
    AUTO = "auto"
    DRIVER = "driver"
    ENDED = "ended"


# -- Field positioning ---------------------------------------------------------

# Quadrants are diagonal triangles split by the two diagonals y=x and y=-x.
# Numbered clockwise starting from TOP:
#   Q0 = TOP    (y >= |x|)
#   Q1 = RIGHT  (x >= |y|)
#   Q2 = BOTTOM (-y >= |x|)
#   Q3 = LEFT   (-x >= |y|)
#
# A toggle lives at the outer midpoint of each quadrant's wall:
#   Q0 -> top    (0, 6)
#   Q1 -> right  (6, 0)
#   Q2 -> bottom (0, -6)
#   Q3 -> left   (-6, 0)
#
# The Autonomous Line is the diagonal y = -x (from top-left to bottom-right).
# Red side: y + x < 0 (LEFT and BOTTOM quadrants).
# Blue side: y + x > 0 (TOP and RIGHT quadrants).


def quadrant_of(x: float, y: float) -> int:
    """Diagonal-quadrant index: 0=TOP, 1=RIGHT, 2=BOTTOM, 3=LEFT.

    Boundaries split at 45° (along y = x and y = -x). On the boundary, ties
    resolve in the order TOP > RIGHT > BOTTOM > LEFT (arbitrary; only matters
    for points exactly on the diagonal lines)."""
    ax = abs(x)
    ay = abs(y)
    if y >= ax:
        return 0
    if x >= ay:
        return 1
    if -y >= ax:
        return 2
    return 3


# -- Game objects --------------------------------------------------------------


@dataclass
class Pin:
    id: int
    half_a_color: PinColor       # immutable physical property
    half_b_color: PinColor       # immutable physical property
    x: float = 0.0
    y: float = 0.0
    in_goal: Optional[int] = None
    in_cup: Optional[int] = None
    scored_half: Optional[int] = None
        # None = not Placed (no visible half)
        # 0    = half_a is the visible half
        # 1    = half_b is the visible half

    def visible_color(self) -> Optional[PinColor]:
        """The color of the currently visible half, or None if not Placed."""
        if self.scored_half is None:
            return None
        return self.half_a_color if self.scored_half == 0 else self.half_b_color


@dataclass
class Cup:
    """Per SC2, a Cup can only be 'Placed' if it's nested on another Placed Pin.
    Cups never sit directly in a goal; they always have a pin below them.

    Fields:
      in_goal:  the goal this cup belongs to (cached from the chain root)
      on_pin:   the pin this cup is nested on (required for SC2 compliance)
    """
    id: int
    x: float = 0.0
    y: float = 0.0
    in_goal: Optional[int] = None
    on_pin: Optional[int] = None      # SC2: cup is nested on this pin
    clear_face_up: bool = True
    nested_pin_ids: list[int] = field(default_factory=list)


@dataclass
class Goal:
    id: int
    type: GoalType
    alliance: Alliance           # RED/BLUE for alliance goals; NEUTRAL for short/tall
    x: float
    y: float
    quadrant: int                # which quadrant's toggle applies to yellow pins here
    in_midfield: bool = False    # tall goal sits in midfield -> yellow ownership by robot majority
    awp_side: Alliance = Alliance.NEUTRAL
        # Which side of the Autonomous Line this goal is on, for AWP exclusion.
        # NEUTRAL = midfield, counts for either alliance's AWP.


@dataclass
class Toggle:
    """A toggle's `state` is what the SCORING engine reads.

    Per SC4 a toggle is only "set to a color" when it's NOT in contact with any
    robot. We model that by having a separate `resting_state` (what the toggle
    settles into when untouched) and overwriting `state = UNSET` while any
    robot is touching it. The app updates `state` each frame from
    `resting_state` + contact detection."""
    id: int
    quadrant: int
    state: ToggleState = ToggleState.YELLOW
    resting_state: ToggleState = ToggleState.YELLOW
    x: float = 0.0
    y: float = 0.0


@dataclass
class Robot:
    id: int
    alliance: Alliance           # RED or BLUE
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0
    holding_pin_id: Optional[int] = None
    holding_cup_id: Optional[int] = None
    touching_toggle_id: Optional[int] = None
    touching_field_perimeter: bool = False
    in_midfield: bool = False


# -- World ---------------------------------------------------------------------


@dataclass
class World:
    """Full game state at one instant."""
    pins: list[Pin] = field(default_factory=list)
    cups: list[Cup] = field(default_factory=list)
    goals: list[Goal] = field(default_factory=list)
    toggles: list[Toggle] = field(default_factory=list)
    robots: list[Robot] = field(default_factory=list)
    phase: Phase = Phase.AUTO
    time_elapsed: float = 0.0

    # Per-alliance match-load counter. SG11 / bill of materials: each alliance
    # has 10 pin match loads available per match. Incremented every time ANY
    # robot (player or bot) on that alliance spawns a pin+cup from a loader.
    match_loads_red: int = 0
    match_loads_blue: int = 0

    # -- Convenience lookups --

    def goal_by_id(self, gid: int) -> Goal:
        return next(g for g in self.goals if g.id == gid)

    def toggle_by_quadrant(self, q: int) -> Toggle:
        return next(t for t in self.toggles if t.quadrant == q)

    def midfield_robot_counts(self) -> tuple[int, int]:
        """Return (red_count, blue_count) of robots currently in midfield."""
        red = sum(1 for r in self.robots if r.in_midfield and r.alliance == Alliance.RED)
        blue = sum(1 for r in self.robots if r.in_midfield and r.alliance == Alliance.BLUE)
        return red, blue

    def midfield_yellow_owner(self) -> Alliance:
        """SC5b — alliance with more midfield robots owns yellow pins in midfield.
        Tie returns Alliance.NONE (no scoring)."""
        red, blue = self.midfield_robot_counts()
        if red > blue:
            return Alliance.RED
        if blue > red:
            return Alliance.BLUE
        return Alliance.NONE

    # -- Stack helpers (SC2 alternation rule) --

    def stack_in_goal(self, goal_id: int) -> list[tuple[str, object]]:
        """Return the ordered stack of (kind, object) in a goal, bottom -> top.

        Per SC2 the stack is forced to alternate: the bottom must be a pin in
        the goal (pin.in_goal=goal_id, pin.in_cup=None), then a cup on that pin
        (cup.on_pin=...), then a pin in that cup (pin.in_cup=...), and so on.

        Returns [] if the goal is empty.
        """
        bottom = next((p for p in self.pins
                       if p.in_goal == goal_id and p.in_cup is None), None)
        if bottom is None:
            return []

        stack: list[tuple[str, object]] = [("pin", bottom)]
        while True:
            last_kind, last_obj = stack[-1]
            if last_kind == "pin":
                cup_above = next((c for c in self.cups
                                  if c.on_pin == last_obj.id), None)
                if cup_above is None:
                    break
                stack.append(("cup", cup_above))
            else:
                pin_above = next((p for p in self.pins
                                  if p.in_cup == last_obj.id), None)
                if pin_above is None:
                    break
                stack.append(("pin", pin_above))
        return stack

    def next_placeable_kind(self, goal_id: int) -> str:
        """Return 'pin' or 'cup' — what kind of object the SC2 rule says you
        can place next on top of this goal's stack."""
        stack = self.stack_in_goal(goal_id)
        if not stack:
            return "pin"
        last_kind, _ = stack[-1]
        return "cup" if last_kind == "pin" else "pin"

    def visible_halves_in_goal(self, goal_id: int) -> dict[int, set[int]]:
        """For each pin in this goal's stack, return the set of half indices
        that are physically visible (= score per SC3):
            0 = top half (half_a)
            1 = bottom half (half_b)

        Physics:
        - Bottom pin (in_cup=None, at index 0 in stack): bottom half is covered
          by the goal interior -> never visible.
        - Top pin (last in stack, no cup above): top half exposed.
        - A pin between cups: its bottom half is inside the lower cup's TOP
          face, its top half is inside the upper cup's BOTTOM face.
        - Cup face visibility:
            clear_face_up=True  -> clear is the cup's TOP face,
                                   gray is the cup's BOTTOM face.
            clear_face_up=False -> opposite.
        - A pin half is visible if the cup face enclosing it is clear, or if
          that side has no cup at all (top of stack only — the goal interior
          does cover the bottom).

        Returns {} if the goal is empty.
        """
        stack = self.stack_in_goal(goal_id)
        out: dict[int, set[int]] = {}

        for i, (kind, obj) in enumerate(stack):
            if kind != "pin":
                continue
            visible: set[int] = set()

            # Bottom half (index 1 = half_b)
            if i == 0:
                # bottom pin -> bottom is in the goal, covered
                pass
            else:
                cup_below = stack[i - 1][1]  # the cup directly under this pin
                # Pin's bottom is enclosed by cup_below's TOP face.
                # clear_face_up=True -> top of cup is clear -> visible.
                if cup_below.clear_face_up:
                    visible.add(1)

            # Top half (index 0 = half_a)
            if i == len(stack) - 1:
                # Top of stack -> top half exposed, visible
                visible.add(0)
            else:
                cup_above = stack[i + 1][1]
                # Pin's top is enclosed by cup_above's BOTTOM face.
                # clear_face_up=False -> bottom of cup is clear -> visible.
                if not cup_above.clear_face_up:
                    visible.add(0)

            out[obj.id] = visible
        return out

    # -- Serialization --

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    @classmethod
    def from_json(cls, s: str) -> World:
        d = json.loads(s)
        return cls(
            pins=[
                Pin(
                    id=p["id"],
                    half_a_color=PinColor(p["half_a_color"]),
                    half_b_color=PinColor(p["half_b_color"]),
                    x=p["x"], y=p["y"],
                    in_goal=p["in_goal"],
                    in_cup=p["in_cup"],
                    scored_half=p["scored_half"],
                )
                for p in d["pins"]
            ],
            cups=[
                Cup(
                    id=c["id"],
                    x=c.get("x", 0.0), y=c.get("y", 0.0),
                    in_goal=c.get("in_goal"),
                    on_pin=c.get("on_pin"),
                    clear_face_up=c.get("clear_face_up", True),
                    nested_pin_ids=c.get("nested_pin_ids", []),
                )
                for c in d["cups"]
            ],
            goals=[
                Goal(
                    id=g["id"],
                    type=GoalType(g["type"]),
                    alliance=Alliance(g["alliance"]),
                    x=g["x"], y=g["y"],
                    quadrant=g["quadrant"],
                    in_midfield=g.get("in_midfield", False),
                    awp_side=Alliance(g.get("awp_side", Alliance.NEUTRAL.value)),
                )
                for g in d["goals"]
            ],
            toggles=[
                Toggle(
                    id=t["id"],
                    quadrant=t["quadrant"],
                    state=ToggleState(t["state"]),
                    resting_state=ToggleState(t.get("resting_state", t["state"])),
                    x=t.get("x", 0.0),
                    y=t.get("y", 0.0),
                )
                for t in d["toggles"]
            ],
            robots=[
                Robot(
                    id=r["id"],
                    alliance=Alliance(r["alliance"]),
                    x=r.get("x", 0.0), y=r.get("y", 0.0),
                    theta=r.get("theta", 0.0),
                    holding_pin_id=r.get("holding_pin_id"),
                    holding_cup_id=r.get("holding_cup_id"),
                    touching_toggle_id=r.get("touching_toggle_id"),
                    touching_field_perimeter=r.get("touching_field_perimeter", False),
                    in_midfield=r.get("in_midfield", False),
                )
                for r in d["robots"]
            ],
            phase=Phase(d["phase"]),
            time_elapsed=d["time_elapsed"],
            match_loads_red=d.get("match_loads_red", 0),
            match_loads_blue=d.get("match_loads_blue", 0),
        )


# -- Construction helpers ------------------------------------------------------

# Common pin templates — caller provides id, x, y.

def red_yellow_pin(id: int, **kw) -> Pin:
    return Pin(id=id, half_a_color=PinColor.RED, half_b_color=PinColor.YELLOW, **kw)


def blue_yellow_pin(id: int, **kw) -> Pin:
    return Pin(id=id, half_a_color=PinColor.BLUE, half_b_color=PinColor.YELLOW, **kw)


def red_blue_pin(id: int, **kw) -> Pin:
    return Pin(id=id, half_a_color=PinColor.RED, half_b_color=PinColor.BLUE, **kw)


def yellow_yellow_pin(id: int, **kw) -> Pin:
    return Pin(id=id, half_a_color=PinColor.YELLOW, half_b_color=PinColor.YELLOW, **kw)


# -- Stack builder ------------------------------------------------------------


def build_stack(
    world: World, goal_id: int, *,
    top_color: PinColor, bottom_color: PinColor,
    n_pins: int, cups_clear_face_up: bool = False,
) -> None:
    """Construct a SC2-compliant stack of `n_pins` pins in the given goal,
    with cups between each pair.

    half_a (top) = `top_color`, half_b (bottom) = `bottom_color` for every pin.
    All N-1 cups use the same `cups_clear_face_up` orientation.

    Cup orientation guide:
      cups_clear_face_up=False  -> cup's clear face is at the BOTTOM, gray at
                                   the TOP. Every pin's TOP half (= top_color)
                                   becomes visible; bottom halves are hidden.
                                   This is "tops visible" mode -> use this if
                                   you want the TOP color to be the scored
                                   half for every pin (= the simple +N per
                                   pin model).
      cups_clear_face_up=True   -> opposite: bottom halves are exposed; top
                                   halves are hidden. The very bottom pin
                                   ends up with 0 visible halves (covered by
                                   goal AND gray-bottom cup above), and the
                                   very top pin has BOTH halves visible.
    """
    prev_pin_id: Optional[int] = None
    prev_cup_id: Optional[int] = None
    next_pin_id = max((p.id for p in world.pins), default=-1) + 1
    next_cup_id = max((c.id for c in world.cups), default=-1) + 1

    for i in range(n_pins):
        pin = Pin(
            id=next_pin_id,
            half_a_color=top_color, half_b_color=bottom_color,
            in_goal=goal_id, in_cup=prev_cup_id,
        )
        world.pins.append(pin)
        prev_pin_id = next_pin_id
        next_pin_id += 1

        if i < n_pins - 1:
            cup = Cup(
                id=next_cup_id, in_goal=goal_id, on_pin=prev_pin_id,
                clear_face_up=cups_clear_face_up,
            )
            world.cups.append(cup)
            prev_cup_id = next_cup_id
            next_cup_id += 1
