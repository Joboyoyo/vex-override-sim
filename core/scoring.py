"""Pure scoring engine for VEX Override 2026-2027.

Single source of truth for points. Pure function, side-effect free, no physics.
Used by HUD, strategy Monte Carlo, and the RL reward function.

Scoring model (per Game Manual v0.1):
- Each pin has two colored halves (half_a_color, half_b_color).
- A pin's scored_half (0 or 1) is the visible half, committed at stack time.
- Visible red half -> 5 to red. Visible blue half -> 5 to blue.
- Visible yellow half -> 10 to whoever owns the pin (SC5):
    * In a quadrant goal: alliance of the quadrant's toggle (RED/BLUE).
      Yellow toggle or unset -> 0 to everyone.
    * In the midfield (tall goal): alliance with more midfield robots.
      Tied -> 0 to everyone.
- Midfield park: +8 per robot in midfield projection at match end (SC6).
- Auto bonus: +12 to higher auto scorer, or +6 to each on tie (SC7).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from core.state import (
    Alliance, Goal, GoalType, Phase, PinColor, ToggleState, World,
)


_RULES_PATH = Path(__file__).with_name("rules.yaml")
_RULES_CACHE: Optional[dict] = None


def load_rules(path: Path = _RULES_PATH) -> dict:
    global _RULES_CACHE
    if _RULES_CACHE is None or path != _RULES_PATH:
        with open(path) as f:
            _RULES_CACHE = yaml.safe_load(f)
    return _RULES_CACHE


# -- Result type ---------------------------------------------------------------


@dataclass
class ScoreLine:
    source: str               # alliance_pin, yellow_pin, midfield, auto_bonus
    alliance: Alliance
    points: int
    detail: str = ""


@dataclass
class ScoreResult:
    red: int = 0
    blue: int = 0
    awp_red: bool = False
    awp_blue: bool = False
    breakdown: list[ScoreLine] = field(default_factory=list)

    def add(self, line: ScoreLine) -> None:
        self.breakdown.append(line)
        if line.alliance == Alliance.RED:
            self.red += line.points
        elif line.alliance == Alliance.BLUE:
            self.blue += line.points

    @property
    def margin(self) -> int:
        return self.red - self.blue


# -- Toggle ownership resolution -----------------------------------------------


def _toggle_yellow_owner(state: ToggleState) -> Alliance:
    """Map a toggle state to which alliance owns yellow pins in that quadrant.
    YELLOW state explicitly assigns ownership to nobody (SC4). UNSET also nobody."""
    if state == ToggleState.RED:
        return Alliance.RED
    if state == ToggleState.BLUE:
        return Alliance.BLUE
    return Alliance.NONE


# -- Scoring components --------------------------------------------------------


def _score_pins(world: World, result: ScoreResult, rules: dict) -> None:
    """Score every VISIBLE half of every pin in every goal's stack.

    Visibility is derived from stack physics (cup orientations + goal coverage)
    via World.visible_halves_in_goal(). Each visible half scores independently
    per SC3. A pin can score 0, 1, or 2 halves depending on the stack.
    """
    alliance_pts = rules["points"]["alliance_pin"]
    yellow_pts = rules["points"]["yellow_pin"]
    midfield_owner = world.midfield_yellow_owner()

    for goal in world.goals:
        visible_by_pin = world.visible_halves_in_goal(goal.id)
        if not visible_by_pin:
            continue
        # Need to map pin_id -> pin object
        pins_in_goal = {p.id: p for p in world.pins
                        if p.in_goal == goal.id}
        for pin_id, half_indices in visible_by_pin.items():
            pin = pins_in_goal.get(pin_id)
            if pin is None:
                continue
            for half_idx in half_indices:
                color = pin.half_a_color if half_idx == 0 else pin.half_b_color
                half_name = "top" if half_idx == 0 else "bottom"

                if color == PinColor.RED:
                    result.add(ScoreLine(
                        source="alliance_pin", alliance=Alliance.RED,
                        points=alliance_pts,
                        detail=f"pin#{pin.id} {half_name} (red) in G{goal.id}",
                    ))
                elif color == PinColor.BLUE:
                    result.add(ScoreLine(
                        source="alliance_pin", alliance=Alliance.BLUE,
                        points=alliance_pts,
                        detail=f"pin#{pin.id} {half_name} (blue) in G{goal.id}",
                    ))
                elif color == PinColor.YELLOW:
                    # SC5: in midfield -> robot majority, else quadrant toggle
                    if goal.in_midfield:
                        owner = midfield_owner
                        src = "midfield majority"
                    else:
                        toggle = world.toggle_by_quadrant(goal.quadrant)
                        owner = _toggle_yellow_owner(toggle.state)
                        src = f"Q{goal.quadrant} toggle={toggle.state.value}"
                    if owner in (Alliance.RED, Alliance.BLUE):
                        result.add(ScoreLine(
                            source="yellow_pin", alliance=owner,
                            points=yellow_pts,
                            detail=f"pin#{pin.id} {half_name} (yellow) in G{goal.id} ({src})",
                        ))
                    # else: 0 to everyone — no line item


def _score_midfield_park(world: World, result: ScoreResult, rules: dict) -> None:
    """+8 per robot in midfield projection at match end (SC6)."""
    if world.phase != Phase.ENDED:
        return
    per_robot = rules["points"]["midfield_park_per_robot"]
    for robot in world.robots:
        if robot.in_midfield:
            result.add(ScoreLine(
                source="midfield",
                alliance=robot.alliance,
                points=per_robot,
                detail=f"robot#{robot.id}",
            ))


def _score_auto_bonus(
    world: World,
    result: ScoreResult,
    rules: dict,
    auto_red: Optional[int],
    auto_blue: Optional[int],
) -> None:
    """SC7 — +12 to auto leader, +6 each on tie (including 0-0)."""
    if auto_red is None or auto_blue is None:
        return
    bonus = rules["points"]["auto_bonus"]
    tie_bonus = rules["points"]["auto_bonus_tie"]
    if auto_red > auto_blue:
        result.add(ScoreLine(
            source="auto_bonus", alliance=Alliance.RED, points=bonus,
            detail=f"auto leader {auto_red}-{auto_blue}",
        ))
    elif auto_blue > auto_red:
        result.add(ScoreLine(
            source="auto_bonus", alliance=Alliance.BLUE, points=bonus,
            detail=f"auto leader {auto_blue}-{auto_red}",
        ))
    else:
        # SC7b — tied auto: each alliance gets 6
        result.add(ScoreLine(
            source="auto_bonus", alliance=Alliance.RED, points=tie_bonus,
            detail=f"auto tie {auto_red}-{auto_blue}",
        ))
        result.add(ScoreLine(
            source="auto_bonus", alliance=Alliance.BLUE, points=tie_bonus,
            detail=f"auto tie {auto_red}-{auto_blue}",
        ))


# -- AWP evaluation ------------------------------------------------------------


def _pins_scored_for(world: World, alliance: Alliance) -> list[tuple[int, int]]:
    """Return [(pin_id, goal_id), ...] for pins that have at least one visible
    half scoring for this alliance. Used by AWP counting.

    Per SC8, pins on the opposing side of the Autonomous Line do not count."""
    out = []
    midfield_owner = world.midfield_yellow_owner()
    opposing = Alliance.BLUE if alliance == Alliance.RED else Alliance.RED

    for goal in world.goals:
        if goal.awp_side == opposing:
            continue
        visible_by_pin = world.visible_halves_in_goal(goal.id)
        pins_in_goal = {p.id: p for p in world.pins if p.in_goal == goal.id}
        for pin_id, half_indices in visible_by_pin.items():
            pin = pins_in_goal.get(pin_id)
            if pin is None:
                continue
            scores_for_us = False
            for half_idx in half_indices:
                color = pin.half_a_color if half_idx == 0 else pin.half_b_color
                if color == PinColor.RED and alliance == Alliance.RED:
                    scores_for_us = True
                elif color == PinColor.BLUE and alliance == Alliance.BLUE:
                    scores_for_us = True
                elif color == PinColor.YELLOW:
                    if goal.in_midfield:
                        if midfield_owner == alliance:
                            scores_for_us = True
                    else:
                        toggle = world.toggle_by_quadrant(goal.quadrant)
                        if _toggle_yellow_owner(toggle.state) == alliance:
                            scores_for_us = True
            if scores_for_us:
                out.append((pin.id, goal.id))
    return out


def _evaluate_awp(world: World, result: ScoreResult, rules: dict) -> None:
    """SC8 — AWP at end of auto.
    Tasks: >=7 pins for alliance, >=3 distinct goals each with >=2 pins,
    neither robot touching field perimeter, no violations (not tracked here)."""
    awp_cfg = rules["awp"]
    min_pins = awp_cfg["pins_placed_for_alliance"]
    min_goals = awp_cfg["distinct_goals_with_min_pins"]
    min_per_goal = awp_cfg["min_pins_per_distinct_goal"]

    for alliance in (Alliance.RED, Alliance.BLUE):
        scored = _pins_scored_for(world, alliance)
        if len(scored) < min_pins:
            continue

        # Group by goal
        per_goal: dict[int, int] = {}
        for _, gid in scored:
            per_goal[gid] = per_goal.get(gid, 0) + 1
        qualifying_goals = sum(1 for c in per_goal.values() if c >= min_per_goal)
        if qualifying_goals < min_goals:
            continue

        # Robots on this alliance must not touch field perimeter
        if any(r.alliance == alliance and r.touching_field_perimeter for r in world.robots):
            continue

        if alliance == Alliance.RED:
            result.awp_red = True
        else:
            result.awp_blue = True


# -- Public entry point --------------------------------------------------------


def score(
    world: World,
    auto_red: Optional[int] = None,
    auto_blue: Optional[int] = None,
    rules: Optional[dict] = None,
    evaluate_awp: bool = False,
) -> ScoreResult:
    """Compute current score from world state.

    auto_red / auto_blue: scores at the moment auto ended. Pass during driver
    and after to apply the auto bonus to the running total.

    evaluate_awp: set True when computing the score at the end of auto to
    populate awp_red / awp_blue fields.
    """
    if rules is None:
        rules = load_rules()
    result = ScoreResult()
    _score_pins(world, result, rules)
    _score_midfield_park(world, result, rules)
    _score_auto_bonus(world, result, rules, auto_red, auto_blue)
    if evaluate_awp:
        _evaluate_awp(world, result, rules)
    return result


def score_at_auto_end(world: World, rules: Optional[dict] = None) -> tuple[int, int, bool, bool]:
    """Snapshot helper for the auto -> driver transition.
    Returns (red, blue, awp_red, awp_blue)."""
    r = score(world, rules=rules, evaluate_awp=True)
    return r.red, r.blue, r.awp_red, r.awp_blue
