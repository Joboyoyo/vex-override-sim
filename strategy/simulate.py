"""Match simulation + Monte Carlo runner.

`play_one_match` builds a final-state World from two strategy specs under
randomness from a MatchScenario, then evaluates it through the scoring engine.

`run_matchup` repeats this N times and aggregates statistics.

Simulation model (intentionally simplified for first-pass MC):
- No defense on opponent's match-load runs (rare in V5RC)
- Toggle conflicts: latest setter wins, then either side may deny at buzzer
- Midfield positioning is a coin flip per robot the strategy commits
- Robot perimeter touch is a single Bernoulli per alliance (AWP filter)

These are *modelling choices*, not rule readings. Edit the scenario to reflect
the real opponent you expect.
"""

from __future__ import annotations

import random
import statistics
from dataclasses import dataclass, field
from typing import Optional

from core.scoring import score, score_at_auto_end
from core.state import (
    Alliance, Cup, Goal, GoalType, Phase, Pin, PinColor, Robot, Toggle,
    ToggleState, World,
)

from .scenario import MatchScenario, StrategySpec


# -- Field construction --------------------------------------------------------


def _make_mc_field() -> World:
    """Diagonal-quadrant field layout matching pygame_view.make_empty_world.
    Red side = LEFT (Q3) + BOTTOM (Q2). Blue side = TOP (Q0) + RIGHT (Q1)."""
    goals = [
        Goal(0, GoalType.ALLIANCE, Alliance.RED,  x=-4.0, y=-2.0, quadrant=3, awp_side=Alliance.RED),
        Goal(1, GoalType.ALLIANCE, Alliance.RED,  x=-2.0, y=-4.0, quadrant=2, awp_side=Alliance.RED),
        Goal(2, GoalType.ALLIANCE, Alliance.BLUE, x=2.0,  y=4.0,  quadrant=0, awp_side=Alliance.BLUE),
        Goal(3, GoalType.ALLIANCE, Alliance.BLUE, x=4.0,  y=2.0,  quadrant=1, awp_side=Alliance.BLUE),
        Goal(4, GoalType.SHORT, Alliance.NEUTRAL, x=-2.0, y=4.0,  quadrant=0, awp_side=Alliance.BLUE),
        Goal(5, GoalType.SHORT, Alliance.NEUTRAL, x=4.0,  y=-2.0, quadrant=1, awp_side=Alliance.BLUE),
        Goal(6, GoalType.SHORT, Alliance.NEUTRAL, x=2.0,  y=-4.0, quadrant=2, awp_side=Alliance.RED),
        Goal(7, GoalType.SHORT, Alliance.NEUTRAL, x=-4.0, y=2.0,  quadrant=3, awp_side=Alliance.RED),
        Goal(8, GoalType.TALL, Alliance.NEUTRAL,  x=0.0,  y=0.0,  quadrant=0,
             in_midfield=True, awp_side=Alliance.NEUTRAL),
    ]
    toggles = [
        Toggle(0, quadrant=0, state=ToggleState.YELLOW, x=0.0,  y=6.0),
        Toggle(1, quadrant=1, state=ToggleState.YELLOW, x=6.0,  y=0.0),
        Toggle(2, quadrant=2, state=ToggleState.YELLOW, x=0.0,  y=-6.0),
        Toggle(3, quadrant=3, state=ToggleState.YELLOW, x=-6.0, y=0.0),
    ]
    return World(goals=goals, toggles=toggles, phase=Phase.AUTO)


# -- Alliance action simulation ------------------------------------------------


@dataclass
class AllianceActions:
    """Aggregated outcome of one alliance's randomized actions."""
    auto_pins: list[tuple[int, str]] = field(default_factory=list)
        # (goal_id, orient: "safe"|"yellow")
    auto_toggles_set: list[int] = field(default_factory=list)
        # quadrant IDs where the toggle was successfully set in auto
    robots_clear_in_auto: bool = True
        # Both robots ended auto not touching field perimeter
    driver_pins: list[tuple[int, str]] = field(default_factory=list)
        # Match-load placements (goal_id, orient)
    robots_in_midfield: int = 0
        # 0/1/2 robots reached midfield at match end


def _simulate_alliance(
    strat: StrategySpec, scenario: MatchScenario, rng: random.Random,
) -> AllianceActions:
    a = AllianceActions()

    # Auto pin attempts — distribute across declared goals round-robin
    for i in range(strat.auto_pin_attempts):
        if rng.random() < scenario.auto_pin_success:
            goal_id = strat.auto_goal_ids[i % len(strat.auto_goal_ids)]
            a.auto_pins.append((goal_id, strat.auto_pin_orient))

    # Auto toggle attempts
    for q in strat.auto_toggle_quadrants:
        if rng.random() < scenario.auto_toggle_set_success:
            a.auto_toggles_set.append(q)

    a.robots_clear_in_auto = rng.random() < scenario.p_robots_clear_perimeter

    # Driver match-load attempts
    for _ in range(strat.match_load_attempts):
        if rng.random() < scenario.driver_pin_success:
            a.driver_pins.append((strat.match_load_goal_id, strat.match_load_orient))

    # Midfield arrival per robot the strategy commits
    for _ in range(strat.midfield_robots):
        if rng.random() < scenario.p_robot_reaches_midfield:
            a.robots_in_midfield += 1

    return a


# -- Pin placement helpers -----------------------------------------------------

# State the simulator tracks per goal as it builds stacks across phases.
# Pins added to the same goal must reference the previous top of stack via
# in_cup / on_pin to satisfy SC2 and the physics-derived visibility.
_StackBuilder = dict[int, dict]   # goal_id -> {"last_pin_id": int|None, "last_cup_id": int|None}


def _new_stack_state() -> _StackBuilder:
    return {}


def _place_pin(
    world: World, alliance: Alliance, goal_id: int, orient: str,
    stack_state: _StackBuilder,
) -> None:
    """Add a pin to the given goal's stack, inserting a cup-between-pins as
    required by SC2. Tracks per-goal stack state via `stack_state`.

    orient="safe":   alliance color is the TOP of the pin (half_a),
                     yellow is the BOTTOM. Cups oriented clear_face_up=False
                     so every pin's top is visible -> +5 per pin guaranteed.
    orient="yellow": yellow is the TOP, alliance color is the BOTTOM. Same
                     cup orientation -> yellow tops are visible -> +10 per
                     pin if toggle owned, else 0 (or +10 to opp if they own).
    """
    # Decide pin colors per alliance + orient
    if alliance == Alliance.RED:
        if orient == "safe":
            top, bot = PinColor.RED, PinColor.YELLOW
        else:
            top, bot = PinColor.YELLOW, PinColor.RED
    else:
        if orient == "safe":
            top, bot = PinColor.BLUE, PinColor.YELLOW
        else:
            top, bot = PinColor.YELLOW, PinColor.BLUE

    state = stack_state.setdefault(
        goal_id, {"last_pin_id": None, "last_cup_id": None})

    # If a previous pin exists, we need to put a cup on it first (SC2).
    if state["last_pin_id"] is not None and state["last_cup_id"] is None:
        next_cup_id = max((c.id for c in world.cups), default=-1) + 1
        world.cups.append(Cup(
            id=next_cup_id, in_goal=goal_id,
            on_pin=state["last_pin_id"], clear_face_up=False,
        ))
        state["last_cup_id"] = next_cup_id

    # Add the new pin (in the cup above the previous pin, if any)
    next_pin_id = max((p.id for p in world.pins), default=-1) + 1
    world.pins.append(Pin(
        id=next_pin_id,
        half_a_color=top, half_b_color=bot,
        in_goal=goal_id, in_cup=state["last_cup_id"],
    ))
    state["last_pin_id"] = next_pin_id
    state["last_cup_id"] = None   # next pin needs a fresh cup again


# -- Buzzer toggle resolution --------------------------------------------------


def _resolve_buzzer_toggles(
    world: World, red_strat: StrategySpec, blue_strat: StrategySpec,
    red_actions: AllianceActions, blue_actions: AllianceActions,
    scenario: MatchScenario, rng: random.Random,
) -> None:
    """For each toggle that was set in auto, check:
      1. Survives natural opponent denial pressure (p_my_toggle_held)
      2. If still owned, may be specifically denied by opponent's deny list
         (p_opp_toggle_denied)
    A denied toggle ends as UNSET (cleanest model — partial states aren't valid).
    """
    # Step 1: pressure on auto-set toggles
    for q in red_actions.auto_toggles_set:
        t = world.toggle_by_quadrant(q)
        if t.state == ToggleState.RED:  # may have been overwritten by blue auto
            if rng.random() > scenario.p_my_toggle_held:
                t.state = ToggleState.UNSET
    for q in blue_actions.auto_toggles_set:
        t = world.toggle_by_quadrant(q)
        if t.state == ToggleState.BLUE:
            if rng.random() > scenario.p_my_toggle_held:
                t.state = ToggleState.UNSET

    # Step 2: active denial — only matters against toggles still owned by opponent
    for q in red_strat.deny_opp_quadrants:
        t = world.toggle_by_quadrant(q)
        if t.state == ToggleState.BLUE and rng.random() < scenario.p_opp_toggle_denied:
            t.state = ToggleState.UNSET
    for q in blue_strat.deny_opp_quadrants:
        t = world.toggle_by_quadrant(q)
        if t.state == ToggleState.RED and rng.random() < scenario.p_opp_toggle_denied:
            t.state = ToggleState.UNSET


# -- One-match simulation ------------------------------------------------------


@dataclass
class MatchOutcome:
    red_score: int
    blue_score: int
    awp_red: bool
    awp_blue: bool
    auto_red: int
    auto_blue: int


def play_one_match(
    red: StrategySpec, blue: StrategySpec,
    scenario: MatchScenario, rng: Optional[random.Random] = None,
) -> MatchOutcome:
    if rng is None:
        rng = random.Random()

    world = _make_mc_field()
    stack_state = _new_stack_state()

    red_actions = _simulate_alliance(red, scenario, rng)
    blue_actions = _simulate_alliance(blue, scenario, rng)

    # Place auto pins
    for goal_id, orient in red_actions.auto_pins:
        _place_pin(world, Alliance.RED, goal_id, orient, stack_state)
    for goal_id, orient in blue_actions.auto_pins:
        _place_pin(world, Alliance.BLUE, goal_id, orient, stack_state)

    # Apply auto toggles — blue overwrites red on contested quadrants (modelling
    # choice; in reality contested = SG7 violation, but symmetric assumption).
    for q in red_actions.auto_toggles_set:
        world.toggle_by_quadrant(q).state = ToggleState.RED
    for q in blue_actions.auto_toggles_set:
        world.toggle_by_quadrant(q).state = ToggleState.BLUE

    # Robots — set perimeter contact for AWP eligibility snapshot
    world.robots = [
        Robot(id=0, alliance=Alliance.RED,
              touching_field_perimeter=not red_actions.robots_clear_in_auto),
        Robot(id=1, alliance=Alliance.RED,
              touching_field_perimeter=not red_actions.robots_clear_in_auto),
        Robot(id=2, alliance=Alliance.BLUE,
              touching_field_perimeter=not blue_actions.robots_clear_in_auto),
        Robot(id=3, alliance=Alliance.BLUE,
              touching_field_perimeter=not blue_actions.robots_clear_in_auto),
    ]

    # Snapshot auto score + AWP eligibility at end of auto
    auto_red, auto_blue, awp_r, awp_b = score_at_auto_end(world)

    # Clear perimeter flags — they only matter for AWP snapshot
    for r in world.robots:
        r.touching_field_perimeter = False

    # Driver pin placements
    for goal_id, orient in red_actions.driver_pins:
        _place_pin(world, Alliance.RED, goal_id, orient, stack_state)
    for goal_id, orient in blue_actions.driver_pins:
        _place_pin(world, Alliance.BLUE, goal_id, orient, stack_state)

    # Buzzer toggle warfare
    _resolve_buzzer_toggles(world, red, blue, red_actions, blue_actions, scenario, rng)

    # Midfield positioning
    for i in range(red_actions.robots_in_midfield):
        world.robots[i].in_midfield = True
    for i in range(blue_actions.robots_in_midfield):
        world.robots[2 + i].in_midfield = True

    world.phase = Phase.ENDED
    result = score(world, auto_red=auto_red, auto_blue=auto_blue)

    return MatchOutcome(
        red_score=result.red, blue_score=result.blue,
        awp_red=awp_r, awp_blue=awp_b,
        auto_red=auto_red, auto_blue=auto_blue,
    )


# -- Monte Carlo aggregation ---------------------------------------------------


@dataclass
class MatchupStats:
    red_name: str
    blue_name: str
    trials: int
    red_scores: list[int]
    blue_scores: list[int]
    awp_red_count: int
    awp_blue_count: int
    red_wins: int
    blue_wins: int
    ties: int

    @property
    def red_mean(self) -> float: return statistics.mean(self.red_scores)
    @property
    def red_std(self) -> float:  return statistics.pstdev(self.red_scores)
    @property
    def blue_mean(self) -> float: return statistics.mean(self.blue_scores)
    @property
    def blue_std(self) -> float:  return statistics.pstdev(self.blue_scores)
    @property
    def red_win_rate(self) -> float:  return self.red_wins / self.trials
    @property
    def blue_win_rate(self) -> float: return self.blue_wins / self.trials
    @property
    def tie_rate(self) -> float:      return self.ties / self.trials
    @property
    def red_awp_rate(self) -> float:  return self.awp_red_count / self.trials
    @property
    def blue_awp_rate(self) -> float: return self.awp_blue_count / self.trials
    @property
    def margin_mean(self) -> float:
        return statistics.mean(r - b for r, b in zip(self.red_scores, self.blue_scores))

    def summary(self) -> str:
        return (
            f"{self.red_name} vs {self.blue_name}  ({self.trials} trials)\n"
            f"  RED:  mean {self.red_mean:6.1f} ± {self.red_std:5.1f}   "
            f"win {self.red_win_rate:5.1%}   AWP {self.red_awp_rate:5.1%}\n"
            f"  BLUE: mean {self.blue_mean:6.1f} ± {self.blue_std:5.1f}   "
            f"win {self.blue_win_rate:5.1%}   AWP {self.blue_awp_rate:5.1%}\n"
            f"  Ties: {self.tie_rate:5.1%}   Mean margin (R-B): {self.margin_mean:+6.1f}"
        )


def run_matchup(
    red: StrategySpec, blue: StrategySpec,
    scenario: MatchScenario, trials: int = 1000,
    seed: Optional[int] = None,
) -> MatchupStats:
    rng = random.Random(seed)
    red_scores: list[int] = []
    blue_scores: list[int] = []
    awp_r = awp_b = 0
    wr = wb = ties = 0

    for _ in range(trials):
        out = play_one_match(red, blue, scenario, rng)
        red_scores.append(out.red_score)
        blue_scores.append(out.blue_score)
        if out.awp_red: awp_r += 1
        if out.awp_blue: awp_b += 1
        if out.red_score > out.blue_score: wr += 1
        elif out.blue_score > out.red_score: wb += 1
        else: ties += 1

    return MatchupStats(
        red_name=red.name, blue_name=blue.name, trials=trials,
        red_scores=red_scores, blue_scores=blue_scores,
        awp_red_count=awp_r, awp_blue_count=awp_b,
        red_wins=wr, blue_wins=wb, ties=ties,
    )
