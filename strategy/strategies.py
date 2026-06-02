"""Predefined strategies. Each factory returns an alliance-aware StrategySpec.

Goal / quadrant layout (matches tests/test_scoring.py::make_empty_world):

    Quadrants:    1 (TL) | 0 (TR)         Red's side: x >= 0 (q0, q3)
                  -------+-------         Blue's side: x < 0  (q1, q2)
                  2 (BL) | 3 (BR)

    Goals:
      0: red alliance,  quadrant 0 (TR)
      1: red alliance,  quadrant 3 (BR)
      2: blue alliance, quadrant 1 (TL)
      3: blue alliance, quadrant 2 (BL)
      4-7: neutral short, one per quadrant
      8: neutral tall, in midfield
"""

from __future__ import annotations

from core.state import Alliance

from .scenario import StrategySpec


# Side-specific constants (diagonal-quadrant layout)
# Red side: LEFT (Q3) + BOTTOM (Q2). Blue side: TOP (Q0) + RIGHT (Q1).
_RED = {
    "alliance_goals": [0, 1],          # G0 (LEFT), G1 (BOTTOM)
    "side_neutral_shorts": [6, 7],     # G6 (BOTTOM), G7 (LEFT)
    "side_goals": [0, 1, 6, 7],
    "own_quadrants": [2, 3],           # BOTTOM, LEFT
    "opp_quadrants": [0, 1],           # TOP, RIGHT
}
_BLUE = {
    "alliance_goals": [2, 3],          # G2 (TOP), G3 (RIGHT)
    "side_neutral_shorts": [4, 5],     # G4 (TOP), G5 (RIGHT)
    "side_goals": [2, 3, 4, 5],
    "own_quadrants": [0, 1],           # TOP, RIGHT
    "opp_quadrants": [2, 3],           # BOTTOM, LEFT
}


def _layout(alliance: Alliance) -> dict:
    return _RED if alliance == Alliance.RED else _BLUE


# -- Strategy templates --------------------------------------------------------


def safe_strategy(alliance: Alliance) -> StrategySpec:
    """User's primary strategy: alliance-color commits everywhere.
    Locks 50 pts of match-load value with zero toggle-dependence."""
    L = _layout(alliance)
    return StrategySpec(
        name=f"safe_{alliance.value}",
        alliance=alliance,
        # Auto: spread 7 pins across 3 distinct goals on your side for AWP
        auto_pin_attempts=7,
        auto_pin_orient="safe",
        auto_goal_ids=[L["alliance_goals"][0], L["side_neutral_shorts"][0], L["side_neutral_shorts"][1]],
        auto_toggle_quadrants=[L["own_quadrants"][0]],  # set 1 own-quadrant toggle for AWP-bonus
        # Driver: 10 match loads, safe orientation, into alliance goal
        match_load_attempts=10,
        match_load_orient="safe",
        match_load_goal_id=L["alliance_goals"][1],
        # Endgame: deny one opponent toggle, park one robot in midfield
        deny_opp_quadrants=L["opp_quadrants"][:1],
        midfield_robots=1,
    )


def yellow_gamble_strategy(alliance: Alliance) -> StrategySpec:
    """Yellow-side commit on everything. Big upside if toggles hold,
    catastrophic loss if denied."""
    L = _layout(alliance)
    return StrategySpec(
        name=f"yellow_gamble_{alliance.value}",
        alliance=alliance,
        auto_pin_attempts=7,
        auto_pin_orient="yellow",
        auto_goal_ids=[L["alliance_goals"][0], L["side_neutral_shorts"][0], L["side_neutral_shorts"][1]],
        auto_toggle_quadrants=L["own_quadrants"],     # try to set BOTH own toggles
        match_load_attempts=10,
        match_load_orient="yellow",
        match_load_goal_id=L["alliance_goals"][1],
        deny_opp_quadrants=[],                        # all-offense; no denial budget
        midfield_robots=2,                            # park both for midfield-yellow ownership
    )


def hybrid_strategy(alliance: Alliance) -> StrategySpec:
    """Safe scoring + aggressive toggle denial. Locks own EV, demolishes opponent's."""
    L = _layout(alliance)
    return StrategySpec(
        name=f"hybrid_{alliance.value}",
        alliance=alliance,
        auto_pin_attempts=7,
        auto_pin_orient="safe",
        auto_goal_ids=[L["alliance_goals"][0], L["side_neutral_shorts"][0], L["side_neutral_shorts"][1]],
        auto_toggle_quadrants=L["own_quadrants"],     # set both own toggles
        match_load_attempts=10,
        match_load_orient="safe",
        match_load_goal_id=L["alliance_goals"][1],
        deny_opp_quadrants=L["opp_quadrants"],        # deny both opponent toggles
        midfield_robots=1,
    )


def do_nothing_strategy(alliance: Alliance) -> StrategySpec:
    """No scoring, no movement. Useful as a baseline opponent for testing."""
    L = _layout(alliance)
    return StrategySpec(
        name=f"nothing_{alliance.value}",
        alliance=alliance,
        auto_pin_attempts=0,
        auto_pin_orient="safe",
        auto_goal_ids=L["alliance_goals"],
        auto_toggle_quadrants=[],
        match_load_attempts=0,
        match_load_orient="safe",
        match_load_goal_id=L["alliance_goals"][1],
        deny_opp_quadrants=[],
        midfield_robots=0,
    )


# -- Registry / factory --------------------------------------------------------

STRATEGY_REGISTRY = {
    "safe": safe_strategy,
    "yellow_gamble": yellow_gamble_strategy,
    "hybrid": hybrid_strategy,
    "do_nothing": do_nothing_strategy,
}


def make_strategy(name: str, alliance: Alliance) -> StrategySpec:
    """Build an alliance-aware StrategySpec by template name."""
    if name not in STRATEGY_REGISTRY:
        raise KeyError(f"Unknown strategy: {name!r}. "
                       f"Available: {list(STRATEGY_REGISTRY)}")
    return STRATEGY_REGISTRY[name](alliance)
