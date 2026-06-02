"""Sanity tests for the strategy Monte Carlo.

Strategy: use deterministic scenarios (everything succeeds with p=1.0) so that
each matchup produces a known closed-form score. If these pass, the simulator
is correctly wired to the scoring engine.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random

import pytest

from core.state import Alliance, ToggleState
from strategy.scenario import MatchScenario
from strategy.simulate import play_one_match, run_matchup
from strategy.strategies import make_strategy


# -- Deterministic baselines ---------------------------------------------------


def test_do_nothing_vs_do_nothing_is_zero_zero_plus_tie_bonus():
    """Both alliances do nothing. Score is just the auto-tie bonus (6+6)."""
    scen = MatchScenario().deterministic(all_success=True)
    red = make_strategy("do_nothing", Alliance.RED)
    blue = make_strategy("do_nothing", Alliance.BLUE)
    out = play_one_match(red, blue, scen, random.Random(0))
    assert out.red_score == 6
    assert out.blue_score == 6
    assert out.awp_red is False
    assert out.awp_blue is False


def test_safe_vs_do_nothing_red_dominates():
    """Red runs safe strategy at perfect execution against a do-nothing blue.

    Red expected breakdown:
      Auto pins: 7 attempts x p=1.0 = 7 pins x 5 pts = 35
      Match loads: 10 x 5 = 50
      Midfield park: 1 robot x 8 = 8
      Auto bonus: +12 (red 35, blue 0)
      Total: 105

    Plus AWP red = True (7 pins / 3 goals / robots clear).
    """
    scen = MatchScenario().deterministic(all_success=True)
    red = make_strategy("safe", Alliance.RED)
    blue = make_strategy("do_nothing", Alliance.BLUE)
    out = play_one_match(red, blue, scen, random.Random(0))
    assert out.red_score == 105
    assert out.blue_score == 0
    assert out.awp_red is True
    assert out.awp_blue is False


def test_safe_vs_safe_mirror_match_ties_auto():
    """Both alliances run identical safe play. Mirror match -> tied auto -> 6 each."""
    scen = MatchScenario().deterministic(all_success=True)
    red = make_strategy("safe", Alliance.RED)
    blue = make_strategy("safe", Alliance.BLUE)
    out = play_one_match(red, blue, scen, random.Random(0))
    # Auto: 35 each (tied) -> +6 bonus each
    # Total each: 35 + 50 + 8 + 6 = 99
    assert out.red_score == 99
    assert out.blue_score == 99
    # AWP: both alliances satisfy 7 pins / 3 goals / clear perimeter
    assert out.awp_red is True
    assert out.awp_blue is True


def test_yellow_gamble_vs_safe_under_deterministic_all_success():
    """Deterministic scenario: everything succeeds with p=1.0.

    Red yellow_gamble plan: 7 auto pins (3-2-2 across G0, G6, G7) + 10 match
    loads in G1. Sets toggles in own quadrants Q2 (BOTTOM) and Q3 (LEFT).
    Goals: G0/G7 are Q3 (LEFT); G6/G1 are Q2 (BOTTOM).

    Blue safe denies opp_quadrants[:1] = [Q2 BOTTOM] -> toggle[2] gets unset.
    Toggle[3] (LEFT) survives.

    Red scoring:
      - 3 G0 pins (Q3 held) -> 30, 2 G7 pins (Q3 held) -> 20  = 50 yellow
      - 2 G6 pins (Q2 denied) -> 0, 10 G1 pins (Q2 denied) -> 0
      - Midfield park: 16
      - Auto bonus: red 70 > blue 35 -> +12
      Total: 50 + 16 + 12 = 78

    Blue safe: 35 + 50 + 8 = 93
    """
    scen = MatchScenario().deterministic(all_success=True)
    red = make_strategy("yellow_gamble", Alliance.RED)
    blue = make_strategy("safe", Alliance.BLUE)
    out = play_one_match(red, blue, scen, random.Random(0))
    assert out.red_score == 78
    assert out.blue_score == 93


def test_yellow_gamble_collapses_when_all_toggles_denied():
    """p_my_toggle_held=0.0 means EVERY toggle red sets gets denied at buzzer.
    Auto snapshot still shows red ahead (toggle denial happens at buzzer, not
    instantly), so auto bonus +12 goes to red. But the underlying pin value
    vanishes -- this is the user's strategic argument materialized.

    Red yellow_gamble:
      Auto pins yellow-side: all toggles UNSET at end -> 0
      Match loads yellow-side: all UNSET -> 0
      Midfield park: 16
      Auto bonus: +12 (snapshot was 70-35 BEFORE denial)
      Total: 28

    Blue safe (immune to denial):
      35 + 50 + 8 = 93
    """
    scen = MatchScenario(
        auto_pin_success=1.0,
        auto_toggle_set_success=1.0,
        driver_pin_success=1.0,
        p_robots_clear_perimeter=1.0,
        p_my_toggle_held=0.0,
        p_opp_toggle_denied=1.0,
        p_robot_reaches_midfield=1.0,
    )
    red = make_strategy("yellow_gamble", Alliance.RED)
    blue = make_strategy("safe", Alliance.BLUE)
    out = play_one_match(red, blue, scen, random.Random(0))
    # Yellow strategy gets auto bonus (from snapshot) but loses all pin value
    assert out.red_score == 28
    assert out.blue_score == 93


# -- Stochastic sanity --------------------------------------------------------


def test_safe_strategy_floor_holds_under_uncertainty():
    """Run safe vs do_nothing under realistic stochastic scenario.
    Even with reduced execution rates, red should dominate clearly."""
    scen = MatchScenario(
        auto_pin_success=0.85, auto_toggle_set_success=0.80,
        driver_pin_success=0.90,
        p_robots_clear_perimeter=0.95,
        p_my_toggle_held=0.5, p_opp_toggle_denied=0.5,
        p_robot_reaches_midfield=0.8,
    )
    red = make_strategy("safe", Alliance.RED)
    blue = make_strategy("do_nothing", Alliance.BLUE)
    stats = run_matchup(red, blue, scen, trials=500, seed=0)
    assert stats.red_win_rate > 0.99
    assert stats.red_mean > 70  # plenty of cushion against the floor
    # AWP needs ALL 7 attempts to succeed (each at 0.85) so expected ~0.32 here.
    # Strategy could add safety margin; for now we just verify AWP is achievable.
    assert stats.red_awp_rate > 0.20


def test_safe_vs_yellow_under_default_scenario_safe_is_competitive():
    """Under default mid-skill scenario, the safe strategy should at least
    be competitive with yellow_gamble (not catastrophically worse).
    Documents the strategy comparison the user is interested in."""
    scen = MatchScenario()   # defaults
    red = make_strategy("safe", Alliance.RED)
    blue = make_strategy("yellow_gamble", Alliance.BLUE)
    stats = run_matchup(red, blue, scen, trials=500, seed=0)
    # Both should be playable strategies; neither completely dominates at p=0.5
    assert 0.2 < stats.red_win_rate < 0.85
    assert stats.red_mean > 50   # safe floor holds
    assert stats.blue_std > 20   # yellow_gamble is high-variance


def test_run_matchup_is_reproducible_with_seed():
    """Same seed -> identical results. Critical for debugging."""
    scen = MatchScenario()
    red = make_strategy("safe", Alliance.RED)
    blue = make_strategy("yellow_gamble", Alliance.BLUE)
    s1 = run_matchup(red, blue, scen, trials=200, seed=42)
    s2 = run_matchup(red, blue, scen, trials=200, seed=42)
    assert s1.red_scores == s2.red_scores
    assert s1.blue_scores == s2.blue_scores
