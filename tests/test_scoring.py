"""Unit tests for the Override scoring engine.

Important: scoring now derives visibility from stack physics, not from a
stored 'scored_half' field. To test scoring, build a proper SC2 stack via
`build_stack()` or construct the chain manually with `in_cup` / `on_pin`.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest

from core.state import (
    Alliance, Cup, Goal, GoalType, Phase, Pin, PinColor, Robot, Toggle,
    ToggleState, World, build_stack,
)
from core.scoring import score


# -- Field setup helpers -------------------------------------------------------


def make_empty_world() -> World:
    """Diagonal-quadrant layout per the user-confirmed manual positions."""
    goals = [
        Goal(id=0, type=GoalType.ALLIANCE, alliance=Alliance.RED,
             x=-4.0, y=-2.0, quadrant=3, awp_side=Alliance.RED),
        Goal(id=1, type=GoalType.ALLIANCE, alliance=Alliance.RED,
             x=-2.0, y=-4.0, quadrant=2, awp_side=Alliance.RED),
        Goal(id=2, type=GoalType.ALLIANCE, alliance=Alliance.BLUE,
             x=2.0,  y=4.0,  quadrant=0, awp_side=Alliance.BLUE),
        Goal(id=3, type=GoalType.ALLIANCE, alliance=Alliance.BLUE,
             x=4.0,  y=2.0,  quadrant=1, awp_side=Alliance.BLUE),
        Goal(id=4, type=GoalType.SHORT, alliance=Alliance.NEUTRAL,
             x=-2.0, y=4.0,  quadrant=0, awp_side=Alliance.BLUE),
        Goal(id=5, type=GoalType.SHORT, alliance=Alliance.NEUTRAL,
             x=4.0,  y=-2.0, quadrant=1, awp_side=Alliance.BLUE),
        Goal(id=6, type=GoalType.SHORT, alliance=Alliance.NEUTRAL,
             x=2.0,  y=-4.0, quadrant=2, awp_side=Alliance.RED),
        Goal(id=7, type=GoalType.SHORT, alliance=Alliance.NEUTRAL,
             x=-4.0, y=2.0,  quadrant=3, awp_side=Alliance.RED),
        Goal(id=8, type=GoalType.TALL, alliance=Alliance.NEUTRAL,
             x=0.0, y=0.0, quadrant=0, in_midfield=True,
             awp_side=Alliance.NEUTRAL),
    ]
    toggles = [
        Toggle(id=0, quadrant=0, state=ToggleState.YELLOW, x=0.0,  y=6.0),
        Toggle(id=1, quadrant=1, state=ToggleState.YELLOW, x=6.0,  y=0.0),
        Toggle(id=2, quadrant=2, state=ToggleState.YELLOW, x=0.0,  y=-6.0),
        Toggle(id=3, quadrant=3, state=ToggleState.YELLOW, x=-6.0, y=0.0),
    ]
    return World(goals=goals, toggles=toggles, phase=Phase.AUTO)


# -- Sanity / empty cases -----------------------------------------------------


def test_empty_world_is_zero_zero():
    r = score(make_empty_world())
    assert r.red == 0 and r.blue == 0
    assert r.awp_red is False and r.awp_blue is False


def test_orphan_pin_with_no_stack_chain_does_not_score():
    """A pin with in_goal set but no proper SC2 chain (no cup above + not
    the unique 'bottom' pin in goal) is invisible to the scoring engine."""
    w = make_empty_world()
    # Two pins both claiming in_goal=0 with in_cup=None — only ONE can be the
    # bottom-of-stack; the other has no valid chain.
    w.pins.append(Pin(id=0, half_a_color=PinColor.RED,
                      half_b_color=PinColor.YELLOW, in_goal=0))
    w.pins.append(Pin(id=1, half_a_color=PinColor.RED,
                      half_b_color=PinColor.YELLOW, in_goal=0))
    r = score(w)
    # Only the first one (lowest id) is the bottom pin per stack_in_goal.
    # That one is also the top pin (no cup above) so its top half scores +5.
    assert r.red == 5
    assert r.blue == 0


# -- Single-pin visibility cases ----------------------------------------------


def test_single_pin_alone_in_goal_scores_top_half_only():
    """One pin in a goal with no cup above: bottom hidden by goal,
    top half (half_a) visible -> scores half_a's color."""
    w = make_empty_world()
    w.pins.append(Pin(id=0, half_a_color=PinColor.RED,
                      half_b_color=PinColor.YELLOW, in_goal=0))
    r = score(w)
    # Top half is RED -> +5 red. Yellow (bottom) hidden by goal.
    assert r.red == 5
    assert r.blue == 0


def test_single_pin_top_color_decides_alliance():
    """A blue/yellow pin alone in a goal: top=blue visible -> +5 blue."""
    w = make_empty_world()
    w.pins.append(Pin(id=0, half_a_color=PinColor.BLUE,
                      half_b_color=PinColor.YELLOW, in_goal=0))
    r = score(w)
    assert r.red == 0 and r.blue == 5


def test_single_yellow_yellow_pin_with_red_toggle():
    """Yellow/yellow pin alone in G0 (which is in Q3 LEFT). Set the Q3 toggle
    to RED -> +10 red for the visible yellow top half."""
    w = make_empty_world()
    # G0 is in quadrant 3 (LEFT), so set toggles[3] (Q3) to RED
    w.toggles[3].state = ToggleState.RED
    w.pins.append(Pin(id=0, half_a_color=PinColor.YELLOW,
                      half_b_color=PinColor.YELLOW, in_goal=0))
    r = score(w)
    assert r.red == 10
    assert r.blue == 0


# -- Stack visibility: the bug the user found --------------------------------


def test_bottom_pin_fully_covered_scores_zero():
    """Stack: pin -> cup(clear-up) -> pin.

    Bottom pin: bottom in goal (hidden), top in cup's gray bottom (hidden).
    -> 0 visible halves -> scores 0.

    Top pin: bottom in cup's clear top (visible) -> half_b. Top exposed -> half_a.
    Both halves visible -> 2 scores.
    """
    w = make_empty_world()
    build_stack(w, 0, top_color=PinColor.RED, bottom_color=PinColor.YELLOW,
                n_pins=2, cups_clear_face_up=True)
    # toggle UNSET, so yellow halves score 0
    r = score(w)
    # Bottom pin: 0 pts (both halves hidden)
    # Top pin: top (red) visible = +5; bottom (yellow) visible but no owner = 0
    # Total: 5
    assert r.red == 5
    assert r.blue == 0


def test_all_cups_clear_down_exposes_top_of_every_pin():
    """Stack of N pins with cups clear_face_up=False -> every pin's TOP half
    is visible (the standard 'top color scored' configuration).

    For N=10 red/yellow pins with top=red: scores 10 * 5 = 50 to red,
    regardless of toggle state."""
    w = make_empty_world()
    build_stack(w, 0, top_color=PinColor.RED, bottom_color=PinColor.YELLOW,
                n_pins=10, cups_clear_face_up=False)
    r = score(w)
    assert r.red == 50
    assert r.blue == 0


def test_all_cups_clear_down_with_yellow_tops_scores_via_toggle():
    """Stack of N pins in G0 (Q3 LEFT), top=yellow, cups clear_face_up=False
    -> every pin's yellow top is visible. Score depends on Q3 toggle (= toggles[3])."""
    w = make_empty_world()
    build_stack(w, 0, top_color=PinColor.YELLOW, bottom_color=PinColor.RED,
                n_pins=10, cups_clear_face_up=False)

    # Q3 toggle UNSET: 0 to everyone
    r = score(w)
    assert r.red == 0 and r.blue == 0

    # Q3 toggle RED: 100 to red
    w.toggles[3].state = ToggleState.RED
    r = score(w)
    assert r.red == 100 and r.blue == 0

    # Q3 toggle BLUE: 100 to blue (the "200 pt swing" demonstrated)
    w.toggles[3].state = ToggleState.BLUE
    r = score(w)
    assert r.red == 0 and r.blue == 100

    # Q3 toggle YELLOW: 0 to everyone
    w.toggles[3].state = ToggleState.YELLOW
    r = score(w)
    assert r.red == 0 and r.blue == 0


def test_yellow_toggle_zeroes_yellows():
    """SC4 — toggle set to yellow means no one owns yellows in that quadrant.
    G0 is in Q3 (LEFT), so set toggles[3] to YELLOW."""
    w = make_empty_world()
    w.toggles[3].state = ToggleState.YELLOW
    build_stack(w, 0, top_color=PinColor.YELLOW, bottom_color=PinColor.YELLOW,
                n_pins=5, cups_clear_face_up=False)
    r = score(w)
    assert r.red == 0 and r.blue == 0


# -- Midfield mechanic --------------------------------------------------------


def test_midfield_yellow_pin_owned_by_majority_robots():
    """Yellow halves in the tall (midfield) goal go to majority-midfield alliance."""
    w = make_empty_world()
    w.phase = Phase.ENDED
    w.robots.append(Robot(id=0, alliance=Alliance.RED, in_midfield=True))
    w.robots.append(Robot(id=1, alliance=Alliance.BLUE, in_midfield=False))
    # Single yellow pin alone in the tall goal — top half visible
    w.pins.append(Pin(id=0, half_a_color=PinColor.YELLOW,
                      half_b_color=PinColor.YELLOW, in_goal=8))
    r = score(w)
    # +10 (yellow to red via midfield majority) + 8 (red robot park)
    assert r.red == 18
    assert r.blue == 0


def test_midfield_yellow_pin_unowned_on_tie():
    w = make_empty_world()
    w.phase = Phase.ENDED
    w.robots.append(Robot(id=0, alliance=Alliance.RED, in_midfield=True))
    w.robots.append(Robot(id=1, alliance=Alliance.BLUE, in_midfield=True))
    w.pins.append(Pin(id=0, half_a_color=PinColor.YELLOW,
                      half_b_color=PinColor.YELLOW, in_goal=8))
    r = score(w)
    # No yellow owner; each alliance parks 1 robot -> 8 each
    assert r.red == 8 and r.blue == 8


# -- Auto bonus + midfield park (SC6 / SC7) ----------------------------------


def test_auto_bonus_goes_to_higher_auto_scorer():
    w = make_empty_world()
    w.phase = Phase.DRIVER
    r = score(w, auto_red=40, auto_blue=25)
    assert r.red == 12 and r.blue == 0


def test_auto_bonus_on_tie_awards_six_each():
    w = make_empty_world()
    w.phase = Phase.DRIVER
    r = score(w, auto_red=30, auto_blue=30)
    assert r.red == 6 and r.blue == 6
    r2 = score(w, auto_red=0, auto_blue=0)
    assert r2.red == 6 and r2.blue == 6


def test_midfield_park_eight_per_robot():
    w = make_empty_world()
    w.phase = Phase.ENDED
    w.robots.append(Robot(id=0, alliance=Alliance.RED, in_midfield=True))
    w.robots.append(Robot(id=1, alliance=Alliance.RED, in_midfield=True))
    w.robots.append(Robot(id=2, alliance=Alliance.BLUE, in_midfield=True))
    w.robots.append(Robot(id=3, alliance=Alliance.BLUE, in_midfield=False))
    r = score(w)
    assert r.red == 16 and r.blue == 8


def test_midfield_park_only_at_match_end():
    w = make_empty_world()
    w.phase = Phase.DRIVER
    w.robots.append(Robot(id=0, alliance=Alliance.RED, in_midfield=True))
    r = score(w)
    assert r.red == 0


# -- AWP (SC8) ----------------------------------------------------------------


def test_awp_requires_seven_pins_three_goals():
    """7 alliance-half pins spread across 3 different goals (>= 2 each).
    Red side goals: G0 (alliance), G1 (alliance), G6, G7 (neutral shorts)."""
    w = make_empty_world()
    w.robots.append(Robot(id=0, alliance=Alliance.RED, touching_field_perimeter=False))
    w.robots.append(Robot(id=1, alliance=Alliance.RED, touching_field_perimeter=False))
    build_stack(w, 0, top_color=PinColor.RED, bottom_color=PinColor.YELLOW,
                n_pins=3, cups_clear_face_up=False)
    build_stack(w, 6, top_color=PinColor.RED, bottom_color=PinColor.YELLOW,
                n_pins=2, cups_clear_face_up=False)
    build_stack(w, 7, top_color=PinColor.RED, bottom_color=PinColor.YELLOW,
                n_pins=2, cups_clear_face_up=False)
    r = score(w, evaluate_awp=True)
    assert r.awp_red is True


def test_awp_fails_when_robot_touches_perimeter():
    w = make_empty_world()
    w.robots.append(Robot(id=0, alliance=Alliance.RED, touching_field_perimeter=True))
    w.robots.append(Robot(id=1, alliance=Alliance.RED))
    build_stack(w, 0, top_color=PinColor.RED, bottom_color=PinColor.YELLOW,
                n_pins=3, cups_clear_face_up=False)
    build_stack(w, 6, top_color=PinColor.RED, bottom_color=PinColor.YELLOW,
                n_pins=2, cups_clear_face_up=False)
    build_stack(w, 7, top_color=PinColor.RED, bottom_color=PinColor.YELLOW,
                n_pins=2, cups_clear_face_up=False)
    r = score(w, evaluate_awp=True)
    assert r.awp_red is False


def test_awp_fails_when_all_pins_in_one_goal():
    w = make_empty_world()
    w.robots.append(Robot(id=0, alliance=Alliance.RED))
    w.robots.append(Robot(id=1, alliance=Alliance.RED))
    build_stack(w, 0, top_color=PinColor.RED, bottom_color=PinColor.YELLOW,
                n_pins=10, cups_clear_face_up=False)
    r = score(w, evaluate_awp=True)
    assert r.awp_red is False


# -- Integration: user's locked-floor strategy --------------------------------


def test_user_strategy_floor_113():
    """Floor:
       - Auto: 7 alliance pins spread 3-2-2 across red-side goals -> 35
       - Driver: 10 alliance pins in red alliance goal G1 -> 50
       - Midfield: 2 red robots parked -> 16
       - Auto bonus: red 35 > blue 0 -> +12
       Total: 113.

       Red-side goals: G0 (alliance, LEFT), G1 (alliance, BOTTOM),
                       G6 (neutral, BOTTOM), G7 (neutral, LEFT)."""
    w = make_empty_world()
    w.phase = Phase.ENDED
    w.robots.append(Robot(id=0, alliance=Alliance.RED, in_midfield=True))
    w.robots.append(Robot(id=1, alliance=Alliance.RED, in_midfield=True))

    build_stack(w, 0, top_color=PinColor.RED, bottom_color=PinColor.YELLOW,
                n_pins=3, cups_clear_face_up=False)
    build_stack(w, 6, top_color=PinColor.RED, bottom_color=PinColor.YELLOW,
                n_pins=2, cups_clear_face_up=False)
    build_stack(w, 7, top_color=PinColor.RED, bottom_color=PinColor.YELLOW,
                n_pins=2, cups_clear_face_up=False)
    build_stack(w, 1, top_color=PinColor.RED, bottom_color=PinColor.YELLOW,
                n_pins=10, cups_clear_face_up=False)

    r = score(w, auto_red=35, auto_blue=0)
    assert r.red == 113


# -- Serialization ------------------------------------------------------------


def test_world_round_trips_through_json():
    w = make_empty_world()
    build_stack(w, 0, top_color=PinColor.RED, bottom_color=PinColor.YELLOW,
                n_pins=2, cups_clear_face_up=False)
    w.robots.append(Robot(id=0, alliance=Alliance.RED, x=0.0, y=0.0))
    w.toggles[0].state = ToggleState.RED
    s = w.to_json()
    w2 = World.from_json(s)
    assert score(w).red == score(w2).red
    assert score(w).blue == score(w2).blue
    assert w2.toggles[0].state == ToggleState.RED
    assert len(w2.pins) == 2
    assert len(w2.cups) == 1
