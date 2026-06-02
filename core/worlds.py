"""World factories — pure-data builders for the field, with no pygame /
rendering dependencies. Importable from anywhere (the live game, the RL
env, the BC/DAGGER trainers, tests).

The canonical starting field is `make_empty_world`. Anything that wants
"the real game" should call this and modify from there.
"""

from __future__ import annotations

import math

from core.state import (
    Alliance, Cup, Goal, GoalType, Phase, Pin, PinColor, Robot, Toggle,
    ToggleState, World, build_stack,
)


def make_empty_world() -> World:
    """Field layout per the user-confirmed manual positions, with the
    user-specified pin/cup starting configuration.

    Autonomous line: diagonal y = -x.
    Red side (y + x < 0): LEFT (Q3) and BOTTOM (Q2) quadrants.
    Blue side (y + x > 0): TOP (Q0) and RIGHT (Q1) quadrants.
    """
    goals = [
        # Alliance goals — 2 red on red side, 2 blue on blue side
        Goal(0, GoalType.ALLIANCE, Alliance.RED,  x=-4.0, y=-2.0, quadrant=3, awp_side=Alliance.RED),
        Goal(1, GoalType.ALLIANCE, Alliance.RED,  x=-2.0, y=-4.0, quadrant=2, awp_side=Alliance.RED),
        Goal(2, GoalType.ALLIANCE, Alliance.BLUE, x=2.0,  y=4.0,  quadrant=0, awp_side=Alliance.BLUE),
        Goal(3, GoalType.ALLIANCE, Alliance.BLUE, x=4.0,  y=2.0,  quadrant=1, awp_side=Alliance.BLUE),
        # Neutral short goals — one per quadrant
        Goal(4, GoalType.SHORT, Alliance.NEUTRAL, x=-2.0, y=4.0,  quadrant=0, awp_side=Alliance.BLUE),
        Goal(5, GoalType.SHORT, Alliance.NEUTRAL, x=4.0,  y=-2.0, quadrant=1, awp_side=Alliance.BLUE),
        Goal(6, GoalType.SHORT, Alliance.NEUTRAL, x=2.0,  y=-4.0, quadrant=2, awp_side=Alliance.RED),
        Goal(7, GoalType.SHORT, Alliance.NEUTRAL, x=-4.0, y=2.0,  quadrant=3, awp_side=Alliance.RED),
        # Tall goal at center, in midfield
        Goal(8, GoalType.TALL,  Alliance.NEUTRAL, x=0.0,  y=0.0,  quadrant=0,
             in_midfield=True, awp_side=Alliance.NEUTRAL),
    ]
    # Toggles at wall midpoints — one per diagonal quadrant
    toggles = [
        Toggle(0, quadrant=0, state=ToggleState.YELLOW, x=0.0,  y=6.0),    # TOP wall
        Toggle(1, quadrant=1, state=ToggleState.YELLOW, x=6.0,  y=0.0),    # RIGHT wall
        Toggle(2, quadrant=2, state=ToggleState.YELLOW, x=0.0,  y=-6.0),   # BOTTOM wall
        Toggle(3, quadrant=3, state=ToggleState.YELLOW, x=-6.0, y=0.0),    # LEFT wall
    ]
    robots = [
        Robot(id=0, alliance=Alliance.RED,  x=-5.5, y=2.0,  theta=0.0),
        Robot(id=1, alliance=Alliance.RED,  x=-5.5, y=-2.0, theta=0.0),
        Robot(id=2, alliance=Alliance.BLUE, x=5.5,  y=2.0,  theta=math.pi),
        Robot(id=3, alliance=Alliance.BLUE, x=5.5,  y=-2.0, theta=math.pi),
    ]
    world = World(goals=goals, toggles=toggles, robots=robots, phase=Phase.AUTO)

    pin_id = 100
    cup_id = 100

    def _add_pin_on_cup(x: float, y: float,
                        top: PinColor, bot: PinColor,
                        clear_face_up: bool) -> None:
        nonlocal pin_id, cup_id
        world.cups.append(Cup(id=cup_id, x=x, y=y, clear_face_up=clear_face_up))
        cup_id += 1
        world.pins.append(Pin(id=pin_id, half_a_color=top, half_b_color=bot,
                                x=x, y=y))
        pin_id += 1

    def _add_empty_cup(x: float, y: float, clear_face_up: bool) -> None:
        nonlocal cup_id
        world.cups.append(Cup(id=cup_id, x=x, y=y, clear_face_up=clear_face_up))
        cup_id += 1

    def _add_loose_pin(x: float, y: float,
                        top: PinColor, bot: PinColor) -> None:
        nonlocal pin_id
        world.pins.append(Pin(id=pin_id, half_a_color=top, half_b_color=bot,
                                x=x, y=y))
        pin_id += 1

    # Midfield pin-on-cups
    _add_pin_on_cup( 2.0, 0.0, PinColor.BLUE, PinColor.RED, clear_face_up=True)
    _add_pin_on_cup( 0.0, 2.0, PinColor.BLUE, PinColor.RED, clear_face_up=True)
    _add_pin_on_cup(-2.0, 0.0, PinColor.RED, PinColor.BLUE, clear_face_up=True)
    _add_pin_on_cup( 0.0,-2.0, PinColor.RED, PinColor.BLUE, clear_face_up=True)

    def _mirror4(x: float, y: float):
        return [(x, y), (-x, y), (-x, -y), (x, -y)]

    # y = x diagonal pin-on-cups (yellow/yellow on clear cups)
    for px, py in _mirror4(2.0, 2.0) + _mirror4(4.0, 4.0):
        if abs(px - py) < 1e-6:
            _add_pin_on_cup(px, py, PinColor.YELLOW, PinColor.YELLOW,
                              clear_face_up=True)

    # y = -x diagonal: empty clear-up cup surrounded by 4 pins
    _SURROUND_OFFSET = 0.7
    for px, py in _mirror4(2.0, 2.0) + _mirror4(4.0, 4.0):
        if abs(px + py) >= 1e-6:
            continue
        _add_empty_cup(px, py, clear_face_up=True)
        _add_loose_pin(px, py + _SURROUND_OFFSET,
                        PinColor.BLUE, PinColor.YELLOW)
        _add_loose_pin(px + _SURROUND_OFFSET, py,
                        PinColor.BLUE, PinColor.YELLOW)
        _add_loose_pin(px, py - _SURROUND_OFFSET,
                        PinColor.RED, PinColor.YELLOW)
        _add_loose_pin(px - _SURROUND_OFFSET, py,
                        PinColor.RED, PinColor.YELLOW)

    # Wall pin-on-cups with flanking empty cups
    for px, py in _mirror4(2.0, 6.0):
        _add_pin_on_cup(px, py, PinColor.YELLOW, PinColor.YELLOW,
                          clear_face_up=False)
        _add_empty_cup(px - 1.0, py, clear_face_up=False)
        _add_empty_cup(px + 1.0, py, clear_face_up=False)
    for px, py in _mirror4(6.0, 2.0):
        _add_pin_on_cup(px, py, PinColor.YELLOW, PinColor.YELLOW,
                          clear_face_up=False)
        _add_empty_cup(px, py - 1.0, clear_face_up=False)
        _add_empty_cup(px, py + 1.0, clear_face_up=False)

    return world


def make_demo_world() -> World:
    """113-pt strategy floor (red alliance). Built on top of make_empty_world
    but clears the starting yellow pin from G6/G7 so build_stack can construct
    a clean SC2 chain there."""
    w = make_empty_world()
    # Remove the starting yellow pin from G6/G7 (G1/G0 don't have one)
    w.pins = [p for p in w.pins
              if not (p.in_goal in (6, 7) and 200 <= p.id < 204)]

    w.phase = Phase.ENDED
    w.robots[0].in_midfield = True
    w.robots[1].in_midfield = True
    w.robots[0].x, w.robots[0].y = 0.5, 0.5
    w.robots[1].x, w.robots[1].y = -0.5, -0.5

    build_stack(w, 0, top_color=PinColor.RED, bottom_color=PinColor.YELLOW,
                n_pins=3, cups_clear_face_up=False)
    build_stack(w, 6, top_color=PinColor.RED, bottom_color=PinColor.YELLOW,
                n_pins=2, cups_clear_face_up=False)
    build_stack(w, 7, top_color=PinColor.RED, bottom_color=PinColor.YELLOW,
                n_pins=2, cups_clear_face_up=False)
    build_stack(w, 1, top_color=PinColor.RED, bottom_color=PinColor.YELLOW,
                n_pins=10, cups_clear_face_up=False)
    return w


__all__ = ["make_empty_world", "make_demo_world"]
