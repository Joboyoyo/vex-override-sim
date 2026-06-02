"""Headless smoke tests for the renderer.

Pygame in tests runs with SDL_VIDEODRIVER=dummy so it doesn't open a window.
These tests verify that:
- The module imports without error
- Drawing functions don't crash on representative worlds
- The empty / demo world factories build worlds that score correctly
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Run pygame headlessly
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from core.scoring import score


def test_pygame_view_imports():
    """The whole render module should import cleanly."""
    from render import pygame_view, draw, hud, colors, coords  # noqa: F401


def test_empty_world_factory_round_trip():
    from render.pygame_view import make_empty_world
    w = make_empty_world()
    assert len(w.goals) == 9
    assert len(w.toggles) == 4
    assert len(w.robots) == 4
    r = score(w)
    assert r.red == 0 and r.blue == 0


def test_demo_world_scores_113_correctly():
    """Demo world matches the strategy floor test."""
    from render.pygame_view import make_demo_world
    w = make_demo_world()
    r = score(w, auto_red=35, auto_blue=0)
    assert r.red == 113


def test_stack_view_renders_empty_goal():
    pygame.init()
    surface = pygame.Surface((1280, 880))
    from render.stack_view import draw_focused_goal
    from render.pygame_view import make_empty_world

    w = make_empty_world()
    area = pygame.Rect(20, 140, 800, 700)
    result = draw_focused_goal(surface, area, w, goal_id=0)

    assert "goal" in result
    assert "stack" in result
    assert "add_zone" in result
    assert len(result["stack"]) == 0   # empty stack
    assert result["add_zone"] is not None


def test_stack_view_renders_demo_goal_with_full_stack():
    pygame.init()
    surface = pygame.Surface((1280, 880))
    from render.stack_view import draw_focused_goal
    from render.pygame_view import make_demo_world

    w = make_demo_world()
    area = pygame.Rect(20, 140, 800, 700)
    # G1 has 10 pins + 9 cups = 19 stack elements
    result = draw_focused_goal(surface, area, w, goal_id=1)
    assert len(result["stack"]) == 19
    # Verify alternation in the rendered output
    for i, (kind, _obj_id, _rect) in enumerate(result["stack"]):
        expected = "pin" if i % 2 == 0 else "cup"
        assert kind == expected


def test_hud_panels_render_without_error():
    pygame.init()
    surface = pygame.Surface((1280, 880))
    from render.hud import (draw_score_breakdown, draw_status_bar,
                            draw_top_bar, draw_help_panel)
    from render.pygame_view import make_demo_world

    w = make_demo_world()
    s = score(w, auto_red=35, auto_blue=0)

    draw_top_bar(surface, pygame.Rect(0, 0, 1280, 120), w, s)
    draw_score_breakdown(surface, pygame.Rect(900, 140, 360, 220), s)
    draw_help_panel(surface, pygame.Rect(900, 380, 360, 200), "R / Y pin")
    draw_status_bar(surface, pygame.Rect(0, 844, 1280, 36), "test status")


def test_demo_world_includes_cups():
    """The demo world should include cups between pins in the stacks
    so the rendering shows realistic alternating structure. May also include
    loose cups on the field (in_goal=None) for the robot to pick up."""
    from render.pygame_view import make_demo_world
    w = make_demo_world()
    # At least some stack cups should exist
    stack_cups = [c for c in w.cups if c.in_goal is not None]
    assert len(stack_cups) > 0


def test_demo_world_still_scores_113():
    """Adding cups to the demo world shouldn't change the scoring outcome
    (cups don't directly score; pins still have scored_half=0)."""
    from render.pygame_view import make_demo_world
    w = make_demo_world()
    r = score(w, auto_red=35, auto_blue=0)
    assert r.red == 113


def test_demo_world_stacks_follow_sc2_alternation():
    """Per SC2: each goal's stack must start with a pin in the goal, then
    alternate cup-on-pin / pin-in-cup. Walk every goal's stack and verify
    the alternation is correctly modeled."""
    from render.pygame_view import make_demo_world
    w = make_demo_world()
    for goal in w.goals:
        stack = w.stack_in_goal(goal.id)
        if not stack:
            continue
        # Bottom must be a pin
        assert stack[0][0] == "pin", f"G{goal.id} stack doesn't start with a pin"
        # Bottom pin must be directly in goal (in_cup=None)
        assert stack[0][1].in_cup is None
        # Alternation: pin, cup, pin, cup, ...
        for i, (kind, _) in enumerate(stack):
            expected = "pin" if i % 2 == 0 else "cup"
            assert kind == expected, (
                f"G{goal.id} stack[{i}] is {kind}, expected {expected}")
        # Each cup must reference the pin below; each pin (except bottom) must
        # reference the cup below
        for i in range(1, len(stack)):
            kind, obj = stack[i]
            below_kind, below_obj = stack[i - 1]
            if kind == "cup":
                assert obj.on_pin == below_obj.id, (
                    f"cup#{obj.id} on_pin mismatch at G{goal.id}[{i}]")
            else:
                assert obj.in_cup == below_obj.id, (
                    f"pin#{obj.id} in_cup mismatch at G{goal.id}[{i}]")


def test_next_placeable_kind_enforces_alternation():
    """The world should expose what the next placement must be."""
    from render.pygame_view import make_empty_world
    from core.state import Pin, Cup, PinColor
    w = make_empty_world()

    # Empty goal: must be pin
    assert w.next_placeable_kind(0) == "pin"

    # Add a pin to G0
    w.pins.append(Pin(id=0, half_a_color=PinColor.RED, half_b_color=PinColor.YELLOW,
                      in_goal=0, scored_half=0))
    # Stack tops out at pin -> next must be cup
    assert w.next_placeable_kind(0) == "cup"

    # Add a cup on that pin
    w.cups.append(Cup(id=0, in_goal=0, on_pin=0, clear_face_up=True))
    # Stack tops out at cup -> next must be pin
    assert w.next_placeable_kind(0) == "pin"

    # Add a pin in the cup
    w.pins.append(Pin(id=1, half_a_color=PinColor.RED, half_b_color=PinColor.YELLOW,
                      in_goal=0, in_cup=0, scored_half=0))
    # Now next must be cup again
    assert w.next_placeable_kind(0) == "cup"


def test_stack_in_goal_walks_full_chain():
    """stack_in_goal should return objects in the correct chain order."""
    from render.pygame_view import make_demo_world
    w = make_demo_world()
    # G1 has 10 pins and 9 cups = 19 stack elements
    stack = w.stack_in_goal(1)
    assert len(stack) == 19
    # Count by kind
    n_pins = sum(1 for k, _ in stack if k == "pin")
    n_cups = sum(1 for k, _ in stack if k == "cup")
    assert n_pins == 10
    assert n_cups == 9


def test_toggle_duel_world_is_minimal():
    """make_toggle_duel_world() returns a stripped-down world for the AI duel:
    no pins, no cups, only 1 toggle (Q1), only 2 robots (one red, one blue)."""
    from render.pygame_view import make_toggle_duel_world
    from core.state import Alliance

    w = make_toggle_duel_world()
    assert len(w.pins) == 0
    assert len(w.cups) == 0
    assert len(w.toggles) == 1
    assert w.toggles[0].quadrant == 1
    assert len(w.robots) == 2
    alliances = sorted(r.alliance for r in w.robots)
    assert alliances == [Alliance.BLUE, Alliance.RED]


def test_duel_bots_flip_toggle_eventually():
    """Run the duel: both bots in POST_LOADS racing for the single Q1 toggle.
    Over ~25 simulated seconds the toggle should change state at least twice
    (each bot flipping it back and forth as they trade contact)."""
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.init()
    pygame.display.set_mode((1400, 900))

    from render.pygame_view import App
    from core.state import ToggleState

    app = App()
    app._enter_duel_mode()
    assert app.duel_mode is True
    assert len(app.bots) == 2
    # Both bots should be in POST_LOADS already
    for b in app.bots:
        assert b.phase == "POST_LOADS"
    tog = app.world.toggles[0]
    initial_state = tog.resting_state    # YELLOW

    seen_states: set = {initial_state}
    dt = 0.05
    for _ in range(int(25.0 / dt)):
        for bot in app.bots:
            bot.update(dt)
        app._resolve_robot_overlaps()
        app._update_toggle_contact()
        seen_states.add(tog.resting_state)

    # The toggle should have been driven AWAY from YELLOW at least once —
    # ideally to both RED and BLUE as the bots trade flips.
    assert seen_states - {ToggleState.YELLOW, ToggleState.UNSET} != set(), (
        f"Toggle never reached an alliance color; seen={seen_states}")

    app._exit_duel_mode()
    assert app.duel_mode is False
    assert len(app.world.pins) > 0   # back to the full populated field

    pygame.quit()


def test_starting_world_matches_user_spec():
    """make_empty_world should populate the field per the explicit user spec:
      - Midfield (4 pin-on-cup): blue/red pins on clear-up cups at (±2, 0)
        and (0, ±2); +x/+y use BLUE-up halves, -x/-y use RED-up.
      - y=x diagonal (4 pin-on-cup): yellow/yellow pin on clear-up cup at
        (±2,±2) and (±4,±4) where y=x.
      - y=-x diagonal (4 EMPTY cups surrounded by 4 pins each): clear-up cup
        with blue/yellow pins on +y/+x sides and red/yellow on -y/-x sides.
      - Walls (8 pin-on-cup + 16 flanking empty cups): opaque-up cups.
      - Totals: 32 pins (4 blue/red + 12 yellow/yellow + 8 blue/yellow +
        8 red/yellow) and 36 cups."""
    from render.pygame_view import make_empty_world
    from core.state import PinColor

    w = make_empty_world()

    counts: dict[tuple[str, str], int] = {}
    for p in w.pins:
        key = tuple(sorted([p.half_a_color.value, p.half_b_color.value]))
        counts[key] = counts.get(key, 0) + 1
    assert counts.get(("blue", "red"), 0) == 4, f"counts: {counts}"
    assert counts.get(("yellow", "yellow"), 0) == 12, f"counts: {counts}"
    assert counts.get(("blue", "yellow"), 0) == 8, f"counts: {counts}"
    assert counts.get(("red", "yellow"), 0) == 8, f"counts: {counts}"
    assert len(w.pins) == 32, f"total pins: {len(w.pins)}"
    assert len(w.cups) == 36, f"total cups: {len(w.cups)}"

    # Pin-on-cup property: yellow/yellow and blue/red pins sit on cups;
    # surrounding pins (blue/yellow, red/yellow) on the y=-x diagonal do NOT.
    cup_positions = {(round(c.x, 3), round(c.y, 3)) for c in w.cups
                     if c.in_goal is None}
    for p in w.pins:
        on_cup = (round(p.x, 3), round(p.y, 3)) in cup_positions
        halves = {p.half_a_color, p.half_b_color}
        if halves == {PinColor.YELLOW} or halves == {PinColor.RED, PinColor.BLUE}:
            assert on_cup, (
                f"yellow/yellow or blue/red pin#{p.id} at "
                f"({p.x}, {p.y}) should be on a cup")
        else:
            assert not on_cup, (
                f"surrounding pin#{p.id} at ({p.x}, {p.y}) should NOT be on a cup")

    # Midfield pin orientations
    for (mx, my) in [(2.0, 0.0), (0.0, 2.0)]:
        ps = [p for p in w.pins if abs(p.x - mx) < 1e-6 and abs(p.y - my) < 1e-6]
        assert len(ps) == 1
        assert ps[0].half_a_color == PinColor.BLUE  # blue side up
        assert ps[0].half_b_color == PinColor.RED
    for (mx, my) in [(-2.0, 0.0), (0.0, -2.0)]:
        ps = [p for p in w.pins if abs(p.x - mx) < 1e-6 and abs(p.y - my) < 1e-6]
        assert len(ps) == 1
        assert ps[0].half_a_color == PinColor.RED   # red side up
        assert ps[0].half_b_color == PinColor.BLUE

    # y=-x diagonal: each of (-4,4), (-2,2), (2,-2), (4,-4) is an EMPTY
    # clear-up cup surrounded by 4 alliance-colored pins.
    for (cx, cy) in [(-4.0, 4.0), (-2.0, 2.0), (2.0, -2.0), (4.0, -4.0)]:
        # Empty cup at center
        cs = [c for c in w.cups
              if abs(c.x - cx) < 1e-6 and abs(c.y - cy) < 1e-6]
        assert len(cs) == 1 and cs[0].clear_face_up
        # No pin at center (the four pins are OFFSET around the cup)
        center_pins = [p for p in w.pins
                       if abs(p.x - cx) < 1e-6 and abs(p.y - cy) < 1e-6]
        assert center_pins == [], (
            f"y=-x diagonal cup at ({cx}, {cy}) should have NO pin on top")


def test_pickup_grabs_pin_and_cup_as_a_set_when_stacked():
    """Pressing E (or R2) near a pin-on-cup setup grabs BOTH the pin AND the
    cup in one action — same as a loader spawn, since the field starts with
    pin-on-cup match-load-style configurations."""
    import os
    import math
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.init()
    pygame.display.set_mode((1400, 900))

    from render.pygame_view import App

    app = App()
    # Park R0 so its front is at one of the midfield pin-on-cup positions (2, 0).
    r0 = app.world.robots[0]
    # The interact "front_x" is robot.x + cos(theta)*0.6, robot.y + sin(theta)*0.6
    # So position the robot at (1.4, 0) facing +x → front lands on (2.0, 0).
    r0.x, r0.y = 1.4, 0.0
    r0.theta = 0.0
    # Move other robots out of the way
    for r in app.world.robots:
        if r.id != 0:
            r.x, r.y = -5.8, 5.8

    # Find the pin and cup at (2, 0) before the interact
    pin_at = next(p for p in app.world.pins
                   if abs(p.x - 2.0) < 1e-6 and abs(p.y) < 1e-6)
    cup_at = next(c for c in app.world.cups
                   if abs(c.x - 2.0) < 1e-6 and abs(c.y) < 1e-6)

    app._try_interact()

    assert r0.holding_pin_id == pin_at.id, (
        f"expected pin#{pin_at.id} in hand, got {r0.holding_pin_id}")
    assert r0.holding_cup_id == cup_at.id, (
        f"expected cup#{cup_at.id} in hand, got {r0.holding_cup_id}")

    pygame.quit()


def test_pickup_grabs_only_one_when_not_stacked():
    """Pickup near a STANDALONE pin (not on a cup) grabs just the pin —
    confirms the pin+cup set-pickup only triggers when they share position."""
    import os
    import math
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.init()
    pygame.display.set_mode((1400, 900))

    from render.pygame_view import App

    app = App()
    r0 = app.world.robots[0]
    # Pick a y=-x surrounding pin (NOT on a cup) — e.g. the blue/yellow on the
    # +y side of the cup at (-2, 2): position (-2, 2.7).
    r0.x, r0.y = -2.0, 2.1
    r0.theta = math.pi / 2     # face +y so front lands at (-2, 2.7)
    for r in app.world.robots:
        if r.id != 0:
            r.x, r.y = -5.8, 5.8

    target_pin = next(p for p in app.world.pins
                       if abs(p.x + 2.0) < 1e-6 and abs(p.y - 2.7) < 1e-6)

    app._try_interact()

    assert r0.holding_pin_id == target_pin.id
    assert r0.holding_cup_id is None, (
        f"surrounding pin should not bring a cup; got cup#{r0.holding_cup_id}")

    pygame.quit()


def test_wing_extends_push_range():
    """With wings deployed, R0 pushes a pin that lies in the wing's thin
    rectangular hitbox at the front bumper — but the same pin is untouched
    without wings (chassis circle can't reach), confirming the wing adds a
    new push zone at the FRONT only."""
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.init()
    pygame.display.set_mode((1400, 900))

    from render.pygame_view import (
        App, ROBOT_COLLISION_RADIUS_FT, PIN_PUSH_RADIUS_FT,
        WING_HALF_WIDTH_FT, WING_LENGTH_FT, WING_BACK_OFFSET_FT,
    )

    app = App()
    r0 = app.world.robots[0]
    r0.x, r0.y = 0.0, 0.0
    r0.theta = 0.0     # face +x
    for r in app.world.robots:
        if r.id != 0:
            r.x, r.y = -5.8, 5.8

    # Place a pin at the wing's perpendicular far-edge: forward = front bumper
    # + WING_LENGTH/2, side = +0.95 ft (just inside the 1.0-ft wing reach).
    # This is outside the chassis circle (radius 0.55 from origin, so a pin
    # at distance > 0.75 from origin is outside chassis push range).
    pin = next(p for p in app.world.pins if p.in_goal is None)
    pin.x = WING_BACK_OFFSET_FT + WING_LENGTH_FT / 2   # ~0.71 ft forward
    pin.y = 0.95                                        # near outer wing edge
    orig = (pin.x, pin.y)

    # Without wings: pin shouldn't move (it's outside the chassis circle)
    app.wings_extended = False
    app._push_loose_objects()
    assert (pin.x, pin.y) == orig, (
        f"pin moved without wings: {orig} → ({pin.x}, {pin.y})")

    # Deploy wings: pin should be pushed sideways (outward) since it's near
    # the wing rectangle's top edge (+y outer face).
    app.wings_extended = True
    app._push_loose_objects()
    assert pin.y > WING_HALF_WIDTH_FT + PIN_PUSH_RADIUS_FT - 1e-6, (
        f"wings should push pin out the +y face to y >= "
        f"{WING_HALF_WIDTH_FT + PIN_PUSH_RADIUS_FT:.3f}; got {pin.y:.3f}")

    pygame.quit()


def test_wing_does_not_push_behind_bumper():
    """The wing's BACK edge sits exactly at the front bumper, so a pin
    sitting just behind the bumper (on the chassis side of the wing) does
    NOT get shoved by the wing — only the chassis circle can reach it."""
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.init()
    pygame.display.set_mode((1400, 900))

    from render.pygame_view import App, WING_BACK_OFFSET_FT, PIN_PUSH_RADIUS_FT

    app = App()
    r0 = app.world.robots[0]
    r0.x, r0.y = 0.0, 0.0
    r0.theta = 0.0
    for r in app.world.robots:
        if r.id != 0:
            r.x, r.y = -5.8, 5.8

    # Place a pin at the WING'S LATERAL FAR edge BUT BEHIND the front bumper.
    # Chassis can't reach (distance from origin > 0.55 + 0.20 = 0.75), and
    # the wing rect starts at x=WING_BACK_OFFSET_FT (=front bumper), so a
    # pin at x just BELOW that with |y| > 1.0 is outside both zones.
    pin = next(p for p in app.world.pins if p.in_goal is None)
    pin.x = WING_BACK_OFFSET_FT - 0.10    # 0.10 ft behind front bumper
    pin.y = 0.95                            # within wing's +y span if reachable
    orig = (pin.x, pin.y)

    app.wings_extended = True
    app._push_loose_objects()
    # The pin is BEHIND the wing's back edge and OUTSIDE the chassis circle
    # (distance ~0.97 from origin > 0.75). It should stay put.
    assert (pin.x, pin.y) == orig, (
        f"wing must not reach behind the bumper: {orig} → ({pin.x}, {pin.y})")

    pygame.quit()
    """A robot driving onto a loose pin shoves it along the line connecting
    their centers, leaving the pin just outside the robot's collision radius."""
    import os
    import math
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.init()
    pygame.display.set_mode((1400, 900))

    from render.pygame_view import (
        App, ROBOT_COLLISION_RADIUS_FT, PIN_PUSH_RADIUS_FT,
    )

    app = App()
    # Put R0 at origin and a fresh pin slightly off-center so it overlaps.
    r0 = app.world.robots[0]
    r0.x, r0.y = 0.0, 0.0
    # Move all other robots far away so they don't perturb the test
    for r in app.world.robots:
        if r.id != 0:
            r.x, r.y = -5.8, 5.8
    # Use the first loose pin we can find; reposition it so it overlaps R0
    pin = next(p for p in app.world.pins if p.in_goal is None)
    pin.x, pin.y = 0.10, 0.05    # well inside ROBOT_COLLISION_RADIUS_FT

    app._push_loose_objects()

    d = math.hypot(pin.x - r0.x, pin.y - r0.y)
    min_d = ROBOT_COLLISION_RADIUS_FT + PIN_PUSH_RADIUS_FT
    assert d >= min_d - 1e-6, (
        f"Pin should be shoved to >= {min_d:.3f} ft from robot; got {d:.3f}")

    pygame.quit()


def test_held_pin_does_not_get_pushed():
    """A pin currently held by a robot must NOT get pushed by `_push_loose_objects`
    (otherwise it would teleport away from the robot every frame)."""
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.init()
    pygame.display.set_mode((1400, 900))

    from render.pygame_view import App

    app = App()
    r0 = app.world.robots[0]
    r0.x, r0.y = 0.0, 0.0
    pin = next(p for p in app.world.pins if p.in_goal is None)
    pin.x, pin.y = 0.0, 0.0
    r0.holding_pin_id = pin.id

    app._push_loose_objects()
    # Position must NOT have changed
    assert pin.x == 0.0 and pin.y == 0.0, (
        f"Held pin moved: ({pin.x}, {pin.y})")

    pygame.quit()


def test_toggle_state_goes_unset_when_robot_touches():
    """SC4: a robot's BACK sensor in contact with a toggle forces its state to
    UNSET — but ONLY when the heading is aligned with the wall's outward
    normal (within ±25°). Off-angle "contact" does not engage the toggle."""
    import os
    import math
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.init()
    pygame.display.set_mode((1400, 900))

    from render.pygame_view import App, BACK_SENSOR_OFFSET_FT
    from core.state import ToggleState, Alliance

    app = App()
    tog = next(t for t in app.world.toggles if t.quadrant == 0)
    tog.resting_state = ToggleState.RED
    app.world.robots[0].x = tog.x
    app.world.robots[0].y = tog.y - BACK_SENSOR_OFFSET_FT   # back lands on tog
    # Perfectly aligned (perpendicular to top wall)
    app.world.robots[0].theta = -math.pi / 2
    app._update_toggle_contact()
    assert tog.state == ToggleState.UNSET, "Aligned BACK on toggle must read UNSET"

    # Same position but FACING the toggle (back is away) — back is far away
    # AND alignment fails. UNSET should not fire.
    app.world.robots[0].theta = math.pi / 2
    app._update_toggle_contact()
    assert tog.state == ToggleState.RED, (
        "Back away from toggle (facing it nose-first) must NOT trigger UNSET")

    # Re-align but with heading rotated 40° off perpendicular. The back sensor
    # is no longer at the toggle (it swings off the AABB), so UNSET shouldn't
    # fire. Even if the sensor still grazed the AABB, the 40° tilt exceeds the
    # 25° engagement tolerance.
    app.world.robots[0].theta = -math.pi / 2 + math.radians(40.0)
    app._update_toggle_contact()
    assert tog.state == ToggleState.RED, (
        "Off-angle (>25°) contact must NOT engage the toggle")

    # Move robot away entirely → still resting_state
    app.world.robots[0].x = 0.0
    app.world.robots[0].y = 0.0
    app.world.robots[0].theta = 0.0
    app._update_toggle_contact()
    assert tog.state == ToggleState.RED, "Free toggle restores to resting_state"

    pygame.quit()


def test_off_angle_back_contact_does_not_engage_toggle():
    """Explicitly probe the alignment gate: robot back GEOMETRICALLY on the
    toggle but heading off by 35° (past the 25° tolerance) → no engagement."""
    import os
    import math
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.init()
    pygame.display.set_mode((1400, 900))

    from render.pygame_view import App, BACK_SENSOR_OFFSET_FT
    from core.state import ToggleState

    app = App()
    tog = next(t for t in app.world.toggles if t.quadrant == 0)
    tog.resting_state = ToggleState.BLUE

    # Position the robot so its back lands at the toggle even when the heading
    # is rotated 35°. Back vector at that heading = (sin 35°, cos 35°).
    angle = math.radians(35.0)
    bx_dir = math.sin(angle)
    by_dir = math.cos(angle)
    r0 = app.world.robots[0]
    r0.x = tog.x - bx_dir * BACK_SENSOR_OFFSET_FT
    r0.y = tog.y - by_dir * BACK_SENSOR_OFFSET_FT
    r0.theta = -math.pi / 2 + angle

    # Sanity: pure geometry says the sensor IS over the toggle AABB...
    assert app._back_sensor_overlaps_geom_only(r0, tog), (
        "test setup: sensor should be geometrically on the toggle")
    # ...but alignment fails (35° > 25°) so engagement is denied.
    assert not app._aligned_to_toggle(r0, tog)
    assert not app._back_sensor_overlaps(r0, tog)

    # SC4 must NOT mark UNSET for an off-angle contact.
    app._update_toggle_contact()
    assert tog.state == ToggleState.BLUE

    # And the cycle action must refuse with a "square up" hint.
    app._cycle_nearest_toggle()
    assert tog.resting_state == ToggleState.BLUE, (
        "Off-angle cycle must NOT change the toggle")
    assert "square up" in app.status.lower() or "align" in app.status.lower()

    pygame.quit()
