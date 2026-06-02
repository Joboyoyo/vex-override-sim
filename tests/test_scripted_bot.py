"""Smoke tests for the ScriptedBot match-load runner.

Verifies the state machine cycles through load → drive → place without
crashing, and that the bot actually deposits a pin into an alliance goal."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ai import ScriptedBot
from core.state import Alliance, Phase
from core.scoring import score
from render.pygame_view import make_empty_world


def test_bot_initializes_idle():
    w = make_empty_world()
    bot = ScriptedBot(robot_id=2, world=w)
    assert bot.state == "IDLE"
    assert bot.alliance == Alliance.BLUE
    assert bot.enabled is True


def test_bot_plans_to_loader_when_hands_empty():
    w = make_empty_world()
    bot = ScriptedBot(robot_id=2, world=w)
    # First tick: hands empty -> plan to drive to loader
    bot.update(0.016)
    assert bot.state == "DRIVING_TO_LOADER"
    # Target should be a blue-side loader (x > 0)
    assert bot.target is not None
    assert bot.target[0] > 0


def test_bot_completes_load_deliver_cycle():
    """Run the bot for ~30 simulated seconds. It should drive to a loader,
    spawn a pin+cup, drive to an alliance goal, and place at least one
    item into the stack. We expect the blue score to be > 0 at the end."""
    w = make_empty_world()
    bot = ScriptedBot(robot_id=2, world=w)
    # Position robot near the right-side area so it can reach a loader quickly
    bot.robot.x = 4.0
    bot.robot.y = 4.0
    bot.robot.theta = 0.0

    dt = 0.05
    for _ in range(int(30.0 / dt)):
        bot.update(dt)

    # After 30s, the bot should have looped at least once and put a pin in a blue goal
    blue_pins_in_goals = [p for p in w.pins
                          if p.in_goal is not None
                          and (p.half_a_color.value == "blue" or p.half_b_color.value == "blue")]
    assert len(blue_pins_in_goals) >= 1, (
        f"Bot didn't place any blue pins. State={bot.state}, "
        f"pins={len(w.pins)}, cups={len(w.cups)}")

    # And the blue score should reflect at least one alliance pin
    w.phase = Phase.ENDED
    r = score(w)
    assert r.blue > 0, f"Blue score still 0; bot didn't score. r={r.red},{r.blue}"


def test_bot_can_be_disabled():
    w = make_empty_world()
    bot = ScriptedBot(robot_id=2, world=w)
    bot.enabled = False
    initial_state = bot.state
    bot.update(0.05)
    # Should remain in IDLE since disabled blocks update
    assert bot.state == initial_state


def test_bot_stops_loading_after_budget_exhausted():
    """After delivering MATCH_LOAD_BUDGET (10) pin+cup pairs, the bot must
    transition to POST_LOADS phase and stop spawning new pins. This is the
    fix for the unbounded-pin-spawn lag problem.

    Post-loads, the bot tries to flip ALL toggles to its alliance color and
    then chases the player (DEFENDING). It no longer parks."""
    from ai.scripted_bot import MATCH_LOAD_BUDGET
    from core.state import ToggleState

    w = make_empty_world()
    bot = ScriptedBot(robot_id=2, world=w)
    bot.robot.x, bot.robot.y = 4.0, 4.0

    # Long enough to do all 10 cycles + flip all 4 toggles
    dt = 0.05
    for _ in range(int(240.0 / dt)):
        bot.update(dt)
        # Stop when we've reached DEFENDING with all toggles set
        if (bot.phase == "POST_LOADS" and bot.state == "DEFENDING"
                and all(t.resting_state == ToggleState.BLUE for t in w.toggles)):
            break

    # Budget cap respected
    assert bot.match_loads_delivered == MATCH_LOAD_BUDGET, (
        f"expected {MATCH_LOAD_BUDGET}, got {bot.match_loads_delivered}")
    assert bot.phase == "POST_LOADS"

    # All FOUR toggles should be BLUE (bot flips every toggle, not just its own)
    blue_toggles = [t for t in w.toggles
                    if t.resting_state == ToggleState.BLUE]
    assert len(blue_toggles) == 4, (
        f"Bot should set all 4 toggles to BLUE; only set {len(blue_toggles)}")


def test_bot_collides_at_ignores_other_robots():
    """Robot-vs-robot collision is now resolved in the post-motion overlap
    pass (so robots can push each other). The per-step collides_at closure
    only blocks on STATIC obstacles (goals). This test confirms that."""
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.init()
    pygame.display.set_mode((1400, 900))

    from render.pygame_view import App

    app = App()
    bot = app.bots[0]
    # Place the bot in clear field
    bot.robot.x, bot.robot.y = 3.0, 0.0
    # Move another robot right on top of the bot
    other = next(r for r in app.world.robots if r.id != bot.robot_id)
    other.x = 3.0
    other.y = 0.0
    # Per-step collision check should NOT report a collision now —
    # the post-motion resolve step handles overlaps instead.
    assert bot.collides_at(bot.robot.x, bot.robot.y) is False, (
        "Bot's collides_at should not block on other robots (resolve step does that)")

    # But it SHOULD still collide with goals
    g8 = next(g for g in app.world.goals if g.id == 8)
    assert bot.collides_at(g8.x, g8.y) is True, (
        "Bot's collides_at should still block on goals")

    pygame.quit()


def test_robot_overlap_resolves_with_50_50_split():
    """Two overlapping robots get pushed apart with a 50/50 split, so either
    robot (player or bot) can shove the other. After resolution the centers
    must be at least 2*radius apart."""
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.init()
    pygame.display.set_mode((1400, 900))

    from render.pygame_view import App, ROBOT_COLLISION_RADIUS_FT
    import math

    app = App()
    r0 = next(r for r in app.world.robots if r.id == 0)
    r2 = next(r for r in app.world.robots if r.id == 2)
    # Place both at (1, 0) — exactly overlapping. Move others away.
    r0.x, r0.y = 1.0, 0.0
    r2.x, r2.y = 1.0, 0.0
    for r in app.world.robots:
        if r.id not in (0, 2):
            r.x, r.y = -5.8, 5.8

    app._resolve_robot_overlaps()

    d = math.hypot(r2.x - r0.x, r2.y - r0.y)
    min_d = 2 * ROBOT_COLLISION_RADIUS_FT
    assert d >= min_d - 1e-6, (
        f"Robots not separated: d={d:.3f}, need >= {min_d:.3f}")

    # 50/50: BOTH must have moved roughly the same distance from (1, 0).
    # The exact-overlap case pushes along +x so r0 ends at smaller x and
    # r2 ends at larger x.
    assert r0.x < 1.0, f"R0 should have been pushed (left), got x={r0.x}"
    assert r2.x > 1.0, f"R2 should have been pushed (right), got x={r2.x}"
    # And their displacements should be approximately equal (50/50)
    d0 = math.hypot(r0.x - 1.0, r0.y - 0.0)
    d2 = math.hypot(r2.x - 1.0, r2.y - 0.0)
    assert abs(d0 - d2) < 0.01, (
        f"50/50 split violated: r0 moved {d0:.3f}, r2 moved {d2:.3f}")

    pygame.quit()


def test_driver_phase_ends_at_105_seconds():
    """The 1:45 driver-period timer auto-transitions phase to ENDED at 105 s."""
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame
    pygame.init()
    pygame.display.set_mode((1400, 900))

    from render.pygame_view import App, DRIVER_DURATION_S
    from core.state import Phase

    app = App()
    # Manually transition to DRIVER (what SPACE does on the first press)
    app._advance_phase()
    assert app.world.phase == Phase.DRIVER
    assert app.world.time_elapsed == 0.0

    # Just before the cap, still DRIVER
    app.world.time_elapsed = DRIVER_DURATION_S - 0.5
    app._tick_match_clock(0.1)
    assert app.world.phase == Phase.DRIVER

    # Cross the cap → ENDED
    app._tick_match_clock(1.0)
    assert app.world.phase == Phase.ENDED
    assert app.world.time_elapsed == DRIVER_DURATION_S

    pygame.quit()


def test_empty_world_starts_with_goals_empty():
    """Per the user-specified starting configuration, all 9 goals start
    EMPTY — no pins are pre-placed in any goal at world creation."""
    w = make_empty_world()
    for g in w.goals:
        in_goal = [p for p in w.pins if p.in_goal == g.id]
        assert in_goal == [], (
            f"G{g.id} should be empty, has {len(in_goal)} pins")


def test_world_growth_is_bounded():
    """Run the bot for plenty of simulated time and verify the world's
    pin/cup count stays bounded by the budget — not unbounded growth that
    causes the per-frame rendering lag."""
    from ai.scripted_bot import MATCH_LOAD_BUDGET

    w = make_empty_world()
    starting_pins = len(w.pins)
    starting_cups = len(w.cups)
    bot = ScriptedBot(robot_id=2, world=w)

    dt = 0.05
    for _ in range(int(300.0 / dt)):  # 5 simulated minutes
        bot.update(dt)

    spawned_pins = len(w.pins) - starting_pins
    spawned_cups = len(w.cups) - starting_cups
    # Each cycle spawns exactly 1 pin + 1 cup; budget = MATCH_LOAD_BUDGET
    assert spawned_pins <= MATCH_LOAD_BUDGET, (
        f"Bot spawned {spawned_pins} pins, expected <= {MATCH_LOAD_BUDGET}")
    assert spawned_cups <= MATCH_LOAD_BUDGET, (
        f"Bot spawned {spawned_cups} cups, expected <= {MATCH_LOAD_BUDGET}")
