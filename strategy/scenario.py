"""MatchScenario + StrategySpec dataclasses.

A MatchScenario encodes the uncertainty in a single number — probabilities of
each action succeeding, each toggle surviving denial, etc. A StrategySpec
encodes one alliance's plan in terms abstract enough to ignore physics.

Both are alliance-agnostic where possible. Goal IDs / quadrant IDs in a
StrategySpec are filled in by the per-alliance factory in strategies.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from core.state import Alliance


OrientChoice = Literal["safe", "yellow"]
# "safe"   = orient pin so the alliance-color half is visible (5 pts, locked)
# "yellow" = orient pin so the yellow half is visible (10 pts if toggle/midfield owned,
#            0 if unset/yellow, 10 to opponent if they own the toggle)


@dataclass
class MatchScenario:
    """Probabilities governing one match. All in [0, 1].

    The defaults represent a competent-but-not-elite team facing an opponent
    of similar caliber. Sweep these to explore the strategy space.
    """

    # --- Action execution ---
    auto_pin_success: float = 0.90
        # Per attempted auto pin placement
    auto_toggle_set_success: float = 0.85
        # Per attempted toggle set in auto (clean approach + release)
    driver_pin_success: float = 0.92
        # Per attempted driver-period pin placement (match load + stack)
    p_robots_clear_perimeter: float = 0.95
        # Both robots end auto not touching the field perimeter (AWP requirement)

    # --- Buzzer toggle warfare ---
    p_my_toggle_held: float = 0.50
        # Probability your auto-set toggle survives opponent denial through to
        # the buzzer. Low = opponent is good at denying; high = your defense holds.
    p_opp_toggle_denied: float = 0.50
        # Probability you successfully deny one of opponent's set toggles at the
        # buzzer (touching it at t=2:00 = not set = unset).

    # --- Midfield positioning ---
    p_robot_reaches_midfield: float = 0.80
        # Per robot the strategy intends to park in midfield

    def deterministic(self, *, all_success: bool = True) -> "MatchScenario":
        """Return a scenario where everything succeeds (or fails) with certainty.
        Useful for testing — strategies should produce closed-form scores."""
        v = 1.0 if all_success else 0.0
        return MatchScenario(
            auto_pin_success=v,
            auto_toggle_set_success=v,
            driver_pin_success=v,
            p_robots_clear_perimeter=v,
            p_my_toggle_held=v,
            p_opp_toggle_denied=v,
            p_robot_reaches_midfield=v,
        )


@dataclass
class StrategySpec:
    """One alliance's plan, after the alliance is known.

    All goal IDs and quadrant IDs are concrete (filled in by the alliance-aware
    factory in strategies.py).
    """

    name: str
    alliance: Alliance

    # --- Auto plan ---
    auto_pin_attempts: int                  # how many pins to try to place
    auto_pin_orient: OrientChoice           # how to orient them
    auto_goal_ids: list[int]                # which goals to spread pins across (round-robin)
    auto_toggle_quadrants: list[int]        # which toggles to attempt to set

    # --- Driver match-load plan ---
    match_load_attempts: int                # how many match loads to try
    match_load_orient: OrientChoice         # red-side (safe) or yellow-side (gamble)
    match_load_goal_id: int                 # which goal to stack them in

    # --- Endgame plan ---
    deny_opp_quadrants: list[int] = field(default_factory=list)
        # Opponent toggle quadrants to deny at the buzzer
    midfield_robots: int = 0
        # Number of robots (0/1/2) to attempt to park in midfield at match end
