"""CLI for running strategy comparisons.

Examples:
    python -m strategy.analyze                          # default tournament
    python -m strategy.analyze --trials 5000            # more trials
    python -m strategy.analyze --sweep p_my_toggle_held # toggle-hold rate sweep
"""

from __future__ import annotations

import argparse

from core.state import Alliance

from .scenario import MatchScenario
from .simulate import run_matchup
from .strategies import STRATEGY_REGISTRY, make_strategy


def tournament(trials: int = 1000, seed: int = 0) -> None:
    """Run a full round-robin of every strategy vs every strategy."""
    scenario = MatchScenario()
    names = list(STRATEGY_REGISTRY)

    print(f"\n=== Tournament  ({trials} trials per matchup, default scenario) ===\n")
    print(f"Scenario: p_my_toggle_held={scenario.p_my_toggle_held}  "
          f"p_opp_toggle_denied={scenario.p_opp_toggle_denied}  "
          f"auto_pin_success={scenario.auto_pin_success}")
    print()

    # Header row
    col_w = 18
    print(" " * col_w + "".join(f"{n:>{col_w}}" for n in names))

    # Each row: red strategy, columns: blue strategy, entry: red win rate
    for red_name in names:
        row = [f"{red_name:>{col_w-2}}: "]
        for blue_name in names:
            red = make_strategy(red_name, Alliance.RED)
            blue = make_strategy(blue_name, Alliance.BLUE)
            stats = run_matchup(red, blue, scenario, trials=trials, seed=seed)
            row.append(f"{stats.red_win_rate:>{col_w-3}.1%} ")
        print("".join(row))
    print()

    # Detail: each strategy vs every other, full summary
    print("\n=== Detailed matchup summaries ===\n")
    for red_name in names:
        for blue_name in names:
            if red_name == blue_name:
                continue
            red = make_strategy(red_name, Alliance.RED)
            blue = make_strategy(blue_name, Alliance.BLUE)
            stats = run_matchup(red, blue, scenario, trials=trials, seed=seed)
            print(stats.summary())
            print()


def sweep(parameter: str, trials: int = 1000, seed: int = 0,
          red_name: str = "safe", blue_name: str = "yellow_gamble") -> None:
    """Sweep a single MatchScenario parameter from 0.0 to 1.0 in steps of 0.1,
    showing how red's win rate against blue varies."""
    print(f"\n=== Sweep: {parameter}  ({red_name} vs {blue_name}, {trials} trials per point) ===\n")
    print(f"  {parameter:>30}    red_mean   blue_mean   red_win   blue_win   margin")
    print("  " + "-" * 80)

    for x in [i / 10 for i in range(0, 11)]:
        scenario = MatchScenario(**{parameter: x})
        red = make_strategy(red_name, Alliance.RED)
        blue = make_strategy(blue_name, Alliance.BLUE)
        stats = run_matchup(red, blue, scenario, trials=trials, seed=seed)
        print(f"  {parameter:>20} = {x:.1f}    "
              f"{stats.red_mean:7.1f}    {stats.blue_mean:7.1f}    "
              f"{stats.red_win_rate:6.1%}    {stats.blue_win_rate:6.1%}    "
              f"{stats.margin_mean:+6.1f}")
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="Strategy Monte Carlo for VEX Override")
    p.add_argument("--trials", type=int, default=1000,
                   help="Trials per matchup (default 1000)")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed (default 0)")
    p.add_argument("--sweep", type=str, default=None,
                   help="Parameter name to sweep (e.g. p_my_toggle_held)")
    p.add_argument("--red", type=str, default="safe",
                   help="Red strategy for sweep mode")
    p.add_argument("--blue", type=str, default="yellow_gamble",
                   help="Blue strategy for sweep mode")
    args = p.parse_args()

    if args.sweep:
        sweep(args.sweep, trials=args.trials, seed=args.seed,
              red_name=args.red, blue_name=args.blue)
    else:
        tournament(trials=args.trials, seed=args.seed)


if __name__ == "__main__":
    main()
