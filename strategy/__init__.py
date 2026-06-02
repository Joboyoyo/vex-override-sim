"""Strategy Monte Carlo for VEX Override.

Lets you simulate matches between named strategies under parameterized
uncertainty, without touching physics or graphics. Built directly on top of
core/scoring.py.

Quickstart:
    from strategy import scenario, simulate, strategies

    s = scenario.MatchScenario()           # default mid-skill parameters
    red = strategies.make_strategy("safe", Alliance.RED)
    blue = strategies.make_strategy("yellow_gamble", Alliance.BLUE)

    stats = simulate.run_matchup(red, blue, s, trials=10_000)
    print(stats.summary())
"""
