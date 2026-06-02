"""Pygame renderer for the Override sim.

Architecture:
- colors.py     -- single source of truth for the palette
- coords.py     -- field (feet) <-> pixel coordinate transforms
- draw.py       -- low-level draw functions for each game element
- hud.py        -- HUD panels (score, toggles, help, timer)
- pygame_view.py -- main loop, input handling, world editor

Run:
    python -m render.pygame_view
"""
