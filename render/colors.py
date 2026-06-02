"""Light-mode palette tuned to match the real V5RC Override field.

Field colors reference Figure FO-1 in the Game Manual:
- Foam-tile gray (medium gray, faint checker)
- Green goals
- Orange alliance stations
- White loaders
- Pink toggles
- White autonomous line / center markings
"""

from __future__ import annotations

# -- Canvas / outer window ----------------------------------------------------

BACKGROUND       = (240, 242, 246)
PANEL_BG         = (255, 255, 255)
PANEL_BG_DARK    = (246, 248, 251)
PANEL_BORDER     = (218, 224, 232)
PANEL_BORDER_FOCUSED = (90, 130, 220)
PANEL_HIGHLIGHT  = (236, 242, 252)
PANEL_SHADOW     = (0, 0, 0, 22)

# -- Field tiles + perimeter --------------------------------------------------

FIELD_BG              = (172, 174, 178)
FIELD_TILE_A          = (178, 180, 184)
FIELD_TILE_B          = (164, 166, 170)
FIELD_TILE_GRIDLINE   = (152, 154, 158)
FIELD_PERIMETER       = (38, 42, 50)
FIELD_PERIMETER_HIGHLIGHT = (90, 94, 102)

AUTONOMOUS_LINE       = (250, 250, 250)
MIDFIELD_LINES        = (250, 250, 250)
MIDFIELD_OVERLAY      = (200, 215, 230, 38)

# -- Alliance colors (VEX-tuned) ----------------------------------------------

RED              = (215, 30, 40)
RED_DEEP         = (165, 22, 32)
RED_DIM          = (250, 215, 218)

BLUE             = (30, 100, 215)
BLUE_DEEP        = (20, 70, 165)
BLUE_DIM         = (210, 225, 250)

YELLOW           = (252, 196, 25)
YELLOW_DEEP      = (210, 158, 14)
YELLOW_DIM       = (255, 240, 190)

# -- Field elements -----------------------------------------------------------

GOAL_GREEN          = (76, 175, 80)
GOAL_GREEN_DEEP     = (40, 120, 45)
GOAL_GREEN_DIM      = (200, 230, 200)
TALL_GOAL_HIGHLIGHT = (130, 200, 110)
TALL_GOAL_GLOW      = (170, 230, 130)

ALLIANCE_STATION_OUTLINE = (180, 100, 16)
ALLIANCE_STATION_FILL    = (255, 152, 28)
ALLIANCE_STATION_BAND    = (210, 124, 14)

LOADER_FILL    = (250, 250, 250)
LOADER_OUTLINE = (140, 145, 155)

TOGGLE_PINK   = (220, 70, 140)
TOGGLE_BORDER = (148, 38, 88)
TOGGLE_UNSET_FILL = (200, 200, 200)
TOGGLE_UNSET_BORDER = (148, 152, 158)

CUP_OPAQUE    = (62, 68, 78)             # gray-up side of cup
CUP_CLEAR     = (210, 222, 232)          # clear-up side
CUP_OUTLINE   = (38, 42, 50)
CUP_RIM_HIGHLIGHT = (170, 180, 192)

# -- Neutrals -----------------------------------------------------------------

NEUTRAL          = (108, 116, 130)
NEUTRAL_DEEP     = (64, 70, 82)
NEUTRAL_DIM      = (200, 204, 210)

GOAL_BORDER      = (38, 42, 50)

# -- Text ---------------------------------------------------------------------

TEXT_PRIMARY     = (22, 26, 34)
TEXT_SECONDARY   = (78, 88, 102)
TEXT_DIM         = (148, 156, 168)
TEXT_HIGHLIGHT   = (180, 90, 20)
TEXT_RED         = RED
TEXT_BLUE        = BLUE
TEXT_SUCCESS     = (40, 150, 80)
TEXT_WARNING     = (218, 130, 36)

# -- State indicators ---------------------------------------------------------

OK_GREEN         = (50, 160, 80)
WARN_AMBER       = (240, 158, 50)
ERR_RED          = (215, 60, 60)


def with_alpha(rgb: tuple[int, int, int], alpha: int) -> tuple[int, int, int, int]:
    return (rgb[0], rgb[1], rgb[2], alpha)


def lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return (int(a[0] + (b[0] - a[0]) * t),
            int(a[1] + (b[1] - a[1]) * t),
            int(a[2] + (b[2] - a[2]) * t))
