"""Field & element drawing functions, tuned to match the real V5RC Override field.

Each function is pure — takes a Surface + state, draws onto it. No globals.

Visual references (from Figure FO-1 in the manual):
- Foam-tile field background (gray with faint checker)
- Goals as green blocks
- Orange alliance stations outside the field perimeter on left & right
- White loaders at the 4 corners adjacent to alliance stations
- Pink toggles at the wall midpoints
- White autonomous markings (center "X" and vertical line)
"""

from __future__ import annotations

import math
from typing import Optional

import pygame
import pygame.gfxdraw

from core.state import (
    Alliance, Cup, Goal, GoalType, Pin, PinColor, Robot, Toggle, ToggleState,
    World,
)

from . import colors as C
from .coords import FieldLayout, WindowLayout


# -- Helpers ------------------------------------------------------------------


def _aa_filled_circle(surface, x: int, y: int, r: int, color) -> None:
    if r <= 0:
        return
    pygame.gfxdraw.filled_circle(surface, x, y, r, color)
    pygame.gfxdraw.aacircle(surface, x, y, r, color)


def _aa_ring(surface, x: int, y: int, r: int, color, thickness: int = 2) -> None:
    for i in range(thickness):
        pygame.gfxdraw.aacircle(surface, x, y, r - i, color)


def _drop_shadow(surface, rect: pygame.Rect, *, radius: int = 4, alpha: int = 40) -> None:
    s = pygame.Surface((rect.width + radius * 2, rect.height + radius * 2), pygame.SRCALPHA)
    pygame.draw.rect(s, (0, 0, 0, alpha),
                     (radius, radius + 2, rect.width, rect.height),
                     border_radius=8)
    surface.blit(s, (rect.left - radius, rect.top - radius))


def _color_for_alliance(a: Alliance) -> tuple[int, int, int]:
    if a == Alliance.RED: return C.RED
    if a == Alliance.BLUE: return C.BLUE
    return C.NEUTRAL


def _color_for_pin_half(color: PinColor) -> tuple[int, int, int]:
    if color == PinColor.RED:    return C.RED
    if color == PinColor.BLUE:   return C.BLUE
    return C.YELLOW


# -- Field background ---------------------------------------------------------


def _draw_tape_segment(surface, p1: tuple[int, int], p2: tuple[int, int],
                       color, thickness: int = 4) -> None:
    pygame.draw.line(surface, color, p1, p2, thickness)


def _draw_double_tape(surface, p1: tuple[int, int], p2: tuple[int, int],
                       color, thickness: int = 4, gap_px: int = 8) -> None:
    """Two parallel tapes from p1 to p2, offset perpendicular to the segment
    by ±gap_px/2 in screen pixels."""
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    length = (dx * dx + dy * dy) ** 0.5
    if length == 0:
        return
    # Unit perpendicular (rotated 90°)
    px = -dy / length
    py = dx / length
    off = gap_px / 2
    o = (px * off, py * off)
    pygame.draw.line(surface, color,
                     (int(p1[0] + o[0]), int(p1[1] + o[1])),
                     (int(p2[0] + o[0]), int(p2[1] + o[1])), thickness)
    pygame.draw.line(surface, color,
                     (int(p1[0] - o[0]), int(p1[1] - o[1])),
                     (int(p2[0] - o[0]), int(p2[1] - o[1])), thickness)


def _draw_loading_zone_l(surface, layout: FieldLayout,
                          corner_ft: tuple[float, float],
                          end1_ft: tuple[float, float],
                          end2_ft: tuple[float, float],
                          color, thickness: int = 5) -> None:
    """L-shaped tape: corner_ft connected to end1_ft and end2_ft."""
    p_c = layout.ft_to_px(*corner_ft)
    p1 = layout.ft_to_px(*end1_ft)
    p2 = layout.ft_to_px(*end2_ft)
    pygame.draw.line(surface, color, p_c, p1, thickness)
    pygame.draw.line(surface, color, p_c, p2, thickness)


def draw_field_background(surface, layout: FieldLayout) -> None:
    """Foam-tile checker + perimeter + autonomous double-tape + quadrant
    diagonal tapes + 4 loading-zone L-tapes + midfield diamond."""
    field_rect = pygame.Rect(layout.origin_x, layout.origin_y,
                             layout.size_px, layout.size_px)
    _drop_shadow(surface, field_rect, radius=8, alpha=60)
    pygame.draw.rect(surface, C.FIELD_BG, field_rect)

    # 6x6 tile checker
    tile_size = layout.size_px // 6
    for i in range(6):
        for j in range(6):
            color = C.FIELD_TILE_A if (i + j) % 2 == 0 else C.FIELD_TILE_B
            pygame.draw.rect(
                surface, color,
                (layout.origin_x + j * tile_size,
                 layout.origin_y + i * tile_size,
                 tile_size, tile_size),
            )

    # Fine gridlines between tiles
    for i in range(1, 6):
        pos = i * tile_size
        pygame.draw.line(surface, C.FIELD_TILE_GRIDLINE,
                         (layout.origin_x + pos, layout.origin_y),
                         (layout.origin_x + pos, layout.origin_y + layout.size_px), 1)
        pygame.draw.line(surface, C.FIELD_TILE_GRIDLINE,
                         (layout.origin_x, layout.origin_y + pos),
                         (layout.origin_x + layout.size_px, layout.origin_y + pos), 1)

    # Midfield overlay tint (drawn under the diagonal tapes)
    cx, cy = layout.center_px
    diamond_r = layout.ft_len_to_px(2.0)
    overlay = pygame.Surface((diamond_r * 2 + 8, diamond_r * 2 + 8), pygame.SRCALPHA)
    pygame.draw.polygon(overlay, C.MIDFIELD_OVERLAY,
                        [(diamond_r + 4, 4), (diamond_r * 2 + 4, diamond_r + 4),
                         (diamond_r + 4, diamond_r * 2 + 4), (4, diamond_r + 4)])
    surface.blit(overlay, (cx - diamond_r - 4, cy - diamond_r - 4))

    # Quadrant diagonal y = x — single white tape from midfield diamond
    # midpoints (1, 1) and (-1, -1) out to where each loading-zone L starts
    # at (5, 5) / (-5, -5). Tape stops at the loading zone, doesn't enter it.
    _draw_tape_segment(surface,
                       layout.ft_to_px(1, 1), layout.ft_to_px(5, 5),
                       C.AUTONOMOUS_LINE, thickness=4)
    _draw_tape_segment(surface,
                       layout.ft_to_px(-1, -1), layout.ft_to_px(-5, -5),
                       C.AUTONOMOUS_LINE, thickness=4)

    # Autonomous line y = -x — DOUBLE parallel tape, from midfield diamond
    # midpoints (-1, 1) / (1, -1) out to the loading-zone start (-5, 5) / (5, -5).
    _draw_double_tape(surface,
                      layout.ft_to_px(-1, 1), layout.ft_to_px(-5, 5),
                      C.AUTONOMOUS_LINE, thickness=4, gap_px=10)
    _draw_double_tape(surface,
                      layout.ft_to_px(1, -1), layout.ft_to_px(5, -5),
                      C.AUTONOMOUS_LINE, thickness=4, gap_px=10)

    # Loading-zone L-tapes — 4 corners, alliance-colored per nearest station.
    # Red station = LEFT wall (x = -6) -> top-left + bottom-left = red.
    # Blue station = RIGHT wall (x = +6) -> top-right + bottom-right = blue.
    _draw_loading_zone_l(surface, layout, (5, 4),  (5, 6),  (6, 4),  C.BLUE)   # top-right
    _draw_loading_zone_l(surface, layout, (5, -4), (5, -6), (6, -4), C.BLUE)   # bot-right
    _draw_loading_zone_l(surface, layout, (-5, 4), (-5, 6), (-6, 4), C.RED)    # top-left
    _draw_loading_zone_l(surface, layout, (-5, -4),(-5, -6),(-6, -4),C.RED)    # bot-left

    # Central midfield diamond outline (drawn on top of tapes so it stays clean)
    diamond_pts = [(cx, cy - diamond_r), (cx + diamond_r, cy),
                   (cx, cy + diamond_r), (cx - diamond_r, cy)]
    pygame.draw.polygon(surface, C.MIDFIELD_LINES, diamond_pts, 4)

    # Field perimeter
    pygame.draw.rect(surface, C.FIELD_PERIMETER, field_rect, 4)
    pygame.draw.rect(surface, C.FIELD_PERIMETER_HIGHLIGHT, field_rect.inflate(-8, -8), 1)


# -- Alliance stations & loaders ---------------------------------------------


def draw_alliance_stations(surface, wl: WindowLayout, world: World) -> None:
    """Orange alliance stations on either side of the field, with simple
    match-load indicator at the bottom."""
    for rect_tuple, alliance, label in [
        (wl.red_station_rect,  Alliance.RED,  "RED"),
        (wl.blue_station_rect, Alliance.BLUE, "BLUE"),
    ]:
        x, y, w, h = rect_tuple
        rect = pygame.Rect(x, y, w, h)

        _drop_shadow(surface, rect, radius=6, alpha=40)
        pygame.draw.rect(surface, C.ALLIANCE_STATION_FILL, rect, border_radius=6)
        pygame.draw.rect(surface, C.ALLIANCE_STATION_OUTLINE, rect, 2, border_radius=6)
        # Top + bottom bands (decorative)
        pygame.draw.rect(surface, C.ALLIANCE_STATION_BAND,
                         (x, y + 4, w, 6))
        pygame.draw.rect(surface, C.ALLIANCE_STATION_BAND,
                         (x, y + h - 10, w, 6))

        # Alliance label rotated 90 degrees
        font = pygame.font.SysFont("Segoe UI", 14, bold=True)
        a_color = C.RED_DEEP if alliance == Alliance.RED else C.BLUE_DEEP
        label_surf = font.render(label, True, a_color)
        rotated = pygame.transform.rotate(label_surf, 90)
        surface.blit(rotated, (x + (w - rotated.get_width()) // 2,
                               y + (h - rotated.get_height()) // 2))


# Loader field positions (feet, centered field coords).
# 2 loaders adjacent to each alliance, mounted on the side wall (x = ±6).
LOADER_POSITIONS_FT = [
    (-6.0, 5.0),    # red — top-left wall mount
    (-6.0, -5.0),   # red — bottom-left wall mount
    (6.0, 5.0),     # blue — top-right wall mount
    (6.0, -5.0),    # blue — bottom-right wall mount
]


def draw_loaders(surface, layout: FieldLayout) -> None:
    """4 wall-mounted loaders. Each rectangle straddles the field perimeter
    so it visually reads as embedded in the wall (half outside, half inside)."""
    long_axis_ft = 1.4        # 1.4 ft along the wall
    short_axis_ft = 0.6       # 0.6 ft perpendicular (sticks out from wall)
    long_px = layout.ft_len_to_px(long_axis_ft)
    short_px = layout.ft_len_to_px(short_axis_ft)
    for x_ft, y_ft in LOADER_POSITIONS_FT:
        cx, cy = layout.ft_to_px(x_ft, y_ft)
        # Wall loaders run along the wall; left/right walls -> vertical long axis
        if abs(x_ft) > abs(y_ft):
            w, h = short_px, long_px
        else:
            w, h = long_px, short_px
        rect = pygame.Rect(cx - w // 2, cy - h // 2, w, h)
        _drop_shadow(surface, rect, radius=3, alpha=50)
        pygame.draw.rect(surface, C.LOADER_FILL, rect, border_radius=4)
        pygame.draw.rect(surface, C.LOADER_OUTLINE, rect, 2, border_radius=4)
        font = pygame.font.SysFont("Segoe UI", 10, bold=True)
        label = font.render("LOAD", True, (90, 100, 115))
        surface.blit(label, (cx - label.get_width() // 2,
                             cy - label.get_height() // 2))


# -- Goals --------------------------------------------------------------------


def draw_goal(surface, goal: Goal, layout: FieldLayout, hovered: bool = False) -> None:
    """Goals rendered as rounded green blocks, all the same size.
    Alliance goals have a colored border; tall goal has a glow."""
    cx, cy = layout.ft_to_px(goal.x, goal.y)

    # All goals are the same size now (per user request)
    size = 36
    fill = C.GOAL_GREEN

    if goal.type == GoalType.TALL:
        border = C.GOAL_GREEN_DEEP
        border_thickness = 3
    elif goal.type == GoalType.ALLIANCE:
        border = _color_for_alliance(goal.alliance)
        border_thickness = 4
    else:  # SHORT
        border = C.GOAL_GREEN_DEEP
        border_thickness = 2

    rect = pygame.Rect(cx - size // 2, cy - size // 2, size, size)

    # Shadow + glow
    _drop_shadow(surface, rect, radius=4, alpha=60)
    if goal.type == GoalType.TALL:
        glow = pygame.Surface((size + 32, size + 32), pygame.SRCALPHA)
        for i in range(10, 0, -1):
            a = 24 - i * 2
            if a < 0: a = 0
            pygame.draw.rect(glow, (*C.TALL_GOAL_GLOW, a),
                             (16 - i, 16 - i, size + 2 * i, size + 2 * i),
                             border_radius=8)
        surface.blit(glow, (rect.left - 16, rect.top - 16))

    pygame.draw.rect(surface, fill, rect, border_radius=8)
    # Inner highlight
    inner_rect = rect.inflate(-8, -8)
    inner_color = C.TALL_GOAL_HIGHLIGHT if goal.type == GoalType.TALL else C.GOAL_GREEN_DIM
    pygame.draw.rect(surface, inner_color, inner_rect, border_radius=4)
    pygame.draw.rect(surface, border, rect, border_thickness, border_radius=8)

    if hovered:
        pygame.draw.rect(surface, C.PANEL_BORDER_FOCUSED,
                         rect.inflate(8, 8), 2, border_radius=10)

    # Goal id label
    font = pygame.font.SysFont("Segoe UI", 11, bold=True)
    label = font.render(f"G{goal.id}", True, C.GOAL_GREEN_DEEP)
    surface.blit(label, (cx - label.get_width() // 2, cy + size // 2 + 2))


# -- Pins ---------------------------------------------------------------------

PIN_ICON_W = 32       # hexagon flat-to-flat width
PIN_ICON_H = 38       # hexagon point-to-point height


def draw_pin_icon(surface, cx: int, cy: int, pin: Pin, *,
                  top_visible: bool, bot_visible: bool,
                  hovered: bool = False) -> None:
    """Small hexagonal pin icon for the top-down field view.

    Top half = pin.half_a_color (visible color when top_visible)
    Bottom half = pin.half_b_color (visible color when bot_visible)
    Hidden halves are darkened so the user sees them but knows they're not scoring.
    """
    W, H = PIN_ICON_W, PIN_ICON_H
    half_h = H // 2
    quarter_h = H // 4

    color_a = _color_for_pin_half(pin.half_a_color)
    color_b = _color_for_pin_half(pin.half_b_color)

    def _dim(c: tuple[int, int, int]) -> tuple[int, int, int]:
        # Heavy darkening so hidden halves clearly differ from visible
        return (c[0] // 3, c[1] // 3, c[2] // 3)

    top_fill = color_a if top_visible else _dim(color_a)
    bot_fill = color_b if bot_visible else _dim(color_b)

    # Top half pentagon (top tip + slanted shoulders + flat bottom at midline)
    top_pts = [
        (cx - W // 2, cy - quarter_h),
        (cx,          cy - half_h),
        (cx + W // 2, cy - quarter_h),
        (cx + W // 2, cy),
        (cx - W // 2, cy),
    ]
    pygame.gfxdraw.filled_polygon(surface, top_pts, top_fill)

    # Bottom half pentagon
    bot_pts = [
        (cx - W // 2, cy),
        (cx + W // 2, cy),
        (cx + W // 2, cy + quarter_h),
        (cx,          cy + half_h),
        (cx - W // 2, cy + quarter_h),
    ]
    pygame.gfxdraw.filled_polygon(surface, bot_pts, bot_fill)

    # Full hexagon outline
    hex_pts = [
        (cx - W // 2, cy - quarter_h),
        (cx,          cy - half_h),
        (cx + W // 2, cy - quarter_h),
        (cx + W // 2, cy + quarter_h),
        (cx,          cy + half_h),
        (cx - W // 2, cy + quarter_h),
    ]
    pygame.draw.polygon(surface, C.GOAL_BORDER, hex_pts, 2)
    # Mid divider
    pygame.draw.line(surface, C.GOAL_BORDER,
                     (cx - W // 2, cy), (cx + W // 2, cy), 2)

    if hovered:
        pygame.draw.polygon(surface, C.PANEL_BORDER_FOCUSED,
                            [(p[0] + (-2 if i in [0, 4, 5] else 2),
                              p[1] + (-2 if i in [0, 1, 2] else 2))
                             for i, p in enumerate(hex_pts)], 2)


# -- Cups (small SQUARE icon for field view) ----------------------------------

CUP_ICON_W = 34
CUP_ICON_H = 22


def draw_cup_icon(surface, cx: int, cy: int, cup: Cup, *,
                  hovered: bool = False) -> None:
    """Small SQUARE cup icon for the top-down field view.

    Top half = top face (clear if clear_face_up else gray)
    Bottom half = the opposite

    Gray half is solid dark; clear half is light with horizontal "glass" lines
    so the two are unambiguously distinguishable.
    """
    W, H = CUP_ICON_W, CUP_ICON_H
    rect = pygame.Rect(cx - W // 2, cy - H // 2, W, H)
    top_rect = pygame.Rect(rect.left, rect.top, W, H // 2)
    bot_rect = pygame.Rect(rect.left, rect.top + H // 2, W, H - H // 2)

    if cup.clear_face_up:
        top_kind, bot_kind = "clear", "gray"
    else:
        top_kind, bot_kind = "gray", "clear"

    def _draw_face(face_rect: pygame.Rect, kind: str) -> None:
        if kind == "gray":
            pygame.draw.rect(surface, C.CUP_OPAQUE, face_rect)
        else:
            pygame.draw.rect(surface, C.CUP_CLEAR, face_rect)
            # Glass lines so user immediately sees this is the see-through half
            for i in range(face_rect.top + 2, face_rect.bottom - 1, 3):
                pygame.draw.line(surface, (160, 190, 210),
                                 (face_rect.left + 2, i),
                                 (face_rect.right - 2, i), 1)

    _draw_face(top_rect, top_kind)
    _draw_face(bot_rect, bot_kind)

    pygame.draw.rect(surface, C.CUP_OUTLINE, rect, 2)
    pygame.draw.line(surface, C.CUP_OUTLINE,
                     (rect.left, cy), (rect.right, cy), 2)

    if hovered:
        pygame.draw.rect(surface, C.PANEL_BORDER_FOCUSED,
                         rect.inflate(6, 6), 2, border_radius=3)


# -- Toggles ------------------------------------------------------------------


TOGGLE_LENGTH_FT = 2.0       # ~2 ft along the wall
TOGGLE_THICKNESS_FT = 0.5    # narrow side


def _toggle_color_for(state: ToggleState) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if state == ToggleState.RED:    return C.RED,    C.RED_DEEP
    if state == ToggleState.BLUE:   return C.BLUE,   C.BLUE_DEEP
    if state == ToggleState.YELLOW: return C.YELLOW, C.YELLOW_DEEP
    return C.TOGGLE_PINK, C.TOGGLE_BORDER


def draw_toggle(surface, toggle: Toggle, layout: FieldLayout, hovered: bool = False) -> None:
    """Pink toggle on field perimeter, ~2 ft long along the wall it sits on.
    When `state == UNSET` (a robot is touching it) we still hint at the
    `resting_state` color underneath, drawn semi-transparently so the player
    can see what color the toggle will settle into when the robot leaves."""
    cx, cy = layout.ft_to_px(toggle.x, toggle.y)

    is_unset = (toggle.state == ToggleState.UNSET)
    if is_unset:
        # Draw a low-opacity wash of the resting_state color so the user knows
        # what the toggle will become as soon as the robot moves off it.
        fill, border = _toggle_color_for(toggle.resting_state)
    else:
        fill, border = _toggle_color_for(toggle.state)

    # Long axis along the wall (2 ft), short axis perpendicular (~0.5 ft).
    # If toggle sits on left/right wall (|x| > |y|), the long axis is vertical.
    # If on top/bottom wall, long axis is horizontal.
    long_px = layout.ft_len_to_px(TOGGLE_LENGTH_FT)
    short_px = max(8, layout.ft_len_to_px(TOGGLE_THICKNESS_FT))

    if abs(toggle.x) > abs(toggle.y):
        w, h = short_px, long_px       # left/right wall → vertical
    else:
        w, h = long_px, short_px       # top/bottom wall → horizontal
    rect = pygame.Rect(cx - w // 2, cy - h // 2, w, h)

    # Glow only when the toggle is fully "set" (not while a robot is touching)
    if not is_unset:
        glow = pygame.Surface((w + 18, h + 18), pygame.SRCALPHA)
        for i in range(8, 0, -1):
            a = 42 - i * 4
            if a < 0: a = 0
            pygame.draw.rect(glow, (*fill, a),
                             (9 - i, 9 - i, w + 2 * i, h + 2 * i),
                             border_radius=3)
        surface.blit(glow, (rect.left - 9, rect.top - 9))

    if is_unset:
        # Semi-transparent wash of resting_state color
        wash = pygame.Surface((w, h), pygame.SRCALPHA)
        pygame.draw.rect(wash, (*fill, 90), (0, 0, w, h), border_radius=3)
        surface.blit(wash, (rect.left, rect.top))
        # Dashed/diagonal pattern to clearly signal "not locked in"
        for x_off in range(-h, w, 6):
            x1 = rect.left + max(0, x_off)
            y1 = rect.top + max(0, -x_off)
            x2 = rect.left + min(w, x_off + h)
            y2 = rect.top + min(h, h + (x_off + h - w) if x_off + h > w else h)
            pygame.draw.line(surface, (*border, 120), (x1, y1), (x2, y2), 1)
        # Outline kept solid so the toggle is still readable
        pygame.draw.rect(surface, border, rect, 2, border_radius=3)
    else:
        pygame.draw.rect(surface, fill, rect, border_radius=3)
        pygame.draw.rect(surface, border, rect, 2, border_radius=3)

    if hovered:
        pygame.draw.rect(surface, C.PANEL_BORDER_FOCUSED,
                         rect.inflate(8, 8), 2, border_radius=5)

    # Quadrant label
    font = pygame.font.SysFont("Segoe UI", 10, bold=True)
    label = font.render(f"Q{toggle.quadrant}", True, C.TEXT_SECONDARY)
    if toggle.x > 5: lp = (cx + 22, cy - 6)
    elif toggle.x < -5: lp = (cx - 36, cy - 6)
    elif toggle.y > 5: lp = (cx - 8, cy - 26)
    else: lp = (cx - 8, cy + 12)
    surface.blit(label, lp)


# -- Robots -------------------------------------------------------------------


ROBOT_LENGTH_FT = 14.0 / 12.0     # 14 inches ≈ 1.167 ft (forward axis)
ROBOT_WIDTH_FT  = 10.0 / 12.0     # 10 inches ≈ 0.833 ft (side axis)

# Optional R0 front wing — when deployed, a thin strip sits in front of the
# bumper: 24" wide perpendicular to heading, 3" thick along heading. The
# strip's BACK edge is at the front bumper so it never reaches behind the
# chassis. Visual matches the push hitbox exactly.
WING_TOTAL_WIDTH_FT = 24.0 / 12.0     # 2.0 ft total span perpendicular to heading
WING_LENGTH_FT       = 3.0 / 12.0     # 0.25 ft thickness along heading


def draw_wings(surface, robot: Robot, layout: FieldLayout) -> None:
    """Draw R0's front wing panels. Called only when wings_extended=True.
    Rendered as two thin strips flanking the chassis at the front bumper —
    the inner half (where the chassis would overlap) is hidden by the body."""
    cx, cy = layout.ft_to_px(robot.x, robot.y)
    half_L      = layout.ft_len_to_px(ROBOT_LENGTH_FT / 2)
    body_half_W = layout.ft_len_to_px(ROBOT_WIDTH_FT / 2)
    wing_half_W = layout.ft_len_to_px(WING_TOTAL_WIDTH_FT / 2)
    wing_thick  = layout.ft_len_to_px(WING_LENGTH_FT)
    # Wing back edge is AT the front bumper; front edge sticks out by thickness
    wing_back   = half_L
    wing_front  = half_L + wing_thick

    cos_t = math.cos(robot.theta)
    sin_t = math.sin(robot.theta)

    def _rot(bx: float, by: float) -> tuple[int, int]:
        wx = bx * cos_t - by * sin_t
        wy = bx * sin_t + by * cos_t
        return (int(cx + wx), int(cy - wy))   # screen y flipped

    fill   = (255, 215,  60, 255)
    border = (160, 110,  20)

    # Two thin panels, one per side. Inner edge at body, outer edge at the
    # full 24" span. The wing is the same thickness over its whole span.
    left_wing = [
        _rot(wing_front, body_half_W),    # inner-front
        _rot(wing_front, wing_half_W),    # outer-front
        _rot(wing_back,  wing_half_W),    # outer-back (at bumper)
        _rot(wing_back,  body_half_W),    # inner-back (at bumper)
    ]
    pygame.gfxdraw.filled_polygon(surface, left_wing, fill)
    pygame.draw.polygon(surface, border, left_wing, 2)

    right_wing = [
        _rot(wing_front, -body_half_W),
        _rot(wing_front, -wing_half_W),
        _rot(wing_back,  -wing_half_W),
        _rot(wing_back,  -body_half_W),
    ]
    pygame.gfxdraw.filled_polygon(surface, right_wing, fill)
    pygame.draw.polygon(surface, border, right_wing, 2)


def draw_robot(surface, robot: Robot, layout: FieldLayout) -> None:
    """Robot rendered as a rotated rectangle, 18" long × 14" wide.
    theta=0 means the robot's forward direction points along +x (field-right)."""
    cx, cy = layout.ft_to_px(robot.x, robot.y)
    color = _color_for_alliance(robot.alliance)
    color_deep = C.RED_DEEP if robot.alliance == Alliance.RED else C.BLUE_DEEP

    L = layout.ft_len_to_px(ROBOT_LENGTH_FT)
    W = layout.ft_len_to_px(ROBOT_WIDTH_FT)
    half_L = L // 2
    half_W = W // 2

    # Corner offsets in body frame: forward = +x, left = +y (in field coords)
    corners_body = [
        ( half_L,  half_W),   # front-left
        ( half_L, -half_W),   # front-right
        (-half_L, -half_W),   # back-right
        (-half_L,  half_W),   # back-left
    ]
    cos_t = math.cos(robot.theta)
    sin_t = math.sin(robot.theta)

    def _rotate_to_screen(bx: int, by: int) -> tuple[int, int]:
        # Rotate by theta in field frame, then map to screen (flip y)
        wx = bx * cos_t - by * sin_t
        wy = bx * sin_t + by * cos_t
        return (int(cx + wx), int(cy - wy))

    corners_px = [_rotate_to_screen(*c) for c in corners_body]

    # Soft shadow polygon
    shadow_pts = [(p[0] + 3, p[1] + 4) for p in corners_px]
    pygame.gfxdraw.filled_polygon(surface, shadow_pts, (0, 0, 0, 60))

    pygame.gfxdraw.filled_polygon(surface, corners_px, color)
    pygame.draw.polygon(surface, color_deep, corners_px, 2)

    # Heading arrow from center to front edge
    front_mid = _rotate_to_screen(half_L, 0)
    pygame.draw.line(surface, (255, 255, 255), (cx, cy), front_mid, 3)

    # Front bumper highlight (small white line along the front edge)
    front_left = corners_px[0]
    front_right = corners_px[1]
    pygame.draw.line(surface, (255, 255, 255), front_left, front_right, 2)

    # Robot id label at center
    font = pygame.font.SysFont("Segoe UI", 10, bold=True)
    label = font.render(f"R{robot.id}", True, (255, 255, 255))
    surface.blit(label, (cx - label.get_width() // 2,
                          cy - label.get_height() // 2))

    # Midfield indicator
    if robot.in_midfield:
        _aa_filled_circle(surface, cx, cy - half_L - 12, 4, C.OK_GREEN)
        _aa_ring(surface, cx, cy - half_L - 12, 4, C.GOAL_BORDER, thickness=1)


# -- World composition --------------------------------------------------------


STACK_ITEM_Y_STEP = 22   # vertical pixels between stack items growing up from the goal


def _score_text_for_pin_half(half_color: PinColor, goal: Goal,
                              world: World) -> tuple[str, tuple[int, int, int]]:
    """Return (label_text, label_color) for a visible pin half."""
    from core.scoring import _toggle_yellow_owner

    if half_color == PinColor.RED:
        return ("+5", C.RED)
    if half_color == PinColor.BLUE:
        return ("+5", C.BLUE)
    # Yellow — depends on quadrant toggle / midfield majority
    if goal.in_midfield:
        owner = world.midfield_yellow_owner()
    else:
        owner = _toggle_yellow_owner(world.toggle_by_quadrant(goal.quadrant).state)
    if owner == Alliance.RED:
        return ("+10", C.RED)
    if owner == Alliance.BLUE:
        return ("+10", C.BLUE)
    return ("+0", C.TEXT_DIM)


def _draw_field_score_label(surface, x_tip: int, y_mid: int,
                             text: str, color, *, side: str = "left") -> None:
    """Small score label with a tiny arrow pointing at (x_tip, y_mid).
    side='left' draws text to the left of the tip; side='right' draws to the right."""
    font = pygame.font.SysFont("Segoe UI", 10, bold=True)
    img = font.render(text, True, color)
    if side == "left":
        surface.blit(img, (x_tip - img.get_width() - 6,
                           y_mid - img.get_height() // 2))
        pygame.draw.line(surface, color, (x_tip - 4, y_mid), (x_tip - 1, y_mid), 1)
        pygame.draw.polygon(surface, color, [
            (x_tip, y_mid),
            (x_tip - 3, y_mid - 2),
            (x_tip - 3, y_mid + 2),
        ])
    else:
        surface.blit(img, (x_tip + 6, y_mid - img.get_height() // 2))
        pygame.draw.line(surface, color, (x_tip + 1, y_mid), (x_tip + 4, y_mid), 1)
        pygame.draw.polygon(surface, color, [
            (x_tip, y_mid),
            (x_tip + 3, y_mid - 2),
            (x_tip + 3, y_mid + 2),
        ])


def draw_stack_in_goal(surface, world: World, goal: Goal, layout: FieldLayout,
                       hovered_pin: Optional[int] = None,
                       hovered_cup: Optional[int] = None) -> list[tuple[str, int, pygame.Rect]]:
    """Render the stack growing UP from a goal as alternating hex/square icons,
    with small score labels next to each visible pin half.

    Returns hit-test rects [(kind, id, Rect), ...] for click handling.
    """
    gx, gy = layout.ft_to_px(goal.x, goal.y)
    stack = world.stack_in_goal(goal.id)
    visibility = world.visible_halves_in_goal(goal.id)
    hits: list[tuple[str, int, pygame.Rect]] = []

    if not stack:
        return hits

    # Starting position for the bottom-most item: just above the goal center
    base_y = gy - 18

    for i, (kind, obj) in enumerate(stack):
        item_y = base_y - i * STACK_ITEM_Y_STEP
        if kind == "pin":
            visible = visibility.get(obj.id, set())
            top_v = (0 in visible)
            bot_v = (1 in visible)
            draw_pin_icon(surface, gx, item_y, obj,
                          top_visible=top_v, bot_visible=bot_v,
                          hovered=(obj.id == hovered_pin))
            hits.append(("pin", obj.id, pygame.Rect(
                gx - PIN_ICON_W // 2, item_y - PIN_ICON_H // 2,
                PIN_ICON_W, PIN_ICON_H)))

            # Score labels for visible halves: top-half label on LEFT,
            # bottom-half label on RIGHT (so they don't collide vertically
            # with adjacent stack items).
            if top_v:
                txt, col = _score_text_for_pin_half(obj.half_a_color, goal, world)
                _draw_field_score_label(
                    surface,
                    x_tip=gx - PIN_ICON_W // 2 - 2,
                    y_mid=item_y - PIN_ICON_H // 4,
                    text=txt, color=col, side="left",
                )
            if bot_v:
                txt, col = _score_text_for_pin_half(obj.half_b_color, goal, world)
                _draw_field_score_label(
                    surface,
                    x_tip=gx + PIN_ICON_W // 2 + 2,
                    y_mid=item_y + PIN_ICON_H // 4,
                    text=txt, color=col, side="right",
                )
        else:
            draw_cup_icon(surface, gx, item_y, obj,
                          hovered=(obj.id == hovered_cup))
            hits.append(("cup", obj.id, pygame.Rect(
                gx - CUP_ICON_W // 2, item_y - CUP_ICON_H // 2,
                CUP_ICON_W, CUP_ICON_H)))

    return hits


def draw_world(surface, world: World, layout: FieldLayout,
               hovered_goal: Optional[int] = None,
               hovered_toggle: Optional[int] = None,
               hovered_pin: Optional[int] = None,
               hovered_cup: Optional[int] = None,
               wings_extended_robot_id: Optional[int] = None) -> dict:
    """Draw the full field in correct z-order. Returns dict of hit-test rects.

    All drawing is clipped to the field rect so that, when the layout is in
    a zoomed (sub-viewport) mode, off-screen field elements don't bleed into
    the side panel area. The previous clip is restored on return."""
    prev_clip = surface.get_clip()
    field_rect = pygame.Rect(layout.origin_x, layout.origin_y,
                              layout.size_px, layout.size_px)
    surface.set_clip(field_rect)
    try:
        return _draw_world_clipped(
            surface, world, layout, hovered_goal, hovered_toggle,
            hovered_pin, hovered_cup, wings_extended_robot_id,
        )
    finally:
        surface.set_clip(prev_clip)


def _draw_world_clipped(surface, world, layout,
                         hovered_goal, hovered_toggle,
                         hovered_pin, hovered_cup,
                         wings_extended_robot_id) -> dict:
    draw_field_background(surface, layout)
    draw_loaders(surface, layout)

    # Goals (background scoring zones)
    for goal in world.goals:
        draw_goal(surface, goal, layout, hovered=(goal.id == hovered_goal))

    # Per-goal stack icons growing UP from each goal
    stack_hits: list[tuple[int, str, int, pygame.Rect]] = []
    for goal in world.goals:
        hits = draw_stack_in_goal(surface, world, goal, layout,
                                  hovered_pin=hovered_pin,
                                  hovered_cup=hovered_cup)
        for h in hits:
            stack_hits.append((goal.id, *h))

    # Loose pins/cups (not in any goal, not nested in any cup) — these are
    # what the robot can pick up. Both halves drawn bright so the user can see
    # the pin's two colors (scoring doesn't apply to loose pins anyway).
    # CUPS DRAWN FIRST, then pins on top — so a pin sharing a position with a
    # cup (the "pin on cup" starting configurations) appears nested inside.
    held_pin_ids = {r.holding_pin_id for r in world.robots if r.holding_pin_id is not None}
    held_cup_ids = {r.holding_cup_id for r in world.robots if r.holding_cup_id is not None}
    for cup in world.cups:
        if cup.in_goal is None and cup.on_pin is None and cup.id not in held_cup_ids:
            px, py = layout.ft_to_px(cup.x, cup.y)
            draw_cup_icon(surface, px, py, cup,
                          hovered=(cup.id == hovered_cup))
    for pin in world.pins:
        if pin.in_goal is None and pin.in_cup is None and pin.id not in held_pin_ids:
            px, py = layout.ft_to_px(pin.x, pin.y)
            draw_pin_icon(surface, px, py, pin,
                          top_visible=True, bot_visible=True,
                          hovered=(pin.id == hovered_pin))

    # Toggles
    for toggle in world.toggles:
        draw_toggle(surface, toggle, layout, hovered=(toggle.id == hovered_toggle))

    # Robots on top
    for robot in world.robots:
        # Wings (if deployed for this robot) drawn FIRST so the chassis on
        # top hides the inner edge where wing meets body.
        if wings_extended_robot_id is not None and robot.id == wings_extended_robot_id:
            draw_wings(surface, robot, layout)
        draw_robot(surface, robot, layout)
        # Held items render at the robot's front
        held_x = robot.x + math.cos(robot.theta) * 0.55
        held_y = robot.y + math.sin(robot.theta) * 0.55
        hpx, hpy = layout.ft_to_px(held_x, held_y)
        if robot.holding_pin_id is not None:
            pin = next((p for p in world.pins if p.id == robot.holding_pin_id), None)
            if pin is not None:
                draw_pin_icon(surface, hpx, hpy, pin,
                              top_visible=True, bot_visible=False)
        if robot.holding_cup_id is not None:
            cup = next((c for c in world.cups if c.id == robot.holding_cup_id), None)
            if cup is not None:
                draw_cup_icon(surface, hpx, hpy + 14, cup)

    return {"stack_hits": stack_hits}
