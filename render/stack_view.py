"""Side-view stack rendering for a single focused goal.

Visual model (matching the user's reference geometry):
- Pins are hexagons with two colored halves (top color + bottom color)
- Cups are circles with one gray (opaque) half and one clear (transparent) half
- Drawing layers:
    1. Pin hexagons (drawn fully colored)
    2. Cups OVER pin junctions:
        - GRAY half = solid fill that visually BLOCKS the pin half behind it
        - CLEAR half = outline only (transparent — pin half shows through)
    3. Goal base drawn ON TOP at the bottom (visually swallows the bottom
       half of the bottommost pin, matching "bottom of pin in goal is covered")
    4. Score arrows for VISIBLE halves only (driven by World.visible_halves_in_goal())
"""

from __future__ import annotations

import math

import pygame
import pygame.gfxdraw

from core.state import (
    Alliance, Cup, Goal, GoalType, Pin, PinColor, ToggleState, World,
)
from core.scoring import _toggle_yellow_owner

from . import colors as C
from . import hud as Hud


# -- Layout constants ---------------------------------------------------------

PIN_W            = 60          # hexagon flat-to-flat width
PIN_H            = 70          # hexagon point-to-point height
PIN_SPACING_Y    = PIN_H       # vertical distance between pin centers
                               # (= pin height -> adjacent pin edges touch)
CUP_R            = 34          # cup radius (slightly larger than pin half-height)

GOAL_BASE_WIDTH  = 200
GOAL_BASE_HEIGHT = 92


# -- Colors -------------------------------------------------------------------


def _pin_color(c: PinColor) -> tuple[int, int, int]:
    if c == PinColor.RED:    return C.RED
    if c == PinColor.BLUE:   return C.BLUE
    return C.YELLOW


def _score_for_visible_color(
    color: PinColor, goal: Goal, world: World,
) -> tuple[int, tuple[int, int, int], str]:
    """Return (points, label_color, label_text) for one visible pin half."""
    if color == PinColor.RED:
        return (5, C.RED, "+5")
    if color == PinColor.BLUE:
        return (5, C.BLUE, "+5")
    # Yellow
    if goal.in_midfield:
        owner = world.midfield_yellow_owner()
    else:
        owner = _toggle_yellow_owner(world.toggle_by_quadrant(goal.quadrant).state)
    if owner == Alliance.RED:
        return (10, C.RED, "+10")
    if owner == Alliance.BLUE:
        return (10, C.BLUE, "+10")
    return (0, C.TEXT_DIM, "+0")


# -- Pin (hexagon) ------------------------------------------------------------


def draw_pin_hex(surface, cx: int, cy: int,
                 top_color: tuple[int, int, int],
                 bot_color: tuple[int, int, int]) -> None:
    """Draw a hexagonal pin centered at (cx, cy). Top half = top_color,
    bottom half = bot_color, with a horizontal line at the midline."""
    W = PIN_W
    half_h = PIN_H // 2          # 35
    quarter_h = PIN_H // 4       # 17 — the slanted segment height

    # Top half pentagon (slanted top + flat bottom at midline)
    top_pts = [
        (cx - W // 2, cy - quarter_h),
        (cx,          cy - half_h),
        (cx + W // 2, cy - quarter_h),
        (cx + W // 2, cy),
        (cx - W // 2, cy),
    ]
    pygame.gfxdraw.filled_polygon(surface, top_pts, top_color)

    # Bottom half pentagon
    bot_pts = [
        (cx - W // 2, cy),
        (cx + W // 2, cy),
        (cx + W // 2, cy + quarter_h),
        (cx,          cy + half_h),
        (cx - W // 2, cy + quarter_h),
    ]
    pygame.gfxdraw.filled_polygon(surface, bot_pts, bot_color)

    # Hexagon outline (thick)
    hex_pts = [
        (cx - W // 2, cy - quarter_h),
        (cx,          cy - half_h),
        (cx + W // 2, cy - quarter_h),
        (cx + W // 2, cy + quarter_h),
        (cx,          cy + half_h),
        (cx - W // 2, cy + quarter_h),
    ]
    pygame.draw.polygon(surface, C.GOAL_BORDER, hex_pts, 3)
    # Middle dividing line
    pygame.draw.line(surface, C.GOAL_BORDER,
                     (cx - W // 2, cy), (cx + W // 2, cy), 3)


# -- Cup (circle with gray + clear halves) ------------------------------------


def _semicircle_points(cx: int, cy: int, r: int,
                       start_deg: int, end_deg: int) -> list[tuple[int, int]]:
    """Return polygon points approximating a semicircle arc from start_deg to
    end_deg, then closing across the diameter."""
    pts = []
    for angle in range(start_deg, end_deg + 1, 4):
        x = cx + int(r * math.cos(math.radians(angle)))
        y = cy + int(r * math.sin(math.radians(angle)))
        pts.append((x, y))
    return pts


def draw_cup_circle(surface, cx: int, cy: int, clear_face_up: bool) -> None:
    """Cup as a circle. Gray half is OPAQUE (visually blocks pin behind it);
    clear half is outline-only (pin behind shows through).

    clear_face_up=True  -> CLEAR on top, GRAY on bottom
    clear_face_up=False -> GRAY on top, CLEAR on bottom

    In screen coords y increases downward:
        angles 180..360 sweep the TOP half of the circle (y < cy)
        angles 0..180   sweep the BOTTOM half (y > cy)
    """
    R = CUP_R

    if clear_face_up:
        gray_pts = _semicircle_points(cx, cy, R, 0, 180)        # bottom half = gray
        gray_label_y = cy + R // 2
        clear_label_y = cy - R // 2
    else:
        gray_pts = _semicircle_points(cx, cy, R, 180, 360)      # top half = gray
        gray_label_y = cy - R // 2
        clear_label_y = cy + R // 2

    # 1. Solid gray fill for the opaque half (this BLOCKS the pin behind)
    pygame.gfxdraw.filled_polygon(surface, gray_pts, C.CUP_OPAQUE)
    pygame.gfxdraw.aapolygon(surface, gray_pts, (10, 10, 14))

    # 2. The CLEAR half: just the outline circle — no fill so the pin shows
    #    through. We add a very faint glass-tint and a couple horizontal
    #    "glass lines" so the user can tell there's a cup boundary there.
    clear_overlay = pygame.Surface((R * 2 + 4, R + 2), pygame.SRCALPHA)
    if clear_face_up:
        # Tint a thin band right above the midline (the bottom edge of the clear half)
        pygame.draw.line(clear_overlay, (180, 210, 230, 80),
                         (4, R - 3), (R * 2, R - 3), 1)
        pygame.draw.line(clear_overlay, (180, 210, 230, 80),
                         (8, R - 7), (R * 2 - 4, R - 7), 1)
        surface.blit(clear_overlay, (cx - R - 2, cy - R))
    else:
        pygame.draw.line(clear_overlay, (180, 210, 230, 80),
                         (4, 3), (R * 2, 3), 1)
        pygame.draw.line(clear_overlay, (180, 210, 230, 80),
                         (8, 7), (R * 2 - 4, 7), 1)
        surface.blit(clear_overlay, (cx - R - 2, cy))

    # 3. Crisp dark outline around the whole cup
    pygame.gfxdraw.aacircle(surface, cx, cy, R, (10, 10, 14))
    pygame.gfxdraw.aacircle(surface, cx, cy, R - 1, (10, 10, 14))

    # 4. Mid divider line — bold, splits the two halves
    pygame.draw.line(surface, (10, 10, 14),
                     (cx - R + 1, cy), (cx + R - 1, cy), 3)

    # 5. Tiny labels in each half so orientation is unambiguous
    label_font = Hud.font("Segoe UI", 9, bold=True)
    gray_img = label_font.render("GRAY", True, (235, 235, 235))
    clear_img = label_font.render("CLEAR", True, (40, 70, 100))
    surface.blit(gray_img, (cx - gray_img.get_width() // 2,
                            gray_label_y - gray_img.get_height() // 2))
    surface.blit(clear_img, (cx - clear_img.get_width() // 2,
                             clear_label_y - clear_img.get_height() // 2))


# -- Goal base ----------------------------------------------------------------


def draw_goal_base(surface, cx: int, y_top: int, goal: Goal) -> None:
    """Goal block at the bottom of the stack view. Drawn OVER pins so the
    bottom pin's lower half appears 'inside' the goal."""
    W = GOAL_BASE_WIDTH
    H = GOAL_BASE_HEIGHT
    rect = pygame.Rect(cx - W // 2, y_top, W, H)

    # Drop shadow
    shadow = pygame.Surface((rect.w + 16, rect.h + 16), pygame.SRCALPHA)
    pygame.draw.rect(shadow, (0, 0, 0, 70),
                     (8, 10, rect.w, rect.h), border_radius=14)
    surface.blit(shadow, (rect.left - 8, rect.top - 8))

    # Border color encodes alliance
    if goal.alliance == Alliance.RED:
        border = C.RED
        border_thickness = 5
    elif goal.alliance == Alliance.BLUE:
        border = C.BLUE
        border_thickness = 5
    else:
        border = C.GOAL_GREEN_DEEP
        border_thickness = 3

    pygame.draw.rect(surface, C.GOAL_GREEN, rect, border_radius=14)
    pygame.draw.rect(surface, C.GOAL_GREEN_DIM,
                     rect.inflate(-16, -16), border_radius=10)
    pygame.draw.rect(surface, border, rect, border_thickness, border_radius=14)

    type_label = {
        GoalType.ALLIANCE: f"GOAL G{goal.id} — {goal.alliance.value.upper()} ALLIANCE",
        GoalType.SHORT: f"GOAL G{goal.id} — neutral short",
        GoalType.TALL:  f"GOAL G{goal.id} — TALL (MIDFIELD)",
    }[goal.type]
    label = Hud.font("Segoe UI", 14, bold=True).render(
        type_label, True, C.TEXT_PRIMARY)
    surface.blit(label, (cx - label.get_width() // 2,
                         y_top + H // 2 - label.get_height() // 2))


# -- Score arrows -------------------------------------------------------------


def _draw_score_arrow(surface, x_text_right: int, y_mid: int,
                      x_tip: int, text: str, color) -> None:
    font = Hud.font("Segoe UI", 18, bold=True)
    img = font.render(text, True, color)
    surface.blit(img, (x_text_right - img.get_width() - 8,
                       y_mid - img.get_height() // 2))
    pygame.draw.line(surface, color,
                     (x_text_right + 2, y_mid), (x_tip - 6, y_mid), 2)
    pygame.draw.polygon(surface, color, [
        (x_tip, y_mid),
        (x_tip - 8, y_mid - 5),
        (x_tip - 8, y_mid + 5),
    ])


# -- Main entry point ---------------------------------------------------------


def draw_focused_goal(surface, area: pygame.Rect, world: World,
                      goal_id: int, hovered_index: int | None = None) -> dict:
    """Render one goal as a side-view tower.

    Drawing order: pins (full color) -> cups (gray halves overlay) -> goal
    base (overlays bottom of bottom pin) -> arrows for visible halves only.

    Returns hit-test info for click handling."""
    goal = world.goal_by_id(goal_id)
    stack = world.stack_in_goal(goal_id)
    visibility = world.visible_halves_in_goal(goal_id)

    cx = area.centerx

    # Goal sits at the bottom of the area
    goal_y_top = area.bottom - GOAL_BASE_HEIGHT - 12
    # Bottom pin center is exactly at the goal top -> bottom half (PIN_H/2)
    # extends below goal_y_top and will be covered by the goal rectangle later.
    bottom_pin_center_y = goal_y_top

    # Compute positions for every stack element
    pin_positions: list[tuple[Pin, int, int]] = []   # (pin, x, y)
    cup_positions: list[tuple[Cup, int, int]] = []   # (cup, x, y)
    hit_rects: list[tuple[str, int, pygame.Rect]] = []

    pin_idx = 0
    for kind, obj in stack:
        if kind == "pin":
            y = bottom_pin_center_y - pin_idx * PIN_SPACING_Y
            pin_positions.append((obj, cx, y))
            hit_rects.append(("pin", obj.id,
                              pygame.Rect(cx - PIN_W // 2, y - PIN_H // 2,
                                          PIN_W, PIN_H)))
            pin_idx += 1
        else:
            # Cup is at the midpoint of the previous pin's center and the
            # next pin's center (= PIN_SPACING_Y / 2 above the previous pin).
            prev_pin_y = pin_positions[pin_idx - 1][2]
            cup_y = prev_pin_y - PIN_SPACING_Y // 2
            cup_positions.append((obj, cx, cup_y))
            hit_rects.append(("cup", obj.id,
                              pygame.Rect(cx - CUP_R, cup_y - CUP_R,
                                          CUP_R * 2, CUP_R * 2)))

    # 1. Draw pins fully colored
    for pin, x, y in pin_positions:
        draw_pin_hex(surface, x, y,
                     _pin_color(pin.half_a_color),
                     _pin_color(pin.half_b_color))

    # 2. Draw cups on top of pin junctions (gray half hides pin underneath,
    #    clear half is outline-only so pin shows through)
    for cup, x, y in cup_positions:
        draw_cup_circle(surface, x, y, cup.clear_face_up)

    # 3. Draw goal base ON TOP (covers bottom half of bottom pin)
    draw_goal_base(surface, cx, goal_y_top, goal)

    # 4. Draw score arrows for visible halves only
    arrow_x_text_right = cx - PIN_W // 2 - 18
    arrow_x_tip = cx - PIN_W // 2 - 4
    for pin, x, y in pin_positions:
        visible = visibility.get(pin.id, set())
        if 0 in visible:
            color = pin.half_a_color
            _, col, txt = _score_for_visible_color(color, goal, world)
            arrow_y = y - PIN_H // 4
            _draw_score_arrow(surface, arrow_x_text_right, arrow_y,
                              arrow_x_tip, txt, col)
        if 1 in visible:
            color = pin.half_b_color
            _, col, txt = _score_for_visible_color(color, goal, world)
            arrow_y = y + PIN_H // 4
            _draw_score_arrow(surface, arrow_x_text_right, arrow_y,
                              arrow_x_tip, txt, col)

    # "+ Click to add next" affordance above the stack
    if pin_positions:
        top_y = pin_positions[-1][2] - PIN_H // 2
    else:
        top_y = goal_y_top
    add_zone = pygame.Rect(cx - 110, top_y - 56, 220, 44)
    next_kind = world.next_placeable_kind(goal_id)
    pygame.draw.rect(surface, C.PANEL_HIGHLIGHT, add_zone, border_radius=10)
    pygame.draw.rect(surface, C.PANEL_BORDER_FOCUSED, add_zone, 2, border_radius=10)
    label = Hud.font("Segoe UI", 12, bold=True).render(
        f"+ Click to add next: {next_kind.upper()}",
        True, C.PANEL_BORDER_FOCUSED)
    surface.blit(label, (add_zone.centerx - label.get_width() // 2,
                         add_zone.centery - label.get_height() // 2))

    return {
        "goal": pygame.Rect(cx - GOAL_BASE_WIDTH // 2, goal_y_top,
                            GOAL_BASE_WIDTH, GOAL_BASE_HEIGHT),
        "stack": hit_rects,
        "add_zone": add_zone,
    }
