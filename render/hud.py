"""HUD panels — light mode, white cards with soft borders + drop shadows."""

from __future__ import annotations

import pygame

from core.scoring import ScoreResult
from core.state import Alliance, Phase, ToggleState, World

from . import colors as C


# -- Font cache ---------------------------------------------------------------

_FONTS: dict[tuple[str, int, bool], pygame.font.Font] = {}


def font(name: str = "Segoe UI", size: int = 14, bold: bool = False) -> pygame.font.Font:
    key = (name, size, bold)
    if key not in _FONTS:
        _FONTS[key] = pygame.font.SysFont(name, size, bold=bold)
    return _FONTS[key]


# -- Primitives ---------------------------------------------------------------


def _draw_panel(surface, rect: pygame.Rect, *, title: str = "") -> pygame.Rect:
    """White panel with rounded corners + subtle shadow + optional title."""
    # Drop shadow
    shadow = pygame.Surface((rect.width + 16, rect.height + 16), pygame.SRCALPHA)
    pygame.draw.rect(shadow, (0, 0, 0, 28),
                     (8, 10, rect.width, rect.height), border_radius=10)
    surface.blit(shadow, (rect.left - 8, rect.top - 8))

    pygame.draw.rect(surface, C.PANEL_BG, rect, border_radius=10)
    pygame.draw.rect(surface, C.PANEL_BORDER, rect, 1, border_radius=10)
    inner = rect.inflate(-28, -24)
    if title:
        label = font("Segoe UI", 11, bold=True).render(
            title.upper(), True, C.TEXT_DIM)
        surface.blit(label, (rect.left + 18, rect.top + 14))
        inner.top += 24
        inner.height -= 24
    return inner


def _draw_text(surface, text: str, x: int, y: int,
               color=C.TEXT_PRIMARY, size: int = 14, bold: bool = False) -> int:
    img = font("Segoe UI", size, bold=bold).render(text, True, color)
    surface.blit(img, (x, y))
    return img.get_height()


# -- Top bar ------------------------------------------------------------------


def draw_top_bar(surface, rect: pygame.Rect, world: World, score: ScoreResult) -> None:
    """Phase + clock on the left, big scores in the center, margin on the right."""
    pygame.draw.rect(surface, C.PANEL_BG, rect)
    pygame.draw.line(surface, C.PANEL_BORDER,
                     (rect.left, rect.bottom - 1), (rect.right, rect.bottom - 1), 1)

    # Phase + timer
    phase_label = {
        Phase.AUTO: "AUTONOMOUS",
        Phase.DRIVER: "DRIVER",
        Phase.ENDED: "MATCH ENDED",
    }[world.phase]
    phase_color = {
        Phase.AUTO: C.YELLOW_DEEP,
        Phase.DRIVER: C.BLUE_DEEP,
        Phase.ENDED: C.TEXT_SECONDARY,
    }[world.phase]

    _draw_text(surface, phase_label, rect.left + 22, rect.top + 18,
               color=phase_color, size=12, bold=True)

    # Driver phase: countdown 1:45 → 0:00. Other phases: show the cap.
    DRIVER_LIMIT = 105
    if world.phase == Phase.DRIVER:
        remaining = max(0, DRIVER_LIMIT - int(world.time_elapsed))
        timer_color = C.TEXT_PRIMARY if remaining > 20 else C.RED
    elif world.phase == Phase.ENDED:
        remaining = 0
        timer_color = C.TEXT_SECONDARY
    else:  # AUTO / pre-match
        remaining = DRIVER_LIMIT
        timer_color = C.TEXT_DIM
    mm, ss = divmod(remaining, 60)
    timer_img = font("Segoe UI", 36, bold=True).render(
        f"{mm}:{ss:02d}", True, timer_color)
    surface.blit(timer_img, (rect.left + 22, rect.top + 36))

    # Big scores, centered
    red_img  = font("Segoe UI", 60, bold=True).render(str(score.red), True, C.RED)
    blue_img = font("Segoe UI", 60, bold=True).render(str(score.blue), True, C.BLUE)
    dash_img = font("Segoe UI", 40, bold=True).render("–", True, C.TEXT_DIM)

    total_w = red_img.get_width() + dash_img.get_width() + blue_img.get_width() + 48
    sx = rect.centerx - total_w // 2
    sy = rect.top + 14

    surface.blit(red_img, (sx, sy))
    surface.blit(dash_img,
                 (sx + red_img.get_width() + 24, sy + 14))
    surface.blit(blue_img,
                 (sx + red_img.get_width() + dash_img.get_width() + 48, sy))

    # AWP underneath
    awp_red_text = "AWP ✓" if score.awp_red else "AWP —"
    awp_blue_text = "AWP ✓" if score.awp_blue else "AWP —"
    awp_red_color = C.OK_GREEN if score.awp_red else C.TEXT_DIM
    awp_blue_color = C.OK_GREEN if score.awp_blue else C.TEXT_DIM
    _draw_text(surface, awp_red_text,
               sx + red_img.get_width() // 2 - 22, sy + 74,
               color=awp_red_color, size=11, bold=True)
    _draw_text(surface, awp_blue_text,
               sx + red_img.get_width() + dash_img.get_width() + 48 +
               blue_img.get_width() // 2 - 22, sy + 74,
               color=awp_blue_color, size=11, bold=True)

    # Margin
    margin = score.red - score.blue
    margin_color = (C.RED if margin > 0 else C.BLUE if margin < 0 else C.TEXT_DIM)
    margin_str = f"{'+' if margin > 0 else ''}{margin}"
    _draw_text(surface, "MARGIN", rect.right - 130, rect.top + 18,
               color=C.TEXT_DIM, size=11, bold=True)
    margin_img = font("Segoe UI", 40, bold=True).render(margin_str, True, margin_color)
    surface.blit(margin_img, (rect.right - 130, rect.top + 34))


# -- Score breakdown ----------------------------------------------------------


def draw_score_breakdown(surface, rect: pygame.Rect, score: ScoreResult) -> None:
    inner = _draw_panel(surface, rect, title="Score breakdown")
    y = inner.top

    by_src_red: dict[str, int] = {}
    by_src_blue: dict[str, int] = {}
    for line in score.breakdown:
        target = by_src_red if line.alliance == Alliance.RED else by_src_blue
        target[line.source] = target.get(line.source, 0) + line.points

    sources = ["alliance_pin", "yellow_pin", "midfield", "auto_bonus"]
    labels = {
        "alliance_pin": "Alliance pins",
        "yellow_pin":   "Yellow pins",
        "midfield":     "Midfield park",
        "auto_bonus":   "Auto bonus",
    }

    _draw_text(surface, "", inner.left, y, size=12)
    _draw_text(surface, "RED", inner.right - 110, y, color=C.RED, size=11, bold=True)
    _draw_text(surface, "BLUE", inner.right - 50, y, color=C.BLUE, size=11, bold=True)
    y += 22
    pygame.draw.line(surface, C.PANEL_BORDER,
                     (inner.left, y - 4), (inner.right, y - 4), 1)

    for src in sources:
        r_val = by_src_red.get(src, 0)
        b_val = by_src_blue.get(src, 0)
        dim = (r_val == 0 and b_val == 0)
        _draw_text(surface, labels[src], inner.left, y,
                   color=(C.TEXT_DIM if dim else C.TEXT_PRIMARY), size=13)
        _draw_text(surface, str(r_val), inner.right - 110, y,
                   color=(C.RED if r_val > 0 else C.TEXT_DIM), size=13, bold=(r_val > 0))
        _draw_text(surface, str(b_val), inner.right - 50, y,
                   color=(C.BLUE if b_val > 0 else C.TEXT_DIM), size=13, bold=(b_val > 0))
        y += 22

    pygame.draw.line(surface, C.PANEL_BORDER,
                     (inner.left, y + 4), (inner.right, y + 4), 1)
    y += 12

    _draw_text(surface, "TOTAL", inner.left, y, color=C.TEXT_HIGHLIGHT, size=14, bold=True)
    _draw_text(surface, str(score.red), inner.right - 110, y,
               color=C.RED, size=18, bold=True)
    _draw_text(surface, str(score.blue), inner.right - 50, y,
               color=C.BLUE, size=18, bold=True)


# -- Toggles panel ------------------------------------------------------------


def draw_toggle_panel(surface, rect: pygame.Rect, world: World) -> None:
    inner = _draw_panel(surface, rect, title="Toggles")
    y = inner.top

    state_color = {
        ToggleState.RED:    C.RED,
        ToggleState.BLUE:   C.BLUE,
        ToggleState.YELLOW: C.YELLOW,
        ToggleState.UNSET:  C.TOGGLE_UNSET_FILL,
    }
    state_label = {
        ToggleState.RED:    "RED",
        ToggleState.BLUE:   "BLUE",
        ToggleState.YELLOW: "YELLOW",
        ToggleState.UNSET:  "UNSET",
    }
    border_color = {
        ToggleState.RED:    C.RED_DEEP,
        ToggleState.BLUE:   C.BLUE_DEEP,
        ToggleState.YELLOW: C.YELLOW_DEEP,
        ToggleState.UNSET:  C.TOGGLE_UNSET_BORDER,
    }

    for tog in sorted(world.toggles, key=lambda t: t.quadrant):
        _draw_text(surface, f"Q{tog.quadrant}", inner.left, y, size=13, bold=True)

        sx = inner.left + 56
        sw, sh = 32, 18
        rect_sw = pygame.Rect(sx, y - 2, sw, sh)
        pygame.draw.rect(surface, state_color[tog.state], rect_sw, border_radius=3)
        pygame.draw.rect(surface, border_color[tog.state], rect_sw, 1, border_radius=3)

        text_color = state_color[tog.state]
        if tog.state == ToggleState.UNSET:
            text_color = C.TEXT_SECONDARY
        elif tog.state == ToggleState.YELLOW:
            text_color = C.YELLOW_DEEP
        _draw_text(surface, state_label[tog.state], sx + sw + 14, y,
                   color=text_color, size=12, bold=True)

        y += 28


# -- Controllers panel --------------------------------------------------------


CONTROLLER_KINDS = ("player", "scripted", "nn", "none")
CONTROLLER_LABELS = {
    "player":   "PLAYER",
    "scripted": "SCRIPT",
    "nn":       "NN",
    "none":     "OFF",
}


def draw_controllers_panel(surface, rect, world, controllers: dict,
                            nn_model_name: str = "") -> list:
    """Draw a 'who's driving who' panel. For each robot, show 4 small
    pill-buttons (PLAYER / SCRIPT / NN / OFF) — the current one is
    highlighted. Returns a list of (robot_id, kind, rect) tuples so
    the click handler in App can dispatch mouse clicks to controller changes.

    If `nn_model_name` is given, also shows "NN model: <name>  (M to cycle)"
    at the bottom of the panel so the user can see which policy checkpoint
    is currently active.
    """
    inner = _draw_panel(surface, rect, title="Controllers")
    hit_zones = []
    y = inner.top
    line_h = 30

    for robot in sorted(world.robots, key=lambda r: r.id):
        # Robot label (left side)
        rcolor = C.RED if robot.alliance == Alliance.RED else C.BLUE
        _draw_text(surface, f"R{robot.id}", inner.left, y,
                    color=rcolor, size=12, bold=True)

        # 4 pill buttons across the rest of the row
        bx = inner.left + 26
        button_w, button_h = 50, 20
        gap = 4
        current = controllers.get(robot.id, "none")
        for kind in CONTROLLER_KINDS:
            br = pygame.Rect(bx, y - 1, button_w, button_h)
            selected = (kind == current)
            fill = rcolor if selected else (235, 235, 235)
            border = C.GOAL_BORDER if not selected else (0, 0, 0)
            text_color = (255, 255, 255) if selected else C.TEXT_SECONDARY
            pygame.draw.rect(surface, fill, br, border_radius=4)
            pygame.draw.rect(surface, border, br, 1, border_radius=4)
            label = CONTROLLER_LABELS[kind]
            _draw_text(surface, label, br.x + 4, br.y + 3,
                        color=text_color, size=11, bold=selected)
            hit_zones.append((robot.id, kind, br))
            bx += button_w + gap

        y += line_h

    # Footer: current NN model name, so the user can see which checkpoint
    # is being used by any robot whose controller is 'nn'. M cycles.
    if nn_model_name:
        _draw_text(surface, "NN model:", inner.left, y + 2,
                    color=C.TEXT_SECONDARY, size=11, bold=False)
        _draw_text(surface, nn_model_name, inner.left + 60, y + 2,
                    color=C.TEXT_HIGHLIGHT, size=11, bold=True)
        _draw_text(surface, "(K to cycle)", inner.left, y + 18,
                    color=C.TEXT_SECONDARY, size=10, bold=False)
    return hit_zones


# -- Help panel ---------------------------------------------------------------


HELP_LINES = [
    ("Left stick Y",   "throttle (linear)"),
    ("Right stick X",  "turn (expo curve, t adjustable)"),
    ("WASD / arrows",  "keyboard arcade drive"),
    ("] [",            "curve t (shape of partial stick)"),
    ("= −",            "turn sensitivity (caps max ω)"),
    ("E / R2",         "interact: loader / goal / pickup (grabs pin+cup set)"),
    ("T / A button",   "BACK into toggle, then cycle (Y → R → B → Y)"),
    ("X / X button",   "deploy / retract front wing (24\" push zone)"),
    ("Y",              "BACK into toggle, then set to alliance color"),
    ("B",              "toggle AI bots on/off"),
    ("F",              "AI-vs-AI toggle duel mode (zoomed quadrant)"),
    ("N",              "in duel: swap red bot ↔ neural-network policy"),
    ("K",              "cycle NN model (ai/*.pt: BC → DAGGER → PPO → …)"),
    ("Click goal",     "auto-place: pin → cup → pin → ... (SC2)"),
    ("Right-click",    "remove TOP-of-stack object"),
    ("1-4 / 5-6",      "pin type / cup orientation"),
    ("SPACE",          "advance phase: AUTO → DRIVER → ENDED"),
    ("D / R",          "load demo / reset field"),
    ("F11 / ESC",      "fullscreen / quit"),
]


def draw_help_panel(surface, rect: pygame.Rect, current_object_label: str) -> None:
    inner = _draw_panel(surface, rect, title="Controls")
    y = inner.top

    for key, desc in HELP_LINES:
        _draw_text(surface, key, inner.left, y,
                   color=C.TEXT_HIGHLIGHT, size=12, bold=True)
        _draw_text(surface, desc, inner.left + 110, y,
                   color=C.TEXT_SECONDARY, size=12)
        y += 18
        if y > inner.bottom - 28:
            break

    pygame.draw.line(surface, C.PANEL_BORDER,
                     (inner.left, y + 6), (inner.right, y + 6), 1)
    y += 14
    _draw_text(surface, "Selected:", inner.left, y,
               color=C.TEXT_DIM, size=11, bold=True)
    _draw_text(surface, current_object_label, inner.left + 90, y,
               color=C.TEXT_HIGHLIGHT, size=12, bold=True)


# -- Status bar ---------------------------------------------------------------


def draw_status_bar(surface, rect: pygame.Rect, status: str) -> None:
    pygame.draw.rect(surface, C.PANEL_BG_DARK, rect)
    pygame.draw.line(surface, C.PANEL_BORDER, (rect.left, rect.top),
                     (rect.right, rect.top), 1)
    _draw_text(surface, status, rect.left + 22, rect.top + 12,
               color=C.TEXT_SECONDARY, size=12)
