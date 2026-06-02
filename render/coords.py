"""Coordinate transforms + responsive window layout.

Field convention:
- Origin (0, 0) at field center
- x in [-6, +6], positive = right (red's side)
- y in [-6, +6], positive = up

Screen y is flipped (positive = down).
"""

from __future__ import annotations

from dataclasses import dataclass


FIELD_FT = 12.0


@dataclass(frozen=True)
class FieldLayout:
    """Where the field is positioned in the window.

    Viewport — the rectangle of FIELD COORDINATES (feet) that maps to the
    on-screen field rect. Defaults to the full field [-6, +6] in both axes.
    For zoom modes (e.g., the toggle-duel quadrant view) pass a smaller
    square viewport. Must be square — x-span must equal y-span — so the
    fixed-size pixel field doesn't stretch."""
    origin_x: int
    origin_y: int
    size_px: int
    view_x_min: float = -6.0
    view_x_max: float = +6.0
    view_y_min: float = -6.0
    view_y_max: float = +6.0

    @property
    def view_span_ft(self) -> float:
        return self.view_x_max - self.view_x_min

    @property
    def px_per_ft(self) -> float:
        return self.size_px / self.view_span_ft

    @property
    def center_px(self) -> tuple[int, int]:
        return (self.origin_x + self.size_px // 2,
                self.origin_y + self.size_px // 2)

    @property
    def right(self) -> int:
        return self.origin_x + self.size_px

    @property
    def bottom(self) -> int:
        return self.origin_y + self.size_px

    def ft_to_px(self, x: float, y: float) -> tuple[int, int]:
        ppf = self.px_per_ft
        return (int(self.origin_x + (x - self.view_x_min) * ppf),
                int(self.origin_y + (self.view_y_max - y) * ppf))

    def px_to_ft(self, px: int, py: int) -> tuple[float, float]:
        ppf = self.px_per_ft
        return ((px - self.origin_x) / ppf + self.view_x_min,
                self.view_y_max - (py - self.origin_y) / ppf)

    def ft_len_to_px(self, feet: float) -> int:
        return int(feet * self.px_per_ft)

    def contains_px(self, px: int, py: int) -> bool:
        return (self.origin_x <= px < self.origin_x + self.size_px and
                self.origin_y <= py < self.origin_y + self.size_px)

    def with_viewport(self,
                      x_min: float, x_max: float,
                      y_min: float, y_max: float) -> "FieldLayout":
        """Return a copy with the viewport replaced."""
        return FieldLayout(
            origin_x=self.origin_x, origin_y=self.origin_y, size_px=self.size_px,
            view_x_min=x_min, view_x_max=x_max,
            view_y_min=y_min, view_y_max=y_max,
        )


@dataclass(frozen=True)
class WindowLayout:
    """Full window layout — recomputed on resize / fullscreen toggle."""
    win_w: int
    win_h: int
    topbar_h: int
    statusbar_h: int
    margin: int
    alliance_strip_w: int      # width of the orange alliance-station strips on either side
    field: FieldLayout
    side_x: int
    side_w: int
    red_station_rect: tuple[int, int, int, int]   # (x, y, w, h)
    blue_station_rect: tuple[int, int, int, int]

    @classmethod
    def compute(cls, win_w: int, win_h: int,
                topbar_h: int = 120, statusbar_h: int = 36,
                margin: int = 18) -> "WindowLayout":
        # We want the right-side HUD to be at least 360 px wide so labels fit.
        min_side_w = 360
        alliance_strip_w = 0   # alliance stations removed per user request

        # Available area (no overhang now that loaders are fully inside)
        h_avail = win_h - topbar_h - statusbar_h - 2 * margin
        w_avail = win_w - 3 * margin - min_side_w

        field_size = max(300, min(h_avail, w_avail))

        field_x = margin
        field_y = topbar_h + margin + max(0, (h_avail - field_size) // 2)

        red_station_rect = (0, 0, 0, 0)
        blue_station_rect = (0, 0, 0, 0)

        side_x = field_x + field_size + margin
        side_w = win_w - side_x - margin

        return cls(
            win_w=win_w, win_h=win_h,
            topbar_h=topbar_h, statusbar_h=statusbar_h,
            margin=margin, alliance_strip_w=alliance_strip_w,
            field=FieldLayout(origin_x=field_x, origin_y=field_y, size_px=field_size),
            side_x=side_x, side_w=side_w,
            red_station_rect=red_station_rect,
            blue_station_rect=blue_station_rect,
        )

    @property
    def topbar_rect(self) -> tuple[int, int, int, int]:
        return (0, 0, self.win_w, self.topbar_h)

    @property
    def statusbar_rect(self) -> tuple[int, int, int, int]:
        return (0, self.win_h - self.statusbar_h, self.win_w, self.statusbar_h)
