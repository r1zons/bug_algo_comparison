"""Geometry primitives shared by maps and beetle physics."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Rectangle:
    """An axis-aligned rectangular obstacle."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def __post_init__(self) -> None:
        if self.xmin >= self.xmax or self.ymin >= self.ymax:
            raise ValueError("Rectangle minimum coordinates must be below maximum coordinates")

    def contains(self, x: float, y: float, radius: float = 0.0) -> bool:
        return (
            self.xmin - radius <= x <= self.xmax + radius
            and self.ymin - radius <= y <= self.ymax + radius
        )

    def hits_segment(self, p: np.ndarray, q: np.ndarray, radius: float = 0.0) -> bool:
        """Return whether a segment hits the radius-inflated rectangle.

        Liang-Barsky clipping checks the whole movement segment, rather than only
        its endpoint, so a long step cannot tunnel through a thin obstacle.
        """
        x0, y0 = p
        dx, dy = q - p
        lower, upper = 0.0, 1.0
        bounds = (
            (-dx, x0 - (self.xmin - radius)),
            (dx, (self.xmax + radius) - x0),
            (-dy, y0 - (self.ymin - radius)),
            (dy, (self.ymax + radius) - y0),
        )
        for direction, offset in bounds:
            if abs(direction) < 1e-12:
                if offset < 0:
                    return False
            else:
                crossing = offset / direction
                if direction < 0:
                    lower = max(lower, crossing)
                else:
                    upper = min(upper, crossing)
                if lower > upper:
                    return False
        return True
