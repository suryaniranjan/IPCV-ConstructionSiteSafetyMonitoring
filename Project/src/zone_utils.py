"""
zone_utils.py
=============
Defines one (or more) polygon danger zones and provides point-in-polygon
membership tests using cv2.pointPolygonTest.

How pointPolygonTest works
--------------------------
OpenCV's cv2.pointPolygonTest uses a ray-casting algorithm:
  - Cast an imaginary horizontal ray from the query point to the right.
  - Count how many times it crosses the polygon boundary.
  - Odd crossings  → inside
  - Even crossings → outside
When measureDist=True it returns the *signed distance* to the nearest edge:
  > 0  →  inside   (positive = how far inside)
  = 0  →  on edge
  < 0  →  outside  (negative = how far outside)
We use measureDist=False for a simple +1 / -1 test which is faster.
"""

import cv2
import numpy as np
from typing import List, Tuple


class DangerZone:
    """
    A single convex or concave polygon danger zone.

    Parameters
    ----------
    points : list of (x, y) tuples defining the polygon vertices IN ORDER
             (clockwise or counter-clockwise, both work)
    """

    def __init__(self, points: List[Tuple[int, int]]):
        if len(points) < 3:
            raise ValueError("A polygon needs at least 3 points.")
        # Convert to the shape cv2 expects: (N, 1, 2) int32
        self.points_raw = points
        self.contour    = np.array(points, dtype=np.int32).reshape((-1, 1, 2))

    # ─────────────────────────────────────────────────────────────────
    def contains_point(self, x: int, y: int) -> bool:
        """
        Return True if pixel (x, y) is inside (or on the boundary of)
        the polygon.
        """
        result = cv2.pointPolygonTest(self.contour, (float(x), float(y)), measureDist=False)
        return result >= 0   # 1 = inside, 0 = on edge, -1 = outside

    # ─────────────────────────────────────────────────────────────────
    def draw(self, frame: np.ndarray, active_violation: bool = False) -> None:
        """
        Draw the zone polygon on *frame* (in-place).

        Visual design
        -------------
        - Filled semi-transparent overlay:
            yellow  → zone is clear
            red     → at least one person is inside
        - Solid border in the same colour family.
        - Vertex markers.
        - "DANGER ZONE" label at the zone centroid.
        """
        overlay = frame.copy()

        if active_violation:
            fill_color   = (0,  30, 200)   # red (BGR)
            border_color = (0,   0, 255)
            text_color   = (0,   0, 255)
            label        = "!! DANGER ZONE !!"
        else:
            fill_color   = (0, 140, 200)   # amber (BGR)
            border_color = (0, 200, 255)
            text_color   = (0, 180, 255)
            label        = "DANGER ZONE"

        # Semi-transparent fill
        cv2.fillPoly(overlay, [self.contour], fill_color)
        alpha = 0.22
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Solid border
        cv2.polylines(frame, [self.contour], isClosed=True, color=border_color, thickness=2)

        # Vertex markers
        for pt in self.points_raw:
            cv2.circle(frame, pt, 5, border_color, -1)

        # Label at centroid
        M   = cv2.moments(self.contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = self.points_raw[0]

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        cv2.rectangle(
            frame,
            (cx - tw // 2 - 6, cy - th - 8),
            (cx + tw // 2 + 6, cy + 4),
            (0, 0, 0), -1,
        )
        cv2.putText(
            frame, label,
            (cx - tw // 2, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            text_color, 2, cv2.LINE_AA,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Multi-zone helper
# ─────────────────────────────────────────────────────────────────────────────

class MultiZone:
    """
    Manages a collection of named DangerZone objects.

    Usage
    -----
    zones = MultiZone()
    zones.add("excavation",    [(100,100),(300,100),(300,300),(100,300)])
    zones.add("material_drop", [(400,150),(600,150),(600,350),(400,350)])

    for name, in_zone in zones.check_point(cx, cy):
        if in_zone:
            print(f"Person is inside zone: {name}")

    zones.draw_all(frame, violations={"excavation"})
    """

    def __init__(self):
        self.zones: dict[str, DangerZone] = {}

    def add(self, name: str, points: List[Tuple[int, int]]) -> None:
        self.zones[name] = DangerZone(points)

    def check_point(self, x: int, y: int):
        """Yield (zone_name, is_inside) for every registered zone."""
        for name, zone in self.zones.items():
            yield name, zone.contains_point(x, y)

    def draw_all(self, frame: np.ndarray, violations: set = None) -> None:
        violations = violations or set()
        for name, zone in self.zones.items():
            zone.draw(frame, active_violation=(name in violations))
