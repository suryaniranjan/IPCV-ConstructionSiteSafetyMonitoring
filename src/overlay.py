"""
overlay.py
==========
Draws a semi-transparent HUD panel in the top-left corner showing:
  • Live FPS
  • People detected this frame
  • Violation count this frame
  • Zone coordinates (collapsed)
  • Flashing ALERT banner when violations are active
"""

import cv2
import numpy as np
import time


class UIOverlay:
    """
    Stateless HUD renderer.  Call draw() each frame.
    """

    # Panel geometry
    PANEL_W = 270
    PANEL_H = 155
    MARGIN  = 12
    PAD     = 10

    # Colours (BGR)
    COL_BG        = (15,  15,  15)
    COL_BORDER    = (0,  200, 255)
    COL_TEXT      = (220, 220, 220)
    COL_VALUE_OK  = (0,  220,  80)
    COL_VALUE_ERR = (0,  60,  255)
    COL_TITLE     = (0,  200, 255)
    COL_ALERT_BG  = (0,  30,  200)
    COL_ALERT_TXT = (255, 255, 255)

    def __init__(self):
        self._alert_toggle_time = 0.0
        self._alert_visible     = True

    # ─────────────────────────────────────────────────────────────────
    def draw(
        self,
        frame: np.ndarray,
        fps: float,
        total_persons: int,
        violations: int,
        zone_points: list,
    ) -> None:

        h, w = frame.shape[:2]
        x0, y0 = self.MARGIN, self.MARGIN

        # ── Semi-transparent panel ────────────────────────────────────
        panel = frame[y0 : y0 + self.PANEL_H, x0 : x0 + self.PANEL_W].copy()
        cv2.rectangle(panel, (0, 0), (self.PANEL_W, self.PANEL_H), self.COL_BG, -1)
        cv2.addWeighted(panel, 0.72, frame[y0:y0+self.PANEL_H, x0:x0+self.PANEL_W], 0.28, 0,
                        frame[y0:y0+self.PANEL_H, x0:x0+self.PANEL_W])

        # Border
        cv2.rectangle(frame, (x0, y0), (x0 + self.PANEL_W, y0 + self.PANEL_H),
                      self.COL_BORDER, 1)

        # ── Title ─────────────────────────────────────────────────────
        ty = y0 + self.PAD + 14
        cv2.putText(frame, "SAFETY MONITOR",
                    (x0 + self.PAD, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, self.COL_TITLE, 1, cv2.LINE_AA)

        # Divider
        ty += 6
        cv2.line(frame, (x0 + self.PAD, ty), (x0 + self.PANEL_W - self.PAD, ty),
                 self.COL_BORDER, 1)

        # ── Rows ──────────────────────────────────────────────────────
        row_h = 22
        rows = [
            ("FPS",       f"{fps:.1f}",           self.COL_VALUE_OK),
            ("Persons",   str(total_persons),      self.COL_VALUE_OK),
            ("Violations",str(violations),
             self.COL_VALUE_ERR if violations > 0 else self.COL_VALUE_OK),
            ("Zone pts",  str(len(zone_points)),   self.COL_VALUE_OK),
        ]

        for i, (key, val, vcol) in enumerate(rows):
            ry = ty + self.PAD + (i + 1) * row_h
            cv2.putText(frame, key + ":",
                        (x0 + self.PAD, ry),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.COL_TEXT, 1, cv2.LINE_AA)
            cv2.putText(frame, val,
                        (x0 + self.PANEL_W - self.PAD - 55, ry),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, vcol, 1, cv2.LINE_AA)

        # ── Flashing ALERT banner (bottom of frame) ───────────────────
        if violations > 0:
            now = time.time()
            if now - self._alert_toggle_time > 0.45:
                self._alert_visible     = not self._alert_visible
                self._alert_toggle_time = now

            if self._alert_visible:
                banner_h = 40
                bx1, by1 = 0, h - banner_h
                bx2, by2 = w, h

                overlay2 = frame.copy()
                cv2.rectangle(overlay2, (bx1, by1), (bx2, by2), self.COL_ALERT_BG, -1)
                cv2.addWeighted(overlay2, 0.75, frame, 0.25, 0, frame)

                msg = f"  ⚠  ALERT: {violations} PERSON(S) IN DANGER ZONE  ⚠"
                (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.putText(frame, msg,
                            ((w - tw) // 2, h - 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            self.COL_ALERT_TXT, 2, cv2.LINE_AA)

        # ── Press Q hint ──────────────────────────────────────────────
        cv2.putText(frame, "Press Q to quit",
                    (w - 150, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA)
