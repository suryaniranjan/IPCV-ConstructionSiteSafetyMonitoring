import cv2
import time
import argparse
import os

from src.detector import PersonDetector
from src.zone_utils import DangerZone
from src.alert import AlertManager
from src.overlay import UIOverlay


# ─────────────────────────────────────────────────────────────────────
#  Polygon danger zone coordinates (x, y)
#
#  VIDEO: testvideo2.mp4  |  Resolution: 1920 x 1080
#
#  This zone is calibrated to cover TWO worker positions:
#
#  1. PIT WORKER (deep in trench, small box in upper frame):
#       Detected as small box ~(495-635, 240-450)
#       Body center cy ≈ 340–362  → zone top must be above 260
#
#  2. FOREGROUND WORKER (walking past pit, large box):
#       Body center cy ≈ 750–870  → zone bottom must reach ~900
#       cx ≈ 220–650 when passing through danger area
#
#  Zone check uses BODY CENTER (cx, cy = mid of bounding box).
# ─────────────────────────────────────────────────────────────────────
DEFAULT_ZONE_POINTS = [
    (220, 240),   # top-left     — covers pit worker top
    (720, 240),   # top-right    — covers pit worker top
    (800, 900),   # bottom-right — covers foreground worker body center
    (180, 900),   # bottom-left  — covers foreground worker body center
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Construction Site Safety Monitor"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: '0' for webcam, or path to video file",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        help="Minimum detection confidence (0.0 – 1.0)",
    )
    parser.add_argument(
        "--save-violations",
        action="store_true",
        default=True,
        help="Save snapshot image on each violation",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="logs/violations.log",
        help="Path to the violation log file",
    )
    return parser.parse_args()


def open_video_source(source_str: str):
    """Open webcam (int) or video file (str)."""
    source = int(source_str) if source_str.isdigit() else source_str
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source_str}")
    return cap


def main():
    args = parse_args()

    # ── Initialise subsystems ──────────────────────────────────────────
    detector = PersonDetector(confidence_threshold=args.confidence)
    zone     = DangerZone(points=DEFAULT_ZONE_POINTS)
    alert    = AlertManager(
        log_path=args.log_file,
        save_violations=args.save_violations,
    )
    overlay  = UIOverlay()

    cap = open_video_source(args.source)

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Video resolution : {frame_w} x {frame_h}")
    print(f"[INFO] Zone points      : {DEFAULT_ZONE_POINTS}")
    print("[INFO] Starting safety monitor. Press 'q' to quit.")

    frame_count   = 0
    fps_timer     = time.time()
    display_fps   = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video stream.")
            break

        frame_count += 1

        # ── FPS calculation (every 30 frames) ─────────────────────────
        if frame_count % 30 == 0:
            elapsed     = time.time() - fps_timer
            display_fps = 30 / elapsed if elapsed > 0 else 0
            fps_timer   = time.time()

        # ── Detection ─────────────────────────────────────────────────
        detections = detector.detect(frame)

        # ── Zone check & labelling ────────────────────────────────────
        total_persons    = len(detections)
        violation_count  = 0
        violation_exists = False

        for (x1, y1, x2, y2, conf) in detections:
            # Body center — robust for both pit workers (small boxes,
            # high in frame) and foreground workers (tall boxes).
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            in_zone = zone.contains_point(cx, cy)

            if in_zone:
                violation_count  += 1
                violation_exists  = True
                color = (0, 0, 255)
                label = f"DANGER  {conf:.0%}"
            else:
                color = (0, 220, 0)
                label = f"Safe  {conf:.0%}"

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Debug dot — shows exact point used for zone check
            cv2.circle(frame, (cx, cy), 5, color, -1)

            # Label
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(
                frame, label,
                (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA,
            )

        # ── Draw danger zone polygon ───────────────────────────────────
        zone.draw(frame, violation_exists)

        # ── Trigger alerts ─────────────────────────────────────────────
        if violation_exists:
            alert.trigger(frame, violation_count)

        # ── HUD overlay ───────────────────────────────────────────────
        overlay.draw(
            frame,
            fps=display_fps,
            total_persons=total_persons,
            violations=violation_count,
            zone_points=DEFAULT_ZONE_POINTS,
        )

        cv2.imshow("Construction Site Safety Monitor", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("[INFO] Quit signal received.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Session ended. Violations logged to: {args.log_file}")


if __name__ == "__main__":
    main()