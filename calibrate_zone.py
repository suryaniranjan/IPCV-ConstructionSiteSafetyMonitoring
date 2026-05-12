"""
calibrate_zone.py
=================
Interactive tool to define your polygon danger zone by clicking on a
reference frame from your video/webcam.

Usage
-----
    python calibrate_zone.py --source 0          # webcam
    python calibrate_zone.py --source video.mp4  # file

Controls
--------
  Left-click       → add a vertex
  Right-click      → remove the last vertex
  ENTER            → finalise and print coordinates
  R                → reset all points
  Q / ESC          → quit without saving
"""

import cv2
import numpy as np
import argparse

points = []
frame_display = None


def mouse_callback(event, x, y, flags, param):
    global points, frame_display
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"  + Added point {len(points)}: ({x}, {y})")

    elif event == cv2.EVENT_RBUTTONDOWN and points:
        removed = points.pop()
        print(f"  - Removed point: {removed}")


def draw_polygon(frame):
    out = frame.copy()
    for pt in points:
        cv2.circle(out, pt, 5, (0, 200, 255), -1)
    if len(points) >= 2:
        contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(out, [contour], isClosed=len(points) >= 3, color=(0, 200, 255), thickness=2)
    if len(points) >= 3:
        overlay = out.copy()
        contour = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [contour], (0, 80, 200))
        cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)
    cv2.putText(out, f"Points: {len(points)}  |  ENTER=save  R=reset  RClick=undo",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return out


def main():
    global points
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    ret, reference_frame = cap.read()
    cap.release()

    if not ret:
        print("[ERROR] Could not read frame from source.")
        return

    cv2.namedWindow("Zone Calibration")
    cv2.setMouseCallback("Zone Calibration", mouse_callback)

    while True:
        display = draw_polygon(reference_frame)
        cv2.imshow("Zone Calibration", display)
        key = cv2.waitKey(30) & 0xFF

        if key == 13:  # ENTER
            if len(points) >= 3:
                print("\n" + "="*55)
                print("  Copy this into main.py as DEFAULT_ZONE_POINTS:")
                print("="*55)
                print("DEFAULT_ZONE_POINTS = [")
                for p in points:
                    print(f"    {p},")
                print("]")
                print("="*55 + "\n")
                break
            else:
                print("[INFO] Need at least 3 points.")

        elif key == ord("r"):
            points = []
            print("[INFO] Points reset.")

        elif key in (ord("q"), 27):
            print("[INFO] Cancelled.")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
