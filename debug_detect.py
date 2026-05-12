# debug_detect.py
import cv2
from src.detector import PersonDetector

detector = PersonDetector()
cap = cv2.VideoCapture("testvideo2.mp4")

frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_num += 1

    detections = detector.detect(frame)
    if detections:
        print(f"Frame {frame_num}: {len(detections)} detection(s)")
        for (x1, y1, x2, y2, conf) in detections:
            print(f"  → box=({x1},{y1},{x2},{y2})  conf={conf:.2f}")

cap.release()
print("Done.")