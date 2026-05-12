# zone_picker.py — run this once on your video to get new coordinates
import cv2

points = []
video_path = "testvideo2.mp4"  # change this

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point added: ({x}, {y})  — total: {points}")
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Frame", frame)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if ret:
    cv2.imshow("Frame", frame)
    cv2.setMouseCallback("Frame", click_event)
    print("Click to define zone polygon corners. Press 'q' when done.")
    while True:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
    print("\nFinal polygon coordinates:")
    print(points)