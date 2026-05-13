"""
app.py
======
Construction Site Safety Monitor — Web Application

Features
--------
  • Login wall  (admin / admin@123)
  • Video upload via browser
  • CV detection pipeline streamed as MJPEG
  • REST API for stats, snapshots, log
  • Session-based auth (Flask sessions)

Usage
-----
    python app.py
    # Then open http://localhost:5000
"""

import cv2
import time
import threading
import os
import json
from datetime import datetime
from functools import wraps

from flask import (
    Flask, Response, jsonify, request,
    session, redirect, url_for, send_from_directory
)
from werkzeug.utils import secure_filename

from src.detector import PersonDetector
from src.zone_utils import DangerZone
from src.alert import AlertManager
from src.overlay import UIOverlay

# ── Config ─────────────────────────────────────────────────────────────────
SECRET_KEY   = "siteguard-secret-2026"
USERNAME     = "admin"
PASSWORD     = "admin@123"
UPLOAD_FOLDER = "uploads"
ALLOWED_EXT   = {"mp4", "avi", "mov", "mkv", "webm"}

DEFAULT_ZONE_POINTS = [
    (220, 240),
    (720, 240),
    (800, 900),
    (180, 900),
]

# ── App setup ───────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = SECRET_KEY
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024   # 500 MB

os.makedirs(UPLOAD_FOLDER,  exist_ok=True)
os.makedirs("violations",   exist_ok=True)
os.makedirs("logs",         exist_ok=True)

# ── Global pipeline state ───────────────────────────────────────────────────
_state = {
    "running":         False,
    "fps":             0.0,
    "total_persons":   0,
    "violations":      0,
    "session_total":   0,
    "uptime_start":    None,
    "source_name":     "",
}
_frame_lock    = threading.Lock()
_latest_frame  = None
_stop_event    = threading.Event()
_pipeline_thread = None


# ═══════════════════════════════════════════════════════════════════════════
#  Auth helpers
# ═══════════════════════════════════════════════════════════════════════════

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated

def api_login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            return jsonify({"ok": False, "msg": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated


# ═══════════════════════════════════════════════════════════════════════════
#  CV Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_pipeline(video_path: str, confidence: float):
    global _latest_frame, _state

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        _state["running"] = False
        return

    detector = PersonDetector(confidence_threshold=confidence)
    zone     = DangerZone(points=DEFAULT_ZONE_POINTS)
    alert    = AlertManager(log_path="logs/violations.log", save_violations=True)
    overlay  = UIOverlay()

    frame_count = 0
    fps_timer   = time.time()
    display_fps = 0.0

    while not _stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 30 == 0:
            elapsed     = time.time() - fps_timer
            display_fps = 30 / elapsed if elapsed > 0 else 0
            fps_timer   = time.time()

        detections       = detector.detect(frame)
        total_persons    = len(detections)
        violation_count  = 0
        violation_exists = False

        for (x1, y1, x2, y2, conf) in detections:
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

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.circle(frame, (cx, cy), 5, color, -1)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(frame, label, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        zone.draw(frame, violation_exists)

        if violation_exists:
            alert.trigger(frame, violation_count)
            _state["session_total"] += 1

        overlay.draw(frame, fps=display_fps, total_persons=total_persons,
                     violations=violation_count, zone_points=DEFAULT_ZONE_POINTS)

        _state["fps"]           = round(display_fps, 1)
        _state["total_persons"] = total_persons
        _state["violations"]    = violation_count

        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        with _frame_lock:
            _latest_frame = jpeg.tobytes()

    cap.release()
    _state["running"] = False


def gen_frames():
    while True:
        with _frame_lock:
            frame = _latest_frame
        if frame:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        time.sleep(0.033)


# ═══════════════════════════════════════════════════════════════════════════
#  Routes — Auth
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/login", methods=["GET"])
def login_page():
    if session.get("logged_in"):
        return redirect(url_for("dashboard"))
    with open(
        os.path.join(os.path.dirname(__file__), "interface.html"),
        encoding="utf-8"
    ) as f:
        return f.read()

@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json(silent=True) or {}
    if data.get("username") == USERNAME and data.get("password") == PASSWORD:
        session["logged_in"] = True
        return jsonify({"ok": True})
    return jsonify({"ok": False, "msg": "Invalid credentials"}), 401

@app.route("/api/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"ok": True})


# ═══════════════════════════════════════════════════════════════════════════
#  Routes — Pages
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/")
@login_required
def dashboard():
    with open(
        os.path.join(os.path.dirname(__file__), "interface.html"),
        encoding="utf-8"
    ) as f:
        return f.read()


# ═══════════════════════════════════════════════════════════════════════════
#  Routes — Video upload
# ═══════════════════════════════════════════════════════════════════════════

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

@app.route("/api/upload", methods=["POST"])
@api_login_required
def api_upload():
    if "video" not in request.files:
        return jsonify({"ok": False, "msg": "No file part"})
    f = request.files["video"]
    if f.filename == "":
        return jsonify({"ok": False, "msg": "No file selected"})
    if not allowed_file(f.filename):
        return jsonify({"ok": False, "msg": "File type not allowed"})

    filename = secure_filename(f.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    f.save(save_path)
    return jsonify({"ok": True, "filename": filename, "path": save_path})


# ═══════════════════════════════════════════════════════════════════════════
#  Routes — Pipeline control
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/api/start", methods=["POST"])
@api_login_required
def api_start():
    global _pipeline_thread, _stop_event, _state, _latest_frame

    if _state["running"]:
        return jsonify({"ok": False, "msg": "Already running"})

    data = request.get_json(silent=True) or {}
    filename   = data.get("filename", "")
    confidence = float(data.get("confidence", 0.3))

    video_path = os.path.join(UPLOAD_FOLDER, secure_filename(filename))
    if not os.path.exists(video_path):
        return jsonify({"ok": False, "msg": f"Video not found: {filename}"})

    _state["running"]       = True
    _state["session_total"] = 0
    _state["source_name"]   = filename
    _state["uptime_start"]  = time.time()
    _latest_frame           = None

    _stop_event.clear()
    _pipeline_thread = threading.Thread(
        target=run_pipeline, args=(video_path, confidence), daemon=True
    )
    _pipeline_thread.start()
    return jsonify({"ok": True})


@app.route("/api/stop", methods=["POST"])
@api_login_required
def api_stop():
    _stop_event.set()
    _state["running"] = False
    return jsonify({"ok": True})


@app.route("/video_feed")
@login_required
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


# ═══════════════════════════════════════════════════════════════════════════
#  Routes — Data
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/api/status")
@api_login_required
def api_status():
    uptime = 0
    if _state["uptime_start"] and _state["running"]:
        uptime = int(time.time() - _state["uptime_start"])
    return jsonify({**_state, "uptime": uptime})

@app.route("/api/snapshots")
@api_login_required
def api_snapshots():
    if not os.path.exists("violations"):
        return jsonify([])
    files = sorted(
        [f for f in os.listdir("violations") if f.endswith(".jpg")],
        reverse=True
    )[:24]
    return jsonify(files)

@app.route("/violations/<path:filename>")
@login_required
def serve_violation(filename):
    return send_from_directory("violations", filename)

@app.route("/api/log")
@api_login_required
def api_log():
    log_path = "logs/violations.log"
    if not os.path.exists(log_path):
        return jsonify([])
    with open(log_path) as f:
        lines = f.readlines()
    return jsonify([l.strip() for l in lines[-40:]][::-1])

@app.route("/api/clear_session_data", methods=["POST"])
@api_login_required
def api_clear():
    """Clear violation snapshots and log for a fresh session."""
    import glob, shutil
    for f in glob.glob("violations/*.jpg"):
        os.remove(f)
    open("logs/violations.log", "w").close()
    return jsonify({"ok": True})


# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    print(f"[INFO] SiteGuard running at http://localhost:{args.port}")
    print(f"[INFO] Login: {USERNAME} / {PASSWORD}")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)