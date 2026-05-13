"""
alert.py
========
Handles all alerting logic:
  1. On-screen text flash  (drawn in overlay.py, triggered here)
  2. Violation log file    (CSV-style, human-readable)
  3. Snapshot images       (saved to violations/ folder)
  4. Optional system beep  (cross-platform, silent if unavailable)

Log format
----------
TIMESTAMP                | VIOLATION_COUNT | SNAPSHOT_PATH
2024-06-01 10:23:45.123  |  2              | violations/snap_20240601_102345.jpg
"""

import cv2
import os
import time
import threading
import datetime
import numpy as np
from typing import Optional


class AlertManager:
    """
    Parameters
    ----------
    log_path        : path to the .log file (created automatically)
    save_violations : if True, save a JPEG snapshot on each violation
    snapshot_dir    : folder where snapshots are stored
    cooldown_secs   : minimum seconds between successive alerts (avoids spam)
    beep            : attempt a system beep on violation
    """

    def __init__(
        self,
        log_path: str         = "logs/violations.log",
        save_violations: bool = True,
        snapshot_dir: str     = "violations",
        cooldown_secs: float  = 2.0,
        beep: bool            = True,
    ):
        self.log_path        = log_path
        self.save_violations = save_violations
        self.snapshot_dir    = snapshot_dir
        self.cooldown_secs   = cooldown_secs
        self.beep_enabled    = beep
        self._last_alert_time: float = 0.0

        # Create directories
        os.makedirs(os.path.dirname(log_path) if os.path.dirname(log_path) else ".", exist_ok=True)
        if save_violations:
            os.makedirs(snapshot_dir, exist_ok=True)

        # Write log header if file is new
        if not os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.write("timestamp,violations,snapshot\n")

    # ─────────────────────────────────────────────────────────────────
    def trigger(self, frame: np.ndarray, violation_count: int) -> None:
        """
        Call this once per frame whenever ≥1 violation is detected.
        Rate-limited by cooldown_secs.
        """
        now = time.time()
        if (now - self._last_alert_time) < self.cooldown_secs:
            return

        self._last_alert_time = now
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # ── Snapshot ──────────────────────────────────────────────────
        snapshot_path = ""
        if self.save_violations:
            fname         = datetime.datetime.now().strftime("snap_%Y%m%d_%H%M%S.jpg")
            snapshot_path = os.path.join(self.snapshot_dir, fname)
            cv2.imwrite(snapshot_path, frame)

        # ── Log entry ─────────────────────────────────────────────────
        with open(self.log_path, "a") as f:
            f.write(f"{timestamp},{violation_count},{snapshot_path}\n")

        print(
            f"[ALERT] {timestamp} — {violation_count} person(s) in danger zone."
            + (f" Snapshot: {snapshot_path}" if snapshot_path else "")
        )

        # ── Beep (non-blocking) ───────────────────────────────────────
        if self.beep_enabled:
            threading.Thread(target=self._beep, daemon=True).start()

    # ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _beep() -> None:
        """
        Cross-platform system beep.  Silently skipped if unavailable.
        """
        try:
            import sys
            if sys.platform == "win32":
                import winsound
                winsound.Beep(880, 300)           # 880 Hz, 300 ms
            elif sys.platform == "darwin":
                os.system("afplay /System/Library/Sounds/Ping.aiff &")
            else:
                # Linux — try print('\a') or beep command
                print("\a", end="", flush=True)   # terminal bell
        except Exception:
            pass   # beep is optional — never crash the main thread


# ─────────────────────────────────────────────────────────────────────────────
#  Stand-alone log reader utility
# ─────────────────────────────────────────────────────────────────────────────

def print_log_summary(log_path: str = "logs/violations.log") -> None:
    """Print a quick summary of the violation log to stdout."""
    if not os.path.exists(log_path):
        print(f"[INFO] No log file found at {log_path}")
        return

    with open(log_path) as f:
        lines = f.readlines()

    entries = [l for l in lines[1:] if l.strip()]   # skip header
    print(f"\n{'─'*50}")
    print(f"  Violation Log Summary: {log_path}")
    print(f"  Total events : {len(entries)}")
    if entries:
        print(f"  First event  : {entries[0].split(',')[0]}")
        print(f"  Last event   : {entries[-1].split(',')[0]}")
    print(f"{'─'*50}\n")
