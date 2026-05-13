"""
detector.py
===========
MobileNet-SSD person detector with multi-scale inference.

Why multi-scale?
----------------
MobileNet-SSD internally resizes every frame to 300x300 before inference.
On a 1280x720 or larger frame a person who occupies only ~60-100 px of
height gets squashed to ~25 px in the blob — too small for reliable
detection.  Running a second pass on an upscaled frame recovers these
small / partially-occluded detections.

Confidence default
------------------
Lowered to 0.3 (from the original 0.5) because construction site footage
has challenging conditions: partial occlusion, low contrast against sandy/
earth backgrounds, and workers at depth in trenches.  False-positive rate
at 0.3 is still low because the zone polygon acts as a second filter —
only detections whose body-centre falls inside the polygon trigger alerts.
"""

import cv2
import numpy as np
import os
from typing import List, Tuple

# MobileNet-SSD class index for "person"
PERSON_CLASS_ID = 15


class PersonDetector:

    def __init__(
        self,
        model_dir: str = "models",
        confidence_threshold: float = 0.3,   # ← lowered from 0.5
        # Second pass: upscale the full frame by this factor before
        # running inference again.  1.5-2.0 works well for distant
        # workers; set to None to disable.
        multiscale_factor: float = 2.0,
        # Minimum box area (px²) to keep — filters tiny false positives
        min_box_area: int = 800,
        # IoU threshold for de-duplication between the two passes
        nms_iou_threshold: float = 0.45,
    ):
        self.confidence_threshold = confidence_threshold
        self.multiscale_factor    = multiscale_factor
        self.min_box_area         = min_box_area
        self.nms_iou_threshold    = nms_iou_threshold

        proto   = os.path.join(model_dir, "deploy.prototxt")
        weights = os.path.join(model_dir, "mobilenet_iter_73000.caffemodel")

        if not os.path.exists(proto) or not os.path.exists(weights):
            raise FileNotFoundError(
                f"\n[ERROR] Model files not found in '{model_dir}/'.\n"
                "Please download them:\n"
                "  prototxt : https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/"
                "master/MobileNetSSD_deploy.prototxt\n"
                "  caffemodel: https://drive.google.com/uc?id=0B3gersZ2cHIxRm5PMWRoTkdHdHc\n"
                "  (or run: python download_models.py)\n"
            )

        print("[INFO] Loading MobileNet-SSD model ...")
        self.net = cv2.dnn.readNetFromCaffe(proto, weights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        print(f"[INFO] Model loaded successfully. Confidence threshold: {self.confidence_threshold}")

    # -----------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Run person detection on *frame*.
        Returns list of (x1, y1, x2, y2, confidence) in frame pixel coords.
        """
        self._orig_h, self._orig_w = frame.shape[:2]

        all_boxes  = []
        all_scores = []

        # ── Pass 1: full frame at native resolution ───────────────────
        boxes, scores = self._infer_on(frame)
        all_boxes.extend(boxes)
        all_scores.extend(scores)

        # ── Pass 2: upscaled frame (recovers small/distant workers) ───
        if self.multiscale_factor and self.multiscale_factor > 1.0:
            up_w     = int(self._orig_w * self.multiscale_factor)
            up_h     = int(self._orig_h * self.multiscale_factor)
            upscaled = cv2.resize(frame, (up_w, up_h), interpolation=cv2.INTER_LINEAR)
            boxes2, scores2 = self._infer_on(upscaled, coord_scale=1.0 / self.multiscale_factor)
            all_boxes.extend(boxes2)
            all_scores.extend(scores2)

        if not all_boxes:
            return []

        return self._nms(all_boxes, all_scores)

    # -----------------------------------------------------------------
    def _infer_on(
        self,
        img: np.ndarray,
        coord_scale: float = 1.0,
    ) -> Tuple[list, list]:
        """
        Run one forward pass on *img*.

        coord_scale : multiply detected pixel coords by this to map them
                      back to the original frame resolution.
                      Pass 1 → 1.0 (no change)
                      Pass 2 → 1/multiscale_factor  (shrink back down)
        """
        ih, iw = img.shape[:2]

        blob = cv2.dnn.blobFromImage(
            img,
            scalefactor=0.007843,
            size=(300, 300),
            mean=(127.5, 127.5, 127.5),
            swapRB=False,
            crop=False,
        )
        self.net.setInput(blob)
        detections = self.net.forward()   # shape: (1, 1, N, 7)

        boxes  = []
        scores = []

        for i in range(detections.shape[2]):
            conf     = float(detections[0, 0, i, 2])
            class_id = int(detections[0, 0, i, 1])

            if class_id != PERSON_CLASS_ID or conf < self.confidence_threshold:
                continue

            # Normalised coords → pixel coords in *img* space
            x1 = int(detections[0, 0, i, 3] * iw)
            y1 = int(detections[0, 0, i, 4] * ih)
            x2 = int(detections[0, 0, i, 5] * iw)
            y2 = int(detections[0, 0, i, 6] * ih)

            # Scale back to original frame space
            x1 = int(x1 * coord_scale)
            y1 = int(y1 * coord_scale)
            x2 = int(x2 * coord_scale)
            y2 = int(y2 * coord_scale)

            # Clamp to original frame boundaries
            x1 = max(0, min(x1, self._orig_w - 1))
            y1 = max(0, min(y1, self._orig_h - 1))
            x2 = max(0, min(x2, self._orig_w - 1))
            y2 = max(0, min(y2, self._orig_h - 1))

            area = (x2 - x1) * (y2 - y1)
            if area < self.min_box_area:
                continue

            boxes.append([x1, y1, x2, y2])
            scores.append(conf)

        return boxes, scores

    # -----------------------------------------------------------------
    def _nms(self, boxes: list, scores: list) -> List[Tuple[int, int, int, int, float]]:
        """Apply NMS and return final (x1, y1, x2, y2, conf) tuples."""
        rects = [[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes]
        idxs  = cv2.dnn.NMSBoxes(
            rects,
            scores,
            self.confidence_threshold,
            self.nms_iou_threshold,
        )

        results = []
        if len(idxs) > 0:
            for i in idxs.flatten():
                x1, y1, x2, y2 = boxes[i]
                results.append((x1, y1, x2, y2, scores[i]))
        return results