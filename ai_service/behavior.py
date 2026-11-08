"""Student behavior classifier using student_behavior_best.pt."""

import logging
from typing import Sequence
import os
from datetime import datetime

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class StudentBehaviorModel:
    """Classify classroom behavior from a crop around a recognized student."""

    ENGAGED_LABELS = {"focused", "raising_hand"}

    def __init__(self, model_path: str, class_names: Sequence[str]):
        self.model_path = model_path
        self.class_names = list(class_names)
        self.model = None

        try:
            from ultralytics import YOLO  # type: ignore[import-not-found]

            self.model = YOLO(self.model_path)
            self._sync_class_names_from_model()
            logger.info("Loaded behavior model: %s", self.model_path)
        except Exception as e:
            logger.error("Failed to load behavior model %s: %s", self.model_path, e)
            self.model = None

    def _sync_class_names_from_model(self) -> None:
        if self.model is None:
            return

        names = getattr(self.model, "names", None)
        if isinstance(names, dict) and names:
            ordered_names = []
            for class_id in sorted(names):
                ordered_names.append(str(names[class_id]))
            self.class_names = ordered_names
            return

        if isinstance(names, (list, tuple)) and names:
            self.class_names = [str(item) for item in names]

    def _class_name(self, class_id: int) -> str:
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return "Unknown"

    @staticmethod
    def _scale_bbox_to_frame(frame: np.ndarray, detect_img: np.ndarray, bbox: np.ndarray) -> tuple[int, int, int, int] | None:
        if frame is None or detect_img is None or bbox is None:
            return None

        fh, fw = frame.shape[:2]
        dh, dw = detect_img.shape[:2]
        if fh <= 0 or fw <= 0 or dh <= 0 or dw <= 0:
            return None

        sx = float(fw) / float(dw)
        sy = float(fh) / float(dh)

        x1, y1, x2, y2 = [float(v) for v in bbox]
        x1 = int(max(0, np.floor(x1 * sx)))
        y1 = int(max(0, np.floor(y1 * sy)))
        x2 = int(min(fw, np.ceil(x2 * sx)))
        y2 = int(min(fh, np.ceil(y2 * sy)))

        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    @staticmethod
    def _crop_behavior_region(frame: np.ndarray, detect_img: np.ndarray, bbox: np.ndarray) -> np.ndarray | None:
        if frame is None or detect_img is None or bbox is None:
            return None

        fh, fw = frame.shape[:2]
        dh, dw = detect_img.shape[:2]
        if fh <= 0 or fw <= 0 or dh <= 0 or dw <= 0:
            return None

        sx = float(fw) / float(dw)
        sy = float(fh) / float(dh)

        x1, y1, x2, y2 = [float(v) for v in bbox]
        x1 *= sx
        x2 *= sx
        y1 *= sy
        y2 *= sy

        face_w = max(1.0, x2 - x1)
        face_h = max(1.0, y2 - y1)
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5

        # Expand from face box to upper-body region where classroom behavior is visible.
        half_w = 1.8 * face_w
        up = 2.1 * face_h
        down = 3.4 * face_h

        rx1 = int(max(0, np.floor(cx - half_w)))
        rx2 = int(min(fw, np.ceil(cx + half_w)))
        ry1 = int(max(0, np.floor(cy - up)))
        ry2 = int(min(fh, np.ceil(cy + down)))

        if rx2 <= rx1 or ry2 <= ry1:
            return None

        crop = frame[ry1:ry2, rx1:rx2]
        if crop is None or crop.size == 0:
            return None

        # Keep inference stable in low-end GPUs/CPUs.
        max_side = max(crop.shape[:2])
        if max_side > 960:
            scale = 960.0 / float(max_side)
            crop = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)), interpolation=cv2.INTER_AREA)

        return crop

    @staticmethod
    def _crop_person_region(frame: np.ndarray, detect_img: np.ndarray, person_bbox: np.ndarray) -> np.ndarray | None:
        scaled = StudentBehaviorModel._scale_bbox_to_frame(frame=frame, detect_img=detect_img, bbox=person_bbox)
        if scaled is None:
            return None

        x1, y1, x2, y2 = scaled
        w = max(1, x2 - x1)
        h = max(1, y2 - y1)

        # Padding đủ rộng để giữ ngữ cảnh toàn thân:
        # - Ngang (+25%): bắt tay/điện thoại, tư thế lệch (using_phone, distracted).
        # - Trên (+20%): đầu + tay giơ cao (raising_hand).
        # - Dưới (+15%): phần ngực/bàn khi ngủ gục (sleep).
        pad_w   = int(round(0.25 * w))
        pad_top = int(round(0.20 * h))
        pad_bot = int(round(0.15 * h))

        fh, fw = frame.shape[:2]
        rx1 = max(0, x1 - pad_w)
        ry1 = max(0, y1 - pad_top)
        rx2 = min(fw, x2 + pad_w)
        ry2 = min(fh, y2 + pad_bot)

        if rx2 <= rx1 or ry2 <= ry1:
            return None

        crop = frame[ry1:ry2, rx1:rx2]
        if crop is None or crop.size == 0:
            return None

        max_side = max(crop.shape[:2])
        if max_side > 960:
            scale = 960.0 / float(max_side)
            crop = cv2.resize(crop, (int(crop.shape[1] * scale), int(crop.shape[0] * scale)), interpolation=cv2.INTER_AREA)

        return crop

    def _predict_crop(self, crop: np.ndarray) -> dict | None:
        if self.model is None:
            return None

        try:
            result = self.model.predict(crop, verbose=False)[0]
        except Exception as e:
            logger.warning("Behavior inference failed: %s", e)
            return None

        class_id = -1
        confidence = 0.0

        probs = getattr(result, "probs", None)
        if probs is not None and getattr(probs, "top1", None) is not None:
            class_id = int(probs.top1)
            confidence = float(probs.top1conf)
        else:
            boxes = getattr(result, "boxes", None)
            if boxes is not None and len(boxes) > 0:
                confs = boxes.conf.tolist()
                best_idx = int(np.argmax(confs))
                class_id = int(boxes.cls[best_idx].item())
                confidence = float(confs[best_idx])

        if class_id < 0:
            return None

        label = self._class_name(class_id)
        engaged = label in self.ENGAGED_LABELS

        return {
            "behavior_class_id": int(class_id),
            "behavior_label": label,
            "behavior_confidence": float(confidence),
            "engaged": bool(engaged),
            "source": "student_behavior_best.pt",
        }

    def evaluate(self, frame: np.ndarray, detect_img: np.ndarray, bbox: np.ndarray) -> dict | None:
        crop = self._crop_behavior_region(frame=frame, detect_img=detect_img, bbox=bbox)
        if crop is None:
            return None

        return self._predict_crop(crop)

    def evaluate_person_bbox(self, frame: np.ndarray, detect_img: np.ndarray, person_bbox: np.ndarray) -> dict | None:
        crop = self._crop_person_region(frame=frame, detect_img=detect_img, person_bbox=person_bbox)
        if crop is None:
            return None
        return self._predict_crop(crop)

    def save_behavior_crop(self, crop: np.ndarray, camera_id: str, user_id: int, label: str) -> str | None:
        try:
            from config import CAPTURE_DIR
        except Exception:
            return None

        try:
            os.makedirs(CAPTURE_DIR, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            uid = int(user_id) if user_id is not None else "unknown"
            
            safe_label = str(label or "unknown").lower()
            safe_label = "".join(c if c.isalnum() or c == "_" else "" for c in safe_label)
            safe_label = safe_label[:20]
            
            filename = f"{camera_id}_{uid}_behavior_{safe_label}_{timestamp}.jpg"
            filepath = os.path.join(CAPTURE_DIR, filename)
            cv2.imwrite(filepath, crop)
            return filename
        except Exception:
            return None
