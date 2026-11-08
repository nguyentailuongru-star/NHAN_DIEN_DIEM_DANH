"""AI Service chính - đọc RTSP, nhận diện khuôn mặt, gửi kết quả về Flask API."""

import cv2
import time
import os
import logging
import threading
import requests
import numpy as np
import argparse
from collections import defaultdict, deque
from datetime import datetime

from config import (
    CAMERAS,
    API_RECOGNITION_URL,
    API_ATTENTION_URL,
    API_CAMERA_SYNC_URL,
    CAMERA_CONNECT_ATTEMPTS,
    CAMERA_RETRY_DELAY_SEC,
    BEHAVIOR_MODEL_FILENAME,
    BEHAVIOR_CLASS_NAMES,
    FACE_MODEL_NAME,
    GPU_ID,
    MODEL_ROOT,
    FRAME_INTERVAL,
    PROCESS_WIDTH,
    PROCESS_HEIGHT,
    DETECTION_WIDTH,
    DETECTION_HEIGHT,
    DETECTION_THRESH,
    VIDEO_DETECTION_WIDTH,
    VIDEO_DETECTION_HEIGHT,
    VIDEO_DETECTION_THRESH,
    VIDEO_PROCESS_MAX_WIDTH,
    VIDEO_PROCESS_MAX_HEIGHT,
    RECOGNITION_COOLDOWN,
    BEHAVIOR_POST_INTERVAL,
    BEHAVIOR_SAVE_IMAGES,
    BEHAVIOR_CAPTURE_INTERVAL,
    CAPTURE_DIR,
    RTSP_TRANSPORT,
    RTSP_OPEN_TIMEOUT_MS,
    RTSP_READ_TIMEOUT_MS,
    RTSP_MAX_DELAY_MS,
    RTSP_BUFFER_SIZE,
    ENABLE_PERSON_TRACKING,
    PERSON_DETECT_MODEL_FILENAME,
    PERSON_TRACKER_CONFIG,
    PERSON_TRACK_CONF,
    PERSON_TRACK_IOU,
    TRACK_STALE_SECONDS,
    TRACK_RECOGNITION_REFRESH_SEC,
    TRACK_BEHAVIOR_WINDOW_SIZE,
    TRACK_BEHAVIOR_MIN_SAMPLES,
)
from face_manager import FaceManager
from face_analysis_local import FaceAnalysisLocal
from model_store import ensure_face_model, ensure_yolo_weights
from behavior import StudentBehaviorModel, write_capture_jpg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class AIService:
    """Service chính xử lý nhận diện khuôn mặt từ camera RTSP."""

    def __init__(self):
        self.face_app = None
        self.person_model = None
        self.person_tracking_enabled = bool(ENABLE_PERSON_TRACKING)
        self.face_manager = FaceManager()
        self.behavior_model = StudentBehaviorModel(
            model_path=os.path.join(MODEL_ROOT, BEHAVIOR_MODEL_FILENAME),
            class_names=BEHAVIOR_CLASS_NAMES,
        )
        # Theo dõi cooldown: {user_id: last_recognition_time}
        self.last_seen = {}
        # Theo dõi tần suất gửi hành vi: {(camera_id, user_id): last_send_ts}
        self.last_behavior_sent = {}
        # Lần lưu ảnh hành vi gần nhất: {(camera_id, user_id): ts} → giới hạn số ảnh trong captures/
        self.last_behavior_capture = {}
        # Camera: {code: worker}, {code: {state, message, since}}
        self._cam_lock = threading.Lock()
        self.camera_workers = {}
        self.camera_status = {}
        # Theo dõi track theo camera: {(camera_id, track_id): state}
        self.track_states = {}
        self.running = False
        self.mode = "rtsp"  # "rtsp" | "video"

        # Debug counters (đặc biệt hữu ích cho video offline)
        self._dbg_frames_processed = 0
        self._dbg_frames_with_face = 0
        self._dbg_frames_with_embedding = 0
        self._dbg_frames_with_match = 0
        self._dbg_best_sim = -1.0
        self._dbg_best_name = None
        self._dbg_last_report = 0.0

    def init_model(self):
        """Khởi tạo InsightFace model."""
        model_dir = ensure_face_model()  # model cài trên máy, tự tải nếu chưa có
        logger.info("Đang load model InsightFace: %s (GPU: %d)", model_dir, GPU_ID)
        self.face_app = FaceAnalysisLocal(model_dir=model_dir, ctx_id=GPU_ID)
        if self.mode == "video":
            det_size = (VIDEO_DETECTION_WIDTH, VIDEO_DETECTION_HEIGHT)
            det_thresh = VIDEO_DETECTION_THRESH
        else:
            det_size = (DETECTION_WIDTH, DETECTION_HEIGHT)
            det_thresh = DETECTION_THRESH

        self.face_app.prepare(ctx_id=GPU_ID, det_size=det_size, det_thresh=det_thresh)
        self._init_person_tracker()
        logger.info("Detector config: det_size=%s, det_thresh=%.2f (mode=%s)", det_size, float(det_thresh), self.mode)
        logger.info("Model đã sẵn sàng!")

    def _init_person_tracker(self):
        if not self.person_tracking_enabled:
            self.person_model = None
            return

        try:
            from ultralytics import YOLO  # type: ignore[import-not-found]

            model_path = ensure_yolo_weights(PERSON_DETECT_MODEL_FILENAME)
            self.person_model = YOLO(model_path)
            logger.info(
                "Loaded person detector for tracking: %s (tracker=%s)",
                model_path,
                PERSON_TRACKER_CONFIG,
            )
        except Exception as e:
            self.person_model = None
            self.person_tracking_enabled = False
            logger.warning("Không thể khởi tạo person tracking, fallback face-only: %s", e)

    def load_known_faces(self):
        """Load danh sách khuôn mặt đã đăng ký từ API."""
        self.face_manager.load_from_api()

    def _is_cooldown(self, user_id):
        """Kiểm tra xem user có đang trong thời gian cooldown không."""
        now = time.time()
        last = self.last_seen.get(user_id, 0)
        return (now - last) < RECOGNITION_COOLDOWN

    def _update_cooldown(self, user_id):
        """Cập nhật thời gian nhận diện mới nhất."""
        self.last_seen[user_id] = time.time()

    def _can_send_behavior(self, camera_id, user_id):
        key = (str(camera_id), int(user_id))
        now = time.time()
        last_ts = self.last_behavior_sent.get(key, 0.0)
        if (now - last_ts) < float(BEHAVIOR_POST_INTERVAL):
            return False
        self.last_behavior_sent[key] = now
        return True

    def _take_behavior_capture_slot(self, camera_id, user_id):
        """True nếu được phép lưu 1 ảnh hành vi cho người này (tối đa 1 ảnh / BEHAVIOR_CAPTURE_INTERVAL)."""
        if not BEHAVIOR_SAVE_IMAGES:
            return False
        if not hasattr(self, "last_behavior_capture"):
            self.last_behavior_capture = {}
        key = (str(camera_id), int(user_id))
        now = time.time()
        if (now - self.last_behavior_capture.get(key, 0.0)) < float(BEHAVIOR_CAPTURE_INTERVAL):
            return False
        self.last_behavior_capture[key] = now
        return True

    def _save_capture(self, frame, camera_id, user_id):
        """Lưu ảnh capture khi nhận diện thành công.

        Đặt tên file không dấu để tránh lỗi URL/encoding trên web.
        """
        os.makedirs(CAPTURE_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{camera_id}_{int(user_id) if user_id is not None else 'unknown'}_{timestamp}.jpg"
        filepath = os.path.join(CAPTURE_DIR, filename)
        write_capture_jpg(filepath, frame)
        return filename

    def _save_face_capture(self, frame, detect_img, bbox, camera_id, user_id, margin: float = 0.25):
        """Lưu ảnh crop khuôn mặt dựa theo bbox detect.

        - bbox là tọa độ theo detect_img.
        - frame là khung hình gốc (độ phân giải đầy đủ).
        """
        try:
            os.makedirs(CAPTURE_DIR, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            uid = int(user_id) if user_id is not None else "unknown"

            # Map bbox từ detect_img → frame gốc
            fh, fw = frame.shape[:2]
            dh, dw = detect_img.shape[:2]
            sx = float(fw) / float(dw) if dw else 1.0
            sy = float(fh) / float(dh) if dh else 1.0

            x1, y1, x2, y2 = [float(v) for v in bbox]
            x1 *= sx
            x2 *= sx
            y1 *= sy
            y2 *= sy

            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            pad = float(margin) * max(bw, bh)

            x1 = int(max(0, np.floor(x1 - pad)))
            y1 = int(max(0, np.floor(y1 - pad)))
            x2 = int(min(fw, np.ceil(x2 + pad)))
            y2 = int(min(fh, np.ceil(y2 + pad)))

            if x2 <= x1 or y2 <= y1:
                raise ValueError("Invalid crop")

            crop = frame[y1:y2, x1:x2]
            if crop is None or crop.size == 0:
                raise ValueError("Empty crop")

            filename = f"{camera_id}_{uid}_face_{timestamp}.jpg"
            filepath = os.path.join(CAPTURE_DIR, filename)
            write_capture_jpg(filepath, crop)
            return filename
        except Exception:
            # Fallback: nếu crop lỗi thì lưu full frame
            return self._save_capture(frame, camera_id, user_id)

    def _send_to_api(self, user_id, user_name, camera_id, image_filename, similarity):
        """Gửi kết quả nhận diện về Flask API."""
        payload = {
            "user_id": user_id,
            "name": user_name,
            "camera_id": camera_id,
            "image": image_filename,
            "similarity": similarity,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            resp = requests.post(API_RECOGNITION_URL, json=payload, timeout=5)
            resp.raise_for_status()
            body = resp.json() if resp.content else {}
            if body.get("status") == "skipped":
                # Server đã có điểm danh gần đây của người này → không lưu ảnh thừa.
                if image_filename:
                    try:
                        os.remove(os.path.join(CAPTURE_DIR, image_filename))
                    except OSError:
                        pass
                logger.info(
                    "Bỏ qua %s: đã điểm danh gần đây (còn %ss nữa mới ghi lại)",
                    user_name, body.get("retry_after"),
                )
                return
            logger.info("Gửi API thành công: %s (sim=%.4f)", user_name, similarity)
        except requests.RequestException as e:
            logger.error("Gửi API thất bại: %s", e)

    def _send_behavior_to_api(self, user_id, user_name, camera_id, behavior_result):
        """Gửi dữ liệu hành vi trong lớp về Flask API."""
        payload = {
            "user_id": user_id,
            "name": user_name,
            "camera_id": camera_id,
            "behavior_class_id": behavior_result.get("behavior_class_id"),
            "behavior_label": behavior_result.get("behavior_label"),
            "behavior_confidence": behavior_result.get("behavior_confidence"),
            "engaged": bool(behavior_result.get("engaged", False)),
            "source": behavior_result.get("source", "student_behavior_best.pt"),
            "image": behavior_result.get("image_file"),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        try:
            resp = requests.post(API_ATTENTION_URL, json=payload, timeout=5)
            resp.raise_for_status()
        except requests.RequestException as e:
            logger.error("Gửi behavior API thất bại: %s", e)

    def _prepare_detect_img(self, frame):
        detect_img = frame
        if self.mode != "video":
            detect_img = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
        else:
            try:
                h, w = frame.shape[:2]
                max_w = int(VIDEO_PROCESS_MAX_WIDTH)
                max_h = int(VIDEO_PROCESS_MAX_HEIGHT)
                if max_w > 0 and max_h > 0:
                    scale = min(float(max_w) / float(w), float(max_h) / float(h), 1.0)
                    if scale < 1.0:
                        new_w = max(1, int(round(w * scale)))
                        new_h = max(1, int(round(h * scale)))
                        detect_img = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            except Exception:
                detect_img = frame
        return detect_img

    def _track_persons(self, detect_img):
        if not self.person_tracking_enabled or self.person_model is None:
            return {}

        try:
            result = self.person_model.track(
                source=detect_img,
                persist=True,
                verbose=False,
                tracker=PERSON_TRACKER_CONFIG,
                conf=float(PERSON_TRACK_CONF),
                iou=float(PERSON_TRACK_IOU),
                classes=[0],
            )[0]
        except Exception as e:
            logger.warning("Person tracking failed, fallback frame: %s", e)
            return {}

        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) <= 0:
            return {}

        ids_data = getattr(boxes, "id", None)
        if ids_data is None:
            return {}

        try:
            xyxy = boxes.xyxy.cpu().numpy()
            cls_arr = boxes.cls.cpu().numpy() if getattr(boxes, "cls", None) is not None else np.zeros(len(xyxy))
            conf_arr = boxes.conf.cpu().numpy() if getattr(boxes, "conf", None) is not None else np.ones(len(xyxy))
            ids_arr = ids_data.cpu().numpy().astype(int)
        except Exception:
            return {}

        tracks = {}
        for i in range(len(xyxy)):
            if i >= len(ids_arr) or i >= len(cls_arr):
                continue
            if int(cls_arr[i]) != 0:
                continue
            tid = int(ids_arr[i])
            tracks[tid] = {
                "bbox": xyxy[i],
                "conf": float(conf_arr[i]) if i < len(conf_arr) else 1.0,
            }
        return tracks

    @staticmethod
    def _bbox_iou(a, b):
        ax1, ay1, ax2, ay2 = [float(v) for v in a]
        bx1, by1, bx2, by2 = [float(v) for v in b]

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0

        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = area_a + area_b - inter
        if denom <= 0.0:
            return 0.0
        return float(inter / denom)

    @staticmethod
    def _face_track_score(face_bbox, person_bbox):
        fx1, fy1, fx2, fy2 = [float(v) for v in face_bbox]
        px1, py1, px2, py2 = [float(v) for v in person_bbox]

        fcx = (fx1 + fx2) * 0.5
        fcy = (fy1 + fy2) * 0.5
        in_person = (px1 <= fcx <= px2) and (py1 <= fcy <= py2)
        if not in_person:
            return -1.0

        person_h = max(1.0, py2 - py1)
        upper_region_ok = fcy <= (py1 + 0.72 * person_h)
        if not upper_region_ok:
            return -1.0

        iou = AIService._bbox_iou(face_bbox, person_bbox)
        return float(iou + 0.15)

    def _match_faces_to_tracks(self, faces, tracks):
        matched = {}
        if not faces or not tracks:
            return matched

        for face in faces:
            best_tid = None
            best_score = -1.0
            for tid, tinfo in tracks.items():
                score = self._face_track_score(face.bbox, tinfo["bbox"])
                if score > best_score:
                    best_score = score
                    best_tid = tid

            if best_tid is None or best_score <= 0.0:
                continue

            prev = matched.get(best_tid)
            if prev is None or best_score > prev[0]:
                matched[best_tid] = (best_score, face)

        return {tid: data[1] for tid, data in matched.items()}

    def _cleanup_track_states(self, camera_id, active_track_ids):
        now = time.time()
        stale_keys = []
        active = set(int(v) for v in active_track_ids)

        for key, state in self.track_states.items():
            cam, tid = key
            if str(cam) != str(camera_id):
                continue
            if int(tid) in active:
                continue
            if (now - float(state.get("last_seen_ts", 0.0))) > float(TRACK_STALE_SECONDS):
                stale_keys.append(key)

        for key in stale_keys:
            self.track_states.pop(key, None)

    def _get_track_state(self, camera_id, track_id):
        key = (str(camera_id), int(track_id))
        state = self.track_states.get(key)
        if state is not None:
            return state

        state = {
            "user_id": None,
            "user_name": None,
            "last_similarity": 0.0,
            "last_recognition_ts": 0.0,
            "last_seen_ts": 0.0,
            "behavior_history": deque(maxlen=int(TRACK_BEHAVIOR_WINDOW_SIZE)),
        }
        self.track_states[key] = state
        return state

    def _aggregate_track_behavior(self, state):
        history = list(state.get("behavior_history") or [])
        if len(history) < int(TRACK_BEHAVIOR_MIN_SAMPLES):
            return None

        label_scores = defaultdict(float)
        label_conf_sums = defaultdict(float)
        label_counts = defaultdict(int)
        class_id_by_label = {}
        engaged_weight = 0.0
        total_weight = 0.0
        last_image_file = None

        for item in history:
            label = str(item.get("behavior_label") or "Unknown")
            conf = float(item.get("behavior_confidence") or 0.0)
            weight = max(0.05, conf)
            label_scores[label] += weight
            label_conf_sums[label] += conf
            label_counts[label] += 1
            class_id_by_label[label] = int(item.get("behavior_class_id", -1))
            total_weight += weight
            if bool(item.get("engaged", False)):
                engaged_weight += weight
            if item.get("image_file"):
                last_image_file = item.get("image_file")

        if not label_scores:
            return None

        best_label = max(label_scores, key=label_scores.get)
        avg_conf = 0.0
        if label_counts[best_label] > 0:
            avg_conf = label_conf_sums[best_label] / float(label_counts[best_label])

        engaged_ratio = (engaged_weight / total_weight) if total_weight > 0 else 0.0
        return {
            "behavior_class_id": int(class_id_by_label.get(best_label, -1)),
            "behavior_label": best_label,
            "behavior_confidence": float(max(0.0, min(1.0, avg_conf))),
            "engaged": bool(engaged_ratio >= 0.5),
            "source": "student_behavior_best.pt",
            "image_file": last_image_file,
        }

    def _process_faces_legacy(self, frame, detect_img, faces, camera_id):
        best_seen = None

        for face in faces:
            embedding = face.embedding
            if embedding is None:
                continue

            self._dbg_frames_with_embedding += 1
            best_seen = self.face_manager.find_best(embedding)
            match = self.face_manager.find_match(embedding)

            if best_seen is not None:
                sim = float(best_seen.get("similarity", -1.0))
                if sim > self._dbg_best_sim:
                    self._dbg_best_sim = sim
                    self._dbg_best_name = best_seen.get("name")

            if match and not self._is_cooldown(match["user_id"]):
                self._update_cooldown(match["user_id"])
                self._dbg_frames_with_match += 1

                image_file = self._save_face_capture(
                    frame=frame,
                    detect_img=detect_img,
                    bbox=face.bbox,
                    camera_id=camera_id,
                    user_id=match["user_id"],
                )

                threading.Thread(
                    target=self._send_to_api,
                    args=(match["user_id"], match["name"], camera_id, image_file, match["similarity"]),
                    daemon=True,
                ).start()

                logger.info("[%s] Nhận diện: %s (sim=%.4f)", camera_id, match["name"], match["similarity"])

            if match:
                behavior_result = self.behavior_model.evaluate(
                    frame=frame,
                    detect_img=detect_img,
                    bbox=face.bbox,
                )
                if behavior_result and self._can_send_behavior(camera_id, match["user_id"]):
                    crop = None
                    if self._take_behavior_capture_slot(camera_id, match["user_id"]):
                        crop = self.behavior_model._crop_behavior_region(frame=frame, detect_img=detect_img, bbox=face.bbox)
                    if crop is not None:
                        image_file = self.behavior_model.save_behavior_crop(
                            crop=crop,
                            camera_id=camera_id,
                            user_id=match["user_id"],
                            label=behavior_result.get("behavior_label", "unknown"),
                        )
                        if image_file:
                            behavior_result["image_file"] = image_file
                    threading.Thread(
                        target=self._send_behavior_to_api,
                        args=(match["user_id"], match["name"], camera_id, behavior_result),
                        daemon=True,
                    ).start()

    def _process_with_tracking(self, frame, detect_img, faces, camera_id):
        tracks = self._track_persons(detect_img)
        if not tracks:
            self._process_faces_legacy(frame=frame, detect_img=detect_img, faces=faces, camera_id=camera_id)
            return

        self._cleanup_track_states(camera_id=camera_id, active_track_ids=tracks.keys())
        face_by_track = self._match_faces_to_tracks(faces=faces, tracks=tracks)
        now = time.time()

        for track_id, tinfo in tracks.items():
            state = self._get_track_state(camera_id=camera_id, track_id=track_id)
            state["last_seen_ts"] = now

            face = face_by_track.get(track_id)
            if face is not None and face.embedding is not None:
                self._dbg_frames_with_embedding += 1

                need_refresh = (
                    state.get("user_id") is None
                    or (now - float(state.get("last_recognition_ts", 0.0))) >= float(TRACK_RECOGNITION_REFRESH_SEC)
                )

                if need_refresh:
                    best_seen = self.face_manager.find_best(face.embedding)
                    match = self.face_manager.find_match(face.embedding)
                    state["last_recognition_ts"] = now

                    if best_seen is not None:
                        sim = float(best_seen.get("similarity", -1.0))
                        if sim > self._dbg_best_sim:
                            self._dbg_best_sim = sim
                            self._dbg_best_name = best_seen.get("name")

                    if match:
                        state["user_id"] = int(match["user_id"])
                        state["user_name"] = match["name"]
                        state["last_similarity"] = float(match.get("similarity") or 0.0)

            if state.get("user_id") is not None and face is not None and not self._is_cooldown(state["user_id"]):
                self._update_cooldown(state["user_id"])
                self._dbg_frames_with_match += 1

                image_file = self._save_face_capture(
                    frame=frame,
                    detect_img=detect_img,
                    bbox=face.bbox,
                    camera_id=camera_id,
                    user_id=state["user_id"],
                )

                threading.Thread(
                    target=self._send_to_api,
                    args=(
                        state["user_id"],
                        state.get("user_name") or "Unknown",
                        camera_id,
                        image_file,
                        float(state.get("last_similarity") or 0.0),
                    ),
                    daemon=True,
                ).start()

                logger.info(
                    "[%s][track=%s] Nhận diện: %s",
                    camera_id,
                    track_id,
                    state.get("user_name") or "Unknown",
                )

            if state.get("user_id") is None:
                continue

            behavior_result = self.behavior_model.evaluate_person_bbox(
                frame=frame,
                detect_img=detect_img,
                person_bbox=tinfo["bbox"],
            )
            if behavior_result is not None:
                state["behavior_history"].append(behavior_result)

            smoothed = self._aggregate_track_behavior(state)
            if smoothed and self._can_send_behavior(camera_id, state["user_id"]):
                smoothed["image_file"] = None
                if self._take_behavior_capture_slot(camera_id, state["user_id"]):
                    crop = self.behavior_model._crop_person_region(frame=frame, detect_img=detect_img, person_bbox=tinfo["bbox"])
                    if crop is not None:
                        smoothed["image_file"] = self.behavior_model.save_behavior_crop(
                            crop=crop,
                            camera_id=camera_id,
                            user_id=state["user_id"],
                            label=smoothed.get("behavior_label", "unknown"),
                        )
                threading.Thread(
                    target=self._send_behavior_to_api,
                    args=(state["user_id"], state.get("user_name") or "Unknown", camera_id, smoothed),
                    daemon=True,
                ).start()

    def process_frame(self, frame, camera_id):
        """Xử lý 1 frame: tracking person + nhận diện khuôn mặt + hành vi."""
        self._dbg_frames_processed += 1
        detect_img = self._prepare_detect_img(frame)

        faces = self.face_app.get(detect_img)
        if faces:
            self._dbg_frames_with_face += 1

        if self.person_tracking_enabled and self.person_model is not None:
            self._process_with_tracking(frame=frame, detect_img=detect_img, faces=faces, camera_id=camera_id)
        else:
            self._process_faces_legacy(frame=frame, detect_img=detect_img, faces=faces, camera_id=camera_id)

        # Thống kê định kỳ để debug trường hợp "chạy nhưng không nhận diện"
        now = time.time()
        if now - self._dbg_last_report >= 5.0:
            self._dbg_last_report = now
            logger.info(
                "[%s] Debug: processed=%d, frames_with_face=%d, frames_with_emb=%d, matches=%d, best_sim=%.3f (%s), tracks=%d",
                camera_id,
                self._dbg_frames_processed,
                self._dbg_frames_with_face,
                self._dbg_frames_with_embedding,
                self._dbg_frames_with_match,
                float(self._dbg_best_sim),
                self._dbg_best_name or "--",
                len([1 for (cam, _) in self.track_states.keys() if str(cam) == str(camera_id)]),
            )

    # ================================================================ camera manager
    # Danh sách camera do admin cấu hình trên web (Flask /api/cameras/ai-sync).
    # Mỗi camera chạy 1 thread. Lỗi kết nối: thử tối đa CAMERA_CONNECT_ATTEMPTS lần rồi DỪNG HẲN,
    # báo trạng thái "error" (dashboard hiện đỏ) và chờ admin bấm "Khởi động lại camera".

    def _set_cam_status(self, code, state, message=""):
        with self._cam_lock:
            cur = self.camera_status.setdefault(code, {"code": code})
            if cur.get("state") == state and cur.get("message") == message:
                return
            cur.update(state=state, message=message, since=datetime.now().strftime("%H:%M:%S %d/%m/%Y"))
        if state == "error":
            logger.error("[camera %s] DỪNG: %s (chờ admin bấm 'Khởi động lại camera')", code, message)

    def _start_camera_worker(self, cam):
        code = str(cam["code"])
        worker = {
            "cfg": dict(cam),
            "stop": threading.Event(),
            "restart": threading.Event(),
        }
        worker["thread"] = threading.Thread(target=self.stream_camera, args=(worker,), daemon=True, name=f"cam-{code}")
        with self._cam_lock:
            self.camera_workers[code] = worker
        worker["thread"].start()

    def _restart_camera(self, code):
        with self._cam_lock:
            worker = self.camera_workers.get(code)
        if worker is None:
            return
        logger.info("[camera %s] Admin yêu cầu khởi động lại", code)
        if worker["thread"].is_alive():
            worker["restart"].set()
        else:
            self._start_camera_worker(worker["cfg"])

    def _apply_camera_list(self, cameras):
        """Đồng bộ thread camera với danh sách admin cấu hình (thêm / bỏ / đổi URL)."""
        desired = {str(c["code"]): c for c in cameras if c.get("code") and c.get("rtsp_url")}
        with self._cam_lock:
            current = dict(self.camera_workers)
        for code, worker in current.items():
            if code not in desired:
                logger.info("[camera %s] Đã bị xóa/tắt trên web → dừng", code)
                worker["stop"].set()
                with self._cam_lock:
                    self.camera_workers.pop(code, None)
                    self.camera_status.pop(code, None)
        for code, cam in desired.items():
            worker = current.get(code)
            if worker is None:
                logger.info("[camera %s] Thêm camera: %s", code, cam.get("name"))
                self._start_camera_worker(cam)
            elif worker["cfg"].get("rtsp_url") != cam["rtsp_url"] or worker["cfg"].get("name") != cam.get("name"):
                logger.info("[camera %s] Cấu hình thay đổi → kết nối lại", code)
                worker["cfg"] = dict(cam)
                if worker["thread"].is_alive():
                    worker["restart"].set()
                else:
                    self._start_camera_worker(cam)

    def _sync_with_server(self):
        """Báo trạng thái camera, nhận danh sách camera + lệnh khởi động lại. Trả về True nếu OK."""
        with self._cam_lock:
            statuses = [dict(v) for v in self.camera_status.values()]
        try:
            resp = requests.post(API_CAMERA_SYNC_URL, json={"cameras": statuses}, timeout=3)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            if not getattr(self, "_sync_warned", False):
                logger.warning("Chưa đồng bộ được cấu hình camera với Flask: %s", e)
                self._sync_warned = True
            return False
        self._sync_warned = False
        self._apply_camera_list(data.get("cameras") or [])
        for code in data.get("restart") or []:
            self._restart_camera(str(code))
        return True

    def stream_camera(self, worker):
        """Đọc stream từ 1 camera RTSP và xử lý (chạy trong thread riêng)."""
        cfg = worker["cfg"]
        code = str(cfg["code"])
        stop, restart = worker["stop"], worker["restart"]

        # Tối ưu FFmpeg backend cho RTSP (stimeout tính theo microseconds)
        try:
            stimeout_us = max(int(RTSP_OPEN_TIMEOUT_MS), int(RTSP_READ_TIMEOUT_MS)) * 1000
        except Exception:
            stimeout_us = 5_000_000
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "|".join([
            f"rtsp_transport;{RTSP_TRANSPORT}",
            f"stimeout;{stimeout_us}",
            f"max_delay;{int(RTSP_MAX_DELAY_MS) * 1000}",
        ])

        max_attempts = max(1, int(CAMERA_CONNECT_ATTEMPTS))
        attempts = 0
        lost_before = False

        while self.running and not stop.is_set():
            if restart.is_set():
                restart.clear()
                attempts = 0
                lost_before = False
            cfg = worker["cfg"]  # có thể đã đổi URL trên web
            rtsp_url = str(cfg["rtsp_url"]).strip()
            source = int(rtsp_url) if rtsp_url.isdigit() else rtsp_url

            attempts += 1
            prefix = "Mất kết nối, đang thử lại" if lost_before else "Đang kết nối"
            self._set_cam_status(code, "connecting", f"{prefix} (lần {attempts}/{max_attempts})")
            logger.info("[camera %s] %s: %s (lần %d/%d)", code, prefix, cfg.get("name"), attempts, max_attempts)

            if isinstance(source, str):
                # Timeout qua tham số OpenCV (FFmpeg mới bỏ "stimeout" → nếu không sẽ treo ~30s khi sai IP)
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG, [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(RTSP_OPEN_TIMEOUT_MS),
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(RTSP_READ_TIMEOUT_MS),
                ])
            else:
                cap = cv2.VideoCapture(source)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, RTSP_BUFFER_SIZE)
            except Exception:
                pass

            if not cap.isOpened():
                cap.release()
                if attempts >= max_attempts:
                    reason = "Mất kết nối camera" if lost_before else "Không kết nối được camera"
                    self._set_cam_status(
                        code, "error",
                        f"{reason} sau {max_attempts} lần thử (sai IP/mật khẩu, camera tắt hoặc mất mạng)",
                    )
                    # Dừng hẳn: chờ admin bấm "Khởi động lại" (hoặc camera bị xóa / dịch vụ dừng)
                    while self.running and not stop.is_set() and not restart.wait(1.0):
                        pass
                    continue
                restart.wait(float(CAMERA_RETRY_DELAY_SEC))
                continue

            attempts = 0
            lost_before = False
            self._set_cam_status(code, "online", "")
            logger.info("[camera %s] Đã kết nối: %s", code, cfg.get("name"))

            lost = False
            next_process_time = time.monotonic()
            while self.running and not stop.is_set() and not restart.is_set():
                # Grab liên tục để luôn lấy frame mới nhất, tránh dồn buffer
                if not cap.grab():
                    lost = True
                    break
                now = time.monotonic()
                if now < next_process_time:
                    time.sleep(0.01)
                    continue
                ret, frame = cap.retrieve()
                if not ret or frame is None:
                    lost = True
                    break
                try:
                    self.process_frame(frame, code)
                except Exception as e:
                    logger.error("Lỗi xử lý frame camera %s: %s", code, e)
                next_process_time = now + float(FRAME_INTERVAL)

            cap.release()
            if lost:
                logger.warning("[camera %s] Mất kết nối", code)
                lost_before = True
                restart.wait(float(CAMERA_RETRY_DELAY_SEC))

        with self._cam_lock:
            if self.camera_workers.get(code) is worker and stop.is_set():
                self.camera_status.pop(code, None)

    def stream_video(self, video_path: str, camera_id: str = "video"):
        """Đọc và xử lý một file video đã lưu.

        Frame sẽ được lấy mẫu theo FPS nguồn và FRAME_INTERVAL để giảm tải.
        """
        # Resolve path: cho phép truyền tên file nếu video nằm trong thư mục video/
        resolved_path = video_path
        if not os.path.isabs(resolved_path):
            resolved_path = os.path.normpath(resolved_path)

        if not os.path.exists(resolved_path):
            candidates = [
                # ai_service/video/<file>
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "video", video_path),
                # <workspace_root>/video/<file>
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "video", video_path),
                # <workspace_root>/<file>
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), video_path),
            ]
            for cand in candidates:
                cand = os.path.normpath(cand)
                if os.path.exists(cand):
                    resolved_path = cand
                    break

        if not os.path.exists(resolved_path):
            raise FileNotFoundError(
                f"Không tìm thấy video: {video_path}. "
                f"Đã thử các vị trí phổ biến như video/{video_path}."
            )

        logger.info("Bắt đầu xử lý video: %s (camera_id=%s)", resolved_path, camera_id)
        cap = cv2.VideoCapture(resolved_path)
        if not cap.isOpened():
            raise RuntimeError(f"Không thể mở video: {resolved_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0

        process_every = max(1, int(round(float(fps) * float(FRAME_INTERVAL))))
        logger.info("Video FPS=%.2f, FRAME_INTERVAL=%.2fs → xử lý mỗi %d frame", fps, FRAME_INTERVAL, process_every)

        frame_idx = 0
        processed = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_idx += 1
                if frame_idx % process_every != 0:
                    continue

                try:
                    self.process_frame(frame, camera_id)
                    processed += 1
                except Exception as e:
                    logger.error("Lỗi xử lý frame video (%s): %s", camera_id, e)

        finally:
            cap.release()

        logger.info("Kết thúc video: total_frames=%d, processed_frames=%d", frame_idx, processed)

    def run_video(self, video_path: str, camera_id: str = "video"):
        """Chạy AI Service trên video offline (không mở RTSP)."""
        logger.info("=" * 60)
        logger.info("AI SERVICE - Nhận diện điểm danh (VIDEO OFFLINE)")
        logger.info("=" * 60)

        self.mode = "video"
        self.track_states = {}
        self.init_model()
        self.load_known_faces()
        self.running = True

        try:
            self.stream_video(video_path=video_path, camera_id=camera_id)
        finally:
            self.running = False

        logger.info(
            "[VIDEO SUMMARY] processed=%d, frames_with_face=%d, frames_with_emb=%d, matches=%d, best_sim=%.3f (%s)",
            self._dbg_frames_processed,
            self._dbg_frames_with_face,
            self._dbg_frames_with_embedding,
            self._dbg_frames_with_match,
            float(self._dbg_best_sim),
            self._dbg_best_name or "--",
        )

    def run(self):
        """Khởi chạy AI Service."""
        logger.info("=" * 60)
        logger.info("AI SERVICE - Nhận diện điểm danh")
        logger.info("=" * 60)

        # 1. Load model
        self.track_states = {}
        self.init_model()

        # 2. Load danh sách khuôn mặt
        self.load_known_faces()

        # 3. Reload embeddings định kỳ (mỗi 5 phút)
        def reload_loop():
            while self.running:
                time.sleep(300)
                self.load_known_faces()

        self.running = True

        reload_thread = threading.Thread(target=reload_loop, daemon=True)
        reload_thread.start()

        # 4. Lấy danh sách camera do admin cấu hình trên web; Flask chưa chạy → dùng mặc định trong config
        if not self._sync_with_server():
            logger.warning("Dùng danh sách camera mặc định trong config.py (chờ Flask để nhận cấu hình từ web)")
            self._apply_camera_list([
                {"code": str(c["id"]), "name": c.get("name"), "rtsp_url": c["rtsp_url"]} for c in CAMERAS
            ])

        # 5. Đồng bộ trạng thái / cấu hình / lệnh khởi động lại với Flask mỗi 2 giây
        def sync_loop():
            while self.running:
                time.sleep(2)
                self._sync_with_server()

        threading.Thread(target=sync_loop, daemon=True, name="camera-sync").start()

        # Giữ main thread chạy
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Đang dừng AI Service...")
            self.running = False
            with self._cam_lock:
                workers = list(self.camera_workers.values())
            for w in workers:
                w["stop"].set()
            for w in workers:
                w["thread"].join(timeout=5)
            logger.info("AI Service đã dừng.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Service - RTSP hoặc video offline")
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Đường dẫn video (mp4/avi/...) để chạy offline. Nếu không truyền thì chạy RTSP theo cấu hình CAMERAS.",
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        default="video",
        help="camera_id gắn vào log/kết quả API khi chạy video offline (mặc định: video)",
    )
    args = parser.parse_args()

    service = AIService()
    if args.video:
        service.run_video(video_path=args.video, camera_id=args.camera_id)
    else:
        service.run()
