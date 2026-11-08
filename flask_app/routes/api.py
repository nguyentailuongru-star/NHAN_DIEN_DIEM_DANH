"""API routes - nhận dữ liệu từ AI Service, quản lý users."""

import os
import json
import sys
import io
import calendar
import threading
import subprocess
import time
import uuid
import shutil
import re
import unicodedata
from pathlib import Path
import numpy as np
from datetime import datetime, date, timedelta
from flask import Blueprint, request, jsonify, current_app, send_file
from werkzeug.utils import secure_filename
from config import Config
from models import (
    db,
    User,
    AttendanceLog,
    AttentionLog,
    Department,
    FaceUploadRequest,
    TeacherNotebookEntry,
    TeacherNotebookAttendanceMark,
)
from auth_utils import api_admin_required, api_employee_required, current_account

api_bp = Blueprint("api", __name__, url_prefix="/api")

DEFAULT_CAMERA_SOURCE = Config.DEFAULT_CAMERA_SOURCE


# ──────────────────────────────────────────────
# Paths / helpers
# ──────────────────────────────────────────────
# Workspace root (chứa flask_app/ và ai_service/)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AI_SERVICE_DIR = os.path.join(REPO_ROOT, "ai_service")
VENV_PY = os.path.join(REPO_ROOT, ".venv", "Scripts", "python.exe")


def _python_executable() -> str:
    # Ưu tiên .venv của workspace để chắc chắn có opencv/onnxruntime/models
    if os.path.exists(VENV_PY):
        return VENV_PY
    return sys.executable


def _ensure_ai_service_importable():
    if AI_SERVICE_DIR not in sys.path:
        sys.path.insert(0, AI_SERVICE_DIR)


def _video_root_dir() -> str:
    # video/ nằm ở root workspace (cùng cấp ai_service, flask_app)
    return os.path.join(REPO_ROOT, "video")


def _captures_roots() -> list[str]:
    return [
        os.path.join(REPO_ROOT, "captures"),
        os.path.join(REPO_ROOT, "ai_service", "captures"),
    ]


def _normalize_log_image_path(image_value) -> str | None:
    """Chuẩn hóa đường dẫn ảnh crop trong log về relative path an toàn."""
    raw = str(image_value or "").strip()
    if not raw:
        return None

    rel = raw.replace("\\", "/")
    if rel.startswith("captures/"):
        rel = rel[len("captures/") :]
    rel = rel.lstrip("/")

    p = Path(rel)
    if p.is_absolute() or ".." in p.parts:
        return None
    if rel in {"", "."}:
        return None
    return rel


def _delete_log_capture_images(images: list[str]) -> dict:
    deleted = 0
    missing = 0
    skipped = 0

    unique_paths = set(images)
    roots = _captures_roots()

    for img in unique_paths:
        rel = _normalize_log_image_path(img)
        if not rel:
            skipped += 1
            continue

        found = False
        for root in roots:
            full = os.path.join(root, rel)
            if os.path.isfile(full):
                try:
                    os.remove(full)
                    deleted += 1
                    found = True
                    break
                except Exception:
                    skipped += 1
                    found = True
                    break

        if not found:
            # Fallback: một số bản ghi cũ chỉ lưu filename
            base = os.path.basename(rel)
            if base and base != rel:
                for root in roots:
                    full = os.path.join(root, base)
                    if os.path.isfile(full):
                        try:
                            os.remove(full)
                            deleted += 1
                            found = True
                            break
                        except Exception:
                            skipped += 1
                            found = True
                            break

        if not found:
            missing += 1

    return {
        "requested": len(unique_paths),
        "deleted": deleted,
        "missing": missing,
        "skipped": skipped,
    }


def _safe_join_video_root(filename: str) -> str:
    root = Path(_video_root_dir()).resolve()
    target = (root / filename).resolve()
    if root not in target.parents and target != root:
        raise ValueError("Invalid video path")
    return str(target)


def _allowed_ext(name: str, exts: set[str]) -> bool:
    return os.path.splitext(name.lower())[1] in exts


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    denom = float(np.linalg.norm(vec))
    if denom <= 0:
        return vec
    return vec / denom


# ──────────────────────────────────────────────
# Behavior model (lazy, video-only analysis)
# ──────────────────────────────────────────────
_BEHAVIOR_MODEL = None
_BEHAVIOR_MODEL_LOCK = threading.Lock()
_BEHAVIOR_CLASS_NAMES = ["distracted", "focused", "raising_hand", "sleep", "using_phone"]


def _ascii_slug(value: str) -> str:
    txt = unicodedata.normalize("NFKD", str(value or ""))
    txt = txt.encode("ascii", "ignore").decode("ascii")
    txt = txt.lower()
    txt = re.sub(r"[^a-z0-9]+", "_", txt).strip("_")
    return txt or "unknown"


def _behavior_model_path() -> str:
    return os.path.join(AI_SERVICE_DIR, "models", "student_behavior_best.pt")


def _get_behavior_model():
    global _BEHAVIOR_MODEL
    if _BEHAVIOR_MODEL is not None:
        return _BEHAVIOR_MODEL

    with _BEHAVIOR_MODEL_LOCK:
        if _BEHAVIOR_MODEL is not None:
            return _BEHAVIOR_MODEL

        _ensure_ai_service_importable()
        model_path = _behavior_model_path()
        if not os.path.isfile(model_path):
            raise RuntimeError(f"Không tìm thấy model hành vi: {model_path}")

        try:
            from ultralytics import YOLO  # type: ignore[import-not-found]
        except Exception as e:
            raise RuntimeError(
                "Không import được ultralytics. Hãy cài dependencies trong môi trường Flask. "
                f"Chi tiết: {str(e).strip()}"
            )

        _BEHAVIOR_MODEL = YOLO(model_path)
        return _BEHAVIOR_MODEL


def _behavior_label_from_id(class_id: int) -> str:
    behavior_names = _behavior_class_names()
    if 0 <= int(class_id) < len(behavior_names):
        return behavior_names[int(class_id)]
    return f"class_{int(class_id)}"


def _behavior_class_names() -> list[str]:
    model = _get_behavior_model()
    names = getattr(model, "names", None)

    if isinstance(names, dict) and names:
        return [str(names[idx]) for idx in sorted(names)]

    if isinstance(names, (list, tuple)) and names:
        return [str(item) for item in names]

    return list(_BEHAVIOR_CLASS_NAMES)


# ──────────────────────────────────────────────
# Face embedding (lazy model)
# ──────────────────────────────────────────────
_FACE_APP = None
_FACE_APP_LOCK = threading.Lock()
_FACE_INFER_LOCK = threading.Lock()

_HAAR_CASCADE = None
_HAAR_LOCK = threading.Lock()


def _get_haar_cascade(cv2):
    global _HAAR_CASCADE
    if _HAAR_CASCADE is not None:
        return _HAAR_CASCADE
    with _HAAR_LOCK:
        if _HAAR_CASCADE is not None:
            return _HAAR_CASCADE
        try:
            base = getattr(cv2, "data", None)
            haar_dir = getattr(base, "haarcascades", None) if base is not None else None
            if not haar_dir:
                _HAAR_CASCADE = None
                return None
            xml_path = os.path.join(haar_dir, "haarcascade_frontalface_default.xml")
            if not os.path.exists(xml_path):
                _HAAR_CASCADE = None
                return None
            cascade = cv2.CascadeClassifier(xml_path)
            # empty() == True means load failed
            if cascade is None or cascade.empty():
                _HAAR_CASCADE = None
                return None
            _HAAR_CASCADE = cascade
            return _HAAR_CASCADE
        except Exception:
            _HAAR_CASCADE = None
            return None


def _get_face_app():
    global _FACE_APP
    if _FACE_APP is not None:
        return _FACE_APP

    with _FACE_APP_LOCK:
        if _FACE_APP is not None:
            return _FACE_APP

        _ensure_ai_service_importable()
        try:
            from face_analysis_local import FaceAnalysisLocal
        except Exception as e:
            raise RuntimeError(
                "Không import được module AI (face_analysis_local). "
                "Hãy chạy Flask bằng Python trong .venv (run_flask.bat) và đảm bảo đã chạy setup.bat. "
                f"Chi tiết: {str(e).strip()}"
            )

        from model_store import ensure_face_model

        # Model cài trên máy (%LOCALAPPDATA%\NhanDienDiemDanh\models), tự tải nếu chưa có
        model_dir = ensure_face_model()
        # ctx_id=0: cố gắng dùng GPU nếu cấu hình CUDA OK; nếu không sẽ fallback CPU (đã patch trong face_analysis_local)
        _FACE_APP = FaceAnalysisLocal(model_dir=model_dir, ctx_id=0)
        # Upload ảnh đăng ký: hạ ngưỡng detect một chút để dễ bắt khuôn mặt hơn
        _FACE_APP.prepare(ctx_id=0, det_size=(640, 640), det_thresh=0.35)
        return _FACE_APP


def _embedding_from_uploaded_images(files) -> tuple[list[float] | None, int, dict]:
    """Trả về (embedding_list, used_count, stats)."""
    if not files:
        return None, 0, {"total": 0}

    _ensure_ai_service_importable()
    try:
        import cv2
    except Exception as e:
        raise RuntimeError(
            "Thiếu OpenCV (cv2). Hãy chạy Flask bằng Python trong .venv (run_flask.bat) và cài requirements. "
            f"Chi tiết: {str(e).strip()}"
        )

    app = _get_face_app()
    embeddings: list[np.ndarray] = []

    def _clahe_bgr(img_bgr: np.ndarray) -> np.ndarray:
        try:
            lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l2 = clahe.apply(l)
            merged = cv2.merge((l2, a, b))
            return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        except Exception:
            return img_bgr

    def _pad_replicate(img_bgr: np.ndarray, pad_ratio: float) -> np.ndarray:
        if not pad_ratio or pad_ratio <= 0:
            return img_bgr
        h0, w0 = img_bgr.shape[:2]
        pad = int(round(min(h0, w0) * float(pad_ratio)))
        if pad <= 0:
            return img_bgr
        return cv2.copyMakeBorder(img_bgr, pad, pad, pad, pad, borderType=cv2.BORDER_REPLICATE)

    def _adaptive_upscale(img_bgr: np.ndarray, target_max: int, max_out: int = 2400) -> tuple[np.ndarray, float]:
        h0, w0 = img_bgr.shape[:2]
        m0 = max(h0, w0)
        if m0 <= 0:
            return img_bgr, 1.0
        if m0 >= target_max:
            return img_bgr, 1.0
        scale = float(target_max) / float(m0)
        # Cap output to avoid very slow detection on CPU
        if m0 * scale > max_out:
            scale = float(max_out) / float(m0)
        if scale <= 1.01:
            return img_bgr, 1.0
        out = cv2.resize(img_bgr, (int(round(w0 * scale)), int(round(h0 * scale))), interpolation=cv2.INTER_LINEAR)
        return out, scale

    def _crop_windows(img_bgr: np.ndarray, ratio: float) -> list[tuple[np.ndarray, dict]]:
        """Tạo các crop kiểu 'zoom-in' (giữa + 4 góc) để bắt mặt khi mặt quá nhỏ."""
        if not ratio or ratio <= 0 or ratio >= 1:
            return [(img_bgr, {"crop": "full", "ratio": 1.0})]

        h0, w0 = img_bgr.shape[:2]
        cw = max(64, int(round(w0 * ratio)))
        ch = max(64, int(round(h0 * ratio)))
        cw = min(cw, w0)
        ch = min(ch, h0)

        def _clip(x, lo, hi):
            return max(lo, min(int(x), hi))

        xs = [0, w0 - cw, (w0 - cw) // 2]
        ys = [0, h0 - ch, (h0 - ch) // 2]

        windows = []
        # corners + center
        points = [
            (xs[0], ys[0], "tl"),
            (xs[1], ys[0], "tr"),
            (xs[0], ys[1], "bl"),
            (xs[1], ys[1], "br"),
            (xs[2], ys[2], "center"),
        ]

        seen = set()
        for x0, y0, name in points:
            x0 = _clip(x0, 0, w0 - cw)
            y0 = _clip(y0, 0, h0 - ch)
            key = (x0, y0)
            if key in seen:
                continue
            seen.add(key)
            crop = img_bgr[y0 : y0 + ch, x0 : x0 + cw]
            windows.append((crop, {"crop": name, "ratio": float(ratio), "x": int(x0), "y": int(y0), "w": int(cw), "h": int(ch)}))

        return windows

    def _haar_best_bbox(img_bgr: np.ndarray):
        """Return best bbox [x1,y1,x2,y2] using Haar cascade or None."""
        cascade = _get_haar_cascade(cv2)
        if cascade is None:
            return None, {"haar_available": False}

        try:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            rects = cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=3,
                flags=getattr(cv2, "CASCADE_SCALE_IMAGE", 0),
                minSize=(48, 48),
            )
            if rects is None or len(rects) == 0:
                return None, {"haar_available": True, "haar_faces": 0}

            # pick the largest face
            rects = list(rects)
            rects.sort(key=lambda r: int(r[2]) * int(r[3]), reverse=True)
            x, y, rw, rh = [int(v) for v in rects[0]]
            return [x, y, x + rw, y + rh], {"haar_available": True, "haar_faces": int(len(rects))}
        except Exception as e:
            return None, {"haar_available": True, "haar_error": str(e).strip()}

    stats = {
        "total": 0,
        "decoded": 0,
        "no_face": 0,
        "no_face_after_retry": 0,
        "no_embedding": 0,
        "used": 0,
        "retried_upscale": 0,
    }

    for f in files:
        if not f or not getattr(f, "filename", ""):
            continue

        stats["total"] += 1

        raw = f.read()
        if not raw:
            continue

        img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue

        stats["decoded"] += 1

        h, w = img.shape[:2]

        # Luôn lưu shape gốc để debug
        stats["last_image_shape"] = {"w": int(w), "h": int(h)}

        # Thử nhiều cấu hình detect để tăng khả năng bắt mặt (ảnh có thể nhỏ / crop sát / ngược sáng)
        attempts = [
            # attempt 1: nhẹ nhất
            {"det_size": (640, 640), "det_thresh": 0.35, "target_max": 0, "pad_ratio": 0.0, "clahe": False},
            # attempt 2: padding + upscale vừa
            {"det_size": (1024, 1024), "det_thresh": 0.25, "target_max": 1200, "pad_ratio": 0.16, "clahe": False},
            # attempt 3: padding + upscale mạnh + tăng tương phản nhẹ
            {"det_size": (1024, 1024), "det_thresh": 0.12, "target_max": 1600, "pad_ratio": 0.22, "clahe": True},
        ]

        faces = []
        used_attempt = None

        # Bảo vệ việc đổi tham số detector (global app)
        with _FACE_INFER_LOCK:
            prev_size = tuple(getattr(app.detector, "input_size", (640, 640)))
            prev_thresh = float(getattr(app.detector, "det_thresh", 0.35))

            try:
                def _run_attempts(img_in: np.ndarray, extra: dict | None = None):
                    tried_local = []
                    faces_local = []
                    used_local = None
                    for a in attempts:
                        det_size = a["det_size"]
                        det_thresh = a["det_thresh"]
                        target_max = int(a.get("target_max") or 0)
                        pad_ratio = float(a.get("pad_ratio") or 0.0)
                        do_clahe = bool(a.get("clahe"))

                        try:
                            app.detector.input_size = det_size
                            app.detector.det_thresh = det_thresh
                        except Exception:
                            pass

                        img_try = img_in
                        if pad_ratio and pad_ratio > 0:
                            img_try = _pad_replicate(img_try, pad_ratio)

                        scale = 1.0
                        if target_max and target_max > 0:
                            img_try, scale = _adaptive_upscale(img_try, target_max=target_max)
                            if scale > 1.01:
                                stats["retried_upscale"] += 1

                        if do_clahe:
                            img_try = _clahe_bgr(img_try)

                        attempt_meta = {
                            "det_size": det_size,
                            "det_thresh": det_thresh,
                            "pad_ratio": pad_ratio,
                            "target_max": target_max,
                            "scale": float(scale),
                            "clahe": do_clahe,
                            "img_try_shape": {"w": int(img_try.shape[1]), "h": int(img_try.shape[0])},
                        }
                        if extra:
                            attempt_meta.update(extra)
                        tried_local.append(attempt_meta)

                        faces_local = app.get(img_try)
                        if faces_local:
                            used_local = attempt_meta
                            break
                    return faces_local, used_local, tried_local

                tried = []
                faces, used_attempt, tried_full = _run_attempts(img)
                tried.extend(tried_full)

                # Fallback: nếu ảnh là screenshot/frame mà mặt quá nhỏ, thử zoom-in qua crop windows
                if not faces:
                    stats["crop_search_used"] = True
                    crop_ratio = 0.65 if max(h, w) >= 900 else 0.75
                    crop_windows = _crop_windows(img, crop_ratio)
                    stats["crop_windows"] = [m for _, m in crop_windows]

                    for crop_img, crop_meta in crop_windows:
                        faces, used_attempt, tried_crop = _run_attempts(crop_img, extra={"crop": crop_meta})
                        tried.extend(tried_crop)
                        if faces:
                            stats["crop_search_success"] = crop_meta
                            break
            finally:
                # Restore defaults
                try:
                    app.detector.input_size = prev_size
                    app.detector.det_thresh = prev_thresh
                except Exception:
                    pass

        if not faces:
            # Fallback cuối: Haar Cascade (bbox) + embedding từ crop (không cần keypoints)
            try:
                img_haar = _pad_replicate(img, 0.35)
                img_haar, haar_scale = _adaptive_upscale(img_haar, target_max=900)
                bbox, haar_info = _haar_best_bbox(img_haar)
                stats.update(haar_info)
                stats["haar_input_shape"] = {"w": int(img_haar.shape[1]), "h": int(img_haar.shape[0])}
                stats["haar_upscale"] = float(haar_scale)

                if bbox is not None:
                    with _FACE_INFER_LOCK:
                        emb = getattr(app, "recognizer", None)
                        if emb is not None and hasattr(emb, "get_embedding_from_bbox"):
                            vec = emb.get_embedding_from_bbox(img_haar, bbox, margin=0.30)
                        else:
                            vec = None

                    if vec is not None:
                        emb_arr = np.asarray(vec, dtype=np.float32)
                        embeddings.append(_l2_normalize(emb_arr))
                        stats["used"] += 1
                        stats["haar_used"] = True
                        stats["haar_bbox"] = {"x1": int(bbox[0]), "y1": int(bbox[1]), "x2": int(bbox[2]), "y2": int(bbox[3])}
                        continue
                    else:
                        stats["no_embedding"] += 1
            except Exception as e:
                stats["haar_fallback_error"] = str(e).strip()

            stats["no_face"] += 1
            stats["no_face_after_retry"] += 1
            try:
                stats["attempts_tried"] = tried
            except Exception:
                pass
            continue

        if used_attempt:
            stats["last_used_attempt"] = used_attempt

        # Lấy embedding của khuôn mặt score cao nhất
        best = None
        best_score = -1.0
        for face in faces:
            if face.embedding is None:
                continue
            score = float(getattr(face, "det_score", 0.0) or 0.0)
            if score > best_score:
                best = face
                best_score = score

        if best is None or best.embedding is None:
            stats["no_embedding"] += 1
            continue

        emb = np.asarray(best.embedding, dtype=np.float32)
        embeddings.append(_l2_normalize(emb))
        stats["used"] += 1

    if not embeddings:
        return None, 0, stats

    mean_emb = _l2_normalize(np.mean(np.stack(embeddings, axis=0), axis=0))
    return mean_emb.astype(np.float32).tolist(), len(embeddings), stats


# ──────────────────────────────────────────────
# Video runner (subprocess)
# ──────────────────────────────────────────────
_VIDEO_PROC = None
_VIDEO_PROC_LOCK = threading.Lock()
_VIDEO_LOGS = None

_GROUP_JOB_LOCK = threading.Lock()
_GROUP_JOBS = {
    "camera": None,
    "video": None,
}

_GROUP_MIN_VISIBLE_FACE_RATIO = 2.0 / 3.0
_GROUP_MIN_LAPLACIAN_VAR = 55.0
_GROUP_MIN_FACE_WIDTH = 58.0
_GROUP_MIN_FACE_HEIGHT = 58.0
_GROUP_MIN_DET_SCORE = 0.30
_GROUP_MAX_NOSE_OFFSET_RATIO = 0.78
_GROUP_MAX_MOUTH_OFFSET_RATIO = 0.90
_GROUP_MIN_EYE_DISTANCE = 12.0


def _video_status():
    global _VIDEO_PROC
    global _VIDEO_LOGS
    if _VIDEO_PROC is None:
        st = {"running": False}
        if _VIDEO_LOGS:
            st.update(_VIDEO_LOGS)
        return st
    code = _VIDEO_PROC.poll()
    if code is None:
        st = {"running": True, "pid": _VIDEO_PROC.pid}
        if _VIDEO_LOGS:
            st.update(_VIDEO_LOGS)
        return st
    st = {"running": False, "exit_code": code}
    if _VIDEO_LOGS:
        st.update(_VIDEO_LOGS)
    return st


# ──────────────────────────────────────────────
# Recognition endpoint (AI Service gọi vào đây)
# ──────────────────────────────────────────────
_RECOGNITION_LOCK = threading.Lock()


@api_bp.route("/recognition", methods=["POST"])
def recognition():
    """Nhận kết quả nhận diện từ AI Service."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data"}), 400

    user_id = data.get("user_id")
    user_name = data.get("name", "Unknown")
    camera_id = data.get("camera_id", "")
    image = data.get("image", "")
    similarity = data.get("similarity", 0)
    log_time = data.get("time")

    if log_time:
        log_time = datetime.strptime(log_time, "%Y-%m-%d %H:%M:%S")
    else:
        log_time = datetime.now()

    cooldown = int(current_app.config.get("ATTENDANCE_COOLDOWN_SECONDS", 300))
    with _RECOGNITION_LOCK:
        # Bỏ qua nếu người này vừa được điểm danh trong khoảng cooldown (mọi camera / mọi tiến trình AI).
        if user_id is not None and cooldown > 0:
            last = (
                AttendanceLog.query.filter(
                    AttendanceLog.user_id == user_id,
                    AttendanceLog.time > log_time - timedelta(seconds=cooldown),
                    AttendanceLog.time <= log_time + timedelta(seconds=cooldown),
                )
                .order_by(AttendanceLog.time.desc())
                .first()
            )
            if last is not None:
                wait = cooldown - int(abs((log_time - last.time).total_seconds()))
                return jsonify({
                    "status": "skipped",
                    "reason": "cooldown",
                    "last_log_id": last.id,
                    "retry_after": max(wait, 0),
                })

        log = AttendanceLog(
            user_id=user_id,
            user_name=user_name,
            camera_id=camera_id,
            time=log_time,
            image=image,
            similarity=similarity,
        )
        db.session.add(log)
        db.session.commit()

    # Emit realtime event qua SocketIO
    try:
        sock = current_app.extensions.get("socketio")
        if sock is not None and hasattr(sock, "emit"):
            sock.emit("new_recognition", log.to_dict())
    except Exception:
        # Không để lỗi SocketIO làm hỏng API (vẫn lưu log vào DB bình thường)
        pass

    return jsonify({"status": "ok", "log_id": log.id})


@api_bp.route("/attention", methods=["POST"])
def attention():
    """Nhận dữ liệu đánh giá hành vi trong lớp từ AI Service.

    Ghi vào attention_logs để tương thích DB cũ:
    - yaw: behavior_class_id
    - attention_score: behavior_confidence
    - attentive: engaged
    - source: sb:<class_id>:<label>
    """
    try:
        behavior_names = _behavior_class_names()
    except Exception:
        behavior_names = list(_BEHAVIOR_CLASS_NAMES)

    def _normalize_label(raw_label, class_id):
        if isinstance(raw_label, str) and raw_label.strip():
            label = raw_label.strip()
            for item in behavior_names:
                if item.lower() == label.lower():
                    return item
        if class_id is not None and 0 <= class_id < len(behavior_names):
            return behavior_names[class_id]
        return "Unknown"

    def _pack_source(label, class_id):
        safe_id = class_id if class_id is not None else -1
        safe_label = str(label or "Unknown").replace(":", "_")
        return f"sb:{safe_id}:{safe_label}"[:30]

    data = request.get_json(silent=True) or {}
    if not data:
        return jsonify({"error": "No data"}), 400

    user_id_raw = data.get("user_id")
    user_id = None
    if user_id_raw is not None and str(user_id_raw).strip() != "":
        try:
            user_id = int(user_id_raw)
        except Exception:
            user_id = None

    if user_id is not None:
        exists = User.query.with_entities(User.id).filter_by(id=user_id).first()
        if not exists:
            user_id = None

    user_name = data.get("name", "Unknown")
    camera_id = data.get("camera_id", "")

    class_id_raw = data.get("behavior_class_id")
    class_id = None
    if class_id_raw is not None and str(class_id_raw).strip() != "":
        try:
            class_id = int(class_id_raw)
        except Exception:
            class_id = None

    label = _normalize_label(data.get("behavior_label"), class_id)
    source = _pack_source(label, class_id)
    log_time = data.get("time")

    try:
        yaw = float(class_id) if class_id is not None else None
    except Exception:
        yaw = None

    pitch = None
    roll = None

    try:
        confidence_raw = data.get("behavior_confidence", data.get("attention_score"))
        attention_score = float(confidence_raw) if confidence_raw is not None else None
    except Exception:
        attention_score = None

    attentive_raw = data.get("engaged", data.get("attentive", False))
    if isinstance(attentive_raw, str):
        attentive = attentive_raw.strip().lower() in {"1", "true", "yes", "on"}
    else:
        attentive = bool(attentive_raw)

    if log_time:
        try:
            log_time = datetime.strptime(log_time, "%Y-%m-%d %H:%M:%S")
        except Exception:
            log_time = datetime.now()
    else:
        log_time = datetime.now()

    image_file = data.get("image")
    if image_file and isinstance(image_file, str):
        image_file = image_file.strip() or None

    log = AttentionLog(
        user_id=user_id,
        user_name=user_name,
        camera_id=camera_id,
        time=log_time,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        attention_score=attention_score,
        attentive=attentive,
        source=source,
        image=image_file,
    )
    db.session.add(log)
    db.session.commit()

    return jsonify({"status": "ok", "attention_log_id": log.id})


@api_bp.route("/attention/logs", methods=["GET"])
@api_admin_required()
def get_attention_logs():
    """Lấy lịch sử attention logs theo bộ lọc."""
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 50, type=int)
    date_str = request.args.get("date")
    user_id = request.args.get("user_id", type=int)
    camera_id = (request.args.get("camera_id") or "").strip()

    query = AttentionLog.query

    if date_str:
        try:
            date_value = datetime.strptime(date_str, "%Y-%m-%d")
            query = query.filter(db.func.date(AttentionLog.time) == date_value.date())
        except Exception:
            pass

    if user_id:
        query = query.filter(AttentionLog.user_id == user_id)

    if camera_id:
        query = query.filter(AttentionLog.camera_id == camera_id)

    pagination = query.order_by(AttentionLog.time.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )

    return jsonify({
        "logs": [item.to_dict() for item in pagination.items],
        "total": pagination.total,
        "page": pagination.page,
        "pages": pagination.pages,
    })


@api_bp.route("/attention/logs/all", methods=["DELETE"])
@api_admin_required()
def delete_all_attention_logs():
    """Xóa toàn bộ log hành vi (attention_logs)."""
    total_logs = AttentionLog.query.count()

    try:
        AttentionLog.query.delete(synchronize_session=False)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Không thể xóa attention logs: {str(e).strip()}"}), 500

    return jsonify({
        "status": "ok",
        "deleted_logs": int(total_logs),
    })


@api_bp.route("/attention/logs/user/<int:user_id>", methods=["DELETE"])
@api_admin_required()
def delete_attention_logs_by_user(user_id):
    """Xóa toàn bộ log hành vi của một user."""
    logs = AttentionLog.query.filter(AttentionLog.user_id == user_id).all()
    total_logs = len(logs)

    if total_logs <= 0:
        return jsonify({
            "status": "ok",
            "deleted_logs": 0,
            "user_id": int(user_id),
        })

    try:
        for item in logs:
            db.session.delete(item)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Không thể xóa attention logs theo user: {str(e).strip()}"}), 500

    return jsonify({
        "status": "ok",
        "deleted_logs": int(total_logs),
        "user_id": int(user_id),
    })


@api_bp.route("/attention/stats", methods=["GET"])
@api_admin_required()
def get_attention_stats():
    """Thống kê hành vi lớp học theo thời gian, user và camera."""
    behavior_names = ["distracted", "focused", "raising_hand", "sleep", "using_phone"]

    def _behavior_label_from_source(src, yaw_value):
        source = str(src or "").strip()
        if source.startswith("sb:"):
            parts = source.split(":", 2)
            if len(parts) == 3 and parts[2]:
                return parts[2]
        try:
            idx = int(yaw_value) if yaw_value is not None else -1
            if 0 <= idx < len(behavior_names):
                return behavior_names[idx]
        except Exception:
            pass
        return "Unknown"

    start_date_str = (request.args.get("start_date") or "").strip()
    end_date_str = (request.args.get("end_date") or "").strip()
    department_id = request.args.get("department_id", type=int)
    camera_id = (request.args.get("camera_id") or "").strip()

    today = datetime.now().date()
    default_start = today - timedelta(days=6)

    try:
        start_day = datetime.strptime(start_date_str, "%Y-%m-%d").date() if start_date_str else default_start
    except Exception:
        start_day = default_start

    try:
        end_day = datetime.strptime(end_date_str, "%Y-%m-%d").date() if end_date_str else today
    except Exception:
        end_day = today

    if start_day > end_day:
        start_day, end_day = end_day, start_day

    start_dt = datetime.combine(start_day, datetime.min.time())
    end_dt = datetime.combine(end_day, datetime.max.time())

    base_query = AttentionLog.query.outerjoin(User, AttentionLog.user_id == User.id)
    base_query = base_query.filter(AttentionLog.time >= start_dt, AttentionLog.time <= end_dt)

    if department_id:
        base_query = base_query.filter(User.department_id == department_id)

    if camera_id:
        base_query = base_query.filter(AttentionLog.camera_id == camera_id)

    summary_row = base_query.with_entities(
        db.func.count(AttentionLog.id),
        db.func.avg(db.cast(AttentionLog.attentive, db.Float)),
        db.func.avg(AttentionLog.attention_score),
    ).first()

    total_samples = int(summary_row[0] or 0)
    overall_engagement_rate = float((summary_row[1] or 0.0) * 100.0)
    overall_confidence = float((summary_row[2] or 0.0) * 100.0)

    timeline_rows = base_query.with_entities(
        db.func.date(AttentionLog.time).label("bucket_day"),
        db.func.count(AttentionLog.id).label("samples"),
        db.func.avg(db.cast(AttentionLog.attentive, db.Float)).label("rate"),
        db.func.avg(AttentionLog.attention_score).label("score"),
    ).group_by(db.func.date(AttentionLog.time)).order_by(db.func.date(AttentionLog.time).asc()).all()

    timeline = []
    for row in timeline_rows:
        day_value = row.bucket_day
        label = day_value.strftime("%d/%m/%Y") if hasattr(day_value, "strftime") else str(day_value)
        timeline.append({
            "day": day_value.strftime("%Y-%m-%d") if hasattr(day_value, "strftime") else str(day_value),
            "label": label,
            "samples": int(row.samples or 0),
            "engagement_rate": float((row.rate or 0.0) * 100.0),
            "behavior_confidence": float((row.score or 0.0) * 100.0),
        })

    # Group by user_id, not User.id - more reliable, ignores NULL User joins
    user_rows = base_query.with_entities(
        AttentionLog.user_id.label("uid"),
        db.func.coalesce(User.name, AttentionLog.user_name, "Unknown").label("uname"),
        db.func.count(AttentionLog.id).label("samples"),
        db.func.sum(db.cast(AttentionLog.attentive, db.Integer)).label("engaged_samples"),
        db.func.avg(db.cast(AttentionLog.attentive, db.Float)).label("rate"),
        db.func.avg(AttentionLog.attention_score).label("score"),
        db.func.max(AttentionLog.time).label("last_seen"),
    ).group_by(AttentionLog.user_id).order_by(db.func.count(AttentionLog.id).desc()).limit(20).all()

    by_user = []
    behavior_counter = {name: 0 for name in behavior_names}
    for row in user_rows:
        engagement_rate = float((row.rate or 0.0) * 100.0)
        confidence = float((row.score or 0.0) * 100.0)
        last_seen = row.last_seen.strftime("%Y-%m-%d %H:%M:%S") if row.last_seen else None

        last_image = None
        try:
            last_log = AttentionLog.query.filter(
                AttentionLog.user_id == row.uid,
                AttentionLog.image.isnot(None)
            ).order_by(AttentionLog.time.desc()).first()
            if last_log:
                last_image = last_log.image
        except Exception:
            pass

        user_behavior_rows = base_query.filter(AttentionLog.user_id == row.uid).with_entities(
            AttentionLog.source,
            AttentionLog.yaw,
            db.func.count(AttentionLog.id).label("samples"),
        ).group_by(AttentionLog.source, AttentionLog.yaw).all()

        per_behavior = []
        for b_row in user_behavior_rows:
            b_label = _behavior_label_from_source(b_row.source, b_row.yaw)
            b_count = int(b_row.samples or 0)
            per_behavior.append({"label": b_label, "samples": b_count})
            if b_label in behavior_counter:
                behavior_counter[b_label] += b_count

        per_behavior.sort(key=lambda x: x["samples"], reverse=True)
        dominant_behavior = per_behavior[0]["label"] if per_behavior else "Unknown"

        by_user.append({
            "user_id": row.uid,
            "name": row.uname or "Unknown",
            "samples": int(row.samples or 0),
            "engaged_samples": int(row.engaged_samples or 0),
            "engagement_rate": engagement_rate,
            "behavior_confidence": confidence,
            "dominant_behavior": dominant_behavior,
            "behaviors": per_behavior,
            "last_seen": last_seen,
            "last_image": last_image,
        })

    camera_expr = db.func.coalesce(AttentionLog.camera_id, "unknown")
    camera_rows = base_query.with_entities(
        camera_expr.label("cam"),
        db.func.count(AttentionLog.id).label("samples"),
        db.func.avg(db.cast(AttentionLog.attentive, db.Float)).label("rate"),
        db.func.avg(AttentionLog.attention_score).label("score"),
    ).group_by(camera_expr).order_by(db.func.count(AttentionLog.id).desc()).all()

    by_camera = []
    for row in camera_rows:
        cam_behavior_rows = base_query.filter(AttentionLog.camera_id == row.cam).with_entities(
            AttentionLog.source,
            AttentionLog.yaw,
            db.func.count(AttentionLog.id).label("samples"),
        ).group_by(AttentionLog.source, AttentionLog.yaw).all()

        behavior_dist = []
        for b_row in cam_behavior_rows:
            b_label = _behavior_label_from_source(b_row.source, b_row.yaw)
            behavior_dist.append({
                "label": b_label,
                "samples": int(b_row.samples or 0),
            })
            if b_label in behavior_counter:
                behavior_counter[b_label] += int(b_row.samples or 0)

        behavior_dist.sort(key=lambda x: x["samples"], reverse=True)

        by_camera.append({
            "camera_id": row.cam,
            "samples": int(row.samples or 0),
            "engagement_rate": float((row.rate or 0.0) * 100.0),
            "behavior_confidence": float((row.score or 0.0) * 100.0),
            "behaviors": behavior_dist,
        })

    overall_behavior_rows = base_query.with_entities(
        AttentionLog.source,
        AttentionLog.yaw,
        db.func.count(AttentionLog.id).label("samples"),
    ).group_by(AttentionLog.source, AttentionLog.yaw).all()

    by_behavior = []
    behavior_counter = {name: 0 for name in behavior_names}
    for row in overall_behavior_rows:
        b_label = _behavior_label_from_source(row.source, row.yaw)
        b_count = int(row.samples or 0)
        if b_label not in behavior_counter:
            behavior_counter[b_label] = 0
        behavior_counter[b_label] += b_count

    for label_name, samples in behavior_counter.items():
        ratio = (float(samples) * 100.0 / float(total_samples)) if total_samples > 0 else 0.0
        by_behavior.append({
            "label": label_name,
            "samples": int(samples),
            "ratio": ratio,
        })

    by_behavior.sort(key=lambda x: x["samples"], reverse=True)

    camera_query = AttentionLog.query.outerjoin(User, AttentionLog.user_id == User.id).filter(
        AttentionLog.time >= start_dt,
        AttentionLog.time <= end_dt,
    )
    if department_id:
        camera_query = camera_query.filter(User.department_id == department_id)

    camera_options_rows = camera_query.with_entities(camera_expr.label("cam")).distinct().order_by(camera_expr.asc()).all()
    camera_options = [row.cam for row in camera_options_rows if row.cam]

    return jsonify({
        "filters": {
            "start_date": start_day.strftime("%Y-%m-%d"),
            "end_date": end_day.strftime("%Y-%m-%d"),
            "department_id": department_id,
            "camera_id": camera_id or None,
        },
        "summary": {
            "total_samples": total_samples,
            "overall_engagement_rate": overall_engagement_rate,
            "overall_behavior_confidence": overall_confidence,
        },
        "timeline": timeline,
        "by_user": by_user,
        "by_camera": by_camera,
        "by_behavior": by_behavior,
        "cameras": camera_options,
    })


@api_bp.route("/attention/debug", methods=["GET"])
def debug_attention_data():
    """Debug endpoint - kiểm tra dữ liệu AttentionLog thực tế."""
    # Tổng số AttentionLog records
    total_records = db.session.query(AttentionLog).count()
    
    # Sample records gần nhất
    recent = db.session.query(AttentionLog).order_by(AttentionLog.time.desc()).limit(5).all()
    recent_data = []
    for log in recent:
        recent_data.append({
            "id": log.id,
            "user_id": log.user_id,
            "user_name": log.user_name,
            "camera_id": log.camera_id,
            "time": log.time.isoformat() if log.time else None,
            "yaw": log.yaw,
            "attention_score": log.attention_score,
            "attentive": log.attentive,
            "source": log.source,
            "image": log.image,
        })
    
    # Sample từ 6 ngày trước đến hôm nay
    today = date.today()
    start_day = today - timedelta(days=6)
    start_dt = datetime.combine(start_day, datetime.min.time())
    end_dt = datetime.combine(today, datetime.max.time())
    
    filtered_count = db.session.query(AttentionLog).filter(
        AttentionLog.time >= start_dt,
        AttentionLog.time <= end_dt
    ).count()
    
    # User count với outerjoin User
    user_rows = db.session.query(AttentionLog).outerjoin(
        User, AttentionLog.user_id == User.id
    ).filter(
        AttentionLog.time >= start_dt,
        AttentionLog.time <= end_dt
    ).with_entities(
        User.id,
        User.name,
        db.func.count(AttentionLog.id)
    ).group_by(User.id, User.name).all()
    
    return jsonify({
        "total_records": total_records,
        "recent_5": recent_data,
        "filtered_by_date_count": filtered_count,
        "date_range": {
            "start": start_dt.isoformat(),
            "end": end_dt.isoformat(),
        },
        "user_groups": len(user_rows),
        "user_details": [{"uid": row[0], "name": row[1], "count": row[2]} for row in user_rows],
    })


# ──────────────────────────────────────────────
# Users CRUD
# ──────────────────────────────────────────────
@api_bp.route("/departments", methods=["GET"])
@api_admin_required()
def get_departments():
    departments = Department.query.filter_by(is_active=True).order_by(Department.name.asc()).all()
    return jsonify({"departments": [d.to_dict() for d in departments]})


@api_bp.route("/departments", methods=["POST"])
@api_admin_required()
def create_department():
    data = request.get_json(silent=True) or {}
    name = (data.get("name") or "").strip()
    if not name:
        return jsonify({"error": "Thiếu tên phòng ban"}), 400

    old = Department.query.filter(db.func.lower(Department.name) == name.lower()).first()
    if old and old.is_active:
        return jsonify({"error": "Phòng ban đã tồn tại"}), 409

    if old and not old.is_active:
        old.is_active = True
        old.name = name
        db.session.commit()
        return jsonify({"status": "ok", "department": old.to_dict()})

    dep = Department(name=name, is_active=True)
    db.session.add(dep)
    db.session.commit()
    return jsonify({"status": "ok", "department": dep.to_dict()}), 201


@api_bp.route("/users", methods=["GET"])
@api_admin_required(allow_local=True)
def get_users():
    """Lấy danh sách users."""
    users = User.query.filter_by(is_active=True).all()
    result = []
    for u in users:
        d = u.to_dict()
        d["has_embedding"] = bool(u.embedding)

        # Thumbnail ảnh khuôn mặt đã upload (mới nhất)
        try:
            faces_dir = os.path.join(current_app.config["UPLOAD_FOLDER"], "faces", f"user_{u.id}")
            face_thumb = None
            if os.path.isdir(faces_dir):
                files = []
                for name in os.listdir(faces_dir):
                    full = os.path.join(faces_dir, name)
                    if not os.path.isfile(full):
                        continue
                    ext = os.path.splitext(name.lower())[1]
                    if ext not in {".jpg", ".jpeg", ".png", ".webp"}:
                        continue
                    try:
                        files.append((os.path.getmtime(full), name))
                    except Exception:
                        files.append((0, name))
                if files:
                    files.sort(reverse=True)
                    face_thumb = f"faces/user_{u.id}/{files[0][1]}"
            d["face_thumbnail"] = face_thumb
        except Exception:
            d["face_thumbnail"] = None

        result.append(d)
    return jsonify({"users": result})


@api_bp.route("/users/embeddings", methods=["GET"])
def get_users_embeddings():
    """Lấy danh sách users kèm embedding (cho AI Service)."""
    users = User.query.filter(
        User.is_active == True,
        User.embedding.isnot(None),
    ).all()

    result = []
    for u in users:
        emb = u.get_embedding()
        if emb:
            result.append({
                "id": u.id,
                "name": u.name,
                "embedding": emb,
            })

    return jsonify({"users": result})


@api_bp.route("/users", methods=["POST"])
@api_admin_required(allow_local=True)
def create_user():
    """Tạo user mới."""
    data = request.get_json()
    if not data or not data.get("name"):
        return jsonify({"error": "Thiếu thông tin name"}), 400

    department_obj = None
    department_id = data.get("department_id")
    if department_id is not None:
        try:
            department_obj = Department.query.filter_by(id=int(department_id), is_active=True).first()
        except Exception:
            department_obj = None

    user = User(
        name=data["name"],
        employee_id=data.get("employee_id"),
        department_id=department_obj.id if department_obj else None,
    )

    # Nếu có embedding
    embedding = data.get("embedding")
    if embedding:
        user.set_embedding(embedding)

    db.session.add(user)
    db.session.commit()

    return jsonify({"status": "ok", "user": user.to_dict()}), 201


@api_bp.route("/users/<int:user_id>", methods=["PUT"])
@api_admin_required(allow_local=True)
def update_user(user_id):
    """Cập nhật thông tin user."""
    user = User.query.get_or_404(user_id)
    data = request.get_json()

    if data.get("name"):
        user.name = data["name"]
    if data.get("employee_id") is not None:
        user.employee_id = data["employee_id"]
    if data.get("department_id") is not None:
        dep = Department.query.filter_by(id=data.get("department_id"), is_active=True).first()
        user.department_id = dep.id if dep else None
    if data.get("embedding"):
        user.set_embedding(data["embedding"])

    db.session.commit()
    return jsonify({"status": "ok", "user": user.to_dict()})


@api_bp.route("/users/<int:user_id>", methods=["DELETE"])
@api_admin_required()
def delete_user(user_id):
    """Xóa user (soft delete)."""
    user = User.query.get_or_404(user_id)
    user.is_active = False
    db.session.commit()
    return jsonify({"status": "ok"})


@api_bp.route("/users/all", methods=["DELETE"])
@api_admin_required()
def delete_all_users():
    """Xóa toàn bộ users và dọn ảnh upload liên quan (avatars/faces)."""
    users = User.query.all()
    total_users = len(users)

    try:
        for user in users:
            db.session.delete(user)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Không thể xóa users: {str(e).strip()}"}), 500

    upload_root = current_app.config.get("UPLOAD_FOLDER")
    removed_dirs = []

    if upload_root and os.path.isdir(upload_root):
        for folder in ("avatars", "faces"):
            p = os.path.join(upload_root, folder)
            if os.path.isdir(p):
                try:
                    shutil.rmtree(p)
                    removed_dirs.append(folder)
                except Exception:
                    pass
            os.makedirs(p, exist_ok=True)

    return jsonify({
        "status": "ok",
        "deleted_users": total_users,
        "cleaned_upload_dirs": removed_dirs,
    })


@api_bp.route("/users/<int:user_id>/avatar", methods=["POST"])
@api_admin_required()
def upload_avatar(user_id):
    """Upload ảnh đại diện cho user."""
    user = User.query.get_or_404(user_id)

    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No filename"}), 400

    filename = secure_filename(f"user_{user_id}_{file.filename}")
    upload_dir = os.path.join(current_app.config["UPLOAD_FOLDER"], "avatars")
    os.makedirs(upload_dir, exist_ok=True)
    file.save(os.path.join(upload_dir, filename))

    user.avatar = f"avatars/{filename}"
    db.session.commit()

    return jsonify({"status": "ok", "avatar": user.avatar})


@api_bp.route("/users/<int:user_id>/face_images", methods=["POST"])
@api_admin_required(allow_local=True)
def upload_face_images(user_id):
    """Upload ảnh khuôn mặt và tự tạo embedding cho user.

    - Nhận multipart form-data, field: files (multiple)
    - Tự detect mặt, lấy embedding và lưu vào users.embedding
    """
    try:
        user = User.query.get_or_404(user_id)

        if "files" not in request.files:
            return jsonify({"error": "No files"}), 400

        files = request.files.getlist("files")
        if not files:
            return jsonify({"error": "No files"}), 400

        # Lưu file (để audit/debug), đồng thời trích embedding từ bytes
        faces_dir = os.path.join(current_app.config["UPLOAD_FOLDER"], "faces", f"user_{user_id}")
        os.makedirs(faces_dir, exist_ok=True)

        # Đọc bytes trước vì save() sẽ dịch con trỏ stream
        raw_files = []
        for f in files:
            if not f.filename:
                continue
            raw = f.read()
            if not raw:
                continue
            raw_files.append((f.filename, raw))

        if not raw_files:
            return jsonify({"error": "File rỗng hoặc không đọc được"}), 400

        # Tạo embedding
        class _MemFile:
            def __init__(self, filename, raw):
                self.filename = filename
                self._raw = raw

            def read(self):
                return self._raw

        embedding, used, stats = _embedding_from_uploaded_images([_MemFile(n, r) for (n, r) in raw_files])
        if embedding is None:
            return jsonify({
                "error": "Không phát hiện khuôn mặt hợp lệ trong ảnh",
                "hint": "Hãy dùng ảnh rõ mặt, đủ sáng, không bị che, mặt chiếm diện tích lớn trong ảnh và mỗi ảnh chỉ 1 người.",
                "stats": stats,
            }), 400

        # Save files
        saved = 0
        for filename, raw in raw_files:
            safe = secure_filename(filename)
            out = os.path.join(faces_dir, f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe}")
            with open(out, "wb") as fp:
                fp.write(raw)
            saved += 1

        user.set_embedding(embedding)
        db.session.commit()

        return jsonify({"status": "ok", "saved": saved, "used_for_embedding": used, "stats": stats})
    except Exception as e:
        # Trả JSON để frontend không bị lỗi parse
        return jsonify({"error": f"Lỗi xử lý upload ảnh: {str(e).strip()}"}), 500


@api_bp.route("/me/attendance", methods=["GET"])
@api_employee_required
def my_attendance():
    acc = current_account()
    if not acc:
        return jsonify({"logs": []})

    logs = AttendanceLog.query.filter_by(user_id=acc.id).order_by(AttendanceLog.time.desc()).limit(200).all()
    return jsonify({"logs": [x.to_dict() for x in logs]})


@api_bp.route("/me/face_uploads", methods=["GET"])
@api_employee_required
def my_face_uploads():
    acc = current_account()
    if not acc:
        return jsonify({"requests": []})

    reqs = FaceUploadRequest.query.filter_by(user_id=acc.id).order_by(FaceUploadRequest.created_at.desc()).all()
    return jsonify({"requests": [r.to_dict() for r in reqs]})


@api_bp.route("/me/face_uploads", methods=["POST"])
@api_employee_required
def upload_my_face_images():
    acc = current_account()
    if not acc:
        return jsonify({"error": "Không tìm thấy hồ sơ nhân viên"}), 400

    if "files" not in request.files:
        return jsonify({"error": "No files"}), 400

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files"}), 400

    pend_root = os.path.join(current_app.config["UPLOAD_FOLDER"], "faces_pending", f"user_{acc.id}")
    os.makedirs(pend_root, exist_ok=True)

    saved_paths = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for idx, f in enumerate(files, 1):
        if not f or not f.filename:
            continue
        safe = secure_filename(f.filename)
        rel = f"faces_pending/user_{acc.id}/{ts}_{idx}_{safe}"
        abs_path = os.path.join(current_app.config["UPLOAD_FOLDER"], rel)
        f.save(abs_path)
        saved_paths.append(rel.replace("\\", "/"))

    if not saved_paths:
        return jsonify({"error": "File rỗng hoặc không đọc được"}), 400

    req_item = FaceUploadRequest(user_id=acc.id, status="pending")
    req_item.set_files(saved_paths)
    db.session.add(req_item)
    db.session.commit()

    return jsonify({"status": "ok", "request": req_item.to_dict(), "saved": len(saved_paths)})


@api_bp.route("/admin/face_uploads/pending", methods=["GET"])
@api_admin_required()
def admin_pending_face_uploads():
    reqs = FaceUploadRequest.query.filter_by(status="pending").order_by(FaceUploadRequest.created_at.asc()).all()
    return jsonify({"requests": [r.to_dict() for r in reqs]})


@api_bp.route("/admin/face_uploads/<int:req_id>/approve", methods=["POST"])
@api_admin_required()
def admin_approve_face_upload(req_id):
    req_item = FaceUploadRequest.query.get_or_404(req_id)
    if req_item.status != "pending":
        return jsonify({"error": "Yêu cầu không còn ở trạng thái pending"}), 409

    files = req_item.get_files()
    raw_files = []
    for rel in files:
        full = os.path.join(current_app.config["UPLOAD_FOLDER"], rel)
        if not os.path.isfile(full):
            continue
        with open(full, "rb") as fp:
            raw = fp.read()
        if raw:
            raw_files.append((os.path.basename(full), raw))

    if not raw_files:
        return jsonify({"error": "Không tìm thấy file ảnh hợp lệ để tạo vector"}), 400

    class _MemFile:
        def __init__(self, filename, raw):
            self.filename = filename
            self._raw = raw

        def read(self):
            return self._raw

    embedding, used, stats = _embedding_from_uploaded_images([_MemFile(n, r) for (n, r) in raw_files])
    if embedding is None:
        return jsonify({"error": "Không tạo được embedding", "stats": stats}), 400

    user = User.query.get(req_item.user_id)
    if user is None:
        return jsonify({"error": "Không tìm thấy nhân viên"}), 404

    user.set_embedding(embedding)
    req_item.status = "approved"
    req_item.reviewed_at = datetime.now()
    req_item.reviewed_by = current_account().id if current_account() else None
    db.session.commit()

    return jsonify({"status": "ok", "used_for_embedding": used, "stats": stats})


@api_bp.route("/admin/face_uploads/<int:req_id>/reject", methods=["POST"])
@api_admin_required()
def admin_reject_face_upload(req_id):
    req_item = FaceUploadRequest.query.get_or_404(req_id)
    if req_item.status != "pending":
        return jsonify({"error": "Yêu cầu không còn ở trạng thái pending"}), 409

    data = request.get_json(silent=True) or {}
    note = (data.get("note") or "").strip() or None

    req_item.status = "rejected"
    req_item.note = note
    req_item.reviewed_at = datetime.now()
    req_item.reviewed_by = current_account().id if current_account() else None
    db.session.commit()

    return jsonify({"status": "ok"})


@api_bp.route("/videos", methods=["GET"])
@api_admin_required()
def list_videos():
    """List các video trong thư mục video/ (root workspace)."""
    root = _video_root_dir()
    os.makedirs(root, exist_ok=True)
    exts = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}

    items = []
    for name in sorted(os.listdir(root)):
        full = os.path.join(root, name)
        if os.path.isfile(full) and _allowed_ext(name, exts):
            items.append({"name": name, "size": os.path.getsize(full)})

    return jsonify({"videos": items, "root": root, "status": _video_status()})


@api_bp.route("/videos/status", methods=["GET"])
@api_admin_required()
def videos_status():
    return jsonify({"status": _video_status()})


@api_bp.route("/behavior-video/analyze", methods=["POST"])
@api_admin_required()
def analyze_behavior_video():
    """Phân tích hành vi trực tiếp từ video, không cần nhận diện khuôn mặt."""
    data = request.get_json(silent=True) or {}
    video_name = (data.get("video") or "").strip()
    camera_id = (data.get("camera_id") or "behavior_video").strip() or "behavior_video"

    if not video_name:
        return jsonify({"error": "Thiếu tên video"}), 400

    try:
        video_path = _safe_join_video_root(video_name)
    except ValueError:
        return jsonify({"error": "Video path không hợp lệ"}), 400

    if not os.path.isfile(video_path):
        return jsonify({"error": f"Không tìm thấy video: {video_name}"}), 404

    frame_step = data.get("frame_step", 12)
    max_images = data.get("max_images", 60)
    conf_threshold = data.get("conf", 0.25)

    try:
        frame_step = max(1, min(120, int(frame_step)))
        max_images = max(1, min(300, int(max_images)))
        conf_threshold = max(0.01, min(0.99, float(conf_threshold)))
    except Exception:
        return jsonify({"error": "Tham số không hợp lệ"}), 400

    try:
        import cv2
    except Exception as e:
        return jsonify({"error": f"Không import được OpenCV: {str(e).strip()}"}), 500

    try:
        model = _get_behavior_model()
    except Exception as e:
        return jsonify({"error": str(e).strip()}), 500

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        return jsonify({"error": "Không mở được video"}), 400

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    out_rel_dir = os.path.join("behavior_video", run_id).replace("\\", "/")
    out_abs_dir = os.path.join(_captures_root_dir(), out_rel_dir.replace("/", os.sep))
    os.makedirs(out_abs_dir, exist_ok=True)

    captures = []
    frames_read = 0
    frames_sampled = 0

    try:
        while len(captures) < max_images:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            frames_read += 1
            if (frames_read % frame_step) != 0:
                continue

            frames_sampled += 1

            try:
                result = model.predict(frame, conf=conf_threshold, verbose=False)[0]
            except Exception:
                continue

            boxes = getattr(result, "boxes", None)
            dets = []
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.tolist()
                cls_list = boxes.cls.tolist()
                conf_list = boxes.conf.tolist()
                for i in range(len(xyxy)):
                    conf = float(conf_list[i])
                    if conf < conf_threshold:
                        continue
                    x1, y1, x2, y2 = [int(v) for v in xyxy[i]]
                    cid = int(cls_list[i])
                    label = _behavior_label_from_id(cid)
                    dets.append({
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "class_id": cid,
                        "label": label,
                        "confidence": conf,
                    })

            if not dets:
                probs = getattr(result, "probs", None)
                if probs is None or getattr(probs, "top1", None) is None:
                    continue
                cid = int(probs.top1)
                conf = float(probs.top1conf)
                if conf < conf_threshold:
                    continue
                h, w = frame.shape[:2]
                dets = [{
                    "x1": 1,
                    "y1": 1,
                    "x2": max(2, int(w) - 1),
                    "y2": max(2, int(h) - 1),
                    "class_id": cid,
                    "label": _behavior_label_from_id(cid),
                    "confidence": conf,
                }]

            dets.sort(key=lambda x: x["confidence"], reverse=True)
            preview = frame.copy()

            for det in dets[:6]:
                x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
                cv2.rectangle(preview, (x1, y1), (x2, y2), (30, 220, 30), 2)
                text = f"{det['label']} {det['confidence'] * 100.0:.1f}%"
                cv2.putText(preview, text, (x1, max(16, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 220, 30), 2)

            primary = dets[0]
            label_slug = _ascii_slug(primary["label"])
            fname = f"{run_id}_{frames_read:06d}_{label_slug}.jpg"
            out_abs = os.path.join(out_abs_dir, fname)

            try:
                cv2.imwrite(out_abs, preview, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            except Exception:
                continue

            time_sec = (float(frames_read) / fps) if fps > 0 else None
            captures.append({
                "image": f"{out_rel_dir}/{fname}",
                "frame_index": int(frames_read),
                "time_sec": time_sec,
                "camera_id": camera_id,
                "label": primary["label"],
                "confidence": float(primary["confidence"]),
                "detections": dets[:6],
            })
    finally:
        cap.release()

    return jsonify({
        "status": "ok",
        "video": video_name,
        "camera_id": camera_id,
        "run_id": run_id,
        "frame_step": frame_step,
        "max_images": max_images,
        "conf": conf_threshold,
        "fps": fps,
        "frames_read": frames_read,
        "frames_sampled": frames_sampled,
        "captures": captures,
        "total": len(captures),
    })


@api_bp.route("/videos/start", methods=["POST"])
@api_admin_required()
def start_video_attendance():
    """Start chạy điểm danh từ video bằng cách spawn ai_service/main.py --video ..."""
    global _VIDEO_PROC
    data = request.get_json(silent=True) or {}
    video_name = (data.get("video") or "").strip()
    camera_id = (data.get("camera_id") or "video").strip() or "video"

    if not video_name:
        return jsonify({"error": "Thiếu tên video"}), 400

    try:
        video_path = _safe_join_video_root(video_name)
    except ValueError:
        return jsonify({"error": "Video path không hợp lệ"}), 400

    if not os.path.exists(video_path):
        return jsonify({"error": f"Không tìm thấy video: {video_name}"}), 404

    with _VIDEO_PROC_LOCK:
        st = _video_status()
        if st.get("running"):
            return jsonify({"error": "Đang có tiến trình video chạy", "status": st}), 409

        main_py = os.path.join(AI_SERVICE_DIR, "main.py")
        cmd = [_python_executable(), main_py, "--video", video_path, "--camera-id", camera_id]

        creationflags = 0
        if os.name == "nt":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

        # Ghi log ra file để debug khi tiến trình thoát ngay
        global _VIDEO_LOGS
        logs_dir = os.path.join(REPO_ROOT, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(logs_dir, f"video_runner_{ts}.out.log")
        err_path = os.path.join(logs_dir, f"video_runner_{ts}.err.log")
        out_fp = open(out_path, "a", encoding="utf-8", errors="replace")
        err_fp = open(err_path, "a", encoding="utf-8", errors="replace")
        _VIDEO_LOGS = {"stdout_log": out_path, "stderr_log": err_path}

        _VIDEO_PROC = subprocess.Popen(
            cmd,
            cwd=REPO_ROOT,
            stdout=out_fp,
            stderr=err_fp,
            creationflags=creationflags,
        )

        # Parent có thể đóng file handle, tiến trình con vẫn ghi được
        try:
            out_fp.close()
            err_fp.close()
        except Exception:
            pass

        return jsonify({"status": "ok", "pid": _VIDEO_PROC.pid, "logs": _VIDEO_LOGS})


@api_bp.route("/videos/stop", methods=["POST"])
@api_admin_required()
def stop_video_attendance():
    """Stop tiến trình điểm danh video nếu đang chạy."""
    global _VIDEO_PROC
    with _VIDEO_PROC_LOCK:
        st = _video_status()
        if not st.get("running"):
            _VIDEO_PROC = None
            return jsonify({"status": "ok", "stopped": False})

        try:
            _VIDEO_PROC.terminate()
        except Exception:
            pass

        return jsonify({"status": "ok", "stopped": True})


# ──────────────────────────────────────────────
# Group similar faces (camera/video)
# ──────────────────────────────────────────────
def _captures_root_dir() -> str:
    root = os.path.join(REPO_ROOT, "captures")
    os.makedirs(root, exist_ok=True)
    return root


def _group_root_dir() -> str:
    root = os.path.join(_captures_root_dir(), "grouped")
    os.makedirs(root, exist_ok=True)
    return root


def _camera_source_value(raw):
    txt = str(raw or DEFAULT_CAMERA_SOURCE).strip()
    if txt.isdigit():
        return int(txt)
    return txt


def _face_crop_from_bbox(frame, bbox, margin: float = 0.25):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    h, w = frame.shape[:2]
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    pad = float(margin) * max(bw, bh)

    x1 = int(max(0, np.floor(x1 - pad)))
    y1 = int(max(0, np.floor(y1 - pad)))
    x2 = int(min(w, np.ceil(x2 + pad)))
    y2 = int(min(h, np.ceil(y2 + pad)))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return None
    return crop


def _face_visible_ratio(frame, bbox):
    x1, y1, x2, y2 = [float(v) for v in bbox]
    h, w = frame.shape[:2]

    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    full_area = bw * bh
    if full_area <= 1.0:
        return 0.0

    cx1 = max(0.0, x1)
    cy1 = max(0.0, y1)
    cx2 = min(float(w), x2)
    cy2 = min(float(h), y2)
    cw = max(0.0, cx2 - cx1)
    ch = max(0.0, cy2 - cy1)
    visible_area = cw * ch

    return float(visible_area) / float(full_area)


def _laplacian_variance(cv2, crop) -> float:
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    except Exception:
        return 0.0
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _face_bbox_size(bbox) -> tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    return bw, bh


def _is_frontal_enough(face) -> bool:
    kps = getattr(face, "kps", None)
    if kps is None:
        return False

    try:
        arr = np.asarray(kps, dtype=np.float32)
        if arr.shape[0] < 5 or arr.shape[1] < 2:
            return False
    except Exception:
        return False

    left_eye = arr[0]
    right_eye = arr[1]
    nose = arr[2]
    mouth_l = arr[3]
    mouth_r = arr[4]

    eye_dist = float(np.linalg.norm(right_eye - left_eye))
    if eye_dist < _GROUP_MIN_EYE_DISTANCE:
        return False

    eye_center_x = float((left_eye[0] + right_eye[0]) * 0.5)
    mouth_center_x = float((mouth_l[0] + mouth_r[0]) * 0.5)
    nose_offset = abs(float(nose[0]) - eye_center_x) / eye_dist
    mouth_offset = abs(mouth_center_x - eye_center_x) / eye_dist

    return (nose_offset <= _GROUP_MAX_NOSE_OFFSET_RATIO) and (mouth_offset <= _GROUP_MAX_MOUTH_OFFSET_RATIO)


def _face_pass_quality_filter(cv2, frame, face, crop):
    det_score = float(getattr(face, "det_score", 0.0) or 0.0)
    if det_score < _GROUP_MIN_DET_SCORE:
        return False, {"reason": "low_score", "det_score": det_score}

    face_w, face_h = _face_bbox_size(face.bbox)
    if face_w < _GROUP_MIN_FACE_WIDTH or face_h < _GROUP_MIN_FACE_HEIGHT:
        return False, {"reason": "small_face", "face_w": face_w, "face_h": face_h}

    visible_ratio = _face_visible_ratio(frame, face.bbox)
    if visible_ratio < _GROUP_MIN_VISIBLE_FACE_RATIO:
        return False, {"visible_ratio": visible_ratio, "lap_var": 0.0, "reason": "partial"}

    if not _is_frontal_enough(face):
        return False, {"visible_ratio": visible_ratio, "lap_var": 0.0, "reason": "side_face"}

    lap_var = _laplacian_variance(cv2, crop)
    if lap_var < _GROUP_MIN_LAPLACIAN_VAR:
        return False, {"visible_ratio": visible_ratio, "lap_var": lap_var, "reason": "blur"}

    return True, {"visible_ratio": visible_ratio, "lap_var": lap_var, "reason": "ok"}


def _job_to_public(job):
    if not job:
        return {"status": "idle"}
    return {
        "id": job.get("id"),
        "source": job.get("source"),
        "status": job.get("status"),
        "created_at": job.get("created_at"),
        "finished_at": job.get("finished_at"),
        "config": job.get("config") or {},
        "progress": job.get("progress") or {},
        "result": job.get("result"),
        "error": job.get("error"),
    }


def _job_get(source: str):
    with _GROUP_JOB_LOCK:
        return _GROUP_JOBS.get(source)


def _job_set(source: str, job: dict):
    with _GROUP_JOB_LOCK:
        _GROUP_JOBS[source] = job


def _job_patch(source: str, job_id: str, **updates):
    with _GROUP_JOB_LOCK:
        job = _GROUP_JOBS.get(source)
        if not job or job.get("id") != job_id:
            return
        for k, v in updates.items():
            job[k] = v


def _bbox_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter_area
    if denom <= 0.0:
        return 0.0
    return float(inter_area) / float(denom)


def _iter_group_tiles(frame, cols: int, rows: int, expand_ratio: float = 1.2):
    h, w = frame.shape[:2]
    tile_w = min(w, max(64, int(round((w / float(cols)) * expand_ratio))))
    tile_h = min(h, max(64, int(round((h / float(rows)) * expand_ratio))))

    max_x = max(0, w - tile_w)
    max_y = max(0, h - tile_h)
    xs = [int(round(v)) for v in np.linspace(0, max_x, cols)] if cols > 1 else [0]
    ys = [int(round(v)) for v in np.linspace(0, max_y, rows)] if rows > 1 else [0]

    seen = set()
    for y in ys:
        for x in xs:
            key = (x, y, tile_w, tile_h)
            if key in seen:
                continue
            seen.add(key)
            yield x, y, frame[y : y + tile_h, x : x + tile_w]


def _translate_face_from_tile(face, offset_x: int, offset_y: int):
    face.bbox = np.asarray(face.bbox, dtype=np.float32).copy()
    face.bbox[[0, 2]] += float(offset_x)
    face.bbox[[1, 3]] += float(offset_y)

    if getattr(face, "kps", None) is not None:
        kps = np.asarray(face.kps, dtype=np.float32).copy()
        kps[:, 0] += float(offset_x)
        kps[:, 1] += float(offset_y)
        face.kps = kps

    return face


def _merge_detected_faces(base_faces: list, incoming_faces: list, offset_x: int = 0, offset_y: int = 0):
    for face in incoming_faces:
        translated = _translate_face_from_tile(face, offset_x, offset_y)
        replaced = False
        for idx, existing in enumerate(base_faces):
            if _bbox_iou(existing.bbox, translated.bbox) >= 0.35:
                existing_score = float(getattr(existing, "det_score", 0.0) or 0.0)
                new_score = float(getattr(translated, "det_score", 0.0) or 0.0)
                if new_score > existing_score:
                    base_faces[idx] = translated
                replaced = True
                break
        if not replaced:
            base_faces.append(translated)


def _detect_faces_for_grouping(app, frame):
    h, w = frame.shape[:2]
    all_faces = []

    attempts = [
        {"det_size": (1024, 1024), "det_thresh": 0.20, "tiles": [(0, 0, frame)]},
    ]

    max_side = max(h, w)
    if max_side >= 1200:
        attempts.append({"det_size": (1024, 1024), "det_thresh": 0.16, "tiles": list(_iter_group_tiles(frame, cols=2, rows=2, expand_ratio=1.25))})
    if max_side >= 1800:
        attempts.append({"det_size": (1024, 1024), "det_thresh": 0.14, "tiles": list(_iter_group_tiles(frame, cols=3, rows=2, expand_ratio=1.15))})

    with _FACE_INFER_LOCK:
        prev_size = tuple(getattr(app.detector, "input_size", (640, 640)))
        prev_thresh = float(getattr(app.detector, "det_thresh", 0.35))
        try:
            for attempt in attempts:
                app.detector.input_size = attempt["det_size"]
                app.detector.det_thresh = float(attempt["det_thresh"])

                for offset_x, offset_y, tile in attempt["tiles"]:
                    try:
                        faces = app.get(tile)
                    except Exception:
                        faces = []
                    if faces:
                        _merge_detected_faces(all_faces, faces, offset_x=offset_x, offset_y=offset_y)
        finally:
            app.detector.input_size = prev_size
            app.detector.det_thresh = prev_thresh

    return all_faces


def _cluster_faces(face_items: list[dict], source: str, job_id: str, threshold: float):
    import cv2

    out_root = os.path.join(_group_root_dir(), source, job_id)
    os.makedirs(out_root, exist_ok=True)

    clusters = []
    total_saved = 0

    for i, item in enumerate(face_items):
        emb = item["embedding"]
        crop = item["crop"]

        best_idx = -1
        best_sim = -1.0
        for idx, c in enumerate(clusters):
            sim = float(np.dot(emb, c["centroid"]))
            if sim > best_sim:
                best_sim = sim
                best_idx = idx

        if best_idx >= 0 and best_sim >= threshold:
            cluster = clusters[best_idx]
            old_count = int(cluster["count"])
            new_centroid = (cluster["centroid"] * old_count + emb) / float(old_count + 1)
            cluster["centroid"] = _l2_normalize(new_centroid)
            cluster["count"] = old_count + 1
            cluster["best_similarity"] = max(float(cluster.get("best_similarity", -1.0)), best_sim)
        else:
            cluster_idx = len(clusters) + 1
            folder_name = f"Chua_nhan_dang{cluster_idx}"
            folder_abs = os.path.join(out_root, folder_name)
            os.makedirs(folder_abs, exist_ok=True)
            cluster = {
                "name": folder_name,
                "folder_abs": folder_abs,
                "centroid": emb.copy(),
                "count": 1,
                "best_similarity": 1.0,
                "preview": None,
            }
            clusters.append(cluster)

        filename = f"face_{i + 1:05d}.jpg"
        img_out = os.path.join(cluster["folder_abs"], filename)
        ok = cv2.imwrite(img_out, crop)
        if ok:
            total_saved += 1
            if cluster["preview"] is None:
                cluster["preview"] = filename

    captures_root = _captures_root_dir()
    summary = []
    for c in clusters:
        rel_folder = os.path.relpath(c["folder_abs"], captures_root).replace("\\", "/")
        preview_url = None
        if c.get("preview"):
            preview_url = "/captures/" + rel_folder + "/" + c["preview"]
        summary.append({
            "name": c["name"],
            "folder": rel_folder,
            "count": int(c["count"]),
            "preview": preview_url,
            "best_similarity": round(float(c.get("best_similarity", 0.0)), 4),
        })

    return {
        "output_dir": os.path.relpath(out_root, REPO_ROOT).replace("\\", "/"),
        "cluster_count": len(summary),
        "total_faces_saved": int(total_saved),
        "clusters": summary,
    }


def _run_group_video_job(job_id: str, video_name: str, sample_every: int, threshold: float, max_frames: int):
    try:
        import cv2

        video_path = _safe_join_video_root(video_name)
        if not os.path.exists(video_path):
            raise RuntimeError(f"Không tìm thấy video: {video_name}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError("Không mở được video")

        app = _get_face_app()
        frame_idx = 0
        sampled = 0
        face_items = []
        skipped_blur = 0
        skipped_partial = 0
        skipped_side_face = 0
        skipped_small_face = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_idx += 1
                if sample_every > 1 and (frame_idx % sample_every) != 0:
                    continue

                sampled += 1
                if max_frames > 0 and sampled > max_frames:
                    break

                faces = _detect_faces_for_grouping(app, frame)

                for face in faces:
                    if getattr(face, "embedding", None) is None:
                        continue
                    crop = _face_crop_from_bbox(frame, face.bbox)
                    if crop is None:
                        continue
                    ok_face, meta = _face_pass_quality_filter(cv2, frame, face, crop)
                    if not ok_face:
                        if meta.get("reason") == "blur":
                            skipped_blur += 1
                        elif meta.get("reason") == "partial":
                            skipped_partial += 1
                        elif meta.get("reason") in {"side_face", "low_score"}:
                            skipped_side_face += 1
                        elif meta.get("reason") == "small_face":
                            skipped_small_face += 1
                        continue
                    emb = _l2_normalize(np.asarray(face.embedding, dtype=np.float32))
                    face_items.append({"embedding": emb, "crop": crop})

                _job_patch(
                    "video",
                    job_id,
                    progress={
                        "frames_read": frame_idx,
                        "frames_sampled": sampled,
                        "faces_found": len(face_items),
                        "skipped_blur": skipped_blur,
                        "skipped_partial": skipped_partial,
                        "skipped_side_face": skipped_side_face,
                        "skipped_small_face": skipped_small_face,
                    },
                )
        finally:
            cap.release()

        result = _cluster_faces(face_items, source="video", job_id=job_id, threshold=threshold)
        result["video"] = video_name
        result["frames_read"] = frame_idx
        result["frames_sampled"] = sampled
        result["skipped_blur"] = skipped_blur
        result["skipped_partial"] = skipped_partial
        result["skipped_side_face"] = skipped_side_face
        result["skipped_small_face"] = skipped_small_face

        _job_patch(
            "video",
            job_id,
            status="done",
            finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            progress={
                "frames_read": frame_idx,
                "frames_sampled": sampled,
                "faces_found": len(face_items),
                "skipped_blur": skipped_blur,
                "skipped_partial": skipped_partial,
                "skipped_side_face": skipped_side_face,
                "skipped_small_face": skipped_small_face,
            },
            result=result,
            error=None,
        )
    except Exception as e:
        _job_patch(
            "video",
            job_id,
            status="failed",
            finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            error=str(e).strip(),
        )


def _run_group_camera_job(job_id: str, camera_source_raw, duration_sec: float, sample_interval: float, threshold: float):
    try:
        import cv2

        camera_source = _camera_source_value(camera_source_raw)
        cap = cv2.VideoCapture(camera_source)
        if not cap.isOpened():
            raise RuntimeError(f"Không mở được camera/source: {camera_source_raw}")

        app = _get_face_app()
        start = time.monotonic()
        next_sample = start
        frames_read = 0
        frames_sampled = 0
        face_items = []
        skipped_blur = 0
        skipped_partial = 0
        skipped_side_face = 0
        skipped_small_face = 0

        try:
            while True:
                if (time.monotonic() - start) >= duration_sec:
                    break

                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                frames_read += 1
                now = time.monotonic()
                if now < next_sample:
                    continue
                next_sample = now + sample_interval

                frames_sampled += 1
                faces = _detect_faces_for_grouping(app, frame)

                for face in faces:
                    if getattr(face, "embedding", None) is None:
                        continue
                    crop = _face_crop_from_bbox(frame, face.bbox)
                    if crop is None:
                        continue
                    ok_face, meta = _face_pass_quality_filter(cv2, frame, face, crop)
                    if not ok_face:
                        if meta.get("reason") == "blur":
                            skipped_blur += 1
                        elif meta.get("reason") == "partial":
                            skipped_partial += 1
                        elif meta.get("reason") in {"side_face", "low_score"}:
                            skipped_side_face += 1
                        elif meta.get("reason") == "small_face":
                            skipped_small_face += 1
                        continue
                    emb = _l2_normalize(np.asarray(face.embedding, dtype=np.float32))
                    face_items.append({"embedding": emb, "crop": crop})

                _job_patch(
                    "camera",
                    job_id,
                    progress={
                        "frames_read": frames_read,
                        "frames_sampled": frames_sampled,
                        "faces_found": len(face_items),
                        "skipped_blur": skipped_blur,
                        "skipped_partial": skipped_partial,
                        "skipped_side_face": skipped_side_face,
                        "skipped_small_face": skipped_small_face,
                    },
                )
        finally:
            cap.release()

        result = _cluster_faces(face_items, source="camera", job_id=job_id, threshold=threshold)
        result["camera_source"] = str(camera_source_raw)
        result["duration_sec"] = duration_sec
        result["frames_read"] = frames_read
        result["frames_sampled"] = frames_sampled
        result["skipped_blur"] = skipped_blur
        result["skipped_partial"] = skipped_partial
        result["skipped_side_face"] = skipped_side_face
        result["skipped_small_face"] = skipped_small_face

        _job_patch(
            "camera",
            job_id,
            status="done",
            finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            progress={
                "frames_read": frames_read,
                "frames_sampled": frames_sampled,
                "faces_found": len(face_items),
                "skipped_blur": skipped_blur,
                "skipped_partial": skipped_partial,
                "skipped_side_face": skipped_side_face,
                "skipped_small_face": skipped_small_face,
            },
            result=result,
            error=None,
        )
    except Exception as e:
        _job_patch(
            "camera",
            job_id,
            status="failed",
            finished_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            error=str(e).strip(),
        )


@api_bp.route("/grouping/status", methods=["GET"])
@api_admin_required()
def grouping_status():
    source = (request.args.get("source") or "camera").strip().lower()
    if source not in {"camera", "video"}:
        return jsonify({"error": "source phải là camera hoặc video"}), 400
    return jsonify(_job_to_public(_job_get(source)))


@api_bp.route("/grouping/camera/start", methods=["POST"])
@api_admin_required()
def grouping_camera_start():
    data = request.get_json(silent=True) or {}
    camera_source = str(data.get("camera_source", DEFAULT_CAMERA_SOURCE)).strip() or DEFAULT_CAMERA_SOURCE
    duration_sec = float(data.get("duration_sec", 20))
    sample_interval = float(data.get("sample_interval", 0.6))
    threshold = float(data.get("threshold", 0.58))

    duration_sec = min(max(duration_sec, 3.0), 600.0)
    sample_interval = min(max(sample_interval, 0.1), 5.0)
    threshold = min(max(threshold, 0.3), 0.95)

    old = _job_get("camera")
    if old and old.get("status") == "running":
        return jsonify({"error": "Đang có job gom ảnh camera chạy"}), 409

    job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    job = {
        "id": job_id,
        "source": "camera",
        "status": "running",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "finished_at": None,
        "config": {
            "camera_source": camera_source,
            "duration_sec": duration_sec,
            "sample_interval": sample_interval,
            "threshold": threshold,
        },
        "progress": {
            "frames_read": 0,
            "frames_sampled": 0,
            "faces_found": 0,
            "skipped_blur": 0,
            "skipped_partial": 0,
            "skipped_side_face": 0,
            "skipped_small_face": 0,
        },
        "result": None,
        "error": None,
    }
    _job_set("camera", job)

    t = threading.Thread(
        target=_run_group_camera_job,
        args=(job_id, camera_source, duration_sec, sample_interval, threshold),
        daemon=True,
    )
    t.start()

    return jsonify(_job_to_public(job)), 202


@api_bp.route("/grouping/video/start", methods=["POST"])
@api_admin_required()
def grouping_video_start():
    data = request.get_json(silent=True) or {}
    video_name = str(data.get("video") or "").strip()
    if not video_name:
        return jsonify({"error": "Thiếu tên video"}), 400

    sample_every = int(data.get("sample_every", 8))
    max_frames = int(data.get("max_frames", 0))
    threshold = float(data.get("threshold", 0.58))

    sample_every = min(max(sample_every, 1), 120)
    max_frames = min(max(max_frames, 0), 20000)
    threshold = min(max(threshold, 0.3), 0.95)

    old = _job_get("video")
    if old and old.get("status") == "running":
        return jsonify({"error": "Đang có job gom ảnh video chạy"}), 409

    job_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    job = {
        "id": job_id,
        "source": "video",
        "status": "running",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "finished_at": None,
        "config": {
            "video": video_name,
            "sample_every": sample_every,
            "max_frames": max_frames,
            "threshold": threshold,
        },
        "progress": {
            "frames_read": 0,
            "frames_sampled": 0,
            "faces_found": 0,
            "skipped_blur": 0,
            "skipped_partial": 0,
            "skipped_side_face": 0,
            "skipped_small_face": 0,
        },
        "result": None,
        "error": None,
    }
    _job_set("video", job)

    t = threading.Thread(
        target=_run_group_video_job,
        args=(job_id, video_name, sample_every, threshold, max_frames),
        daemon=True,
    )
    t.start()

    return jsonify(_job_to_public(job)), 202


# ──────────────────────────────────────────────
# Attendance Logs
# ──────────────────────────────────────────────
def _parse_month_arg(month_text: str) -> tuple[date, date, int]:
    try:
        dt = datetime.strptime(month_text, "%Y-%m")
    except Exception:
        raise ValueError("month phải có định dạng YYYY-MM")

    days_in_month = calendar.monthrange(dt.year, dt.month)[1]
    month_start = date(dt.year, dt.month, 1)
    month_end = date(dt.year, dt.month, days_in_month)
    return month_start, month_end, days_in_month


def _build_department_attendance_map(department_id: int, month_start: date, month_end: date):
    users = User.query.filter_by(department_id=department_id, is_active=True).order_by(User.name.asc()).all()

    start_dt = datetime.combine(month_start, datetime.min.time())
    end_dt = datetime.combine(month_end, datetime.max.time())

    rows = db.session.query(
        AttendanceLog.user_id,
        db.func.date(AttendanceLog.time).label("log_date"),
    ).join(User, AttendanceLog.user_id == User.id).filter(
        User.department_id == department_id,
        User.is_active == True,
        AttendanceLog.user_id.isnot(None),
        AttendanceLog.time >= start_dt,
        AttendanceLog.time <= end_dt,
    ).all()

    present_by_user_day: dict[tuple[int, date], bool] = {}
    present_by_day: dict[date, set[int]] = {}

    for uid, log_day in rows:
        if uid is None or log_day is None:
            continue
        normalized_day = log_day if isinstance(log_day, date) else datetime.strptime(str(log_day), "%Y-%m-%d").date()
        present_by_user_day[(int(uid), normalized_day)] = True
        if normalized_day not in present_by_day:
            present_by_day[normalized_day] = set()
        present_by_day[normalized_day].add(int(uid))

    return users, present_by_user_day, present_by_day


def _build_manual_mark_map(department_id: int, month_start: date, month_end: date):
    rows = TeacherNotebookAttendanceMark.query.filter(
        TeacherNotebookAttendanceMark.department_id == department_id,
        TeacherNotebookAttendanceMark.lesson_date >= month_start,
        TeacherNotebookAttendanceMark.lesson_date <= month_end,
    ).all()

    manual_map: dict[tuple[int, date], bool] = {}
    for row in rows:
        manual_map[(int(row.user_id), row.lesson_date)] = bool(row.is_present)
    return manual_map


@api_bp.route("/teacher-notebook/entries", methods=["GET"])
@api_admin_required()
def get_teacher_notebook_entries():
    department_id = request.args.get("department_id", type=int)
    month_text = (request.args.get("month") or "").strip()
    if not department_id or not month_text:
        return jsonify({"error": "Thiếu department_id hoặc month"}), 400

    try:
        month_start, month_end, _ = _parse_month_arg(month_text)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    entries = TeacherNotebookEntry.query.filter(
        TeacherNotebookEntry.department_id == department_id,
        TeacherNotebookEntry.lesson_date >= month_start,
        TeacherNotebookEntry.lesson_date <= month_end,
    ).order_by(TeacherNotebookEntry.lesson_date.asc(), TeacherNotebookEntry.period_no.asc()).all()

    return jsonify({"entries": [e.to_dict() for e in entries]})


@api_bp.route("/teacher-notebook/entries", methods=["POST"])
@api_admin_required()
def upsert_teacher_notebook_entry():
    data = request.get_json(silent=True) or {}

    department_id = data.get("department_id")
    lesson_date_text = (data.get("lesson_date") or "").strip()
    period_no = data.get("period_no")
    lesson_content = (data.get("lesson_content") or "").strip()
    note = (data.get("note") or "").strip() or None

    if not department_id or not lesson_date_text or not period_no or not lesson_content:
        return jsonify({"error": "Thiếu dữ liệu bắt buộc"}), 400

    department = Department.query.filter_by(id=department_id, is_active=True).first()
    if not department:
        return jsonify({"error": "Lớp/phòng ban không tồn tại"}), 404

    try:
        parsed_date = datetime.strptime(lesson_date_text, "%Y-%m-%d").date()
    except Exception:
        return jsonify({"error": "lesson_date phải có định dạng YYYY-MM-DD"}), 400

    try:
        period_no = int(period_no)
    except Exception:
        return jsonify({"error": "period_no không hợp lệ"}), 400

    if period_no <= 0:
        return jsonify({"error": "period_no phải lớn hơn 0"}), 400

    entry = TeacherNotebookEntry.query.filter_by(
        department_id=department.id,
        lesson_date=parsed_date,
        period_no=period_no,
    ).first()

    if not entry:
        entry = TeacherNotebookEntry(
            department_id=department.id,
            lesson_date=parsed_date,
            period_no=period_no,
            lesson_content=lesson_content,
            note=note,
        )
        db.session.add(entry)
    else:
        entry.lesson_content = lesson_content
        entry.note = note

    db.session.commit()
    return jsonify({"status": "ok", "entry": entry.to_dict()})


@api_bp.route("/teacher-notebook/entries/<int:entry_id>", methods=["DELETE"])
@api_admin_required()
def delete_teacher_notebook_entry(entry_id):
    entry = TeacherNotebookEntry.query.get_or_404(entry_id)
    db.session.delete(entry)
    db.session.commit()
    return jsonify({"status": "ok"})


@api_bp.route("/teacher-notebook/attendance-mark", methods=["POST"])
@api_admin_required()
def upsert_teacher_notebook_attendance_mark():
    data = request.get_json(silent=True) or {}

    department_id = data.get("department_id")
    user_id = data.get("user_id")
    lesson_date_text = (data.get("lesson_date") or "").strip()
    is_present = data.get("is_present")

    if not department_id or not user_id or not lesson_date_text or is_present is None:
        return jsonify({"error": "Thiếu dữ liệu bắt buộc"}), 400

    department = Department.query.filter_by(id=department_id, is_active=True).first()
    if not department:
        return jsonify({"error": "Lớp/phòng ban không tồn tại"}), 404

    user = User.query.filter_by(id=user_id, is_active=True).first()
    if not user or user.department_id != department.id:
        return jsonify({"error": "Học viên không thuộc lớp/phòng ban đã chọn"}), 400

    try:
        lesson_day = datetime.strptime(lesson_date_text, "%Y-%m-%d").date()
    except Exception:
        return jsonify({"error": "lesson_date phải có định dạng YYYY-MM-DD"}), 400

    if isinstance(is_present, str):
        is_present = is_present.strip().lower() in {"1", "true", "yes", "on"}
    else:
        is_present = bool(is_present)

    editor = current_account()
    mark = TeacherNotebookAttendanceMark.query.filter_by(
        department_id=department.id,
        user_id=user.id,
        lesson_date=lesson_day,
    ).first()

    if not mark:
        mark = TeacherNotebookAttendanceMark(
            department_id=department.id,
            user_id=user.id,
            lesson_date=lesson_day,
            is_present=is_present,
            updated_by=editor.id if editor else None,
        )
        db.session.add(mark)
    else:
        mark.is_present = is_present
        mark.updated_by = editor.id if editor else None

    db.session.commit()
    return jsonify({"status": "ok", "mark": mark.to_dict()})


@api_bp.route("/teacher-notebook/attendance-matrix", methods=["GET"])
@api_admin_required()
def teacher_notebook_attendance_matrix():
    department_id = request.args.get("department_id", type=int)
    month_text = (request.args.get("month") or "").strip()
    if not department_id or not month_text:
        return jsonify({"error": "Thiếu department_id hoặc month"}), 400

    department = Department.query.filter_by(id=department_id, is_active=True).first()
    if not department:
        return jsonify({"error": "Lớp/phòng ban không tồn tại"}), 404

    try:
        month_start, month_end, _ = _parse_month_arg(month_text)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    users, present_by_user_day, _ = _build_department_attendance_map(department_id, month_start, month_end)
    manual_marks = _build_manual_mark_map(department_id, month_start, month_end)

    lesson_dates_rows = db.session.query(TeacherNotebookEntry.lesson_date).filter(
        TeacherNotebookEntry.department_id == department_id,
        TeacherNotebookEntry.lesson_date >= month_start,
        TeacherNotebookEntry.lesson_date <= month_end,
    ).distinct().order_by(TeacherNotebookEntry.lesson_date.asc()).all()
    lesson_dates = [r[0] for r in lesson_dates_rows if r[0]]

    students = []
    for u in users:
        marks = []
        manual_flags = []
        total_present = 0
        for lesson_day in lesson_dates:
            key = (u.id, lesson_day)
            if key in manual_marks:
                has_present = bool(manual_marks[key])
                is_manual = True
            else:
                has_present = bool(present_by_user_day.get(key))
                is_manual = False

            marks.append(has_present)
            manual_flags.append(is_manual)
            if has_present:
                total_present += 1

        students.append({
            "id": u.id,
            "name": u.name,
            "employee_id": u.employee_id,
            "attendance_marks": marks,
            "attendance_manual_flags": manual_flags,
            "total_present": total_present,
        })

    return jsonify({
        "department": department.to_dict(),
        "month": month_text,
        "lesson_dates": [d.strftime("%Y-%m-%d") for d in lesson_dates],
        "lesson_date_labels": [d.strftime("%d/%m/%Y") for d in lesson_dates],
        "students": students,
    })


@api_bp.route("/teacher-notebook/export", methods=["GET"])
@api_admin_required()
def export_teacher_notebook_excel():
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Border, Font, Side
        from openpyxl.utils import get_column_letter
    except Exception:
        return jsonify({"error": "Thiếu thư viện openpyxl. Vui lòng cài trong flask_app: pip install -r requirements.txt"}), 500

    department_id = request.args.get("department_id", type=int)
    month_text = (request.args.get("month") or "").strip()
    subject = (request.args.get("subject") or "").strip()
    period_count = request.args.get("period_count", type=int) or 0

    if not department_id or not month_text:
        return jsonify({"error": "Thiếu department_id hoặc month"}), 400

    department = Department.query.filter_by(id=department_id, is_active=True).first()
    if not department:
        return jsonify({"error": "Lớp/phòng ban không tồn tại"}), 404

    try:
        month_start, month_end, _ = _parse_month_arg(month_text)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    users, present_by_user_day, _ = _build_department_attendance_map(department_id, month_start, month_end)
    manual_marks = _build_manual_mark_map(department_id, month_start, month_end)

    entries = TeacherNotebookEntry.query.filter(
        TeacherNotebookEntry.department_id == department_id,
        TeacherNotebookEntry.lesson_date >= month_start,
        TeacherNotebookEntry.lesson_date <= month_end,
    ).order_by(TeacherNotebookEntry.lesson_date.asc(), TeacherNotebookEntry.period_no.asc()).all()

    lesson_dates = sorted({e.lesson_date for e in entries if e.lesson_date})

    effective_present_by_day: dict[date, set[int]] = {}
    for lesson_day in lesson_dates:
        effective_present_by_day[lesson_day] = set()
        for user in users:
            key = (user.id, lesson_day)
            if key in manual_marks:
                is_present = bool(manual_marks[key])
            else:
                is_present = bool(present_by_user_day.get(key))
            if is_present:
                effective_present_by_day[lesson_day].add(user.id)

    wb = Workbook()
    ws1 = wb.active
    ws1.title = "Bang cham cong"
    ws2 = wb.create_sheet("Theo doi giang day")

    thin = Side(border_style="thin", color="000000")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    # Sheet 1: BANG CHAM CONG NGHI HOC LOP
    date_col_count = max(len(lesson_dates), 1)
    last_col_1 = 2 + date_col_count + 1
    ws1.merge_cells(start_row=1, start_column=1, end_row=1, end_column=last_col_1)
    ws1.cell(row=1, column=1, value="BANG CHAM CONG NGHI HOC LOP").font = Font(bold=True, size=16)
    ws1.cell(row=1, column=1).alignment = Alignment(horizontal="center")

    ws1.merge_cells(start_row=2, start_column=1, end_row=2, end_column=8)
    ws1.cell(row=2, column=1, value=f"Mon hoc: {subject or '...'}")
    ws1.merge_cells(start_row=2, start_column=9, end_row=2, end_column=last_col_1)
    ws1.cell(row=2, column=9, value=f"So tiet: {period_count if period_count > 0 else '...'}")

    ws1.cell(row=4, column=1, value="TT")
    ws1.cell(row=4, column=2, value="Hoc va ten")
    if lesson_dates:
        for idx, lesson_day in enumerate(lesson_dates, start=1):
            ws1.cell(row=4, column=2 + idx, value=lesson_day.strftime("%d/%m/%Y"))
    else:
        ws1.cell(row=4, column=3, value="(Chua co tiet hoc)")
    ws1.cell(row=4, column=last_col_1, value="Tong")

    for c in range(1, last_col_1 + 1):
        ws1.cell(row=4, column=c).font = Font(bold=True)
        ws1.cell(row=4, column=c).alignment = Alignment(horizontal="center", vertical="center")

    for idx, user in enumerate(users, start=1):
        row = 4 + idx
        ws1.cell(row=row, column=1, value=idx)
        ws1.cell(row=row, column=2, value=user.name)

        present_total = 0
        if lesson_dates:
            for col_idx, lesson_day in enumerate(lesson_dates, start=1):
                present = user.id in effective_present_by_day.get(lesson_day, set())
                ws1.cell(row=row, column=2 + col_idx, value="x" if present else "")
                if present:
                    present_total += 1
        ws1.cell(row=row, column=last_col_1, value=present_total)

    max_row_1 = 4 + max(len(users), 1)
    for r in range(4, max_row_1 + 1):
        for c in range(1, last_col_1 + 1):
            cell = ws1.cell(row=r, column=c)
            cell.border = border
            cell.alignment = Alignment(horizontal="center", vertical="center")

    ws1.column_dimensions["A"].width = 5
    ws1.column_dimensions["B"].width = 28
    for col_idx in range(3, last_col_1):
        ws1.column_dimensions[get_column_letter(col_idx)].width = 12

    # Sheet 2: THEO DOI GIANG DAY - HOC TAP TREN LOP
    ws2.merge_cells("A1:E1")
    ws2["A1"] = "THEO DOI GIANG DAY - HOC TAP TREN LOP"
    ws2["A1"].font = Font(bold=True, size=16)
    ws2["A1"].alignment = Alignment(horizontal="center")

    ws2["A2"] = f"Mon hoc: {subject or '...'}"
    ws2["A3"] = f"So tiet: {period_count if period_count > 0 else '...'}"
    ws2["A4"] = f"Lop: {department.name}"

    headers = ["Ngay", "Tiet thu", "Noi dung giang day", "Quan so", "Ten HV va li do nghi hoc"]
    for idx, h in enumerate(headers, start=1):
        ws2.cell(row=6, column=idx, value=h)
        ws2.cell(row=6, column=idx).font = Font(bold=True)
        ws2.cell(row=6, column=idx).alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        ws2.cell(row=6, column=idx).border = border

    total_students = len(users)
    user_name_by_id = {u.id: u.name for u in users}

    start_row_2 = 7
    for offset, entry in enumerate(entries):
        row = start_row_2 + offset
        present_ids = effective_present_by_day.get(entry.lesson_date, set())
        absent_names = [user_name_by_id[u.id] for u in users if u.id not in present_ids]
        absent_text = ", ".join(absent_names)
        if entry.note:
            absent_text = f"{absent_text} | Ghi chu: {entry.note}" if absent_text else f"Ghi chu: {entry.note}"

        ws2.cell(row=row, column=1, value=entry.lesson_date.strftime("%d/%m/%Y"))
        ws2.cell(row=row, column=2, value=entry.period_no)
        ws2.cell(row=row, column=3, value=entry.lesson_content)
        ws2.cell(row=row, column=4, value=min(len(present_ids), total_students))
        ws2.cell(row=row, column=5, value=absent_text)

        for c in range(1, 6):
            ws2.cell(row=row, column=c).border = border
            ws2.cell(row=row, column=c).alignment = Alignment(vertical="center", wrap_text=True)

    if not entries:
        for c in range(1, 6):
            ws2.cell(row=start_row_2, column=c).border = border
        ws2.cell(row=start_row_2, column=1, value="(Chua co noi dung giang day trong thang)")

    ws2.column_dimensions["A"].width = 14
    ws2.column_dimensions["B"].width = 10
    ws2.column_dimensions["C"].width = 45
    ws2.column_dimensions["D"].width = 10
    ws2.column_dimensions["E"].width = 45

    out = io.BytesIO()
    wb.save(out)
    out.seek(0)

    safe_month = month_text.replace("-", "")
    filename = f"so_tay_giao_vien_{department.name}_{safe_month}.xlsx"
    return send_file(
        out,
        as_attachment=True,
        download_name=filename,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


@api_bp.route("/logs", methods=["GET"])
@api_admin_required()
def get_logs():
    """Lấy lịch sử điểm danh tổng hợp theo phòng ban."""
    page = request.args.get("page", 1, type=int)
    per_page = request.args.get("per_page", 50, type=int)
    date_str = request.args.get("date")
    user_id = request.args.get("user_id", type=int)
    department_id_raw = request.args.get("department_id")
    unassigned = str(request.args.get("unassigned", "0")).strip().lower() in {"1", "true", "yes"}

    query = AttendanceLog.query.outerjoin(User, AttendanceLog.user_id == User.id).outerjoin(
        Department, User.department_id == Department.id
    )

    if date_str:
        try:
            date = datetime.strptime(date_str, "%Y-%m-%d")
            query = query.filter(
                db.func.date(AttendanceLog.time) == date.date()
            )
        except ValueError:
            pass

    if user_id:
        query = query.filter(AttendanceLog.user_id == user_id)

    if unassigned:
        query = query.filter(User.department_id.is_(None))
    elif department_id_raw not in (None, ""):
        try:
            department_id = int(department_id_raw)
        except (TypeError, ValueError):
            return jsonify({"error": "department_id không hợp lệ"}), 400
        query = query.filter(User.department_id == department_id)

    # Khi có lọc theo phòng cụ thể, trả về danh sách chi tiết bản ghi điểm danh.
    if unassigned or department_id_raw not in (None, ""):
        pagination = query.order_by(AttendanceLog.time.desc()).paginate(
            page=page, per_page=per_page, error_out=False
        )

        detail_rows = []
        for log in pagination.items:
            item = log.to_dict()
            item["room_name"] = log.user.department_rel.name if log.user and log.user.department_rel else "Chua phan phong"
            detail_rows.append(item)

        return jsonify({
            "logs": detail_rows,
            "total": pagination.total,
            "page": pagination.page,
            "pages": pagination.pages,
            "mode": "detail",
        })

    grouped_query = query.with_entities(
        Department.id.label("department_id"),
        db.func.coalesce(Department.name, "Chua phan phong").label("room_name"),
        db.func.count(AttendanceLog.id).label("attendance_count"),
        db.func.count(db.func.distinct(AttendanceLog.user_id)).label("unique_users"),
        db.func.max(AttendanceLog.time).label("last_time"),
        db.func.group_concat(db.func.distinct(AttendanceLog.camera_id)).label("cameras"),
    ).group_by(Department.id, Department.name).order_by(db.func.max(AttendanceLog.time).desc())

    pagination = grouped_query.paginate(
        page=page, per_page=per_page, error_out=False
    )

    rows = []
    for item in pagination.items:
        camera_text = item.cameras or ""
        cameras = [c.strip() for c in str(camera_text).split(",") if c and c.strip()]
        rows.append({
            "department_id": item.department_id,
            "room_name": item.room_name,
            "attendance_count": int(item.attendance_count or 0),
            "unique_users": int(item.unique_users or 0),
            "last_time": item.last_time.strftime("%Y-%m-%d %H:%M:%S") if item.last_time else None,
            "cameras": cameras,
        })

    return jsonify({
        "logs": rows,
        "total": pagination.total,
        "page": pagination.page,
        "pages": pagination.pages,
        "mode": "grouped",
    })


@api_bp.route("/logs/today", methods=["GET"])
@api_admin_required()
def get_today_logs():
    """Lấy log điểm danh hôm nay."""
    today = datetime.now().date()
    logs = AttendanceLog.query.filter(
        db.func.date(AttendanceLog.time) == today
    ).order_by(AttendanceLog.time.desc()).all()

    return jsonify({"logs": [log.to_dict() for log in logs]})


@api_bp.route("/logs/all", methods=["DELETE"])
@api_admin_required()
def delete_all_logs():
    """Xóa toàn bộ log điểm danh, tùy chọn xóa luôn ảnh crop tương ứng."""
    delete_crops_q = str(request.args.get("delete_crops", "1")).strip().lower()
    delete_crops = delete_crops_q not in {"0", "false", "no"}

    logs = AttendanceLog.query.with_entities(AttendanceLog.id, AttendanceLog.image).all()
    total_logs = len(logs)
    images = [img for _, img in logs if img]

    try:
        AttendanceLog.query.delete(synchronize_session=False)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Không thể xóa logs: {str(e).strip()}"}), 500

    crop_cleanup = {
        "requested": 0,
        "deleted": 0,
        "missing": 0,
        "skipped": 0,
    }
    if delete_crops and images:
        crop_cleanup = _delete_log_capture_images(images)

    return jsonify({
        "status": "ok",
        "deleted_logs": total_logs,
        "delete_crops": delete_crops,
        "crop_cleanup": crop_cleanup,
    })


@api_bp.route("/logs/user/<int:user_id>", methods=["DELETE"])
@api_admin_required()
def delete_logs_by_user(user_id):
    """Xóa toàn bộ log điểm danh của một user, tùy chọn xóa luôn ảnh crop."""
    delete_crops_q = str(request.args.get("delete_crops", "1")).strip().lower()
    delete_crops = delete_crops_q not in {"0", "false", "no"}

    logs = AttendanceLog.query.filter(AttendanceLog.user_id == user_id).with_entities(
        AttendanceLog.id,
        AttendanceLog.image,
    ).all()
    total_logs = len(logs)
    images = [img for _, img in logs if img]

    if total_logs <= 0:
        return jsonify({
            "status": "ok",
            "deleted_logs": 0,
            "delete_crops": delete_crops,
            "crop_cleanup": {
                "requested": 0,
                "deleted": 0,
                "missing": 0,
                "skipped": 0,
            },
            "user_id": int(user_id),
        })

    try:
        AttendanceLog.query.filter(AttendanceLog.user_id == user_id).delete(synchronize_session=False)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Không thể xóa logs theo user: {str(e).strip()}"}), 500

    crop_cleanup = {
        "requested": 0,
        "deleted": 0,
        "missing": 0,
        "skipped": 0,
    }
    if delete_crops and images:
        crop_cleanup = _delete_log_capture_images(images)

    return jsonify({
        "status": "ok",
        "deleted_logs": int(total_logs),
        "delete_crops": delete_crops,
        "crop_cleanup": crop_cleanup,
        "user_id": int(user_id),
    })


@api_bp.route("/stats", methods=["GET"])
@api_admin_required()
def get_stats():
    """Thống kê tổng quan."""
    today = datetime.now().date()

    total_users = User.query.filter_by(is_active=True).count()
    today_count = AttendanceLog.query.filter(
        db.func.date(AttendanceLog.time) == today
    ).distinct(AttendanceLog.user_id).count()
    total_logs = AttendanceLog.query.count()

    # Người điểm danh gần nhất
    latest = AttendanceLog.query.order_by(AttendanceLog.time.desc()).first()

    return jsonify({
        "total_users": total_users,
        "today_attendance": today_count,
        "total_logs": total_logs,
        "latest": latest.to_dict() if latest else None,
    })
