"""Camera: cấu hình (admin), trạng thái và nút khởi động lại.

- Danh sách camera lưu trong bảng `cameras` (SQLite). Lần đầu chạy nạp từ cấu hình mặc định
  (ai_service/config.py → CAMERAS, nếu không có thì flask_app/config.py → CAMERA_SOURCES).
- AI service gọi /api/cameras/ai-sync mỗi 2 giây: báo trạng thái từng camera, nhận lại danh sách
  camera đang bật + lệnh khởi động lại. Nhờ vậy admin sửa camera trên web là AI áp dụng ngay.
- Không tự kết nối lại vô hạn: lỗi → trạng thái "error" (báo đỏ), chờ admin bấm "Khởi động lại".
"""

import base64
import importlib.util
import os
import re
import threading
import time
from datetime import datetime
from urllib.parse import quote, urlsplit

from flask import Blueprint, jsonify, request

from auth_utils import api_admin_required, is_local_request
from config import Config
from models import Camera, db

cameras_bp = Blueprint("cameras", __name__, url_prefix="/api/cameras")

AI_SYNC_TIMEOUT_SEC = 10   # quá thời gian này không nhận sync → coi AI service không chạy
PREVIEW_STALE_SEC = 10     # luồng xem quá thời gian này không có hình → báo lỗi

_lock = threading.Lock()
_ai_state = {"last_sync": 0.0, "cameras": {}}   # cameras: {code: {...}}
_pending_restart_codes = set()
_preview = {}                                   # {camera_id: {"state", "message", "last_frame", "since"}}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _now_text():
    return datetime.now().strftime("%H:%M:%S %d/%m/%Y")


# ======================================================================= DB
def _default_cameras():
    """Danh sách mặc định của hệ thống: ưu tiên ai_service/config.py (có mã + tên camera)."""
    try:
        path = os.path.join(REPO_ROOT, "ai_service", "config.py")
        spec = importlib.util.spec_from_file_location("_ai_default_config", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cams = [
            {"code": str(c["id"]), "name": str(c.get("name") or f"Camera {c['id']}"), "rtsp_url": str(c["rtsp_url"])}
            for c in getattr(mod, "CAMERAS", []) if c.get("rtsp_url")
        ]
        if cams:
            return cams
    except Exception:
        pass
    return [
        {"code": str(i + 1), "name": f"Camera {i + 1}", "rtsp_url": str(src)}
        for i, src in enumerate(Config.CAMERA_SOURCES or [])
    ]


def ensure_default_cameras():
    """Gọi khi khởi động app: bảng cameras trống → nạp danh sách mặc định."""
    if Camera.query.count() > 0:
        return
    for i, c in enumerate(_default_cameras()):
        db.session.add(Camera(code=c["code"], name=c["name"], rtsp_url=c["rtsp_url"], enabled=True, sort_order=i))
    db.session.commit()


def list_cameras(enabled_only=False):
    q = Camera.query
    if enabled_only:
        q = q.filter(Camera.enabled.is_(True))
    return q.order_by(Camera.sort_order.asc(), Camera.id.asc()).all()


# ================================================================== preview
def preview_update(camera_id, state, message=""):
    with _lock:
        cur = _preview.setdefault(camera_id, {})
        if state == "online":
            cur["last_frame"] = time.time()
        if cur.get("state") != state or (message and cur.get("message") != message):
            cur.update(state=state, message=message, since=_now_text())


def _preview_snapshot(camera_id):
    cur = dict(_preview.get(camera_id) or {})
    if cur.get("state") == "online" and time.time() - float(cur.get("last_frame") or 0) > PREVIEW_STALE_SEC:
        cur["state"] = "error"
        cur["message"] = f"Không nhận được hình quá {PREVIEW_STALE_SEC} giây"
    return cur or {"state": "idle", "message": ""}


# =============================================================== AI service
@cameras_bp.route("/ai-sync", methods=["POST"])
def ai_sync():
    """AI service báo trạng thái + nhận danh sách camera và lệnh khởi động lại (chỉ máy local)."""
    if not is_local_request():
        return jsonify({"error": "forbidden"}), 403
    data = request.get_json(silent=True) or {}
    cams = list_cameras(enabled_only=True)
    with _lock:
        _ai_state["last_sync"] = time.time()
        _ai_state["cameras"] = {str(c.get("code")): c for c in (data.get("cameras") or []) if c.get("code")}
        restart = sorted(_pending_restart_codes)
        _pending_restart_codes.clear()
    return jsonify({
        "cameras": [{"code": c.code, "name": c.name, "rtsp_url": c.rtsp_url} for c in cams],
        "restart": restart,
    })


def _ai_status_for(cam, ai_alive, ai_cams, pending):
    if not cam.enabled:
        return {"state": "disabled", "message": "Camera đang tắt"}
    if not ai_alive:
        return {"state": "offline", "message": "AI service chưa chạy"}
    if cam.code in pending:
        return {"state": "connecting", "message": "Đang gửi lệnh khởi động lại..."}
    c = ai_cams.get(cam.code)
    if not c:
        return {"state": "connecting", "message": "AI đang nhận cấu hình camera..."}
    return {"state": c.get("state"), "message": c.get("message", ""), "since": c.get("since")}


# ================================================================ admin API
@cameras_bp.route("", methods=["GET"])
@api_admin_required()
def cameras_list():
    return jsonify({"cameras": [c.to_dict() for c in list_cameras()]})


@cameras_bp.route("/status", methods=["GET"])
@api_admin_required()
def cameras_status():
    cams = list_cameras()
    with _lock:
        ai_alive = (time.time() - _ai_state["last_sync"]) <= AI_SYNC_TIMEOUT_SEC
        ai_cams = dict(_ai_state["cameras"])
        pending = set(_pending_restart_codes)
        items = [{
            "id": c.id,
            "code": c.code,
            "enabled": bool(c.enabled),
            "preview": _preview_snapshot(c.id),
            "ai": _ai_status_for(c, ai_alive, ai_cams, pending),
        } for c in cams]
    return jsonify({"ai_service_running": ai_alive, "cameras": items})


@cameras_bp.route("/<int:cam_id>/restart", methods=["POST"])
@api_admin_required()
def camera_restart(cam_id):
    cam = Camera.query.get_or_404(cam_id)
    with _lock:
        _preview.pop(cam.id, None)
        ai_alive = (time.time() - _ai_state["last_sync"]) <= AI_SYNC_TIMEOUT_SEC
        if ai_alive and cam.enabled:
            _pending_restart_codes.add(cam.code)
    return jsonify({"status": "ok", "ai_restart_sent": bool(ai_alive and cam.enabled)})


# ---------------------------------------------------------- cấu hình (CRUD)
_CODE_RE = re.compile(r"^[A-Za-z0-9_-]{1,50}$")


def _build_url(data, existing: Camera | None):
    """Tạo URL RTSP từ form. Mật khẩu để trống khi sửa = giữ mật khẩu cũ."""
    mode = (data.get("mode") or "fields").strip()
    if mode == "keep":
        if existing is None or not existing.rtsp_url:
            raise ValueError("Chưa nhập thông tin luồng camera")
        return existing.rtsp_url
    if mode == "url":
        url = (data.get("rtsp_url") or "").strip()
        if not url:
            raise ValueError("Chưa nhập URL camera")
        return url
    if mode == "webcam":
        return str(int(data.get("webcam_index") or 0))

    host = (data.get("host") or "").strip()
    if not host:
        raise ValueError("Chưa nhập địa chỉ IP camera")
    if not re.match(r"^[A-Za-z0-9.\-]+$", host):
        raise ValueError("Địa chỉ IP/host không hợp lệ")
    try:
        port = int(data.get("port") or 554)
    except (TypeError, ValueError):
        raise ValueError("Cổng không hợp lệ")
    if not 1 <= port <= 65535:
        raise ValueError("Cổng phải từ 1 đến 65535")
    user = (data.get("username") or "").strip()
    password = data.get("password") or ""

    # Mật khẩu (đã mã hóa URL) đưa vào URL mới
    pw_encoded = quote(password, safe="") if password else ""
    if not password and existing is not None:
        # Sửa camera mà để trống mật khẩu → giữ mật khẩu cũ
        try:
            old = urlsplit(existing.rtsp_url)
            if old.password:
                pw_encoded = old.password
        except ValueError:
            pass

    path = (data.get("path") or "").strip()
    if path and not path.startswith("/"):
        path = "/" + path

    auth = ""
    if user:
        auth = quote(user, safe="") + (":" + pw_encoded if pw_encoded else "") + "@"
    return f"rtsp://{auth}{host}:{port}{path}"


def _apply(cam: Camera, data, is_new):
    name = (data.get("name") or "").strip()
    code = (data.get("code") or "").strip()
    if not name:
        raise ValueError("Chưa nhập tên camera")
    if not _CODE_RE.match(code):
        raise ValueError("Mã camera chỉ gồm chữ, số, '-', '_' (tối đa 50 ký tự)")
    dup = Camera.query.filter(Camera.code == code, Camera.id != (cam.id or 0)).first()
    if dup:
        raise ValueError(f"Mã camera '{code}' đã được dùng cho '{dup.name}'")
    cam.name = name
    cam.code = code
    cam.rtsp_url = _build_url(data, None if is_new else cam)
    cam.enabled = bool(data.get("enabled", True))


@cameras_bp.route("", methods=["POST"])
@api_admin_required()
def camera_create():
    data = request.get_json(silent=True) or {}
    cam = Camera(sort_order=(db.session.query(db.func.max(Camera.sort_order)).scalar() or 0) + 1)
    try:
        _apply(cam, data, is_new=True)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    db.session.add(cam)
    db.session.commit()
    return jsonify({"status": "ok", "camera": cam.to_dict()})


@cameras_bp.route("/<int:cam_id>", methods=["PUT"])
@api_admin_required()
def camera_update(cam_id):
    cam = Camera.query.get_or_404(cam_id)
    data = request.get_json(silent=True) or {}
    old_code = cam.code
    try:
        _apply(cam, data, is_new=False)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    db.session.commit()
    with _lock:
        _preview.pop(cam.id, None)
        _pending_restart_codes.discard(old_code)
    return jsonify({"status": "ok", "camera": cam.to_dict()})


@cameras_bp.route("/<int:cam_id>", methods=["DELETE"])
@api_admin_required()
def camera_delete(cam_id):
    cam = Camera.query.get_or_404(cam_id)
    db.session.delete(cam)
    db.session.commit()
    with _lock:
        _preview.pop(cam_id, None)
    return jsonify({"status": "ok"})


@cameras_bp.route("/reset-defaults", methods=["POST"])
@api_admin_required()
def camera_reset_defaults():
    """Xóa cấu hình trên web, quay về danh sách mặc định của hệ thống."""
    Camera.query.delete()
    db.session.commit()
    ensure_default_cameras()
    with _lock:
        _preview.clear()
    return jsonify({"status": "ok", "cameras": [c.to_dict() for c in list_cameras()]})


@cameras_bp.route("/test", methods=["POST"])
@api_admin_required()
def camera_test():
    """Thử kết nối 1 camera (từ form hoặc camera đã lưu), trả về ảnh chụp nhỏ nếu thành công."""
    data = request.get_json(silent=True) or {}
    existing = Camera.query.get(int(data["id"])) if data.get("id") else None
    try:
        url = _build_url(data, existing) if data.get("mode") else (existing.rtsp_url if existing else "")
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    if not url:
        return jsonify({"ok": False, "error": "Thiếu thông tin camera"}), 400

    try:
        import cv2
    except Exception:
        return jsonify({"ok": False, "error": "Thiếu OpenCV trên máy chủ"}), 500

    started = time.time()
    if url.isdigit():
        cap = cv2.VideoCapture(int(url))
    else:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000,
        ])
    try:
        if not cap.isOpened():
            return jsonify({"ok": False, "error": "Không kết nối được: sai IP/cổng, sai tài khoản/mật khẩu, sai đường dẫn hoặc camera đang tắt"})
        frame = None
        for _ in range(15):
            ok, frame = cap.read()
            if ok and frame is not None:
                break
            frame = None
        if frame is None:
            return jsonify({"ok": False, "error": "Kết nối được nhưng không nhận được hình"})
        h, w = frame.shape[:2]
        scale = 360.0 / max(h, w)
        thumb = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame
        _, jpg = cv2.imencode(".jpg", thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        return jsonify({
            "ok": True,
            "resolution": f"{w}x{h}",
            "elapsed_ms": int((time.time() - started) * 1000),
            "snapshot": "data:image/jpeg;base64," + base64.b64encode(jpg.tobytes()).decode("ascii"),
        })
    finally:
        cap.release()
