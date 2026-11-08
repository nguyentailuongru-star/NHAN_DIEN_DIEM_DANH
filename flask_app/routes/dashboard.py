"""Dashboard routes - render HTML pages."""

import os
import time

from flask import Blueprint, Response, current_app, render_template, request, stream_with_context
from auth_utils import page_admin_required, page_employee_required
from config import Config

dashboard_bp = Blueprint("dashboard", __name__)


def _camera_source_value(raw):
    txt = str(raw or Config.DEFAULT_CAMERA_SOURCE).strip()
    if txt.isdigit():
        return int(txt)
    return txt


def _open_camera_capture(camera_source):
    try:
        import cv2
    except Exception as e:
        raise RuntimeError(f"Không import được OpenCV: {str(e).strip()}")

    source_value = _camera_source_value(camera_source)
    if isinstance(source_value, str) and source_value.lower().startswith("rtsp://"):
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|max_delay;500000"
        # Timeout 5s (FFmpeg mới bỏ "stimeout" → nếu không đặt ở đây sẽ treo ~30s khi sai IP)
        cap = cv2.VideoCapture(source_value, cv2.CAP_FFMPEG, [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000,
        ])
    else:
        cap = cv2.VideoCapture(source_value)

    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass

    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Không mở được camera/source: {camera_source}")

    return cap, cv2


def _camera_stream_generator(cap, cv2, cam_idx=None):
    from routes.cameras import preview_update

    try:
        last_ok_at = time.monotonic()
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                if (time.monotonic() - last_ok_at) > 10.0:
                    if cam_idx is not None:
                        preview_update(cam_idx, "error", "Mất kết nối camera (không nhận được hình)")
                    break
                time.sleep(0.15)
                continue

            last_ok_at = time.monotonic()
            if cam_idx is not None:
                preview_update(cam_idx, "online")
            encoded_ok, jpg = cv2.imencode(".jpg", frame, encode_params)
            if not encoded_ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
            )
            time.sleep(0.03)
    except GeneratorExit:
        # Người xem đóng trang → không phải lỗi camera
        if cam_idx is not None:
            preview_update(cam_idx, "idle")
        raise
    finally:
        try:
            cap.release()
        except Exception:
            pass


@dashboard_bp.route("/")
@page_admin_required
def index():
    """Trang dashboard chính."""
    return render_template("dashboard.html", cameras=_camera_list())


def _camera_list():
    """Camera đang bật (cấu hình trên trang Cấu hình camera). KHÔNG đưa user/mật khẩu RTSP ra trang web."""
    from routes.cameras import list_cameras

    return [
        {"id": c.id, "code": c.code, "name": c.name, "host": c.to_dict()["host"]}
        for c in list_cameras(enabled_only=True)
    ]


@dashboard_bp.route("/cameras")
@page_admin_required
def cameras_page():
    """Trang cấu hình camera (thêm/sửa/xóa luồng RTSP)."""
    return render_template("cameras.html")


@dashboard_bp.route("/camera-stream")
@page_admin_required
def camera_stream():
    """Phát MJPEG trực tiếp từ camera đã cấu hình (?cam=<id>) cho dashboard."""
    from models import Camera
    from routes.cameras import preview_update

    cam_id = request.args.get("cam", type=int)
    cam = Camera.query.get(cam_id) if cam_id is not None else None
    if cam is not None:
        camera_source = cam.rtsp_url.strip()
    else:
        camera_source = (request.args.get("source") or Config.DEFAULT_CAMERA_SOURCE).strip() or Config.DEFAULT_CAMERA_SOURCE

    mjpeg_headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
    }

    # AI service đang chạy → dùng ảnh từ kết nối camera chạy ngầm của AI (không mở thêm RTSP)
    if cam is not None and cam.enabled:
        from routes.cameras import ai_camera_info

        ai_alive, ai_cam = ai_camera_info(cam.code)
        if ai_alive:
            if ai_cam.get("state") == "error":
                preview_update(cam.id, "error", ai_cam.get("message") or "Camera đang lỗi")
                return Response("Camera đang lỗi", status=503, mimetype="text/plain")
            return Response(
                stream_with_context(_live_relay_generator(cam.id, cam.code)),
                mimetype="multipart/x-mixed-replace; boundary=frame",
                headers=mjpeg_headers,
            )

    # AI service không chạy → Flask tự kết nối thẳng tới camera
    preview_idx = cam.id if cam is not None else None
    if preview_idx is not None:
        preview_update(preview_idx, "connecting", "Đang kết nối...")
    try:
        cap, cv2 = _open_camera_capture(camera_source)
    except Exception as e:
        current_app.logger.error("Camera stream open failed (cam=%s): %s", preview_idx, e)
        if preview_idx is not None:
            preview_update(preview_idx, "error", "Không mở được camera (sai IP, camera tắt hoặc mất mạng)")
        # Không trả URL/mật khẩu RTSP về trình duyệt
        return Response("Không mở được camera", status=503, mimetype="text/plain")

    return Response(
        stream_with_context(_camera_stream_generator(cap, cv2, preview_idx)),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers=mjpeg_headers,
    )


LIVE_WAIT_SEC = 25  # chờ ảnh đầu tiên (AI có thể đang thử kết nối 3 lần) / tối đa không có ảnh mới


def _live_relay_generator(cam_id, code):
    """Phát lại ảnh mà AI service gửi sang (camera chạy ngầm liên tục ở AI)."""
    from routes.cameras import ai_camera_info, live_frame, mark_viewer, preview_update

    last_ts = 0.0
    last_new = time.time()
    preview_update(cam_id, "connecting", "Đang lấy hình từ AI service...")
    try:
        while True:
            mark_viewer(code)
            item = live_frame(code)
            if item and item[0] > last_ts:
                last_ts, jpg = item
                last_new = time.time()
                preview_update(cam_id, "online")
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            else:
                ai_alive, ai_cam = ai_camera_info(code)
                if ai_cam.get("state") == "error":
                    preview_update(cam_id, "error", ai_cam.get("message") or "Camera đang lỗi")
                    break
                if not ai_alive:
                    preview_update(cam_id, "error", "AI service đã dừng")
                    break
                if time.time() - last_new > LIVE_WAIT_SEC:
                    preview_update(cam_id, "error", "Không nhận được hình từ camera")
                    break
            time.sleep(0.05)
    except GeneratorExit:
        preview_update(cam_id, "idle")
        raise


@dashboard_bp.route("/video-recognition")
@page_admin_required
def video_recognition():
    """Trang nhận dạng/điểm danh từ video offline."""
    return render_template("video_recognition.html")


@dashboard_bp.route("/behavior-video")
@page_admin_required
def behavior_video_page():
    """Trang phân tích hành vi từ video (không nhận diện khuôn mặt)."""
    return render_template("behavior_video.html")


@dashboard_bp.route("/group-camera")
@page_admin_required
def group_camera_page():
    """Trang gom khuôn mặt từ camera."""
    from routes.cameras import list_cameras

    cams = list_cameras(enabled_only=True)
    sources = [c.rtsp_url for c in cams] or list(Config.CAMERA_SOURCES or [])
    return render_template(
        "group_camera.html",
        default_camera_source=sources[0] if sources else Config.DEFAULT_CAMERA_SOURCE,
        camera_sources=sources,
    )


@dashboard_bp.route("/group-video")
@page_admin_required
def group_video_page():
    """Trang gom khuôn mặt từ video."""
    return render_template("group_video.html")


@dashboard_bp.route("/users")
@page_admin_required
def users_page():
    """Trang quản lý users."""
    return render_template("users.html")


@dashboard_bp.route("/logs")
@page_admin_required
def logs_page():
    """Trang xem lịch sử điểm danh."""
    return render_template("logs.html")


@dashboard_bp.route("/face-uploads")
@page_admin_required
def face_upload_requests_page():
    """Trang admin duyệt ảnh nhân viên upload."""
    return render_template("face_uploads.html")


@dashboard_bp.route("/teacher-notebook")
@page_admin_required
def teacher_notebook_page():
    """Trang sổ tay giáo viên theo lớp/phòng ban."""
    return render_template("teacher_notebook.html")


@dashboard_bp.route("/attention-dashboard")
@page_admin_required
def attention_dashboard_page():
    """Trang dashboard hành vi lớp theo phòng/lớp."""
    return render_template("attention_dashboard.html")


@dashboard_bp.route("/me")
@page_employee_required
def my_attendance_page():
    """Trang nhân viên: xem điểm danh của tôi + upload ảnh."""
    return render_template("my_attendance.html")
