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
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000|max_delay;500000"
        cap = cv2.VideoCapture(source_value, cv2.CAP_FFMPEG)
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


def _camera_stream_generator(cap, cv2):
    try:
        last_ok_at = time.monotonic()
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 80]

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                if (time.monotonic() - last_ok_at) > 10.0:
                    break
                time.sleep(0.15)
                continue

            last_ok_at = time.monotonic()
            encoded_ok, jpg = cv2.imencode(".jpg", frame, encode_params)
            if not encoded_ok:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
            )
            time.sleep(0.03)
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
    """Danh sách camera để hiển thị: chỉ tên + IP, KHÔNG đưa user/mật khẩu RTSP ra trang web."""
    from urllib.parse import urlparse

    cams = []
    for idx, src in enumerate(Config.CAMERA_SOURCES or [Config.DEFAULT_CAMERA_SOURCE]):
        txt = str(src).strip()
        if txt.isdigit():
            host = f"Webcam {txt}"
        else:
            try:
                u = urlparse(txt)
                host = u.hostname or txt
            except ValueError:
                host = "Camera"
        cams.append({"index": idx, "name": f"Camera {idx + 1}", "host": host})
    return cams


@dashboard_bp.route("/camera-stream")
@page_admin_required
def camera_stream():
    """Phát MJPEG trực tiếp từ camera/source cho dashboard."""
    cam_idx = request.args.get("cam", type=int)
    sources = Config.CAMERA_SOURCES or [Config.DEFAULT_CAMERA_SOURCE]
    if cam_idx is not None and 0 <= cam_idx < len(sources):
        camera_source = str(sources[cam_idx]).strip()
    else:
        camera_source = (request.args.get("source") or Config.DEFAULT_CAMERA_SOURCE).strip() or Config.DEFAULT_CAMERA_SOURCE
    try:
        cap, cv2 = _open_camera_capture(camera_source)
    except Exception as e:
        current_app.logger.error("Camera stream open failed: %s", e)
        return Response(str(e).strip(), status=503, mimetype="text/plain")

    return Response(
        stream_with_context(_camera_stream_generator(cap, cv2)),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )


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
    return render_template(
        "group_camera.html",
        default_camera_source=Config.DEFAULT_CAMERA_SOURCE,
        camera_sources=Config.CAMERA_SOURCES,
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
