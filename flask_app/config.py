"""Cấu hình Flask app."""

import os


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "nhan-dien-diem-danh-secret-key-2024")
    CAMERA_SOURCES = [
        "rtsp://admin:admin1630688@192.168.1.53:554/onvif1",
        "rtsp://admin:L2295266@192.168.1.135:54/cam/realmonitor?channel=1&subtype=0",
    ]
    DEFAULT_CAMERA_SOURCE = os.environ.get("DEFAULT_CAMERA_SOURCE", CAMERA_SOURCES[0])

    # SQLite: toàn bộ dữ liệu nằm trong 1 file, không cần cài MySQL server.
    # Có thể đổi đường dẫn qua biến môi trường SQLITE_PATH,
    # hoặc dùng DB khác bằng DATABASE_URL (vd: mysql+pymysql://user:pass@host/db).
    REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    SQLITE_PATH = os.environ.get(
        "SQLITE_PATH", os.path.join(REPO_ROOT, "database", "diem_danh.db")
    )

    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", "sqlite:///" + SQLITE_PATH.replace("\\", "/")
    )
    SQLALCHEMY_ENGINE_OPTIONS = (
        {
            # Flask-SocketIO + thread xử lý video dùng chung engine → cho phép đa luồng.
            # timeout: chờ tối đa 30s khi DB đang bị ghi thay vì báo "database is locked".
            "connect_args": {"check_same_thread": False, "timeout": 30},
        }
        if SQLALCHEMY_DATABASE_URI.startswith("sqlite")
        else {"pool_pre_ping": True}
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Upload
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
