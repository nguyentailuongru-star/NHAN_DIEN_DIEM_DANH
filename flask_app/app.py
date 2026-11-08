"""Flask Application - main entry point."""

import os
from flask import Flask, abort, send_from_directory
from flask_socketio import SocketIO
from sqlalchemy import event, inspect, text
from sqlalchemy.engine import Engine
from config import Config
from models import db
from auth_utils import current_account

socketio = SocketIO()


@event.listens_for(Engine, "connect")
def _set_sqlite_pragmas(dbapi_conn, _record):
    """Cấu hình mỗi kết nối SQLite: WAL cho đọc/ghi đồng thời, bật khóa ngoại."""
    import sqlite3

    if not isinstance(dbapi_conn, sqlite3.Connection):
        return
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")      # web đọc trong lúc AI ghi không bị khóa
    cur.execute("PRAGMA synchronous=NORMAL")    # an toàn với WAL, ghi nhanh hơn FULL
    cur.execute("PRAGMA foreign_keys=ON")       # SQLite mặc định tắt khóa ngoại
    cur.execute("PRAGMA busy_timeout=30000")
    cur.close()


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Tạo thư mục upload + thư mục chứa file SQLite
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    if app.config["SQLALCHEMY_DATABASE_URI"].startswith("sqlite"):
        os.makedirs(os.path.dirname(os.path.abspath(app.config["SQLITE_PATH"])), exist_ok=True)

    # Init extensions
    db.init_app(app)
    socketio.init_app(app, cors_allowed_origins="*")

    # Register blueprints
    from routes.api import api_bp
    from routes.dashboard import dashboard_bp
    from routes.auth import auth_bp
    app.register_blueprint(api_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(auth_bp)

    @app.context_processor
    def inject_auth_context():
        acc = current_account()
        return {
            "current_account": acc,
            "current_role": acc.role if acc else None,
        }

    @app.route("/uploads/<path:filename>")
    def uploaded_file(filename: str):
        """Serve uploaded files (avatars, face images) from UPLOAD_FOLDER."""
        upload_root = app.config.get("UPLOAD_FOLDER")
        if not upload_root:
            abort(404)
        return send_from_directory(upload_root, filename)

    @app.route("/video/<path:filename>")
    def video_file(filename: str):
        """Serve offline video files from workspace root video/ directory."""
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        video_root = os.path.join(repo_root, "video")
        if not os.path.isdir(video_root):
            abort(404)
        # send_from_directory blocks path traversal; conditional enables range requests.
        return send_from_directory(video_root, filename, conditional=True)

    @app.route("/captures/<path:filename>")
    def capture_file(filename: str):
        """Serve recognition capture images saved by AI service under captures/."""
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        captures_root = os.path.join(repo_root, "captures")

        # Fallback: một số lần chạy AI service có thể lưu tương đối theo cwd → ai_service/captures
        alt_captures_root = os.path.join(repo_root, "ai_service", "captures")

        # Ưu tiên root/captures
        if os.path.isfile(os.path.join(captures_root, filename)):
            return send_from_directory(captures_root, filename, conditional=True)

        if os.path.isfile(os.path.join(alt_captures_root, filename)):
            return send_from_directory(alt_captures_root, filename, conditional=True)

        abort(404)

    # Tạo bảng DB nếu chưa có
    with app.app_context():
        _ensure_schema_compat()
        db.create_all()

    return app


def _ensure_schema_compat():
    """Best-effort schema updates for existing databases without Alembic."""
    inspector = inspect(db.engine)
    tables = set(inspector.get_table_names())

    # Thêm các cột mới nếu DB cũ còn thiếu (cú pháp chung cho SQLite và MySQL).
    add_columns = {
        "users": [
            ("department_id", "INTEGER NULL"),
            ("username", "VARCHAR(50) NULL"),
            ("password_hash", "VARCHAR(255) NULL"),
            ("role", "VARCHAR(20) NOT NULL DEFAULT 'employee'"),
        ],
        "face_upload_requests": [("reviewed_by", "INTEGER NULL")],
        "attention_logs": [("image", "VARCHAR(255) NULL")],
    }
    for table, cols in add_columns.items():
        if table not in tables:
            continue
        existing = {c["name"] for c in inspector.get_columns(table)}
        for col_name, col_def in cols:
            if col_name not in existing:
                db.session.execute(text(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def}"))
        db.session.commit()

    if db.engine.dialect.name == "sqlite":
        return

    # Phần dưới chỉ dành cho DB MySQL cũ (cú pháp riêng của MySQL).
    _ensure_mysql_legacy_schema(inspector, tables)


def _ensure_mysql_legacy_schema(inspector, tables):
    if "users" in tables:
        user_cols = {c["name"] for c in inspector.get_columns("users")}
        try:
            db.session.execute(text("ALTER TABLE users ADD UNIQUE INDEX uq_users_username (username)"))
        except Exception:
            pass

        if "department" in user_cols:
            try:
                db.session.execute(text("ALTER TABLE users DROP INDEX idx_department"))
            except Exception:
                pass
            try:
                db.session.execute(text("ALTER TABLE users DROP COLUMN department"))
            except Exception:
                pass

        db.session.commit()

    # Migrate dữ liệu từ bảng accounts cũ sang users nếu bảng cũ còn tồn tại.
    if "accounts" in tables and "users" in tables:
        try:
            db.session.execute(text(
                """
                UPDATE users u
                JOIN accounts a ON a.user_id = u.id
                SET u.username = COALESCE(u.username, a.username),
                    u.password_hash = COALESCE(u.password_hash, a.password_hash),
                    u.role = COALESCE(u.role, a.role),
                    u.is_active = COALESCE(u.is_active, a.is_active)
                WHERE a.user_id IS NOT NULL
                """
            ))
            db.session.execute(text(
                """
                UPDATE face_upload_requests fur
                JOIN accounts a ON fur.reviewed_by = a.id
                SET fur.reviewed_by = a.user_id
                WHERE a.user_id IS NOT NULL
                """
            ))
            db.session.commit()
        except Exception:
            db.session.rollback()

    if "face_upload_requests" in tables:
        try:
            db.session.execute(text("ALTER TABLE face_upload_requests DROP FOREIGN KEY face_upload_requests_ibfk_2"))
            db.session.commit()
        except Exception:
            db.session.rollback()

        try:
            db.session.execute(text("ALTER TABLE face_upload_requests ADD CONSTRAINT fk_face_upload_reviewed_user FOREIGN KEY (reviewed_by) REFERENCES users(id) ON DELETE SET NULL"))
            db.session.commit()
        except Exception:
            db.session.rollback()

    if "accounts" in tables:
        try:
            db.session.execute(text("DROP TABLE accounts"))
            db.session.commit()
        except Exception:
            db.session.rollback()


# SocketIO events
@socketio.on("connect")
def handle_connect():
    print("Client connected")


@socketio.on("disconnect")
def handle_disconnect():
    print("Client disconnected")


if __name__ == "__main__":
    app = create_app()
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
