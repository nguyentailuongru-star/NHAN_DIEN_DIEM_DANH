"""SQLAlchemy models."""

from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import json
import numpy as np

db = SQLAlchemy()


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(50), unique=True, nullable=True)
    password_hash = db.Column(db.String(255), nullable=True)
    role = db.Column(db.String(20), nullable=False, default="employee")
    name = db.Column(db.String(100), nullable=False)
    employee_id = db.Column(db.String(50), unique=True, nullable=True)
    department_id = db.Column(db.Integer, db.ForeignKey("departments.id"), nullable=True)
    embedding = db.Column(db.Text, nullable=True)  # JSON string of float list
    avatar = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
    is_active = db.Column(db.Boolean, default=True)

    logs = db.relationship("AttendanceLog", backref="user", lazy="dynamic")
    face_upload_requests = db.relationship(
        "FaceUploadRequest",
        backref="employee_user",
        foreign_keys="FaceUploadRequest.user_id",
        lazy="dynamic",
    )
    reviewed_upload_requests = db.relationship("FaceUploadRequest", backref="reviewer_user", foreign_keys="FaceUploadRequest.reviewed_by", lazy="dynamic")

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        if not self.password_hash:
            return False
        return check_password_hash(self.password_hash, password)

    def set_embedding(self, embedding_array):
        """Lưu numpy array thành JSON string."""
        if isinstance(embedding_array, np.ndarray):
            self.embedding = json.dumps(embedding_array.tolist())
        else:
            self.embedding = json.dumps(embedding_array)

    def get_embedding(self):
        """Đọc embedding từ JSON string thành list."""
        if self.embedding:
            return json.loads(self.embedding)
        return None

    def to_dict(self):
        department_name = self.department_rel.name if self.department_rel else None
        return {
            "id": self.id,
            "username": self.username,
            "role": self.role,
            "name": self.name,
            "employee_id": self.employee_id,
            "department": department_name,
            "department_id": self.department_id,
            "avatar": self.avatar,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else None,
            "is_active": self.is_active,
        }


class Camera(db.Model):
    """Camera RTSP do admin cấu hình (thay thế danh sách mặc định trong config)."""

    __tablename__ = "cameras"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    code = db.Column(db.String(50), unique=True, nullable=False)  # mã camera ghi trong log điểm danh
    name = db.Column(db.String(100), nullable=False)
    rtsp_url = db.Column(db.String(500), nullable=False)
    enabled = db.Column(db.Boolean, nullable=False, default=True)
    sort_order = db.Column(db.Integer, nullable=False, default=0)
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    def url_parts(self):
        """Tách URL thành các trường cho form sửa (KHÔNG trả mật khẩu)."""
        from urllib.parse import urlsplit, unquote

        url = (self.rtsp_url or "").strip()
        if url.isdigit():
            return {"kind": "webcam", "webcam_index": int(url)}
        try:
            u = urlsplit(url)
            if u.scheme.lower() != "rtsp":
                return {"kind": "custom", "host": u.hostname or ""}
            path = u.path or ""
            if u.query:
                path += "?" + u.query
            return {
                "kind": "rtsp",
                "host": u.hostname or "",
                "port": u.port or 554,
                "username": unquote(u.username or ""),
                "has_password": bool(u.password),
                "path": path,
            }
        except ValueError:
            return {"kind": "custom"}

    def masked_url(self):
        from urllib.parse import urlsplit

        url = (self.rtsp_url or "").strip()
        try:
            u = urlsplit(url)
            if u.password:
                netloc = f"{u.username}:••••@{u.hostname}" + (f":{u.port}" if u.port else "")
                return u._replace(netloc=netloc).geturl()
        except ValueError:
            pass
        return url

    def to_dict(self):
        parts = self.url_parts()
        return {
            "id": self.id,
            "code": self.code,
            "name": self.name,
            "enabled": bool(self.enabled),
            "sort_order": self.sort_order,
            "url_masked": self.masked_url(),
            "host": parts.get("host") or (f"Webcam {parts.get('webcam_index')}" if parts.get("kind") == "webcam" else ""),
            "parts": parts,
            "updated_at": self.updated_at.strftime("%Y-%m-%d %H:%M:%S") if self.updated_at else None,
        }


class Department(db.Model):
    __tablename__ = "departments"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.now)
    is_active = db.Column(db.Boolean, default=True)

    users = db.relationship("User", backref="department_rel", lazy="dynamic")

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else None,
            "is_active": self.is_active,
        }


class FaceUploadRequest(db.Model):
    __tablename__ = "face_upload_requests"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    files_json = db.Column(db.Text, nullable=False)  # JSON list[str]
    status = db.Column(db.String(20), nullable=False, default="pending")  # pending/approved/rejected
    note = db.Column(db.String(255), nullable=True)
    reviewed_by = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
    reviewed_at = db.Column(db.DateTime, nullable=True)

    def set_files(self, files):
        self.files_json = json.dumps(list(files or []), ensure_ascii=False)

    def get_files(self):
        try:
            return json.loads(self.files_json or "[]")
        except Exception:
            return []

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "user_name": self.employee_user.name if self.employee_user else None,
            "status": self.status,
            "note": self.note,
            "reviewed_by": self.reviewer_user.username if self.reviewer_user else None,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else None,
            "reviewed_at": self.reviewed_at.strftime("%Y-%m-%d %H:%M:%S") if self.reviewed_at else None,
            "files": self.get_files(),
        }


class AttendanceLog(db.Model):
    __tablename__ = "attendance_logs"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    user_name = db.Column(db.String(100), nullable=True)
    camera_id = db.Column(db.String(50), nullable=True)
    time = db.Column(db.DateTime, default=datetime.now, index=True)
    image = db.Column(db.String(255), nullable=True)
    similarity = db.Column(db.Float, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "user_name": self.user_name or (self.user.name if self.user else "Unknown"),
            "camera_id": self.camera_id,
            "time": self.time.strftime("%Y-%m-%d %H:%M:%S") if self.time else None,
            "image": self.image,
            "similarity": self.similarity,
        }


class AttentionLog(db.Model):
    __tablename__ = "attention_logs"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True, index=True)
    user_name = db.Column(db.String(100), nullable=True)
    camera_id = db.Column(db.String(50), nullable=True)
    time = db.Column(db.DateTime, default=datetime.now, index=True)
    yaw = db.Column(db.Float, nullable=True)
    pitch = db.Column(db.Float, nullable=True)
    roll = db.Column(db.Float, nullable=True)
    attention_score = db.Column(db.Float, nullable=True)
    attentive = db.Column(db.Boolean, nullable=False, default=False)
    source = db.Column(db.String(30), nullable=False, default="insightface")
    image = db.Column(db.String(255), nullable=True)

    user = db.relationship("User", backref=db.backref("attention_logs", lazy="dynamic"))

    @staticmethod
    def _behavior_names():
        return ["distracted", "focused", "raising_hand", "sleep", "using_phone"]

    def _behavior_label_from_source(self):
        src = (self.source or "").strip()
        if src.startswith("sb:"):
            # Format: sb:<class_id>:<label>
            parts = src.split(":", 2)
            if len(parts) == 3 and parts[2]:
                return parts[2]

        try:
            cid = int(self.yaw) if self.yaw is not None else -1
        except Exception:
            cid = -1

        names = self._behavior_names()
        if 0 <= cid < len(names):
            return names[cid]
        return "Unknown"

    def _behavior_class_id(self):
        try:
            return int(self.yaw) if self.yaw is not None else None
        except Exception:
            return None

    def to_dict(self):
        behavior_label = self._behavior_label_from_source()
        behavior_class_id = self._behavior_class_id()
        return {
            "id": self.id,
            "user_id": self.user_id,
            "user_name": self.user_name or (self.user.name if self.user else "Unknown"),
            "camera_id": self.camera_id,
            "time": self.time.strftime("%Y-%m-%d %H:%M:%S") if self.time else None,
            "behavior_class_id": behavior_class_id,
            "behavior_label": behavior_label,
            "behavior_confidence": self.attention_score,
            "engaged": bool(self.attentive),
            # Giữ lại key cũ để tương thích các màn hình/clients chưa nâng cấp.
            "yaw": self.yaw,
            "pitch": self.pitch,
            "roll": self.roll,
            "attention_score": self.attention_score,
            "attentive": bool(self.attentive),
            "source": self.source,
        }


class TeacherNotebookEntry(db.Model):
    __tablename__ = "teacher_notebook_entries"
    __table_args__ = (
        db.UniqueConstraint("department_id", "lesson_date", "period_no", name="uq_teacher_notebook_department_date_period"),
    )

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    department_id = db.Column(db.Integer, db.ForeignKey("departments.id", ondelete="CASCADE"), nullable=False)
    lesson_date = db.Column(db.Date, nullable=False)
    period_no = db.Column(db.Integer, nullable=False)
    lesson_content = db.Column(db.Text, nullable=False)
    note = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    department = db.relationship("Department", backref=db.backref("teacher_notebook_entries", lazy="dynamic"))

    def to_dict(self):
        return {
            "id": self.id,
            "department_id": self.department_id,
            "department_name": self.department.name if self.department else None,
            "lesson_date": self.lesson_date.strftime("%Y-%m-%d") if self.lesson_date else None,
            "period_no": self.period_no,
            "lesson_content": self.lesson_content,
            "note": self.note,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else None,
            "updated_at": self.updated_at.strftime("%Y-%m-%d %H:%M:%S") if self.updated_at else None,
        }


class TeacherNotebookAttendanceMark(db.Model):
    __tablename__ = "teacher_notebook_attendance_marks"
    __table_args__ = (
        db.UniqueConstraint("department_id", "user_id", "lesson_date", name="uq_teacher_notebook_attendance_mark"),
    )

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    department_id = db.Column(db.Integer, db.ForeignKey("departments.id"), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    lesson_date = db.Column(db.Date, nullable=False)
    is_present = db.Column(db.Boolean, nullable=False, default=True)
    updated_by = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.now)
    updated_at = db.Column(db.DateTime, default=datetime.now, onupdate=datetime.now)

    department = db.relationship("Department", backref=db.backref("teacher_notebook_attendance_marks", lazy="dynamic"))
    student = db.relationship("User", foreign_keys=[user_id])
    editor = db.relationship("User", foreign_keys=[updated_by])

    def to_dict(self):
        return {
            "id": self.id,
            "department_id": self.department_id,
            "user_id": self.user_id,
            "lesson_date": self.lesson_date.strftime("%Y-%m-%d") if self.lesson_date else None,
            "is_present": bool(self.is_present),
            "updated_by": self.updated_by,
            "updated_at": self.updated_at.strftime("%Y-%m-%d %H:%M:%S") if self.updated_at else None,
        }
