"""Auth helpers: session utilities and decorators for page/API access."""

from functools import wraps
from flask import session, redirect, url_for, jsonify, request, g
from models import User


def _load_current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    return User.query.filter_by(id=user_id, is_active=True).first()


def current_user():
    if hasattr(g, "_current_user"):
        return g._current_user
    g._current_user = _load_current_user()
    return g._current_user


def current_account():
    """Alias giữ tương thích template cũ."""
    return current_user()


def current_role():
    u = current_user()
    return u.role if u else None


def is_local_request() -> bool:
    addr = (request.remote_addr or "").strip()
    return addr in {"127.0.0.1", "::1", "localhost"}


def login_user(user: User):
    session["user_id"] = user.id


def logout_user():
    session.pop("user_id", None)


def page_login_required(view_fn):
    @wraps(view_fn)
    def _wrapped(*args, **kwargs):
        if not current_user():
            return redirect(url_for("auth.login_page"))
        return view_fn(*args, **kwargs)

    return _wrapped


def page_admin_required(view_fn):
    @wraps(view_fn)
    def _wrapped(*args, **kwargs):
        u = current_user()
        if not u:
            return redirect(url_for("auth.login_page"))
        if u.role != "admin":
            return redirect(url_for("dashboard.my_attendance_page"))
        return view_fn(*args, **kwargs)

    return _wrapped


def page_employee_required(view_fn):
    @wraps(view_fn)
    def _wrapped(*args, **kwargs):
        u = current_user()
        if not u:
            return redirect(url_for("auth.login_page"))
        if u.role != "employee":
            return redirect(url_for("dashboard.index"))
        return view_fn(*args, **kwargs)

    return _wrapped


def api_admin_required(allow_local: bool = False):
    def _decorator(view_fn):
        @wraps(view_fn)
        def _wrapped(*args, **kwargs):
            if allow_local and is_local_request():
                return view_fn(*args, **kwargs)

            u = current_user()
            if not u:
                return jsonify({"error": "Unauthorized"}), 401
            if u.role != "admin":
                return jsonify({"error": "Forbidden"}), 403
            return view_fn(*args, **kwargs)

        return _wrapped

    return _decorator


def api_employee_required(view_fn):
    @wraps(view_fn)
    def _wrapped(*args, **kwargs):
        u = current_user()
        if not u:
            return jsonify({"error": "Unauthorized"}), 401
        if u.role != "employee":
            return jsonify({"error": "Forbidden"}), 403
        return view_fn(*args, **kwargs)

    return _wrapped
