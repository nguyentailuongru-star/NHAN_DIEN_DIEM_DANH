"""Authentication routes: login/register/logout."""

from flask import Blueprint, render_template, request, redirect, url_for, flash
from models import db, User, Department
from auth_utils import login_user, logout_user, current_user

auth_bp = Blueprint("auth", __name__)


def _has_admin() -> bool:
    return User.query.filter_by(role="admin", is_active=True).first() is not None


@auth_bp.route("/login", methods=["GET", "POST"])
def login_page():
    u = current_user()
    if u:
        if u.role == "admin":
            return redirect(url_for("dashboard.index"))
        return redirect(url_for("dashboard.my_attendance_page"))

    if request.method == "GET":
        return render_template("login.html")

    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""

    user = User.query.filter_by(username=username, is_active=True).first()
    if not user or not user.check_password(password):
        flash("Sai tài khoản hoặc mật khẩu", "danger")
        return render_template("login.html"), 401

    login_user(user)
    if user.role == "admin":
        return redirect(url_for("dashboard.index"))
    return redirect(url_for("dashboard.my_attendance_page"))


@auth_bp.route("/register", methods=["GET", "POST"])
def register_page():
    if request.method == "GET":
        departments = Department.query.filter_by(is_active=True).order_by(Department.name.asc()).all()
        return render_template("register.html", departments=departments, admin_exists=_has_admin())

    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""
    full_name = (request.form.get("full_name") or "").strip()
    employee_id = (request.form.get("employee_id") or "").strip() or None
    department_id_raw = (request.form.get("department_id") or "").strip()

    if not username or not password or not full_name:
        flash("Vui lòng nhập đầy đủ Username, Mật khẩu và Họ tên", "danger")
        return redirect(url_for("auth.register_page"))

    if User.query.filter_by(username=username).first():
        flash("Username đã tồn tại", "danger")
        return redirect(url_for("auth.register_page"))

    if employee_id and User.query.filter_by(employee_id=employee_id).first():
        flash("Mã nhân viên đã tồn tại", "danger")
        return redirect(url_for("auth.register_page"))

    role = "employee"
    if not _has_admin() and (request.form.get("register_as_admin") == "on"):
        role = "admin"

    department_id = None
    if department_id_raw.isdigit():
        department_id = int(department_id_raw)

    department = Department.query.filter_by(id=department_id, is_active=True).first() if department_id else None

    user = User(
        username=username,
        name=full_name,
        employee_id=employee_id,
        department_id=department.id if department else None,
        role=role,
        is_active=True,
    )
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    login_user(user)
    flash("Đăng ký thành công", "success")

    if user.role == "admin":
        return redirect(url_for("dashboard.index"))
    return redirect(url_for("dashboard.my_attendance_page"))


@auth_bp.route("/logout")
def logout_page():
    logout_user()
    return redirect(url_for("auth.login_page"))
