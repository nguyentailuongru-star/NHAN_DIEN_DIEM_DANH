r"""Tải trước model AI về máy (không bắt buộc - hệ thống tự tải ở lần chạy đầu).

Model được cài tại %LOCALAPPDATA%\NhanDienDiemDanh\models (đổi bằng AI_MODEL_HOME).
"""

import logging

from model_store import ensure_face_model, ensure_yolo_weights, model_home

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("=" * 60)
    print("CÀI MODEL AI VÀO MÁY")
    print("=" * 60)
    print("Thư mục model:", model_home())
    print("buffalo_l  ->", ensure_face_model())
    print("yolov8n.pt ->", ensure_yolo_weights("yolov8n.pt"))
    print("Xong.")
