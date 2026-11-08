"""Quản lý model AI cài trên máy (ngoài thư mục dự án).

Model tải được từ nguồn chính thức được lưu một lần cho cả máy tại:
    %LOCALAPPDATA%\\NhanDienDiemDanh\\models     (đổi bằng biến môi trường AI_MODEL_HOME)

- buffalo_l (InsightFace): chỉ giữ 2 file hệ thống dùng
    det_10g.onnx   (SCRFD - tìm khuôn mặt)
    w600k_r50.onnx (ArcFace - vector đặc trưng 512 chiều)
- yolov8n.pt (Ultralytics): bám theo người

Lần chạy đầu trên máy mới sẽ tự tải (~280MB cho buffalo_l). Có thể tải trước bằng:
    python ai_service/download_model.py
"""

import logging
import os
import shutil
import sys
import tempfile
import time
import urllib.request
import zipfile

logger = logging.getLogger(__name__)

BUFFALO_URLS = [
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
]
FACE_MODEL_NAME = "buffalo_l"
FACE_MODEL_FILES = ["det_10g.onnx", "w600k_r50.onnx"]


def model_home() -> str:
    custom = os.environ.get("AI_MODEL_HOME", "").strip()
    if custom:
        root = custom
    else:
        base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".cache")
        root = os.path.join(base, "NhanDienDiemDanh", "models")
    os.makedirs(root, exist_ok=True)
    return root


class _DirLock:
    """Khóa đơn giản bằng file để Flask và AI service không cùng tải 1 model."""

    def __init__(self, path, timeout=1800):
        self.path = path
        self.timeout = timeout
        self.fd = None

    def __enter__(self):
        start = time.time()
        while True:
            try:
                self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                return self
            except FileExistsError:
                # Khóa cũ của tiến trình đã chết (quá 1 giờ) → bỏ
                try:
                    if time.time() - os.path.getmtime(self.path) > 3600:
                        os.remove(self.path)
                        continue
                except OSError:
                    pass
                if time.time() - start > self.timeout:
                    raise TimeoutError(f"Chờ khóa tải model quá lâu: {self.path}")
                time.sleep(1)

    def __exit__(self, *exc):
        if self.fd is not None:
            os.close(self.fd)
        try:
            os.remove(self.path)
        except OSError:
            pass


def _progress(block_num, block_size, total_size):
    done = block_num * block_size
    if total_size > 0:
        pct = min(100.0, done * 100.0 / total_size)
        sys.stdout.write(f"\r  Đang tải: {pct:5.1f}% ({done / 1048576:.0f}/{total_size / 1048576:.0f} MB)")
    else:
        sys.stdout.write(f"\r  Đang tải: {done / 1048576:.0f} MB")
    sys.stdout.flush()


def _has_files(folder, files):
    return all(os.path.isfile(os.path.join(folder, f)) and os.path.getsize(os.path.join(folder, f)) > 0 for f in files)


def ensure_face_model() -> str:
    """Trả về thư mục buffalo_l chứa đủ model; tự tải nếu máy chưa có."""
    target = os.path.join(model_home(), FACE_MODEL_NAME)
    if _has_files(target, FACE_MODEL_FILES):
        return target

    with _DirLock(os.path.join(model_home(), f".{FACE_MODEL_NAME}.lock")):
        if _has_files(target, FACE_MODEL_FILES):  # tiến trình khác vừa tải xong
            return target

        os.makedirs(target, exist_ok=True)
        last_err = None
        for url in BUFFALO_URLS:
            fd, zip_path = tempfile.mkstemp(suffix=".zip", dir=model_home())
            os.close(fd)
            try:
                logger.info("Máy chưa có model %s → đang tải từ %s", FACE_MODEL_NAME, url)
                print(f"Đang tải model {FACE_MODEL_NAME} (~280MB, chỉ tải 1 lần cho máy này)...")
                urllib.request.urlretrieve(url, zip_path, _progress)
                print()
                with zipfile.ZipFile(zip_path) as zf:
                    for member in zf.infolist():
                        name = os.path.basename(member.filename)
                        if name in FACE_MODEL_FILES:
                            tmp_out = os.path.join(target, name + ".part")
                            with zf.open(member) as src, open(tmp_out, "wb") as dst:
                                shutil.copyfileobj(src, dst)
                            os.replace(tmp_out, os.path.join(target, name))
                if _has_files(target, FACE_MODEL_FILES):
                    logger.info("Đã cài model %s tại %s", FACE_MODEL_NAME, target)
                    return target
                last_err = RuntimeError("File zip không chứa đủ model cần thiết")
            except Exception as e:  # thử URL kế tiếp
                last_err = e
                logger.warning("Tải model từ %s thất bại: %s", url, e)
            finally:
                try:
                    os.remove(zip_path)
                except OSError:
                    pass

    raise RuntimeError(
        f"Không tải được model {FACE_MODEL_NAME}: {last_err}\n"
        f"Tải thủ công buffalo_l.zip từ https://github.com/deepinsight/insightface/releases (v0.7), "
        f"giải nén {', '.join(FACE_MODEL_FILES)} vào: {target}"
    )


def ensure_yolo_weights(name: str = "yolov8n.pt") -> str:
    """Trả về đường dẫn file trọng số YOLO chính thức; tự tải bằng ultralytics nếu chưa có."""
    path = os.path.join(model_home(), name)
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return path
    with _DirLock(os.path.join(model_home(), f".{name}.lock")):
        if not os.path.isfile(path):
            from ultralytics.utils.downloads import attempt_download_asset  # type: ignore[import-not-found]

            logger.info("Máy chưa có %s → đang tải về %s", name, path)
            attempt_download_asset(path)
    if not os.path.isfile(path):
        raise RuntimeError(f"Không tải được {name} về {path}")
    return path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    print("Thư mục model:", model_home())
    print("buffalo_l :", ensure_face_model())
    print("yolov8n   :", ensure_yolo_weights())
