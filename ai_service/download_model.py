"""Tải model InsightFace buffalo_l về thư mục models/ trong dự án."""

import os
import zipfile
import urllib.request
import sys

# URL model buffalo_l - GitHub Releases (chính thức)
MODEL_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
# Backup URLs
BACKUP_URLS = [
    "https://insightface.ai/files/models/buffalo_l.zip",
]

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_NAME = "buffalo_l"
MODEL_DIR = os.path.join(MODELS_DIR, MODEL_NAME)
ZIP_PATH = os.path.join(MODELS_DIR, f"{MODEL_NAME}.zip")

REQUIRED_FILES = [
    "1k3d68.onnx",
    "2d106det.onnx",
    "det_10g.onnx",
    "genderage.onnx",
    "w600k_r50.onnx",
]


def check_model_exists():
    """Kiểm tra model đã có đủ chưa."""
    if not os.path.exists(MODEL_DIR):
        return False
    for f in REQUIRED_FILES:
        if not os.path.exists(os.path.join(MODEL_DIR, f)):
            return False
    return True


def download_progress(block_num, block_size, total_size):
    """Hiển thị tiến trình download."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\r  Downloading: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)")
    else:
        mb_downloaded = downloaded / (1024 * 1024)
        sys.stdout.write(f"\r  Downloading: {mb_downloaded:.1f} MB")
    sys.stdout.flush()


def download_model():
    """Tải và giải nén model."""
    if check_model_exists():
        print(f"Model '{MODEL_NAME}' đã có sẵn tại: {MODEL_DIR}")
        return True

    os.makedirs(MODELS_DIR, exist_ok=True)

    # Thử download
    urls = [MODEL_URL] + BACKUP_URLS
    success = False

    for url in urls:
        try:
            print(f"Đang tải model '{MODEL_NAME}'...")
            print(f"  URL: {url}")
            urllib.request.urlretrieve(url, ZIP_PATH, download_progress)
            print()  # newline after progress
            success = True
            break
        except Exception as e:
            print(f"\n  Lỗi tải từ URL này: {e}")
            continue

    if not success:
        print("\n" + "=" * 60)
        print("KHÔNG THỂ TẢI TỰ ĐỘNG!")
        print("=" * 60)
        print(f"\nVui lòng tải thủ công model buffalo_l:")
        print(f"  1. Tải file buffalo_l.zip từ:")
        print(f"     https://github.com/deepinsight/insightface/releases")
        print(f"     (Mục 'Model Zoo' → buffalo_l)")
        print(f"  2. Đặt file zip vào: {MODELS_DIR}")
        print(f"  3. Chạy lại script này để giải nén")
        print(f"\n  Hoặc giải nén thủ công vào: {MODEL_DIR}")
        print(f"  Cần có các file: {', '.join(REQUIRED_FILES)}")
        return False

    # Giải nén
    print("Đang giải nén...")
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
            zf.extractall(MODELS_DIR)
        print(f"Giải nén xong!")

        # Nếu file onnx nằm ngoài (không trong subfolder buffalo_l) thì move vào
        os.makedirs(MODEL_DIR, exist_ok=True)
        for f in REQUIRED_FILES:
            src = os.path.join(MODELS_DIR, f)
            dst = os.path.join(MODEL_DIR, f)
            if os.path.exists(src) and not os.path.exists(dst):
                os.rename(src, dst)

    except zipfile.BadZipFile:
        print("File zip bị lỗi! Vui lòng tải lại.")
        if os.path.exists(ZIP_PATH):
            os.remove(ZIP_PATH)
        return False

    # Xóa file zip
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
        print("Đã xóa file zip.")

    # Verify
    if check_model_exists():
        print(f"\nModel đã sẵn sàng tại: {MODEL_DIR}")
        print(f"Các file: {', '.join(REQUIRED_FILES)}")
        return True
    else:
        print(f"\nCẤNH BÁO: Thiếu file model. Kiểm tra thư mục: {MODEL_DIR}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("DOWNLOAD MODEL INSIGHTFACE - buffalo_l")
    print("=" * 60)
    download_model()
