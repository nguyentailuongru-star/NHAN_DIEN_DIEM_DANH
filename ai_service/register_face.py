"""Tool đăng ký khuôn mặt - chụp ảnh từ webcam, extract embedding, gửi lên API."""

import cv2
import sys
import os
import requests
import numpy as np
from face_analysis_local import FaceAnalysisLocal
from model_store import ensure_face_model

API_BASE = "http://localhost:5000"
MIN_SAMPLES = 5
MAX_SAMPLES = 20
BEST_SHARP_SAMPLES = 10


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Chuẩn hóa L2 để ổn định cosine similarity khi so khớp."""
    norm = float(np.linalg.norm(vec))
    if norm <= 0:
        return vec
    return vec / norm


def _face_sharpness(frame: np.ndarray, bbox) -> float:
    """Đo độ nét vùng mặt bằng phương sai Laplacian; càng cao càng nét."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    face_roi = frame[y1:y2, x1:x2]
    if face_roi.size == 0:
        return 0.0

    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def register_face(name, employee_id=None, department=None):
    """Đăng ký khuôn mặt mới cho 1 nhân viên."""

    # Load model
    print("Đang load model InsightFace...")
    app = FaceAnalysisLocal(model_dir=ensure_face_model(), ctx_id=0)
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Mở webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở webcam!")
        return

    print(
        f"Webcam đã mở. SPACE=chụp | ENTER=hoàn tất (khi >= {MIN_SAMPLES}) | "
        f"ESC=hủy. Yêu cầu {MIN_SAMPLES}-{MAX_SAMPLES} ảnh."
    )

    samples = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces để hiển thị preview
        display = frame.copy()
        faces = app.get(frame)

        for face in faces:
            bbox = face.bbox.astype(int)
            cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(display, "Face detected", (bbox[0], bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(
            display,
            f"Samples: {len(samples)}/{MAX_SAMPLES} (min {MIN_SAMPLES})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        cv2.imshow(f"Dang ky: {name}", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            print("Đã hủy.")
            break
        elif key == 32:  # SPACE
            if faces:
                # Chọn khuôn mặt có det_score cao nhất để giảm nhiễu khi khung có nhiều mặt
                best_face = max(faces, key=lambda f: float(getattr(f, "det_score", 0.0) or 0.0))
                emb = np.asarray(best_face.embedding, dtype=np.float32)
                emb = _l2_normalize(emb)
                sharpness = _face_sharpness(frame, best_face.bbox)
                samples.append({"embedding": emb, "sharpness": sharpness})
                print(
                    f"Đã capture ảnh {len(samples)}/{MAX_SAMPLES} "
                    f"({len(emb)} chiều, sharpness={sharpness:.2f})"
                )

                if len(samples) >= MAX_SAMPLES:
                    print(f"Đã đủ {MAX_SAMPLES} ảnh, tự động hoàn tất.")
                    break
            else:
                print("Không phát hiện khuôn mặt! Thử lại.")
        elif key == 13:  # ENTER
            if len(samples) >= MIN_SAMPLES:
                print(f"Hoàn tất với {len(samples)} ảnh.")
                break
            print(f"Cần ít nhất {MIN_SAMPLES} ảnh, hiện tại mới có {len(samples)} ảnh.")

    cap.release()
    cv2.destroyAllWindows()

    if len(samples) < MIN_SAMPLES:
        print(f"Chưa đủ số ảnh tối thiểu ({MIN_SAMPLES}), hủy đăng ký.")
        return

    # Lọc theo độ nét: chỉ giữ các mẫu sắc nét nhất trước khi gộp embedding
    sorted_samples = sorted(samples, key=lambda s: s["sharpness"], reverse=True)
    keep_count = min(len(sorted_samples), BEST_SHARP_SAMPLES)
    selected = sorted_samples[:keep_count]

    print(
        f"Chọn {keep_count}/{len(samples)} ảnh nét nhất "
        f"(min={selected[-1]['sharpness']:.2f}, max={selected[0]['sharpness']:.2f}) để tính embedding."
    )

    selected_embeddings = [s["embedding"] for s in selected]
    embedding = _l2_normalize(np.mean(np.stack(selected_embeddings, axis=0), axis=0)).astype(np.float32)

    # Gửi lên API
    payload = {
        "name": name,
        "employee_id": employee_id,
        "department": department,
        "embedding": embedding.tolist(),
    }

    try:
        resp = requests.post(f"{API_BASE}/api/users", json=payload, timeout=10)
        resp.raise_for_status()
        result = resp.json()
        print(f"Đăng ký thành công! User ID: {result['user']['id']}")
    except requests.RequestException as e:
        print(f"Lỗi gửi API: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cách dùng: python register_face.py <tên> [mã_nv] [phòng_ban]")
        print("Ví dụ:    python register_face.py 'Nguyen Van A' NV001 'Phòng IT'")
        sys.exit(1)

    name = sys.argv[1]
    emp_id = sys.argv[2] if len(sys.argv) > 2 else None
    dept = sys.argv[3] if len(sys.argv) > 3 else None

    register_face(name, emp_id, dept)
