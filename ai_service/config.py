"""Cấu hình cho AI Service."""

# RTSP Camera config
CAMERAS = [
    {
        "id": "3",
        "name": "Camera yosee 192.168.1.53",
        "rtsp_url": "rtsp://admin:admin1630688@192.168.1.53:554/onvif1",
    },
    {
        "id": "56",
        "name": "cam_dahua_192_168_1_54",
        "rtsp_url": "rtsp://admin:L2295266@192.168.1.54:554/cam/realmonitor?channel=1&subtype=0",
    },
]

# Flask API endpoint
API_BASE_URL = "http://localhost:5000"
API_RECOGNITION_URL = f"{API_BASE_URL}/api/recognition"
API_GET_USERS_URL = f"{API_BASE_URL}/api/users/embeddings"
API_ATTENTION_URL = f"{API_BASE_URL}/api/attention"

# InsightFace config
FACE_MODEL_NAME = "buffalo_l"
GPU_ID = 0  # -1 cho CPU

# Đường dẫn model trong dự án (thay vì ~/.insightface/models)
# FaceAnalysis tìm onnx tại: {root}/{name}/*.onnx → models/buffalo_l/*.onnx
import os

# Workspace root (chứa ai_service/, flask_app/, video/, captures/ ...)
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# Ngưỡng so sánh khuôn mặt (cosine similarity)
FACE_MATCH_THRESHOLD = 0.55

# Khoảng cách giữa các lần xử lý frame (giây)
FRAME_INTERVAL = 0.5

# Kích thước frame xử lý (resize để tăng tốc)
# Với camera 5MP, nên dùng 640x360 hoặc 640x480 cho pipeline AI.
PROCESS_WIDTH = 640
PROCESS_HEIGHT = 360

# Kích thước input cho detector (SCRFD thường ổn định nhất với 640x640)
DETECTION_WIDTH = 640
DETECTION_HEIGHT = 640

# Ngưỡng detect cho SCRFD (càng thấp càng dễ bắt mặt nhưng dễ false positive)
# Video offline/ảnh tối/mặt nhỏ nên để thấp hơn mặc định 0.5.
DETECTION_THRESH = 0.4

# Cấu hình riêng cho VIDEO OFFLINE: det_size lớn hơn giúp bắt mặt nhỏ tốt hơn.
VIDEO_DETECTION_WIDTH = 1024
VIDEO_DETECTION_HEIGHT = 1024
VIDEO_DETECTION_THRESH = 0.25

# Video offline thường chứa mặt nhỏ/xa: không nên resize xuống quá nhỏ như 640x360.
# Ta chỉ *giảm* kích thước nếu video quá lớn để tránh nặng máy; còn lại giữ nguyên.
VIDEO_PROCESS_MAX_WIDTH = 1280
VIDEO_PROCESS_MAX_HEIGHT = 720

# Tối ưu OpenCV FFmpeg cho RTSP (giảm trễ/buffer, ổn định WiFi)
# Nếu WiFi yếu và hay mất gói, TCP thường ổn định hơn UDP nhưng có thể tăng latency.
RTSP_TRANSPORT = "tcp"  # "tcp" | "udp"
RTSP_OPEN_TIMEOUT_MS = 5000
RTSP_READ_TIMEOUT_MS = 5000
RTSP_MAX_DELAY_MS = 500
RTSP_BUFFER_SIZE = 1

# Thời gian cooldown giữa 2 lần điểm danh cùng 1 người (giây)
RECOGNITION_COOLDOWN = 300

# Student behavior model config
BEHAVIOR_MODEL_FILENAME = "student_behavior_best.pt"
BEHAVIOR_POST_INTERVAL = 5.0  # giây, gửi đánh giá hành vi định kỳ theo user+camera
BEHAVIOR_CLASS_NAMES = [
    "distracted",
    "focused",
    "raising_hand",
    "sleep",
    "using_phone",
]

# Person tracking pipeline (YOLO -> ByteTrack)
ENABLE_PERSON_TRACKING = True
PERSON_DETECT_MODEL_FILENAME = "yolov8n.pt"
PERSON_TRACKER_CONFIG = "bytetrack.yaml"
PERSON_TRACK_CONF = 0.25
PERSON_TRACK_IOU = 0.5

# Theo dõi map track -> user và gom hành vi theo nhiều frame
TRACK_STALE_SECONDS = 8.0
TRACK_RECOGNITION_REFRESH_SEC = 2.0
TRACK_BEHAVIOR_WINDOW_SIZE = 12
TRACK_BEHAVIOR_MIN_SAMPLES = 4

# Thư mục lưu ảnh nhận diện (absolute để không phụ thuộc cwd)
CAPTURE_DIR = os.path.join(REPO_ROOT, "captures")
