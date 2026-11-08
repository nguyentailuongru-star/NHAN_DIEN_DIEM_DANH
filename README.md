# Hệ thống Nhận diện Điểm danh bằng AI

## Kiến trúc hệ thống

```
Camera RTSP
    ↓
AI Service (Python + GPU)     ← InsightFace / ArcFace + student_behavior_best.pt
    ↓ (HTTP POST)
Flask API (Backend)
    ↓
SQLite Database (database/diem_danh.db)
    ↓
Dashboard (Web Realtime)      ← Socket.IO
```

## Cấu trúc thư mục

```
NHAN_DIEN_DIEM_DANH/
├── ai_service/               # Service AI nhận diện
│   ├── main.py               # Chính: đọc RTSP → detect → recognize → POST API
│   ├── face_manager.py       # Quản lý & so sánh embeddings
│   ├── register_face.py      # Tool đăng ký khuôn mặt qua webcam
│   ├── config.py             # Cấu hình camera, threshold, ...
│   └── requirements.txt
├── flask_app/                # Flask Backend + Dashboard
│   ├── app.py                # Entry point Flask + SocketIO
│   ├── config.py             # Cấu hình DB, upload, ...
│   ├── models.py             # SQLAlchemy models
│   ├── routes/
│   │   ├── api.py            # REST API endpoints
│   │   └── dashboard.py      # Render HTML pages
│   ├── templates/            # Jinja2 templates
│   ├── static/               # CSS + JS
│   └── requirements.txt
└── database/
    └── schema.sql            # SQL tạo bảng
```

## Hướng dẫn cài đặt

### 1. Database (SQLite – không cần cài đặt)

Hệ thống dùng **SQLite**: toàn bộ dữ liệu nằm trong file `database/diem_danh.db`.
File được tạo tự động khi chạy Flask lần đầu. Sao lưu = copy file này (khi Flask đang tắt).

Chuyển dữ liệu cũ từ MySQL (chỉ cần làm 1 lần):
```bash
cd flask_app
python migrate_to_sqlite.py                      # từ file dump database/diem_danh.sql
python migrate_to_sqlite.py --mysql-url "mysql+pymysql://root:@localhost:3306/diem_danh?charset=utf8mb4"   # đọc trực tiếp MySQL
```
Thêm `--force` để xóa file SQLite hiện có và chuyển lại từ đầu.

Biến môi trường (tùy chọn):
- `SQLITE_PATH`: đổi vị trí file `.db`
- `DATABASE_URL`: dùng DB khác (vd. MySQL, cần `pip install pymysql`)

### 2. Cài đặt Flask Backend

```bash
cd flask_app
pip install -r requirements.txt
```

Chạy Flask:
```bash
cd flask_app
python app.py
```

Truy cập dashboard: http://localhost:5000

### 3. Cài đặt AI Service

```bash
cd ai_service
pip install -r requirements.txt
```

> **Lưu ý GPU (GTX 1050)**: AI Service đang dùng `onnxruntime-gpu==1.16.3` (CUDA 11.8). Bạn cần:
> - NVIDIA Driver đủ mới
> - CUDA Toolkit 11.8
> - cuDNN 8.x tương ứng với CUDA 11.8
>
> Nếu muốn chạy CPU: thay `onnxruntime-gpu==1.16.3` bằng `onnxruntime==1.16.3` trong `ai_service/requirements.txt` và đặt `GPU_ID = -1` trong config.

Cấu hình camera RTSP trong `ai_service/config.py`:
```python
CAMERAS = [
    {
        "id": "cam_192_168_1_3",
        "name": "Camera RTSP 192.168.1.3",
        "rtsp_url": "rtsp://admin:admin1630688@192.168.1.3:554/onvif1",
    },
]
```

> **Khuyến nghị camera Dahua 5MP**:
> - Ưu tiên stream **H.264** (tránh H.265/HEVC vì OpenCV/FFmpeg trên Windows có thể không decode được tuỳ build)
> - Dùng **Sub stream** cho AI: độ phân giải thấp (ví dụ 640x360 hoặc 720p), 10–15 FPS
> - Main stream giữ chất lượng cao để lưu hình/giám sát (nếu cần)

### 4. Đăng ký khuôn mặt

Dùng webcam để đăng ký:
```bash
cd ai_service
python register_face.py "Nguyen Van A" NV001 "Phong IT"
```

### 5. Chạy AI Service

```bash
cd ai_service
python main.py
```

### 6. (Tuỳ chọn) Chạy AI Service qua video đã lưu

Nếu bạn muốn test/điểm danh bằng video (mp4/avi/...) thay vì RTSP:

```bash
cd ai_service
python main.py --video "duong_dan\video.mp4" --camera-id "video_cam01"
```

AI Service sẽ lấy mẫu frame theo `FRAME_INTERVAL` để giảm tải, nhận diện như bình thường và vẫn gửi kết quả về Flask API.

## Luồng hoạt động

1. **AI Service** đọc stream RTSP từ camera
2. Mỗi frame → InsightFace detect + extract embedding (vector 512 chiều)
3. So sánh embedding với danh sách đã đăng ký (cosine similarity)
4. Nếu match (>= threshold 0.5) → HTTP POST kết quả về Flask API
5. Flask API lưu vào SQLite + emit Socket.IO event
6. Dashboard nhận event realtime → hiển thị ngay lập tức

## API Endpoints

| Method | URL | Mô tả |
|--------|-----|--------|
| POST | `/api/recognition` | AI Service gửi kết quả nhận diện |
| GET | `/api/users` | Lấy danh sách nhân viên |
| GET | `/api/users/embeddings` | Lấy embeddings (cho AI Service) |
| POST | `/api/users` | Thêm nhân viên mới |
| PUT | `/api/users/<id>` | Cập nhật thông tin nhân viên |
| DELETE | `/api/users/<id>` | Xóa nhân viên (soft delete) |
| POST | `/api/users/<id>/avatar` | Upload ảnh đại diện |
| POST | `/api/users/<id>/face_images` | Upload ảnh khuôn mặt → tự tạo embedding |
| GET | `/api/logs` | Lấy lịch sử điểm danh (phân trang) |
| GET | `/api/logs/today` | Lấy log điểm danh hôm nay |
| GET | `/api/stats` | Thống kê tổng quan |

## Thao tác trên Web

### 1) Thêm sinh viên/nhân viên + ảnh khuôn mặt (để nhận dạng)

- Vào trang `/users` → **Thêm nhân viên**
- Sau khi tạo xong, bấm nút **Upload ảnh khuôn mặt** (icon khung mặt) ở cột thao tác
- Chọn 3–10 ảnh rõ mặt → hệ thống tự detect mặt, tạo embedding và lưu vào DB

### 2) Chọn video để điểm danh (offline)

- Chép video vào thư mục `video/` (tại root workspace)
- Vào Dashboard → mục **Điểm danh từ video (offline)**
- Chọn video → nhấn **Start** (có thể đặt `Camera ID` tuỳ ý)
- Khi nhận diện được, Dashboard sẽ hiện realtime như bình thường

## Camera RTSP URLs

| Hãng | URL mẫu |
|------|---------|
| Hikvision | `rtsp://user:pass@ip:554/Streaming/Channels/101` |
| Dahua | `rtsp://user:pass@ip:554/cam/realmonitor?channel=1&subtype=0` |

## Cấu hình

### Ngưỡng nhận diện (`ai_service/config.py`)
- `FACE_MATCH_THRESHOLD = 0.5` — cosine similarity threshold
- `0.4`: dễ match nhầm
- `0.5–0.6`: chuẩn (khuyến nghị)
- `RECOGNITION_COOLDOWN = 60` — số giây giữa 2 lần điểm danh cùng 1 người

# COPY MÁY KHÁC

Có, copy được toàn bộ project để làm tiếp.
Khuyến nghị không copy .venv giữa 2 máy khác cấu hình, vì thường không ổn định (đặc biệt với CUDA/driver).
Nên copy source + models + database backup, rồi trên máy mới cài lại thư viện bằng requirements.
Checklist khi sang máy mới

Copy project + model:
buffalo_l
Tạo môi trường và cài libs:
python -m venv .venv
.python.exe -m pip install -r ai_service/requirements.txt
.python.exe -m pip install -r flask_app/requirements.txt
DB:
