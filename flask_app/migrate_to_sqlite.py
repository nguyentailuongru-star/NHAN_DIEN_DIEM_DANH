"""Chuyển dữ liệu từ MySQL sang SQLite.

Cách dùng (chạy trong thư mục flask_app):

    # 1) Từ file dump phpMyAdmin/mysqldump (mặc định: ../database/diem_danh.sql)
    python migrate_to_sqlite.py
    python migrate_to_sqlite.py --sql-dump D:/backup/diem_danh.sql

    # 2) Đọc trực tiếp từ MySQL đang chạy (cần: pip install pymysql)
    python migrate_to_sqlite.py --mysql-url "mysql+pymysql://root:@localhost:3306/diem_danh?charset=utf8mb4"

    # Ghi đè file SQLite đã có dữ liệu
    python migrate_to_sqlite.py --force
"""

import argparse
import os
import re
import sys
from datetime import date, datetime

from sqlalchemy import Boolean, Date, DateTime, Integer, create_engine, func, select

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from config import Config  # noqa: E402

# Thứ tự chèn theo khóa ngoại (bảng cha trước).
TABLE_ORDER = [
    "departments",
    "users",
    "attendance_logs",
    "attention_logs",
    "face_upload_requests",
    "teacher_notebook_entries",
    "teacher_notebook_attendance_marks",
]


# ---------------------------------------------------------------------------
# Đọc file dump MySQL
# ---------------------------------------------------------------------------

_INSERT_RE = re.compile(r"INSERT INTO\s+`?(\w+)`?\s*\(([^)]*)\)\s*VALUES\s*", re.IGNORECASE)
_MYSQL_ESCAPES = {"0": "\0", "b": "\b", "n": "\n", "r": "\r", "t": "\t", "Z": "\x1a"}


def _parse_values(sql: str, pos: int):
    """Đọc các tuple (...),(...); bắt đầu từ pos. Trả về (rows, vị trí sau dấu ;)."""
    rows, row, n = [], None, len(sql)
    while pos < n:
        ch = sql[pos]
        if ch == "(" and row is None:
            row = []
            pos += 1
        elif ch == ")" and row is not None:
            rows.append(row)
            row = None
            pos += 1
        elif ch == ";" and row is None:
            return rows, pos + 1
        elif ch in " \t\r\n," :
            pos += 1
        elif ch == "'":
            pos += 1
            buf = []
            while True:
                c = sql[pos]
                if c == "\\":
                    nxt = sql[pos + 1]
                    buf.append(_MYSQL_ESCAPES.get(nxt, nxt))
                    pos += 2
                elif c == "'":
                    if pos + 1 < n and sql[pos + 1] == "'":
                        buf.append("'")
                        pos += 2
                    else:
                        pos += 1
                        break
                else:
                    buf.append(c)
                    pos += 1
            row.append("".join(buf))
        else:
            m = re.compile(r"[^,)\s]+").match(sql, pos)
            token = m.group(0)
            pos = m.end()
            if token.upper() == "NULL":
                row.append(None)
            else:
                try:
                    row.append(int(token))
                except ValueError:
                    row.append(float(token))
    raise ValueError("File dump bị cắt ngang: thiếu dấu ';' kết thúc lệnh INSERT")


def read_dump(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        sql = f.read()
    data, pos = {}, 0
    while True:
        m = _INSERT_RE.search(sql, pos)
        if not m:
            break
        table = m.group(1)
        cols = [c.strip().strip("`") for c in m.group(2).split(",")]
        rows, pos = _parse_values(sql, m.end())
        data.setdefault(table, []).extend(dict(zip(cols, r)) for r in rows)
    return data


def read_mysql(url: str, table_names) -> dict:
    from sqlalchemy import MetaData

    engine = create_engine(url)
    meta = MetaData()
    meta.reflect(bind=engine)
    data = {}
    with engine.connect() as conn:
        for name in table_names:
            if name in meta.tables:
                data[name] = [dict(r._mapping) for r in conn.execute(meta.tables[name].select())]
    return data


# ---------------------------------------------------------------------------
# Chuẩn hóa kiểu dữ liệu theo model
# ---------------------------------------------------------------------------

def _convert(value, col_type):
    if value is None:
        return None
    if isinstance(col_type, DateTime):
        if isinstance(value, datetime):
            return value
        s = str(value)
        if s.startswith("0000-00-00"):
            return None
        return datetime.fromisoformat(s)
    if isinstance(col_type, Date):
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        s = str(value)
        if s.startswith("0000-00-00"):
            return None
        return date.fromisoformat(s[:10])
    if isinstance(col_type, Boolean):
        return bool(int(value))
    if isinstance(col_type, Integer) and isinstance(value, str) and value.strip():
        return int(value)
    return value


def main():
    parser = argparse.ArgumentParser(description="Chuyển dữ liệu MySQL → SQLite")
    parser.add_argument("--sql-dump", default=os.path.join(Config.REPO_ROOT, "database", "diem_danh.sql"))
    parser.add_argument("--mysql-url", help="Đọc trực tiếp từ MySQL thay vì file dump")
    parser.add_argument("--force", action="store_true", help="Xóa file SQLite cũ trước khi chuyển")
    args = parser.parse_args()

    if not Config.SQLALCHEMY_DATABASE_URI.startswith("sqlite"):
        sys.exit("DATABASE_URL đang trỏ tới DB không phải SQLite, dừng lại.")

    db_path = Config.SQLITE_PATH
    if os.path.exists(db_path):
        if not args.force:
            sys.exit(f"Đã có file {db_path}. Thêm --force để xóa và chuyển lại từ đầu.")
        for suffix in ("", "-wal", "-shm"):
            if os.path.exists(db_path + suffix):
                os.remove(db_path + suffix)

    # create_app() tạo file SQLite + toàn bộ bảng theo models.py
    from app import create_app
    from models import db

    app = create_app()
    with app.app_context():
        tables = db.metadata.tables
        ordered = TABLE_ORDER + [t for t in tables if t not in TABLE_ORDER]

        if args.mysql_url:
            print("Đọc dữ liệu từ MySQL ...")
            source = read_mysql(args.mysql_url, ordered)
        else:
            print(f"Đọc file dump: {args.sql_dump}")
            source = read_dump(args.sql_dump)

        skipped = sorted(set(source) - set(tables))
        if skipped:
            print(f"Bỏ qua bảng không còn dùng: {', '.join(skipped)}")

        with db.engine.begin() as conn:
            for name in ordered:
                rows = source.get(name) or []
                if not rows:
                    print(f"  {name:<36} 0 dòng")
                    continue
                table = tables[name]
                col_types = {c.name: c.type for c in table.columns}
                clean = [
                    {k: _convert(v, col_types[k]) for k, v in row.items() if k in col_types}
                    for row in rows
                ]
                conn.execute(table.insert(), clean)
                count = conn.execute(select(func.count()).select_from(table)).scalar()
                print(f"  {name:<36} {count} dòng")

        # Gộp file -wal vào .db và đóng kết nối để file .db đầy đủ, dễ copy/backup.
        with db.engine.connect() as conn:
            conn.exec_driver_sql("PRAGMA wal_checkpoint(TRUNCATE)")
        db.engine.dispose()

        print(f"\nXong. File SQLite: {db_path}")


if __name__ == "__main__":
    main()
