@echo off & (where pythonw >nul 2>&1 && start "" pythonw -x "%~f0" %* || (echo Khong tim thay Python 3.11 trong PATH. Hay cai Python va tick "Add python.exe to PATH". & pause)) & exit /b
# -*- coding: utf-8 -*-
# ============================================================================
#  GIAO DIEN KHOI DONG - He thong nhan dien diem danh
#  Bam dup file nay. Dong dau la lenh batch goi "pythonw -x" de chay chinh file
#  nay bang Python (-x bo qua dong dau), nen chi can 1 file duy nhat.
# ============================================================================
import ctypes
import os
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import webbrowser
from datetime import datetime

import tkinter as tk
from tkinter import messagebox, ttk

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(ROOT, "logs")
FLASK_PORT = 5000
DASHBOARD_URL = f"http://localhost:{FLASK_PORT}"
NO_WINDOW = 0x08000000  # CREATE_NO_WINDOW

# Chay giao dien bang pythonw.exe, nhung service phai chay bang python.exe (co stdout).
_exe_dir = os.path.dirname(sys.executable)
PYTHON = os.path.join(_exe_dir, "python.exe")
if not os.path.exists(PYTHON):
    PYTHON = sys.executable

ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
ERR_RE = re.compile(r"Traceback|\bERROR\b|Error:|Exception|\[E:|lỗi|Không thể", re.IGNORECASE)
WARN_RE = re.compile(r"\bWARN(ING)?\b|\[W:", re.IGNORECASE)

COLORS = {
    "bg": "#f4f6fa", "card": "#ffffff", "text": "#1f2937", "muted": "#6b7280",
    "green": "#16a34a", "red": "#dc2626", "amber": "#d97706", "gray": "#9ca3af",
    "accent": "#2563eb", "log_bg": "#111827", "log_fg": "#e5e7eb",
}

try:
    import psutil
except ImportError:  # psutil di kem ultralytics; neu thieu thi bao ro
    psutil = None


def _norm(path):
    return os.path.normcase(os.path.abspath(path)) if path else ""


def kill_tree(pid):
    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW)


class Service:
    def __init__(self, key, title, folder, script, log_queue):
        self.key = key
        self.title = title
        self.cwd = os.path.join(ROOT, folder)
        self.script = script
        self.log_queue = log_queue
        self.proc = None
        self.log_path = os.path.join(LOG_DIR, f"{key}.log")
        self.started_at = None
        self.stopping = False

    # ---- tim tien trinh (ke ca tien trinh chay tu truoc / tien trinh con cua Flask debug) ----
    def find_pids(self):
        if psutil is None:
            return [self.proc.pid] if self.proc and self.proc.poll() is None else []
        found = []
        want_cwd = _norm(self.cwd)
        for p in psutil.process_iter(["name", "cmdline", "cwd", "create_time"]):
            info = p.info
            if not (info.get("name") or "").lower().startswith("python"):
                continue
            cmd = info.get("cmdline") or []
            if not cmd or not os.path.basename(cmd[-1]).lower() == self.script:
                continue
            if _norm(info.get("cwd")) != want_cwd:
                continue
            found.append((p.pid, info.get("create_time") or time.time()))
        return found

    def owned(self):
        return self.proc is not None and self.proc.poll() is None

    # ---- dieu khien ----
    def start(self):
        if self.find_pids():
            self.log("[launcher] Service đang chạy từ trước, bỏ qua. "
                     "Bấm 'Khởi động lại' nếu muốn xem log.")
            return
        os.makedirs(LOG_DIR, exist_ok=True)
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        self.log(f"[launcher] Khởi động: {PYTHON} {self.script}  (thư mục {self.cwd})")
        self.proc = subprocess.Popen(
            [PYTHON, "-u", self.script], cwd=self.cwd, env=env,
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            creationflags=NO_WINDOW,
        )
        self.started_at = time.time()
        threading.Thread(target=self._pump, args=(self.proc,), daemon=True).start()

    def stop(self):
        pids = [pid for pid, _ in self.find_pids()]
        if self.owned():
            pids.insert(0, self.proc.pid)
        if not pids:
            return
        self.log("[launcher] Đang dừng...")
        self.stopping = True
        for pid in dict.fromkeys(pids):
            kill_tree(pid)
        if self.proc:
            try:
                self.proc.wait(timeout=5)
            except Exception:
                pass
        time.sleep(0.3)  # cho luồng đọc log kết thúc trước khi bỏ cờ
        self.stopping = False
        self.proc = None
        self.started_at = None
        self.log("[launcher] Đã dừng.")

    # ---- log ----
    def log(self, line):
        stamp = datetime.now().strftime("%H:%M:%S")
        self._emit(f"{stamp} {line}")

    def _emit(self, line):
        self.log_queue.put((self.key, line))
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def _pump(self, proc):
        for raw in iter(proc.stdout.readline, b""):
            line = ANSI_RE.sub("", raw.decode("utf-8", errors="replace")).rstrip()
            if line:
                self._emit(line)
        code = proc.wait()
        if not self.stopping:
            self.log(f"[launcher] LỖI: tiến trình tự thoát (mã {code}). Xem log phía trên.")


class LauncherApp:
    def __init__(self, root):
        self.root = root
        self.log_queue = queue.Queue()
        self.status_queue = queue.Queue()
        self.services = {
            "ai_service": Service("ai_service", "AI Service (camera + nhận diện)", "ai_service", "main.py", self.log_queue),
            "flask": Service("flask", "Flask Web (dashboard)", "flask_app", "app.py", self.log_queue),
        }
        self.flask_ready = False
        self.pending_open_browser = False
        self.busy = False

        self._build_ui()
        self._start_status_thread()
        self.root.after(100, self._drain_queues)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        if psutil is None:
            messagebox.showwarning("Thiếu psutil",
                                   "Không có thư viện psutil nên không phát hiện được service chạy từ trước.\n"
                                   "Cài bằng: python -m pip install psutil")

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        r = self.root
        r.title("Khởi động - Hệ thống nhận diện điểm danh")
        r.geometry("1000x700")
        r.minsize(820, 560)
        r.configure(bg=COLORS["bg"])

        style = ttk.Style(r)
        style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 10), background=COLORS["bg"], foreground=COLORS["text"])
        style.configure("Card.TFrame", background=COLORS["card"], relief="solid", borderwidth=1)
        style.configure("Card.TLabel", background=COLORS["card"])
        style.configure("Muted.TLabel", background=COLORS["card"], foreground=COLORS["muted"], font=("Segoe UI", 9))
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 16), background=COLORS["bg"])
        style.configure("Sub.TLabel", foreground=COLORS["muted"], background=COLORS["bg"], font=("Segoe UI", 9))
        style.configure("CardTitle.TLabel", font=("Segoe UI Semibold", 11), background=COLORS["card"])
        style.configure("Status.TLabel", font=("Segoe UI Semibold", 10), background=COLORS["card"])
        style.configure("Big.TButton", font=("Segoe UI Semibold", 11), padding=(16, 8))
        style.configure("Accent.TButton", font=("Segoe UI Semibold", 11), padding=(16, 8),
                        background=COLORS["accent"], foreground="white")
        style.map("Accent.TButton", background=[("active", "#1d4ed8"), ("disabled", "#93c5fd")])
        style.configure("TButton", padding=(10, 5))
        style.configure("TNotebook", background=COLORS["bg"])
        style.configure("TNotebook.Tab", padding=(14, 6))

        header = ttk.Frame(r, padding=(18, 14, 18, 6))
        header.pack(fill="x")
        ttk.Label(header, text="HỆ THỐNG NHẬN DIỆN ĐIỂM DANH", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text=f"Python: {PYTHON}    •    Thư mục: {ROOT}", style="Sub.TLabel").pack(anchor="w")

        # Thanh nut chinh
        bar = ttk.Frame(r, padding=(18, 6))
        bar.pack(fill="x")
        self.btn_start_all = ttk.Button(bar, text="▶  Khởi động tất cả", style="Accent.TButton", command=self.start_all)
        self.btn_start_all.pack(side="left")
        self.btn_stop_all = ttk.Button(bar, text="■  Dừng tất cả", style="Big.TButton", command=self.stop_all)
        self.btn_stop_all.pack(side="left", padx=8)
        self.btn_restart_all = ttk.Button(bar, text="⟳  Khởi động lại", style="Big.TButton", command=self.restart_all)
        self.btn_restart_all.pack(side="left")
        ttk.Button(bar, text="Thư mục log", command=lambda: os.startfile(LOG_DIR) if os.path.isdir(LOG_DIR) else None).pack(side="right")
        self.btn_open = ttk.Button(bar, text="Mở Dashboard", command=lambda: webbrowser.open(DASHBOARD_URL))
        self.btn_open.pack(side="right", padx=8)
        self.auto_open = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="Tự mở dashboard khi sẵn sàng", variable=self.auto_open).pack(side="right", padx=8)

        # The trang thai tung service
        cards = ttk.Frame(r, padding=(18, 8))
        cards.pack(fill="x")
        self.card_widgets = {}
        for i, (key, svc) in enumerate(self.services.items()):
            card = ttk.Frame(cards, style="Card.TFrame", padding=12)
            card.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 6, 6 if i == 0 else 0))
            cards.columnconfigure(i, weight=1)

            top = ttk.Frame(card, style="Card.TFrame", borderwidth=0)
            top.pack(fill="x")
            dot = tk.Canvas(top, width=14, height=14, bg=COLORS["card"], highlightthickness=0)
            dot.pack(side="left", padx=(0, 8))
            oval = dot.create_oval(2, 2, 12, 12, fill=COLORS["gray"], outline="")
            ttk.Label(top, text=svc.title, style="CardTitle.TLabel").pack(side="left")

            status = ttk.Label(card, text="Đang kiểm tra...", style="Status.TLabel")
            status.pack(anchor="w", pady=(8, 0))
            detail = ttk.Label(card, text="", style="Muted.TLabel")
            detail.pack(anchor="w")

            btns = ttk.Frame(card, style="Card.TFrame", borderwidth=0)
            btns.pack(anchor="w", pady=(10, 0))
            b1 = ttk.Button(btns, text="Khởi động", command=lambda k=key: self.run_bg(self.services[k].start))
            b1.pack(side="left")
            b2 = ttk.Button(btns, text="Dừng", command=lambda k=key: self.run_bg(self.services[k].stop))
            b2.pack(side="left", padx=6)
            b3 = ttk.Button(btns, text="Khởi động lại", command=lambda k=key: self.run_bg(self._restart_one, k))
            b3.pack(side="left")
            self.card_widgets[key] = dict(dot=dot, oval=oval, status=status, detail=detail, buttons=(b1, b2, b3))

        # Log
        logs = ttk.Frame(r, padding=(18, 8, 18, 4))
        logs.pack(fill="both", expand=True)
        tools = ttk.Frame(logs)
        tools.pack(fill="x", pady=(0, 4))
        ttk.Label(tools, text="Log", font=("Segoe UI Semibold", 11)).pack(side="left")
        self.autoscroll = tk.BooleanVar(value=True)
        ttk.Button(tools, text="Xoá log đang xem", command=self.clear_current_log).pack(side="right")
        ttk.Checkbutton(tools, text="Tự cuộn", variable=self.autoscroll).pack(side="right", padx=8)

        self.notebook = ttk.Notebook(logs)
        self.notebook.pack(fill="both", expand=True)
        self.log_widgets = {}
        for key, svc in self.services.items():
            frame = ttk.Frame(self.notebook)
            text = tk.Text(frame, wrap="none", bg=COLORS["log_bg"], fg=COLORS["log_fg"],
                           insertbackground="white", font=("Consolas", 9), relief="flat", padx=8, pady=6)
            ys = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
            xs = ttk.Scrollbar(frame, orient="horizontal", command=text.xview)
            text.configure(yscrollcommand=ys.set, xscrollcommand=xs.set, state="disabled")
            text.grid(row=0, column=0, sticky="nsew")
            ys.grid(row=0, column=1, sticky="ns")
            xs.grid(row=1, column=0, sticky="ew")
            frame.rowconfigure(0, weight=1)
            frame.columnconfigure(0, weight=1)
            text.tag_configure("err", foreground="#f87171")
            text.tag_configure("warn", foreground="#fbbf24")
            text.tag_configure("launcher", foreground="#60a5fa")
            self.notebook.add(frame, text=svc.title.split(" (")[0])
            self.log_widgets[key] = text

        self.statusbar = ttk.Label(r, text="", style="Sub.TLabel", padding=(18, 4, 18, 8))
        self.statusbar.pack(fill="x")

    # ------------------------------------------------------------ actions
    def run_bg(self, fn, *args):
        def worker():
            self.busy = True
            try:
                fn(*args)
            except Exception as e:
                self.log_queue.put(("flask", f"[launcher] Lỗi: {e}"))
            finally:
                self.busy = False
        threading.Thread(target=worker, daemon=True).start()

    def _restart_one(self, key):
        self.services[key].stop()
        time.sleep(0.8)
        self.services[key].start()
        if key == "flask":
            self.pending_open_browser = self.auto_open.get()

    def start_all(self):
        self.pending_open_browser = self.auto_open.get()
        self.run_bg(lambda: [s.start() for s in (self.services["flask"], self.services["ai_service"])])

    def stop_all(self):
        self.pending_open_browser = False
        self.run_bg(lambda: [s.stop() for s in self.services.values()])

    def restart_all(self):
        def job():
            for s in self.services.values():
                s.stop()
            time.sleep(0.8)
            self.pending_open_browser = self.auto_open.get()
            self.services["flask"].start()
            self.services["ai_service"].start()
        self.run_bg(job)

    def clear_current_log(self):
        idx = self.notebook.index(self.notebook.select())
        text = list(self.log_widgets.values())[idx]
        text.configure(state="normal")
        text.delete("1.0", "end")
        text.configure(state="disabled")

    # ------------------------------------------------------- status loop
    def _start_status_thread(self):
        def loop():
            while True:
                snapshot = {}
                for key, svc in self.services.items():
                    try:
                        snapshot[key] = (svc.find_pids(), svc.owned())
                    except Exception:
                        snapshot[key] = ([], svc.owned())
                ready = False
                if snapshot["flask"][0]:
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{FLASK_PORT}/login", timeout=1.5) as resp:
                            ready = resp.status < 500
                    except Exception:
                        ready = False
                self.status_queue.put((snapshot, ready))
                time.sleep(1.5)
        threading.Thread(target=loop, daemon=True).start()

    def _apply_status(self, snapshot, ready):
        any_running = False
        for key, (pids, owned) in snapshot.items():
            w = self.card_widgets[key]
            if pids:
                any_running = True
                uptime = int(time.time() - min(t for _, t in pids))
                h, rem = divmod(uptime, 3600)
                m, s = divmod(rem, 60)
                pid_text = ", ".join(str(p) for p, _ in pids)
                if key == "flask":
                    label, color = (f"Sẵn sàng  •  {DASHBOARD_URL}", COLORS["green"]) if ready else ("Đang khởi động...", COLORS["amber"])
                else:
                    label, color = "Đang chạy", COLORS["green"]
                extra = "" if owned else "   (chạy từ trước - không có log, bấm Khởi động lại để xem)"
                w["detail"].configure(text=f"PID {pid_text}   •   thời gian chạy {h:02d}:{m:02d}:{s:02d}{extra}")
            else:
                label, color = "Đã dừng", COLORS["gray"]
                w["detail"].configure(text="")
            w["status"].configure(text=label, foreground=color)
            w["dot"].itemconfigure(w["oval"], fill=color)

        self.flask_ready = ready
        if ready and self.pending_open_browser:
            self.pending_open_browser = False
            webbrowser.open(DASHBOARD_URL)

        state = "disabled" if self.busy else "normal"
        for b in (self.btn_start_all, self.btn_stop_all, self.btn_restart_all):
            b.configure(state=state)
        for w in self.card_widgets.values():
            for b in w["buttons"]:
                b.configure(state=state)
        self.btn_open.configure(state="normal" if ready else "disabled")
        self.statusbar.configure(text=("Đang xử lý..." if self.busy else
                                       ("Hệ thống đang chạy." if any_running else "Hệ thống đã dừng.")) +
                                 f"   Log được lưu tại: {LOG_DIR}")

    def _drain_queues(self):
        latest = None
        while True:
            try:
                latest = self.status_queue.get_nowait()
            except queue.Empty:
                break
        if latest:
            self._apply_status(*latest)

        pending = {}
        for _ in range(800):
            try:
                key, line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            pending.setdefault(key, []).append(line)
        for key, lines in pending.items():
            text = self.log_widgets[key]
            text.configure(state="normal")
            for line in lines:
                tag = ("launcher" if "[launcher]" in line else
                       "err" if ERR_RE.search(line) else
                       "warn" if WARN_RE.search(line) else None)
                text.insert("end", line + "\n", tag)
            excess = int(text.index("end-1c").split(".")[0]) - 3000
            if excess > 0:
                text.delete("1.0", f"{excess + 1}.0")
            text.configure(state="disabled")
            if self.autoscroll.get():
                text.see("end")
        self.root.after(150, self._drain_queues)

    def on_close(self):
        running = any(s.find_pids() for s in self.services.values())
        if running:
            if not messagebox.askokcancel("Thoát", "Thoát sẽ DỪNG tất cả service (Flask + AI).\nTiếp tục?"):
                return
            self.statusbar.configure(text="Đang dừng service...")
            self.root.update_idletasks()
            for s in self.services.values():
                s.stop()
        self.root.destroy()


def main():
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    # Chi cho phep mo 1 cua so khoi dong
    ctypes.windll.kernel32.CreateMutexW(None, False, "NhanDienDiemDanh_Launcher")
    if ctypes.windll.kernel32.GetLastError() == 183:  # ERROR_ALREADY_EXISTS
        ctypes.windll.user32.MessageBoxW(None, "Cửa sổ khởi động đang mở rồi.", "Thông báo", 0x40)
        return
    root = tk.Tk()
    LauncherApp(root)
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        err = traceback.format_exc()
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            with open(os.path.join(LOG_DIR, "launcher_error.log"), "a", encoding="utf-8") as f:
                f.write(f"\n[{datetime.now()}]\n{err}")
        except OSError:
            pass
        ctypes.windll.user32.MessageBoxW(None, err[-1500:], "Lỗi giao diện khởi động", 0x10)
