import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import tkinter.filedialog as fd
from tkinter import messagebox
import customtkinter as ctk
import cv2

VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.webm'}
MAX_WORKERS = 4

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class VideoFrameExtractor(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Video Frame Extractor Pro")
        self.geometry("720x450")

        self.input_folder = ctk.StringVar()
        self.output_folder = ctk.StringVar()
        self.auto_output = ctk.BooleanVar(value=True)
        self.stop_event = threading.Event()
        self.is_running = False

        self.setup_ui()

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=1)

        f1 = ctk.CTkFrame(self)
        f1.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        f1.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(f1, text="Thư mục video:", font=("Arial", 13, "bold")).grid(row=0, column=0, sticky="w", padx=15, pady=(10,0))
        ctk.CTkEntry(f1, textvariable=self.input_folder).grid(row=1, column=0, padx=15, pady=10, sticky="ew")
        ctk.CTkButton(f1, text="Chọn", width=80, command=self.select_input).grid(row=1, column=1, padx=10)

        f2 = ctk.CTkFrame(self)
        f2.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        f2.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(f2, text="Thư mục lưu ảnh:", font=("Arial", 13, "bold")).grid(row=0, column=0, sticky="w", padx=15, pady=(10,0))
        self.out_entry = ctk.CTkEntry(f2, textvariable=self.output_folder, state="disabled")
        self.out_entry.grid(row=1, column=0, padx=15, pady=10, sticky="ew")
        self.out_btn = ctk.CTkButton(f2, text="Chọn", width=80, command=self.select_output, state="disabled")
        self.out_btn.grid(row=1, column=1, padx=10)

        ctk.CTkCheckBox(f2, text="Lưu cùng thư mục gốc", variable=self.auto_output, command=self.toggle_output).grid(row=2, column=0, padx=15, pady=(0,10), sticky="w")

        f3 = ctk.CTkFrame(self)
        f3.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        f3.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(f3, text="Thời điểm (mm:ss):", font=("Arial", 13, "bold")).grid(row=0, column=0, sticky="w", padx=15, pady=(10,0))
        self.time_entry = ctk.CTkEntry(f3, width=120, placeholder_text="00:15")
        self.time_entry.insert(0, "00:15")
        self.time_entry.grid(row=1, column=0, padx=15, pady=10)

        self.btn_action = ctk.CTkButton(f3, text="Bắt đầu tách", command=self.toggle_processing, height=40, font=("Arial", 14, "bold"))
        self.btn_action.grid(row=1, column=1, padx=15, pady=10, sticky="ew")

        f4 = ctk.CTkFrame(self, fg_color="transparent")
        f4.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        f4.grid_columnconfigure(0, weight=1)

        self.status_lbl = ctk.CTkLabel(f4, text="Sẵn sàng", text_color="gray")
        self.status_lbl.grid(row=0, column=0, sticky="ew")
        self.progress = ctk.CTkProgressBar(f4, height=12)
        self.progress.grid(row=1, column=0, pady=5, sticky="ew")
        self.progress.set(0)

    def select_input(self):
        path = fd.askdirectory()
        if path:
            self.input_folder.set(path)
            if self.auto_output.get():
                self.output_folder.set(path)
            count = len([f for f in Path(path).glob('*') if f.suffix.lower() in VIDEO_EXTENSIONS])
            self.status_lbl.configure(text=f"Tìm thấy {count} video", text_color="white")

    def select_output(self):
        path = fd.askdirectory()
        if path: self.output_folder.set(path)

    def toggle_output(self):
        state = "disabled" if self.auto_output.get() else "normal"
        self.out_entry.configure(state=state)
        self.out_btn.configure(state=state)
        if self.auto_output.get() and self.input_folder.get():
            self.output_folder.set(self.input_folder.get())

    def parse_time(self, time_str):
        try:
            parts = list(map(int, time_str.strip().split(':')))
            if len(parts) == 1: return parts[0]
            if len(parts) == 2: return parts[0]*60 + parts[1]
            if len(parts) == 3: return parts[0]*3600 + parts[1]*60 + parts[2]
        except: return None
        return None

    def toggle_processing(self):
        if self.is_running:
            self.stop_event.set()
            self.btn_action.configure(text="Đang dừng...", state="disabled")
            return

        in_path = self.input_folder.get()
        out_path = self.output_folder.get()
        seconds = self.parse_time(self.time_entry.get())

        if not in_path or not out_path:
            messagebox.showwarning("Thiếu thông tin", "Vui lòng chọn thư mục input/output.")
            return
        if seconds is None:
            messagebox.showwarning("Lỗi", "Thời gian không hợp lệ.")
            return

        self.is_running = True
        self.stop_event.clear()
        self.btn_action.configure(text="Dừng lại", fg_color="#c0392b", hover_color="#e74c3c")

        threading.Thread(target=self.process_batch, args=(in_path, out_path, seconds), daemon=True).start()

    def process_one_video(self, file_path: Path, out_dir: Path, target_sec: int):
        if self.stop_event.is_set(): return False

        try:
            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened(): return False

            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0: return False

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_idx = min(int(fps * target_sec), total_frames - 1)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()

            if ret:
                safe_name = file_path.stem.translate(str.maketrans('<>:"/\\|?*', '_________'))
                out_file = out_dir / f"{safe_name}.jpg"

                is_success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if is_success:
                    with open(out_file, "wb") as f:
                        f.write(buffer)
                return True
        except:
            pass
        return False

    def process_batch(self, in_dir, out_dir, seconds):
        files = [f for f in Path(in_dir).glob('*') if f.suffix.lower() in VIDEO_EXTENSIONS]
        total = len(files)

        if total == 0:
            self.after(0, lambda: messagebox.showinfo("Thông báo", "Không tìm thấy video nào!"))
            self.reset_ui()
            return

        Path(out_dir).mkdir(parents=True, exist_ok=True)

        completed = 0
        success_count = 0

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(self.process_one_video, f, Path(out_dir), seconds): f for f in files}

            for future in futures:
                if self.stop_event.is_set(): break
                if future.result(): success_count += 1
                completed += 1

                prog = completed / total
                msg = f"Đang xử lý: {completed}/{total} ({int(prog*100)}%)"
                self.after(0, lambda p=prog, m=msg: self.update_progress(p, m))

        final_msg = "Đã dừng bởi người dùng." if self.stop_event.is_set() else "Hoàn tất!"
        self.after(0, lambda: messagebox.showinfo(final_msg, f"Đã xuất thành công: {success_count}/{total} ảnh"))
        self.after(0, self.reset_ui)

    def update_progress(self, val, msg):
        self.progress.set(val)
        self.status_lbl.configure(text=msg)

    def reset_ui(self):
        self.is_running = False
        self.btn_action.configure(text="Bắt đầu tách", fg_color="#1f6aa5", hover_color="#144870", state="normal")
        self.status_lbl.configure(text="Sẵn sàng")
        self.progress.set(0)

if __name__ == "__main__":
    VideoFrameExtractor().mainloop()