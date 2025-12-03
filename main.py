import customtkinter as ctk
import tkinter.filedialog as fd
from tkinter import messagebox
import os
import cv2
import threading
import unicodedata
from pathlib import Path

VIDEO_EXTENSIONS = ('.mp4', '.mkv', '.avi', '.mov')

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
class VideoFrameExtractor(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Tách Frame từ Video")
        self.geometry("700x480")
        self.configure(fg_color="#101010")

        self.input_folder = ctk.StringVar()
        self.output_folder = ctk.StringVar()
        self.is_processing = False
        self.auto_output = ctk.BooleanVar(value=True)
        self.setup_ui()

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=1)
        f1 = ctk.CTkFrame(self)
        f1.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        f1.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(f1, text="📂 Thư mục video:", font=ctk.CTkFont(size=13, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=15, pady=(8, 0))
        ctk.CTkEntry(f1, textvariable=self.input_folder).grid(
            row=1, column=0, padx=15, pady=8, sticky="ew")
        ctk.CTkButton(f1, text="Chọn", command=self.select_input_folder,
                      width=90).grid(row=1, column=1, padx=10, pady=8)

        f2 = ctk.CTkFrame(self)
        f2.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        f2.grid_columnconfigure(0, weight=1)
        
        ctk.CTkLabel(f2, text="💾 Thư mục lưu ảnh:", font=ctk.CTkFont(size=13, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=15, pady=(8, 0))
        self.output_entry = ctk.CTkEntry(f2, textvariable=self.output_folder, state="disabled")
        self.output_entry.grid(row=1, column=0, padx=15, pady=8, sticky="ew")
        self.output_button = ctk.CTkButton(f2, text="Chọn", command=self.select_output_folder,
                      width=90, state="disabled")
        self.output_button.grid(row=1, column=1, padx=10, pady=8)
        ctk.CTkCheckBox(f2, text="Tự động (cùng thư mục video)", variable=self.auto_output,
                       command=self.toggle_output_folder).grid(row=2, column=0, columnspan=2, padx=15, pady=(0, 8), sticky="w")

        f3 = ctk.CTkFrame(self)
        f3.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        f3.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(f3, text="⏱ Thời gian (mm:ss):", font=ctk.CTkFont(size=13, weight="bold")).grid(
            row=0, column=0, sticky="w", padx=15, pady=(8, 0))
        self.time_entry = ctk.CTkEntry(f3, width=110, placeholder_text="00:15")
        self.time_entry.insert(0, "00:15")
        self.time_entry.grid(row=1, column=0, padx=15, pady=8)
        self.extract_button = ctk.CTkButton(
            f3, text="✨ Tách ảnh", command=self.run_extraction,
            height=40, font=ctk.CTkFont(size=15, weight="bold")
        )
        self.extract_button.grid(row=1, column=1, padx=15, pady=8, sticky="ew")

        f4 = ctk.CTkFrame(self)
        f4.grid(row=3, column=0, padx=20, pady=(10, 20), sticky="ew")
        f4.grid_columnconfigure(0, weight=1)
        
        self.status_label = ctk.CTkLabel(
            f4, text="✅ Sẵn sàng", anchor="center", font=ctk.CTkFont(size=13))
        self.status_label.grid(row=0, column=0, padx=15, pady=(12, 6), sticky="ew")
        self.progress_bar = ctk.CTkProgressBar(f4, height=14, progress_color="#4da6ff")
        self.progress_bar.grid(row=1, column=0, padx=20, pady=(0, 12), sticky="ew")
        self.progress_bar.set(0)

    def select_input_folder(self):
        path = fd.askdirectory()
        if path:
            self.input_folder.set(path)
            if self.auto_output.get():
                self.output_folder.set(path)
            try:
                count = len([f for f in os.listdir(path)
                            if f.lower().endswith(VIDEO_EXTENSIONS)])
                self.status_label.configure(
                    text=f"Đã tìm thấy {count} video" if count else "Không tìm thấy video nào")
            except OSError as e:
                messagebox.showerror("Lỗi", f"Không thể đọc thư mục: {e}")

    def select_output_folder(self):
        path = fd.askdirectory()
        if path:
            self.output_folder.set(path)

    def toggle_output_folder(self):
        if self.auto_output.get():
            self.output_entry.configure(state="disabled")
            self.output_button.configure(state="disabled")
            if self.input_folder.get():
                self.output_folder.set(self.input_folder.get())
        else:
            self.output_entry.configure(state="normal")
            self.output_button.configure(state="normal")

    def run_extraction(self):
        if self.is_processing:
            return
        if not self.input_folder.get():
            messagebox.showerror("Thiếu thông tin", "Vui lòng chọn thư mục video.")
            return
        if self.auto_output.get():
            self.output_folder.set(self.input_folder.get())
        if not self.output_folder.get() or not self.time_entry.get():
            messagebox.showerror("Thiếu thông tin", "Vui lòng chọn đủ thư mục và thời gian.")
            return
        self.is_processing = True
        self.extract_button.configure(state="disabled")
        threading.Thread(target=self.extract_frames_with_progress, daemon=True).start()

    def validate_time(self, time_str):
        try:
            m, s = map(int, time_str.strip().split(':'))
            if not (0 <= m <= 999 and 0 <= s <= 59):
                return None
            return m * 60 + s
        except:
            return None

    def extract_frames_with_progress(self):
        target_sec = self.validate_time(self.time_entry.get())
        if target_sec is None:
            self.status_label.configure(text="❌ Lỗi: Thời gian không hợp lệ")
            messagebox.showerror("Lỗi", "Thời gian không hợp lệ (mm:ss).")
            self.reset_ui()
            return

        input_path, output_path = self.input_folder.get(), self.output_folder.get()
        os.makedirs(output_path, exist_ok=True)

        videos = [f for f in os.listdir(
            input_path) if f.lower().endswith(VIDEO_EXTENSIONS)]
        total = len(videos)
        if not total:
            self.status_label.configure(text="❌ Không tìm thấy video nào")
            messagebox.showerror("Lỗi", "Không có video trong thư mục.")
            self.reset_ui()
            return

        count, log = 0, []
        for i, file in enumerate(videos, start=1):
            self.progress_bar.set(i/total)
            self.status_label.configure(text=f"⏳ Đang xử lý: {i}/{total}")
            full_path = os.path.join(input_path, file)

            try:
                cap = cv2.VideoCapture(full_path)
                if not cap.isOpened():
                    log.append(
                        f"Lỗi: Không thể mở {file} (file có thể bị hỏng hoặc định dạng không hỗ trợ)")
                    continue

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total_frames <= 0:
                    log.append(f"Lỗi: {file} không có frames hoặc bị hỏng")
                    continue

                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    log.append(f"Lỗi: Không thể đọc FPS của {file}")
                    continue

                frame_pos = min(int(fps * target_sec), total_frames - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_pos))
                ret, frame = cap.read()
                if not ret:
                    log.append(
                        f"Lỗi: Không thể đọc frame tại {target_sec}s của {file}")
                    continue

                safe_name = unicodedata.normalize('NFKD', Path(file).stem)
                safe_name = safe_name.encode('ascii', 'ignore').decode()
                safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in safe_name).strip()
                safe_name = safe_name or f"image_{i}"
                filename = f"{safe_name}.jpg"
                save_path = os.path.join(output_path, filename)

                if cv2.imwrite(save_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85]):
                    count += 1
                    if count <= 10:
                        log.append(f"{Path(file).stem} → {filename}")
            finally:
                cap.release()

        self.progress_bar.set(1)
        self.status_label.configure(text=f"🎉 Đã xuất {count}/{total} ảnh")

        msg = f"Đã xuất {count} ảnh vào:\n{output_path}"
        if log:
            msg += "\n\n" + "\n".join(log)
        messagebox.showinfo("Hoàn tất", msg)
        self.reset_ui()

    def reset_ui(self):
        self.is_processing = False
        self.extract_button.configure(state="normal")


if __name__ == "__main__":
    VideoFrameExtractor().mainloop()
