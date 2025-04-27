import customtkinter as ctk
import tkinter.filedialog as fd
import os
import cv2
from pathlib import Path
from tkinter import messagebox
import threading
import unicodedata
import re
import uuid

ctk.set_appearance_mode("system")
ctk.set_default_color_theme("blue")


class VideoFrameExtractor(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Video Frame Extractor")
        self.geometry("600x320")
        self.input_folder = ctk.StringVar()
        self.output_folder = ctk.StringVar()
        self.is_processing = False
        self.setup_ui()

    def setup_ui(self):
        self.grid_columnconfigure(0, weight=1)

        # --- INPUT ---
        f1 = ctk.CTkFrame(self)
        f1.grid(row=0, column=0, padx=20, pady=(20, 5), sticky="ew")
        f1.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(f1, text="Video folder:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, sticky="w", padx=10)
        ctk.CTkEntry(f1, textvariable=self.input_folder).grid(row=1, column=0, padx=15, pady=5, sticky="ew")
        ctk.CTkButton(f1, text="Browse", command=self.select_input_folder, width=80).grid(row=1, column=1, padx=(5, 15))

        # --- OUTPUT ---
        f2 = ctk.CTkFrame(self)
        f2.grid(row=1, column=0, padx=20, pady=5, sticky="ew")
        f2.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(f2, text="Output folder:", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, sticky="w", padx=10)
        ctk.CTkEntry(f2, textvariable=self.output_folder).grid(row=1, column=0, padx=15, pady=5, sticky="ew")
        ctk.CTkButton(f2, text="Browse", command=self.select_output_folder, width=80).grid(row=1, column=1, padx=(5, 15))

        # --- TIME + BUTTON ---
        f3 = ctk.CTkFrame(self)
        f3.grid(row=2, column=0, padx=20, pady=5, sticky="ew")
        f3.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(f3, text="Timestamp (mm:ss):", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, sticky="w", padx=10)
        self.time_entry = ctk.CTkEntry(f3, width=100, placeholder_text="01:30")
        self.time_entry.grid(row=1, column=0, padx=15, pady=5)
        self.extract_button = ctk.CTkButton(
            f3, text="Extract Frame", command=self.run_extraction, height=35, font=ctk.CTkFont(size=15, weight="bold")
        )
        self.extract_button.grid(row=1, column=1, padx=(5, 15), sticky="ew")

        # --- STATUS ---
        f4 = ctk.CTkFrame(self)
        f4.grid(row=3, column=0, padx=20, pady=(5, 20), sticky="ew")
        f4.grid_columnconfigure(0, weight=1)
        self.status_label = ctk.CTkLabel(f4, text="Ready", anchor="center")
        self.status_label.grid(row=0, column=0, padx=15, pady=5, sticky="ew")
        self.progress_bar = ctk.CTkProgressBar(f4)
        self.progress_bar.grid(row=1, column=0, padx=15, pady=5, sticky="ew")
        self.progress_bar.set(0)

    def select_input_folder(self):
        path = fd.askdirectory()
        if path:
            self.input_folder.set(path)
            count = len([f for f in os.listdir(path) if f.lower().endswith(('.mp4', '.mkv', '.avi', '.mov'))])
            self.status_label.configure(text=f"Found {count} video(s)" if count else "No videos found")

    def select_output_folder(self):
        path = fd.askdirectory()
        if path:
            self.output_folder.set(path)

    def run_extraction(self):
        if self.is_processing:
            return
        if not self.input_folder.get() or not self.output_folder.get() or not self.time_entry.get():
            messagebox.showerror("Missing Info", "Please select folders and enter a timestamp.")
            return
        self.is_processing = True
        self.extract_button.configure(state="disabled")
        threading.Thread(target=self.extract_frames_with_progress, daemon=True).start()

    def extract_frames_with_progress(self):
        try:
            m, s = map(int, self.time_entry.get().strip().split(':'))
            target_sec = m * 60 + s
        except:
            self.status_label.configure(text="Error: Invalid timestamp")
            messagebox.showerror("Error", "Invalid timestamp format (mm:ss).")
            self.reset_ui()
            return

        os.makedirs(self.output_folder.get(), exist_ok=True)
        videos = [f for f in os.listdir(self.input_folder.get()) if f.lower().endswith(('.mp4', '.mkv', '.avi', '.mov'))]
        if not videos:
            self.status_label.configure(text="No videos found")
            messagebox.showerror("Error", "No videos in selected folder.")
            self.reset_ui()
            return

        count, log = 0, []
        for i, file in enumerate(videos):
            self.progress_bar.set(i / len(videos))
            self.status_label.configure(text=f"Processing: {i+1}/{len(videos)}")
            try:
                cap = cv2.VideoCapture(os.path.join(self.input_folder.get(), file))
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * target_sec))
                ret, frame = cap.read()
                if ret:
                    name = Path(file).stem
                    safe = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode()
                    safe = re.sub(r'[\\/*?:"<>|]', "_", safe).strip()
                    if not safe:
                        safe = f"image_{uuid.uuid4().hex[:8]}"
                    out_path = os.path.join(self.output_folder.get(), f"{safe}.jpg")
                    if cv2.imwrite(out_path, frame):
                        count += 1
                        log.append(f"{name} → {Path(out_path).name}")
                cap.release()
            except:
                continue

        self.progress_bar.set(1)
        self.status_label.configure(text=f"Extracted {count}/{len(videos)} frames")
        message = f"Extracted {count} frames to:\n{self.output_folder.get()}"
        if count <= 10:
            message += "\n\n" + "\n".join(log)
        messagebox.showinfo("Done", message)
        self.reset_ui()

    def reset_ui(self):
        self.is_processing = False
        self.extract_button.configure(state="normal")


if __name__ == "__main__":
    VideoFrameExtractor().mainloop()
