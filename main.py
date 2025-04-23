import cv2
import os
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox


def extract_frames(folder_path, time_str):
    minutes, seconds = map(int, time_str.split(':'))
    target_time_sec = minutes * 60 + seconds
    output_folder = os.path.join(folder_path, 'frames')
    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(folder_path):
        if not file.lower().endswith(('.mp4', '.mkv', '.avi', '.mov')):
            continue

        video_path = os.path.join(folder_path, file)
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(fps * target_time_sec)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = cap.read()
        if ret:
            name = Path(file).stem
            output_path = os.path.join(output_folder, f'{name}.jpg')
            cv2.imwrite(output_path, frame)

        cap.release()

    messagebox.showinfo("Hoàn tất", f"Đã xuất ảnh vào: {output_folder}")


def browse_folder():
    folder_selected = filedialog.askdirectory()
    if folder_selected:
        folder_path.set(folder_selected)


def run():
    if not folder_path.get() or not time_entry.get():
        messagebox.showerror("Lỗi", "Chọn thư mục và nhập thời gian.")
        return
    extract_frames(folder_path.get(), time_entry.get())


# GUI
root = tk.Tk()
root.title("Video Frame Extractor")

folder_path = tk.StringVar()

tk.Label(root, text="Thư mục chứa video:").grid(row=0, column=0, sticky="w")
tk.Entry(root, textvariable=folder_path, width=40).grid(row=0, column=1)
tk.Button(root, text="Chọn...", command=browse_folder).grid(row=0, column=2)

tk.Label(root, text="Thời gian (mm:ss):").grid(row=1, column=0, sticky="w")
time_entry = tk.Entry(root)
time_entry.grid(row=1, column=1)

tk.Button(root, text="Tách ảnh", command=run).grid(row=2, column=1, pady=10)

root.mainloop()
