import cv2
import os
from pathlib import Path

# --- Cấu hình ---
input_folder = 'videos'
output_folder = 'frames'
time_str = '2:32'  # thời gian: "phút:giây"

# --- Xử lý ---
minutes, seconds = map(int, time_str.split(':'))
target_time_sec = minutes * 60 + seconds

os.makedirs(output_folder, exist_ok=True)

for file in os.listdir(input_folder):
    if not file.lower().endswith(('.mp4', '.mkv', '.avi', '.mov')):
        continue

    video_path = os.path.join(input_folder, file)
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
