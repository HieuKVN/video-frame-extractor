import difflib
import html
import multiprocessing
import os
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from xml.etree.ElementTree import Element, SubElement, tostring

import cv2
from PySide6.QtCore import QObject, Qt, QThread, Signal
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent, QFont, QTextCharFormat, QTextCursor, QKeySequence, QShortcut
from PySide6.QtWidgets import (QApplication, QFileDialog, QFrame,
                               QHBoxLayout, QLabel, QLineEdit, QMainWindow,
                               QMessageBox, QProgressBar, QPushButton,
                               QTabWidget, QTextEdit, QVBoxLayout, QWidget)

try:
    from unidecode import unidecode
except ImportError:
    import unicodedata
    def unidecode(s: str) -> str:
        return "".join([c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c)])

DEFAULT_FOLDER = "F:/Comp 1"
DEFAULT_FPS = 30
DEFAULT_TIME_EXTRACT = "00:15"
JPEG_QUALITY = 90
DEFAULT_CLIP_DURATION = 300
FUZZY_MATCH_CUTOFF = 0.6
MAX_LOG_LINES = 1000

VIDEO_EXTENSIONS: Set[str] = {'.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.webm'}
IMAGE_EXTENSIONS: Set[str] = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
MAX_WORKERS = min(32, (multiprocessing.cpu_count() or 1) + 4)

TIMECODE_PATTERN = re.compile(r'^\d{2}:\d{2}:\d{2}:\d{2}$')
TIME_EXTRACT_PATTERN = re.compile(r'^(\d+|\d{2}:\d{2}|\d{2}:\d{2}:\d{2})$')

COLORS = {
    "SUCCESS": "#a6e3a1", "ERROR": "#f38ba8", "WARNING": "#fab387",
    "INFO": "#89dceb", "OK": "#a6e3a1", "WARN": "#fab387",
    "ERR": "#f38ba8", "HEADER": "#b4befe"
}

ICONS = {
    "INFO": "ℹ", "SUCCESS": "✓", "ERROR": "✗", "WARNING": "⚠",
    "OK": "✓", "WARN": "⚠", "ERR": "✗", "HEADER": "▸"
}

def sanitize_filename(filename: str) -> str:
    return filename.translate(str.maketrans('<>:"/\\|?*', '_________'))

def get_timestamp() -> str:
    return datetime.now().strftime("%H:%M:%S")

def parse_timecode_to_seconds(time_str: str) -> Optional[int]:
    try:
        parts = list(map(int, time_str.strip().split(':')))
        if len(parts) == 1: return parts[0]
        if len(parts) == 2: return parts[0] * 60 + parts[1]
        if len(parts) == 3: return parts[0] * 3600 + parts[1] * 60 + parts[2]
    except (ValueError, AttributeError): pass
    return None

def parse_timeline_timecode(tc_str: str, fps: int) -> Optional[int]:
    try:
        parts = list(map(int, tc_str.split(':')))
        if len(parts) == 4:
            return (parts[0] * 3600 + parts[1] * 60 + parts[2]) * fps + parts[3]
    except (ValueError, IndexError): pass
    return None

def validate_timecode(tc: str) -> bool:
    return bool(TIMECODE_PATTERN.match(tc.strip()))

def validate_time_extract(time: str) -> bool:
    return bool(TIME_EXTRACT_PATTERN.match(time.strip()))

def create_log_formatter(color: str, bold: bool = False) -> QTextCharFormat:
    fmt = QTextCharFormat()
    fmt.setForeground(QColor(color))
    if bold: fmt.setFontWeight(QFont.Bold)
    return fmt

@lru_cache(maxsize=10)
def get_image_files_cached(folder: str) -> Tuple[str, ...]:
    try:
        path = Path(folder)
        return tuple(f.name for f in path.iterdir() if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS)
    except Exception: return tuple()

class WorkerSignals(QObject):
    log = Signal(str, str)
    progress = Signal(int, int, int)
    finished = Signal(bool, int, int, float)

class ScanSignals(QObject):
    log = Signal(str, str)
    status = Signal(str)
    progress = Signal(int, int)
    finished = Signal(bool)

class FrameExtractorWorker(QThread):
    def __init__(self, folder: str, seconds: int, stop_event: threading.Event):
        super().__init__()
        self.folder, self.seconds, self.stop_event = folder, seconds, stop_event
        self.signals = WorkerSignals()

    def run(self) -> None:
        in_path = Path(self.folder)
        video_files = [f for f in in_path.glob('*') if f.suffix.lower() in VIDEO_EXTENSIONS]
        total = len(video_files)

        if total == 0:
            self.signals.log.emit("No video files found", "WARNING")
            self.signals.finished.emit(True, 0, 0, 0.0)
            return

        self.signals.log.emit(f"Found {total} video(s)", "INFO")
        done, success, start_time = 0, 0, time.time()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(self._extract, f, in_path, self.seconds): f for f in video_files}
            for future in as_completed(futures):
                if self.stop_event.is_set(): break
                file = futures[future]
                try:
                    if future.result():
                        success += 1
                        self.signals.log.emit(f"Extracted: {file.name}", "SUCCESS")
                    else:
                        self.signals.log.emit(f"Failed: {file.name}", "ERROR")
                except Exception as e:
                    self.signals.log.emit(f"Error {file.name}: {str(e)}", "ERROR")
                done += 1
                self.signals.progress.emit(done, total, success)

        elapsed = time.time() - start_time
        self.signals.finished.emit(self.stop_event.is_set(), success, total, elapsed)

    def _extract(self, path: Path, out_dir: Path, sec: int) -> bool:
        cap = None
        try:
            cap = cv2.VideoCapture(str(path))
            if not cap.isOpened(): return False
            fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0: return False

            target_frame = max(0, min(int(fps * sec), total_frames - 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            if not ret or frame is None: return False

            safe_name = sanitize_filename(unidecode(path.stem))
            return cv2.imwrite(str(out_dir / f"{safe_name}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        except Exception: return False
        finally:
            if cap: cap.release()

class TimelineScanWorker(QThread):
    def __init__(self, folder: str, timeline_text: str):
        super().__init__()
        self.folder, self.timeline_text = folder, timeline_text
        self.signals, self.processed_clips = ScanSignals(), []

    def run(self) -> None:
        has_error = False
        files = list(get_image_files_cached(self.folder))
        if not files:
            self.signals.log.emit("No images found", "ERR")
            self.signals.finished.emit(True)
            return

        c_exact, c_no_ext, c_unsigned = self._build_caches(files)
        lines = [l.strip() for l in self.timeline_text.strip().split('\n') if l.strip()]
        total_lines = len(lines)

        for i, line in enumerate(lines):
            self.signals.progress.emit(i + 1, total_lines)
            parts = line.split(' ', 1)
            if len(parts) < 2 or not validate_timecode(parts[0]):
                self.signals.log.emit(f"Line {i+1}: Invalid format", "ERR")
                has_error = True
                continue

            tc, name = parts[0], parts[1].strip()
            found_file, status = self._find_file(name, c_exact, c_no_ext, c_unsigned)

            if found_file:
                self._log_match(name, found_file, status)
                start_frame = (2 * DEFAULT_FPS) if tc == "00:00:00:00" else parse_timeline_timecode(tc, DEFAULT_FPS)
                if start_frame is None:
                    has_error = True
                    continue
                self.processed_clips.append({"name": html.escape(name), "filename": html.escape(found_file),
                                           "path": f"{self.folder}/{html.escape(found_file)}", "start": start_frame})
            else:
                self.signals.log.emit(f"[MISSING] {name}", "ERR")
                has_error = True

        self._calc_durations()
        self.signals.status.emit("Ready" if not has_error else "Errors Found")
        self.signals.finished.emit(has_error)

    def _build_caches(self, files):
        e, n, u = {}, {}, {}
        for f in files:
            low = f.lower()
            no_ext = os.path.splitext(low)[0]
            e[low], n[no_ext], u[unidecode(no_ext)] = f, f, f
        return e, n, u

    def _find_file(self, target, c_e, c_n, c_u):
        low = target.strip().lower()
        if any(low.endswith(ext) for ext in IMAGE_EXTENSIONS) and low in c_e: return c_e[low], "OK"
        if low in c_n: return c_n[low], "OK"
        unsig = unidecode(low)
        if unsig in c_u: return c_u[unsig], "WARN_UNSIGNED"
        fuzzy = difflib.get_close_matches(unsig, c_u.keys(), n=1, cutoff=FUZZY_MATCH_CUTOFF)
        return (c_u[fuzzy[0]], "WARN_FUZZY") if fuzzy else (None, "MISSING")

    def _log_match(self, name, found, status):
        lvl = "OK" if status == "OK" else "WARN"
        prefix = "" if status == "OK" else f"[{status}] "
        self.signals.log.emit(f"{prefix}{name} ➜ {found}", lvl)

    def _calc_durations(self):
        for i, clip in enumerate(self.processed_clips):
            clip['end'] = self.processed_clips[i+1]['start'] if i < len(self.processed_clips)-1 else clip['start'] + (DEFAULT_CLIP_DURATION * DEFAULT_FPS)
            clip['duration'] = clip['end'] - clip['start']

class LogMixin:
    def setup_log_widget(self, widget): self.log_widget = widget

    def log(self, msg, level="INFO"):
        cursor = self.log_widget.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.insertText(f"[{get_timestamp()}] ", create_log_formatter("#6c7086"))

        color = COLORS.get(level, "#cdd6f4")
        bold = level in ["SUCCESS", "ERROR", "OK", "ERR", "HEADER"]
        cursor.insertText(f"{ICONS.get(level, '•')} {msg}\n", create_log_formatter(color, bold))

        self.log_widget.setTextCursor(cursor)
        self.log_widget.ensureCursorVisible()
        if self.log_widget.document().lineCount() > MAX_LOG_LINES:
            c = QTextCursor(self.log_widget.document().findBlockByLineNumber(0))
            c.movePosition(QTextCursor.Down, QTextCursor.KeepAnchor, 100)
            c.removeSelectedText()

class FrameExtractorTab(QWidget, LogMixin):
    def __init__(self):
        super().__init__()
        self.stop_event, self.is_running, self.worker = threading.Event(), False, None
        self.init_ui()
        self.setup_shortcuts()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)

        title = QLabel("🎬 Frame Extractor")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        title.setStyleSheet("color: #00d9ff;")
        layout.addWidget(title)

        self.folder_input = QLineEdit(DEFAULT_FOLDER)
        self.folder_input.setPlaceholderText("Drag folder here...")
        self.folder_input.setStyleSheet("background: #11111b; color: #cdd6f4; padding: 8px; border-radius: 6px;")

        f_layout = QHBoxLayout()
        f_layout.addWidget(self.folder_input)
        btn_br = QPushButton("📂 Browse")
        btn_br.clicked.connect(self.browse_folder)
        f_layout.addWidget(btn_br)
        layout.addLayout(f_layout)

        self.time_input = QLineEdit(DEFAULT_TIME_EXTRACT)
        self.time_input.setFixedWidth(100)
        layout.addWidget(QLabel("⏱ Extract at (SS, MM:SS, HH:MM:SS):"))
        layout.addWidget(self.time_input)

        self.run_btn = QPushButton("▶ Start Extraction")
        self.run_btn.setMinimumHeight(40)
        self.run_btn.setStyleSheet("background: #00d9ff; color: #11111b; font-weight: bold; border-radius: 8px;")
        self.run_btn.clicked.connect(self.toggle_extraction)
        layout.addWidget(self.run_btn)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        self.progress_label = QLabel("Ready")
        layout.addWidget(self.progress_label)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("background: #11111b; color: #cdd6f4; border-radius: 8px;")
        layout.addWidget(self.log_text)
        self.setup_log_widget(self.log_text)
        self.setAcceptDrops(True)

    def setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+O"), self).activated.connect(self.browse_folder)
        QShortcut(QKeySequence("Ctrl+R"), self).activated.connect(self.toggle_extraction)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls(): e.acceptProposedAction()

    def dropEvent(self, e):
        path = e.mimeData().urls()[0].toLocalFile()
        if os.path.isdir(path): self.folder_input.setText(path)

    def browse_folder(self):
        f = QFileDialog.getExistingDirectory(self, "Select Folder", self.folder_input.text())
        if f: self.folder_input.setText(f)

    def toggle_extraction(self):
        if self.is_running:
            self.stop_event.set()
            self.run_btn.setText("Stopping...")
        else:
            self.start_extraction()

    def start_extraction(self):
        t_str = self.time_input.text()
        if not validate_time_extract(t_str): return
        sec = parse_timecode_to_seconds(t_str)
        if not os.path.exists(self.folder_input.text()): return

        self.is_running = True
        self.stop_event.clear()
        self.run_btn.setText("⏹ Stop")
        self.run_btn.setStyleSheet("background: #f38ba8; color: white; border-radius: 8px;")

        self.worker = FrameExtractorWorker(self.folder_input.text(), sec, self.stop_event)
        self.worker.signals.log.connect(self.log)
        self.worker.signals.progress.connect(self.update_progress)
        self.worker.signals.finished.connect(self.extraction_finished)
        self.worker.start()

    def update_progress(self, d, t, s):
        p = int((d/t)*100)
        self.progress_bar.setValue(p)
        self.progress_label.setText(f"Processing: {d}/{t} ({p}%) - Success: {s}")

    def extraction_finished(self, st, s, t, e):
        self.is_running = False
        self.run_btn.setText("▶ Start Extraction")
        self.run_btn.setStyleSheet("background: #00d9ff; color: #11111b; font-weight: bold; border-radius: 8px;")
        if not st: self.log(f"Complete! {s}/{t} in {e:.1f}s", "SUCCESS")

class TimelineGeneratorTab(QWidget, LogMixin):
    def __init__(self):
        super().__init__()
        self.scan_has_error, self.processed_clips, self.worker = False, [], None
        self.init_ui()
        self.setup_shortcuts()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("📝 Timeline Input (HH:MM:SS:FF Name):"))
        self.timeline_input = QTextEdit("00:00:00:00 Clip A")
        self.timeline_input.setStyleSheet("background: #11111b; color: #cdd6f4;")
        layout.addWidget(self.timeline_input)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setStyleSheet("background: #11111b; color: #cdd6f4;")
        layout.addWidget(self.result_text)
        self.setup_log_widget(self.result_text)

        self.scan_progress = QProgressBar()
        self.scan_progress.setVisible(False)
        layout.addWidget(self.scan_progress)

        b_layout = QHBoxLayout()
        self.scan_btn = QPushButton("🔍 Scan Images")
        self.scan_btn.clicked.connect(self.start_scan)
        self.export_btn = QPushButton("💾 Export XML")
        self.export_btn.clicked.connect(self.export_xml)
        b_layout.addWidget(self.scan_btn)
        b_layout.addWidget(self.export_btn)
        layout.addLayout(b_layout)

    def setup_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self.start_scan)
        QShortcut(QKeySequence("Ctrl+E"), self).activated.connect(self.export_xml)

    def start_scan(self):
        self.scan_btn.setEnabled(False)
        self.scan_progress.setVisible(True)
        self.worker = TimelineScanWorker(DEFAULT_FOLDER, self.timeline_input.toPlainText())
        self.worker.signals.log.connect(self.log)
        self.worker.signals.progress.connect(lambda c, t: self.scan_progress.setValue(int((c/t)*100)))
        self.worker.signals.finished.connect(self.scan_finished)
        self.worker.start()

    def scan_finished(self, err):
        self.scan_has_error = err
        self.processed_clips = self.worker.processed_clips
        self.scan_btn.setEnabled(True)
        self.scan_progress.setVisible(False)

    def export_xml(self):
        if not self.processed_clips or self.scan_has_error: return
        path, _ = QFileDialog.getSaveFileName(self, "Save XML", "", "XML (*.xml)")
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(self._gen_xml())
                self.log(f"Exported to {path}", "SUCCESS")
            except Exception as e: self.log(str(e), "ERROR")

    def _gen_xml(self) -> str:
        root = Element('xmeml', version='4')
        seq = SubElement(root, 'sequence')
        SubElement(seq, 'name').text = 'TIMELINE'
        media = SubElement(seq, 'media')
        video = SubElement(media, 'video')
        track = SubElement(video, 'track')

        for i, clip in enumerate(self.processed_clips):
            item = SubElement(track, 'clipitem', id=f'c{i}')
            SubElement(item, 'name').text = clip['name']
            SubElement(item, 'duration').text = str(clip['duration'])
            SubElement(item, 'start').text = str(clip['start'])
            SubElement(item, 'end').text = str(clip['end'])
            file = SubElement(item, 'file', id=f'f{i}')
            SubElement(file, 'name').text = clip['filename']
            SubElement(file, 'pathurl').text = f"file://localhost/{clip['path']}"

        return '<?xml version="1.0" encoding="UTF-8"?>\n' + tostring(root, encoding='unicode')

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AIO Video Tool")
        self.resize(900, 700)
        tabs = QTabWidget()
        tabs.addTab(FrameExtractorTab(), "Extractor")
        tabs.addTab(TimelineGeneratorTab(), "Timeline")
        self.setCentralWidget(tabs)
        self.setStyleSheet("QMainWindow { background: #181825; }")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())