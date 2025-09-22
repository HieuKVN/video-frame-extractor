# Video Frame Extractor

Simple app to **extract frames** from multiple videos at a **specific timestamp**.

_AI was used during the development process._

## Tech Stack

- Python
- OpenCV
- CustomTkinter

## Install

```bash
git clone https://github.com/ayaka-chann/video-frame-extractor.git
cd video-frame-extractor
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

or build `.exe`:

```bash
pip install pyinstaller
pyinstaller --noconsole --onefile --name "FrameExtractor" main.py
```

- `.exe` will be in the `dist/` folder

## How to use

- Select input folder (video files: `.mp4`, `.mkv`, `.avi`, `.mov`)
- Select output folder
- Enter timestamp (format `mm:ss`)
- Click **Extract Frame**

## Notes

- Special characters in filenames are auto-fixed
- Corrupt videos are skipped
- Shows total frames extracted when done
