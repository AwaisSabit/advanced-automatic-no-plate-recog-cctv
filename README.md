# PlateForge — Advanced Automatic Number Plate Recognition from CCTV

Read number plates from CCTV video: find moving vehicles, detect the
plate region, read the plate text with OCR, and log every plate with
its time and a snapshot.

> Built as part of a 30-day portfolio challenge, Nov–Dec 2026.

## Problem

Hours of CCTV footage hide the one vehicle you need. Finding its
plate by hand means watching the whole recording. PlateForge scans
the video, picks out the moving vehicles, reads their plates, and
writes a searchable log, so the plate is found in minutes, not hours.

## Features

- Video loading and metadata (duration, fps, frame count, resolution, codec)
- Frame extraction by interval
- Motion detection to find moving vehicles
- Plate region detection with contours and edge detection
- Plate candidate scoring and filtering by size and shape
- Plate cropping, zoom, and image enhancement
- OCR reading of plate text with a lightweight on-device OCR library
- Plate text cleaning and Pakistani format checking
- Confidence scoring and clear-plate filtering
- Logging to CSV and JSON, duplicate suppression, and plate search
- Snapshots with a box drawn around the plate, and a plate report
- Batch processing for multiple videos
- Command line interface with `info`, `detect`, `read`, `log`, `search` and `batch`

## Stack

- Python 3.11+
- [OpenCV](https://opencv.org/) (`opencv-python-headless`) for video, detection and enhancement
- [RapidOCR](https://github.com/RapidAI/RapidOCR) (`rapidocr-onnxruntime`, CPU-only, no torch) for reading plate text
- [numpy](https://numpy.org/) and [Pillow](https://python-pillow.org/) for image work
- pytest for tests

## Install

Requires Python 3.11 or newer.

```bash
git clone https://github.com/AwaisSabit/advanced-automatic-no-plate-recog-cctv.git
cd advanced-automatic-no-plate-recog-cctv
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .[test]           # editable install, including pytest
```

## Usage

### Video info

```bash
plateforge info cctv.mp4
```

```text
Video: cctv.mp4
Duration: 5.00s
FPS: 10.0
Frames: 50
Resolution: 640x360
Codec: FMP4
```

### Detect plate regions

```bash
plateforge detect cctv.mp4 --every-seconds 1
```

### Read plates and write the log

```bash
plateforge read cctv.mp4 --output-dir readings
```

This writes `readings/plates.csv`, `readings/plates.json`,
`readings/report.md` and a snapshot PNG per plate in
`readings/snapshots/`.

### Search the log

```bash
plateforge log readings/plates.csv
plateforge search readings/plates.csv --plate ABC-123
plateforge search readings/plates.csv --start 10 --end 60
```

### Batch processing

```bash
plateforge batch examples/batch-list.txt --output-dir results
```

### As a library

```python
from plateforge.batch import process_video

entries = process_video("cctv.mp4", "readings", every_seconds=1)
for entry in entries:
    print(entry.video_time, entry.plate, entry.confidence)
```

### Demo

```bash
python examples/demo.py        # writes examples/output/*.png and the log
```

## Tests

```bash
python -m pytest -q
```

## Limitations

- OCR accuracy depends on plate size, angle, light and blur. A small,
  tilted or dirty plate may read wrong or not at all. PlateForge never
  promises 100% accuracy; every reading carries a confidence score,
  and low-confidence readings are easy to spot in the log.
- Detection is contour-based, not a trained detector, so it can miss
  plates or pick up plate-shaped signs.
- Digital zoom cannot add detail that is not in the pixels.
- The OCR models are downloaded by RapidOCR on first use and need
  internet access that first time.

## License

MIT — see [LICENSE](LICENSE).
