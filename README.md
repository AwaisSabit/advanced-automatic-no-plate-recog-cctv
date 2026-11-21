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
git clone https://github.com/AwaisSabit/advanced-anpr-cctv.git
cd advanced-anpr-cctv
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .[test]           # editable install, including pytest
```

## Usage

```bash
plateforge info cctv.mp4
plateforge detect cctv.mp4 --output-dir detections
plateforge read cctv.mp4 --output-dir readings
plateforge log readings/plates.csv --search ABC-123
plateforge search readings/plates.csv --plate ABC-123
plateforge batch videos.txt --output-dir results
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
