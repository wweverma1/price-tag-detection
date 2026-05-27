# Price Tag Detection Pipeline
Terminal and GUI python pipeline for shelf price-tag detection using a fine tuned YOLOv8n model, its handover to EasyOCR for OCR and finally reporting the results in an easy to understand way.

## Setup
1. Create and activate a virtual environment.

```bash
# create virtual environment
virtualenv venv

# activate the virtual environment on windows
venv/Scripts/activate

# activate the virtual environment on linux
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure model weights exist at:
`weights/best.pt`


## Run GUI

```bash
python inference_gui.py
```

### Outputs
The GUI writes:
- Summary JSON to: `runs/inference_gui/detections_<timestamp>.json`
- Annotated images are saved upon user's request

Detection JSON shape per tag:

```json
{
  "tag_id": 1,
  "confidence": 0.97,
  "bbox": [35, 121, 92, 162],
  "ocr": {
    "value": 359,
    "confidence": 0.76
  }
}
```

### Demo

https://github.com/user-attachments/assets/e254a315-e6b0-48c3-a855-17d6bb277b38

## Run In Terminal

Basic usage:

```bash
python inference_cli.py <input1> [<input2> ...]
```

`<input>` can be:
- a single image file path
- a directory path
- a glob pattern

Examples:

```bash
python inference_cli.py ./images
python inference_cli.py ./images --recursive
python inference_cli.py "./images/*.jpg" "./more_images/*.png"
python inference_cli.py ./images --threshold 0.5
python inference_cli.py ./images --weights ./weights/best.pt --output-dir ./runs/inference_cli
```

Available options:
- `--weights` path to YOLO weights (default: `weights/best.pt`)
- `--threshold` YOLO confidence threshold between `0.0` and `1.0` (default: `0.4`)
- `--output-dir` output directory for summary JSON and annotated images (default: `runs/inference_cli`)
- `--recursive` recursively scan directories for images

### Outputs
The CLI writes:
- Summary JSON to: `<output-dir>/detections_<timestamp>.json`
- Annotated images to: `<output-dir>/annotated/`

Detection JSON shape per tag:

```json
{
  "tag_id": 1,
  "confidence": 0.97,
  "bbox": [35, 121, 92, 162],
  "ocr": {
    "value": 359,
    "confidence": 0.76
  }
}
```

## Acknowledgements
1. [Ultralytics](https://ultralytics.com/)
2. [OpenCV](https://github.com/opencv/opencv-python)
3. [PyTorch](https://pytorch.org/)
4. [EasyOCR](https://github.com/jaidedai/easyocr)
