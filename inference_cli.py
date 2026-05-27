from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ocr import PriceOCRService, read_image_bgr
from ultralytics import YOLO

WEIGHTS_PATH = Path("./weights/best.pt")
OUTPUT_RELATIVE_DIR = Path("runs") / "inference_cli"
INITIAL_THRESHOLD = 0.4
TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
SUMMARY_FILENAME_TEMPLATE = "detections_{timestamp}.json"
ANNOTATED_FILENAME_TEMPLATE = "{stem}_detections_{timestamp}{suffix}"
ANNOTATED_SUBDIR = "annotated"
IMAGE_EXTENSIONS = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
)
PREVIEW_OVERLAY_ALPHA = 0.28
BOX_COLOR_BGR = (204, 102, 255)
BOX_LABEL_TEXT_COLOR_BGR = (255, 255, 255)
LABEL_FONT_SCALE = 0.72
LABEL_FONT_THICKNESS = 2
type OCRNumber = int | float


@dataclass
class PriceTagDetection:
    tag_id: int
    confidence: float
    bbox: list[int]
    ocr_value: OCRNumber | None = None
    ocr_confidence: float | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run shelf price-tag detection and OCR in terminal mode.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input image file(s), folder(s), or glob pattern(s).",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=WEIGHTS_PATH,
        help=f"Path to YOLO model weights. Default: {WEIGHTS_PATH.as_posix()}",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=INITIAL_THRESHOLD,
        help=f"YOLO confidence threshold (0.0-1.0). Default: {INITIAL_THRESHOLD}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd() / OUTPUT_RELATIVE_DIR,
        help="Directory for JSON summary and annotated images.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively collect images when an input is a directory.",
    )
    return parser.parse_args()


def _validate_threshold(threshold: float) -> float:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0.0 and 1.0.")
    return round(float(threshold), 1)


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _collect_images_from_dir(directory: Path, recursive: bool) -> list[Path]:
    iterator = directory.rglob("*") if recursive else directory.glob("*")
    return sorted(path.resolve() for path in iterator if _is_image_file(path))


def _resolve_input_images(inputs: list[str], recursive: bool) -> list[Path]:
    seen: set[Path] = set()
    resolved: list[Path] = []

    def add_candidate(path: Path) -> None:
        normalized = path.expanduser().resolve()
        if normalized in seen:
            return
        seen.add(normalized)
        resolved.append(normalized)

    for entry in inputs:
        direct_path = Path(entry).expanduser()

        if direct_path.is_file():
            if _is_image_file(direct_path):
                add_candidate(direct_path)
            else:
                print(f"[warn] Skipping non-image file: {direct_path}")
            continue

        if direct_path.is_dir():
            for image_path in _collect_images_from_dir(direct_path, recursive):
                add_candidate(image_path)
            continue

        matches = sorted(Path(match).resolve() for match in glob.glob(entry))
        if not matches:
            print(f"[warn] No matches for input: {entry}")
            continue
        for match in matches:
            if _is_image_file(match):
                add_candidate(match)
            else:
                print(f"[warn] Skipping non-image match: {match}")

    return resolved


def _format_ocr_value(value: OCRNumber) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _format_detection_text(det: PriceTagDetection) -> str:
    if det.ocr_value is None:
        return f"Conf: {det.confidence:.2f}"
    ocr_text = _format_ocr_value(det.ocr_value)
    if det.ocr_confidence is None:
        return f"Conf: {det.confidence:.2f} | OCR: {ocr_text}"
    return f"Conf: {det.confidence:.2f} | OCR: {ocr_text} ({det.ocr_confidence:.2f})"


def _serialize_detection(det: PriceTagDetection) -> dict[str, Any]:
    return {
        "tag_id": det.tag_id,
        "confidence": det.confidence,
        "bbox": det.bbox,
        "ocr": {
            "value": det.ocr_value,
            "confidence": det.ocr_confidence,
        },
    }


def _compose_annotated_image(
    image: np.ndarray,
    detections: list[PriceTagDetection],
) -> np.ndarray:
    draw_image = image.copy()

    if detections:
        dimmed = cv2.convertScaleAbs(draw_image, alpha=PREVIEW_OVERLAY_ALPHA, beta=0)
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            x1 = max(0, min(x1, draw_image.shape[1] - 1))
            y1 = max(0, min(y1, draw_image.shape[0] - 1))
            x2 = max(x1 + 1, min(x2, draw_image.shape[1]))
            y2 = max(y1 + 1, min(y2, draw_image.shape[0]))
            dimmed[y1:y2, x1:x2] = draw_image[y1:y2, x1:x2]
        draw_image = dimmed

    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(draw_image, (x1, y1), (x2, y2), BOX_COLOR_BGR, 2)

        label_text = _format_detection_text(det)
        (text_w, text_h), _ = cv2.getTextSize(
            label_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            LABEL_FONT_SCALE,
            LABEL_FONT_THICKNESS,
        )
        center_x = x1 + ((x2 - x1) // 2)
        text_x = max(4, min(center_x - (text_w // 2), draw_image.shape[1] - text_w - 4))
        text_y = max(text_h + 6, y1 - 10)
        cv2.putText(
            draw_image,
            label_text,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            LABEL_FONT_SCALE,
            BOX_LABEL_TEXT_COLOR_BGR,
            LABEL_FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return draw_image


def _write_image_file(target_path: Path, image: np.ndarray) -> bool:
    suffix = target_path.suffix.lower() or ".jpg"
    if suffix == ".jpeg":
        suffix = ".jpg"
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        return False
    encoded.tofile(str(target_path))
    return True


def _write_summary(
    output_dir: Path,
    detections_payload: dict[str, list[dict[str, Any]]],
    threshold: float,
    timestamp: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / SUMMARY_FILENAME_TEMPLATE.format(timestamp=timestamp)
    summary_data = {
        "created_at": datetime.now().isoformat(),
        "threshold": threshold,
        "images": detections_payload,
    }
    summary_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    return summary_path


def _run_pipeline(args: argparse.Namespace) -> int:
    try:
        threshold = _validate_threshold(args.threshold)
    except ValueError as exc:
        print(f"[error] {exc}")
        return 2

    weights_path = args.weights.expanduser().resolve()
    if not weights_path.is_file():
        print(f"[error] Model weights not found: {weights_path}")
        return 2

    image_paths = _resolve_input_images(args.inputs, args.recursive)
    if not image_paths:
        print("[error] No valid image files found from provided inputs.")
        return 2

    output_dir = args.output_dir.expanduser().resolve()
    print(f"[info] Loading YOLO model from: {weights_path}")
    try:
        model = YOLO(str(weights_path))
    except Exception as exc:  # noqa: BLE001
        print(f"[error] Failed to load model: {exc}")
        return 1

    ocr_service = PriceOCRService()
    timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
    detections_payload: dict[str, list[dict[str, Any]]] = {}
    total = len(image_paths)
    annotated_dir = output_dir / ANNOTATED_SUBDIR

    for idx, image_path in enumerate(image_paths, start=1):
        print(f"[detect] Image {idx}/{total}: {image_path.name}")
        try:
            result = model.predict(
                source=str(image_path),
                conf=threshold,
                show=False,
                save=False,
                verbose=False,
            )[0]
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] Inference failed for {image_path.name}: {exc}")
            continue

        detections: list[PriceTagDetection] = []
        for det_index, box in enumerate(result.boxes, start=1):
            coords = [int(c) for c in box.xyxy[0].tolist()]
            confidence = float(box.conf[0].item())
            detections.append(
                PriceTagDetection(
                    tag_id=det_index,
                    confidence=confidence,
                    bbox=coords,
                    ocr_value=None,
                    ocr_confidence=None,
                )
            )
        
        print(f"[detect] {image_path.name} found {len(detections)} price tags")

        raw_image = read_image_bgr(image_path)
        tag_count = len(detections)
        for tag_position, detection in enumerate(detections, start=1):
            print(
                "[ocr] Image "
                f"{idx}/{total} | Tag {tag_position}/{max(1, tag_count)} "
                f"(id={detection.tag_id})"
            )
            if raw_image is None:
                continue
            try:
                ocr_value, ocr_confidence = ocr_service.resolve_price(
                    raw_image, detection.bbox
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    "[error] OCR failed for "
                    f"{image_path.name}, tag {detection.tag_id}: {exc}"
                )
                return 1
            detection.ocr_value = ocr_value
            detection.ocr_confidence = ocr_confidence

        detections_payload[str(image_path)] = [
            _serialize_detection(d) for d in detections
        ]

        for det in detections:
            print(f"Tag {det.tag_id:02d} | {_format_detection_text(det)}")

        if raw_image is not None:
            annotated_dir.mkdir(parents=True, exist_ok=True)
            annotated_image = _compose_annotated_image(raw_image, detections)
            annotated_path = annotated_dir / ANNOTATED_FILENAME_TEMPLATE.format(
                stem=image_path.stem,
                timestamp=timestamp,
                suffix=image_path.suffix.lower() or ".jpg",
            )
            if _write_image_file(annotated_path, annotated_image):
                print(f"[save] Annotated image: {annotated_path}")
            else:
                print(f"[warn] Failed to write annotated image: {annotated_path}")

    summary_path = _write_summary(output_dir, detections_payload, threshold, timestamp)
    print(f"[summary] Saved: {summary_path}")
    return 0


def main() -> None:
    args = _parse_args()
    raise SystemExit(_run_pipeline(args))


if __name__ == "__main__":
    main()
