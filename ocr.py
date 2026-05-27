from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import cv2
import numpy as np

OCR_LANGUAGES = ("en",)
OCR_UPSCALE_FACTOR = 2.5
PRICE_VALUE_CHARS = frozenset("0123456789.")
OPTICAL_DIGIT_CORRECTION_MAP = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "S": "5",
        "s": "5",
        "I": "1",
        "l": "1",
        "i": "1",
        "|": "1",
    }
)
COMMA_CHARACTERS = (",", "，")
type OCRNumber = int | float
type OCRResolveResult = tuple[OCRNumber | None, float | None]


class OCRReader(Protocol):
    def readtext(self, image: np.ndarray) -> list[Any]: ...


@dataclass(frozen=True)
class OCRCandidate:
    value: OCRNumber
    center_x: float
    height: float
    confidence: float


def read_image_bgr(image_path: Path) -> np.ndarray | None:
    buffer = np.fromfile(str(image_path), dtype=np.uint8)
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


class PriceOCRService:
    def __init__(
        self,
        reader: OCRReader | None = None,
        *,
        languages: Sequence[str] = OCR_LANGUAGES,
        upscale_factor: float = OCR_UPSCALE_FACTOR,
    ):
        self._reader = reader
        self.languages = tuple(languages)
        self.upscale_factor = upscale_factor

    @property
    def reader(self) -> OCRReader:
        if self._reader is None:
            import easyocr

            self._reader = easyocr.Reader(list(self.languages))
        return self._reader

    def resolve_price(self, image: np.ndarray, bbox: Sequence[int]) -> OCRResolveResult:
        crop = self.crop_price_tag(image, bbox)
        if crop is None:
            return None, None

        processed = self.preprocess_for_ocr(crop, self.upscale_factor)
        result = self.reader.readtext(processed)
        return self.select_price_value(result)

    def resolve_price_from_file(
        self, image_path: Path, bbox: Sequence[int]
    ) -> OCRResolveResult:
        image = read_image_bgr(image_path)
        if image is None:
            return None, None
        return self.resolve_price(image, bbox)

    @staticmethod
    def crop_price_tag(image: np.ndarray, bbox: Sequence[int]) -> np.ndarray | None:
        if image.size == 0:
            return None

        height, width = image.shape[:2]
        normalized_bbox = PriceOCRService._normalize_xyxy_bbox(bbox)
        if normalized_bbox is None:
            return None

        x1, y1, x2, y2 = normalized_bbox
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            return None
        return image[y1:y2, x1:x2].copy()

    @staticmethod
    def preprocess_for_ocr(
        img: np.ndarray, upscale_factor: float = OCR_UPSCALE_FACTOR
    ) -> np.ndarray:
        upscaled = cv2.resize(
            img,
            None,
            fx=upscale_factor,
            fy=upscale_factor,
            interpolation=cv2.INTER_CUBIC,
        )
        gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
        return gray

    @classmethod
    def select_price_value(cls, result: Sequence[Any]) -> OCRResolveResult:
        candidates = [
            candidate
            for item in result
            if (candidate := cls._candidate_from_result(item)) is not None
        ]
        if not candidates:
            return None, None
        selected = max(
            candidates,
            key=lambda candidate: candidate.height * candidate.center_x,
        )
        return selected.value, selected.confidence

    @staticmethod
    def _normalize_xyxy_bbox(bbox: Sequence[Any]) -> tuple[int, int, int, int] | None:
        if len(bbox) < 4:
            return None
        try:
            x1, y1, x2, y2 = [int(round(float(value))) for value in bbox[:4]]
        except (TypeError, ValueError):
            return None
        return x1, y1, x2, y2

    @classmethod
    def _candidate_from_result(cls, item: Any) -> OCRCandidate | None:
        if not isinstance(item, Sequence) or len(item) < 2:
            return None

        bbox = item[0]
        value_text = cls._normalize_price_value(item[1])
        if value_text is None:
            return None

        numeric_value = cls._parse_numeric_value(value_text)
        if numeric_value is None:
            return None

        confidence = 0.0
        if len(item) >= 3:
            try:
                confidence = float(item[2])
            except (TypeError, ValueError):
                confidence = 0.0

        return OCRCandidate(
            value=numeric_value,
            center_x=cls._bbox_center_x(bbox),
            height=cls._bbox_height(bbox),
            confidence=confidence,
        )

    @staticmethod
    def _normalize_price_value(value: Any) -> str | None:
        compact = "".join(str(value).split())
        if not compact:
            return None

        corrected = compact.translate(OPTICAL_DIGIT_CORRECTION_MAP)
        for comma in COMMA_CHARACTERS:
            corrected = corrected.replace(comma, "")

        if not corrected:
            return None
        if any(char not in PRICE_VALUE_CHARS for char in corrected):
            return None
        if not any(char.isdigit() for char in corrected):
            return None
        return corrected

    @staticmethod
    def _parse_numeric_value(value: str) -> OCRNumber | None:
        if value.count(".") > 1:
            return None
        if value == ".":
            return None
        try:
            parsed = float(value)
        except ValueError:
            return None
        if parsed.is_integer() and "." not in value:
            return int(parsed)
        return parsed

    @staticmethod
    def _bbox_height(bbox: Any) -> float:
        points = np.asarray(bbox, dtype=float)
        if points.ndim == 1 and points.size >= 4:
            return abs(float(points[3]) - float(points[1]))
        if points.ndim != 2 or points.shape[1] < 2:
            return 0.0
        y_values = points[:, 1]
        return float(np.max(y_values) - np.min(y_values))

    @staticmethod
    def _bbox_center_x(bbox: Any) -> float:
        points = np.asarray(bbox, dtype=float)
        if points.ndim == 1 and points.size >= 4:
            top_left_x = float(points[0])
            bottom_right_x = float(points[2])
            return (top_left_x + bottom_right_x) / 2.0
        if points.ndim != 2 or points.shape[1] < 1:
            return 0.0
        if points.shape[0] >= 3:
            top_left_x = float(points[0, 0])
            bottom_right_x = float(points[2, 0])
            return (top_left_x + bottom_right_x) / 2.0
        x_values = points[:, 0]
        return float((np.min(x_values) + np.max(x_values)) / 2.0)
