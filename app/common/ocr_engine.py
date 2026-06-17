import os
import tempfile
import time
from typing import Any, List

from PIL import Image

from app.common.logging_config import get_logger

logger = get_logger("app.ocr")

_ocr_instance = None
_ocr_low_score_instance = None


def get_ocr(*, low_score: bool = False):
    global _ocr_instance, _ocr_low_score_instance
    if low_score:
        if _ocr_low_score_instance is None:
            from paddleocr import PaddleOCR

            _ocr_low_score_instance = PaddleOCR(
                use_angle_cls=True,
                lang="ch",
                show_log=False,
                drop_score=0.25,
            )
        return _ocr_low_score_instance

    if _ocr_instance is None:
        from paddleocr import PaddleOCR

        _ocr_instance = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
    return _ocr_instance


def run_ocr(image_path: str, cls: bool = True, *, low_score: bool = False) -> List[Any]:
    """Run OCR and return the first page result list."""
    result = get_ocr(low_score=low_score).ocr(image_path, cls=cls)
    if not result or not result[0]:
        return []
    return result[0]


def _dummy_image_path() -> str:
    """Minimal blank image to warm detection + recognition pipelines."""
    image = Image.new("RGB", (128, 64), color=(255, 255, 255))
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    image.save(tmp.name, quality=85)
    return tmp.name


def warmup_ocr(*, run_dummy_ocr: bool = True) -> None:
    """Load PaddleOCR singletons at startup to avoid first-request cold start."""
    total_started = time.perf_counter()

    started = time.perf_counter()
    get_ocr(low_score=False)
    default_ms = int((time.perf_counter() - started) * 1000)
    logger.info("ocr warmup default instance ready elapsed_ms=%d", default_ms)

    started = time.perf_counter()
    get_ocr(low_score=True)
    low_score_ms = int((time.perf_counter() - started) * 1000)
    logger.info("ocr warmup low_score instance ready elapsed_ms=%d", low_score_ms)

    if run_dummy_ocr:
        dummy_path = _dummy_image_path()
        try:
            started = time.perf_counter()
            run_ocr(dummy_path, low_score=False)
            infer_ms = int((time.perf_counter() - started) * 1000)
            logger.info("ocr warmup dummy infer done elapsed_ms=%d", infer_ms)
        finally:
            if os.path.exists(dummy_path):
                os.remove(dummy_path)

    total_ms = int((time.perf_counter() - total_started) * 1000)
    logger.info("ocr warmup complete total_ms=%d", total_ms)
