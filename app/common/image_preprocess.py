import os
import tempfile
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image


def _save_cv_image(image: np.ndarray, suffix: str = ".jpg") -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    cv2.imwrite(tmp.name, image)
    return tmp.name


def resize_image(image_path: str, scale: float) -> str:
    img = Image.open(image_path)
    resized = img.resize(
        (int(img.width * scale), int(img.height * scale)),
        Image.LANCZOS,
    )
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    tmp.close()
    resized.save(tmp.name, quality=95)
    return tmp.name


def enhance_for_embossed_text(image_path: str) -> str:
    """Improve embossed silver digits on patterned card backgrounds."""
    image = cv2.imread(image_path)
    if image is None:
        return image_path

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)
    return _save_cv_image(sharpened)


def _pad_image(image: np.ndarray, padding_pct: float = 0.06) -> np.ndarray:
    """Add white border so edge digits are not clipped by OCR."""
    height, width = image.shape[:2]
    pad_x = max(int(width * padding_pct), 12)
    pad_y = max(int(height * padding_pct), 8)
    if len(image.shape) == 2:
        border_value = 255
    else:
        border_value = (255, 255, 255)
    return cv2.copyMakeBorder(
        image, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=border_value
    )


def crop_card_number_band(image_path: str) -> str:
    """Crop the horizontal band where embossed card numbers usually appear."""
    image = cv2.imread(image_path)
    if image is None:
        return image_path

    height, width = image.shape[:2]
    y1 = int(height * 0.28)
    y2 = int(height * 0.58)
    cropped = image[y1:y2, 0:width]
    padded = _pad_image(cropped)
    return _save_cv_image(padded)


def enhance_for_printed_digits(image_path: str) -> str:
    """Improve dark printed digits on patterned card backgrounds (e.g. ABC debit)."""
    image = cv2.imread(image_path)
    if image is None:
        return image_path

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced, 5, 50, 50)
    binary = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 8
    )
    return _save_cv_image(binary)


def build_primary_variants(image_path: str) -> List[Tuple[str, str]]:
    """Fast path: single OCR on the uploaded image."""
    return [("original", image_path)]


def build_fallback_variants(image_path: str) -> List[Tuple[str, str]]:
    """Second tier when primary OCR lacks a confident 19-digit BIN match."""
    return [
        ("scale_1.5", resize_image(image_path, 1.5)),
        ("enhanced", enhance_for_embossed_text(image_path)),
        ("scale_2.0", resize_image(image_path, 2.0)),
    ]


def build_number_band_variants(image_path: str) -> List[Tuple[str, str]]:
    """Last tier for embossed cards / hard backgrounds (crop + enhance)."""
    number_band = crop_card_number_band(image_path)
    return [
        ("number_band", number_band),
        ("number_band_enhanced", enhance_for_embossed_text(number_band)),
        ("number_band_printed", enhance_for_printed_digits(number_band)),
    ]


def build_ocr_variants(image_path: str) -> List[Tuple[str, str]]:
    """
    Return all (variant_name, image_path) pairs for multi-pass OCR.
    Prefer tiered helpers + early exit in bankcard_parser for faster responses.
    """
    variants = build_primary_variants(image_path)
    variants.extend(build_fallback_variants(image_path))
    variants.extend(build_number_band_variants(image_path))
    return variants


def cleanup_temp_paths(paths: List[str], keep: str) -> None:
    for path in paths:
        if path != keep and os.path.exists(path):
            os.remove(path)
