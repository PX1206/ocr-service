import math
import os
import tempfile
from typing import Any, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps


def _save_cv_image(image: np.ndarray, suffix: str = ".jpg") -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    cv2.imwrite(tmp.name, image)
    return tmp.name


def apply_exif_orientation(image_path: str) -> str:
    """按 EXIF Orientation 转正（手机竖拍常见），无需旋转则返回原路径。"""
    try:
        with Image.open(image_path) as img:
            transposed = ImageOps.exif_transpose(img)
            if transposed is None or transposed is img:
                return image_path
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            tmp.close()
            transposed.convert("RGB").save(tmp.name, quality=95)
            return tmp.name
    except Exception:
        return image_path


def rotate_image_cw(image_path: str, degrees: int) -> str:
    """顺时针旋转整图 0/90/180/270，返回新临时文件路径（0 度返回原路径）。"""
    deg = int(degrees) % 360
    if deg == 0:
        return image_path
    if deg not in (90, 180, 270):
        raise ValueError(f"unsupported rotation degrees: {degrees}")

    image = cv2.imread(image_path)
    if image is None:
        return image_path

    if deg == 90:
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif deg == 180:
        rotated = cv2.rotate(image, cv2.ROTATE_180)
    else:
        rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return _save_cv_image(rotated)


def _box_text_angle_deg(box: Sequence[Sequence[float]]) -> float:
    """沿文字行方向的角度（图像坐标 y 向下，atan2，单位度，归一化到 [0, 360)）。"""
    pts = np.asarray(box, dtype=float)
    if pts.shape[0] < 2:
        return 0.0
    d01 = float(np.linalg.norm(pts[1] - pts[0]))
    d12 = float(np.linalg.norm(pts[2] - pts[1])) if pts.shape[0] > 2 else 0.0
    if d01 >= d12:
        vec = pts[1] - pts[0]
    else:
        vec = pts[2] - pts[1]
    angle = math.degrees(math.atan2(float(vec[1]), float(vec[0]))) % 360.0
    return angle


def estimate_upright_rotation_cw(ocr_result: List[Any]) -> int:
    """
    根据 OCR 框的文字走向，估计把整图转到「横向从左到右」所需的顺时针角度。

    Paddle angle_cls 只纠正单行识别，不转整图；斜着/竖着上传时坐标「取右侧」会失效，
    需先整图旋转再二次 OCR。
    """
    if not ocr_result:
        return 0

    votes = {0: 0.0, 90: 0.0, 180: 0.0, 270: 0.0}
    for item in ocr_result:
        try:
            box = item[0]
            text = item[1][0] if item[1] else ""
        except (TypeError, IndexError, KeyError):
            continue
        if not box or not text:
            continue
        weight = float(max(len(str(text).strip()), 1))
        angle = _box_text_angle_deg(box)
        snap = int(round(angle / 90.0) % 4) * 90
        # 文字当前沿 snap° 方向阅读；转到 0°（水平向右）需顺时针旋转 (360-snap)%360
        cw = (360 - snap) % 360
        votes[cw] += weight

    best = max(votes.items(), key=lambda kv: kv[1])[0]
    # 票数过低或几乎打平则不转，避免噪声误旋
    total = sum(votes.values())
    if total <= 0 or votes[best] < total * 0.45:
        return 0
    return int(best)


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
