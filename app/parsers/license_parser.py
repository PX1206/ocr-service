import re
import time
from typing import Any, Dict, List, Optional, Tuple

from app.common.image_preprocess import cleanup_temp_paths, enhance_for_embossed_text
from app.common.logging_config import get_logger
from app.common.ocr_engine import run_ocr

logger = get_logger("app.license")

# GB/T 32100-2015 charset (excludes I, O, Z, S, V)
CREDIT_CODE_STRICT = re.compile(r"[0-9A-HJ-NPQRTUWXY]{18}", re.IGNORECASE)
CREDIT_CODE_LOOSE = re.compile(r"[0-9A-Z]{18}", re.IGNORECASE)

CREDIT_CODE_LABELS = ("统一社会信用代码", "社会信用代码", "信用代码")
NAME_LABELS = ("名称", "名 称", "企业名称", "公司名称")
ADDRESS_LABELS = ("住所", "住 所", "住  所", "注册地址")
ADDRESS_HINT_PATTERN = re.compile(
    r"[\u4e00-\u9fa5]{2,}(?:省|市|自治区|区|县|镇|乡|村|路|街|道|弄|号|园|室|楼|层|大道|广场)",
)
ADDRESS_INLINE_PATTERN = re.compile(
    r"住\s*所\s*[：: ]*([\u4e00-\u9fa50-9A-Za-z\-#（）()、，,.·\s]+?)"
    r"(?=统一社会信用代码|社会信用代码|经营范围|注册资本|成立日期|营业期限|类型|法定代表人|登记机关|$)",
    re.S,
)
ADDRESS_STOP_LABELS = (
    "法定代表人",
    "注册资本",
    "成立日期",
    "营业期限",
    "经营范围",
    "统一社会信用代码",
    "社会信用代码",
    "类型",
    "登记机关",
)
STREET_ADDRESS_PATTERN = re.compile(
    r"[\u4e00-\u9fa5]{2,}"
    r"(?:省|市|自治区|区|县|镇|乡|村|大道|路|街|街道|巷|弄|号|园|室|楼|层|段|广场|开发区|高新区)"
    r"[\u4e00-\u9fa50-9A-Za-z#（）()、，,.·\-层室号楼段路街乡镇村园广场开发区高新区]{4,}",
)

FIELD_STOP_LABELS = (
    "经营范围",
    "注册资本",
    "成立日期",
    "营业期限",
    "统一社会信用代码",
    "社会信用代码",
    "类型",
    "法定代表人",
    "名称",
    "登记机关",
    "组成形式",
)

COMPANY_SUFFIXES = (
    "有限责任公司",
    "股份有限公司",
    "有限公司",
    "合伙企业",
    "个体工商户",
    "合作社",
    "分公司",
    "公司",
    "企业",
    "中心",
    "商行",
    "工作室",
)


def _empty_result() -> Dict[str, Optional[str]]:
    return {
        "company_name": None,
        "credit_code": None,
        "address": None,
    }


def _extract_lines(ocr_result: List[Any]) -> List[str]:
    return [item[1][0].strip() for item in ocr_result if item[1][0].strip()]


def _rows_from_ocr(ocr_result: List[Any], y_threshold: float = 18.0) -> List[str]:
    items: List[Dict[str, Any]] = []
    for item in ocr_result:
        text = item[1][0].strip()
        if not text:
            continue
        box = item[0]
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        items.append({"text": text, "y": sum(ys) / len(ys), "x": min(xs)})

    if not items:
        return []

    items.sort(key=lambda t: (t["y"], t["x"]))
    rows: List[List[Dict[str, Any]]] = []
    for item in items:
        if not rows or abs(item["y"] - rows[-1][0]["y"]) > y_threshold:
            rows.append([item])
        else:
            rows[-1].append(item)

    out: List[str] = []
    for row in rows:
        row.sort(key=lambda t: t["x"])
        out.append("".join(part["text"] for part in row))
    return out


def _merge_split_label_lines(lines: List[str]) -> List[str]:
    merged: List[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        norm = re.sub(r"\s+", "", line)
        if norm == "名" and i + 1 < len(lines):
            merged.append(f"{line}{lines[i + 1]}")
            i += 2
            continue
        if norm in {"住", "所", "住所"} and i + 1 < len(lines):
            next_norm = re.sub(r"\s+", "", lines[i + 1])
            if norm == "所" and _looks_like_address(next_norm):
                merged.append(f"{line}{lines[i + 1]}")
                i += 2
                continue
            if norm in {"住", "住所"} or next_norm in {"所", "住所"}:
                merged.append(f"{line}{lines[i + 1]}")
                i += 2
                continue
        if norm.endswith("住") and i + 1 < len(lines):
            merged.append(f"{line}{lines[i + 1]}")
            i += 2
            continue
        merged.append(line)
        i += 1
    return merged


def _dedup_lines(lines: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for line in lines:
        key = re.sub(r"\s+", "", line)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(line)
    return out


def _normalize_credit_code(raw: str) -> Optional[str]:
    compact = re.sub(r"[\s\-_]", "", raw).upper()
    if len(compact) != 18:
        return None

    fixes = (
        compact.replace("O", "0")
        .replace("I", "1")
        .replace("L", "1")
        .replace("B", "8")
        .replace("S", "5")
        .replace("Z", "2")
    )

    for candidate in (compact, fixes):
        if CREDIT_CODE_STRICT.fullmatch(candidate):
            return candidate
    if CREDIT_CODE_LOOSE.fullmatch(compact):
        return compact
    return None


def _find_credit_code_in_text(text: str) -> Optional[str]:
    compact = re.sub(r"[\s\-_]", "", text).upper()
    for pattern in (CREDIT_CODE_STRICT, CREDIT_CODE_LOOSE):
        for match in pattern.finditer(compact):
            normalized = _normalize_credit_code(match.group(0))
            if normalized:
                return normalized
    return None


def _extract_credit_code(lines: List[str], full_text: str) -> Optional[str]:
    for line in lines:
        normalized_line = line.replace(" ", "")
        for label in CREDIT_CODE_LABELS:
            if label not in normalized_line:
                continue
            idx = normalized_line.find(label)
            tail = normalized_line[idx + len(label) :].strip(" :：")
            code = _find_credit_code_in_text(tail)
            if code:
                return code

    return _find_credit_code_in_text(full_text)


def _looks_like_company_name(text: str) -> bool:
    value = text.strip(" :：")
    if len(value) < 4:
        return False
    if ADDRESS_HINT_PATTERN.search(value) and not any(suffix in value for suffix in COMPANY_SUFFIXES):
        return False
    if any(suffix in value for suffix in COMPANY_SUFFIXES):
        return True
    return len(value) >= 6 and bool(re.search(r"[\u4e00-\u9fa5]{4,}", value))


def _strip_name_label(line: str) -> str:
    normalized = line.replace(" ", "")
    for label in NAME_LABELS:
        if label in normalized and "法定代表人" not in normalized:
            idx = normalized.find(label)
            return _normalize_company_name(normalized[idx + len(label) :].strip(" :："))
    return ""


def _normalize_company_name(value: str) -> str:
    value = value.strip(" :：")
    if not value:
        return value
    if value.startswith("称") and _looks_like_company_name(value[1:]):
        return value[1:]
    return value


def _is_field_label_line(line: str) -> bool:
    normalized = line.replace(" ", "")
    return any(label in normalized for label in FIELD_STOP_LABELS)


def _clean_address_text(raw: str) -> str:
    value = re.sub(r"\s+", "", raw).strip(" :：")
    for label in ("住所", "注册地址", "经营场所"):
        if value.startswith(label):
            value = value[len(label) :].strip(" :：")
    return value


def _looks_like_address(text: str) -> bool:
    value = _clean_address_text(text)
    if len(value) < 6:
        return False
    if ADDRESS_HINT_PATTERN.search(value):
        return True
    if re.search(r"[\u4e00-\u9fa5]{4,}", value):
        return True
    return bool(re.search(r"\d", value)) and len(value) >= 8


def _extract_address_from_full_text(full_text: str) -> Optional[str]:
    compact = re.sub(r"\s+", "", full_text)
    match = ADDRESS_INLINE_PATTERN.search(compact)
    if match:
        addr = _clean_address_text(match.group(1))
        if _looks_like_address(addr):
            return addr
    return None


def _line_has_address_label(normalized: str) -> bool:
    return any(label.replace(" ", "") in normalized for label in ADDRESS_LABELS)


def _strip_address_label_prefix(normalized: str) -> str:
    for label in ADDRESS_LABELS:
        label_compact = label.replace(" ", "")
        if normalized.startswith(label_compact):
            return normalized[len(label_compact) :].strip(" :：")
    if normalized.startswith("所") and len(normalized) > 1:
        tail = normalized[1:].strip(" :：")
        if _looks_like_address(tail):
            return tail
    return normalized


def _extract_address_from_lines(lines: List[str]) -> Optional[str]:
    merged_lines = _merge_split_label_lines(lines)
    for i, line in enumerate(merged_lines):
        normalized = re.sub(r"\s+", "", line)
        if not _line_has_address_label(normalized):
            continue
        for label in ADDRESS_LABELS:
            label_compact = label.replace(" ", "")
            if label_compact not in normalized:
                continue
            idx = normalized.find(label_compact)
            label_len = len(label_compact)
            parts: List[str] = []
            head = normalized[idx + label_len :].strip(" :：")
            if head:
                parts.append(head)
            for j in range(i + 1, len(merged_lines)):
                next_line = merged_lines[j].strip(" :：")
                if not next_line:
                    continue
                next_norm = re.sub(r"\s+", "", next_line)
                if any(stop in next_norm for stop in ADDRESS_STOP_LABELS):
                    break
                if _is_field_label_line(next_line):
                    break
                parts.append(_strip_address_label_prefix(next_norm))
            addr = _clean_address_text("".join(parts))
            if _looks_like_address(addr):
                return addr
    return None


def _extract_address_between_row_labels(rows: List[str]) -> Optional[str]:
    if not rows:
        return None
    for i, row in enumerate(rows):
        normalized = re.sub(r"\s+", "", row)
        if not _line_has_address_label(normalized):
            continue
        for label in ADDRESS_LABELS:
            label_compact = label.replace(" ", "")
            if label_compact not in normalized:
                continue
            idx = normalized.find(label_compact)
            tail = normalized[idx + len(label_compact) :].strip(" :：")
            if _looks_like_address(tail):
                return _clean_address_text(tail)
        parts: List[str] = []
        for j in range(i + 1, len(rows)):
            next_norm = re.sub(r"\s+", "", rows[j])
            if not next_norm:
                continue
            if any(stop in next_norm for stop in ADDRESS_STOP_LABELS):
                break
            if next_norm in {"住", "所", "住所"}:
                continue
            parts.append(_strip_address_label_prefix(next_norm))
        addr = _clean_address_text("".join(parts))
        if _looks_like_address(addr):
            return addr
    return None


def _extract_address_by_street_pattern(lines: List[str]) -> Optional[str]:
    candidates: List[str] = []
    for line in lines:
        value = _clean_address_text(line)
        if "统一社会信用代码" in value or _is_field_label_line(line):
            continue
        for match in STREET_ADDRESS_PATTERN.finditer(value):
            candidate = match.group(0).strip(" :：")
            if len(candidate) >= 8 and _looks_like_address(candidate):
                candidates.append(candidate)
    if not candidates:
        return None
    return max(candidates, key=len)


def _extract_address_fallback(lines: List[str]) -> Optional[str]:
    candidates: List[str] = []
    for line in lines:
        value = _clean_address_text(line)
        if _looks_like_address(value) and "统一社会信用代码" not in value:
            if not _is_field_label_line(line):
                candidates.append(value)
    if not candidates:
        return None
    return max(candidates, key=len)


def _extract_address(lines: List[str], full_text: str, ocr_rows: Optional[List[str]] = None) -> Optional[str]:
    for source in (
        _extract_address_from_lines(lines),
        _extract_address_between_row_labels(ocr_rows or []),
        _extract_address_from_full_text(full_text),
        _extract_address_from_lines(ocr_rows or []),
        _extract_address_by_street_pattern(lines),
        _extract_address_by_street_pattern(ocr_rows or []),
        _extract_address_fallback(lines),
        _extract_address_fallback(ocr_rows or []),
    ):
        if source:
            return source
    return None


def _extract_company_name(lines: List[str]) -> Optional[str]:
    for i, line in enumerate(lines):
        normalized = line.replace(" ", "")
        if "法定代表人" in normalized:
            continue
        if not any(label in normalized for label in NAME_LABELS):
            continue

        value = _strip_name_label(line)
        if _looks_like_company_name(value):
            return _normalize_company_name(value)

        if i + 1 < len(lines):
            next_line = _normalize_company_name(lines[i + 1].strip(" :："))
            if _looks_like_company_name(next_line) and "统一社会信用代码" not in next_line:
                return next_line

    for line in lines:
        value = _normalize_company_name(line.strip(" :："))
        if _looks_like_company_name(value) and "统一社会信用代码" not in value:
            return value

    return None


def _parse_lines(lines: List[str], ocr_rows: Optional[List[str]] = None) -> Dict[str, Optional[str]]:
    dedup = _dedup_lines(lines)
    full_text = "".join(dedup)
    data = _empty_result()
    data["credit_code"] = _extract_credit_code(dedup, full_text)
    data["company_name"] = _extract_company_name(dedup)
    data["address"] = _extract_address(dedup, full_text, ocr_rows)
    return data


def _is_confident(data: Dict[str, Optional[str]]) -> bool:
    return bool(data.get("company_name") and data.get("credit_code") and data.get("address"))


def _mask_credit_code(code: Optional[str]) -> Optional[str]:
    if not code or len(code) < 8:
        return code
    return f"{code[:4]}****{code[-4:]}"


def _run_variant(variant_name: str, variant_path: str) -> Tuple[List[str], List[str], int]:
    started = time.perf_counter()
    ocr_result = run_ocr(variant_path)
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    if not ocr_result:
        logger.info("[%s] ocr_empty elapsed_ms=%d", variant_name, elapsed_ms)
        return [], [], elapsed_ms

    lines = _extract_lines(ocr_result)
    rows = _rows_from_ocr(ocr_result)
    logger.info(
        "[%s] ocr_done elapsed_ms=%d boxes=%d lines=%d rows=%d",
        variant_name,
        elapsed_ms,
        len(ocr_result),
        len(lines),
        len(rows),
    )
    return lines, rows, elapsed_ms


def parse_business_license(image_path: str, debug: bool = False) -> Dict[str, Optional[str]]:
    """Parse Chinese business license (营业执照) — company name, credit code and address."""
    temp_paths: List[str] = [image_path]
    all_lines: List[str] = []
    all_rows: List[str] = []
    total_started = time.perf_counter()
    variants_run = 0
    result = _empty_result()

    logger.info("parse_business_license start")

    try:
        for variant_name, variant_path in (
            ("original", image_path),
            ("enhanced", None),
        ):
            if variant_path is None:
                variant_path = enhance_for_embossed_text(image_path)
            temp_paths.append(variant_path)
            variants_run += 1

            lines, rows, _elapsed_ms = _run_variant(variant_name, variant_path)
            if debug and lines:
                print(f"\n========== LICENSE OCR {variant_name} ==========")
                for line in lines:
                    print(line)

            all_lines.extend(lines)
            all_rows.extend(rows)
            result = _parse_lines(all_lines, all_rows)

            if _is_confident(result):
                total_ms = int((time.perf_counter() - total_started) * 1000)
                logger.info(
                    "parse_business_license early_exit variant=%s variants_run=%d total_ms=%d "
                    "company_name=%s credit_code=%s address=%s",
                    variant_name,
                    variants_run,
                    total_ms,
                    result.get("company_name"),
                    _mask_credit_code(result.get("credit_code")),
                    result.get("address"),
                )
                return result

        total_ms = int((time.perf_counter() - total_started) * 1000)
        logger.info(
            "parse_business_license done variants_run=%d total_ms=%d company_name=%s credit_code=%s address=%s",
            variants_run,
            total_ms,
            result.get("company_name"),
            _mask_credit_code(result.get("credit_code")),
            result.get("address"),
        )
        return result
    finally:
        cleanup_temp_paths(temp_paths, keep=image_path)


parse_license = parse_business_license
