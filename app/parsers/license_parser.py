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
ADDRESS_LABELS = ("住所", "住 所", "住  所", "注册地址", "经营场所", "经营地址")
ADDRESS_HINT_PATTERN = re.compile(
    r"[\u4e00-\u9fa5]{2,}(?:省|市|自治区|区|县|镇|乡|村|路|街|道|弄|号|园|室|楼|层|大道|广场)",
)
ADDRESS_INLINE_PATTERN = re.compile(
    r"(?:住\s*所|注册地址|经营场所|经营地址)\s*[：: ]*"
    r"([\u4e00-\u9fa50-9A-Za-z\-#（）()、，,.·\s]+?)"
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
    "国家企业信用信息公示系统",
    "公示系统",
)
# 执照页脚网址、公示系统说明等，绝不能当住所
ADDRESS_NOISE_PATTERN = re.compile(
    r"http|https|www\.|gsxt\.gov|\.gov\.cn|网址|网站|"
    r"公示系统|信用信息公示|扫描二维码|登录|系统了解更多",
    re.IGNORECASE,
)
# 住所正文 + 最多续写行数（执照换行通常 1～2 行）
ADDRESS_MAX_CONTINUATION_LINES = 2
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

# 执照二维码旁说明、水印、栏目名等，绝不能当企业名称
NAME_NOISE_PATTERN = re.compile(
    r"扫描|二维码|登录|微信|支付宝|公示系统|信用信息|查询本企业|请使用|"
    r"国徽|登记机关|副本|正本|营业执照|统一社会信用代码|社会信用代码|"
    r"法定代表人|经营范围|注册资本|成立日期|营业期限|组成形式|"
    r"市场监督|工商行政|核准日期"
)
NAME_FIELD_PREFIXES = (
    "类型",
    "经营范围",
    "注册资本",
    "成立日期",
    "营业期限",
    "法定代表人",
    "登记机关",
    "统一社会信用代码",
    "社会信用代码",
    "组成形式",
    "住所",
    "名称",
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


def _has_company_suffix(value: str) -> bool:
    return any(suffix in value for suffix in COMPANY_SUFFIXES)


def _is_name_noise(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return True
    if NAME_NOISE_PATTERN.search(compact):
        return True
    if any(compact.startswith(prefix) for prefix in NAME_FIELD_PREFIXES):
        return True
    # 仅组织形式描述，无字号：如「有限责任公司(自然人投资或控股)」
    for suffix in COMPANY_SUFFIXES:
        if compact.startswith(suffix):
            rest = compact[len(suffix) :]
            if not rest or re.fullmatch(r"[\(（].+[\)）]", rest):
                return True
    return False


def _looks_like_company_name(text: str, *, require_suffix: bool = False) -> bool:
    value = text.strip(" :：\"'「」『』“”")
    if len(value) < 4:
        return False
    if _is_name_noise(value):
        return False
    if _find_credit_code_in_text(value):
        return False
    if ADDRESS_HINT_PATTERN.search(value) and not _has_company_suffix(value):
        return False
    if _has_company_suffix(value):
        return bool(re.search(r"[\u4e00-\u9fa5]{2,}", value))
    if require_suffix:
        return False
    # 仅用于「名称」标签旁：个体字号等可能无标准后缀
    return len(value) >= 4 and bool(re.search(r"[\u4e00-\u9fa5]{4,}", value))


def _strip_name_label(line: str) -> str:
    normalized = line.replace(" ", "")
    for label in NAME_LABELS:
        if label in normalized and "法定代表人" not in normalized:
            idx = normalized.find(label)
            return _normalize_company_name(normalized[idx + len(label) :].strip(" :："))
    return ""


def _normalize_company_name(value: str) -> str:
    value = value.strip(" :：\"'「」『』“”")
    if not value:
        return value
    if value.startswith("称") and _looks_like_company_name(value[1:], require_suffix=False):
        return value[1:]
    return value


def _company_name_score(name: str) -> int:
    score = min(len(name), 40)
    for i, suffix in enumerate(COMPANY_SUFFIXES):
        if suffix in name:
            # 越靠前的后缀越具体
            score = max(score, 100 - i) + min(len(name), 40)
            break
    return score


def _pick_best_company_name(candidates: List[str]) -> Optional[str]:
    cleaned = []
    seen = set()
    for raw in candidates:
        name = _normalize_company_name(raw)
        if not name or name in seen:
            continue
        seen.add(name)
        cleaned.append(name)
    if not cleaned:
        return None
    return max(cleaned, key=_company_name_score)


def _is_field_label_line(line: str) -> bool:
    normalized = line.replace(" ", "")
    return any(label in normalized for label in FIELD_STOP_LABELS)


def _extract_company_name(lines: List[str]) -> Optional[str]:
    merged = _merge_split_label_lines(lines)
    labeled: List[str] = []
    with_suffix: List[str] = []

    for i, line in enumerate(merged):
        normalized = line.replace(" ", "")
        if "法定代表人" in normalized:
            continue
        if not any(label in normalized for label in NAME_LABELS):
            continue

        value = _strip_name_label(line)
        if _looks_like_company_name(value, require_suffix=False):
            labeled.append(value)

        if i + 1 < len(merged):
            next_line = _normalize_company_name(merged[i + 1])
            if (
                _looks_like_company_name(next_line, require_suffix=False)
                and "统一社会信用代码" not in next_line
            ):
                labeled.append(next_line)

    for line in merged:
        value = _normalize_company_name(line)
        if _looks_like_company_name(value, require_suffix=True) and "统一社会信用代码" not in value:
            with_suffix.append(value)

    labeled_with_suffix = [c for c in labeled if _has_company_suffix(c)]
    # 优先：名称标签旁且带组织形式 → 全文带组织形式 → 名称标签旁无后缀字号
    return (
        _pick_best_company_name(labeled_with_suffix)
        or _pick_best_company_name(with_suffix)
        or _pick_best_company_name(labeled)
    )


def _clean_address_text(raw: str) -> str:
    value = re.sub(r"\s+", "", raw).strip(" :：")
    for label in ("住所", "注册地址", "经营场所", "经营地址"):
        if value.startswith(label):
            value = value[len(label) :].strip(" :：")
    return value


def _is_address_noise(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return True
    return bool(ADDRESS_NOISE_PATTERN.search(compact))


def _truncate_address_at_stops(value: str) -> str:
    """同一行里若拼进了下一栏目（经营范围等），截到栏目名之前。"""
    cut_at = None
    for stop in ADDRESS_STOP_LABELS:
        idx = value.find(stop)
        if idx > 0 and (cut_at is None or idx < cut_at):
            cut_at = idx
    if cut_at is not None:
        return value[:cut_at].rstrip(" :：")
    return value


def _looks_like_address(text: str) -> bool:
    value = _clean_address_text(text)
    if len(value) < 4:
        return False
    if _is_address_noise(value):
        return False
    if ADDRESS_HINT_PATTERN.search(value):
        return True
    return bool(STREET_ADDRESS_PATTERN.search(value)) or (
        len(value) >= 6 and bool(re.search(r"[\u4e00-\u9fa5]{4,}", value))
    )


def _line_has_address_label(normalized: str) -> bool:
    return any(label.replace(" ", "") in normalized for label in ADDRESS_LABELS)


def _strip_address_label_prefix(normalized: str) -> str:
    for label in ADDRESS_LABELS:
        label_compact = label.replace(" ", "")
        if normalized.startswith(label_compact):
            return normalized[len(label_compact) :].strip(" :：")
    if normalized.startswith("所") and len(normalized) > 1:
        return normalized[1:].strip(" :：")
    return normalized


def _is_address_field_stop(normalized: str) -> bool:
    """下一栏目 / 页脚，不再并入住所。"""
    if not normalized:
        return True
    if normalized in {"住", "所", "住所", "注册地址", "经营场所", "经营地址"}:
        return False
    if _is_address_noise(normalized):
        return True
    if any(stop in normalized for stop in ADDRESS_STOP_LABELS):
        return True
    if _is_field_label_line(normalized):
        return True
    return False


def _collect_address_after_label(
    lines: List[str], label_index: int, label_compact: str
) -> Optional[str]:
    """
    通用规则：命中「住所/注册地址/经营场所」后，
    取本行正文 + 紧随其后的续行，直到下一栏目标签。
    不依赖具体断词，适用于任意换行执照。
    """
    if label_index < 0 or label_index >= len(lines):
        return None
    normalized = re.sub(r"\s+", "", lines[label_index])
    idx = normalized.find(label_compact)
    if idx < 0:
        return None

    parts: List[str] = []
    head = _truncate_address_at_stops(normalized[idx + len(label_compact) :].strip(" :："))
    if head and not _is_address_noise(head):
        parts.append(head)

    # 标签独占一行时，下一行才是正文，仍计入续写额度
    continuation_budget = ADDRESS_MAX_CONTINUATION_LINES
    if not head:
        continuation_budget += 1

    taken = 0
    for j in range(label_index + 1, len(lines)):
        next_norm = re.sub(r"\s+", "", lines[j])
        if not next_norm:
            continue
        if next_norm in {"住", "所", "住所", "注册地址", "经营场所", "经营地址"}:
            continue
        if _is_address_field_stop(next_norm):
            break
        piece = _truncate_address_at_stops(_strip_address_label_prefix(next_norm))
        if not piece or _is_address_noise(piece):
            break
        # 续行本身不必「像完整地址」，直接拼上（如「理所院内」）
        parts.append(piece)
        taken += 1
        if taken >= continuation_budget:
            break

    addr = _clean_address_text("".join(parts))
    if addr and (_looks_like_address(addr) or len(addr) >= 6):
        return addr
    return None


def _extract_address_from_full_text(full_text: str) -> Optional[str]:
    compact = re.sub(r"\s+", "", full_text)
    match = ADDRESS_INLINE_PATTERN.search(compact)
    if match:
        addr = _clean_address_text(match.group(1))
        cut = ADDRESS_NOISE_PATTERN.search(addr)
        if cut:
            addr = addr[: cut.start()].rstrip(" :：")
        addr = _truncate_address_at_stops(addr)
        if _looks_like_address(addr) or len(addr) >= 6:
            return addr
    return None


def _boxes_from_ocr(ocr_result: List[Any]) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for item in ocr_result:
        text = item[1][0].strip()
        if not text:
            continue
        box = item[0]
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        items.append(
            {
                "text": text,
                "y": sum(ys) / len(ys),
                "x0": min(xs),
                "x1": max(xs),
                "y0": min(ys),
                "y1": max(ys),
            }
        )
    items.sort(key=lambda t: (t["y"], t["x0"]))
    return items


def _extract_address_from_boxes(boxes: List[Dict[str, Any]]) -> Optional[str]:
    """空间版同一规则：标签行正文 + 正下方最多 N 行续写。"""
    if not boxes:
        return None
    best: Optional[str] = None
    for box in boxes:
        norm = re.sub(r"\s+", "", box["text"])
        label_hit = None
        for label in ADDRESS_LABELS:
            compact = label.replace(" ", "")
            if compact in norm or norm in {"住", "所", "住所"}:
                label_hit = compact if compact in norm else "住所"
                break
        if not label_hit:
            continue

        parts: List[str] = []
        if label_hit in norm and norm not in {"住", "所", "住所"}:
            head = _truncate_address_at_stops(
                norm[norm.find(label_hit) + len(label_hit) :].strip(" :：")
            )
            if head and not _is_address_noise(head):
                parts.append(head)

        row_h = max(12.0, box["y1"] - box["y0"])
        value_x0 = box["x1"]
        # 同行右侧
        for other in boxes:
            if other is box:
                continue
            if abs(other["y"] - box["y"]) > row_h * 0.7:
                continue
            if other["x0"] < box["x0"] + 8:
                continue
            piece = _truncate_address_at_stops(_clean_address_text(other["text"]))
            if piece and not _is_address_noise(piece) and not _is_address_field_stop(piece):
                parts.append(piece)
                value_x0 = min(value_x0, other["x0"])

        # 正下方续行
        below = [
            other
            for other in boxes
            if other is not box
            and other["y0"] >= box["y1"] - 4
            and other["y0"] <= box["y1"] + row_h * (ADDRESS_MAX_CONTINUATION_LINES + 0.8)
            and not (other["x1"] < value_x0 - 10 and other["x0"] < box["x0"] + 20)
        ]
        below.sort(key=lambda b: (b["y"], b["x0"]))
        taken = 0
        last_y = None
        for other in below:
            piece = _truncate_address_at_stops(_clean_address_text(other["text"]))
            if not piece or _is_address_noise(piece) or _is_address_field_stop(piece):
                if piece and _is_address_field_stop(piece):
                    break
                continue
            # 按视觉行计数
            if last_y is None or abs(other["y"] - last_y) > row_h * 0.6:
                taken += 1
                last_y = other["y"]
            if taken > ADDRESS_MAX_CONTINUATION_LINES:
                break
            parts.append(piece)

        addr = _clean_address_text("".join(parts))
        if addr and (_looks_like_address(addr) or len(addr) >= 6):
            if best is None or len(addr) > len(best):
                best = addr
    return best


def _extract_address_from_lines(lines: List[str]) -> Optional[str]:
    merged_lines = _merge_split_label_lines(lines)
    best: Optional[str] = None
    for i, line in enumerate(merged_lines):
        normalized = re.sub(r"\s+", "", line)
        if not _line_has_address_label(normalized):
            continue
        for label in ADDRESS_LABELS:
            label_compact = label.replace(" ", "")
            if label_compact not in normalized:
                continue
            addr = _collect_address_after_label(merged_lines, i, label_compact)
            if addr and (best is None or len(addr) > len(best)):
                best = addr
    return best


def _extract_address(
    lines: List[str],
    full_text: str,
    ocr_rows: Optional[List[str]] = None,
    boxes: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """优先走「标签行 + 续行」；找不到标签时才兜底。"""
    labeled_candidates: List[str] = []
    for source in (
        _extract_address_from_boxes(boxes or []),
        _extract_address_from_lines(lines),
        _extract_address_from_lines(ocr_rows or []),
        _extract_address_from_full_text(full_text),
    ):
        if source and not _is_address_noise(source):
            labeled_candidates.append(source)
    if labeled_candidates:
        return max(labeled_candidates, key=len)

    # 无标签时的弱兜底：带行政区划/道路特征的最长一行
    fallback: List[str] = []
    for line in lines + (ocr_rows or []):
        value = _clean_address_text(line)
        if _looks_like_address(value) and not _is_address_noise(value) and not _is_field_label_line(line):
            fallback.append(value)
    if not fallback:
        return None
    return max(fallback, key=len)


def _parse_lines(
    lines: List[str],
    ocr_rows: Optional[List[str]] = None,
    boxes: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Optional[str]]:
    dedup = _dedup_lines(lines)
    full_text = "".join(dedup)
    data = _empty_result()
    data["credit_code"] = _extract_credit_code(dedup, full_text)
    data["company_name"] = _extract_company_name(dedup)
    data["address"] = _extract_address(dedup, full_text, ocr_rows, boxes)
    return data


def _is_confident(data: Dict[str, Optional[str]]) -> bool:
    return bool(data.get("company_name") and data.get("credit_code") and data.get("address"))


def _mask_credit_code(code: Optional[str]) -> Optional[str]:
    if not code or len(code) < 8:
        return code
    return f"{code[:4]}****{code[-4:]}"


def _run_variant(
    variant_name: str, variant_path: str, *, low_score: bool = False
) -> Tuple[List[str], List[str], List[Dict[str, Any]], int]:
    started = time.perf_counter()
    ocr_result = run_ocr(variant_path, low_score=low_score)
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    if not ocr_result:
        logger.info("[%s] ocr_empty elapsed_ms=%d low_score=%s", variant_name, elapsed_ms, low_score)
        return [], [], [], elapsed_ms

    lines = _extract_lines(ocr_result)
    rows = _rows_from_ocr(ocr_result)
    boxes = _boxes_from_ocr(ocr_result)
    logger.info(
        "[%s] ocr_done elapsed_ms=%d boxes=%d lines=%d rows=%d low_score=%s",
        variant_name,
        elapsed_ms,
        len(ocr_result),
        len(lines),
        len(rows),
        low_score,
    )
    return lines, rows, boxes, elapsed_ms


def parse_business_license(image_path: str, debug: bool = False) -> Dict[str, Optional[str]]:
    """Parse Chinese business license (营业执照) — company name, credit code and address."""
    temp_paths: List[str] = [image_path]
    all_lines: List[str] = []
    all_rows: List[str] = []
    all_boxes: List[Dict[str, Any]] = []
    total_started = time.perf_counter()
    variants_run = 0
    result = _empty_result()

    logger.info("parse_business_license start")

    try:
        variants = [
            ("original", image_path, False),
            ("enhanced", None, False),
            # 住所换行小字（如「院内」）常被高阈值丢掉，补一轮低阈值
            ("original_low_score", image_path, True),
        ]
        for variant_name, variant_path, low_score in variants:
            if variant_path is None:
                variant_path = enhance_for_embossed_text(image_path)
                temp_paths.append(variant_path)
            variants_run += 1

            lines, rows, boxes, _elapsed_ms = _run_variant(
                variant_name, variant_path, low_score=low_score
            )
            if debug and lines:
                print(f"\n========== LICENSE OCR {variant_name} ==========")
                for line in lines:
                    print(line)

            all_lines.extend(lines)
            all_rows.extend(rows)
            all_boxes.extend(boxes)
            result = _parse_lines(all_lines, all_rows, all_boxes)

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
