import re
import time
from typing import Any, Dict, List, Optional, Tuple

from app.common.id_checksum import id_card_checksum_valid
from app.common.image_preprocess import cleanup_temp_paths, enhance_for_embossed_text
from app.common.logging_config import get_logger
from app.common.ocr_engine import run_ocr

logger = get_logger("app.idcard")

ID_NUMBER_PATTERN = re.compile(r"\d{17}[\dXx]")
BIRTH_PATTERN = re.compile(r"(\d{4}年\d{1,2}月\d{1,2})")
VALID_DATE_PATTERN = re.compile(r"\d{4}[.\-年]\d{1,2}[.\-月]\d{1,2}")


def _empty_result() -> Dict[str, Optional[str]]:
    return {
        "side": None,
        "name": None,
        "gender": None,
        "nation": None,
        "birth": None,
        "address": None,
        "id_no": None,
        "issue_authority": None,
        "valid_start": None,
        "valid_end": None,
    }


def _normalize_id_number(text: str) -> Optional[str]:
    compact = re.sub(r"[\s\-]", "", text).upper()
    match = ID_NUMBER_PATTERN.search(compact)
    if not match:
        return None

    candidate = match.group(0)
    if id_card_checksum_valid(candidate):
        return candidate

    fixes = (
        candidate.replace("O", "0")
        .replace("o", "0")
        .replace("I", "1")
        .replace("l", "1")
        .replace("B", "8")
        .replace("S", "5")
    )
    if id_card_checksum_valid(fixes):
        return fixes

    # Checksum failed but pattern matched — still return for manual review.
    return candidate


def _extract_lines(ocr_result: List[Any]) -> List[str]:
    return [item[1][0].strip() for item in ocr_result if item[1][0].strip()]


def _detect_side(full_text: str, id_match: Optional[re.Match]) -> str:
    if id_match:
        return "front"
    if "签发机关" in full_text or "有效期限" in full_text:
        return "back"
    return "unknown"


def _parse_gender_nation_line(line: str, data: Dict[str, Optional[str]]) -> None:
    if "性别" in line:
        gender_part = line.split("民族", 1)[0] if "民族" in line else line
        value = gender_part.replace("性别", "").strip(" :：")
        if value in ("男", "女"):
            data["gender"] = value
        else:
            match = re.search(r"[男女]", value)
            if match:
                data["gender"] = match.group(0)

    if "民族" in line:
        idx = line.find("民族")
        nation_val = line[idx + 2:].strip(" :：")
        nation_match = re.match(r"([\u4e00-\u9fa5]{1,10})", nation_val)
        if nation_match:
            data["nation"] = nation_match.group(1)
        elif nation_val:
            data["nation"] = nation_val


def _parse_front(texts: List[str], full_text: str, id_match: Optional[re.Match]) -> Dict[str, Optional[str]]:
    data = _empty_result()
    data["side"] = "front"

    for line in texts:
        if "姓名" in line:
            data["name"] = line.replace("姓名", "").strip(" :：")

        elif "性别" in line or "民族" in line:
            _parse_gender_nation_line(line, data)

    if id_match:
        data["id_no"] = _normalize_id_number(id_match.group(0))
    else:
        data["id_no"] = _normalize_id_number(full_text)

    birth_match = BIRTH_PATTERN.search(full_text)
    if birth_match:
        birth = birth_match.group(1)
        if not birth.endswith("日"):
            birth += "日"
        data["birth"] = birth

    addr_lines: List[str] = []
    collecting = False
    for line in texts:
        if "住址" in line:
            collecting = True
            addr_lines.append(line.replace("住址", "").strip(" :："))
            continue

        if collecting:
            if ID_NUMBER_PATTERN.search(line) or "公民" in line:
                break
            addr_lines.append(line)

    if addr_lines:
        data["address"] = "".join(addr_lines)

    return data


def _parse_back(texts: List[str], full_text: str) -> Dict[str, Optional[str]]:
    data = _empty_result()
    data["side"] = "back"

    for idx, line in enumerate(texts):
        if "签发机关" in line:
            value = line.replace("签发机关", "").strip(" :：")
            if value:
                data["issue_authority"] = value
            elif idx + 1 < len(texts):
                data["issue_authority"] = texts[idx + 1].strip()

        elif "有效期限" in line:
            value = line.replace("有效期限", "")
            dates = VALID_DATE_PATTERN.findall(value)
            if len(dates) >= 2:
                data["valid_start"] = dates[0]
                data["valid_end"] = dates[1]
            elif len(dates) == 1 and "长期" in value:
                data["valid_start"] = dates[0]
                data["valid_end"] = "长期"

    if not data["valid_start"]:
        dates = VALID_DATE_PATTERN.findall(full_text)
        if len(dates) >= 2:
            data["valid_start"] = dates[0]
            data["valid_end"] = dates[1]
        elif len(dates) == 1 and "长期" in full_text:
            data["valid_start"] = dates[0]
            data["valid_end"] = "长期"

    if not data["issue_authority"]:
        match = re.search(r"签发机关\s*(.+?)(?:有效期限|$)", full_text, re.S)
        if match:
            authority = re.sub(r"\s+", "", match.group(1))
            if authority:
                data["issue_authority"] = authority

    return data


def _dedup_lines(lines: List[str]) -> List[str]:
    dedup_texts: List[str] = []
    seen = set()
    for line in lines:
        if line not in seen:
            seen.add(line)
            dedup_texts.append(line)
    return dedup_texts


def _resolve_parse(
    dedup_texts: List[str],
    merged_text: str,
    side: str,
) -> Dict[str, Optional[str]]:
    id_match = ID_NUMBER_PATTERN.search(re.sub(r"[\s\-]", "", merged_text))

    if side == "front":
        return _parse_front(dedup_texts, merged_text, id_match)
    if side == "back":
        return _parse_back(dedup_texts, merged_text)

    detected = _detect_side(merged_text, id_match)
    if detected == "front":
        return _parse_front(dedup_texts, merged_text, id_match)
    if detected == "back":
        return _parse_back(dedup_texts, merged_text)

    front = _parse_front(dedup_texts, merged_text, id_match)
    if front.get("id_no") or front.get("name"):
        return front

    data = _parse_back(dedup_texts, merged_text)
    data["side"] = "unknown"
    return data


def _is_confident_front(data: Dict[str, Optional[str]]) -> bool:
    id_no = data.get("id_no")
    if not id_no or not data.get("name"):
        return False
    return id_card_checksum_valid(id_no)


def _is_confident_back(data: Dict[str, Optional[str]]) -> bool:
    return bool(
        data.get("issue_authority")
        and data.get("valid_start")
        and data.get("valid_end")
    )


def _is_confident_result(data: Dict[str, Optional[str]], side: str) -> bool:
    if side == "front":
        return _is_confident_front(data)
    if side == "back":
        return _is_confident_back(data)

    detected = data.get("side")
    if detected == "front":
        return _is_confident_front(data)
    if detected == "back":
        return _is_confident_back(data)
    return False


def _mask_id_no(id_no: Optional[str]) -> Optional[str]:
    if not id_no or len(id_no) < 8:
        return id_no
    return f"{id_no[:6]}****{id_no[-4:]}"


def _run_idcard_variant(variant_name: str, variant_path: str) -> Tuple[List[str], int]:
    started = time.perf_counter()
    ocr_result = run_ocr(variant_path)
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    if not ocr_result:
        logger.info("[%s] ocr_empty elapsed_ms=%d", variant_name, elapsed_ms)
        return [], elapsed_ms

    lines = _extract_lines(ocr_result)
    logger.info("[%s] ocr_done elapsed_ms=%d boxes=%d lines=%d", variant_name, elapsed_ms, len(ocr_result), len(lines))
    return lines, elapsed_ms


def parse_idcard(image_path: str, side: str = "auto", debug: bool = False) -> Dict[str, Optional[str]]:
    """
    Parse Chinese resident ID card image.

    side: auto | front | back
    """
    temp_paths: List[str] = [image_path]
    all_lines: List[str] = []
    total_started = time.perf_counter()
    variants_run = 0
    result = _empty_result()

    logger.info("parse_idcard start side=%s", side)

    try:
        for variant_name, variant_path in (
            ("original", image_path),
            ("enhanced", None),
        ):
            if variant_path is None:
                variant_path = enhance_for_embossed_text(image_path)
            temp_paths.append(variant_path)
            variants_run += 1

            lines, _elapsed_ms = _run_idcard_variant(variant_name, variant_path)
            if debug and lines:
                print(f"\n========== ID OCR {variant_name} ==========")
                for line in lines:
                    print(line)

            all_lines.extend(lines)
            dedup_texts = _dedup_lines(all_lines)
            merged_text = "".join(dedup_texts)
            result = _resolve_parse(dedup_texts, merged_text, side)

            if _is_confident_result(result, side):
                total_ms = int((time.perf_counter() - total_started) * 1000)
                logger.info(
                    "parse_idcard early_exit variant=%s variants_run=%d total_ms=%d "
                    "side=%s id_no=%s name=%s",
                    variant_name,
                    variants_run,
                    total_ms,
                    result.get("side"),
                    _mask_id_no(result.get("id_no")),
                    result.get("name"),
                )
                return result

        total_ms = int((time.perf_counter() - total_started) * 1000)
        logger.info(
            "parse_idcard done variants_run=%d total_ms=%d side=%s id_no=%s name=%s",
            variants_run,
            total_ms,
            result.get("side"),
            _mask_id_no(result.get("id_no")),
            result.get("name"),
        )
        return result
    finally:
        cleanup_temp_paths(temp_paths, keep=image_path)


# Alias for HTTP layer / external callers
parse_id_card = parse_idcard


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "/data/ai/upload/test/111.png"
    result = parse_idcard(target, debug=True)
    print("\n===== 解析结果 =====")
    for key, value in result.items():
        print(key, ":", value)
