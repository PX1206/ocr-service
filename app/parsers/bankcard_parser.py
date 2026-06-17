import os
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from app.common.image_preprocess import (
    build_fallback_variants,
    build_number_band_variants,
    build_primary_variants,
    cleanup_temp_paths,
)
from app.common.logging_config import get_logger
from app.common.luhn import luhn_check
from app.common.ocr_engine import run_ocr

logger = get_logger("app.bankcard")

BANK_NAMES = [
    "中国工商银行",
    "中国农业银行",
    "中国银行",
    "中国建设银行",
    "交通银行",
    "招商银行",
    "邮政储蓄银行",
    "浦发银行",
    "中信银行",
    "民生银行",
    "兴业银行",
    "平安银行",
    "广发银行",
    "光大银行",
    "华夏银行",
]

NOISE_KEYWORDS = (
    "VALID",
    "MONTH",
    "YEAR",
    "THRU",
    "DEBIT",
    "CREDIT",
    "银联",
    "UnionPay",
    "有效期",
    "有效期限",
)

# Common BIN prefixes per bank — used to rank ambiguous Luhn repairs.
BANK_BINS: Dict[str, Tuple[str, ...]] = {
    "中国农业银行": ("95599", "622848", "103"),
    "中国工商银行": ("622200", "622202", "9558"),
    "中国银行": ("621785", "621660", "456351", "621256"),
    "中国建设银行": ("622700", "436742", "621081"),
    "交通银行": ("622260", "622258", "521899"),
    "招商银行": ("622588", "622575", "621483"),
    "邮政储蓄银行": ("622188", "621096", "621799"),
}

# Prefer standard card lengths; 17-digit hits are often truncated OCR reads.
LENGTH_PRIORITY = {19: 0, 16: 1, 18: 2, 17: 3, 15: 4}

ALL_KNOWN_BINS = tuple(prefix for prefixes in BANK_BINS.values() for prefix in prefixes)

DATE_LIKE = re.compile(r"^(0[1-9]|1[0-2])/?\d{0,2}$")


def _box_metrics(box: List[List[float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    return min(xs), min(ys), max(xs), max(ys)


def _is_noise_text(text: str) -> bool:
    upper = text.upper()
    return any(keyword in text or keyword in upper for keyword in NOISE_KEYWORDS)


def _is_date_like_digits(digits: str) -> bool:
    if len(digits) > 4:
        return False
    return bool(DATE_LIKE.match(digits))


def _ocr_item_text(item: Any) -> str:
    return item[1][0]


def _ocr_item_score(item: Any) -> Optional[float]:
    if len(item[1]) > 1:
        return float(item[1][1])
    return None


def _log_ocr_variant(variant_name: str, ocr_result: List[Any]) -> None:
    """Log raw OCR boxes and boxes dropped by row extraction filters."""
    raw_boxes: List[str] = []
    skipped_boxes: List[str] = []

    for item in ocr_result:
        text = _ocr_item_text(item)
        score = _ocr_item_score(item)
        digits = re.sub(r"\D", "", text)
        entry = f"text={text!r} score={score:.3f}" if score is not None else f"text={text!r}"
        if digits:
            entry += f" digits={digits!r}"

        raw_boxes.append(entry)

        if _is_noise_text(text):
            skipped_boxes.append(f"noise: {entry}")
        elif len(digits) == 1:
            skipped_boxes.append(f"single_digit: {entry}")
        elif len(digits) >= 2 and _is_date_like_digits(digits):
            skipped_boxes.append(f"date_like: {entry}")
        elif len(digits) < 2:
            skipped_boxes.append(f"non_digit: {entry}")

    logger.info("[%s] ocr_boxes=%d raw=%s", variant_name, len(ocr_result), raw_boxes)
    if skipped_boxes:
        logger.info("[%s] filtered_boxes=%s", variant_name, skipped_boxes)


def _log_digit_clusters(variant_name: str, ocr_result: List[Any]) -> None:
    rows = _extract_digit_rows(ocr_result)
    if not rows:
        logger.info("[%s] digit_rows=0", variant_name)
        return

    clusters = _cluster_rows(rows)
    for index, cluster in enumerate(clusters):
        merged = _merge_cluster(cluster)
        parts = [
            f"{row['text']!r}->{row['digits']!r}@x={row['x']:.0f}"
            for row in sorted(cluster, key=lambda row: row["x"])
        ]
        logger.info(
            "[%s] digit_cluster[%d] merged=%s len=%d parts=%s",
            variant_name,
            index,
            merged,
            len(merged),
            parts,
        )


def _extract_digit_rows(ocr_result: List[Any], y_threshold: float = 24.0) -> List[dict]:
    rows: List[dict] = []

    for item in ocr_result:
        box = item[0]
        text = item[1][0]
        if _is_noise_text(text):
            continue

        digits = re.sub(r"\D", "", text)
        if len(digits) < 2 or _is_date_like_digits(digits):
            continue

        x_min, y_min, x_max, y_max = _box_metrics(box)
        rows.append(
            {
                "x": x_min,
                "y": y_min,
                "x_max": x_max,
                "y_max": y_max,
                "digits": digits,
                "text": text,
            }
        )

    rows.sort(key=lambda row: (row["y"], row["x"]))
    return rows


def _cluster_rows(rows: List[dict], y_threshold: float = 24.0) -> List[List[dict]]:
    if not rows:
        return []

    clusters: List[List[dict]] = [[rows[0]]]
    for row in rows[1:]:
        cluster_y = sum(item["y"] for item in clusters[-1]) / len(clusters[-1])
        if abs(row["y"] - cluster_y) <= y_threshold:
            clusters[-1].append(row)
        else:
            clusters.append([row])
    return clusters


def _merge_cluster(cluster: List[dict]) -> str:
    cluster.sort(key=lambda row: row["x"])
    return "".join(row["digits"] for row in cluster)


def _bin_match_score(card_no: str, bank_name: Optional[str]) -> int:
    if bank_name:
        prefixes = BANK_BINS.get(bank_name, ())
        for prefix in prefixes:
            if card_no.startswith(prefix):
                return 0
        return 1

    for prefix in ALL_KNOWN_BINS:
        if card_no.startswith(prefix):
            return 0
    return 1


def _rank_luhn_hits(
    hits: List[str], bank_name: Optional[str] = None
) -> List[Tuple[str, int]]:
    """Return (card_no, count) sorted best-first."""
    ranked = Counter(hits).most_common()

    def sort_key(item: Tuple[str, int]) -> Tuple[int, int, int, int]:
        card_no, count = item
        length_rank = LENGTH_PRIORITY.get(len(card_no), 10)
        bin_rank = _bin_match_score(card_no, bank_name)
        return (length_rank, bin_rank, -len(card_no), -count)

    return sorted(ranked, key=sort_key)


def _repair_card_digits(
    digits: str,
    min_len: int = 16,
    max_len: int = 19,
    max_missing: int = 2,
) -> List[str]:
    """Recover Luhn-valid card numbers when OCR drops edge digits."""
    if not digits.isdigit() or len(digits) < min_len - max_missing:
        return []

    found: List[str] = []
    current_len = len(digits)

    for target in range(max_len, min_len - 1, -1):
        missing = target - current_len
        if missing < 0:
            continue
        if missing == 0:
            if luhn_check(digits):
                found.append(digits)
            continue
        if missing > max_missing:
            continue

        if missing == 1:
            for d in "0123456789":
                for candidate in (d + digits, digits + d):
                    if len(candidate) == target and luhn_check(candidate):
                        found.append(candidate)
        elif missing == 2:
            for d1 in "0123456789":
                for d2 in "0123456789":
                    candidate = d1 + digits + d2
                    if len(candidate) == target and luhn_check(candidate):
                        found.append(candidate)

    return found


def _collect_repair_candidates(
    candidates: List[str],
    bank_name: Optional[str] = None,
    *,
    log_detail: bool = True,
) -> List[str]:
    """Try Luhn repair on longest digit strings (edge digits often missing)."""
    digit_strings = sorted(
        {re.sub(r"\D", "", text) for text in candidates},
        key=len,
        reverse=True,
    )
    repairs: List[str] = []
    repair_sources: List[str] = []
    for digits in digit_strings[:5]:
        if len(digits) >= 14:
            repaired = _repair_card_digits(digits)
            if repaired:
                repair_sources.append(f"{digits}(+{len(repaired)})")
            repairs.extend(repaired)

    if log_detail:
        if repair_sources:
            ranked = _rank_luhn_hits(repairs, bank_name)[:5]
            logger.info(
                "repair_sources=%s repair_hits=%d top5=%s",
                repair_sources,
                len(repairs),
                ranked,
            )
        else:
            logger.info(
                "repair_sources=[] repair_hits=0 repair_inputs=%s",
                digit_strings[:5],
            )
    return repairs


def _find_luhn_candidates(digits: str, min_len: int = 16, max_len: int = 19) -> List[str]:
    if len(digits) < min_len:
        return []

    found: List[str] = []
    for length in range(max_len, min_len - 1, -1):
        for start in range(len(digits) - length + 1):
            candidate = digits[start : start + length]
            if luhn_check(candidate):
                found.append(candidate)
    return found


def _collect_candidates(ocr_result: List[Any]) -> List[str]:
    candidates: List[str] = []

    for item in ocr_result:
        text = item[1][0]
        if _is_noise_text(text):
            continue
        digits = re.sub(r"\D", "", text)
        if len(digits) >= 4:
            candidates.append(digits)
            candidates.extend(_find_luhn_candidates(digits))

    rows = _extract_digit_rows(ocr_result)
    for cluster in _cluster_rows(rows):
        merged = _merge_cluster(cluster)
        if len(merged) >= 8:
            candidates.append(merged)
            candidates.extend(_find_luhn_candidates(merged))

    # Fallback: only merge rows in the widest cluster (likely card number line).
    if rows:
        clusters = _cluster_rows(rows)
        widest = max(clusters, key=lambda cluster: sum(row["x_max"] - row["x"] for row in cluster))
        merged = _merge_cluster(widest)
        if len(merged) >= 8:
            candidates.append(merged)
            candidates.extend(_find_luhn_candidates(merged))

    return candidates


def _pick_best_card(
    candidates: List[str],
    bank_name: Optional[str] = None,
    *,
    log_detail: bool = True,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Return (card_no, card_no_partial).
    card_no is set only when a 16-19 digit Luhn-valid number is found.
    """
    if not candidates:
        if log_detail:
            logger.info("pick_best_card: no candidates")
        return None, None

    unique_candidates = sorted({re.sub(r"\D", "", text) for text in candidates}, key=len, reverse=True)
    if log_detail:
        logger.info(
            "pick_best_card: candidate_count=%d unique_digit_strings=%s bank=%s",
            len(candidates),
            unique_candidates[:8],
            bank_name,
        )

    luhn_hits: List[str] = []
    for text in candidates:
        luhn_hits.extend(_find_luhn_candidates(text))
    luhn_hits.extend(_collect_repair_candidates(candidates, bank_name, log_detail=log_detail))

    if luhn_hits:
        ranked = _rank_luhn_hits(luhn_hits, bank_name)
        winner, winner_count = ranked[0]
        if log_detail:
            logger.info(
                "pick_best_card: luhn_hits=%d winner=%s len=%d count=%d top3=%s",
                len(luhn_hits),
                winner,
                len(winner),
                winner_count,
                ranked[:3],
            )
            if len(winner) in (17, 18):
                logger.warning(
                    "pick_best_card: non-standard length=%d; OCR may have dropped edge digits",
                    len(winner),
                )
        return winner, None

    digit_only = [re.sub(r"\D", "", text) for text in candidates]
    digit_only = [text for text in digit_only if len(text) >= 12]
    if not digit_only:
        if log_detail:
            logger.info("pick_best_card: no luhn hit and no usable partial digits")
        return None, None

    digit_only.sort(key=len, reverse=True)
    partial = digit_only[0]
    if log_detail:
        logger.info(
            "pick_best_card: luhn miss, partial fallback=%s len=%d",
            partial[:19],
            len(partial),
        )
    if len(partial) >= 16:
        return None, partial[:19]
    return None, partial


def _detect_bank_name(full_text: str) -> Optional[str]:
    for bank in BANK_NAMES:
        if bank in full_text:
            return bank
    return None


def _is_confident_card(
    card_no: Optional[str],
    bank_name: Optional[str],
    candidates: List[str],
) -> bool:
    """True when OCR directly read a 19-digit number matching the bank BIN."""
    if not card_no or len(card_no) != 19 or not luhn_check(card_no):
        return False
    if _bin_match_score(card_no, bank_name) != 0:
        return False
    digit_set = {re.sub(r"\D", "", text) for text in candidates}
    return card_no in digit_set


def _run_variant_ocr(
    variant_name: str,
    variant_path: str,
    *,
    debug: bool,
    data: Dict[str, Optional[str]],
    all_candidates: List[str],
) -> None:
    use_low_score = variant_name.startswith("number_band")
    started = time.perf_counter()
    ocr_result = run_ocr(variant_path, low_score=use_low_score)
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    if not ocr_result:
        logger.info("[%s] ocr_empty elapsed_ms=%d", variant_name, elapsed_ms)
        return

    logger.info("[%s] ocr_done elapsed_ms=%d boxes=%d", variant_name, elapsed_ms, len(ocr_result))
    _log_ocr_variant(variant_name, ocr_result)
    _log_digit_clusters(variant_name, ocr_result)

    if debug:
        print(f"\n========== OCR {variant_name} ==========")
        for item in ocr_result:
            print(_ocr_item_text(item))

    full_text = "".join(_ocr_item_text(item) for item in ocr_result)
    if not data["bank_name"]:
        data["bank_name"] = _detect_bank_name(full_text)

    candidates = _collect_candidates(ocr_result)
    logger.info("[%s] candidates=%s", variant_name, candidates)
    if debug:
        print("candidates:", candidates)

    all_candidates.extend(candidates)


def parse_bank_card(image_path: str, debug: bool = False) -> Dict[str, Optional[str]]:
    data: Dict[str, Optional[str]] = {
        "card_no": None,
        "card_no_partial": None,
        "bank_name": None,
        "holder_name": None,
    }

    all_candidates: List[str] = []
    temp_paths: List[str] = []
    image_label = os.path.basename(image_path)
    variant_tiers = (
        ("primary", build_primary_variants),
        ("fallback", build_fallback_variants),
        ("number_band", build_number_band_variants),
    )
    total_started = time.perf_counter()
    variants_run = 0

    logger.info("parse_bank_card start image=%s tiers=%d", image_label, len(variant_tiers))

    try:
        for tier_name, build_tier in variant_tiers:
            tier_variants = build_tier(image_path)
            logger.info("tier=%s variants=%d", tier_name, len(tier_variants))

            for variant_name, variant_path in tier_variants:
                temp_paths.append(variant_path)
                variants_run += 1
                _run_variant_ocr(
                    variant_name,
                    variant_path,
                    debug=debug,
                    data=data,
                    all_candidates=all_candidates,
                )

                card_no, partial = _pick_best_card(
                    all_candidates,
                    data["bank_name"],
                    log_detail=False,
                )
                if _is_confident_card(card_no, data["bank_name"], all_candidates):
                    data["card_no"] = card_no
                    data["card_no_partial"] = None
                    total_ms = int((time.perf_counter() - total_started) * 1000)
                    logger.info(
                        "parse_bank_card early_exit tier=%s variant=%s variants_run=%d "
                        "total_ms=%d image=%s bank=%s card_no=%s",
                        tier_name,
                        variant_name,
                        variants_run,
                        total_ms,
                        image_label,
                        data["bank_name"],
                        data["card_no"],
                    )
                    return data

                if partial and not card_no:
                    data["card_no_partial"] = partial

        card_no, partial = _pick_best_card(all_candidates, data["bank_name"])
        data["card_no"] = card_no
        data["card_no_partial"] = partial if card_no is None else None
        total_ms = int((time.perf_counter() - total_started) * 1000)
        logger.info(
            "parse_bank_card done variants_run=%d total_ms=%d image=%s bank=%s card_no=%s partial=%s",
            variants_run,
            total_ms,
            image_label,
            data["bank_name"],
            data["card_no"],
            data["card_no_partial"],
        )
        return data
    finally:
        cleanup_temp_paths(temp_paths, keep=image_path)


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "/data/ai/upload/test/333.png"
    result = parse_bank_card(target, debug=True)
    print("\n===== result =====")
    for key, value in result.items():
        print(f"{key}: {value}")
