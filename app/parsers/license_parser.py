import os
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

from app.common.image_preprocess import (
    apply_exif_orientation,
    cleanup_temp_paths,
    enhance_for_embossed_text,
    estimate_upright_rotation_cw,
    rotate_image_cw,
)
from app.common.logging_config import get_logger
from app.common.ocr_engine import run_ocr

logger = get_logger("app.license")

# GB/T 32100-2015 charset (excludes I, O, Z, S, V)
CREDIT_CODE_STRICT = re.compile(r"[0-9A-HJ-NPQRTUWXY]{18}", re.IGNORECASE)
CREDIT_CODE_LOOSE = re.compile(r"[0-9A-Z]{18}", re.IGNORECASE)

CREDIT_CODE_LABELS = ("统一社会信用代码", "社会信用代码", "信用代码")
NAME_LABELS = ("名称", "名 称", "企业名称", "公司名称")
ADDRESS_LABELS = ("住所", "住 所", "住  所", "注册地址", "经营场所", "经营地址")
# 执照栏目标签：用于坐标硬停（下一字段 Y 上界）
SPATIAL_BOUNDARY_LABELS = (
    "统一社会信用代码",
    "社会信用代码",
    "信用代码",
    "名称",
    "企业名称",
    "公司名称",
    "类型",
    "法定代表人",
    "负责人",
    "经营范围",
    "注册资本",
    "成立日期",
    "营业期限",
    "住所",
    "注册地址",
    "经营场所",
    "经营地址",
    "登记机关",
    "组成形式",
)
NAME_LABEL_FRAGMENTS = frozenset({"名", "名称"})
ADDRESS_LABEL_FRAGMENTS = frozenset({"住", "所", "住所", "注册地址", "经营场所", "经营地址"})
ADDRESS_HINT_PATTERN = re.compile(
    r"[\u4e00-\u9fa5]{2,}(?:省|市|自治区|区|县|镇|乡|村|路|街|道|弄|号|园|室|楼|层|大道|广场)",
)
ADDRESS_INLINE_PATTERN = re.compile(
    r"(?:住\s*所|注册地址|经营场所|经营地址)\s*[：: ]*"
    r"([\u4e00-\u9fa50-9A-Za-z\-#（）()、，,.·\s]+?)"
    r"(?=统一社会信用代码|社会信用代码|经营范围|注册资本|成立日期|营业期限|类型|法定代表人|负责人|登记机关|$)",
    re.S,
)
ADDRESS_STOP_LABELS = (
    "法定代表人",
    "负责人",
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
    "副本号",
    "（副本",
    "(副本",
    "市场主体",
    "年度报告",
    "每年1月1日",
)
# 执照页脚网址、年报提示、公示系统说明等，绝不能当住所
ADDRESS_NOISE_PATTERN = re.compile(
    r"http|https|www\.|gsxt\.gov|\.gov\.cn|网址|网站|"
    r"公示系统|信用信息公示|扫描二维码|登录|系统了解更多|"
    r"副本号|副本\)|副本）|（副本|^\s*本）|"
    r"市场主体|年度报告|报送年度|每年1月1日至6月30日",
    re.IGNORECASE,
)
# 住所正文 + 最多续写行数（执照换行通常 1～2 行）
ADDRESS_MAX_CONTINUATION_LINES = 2
# 下一行顶边离上一行底边超过约 1.5 个字高就停止。
# 换行是贴着的；右下角公章隔着空白，只限制行数仍会把章上的单字拼进住所。
CONTINUATION_MAX_GAP_RATIO = 1.5
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
    "负责人",
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
    r"法定代表人|负责人|经营范围|注册资本|成立日期|营业期限|组成形式|"
    r"市场监督|工商行政|核准日期"
)
NAME_FIELD_PREFIXES = (
    "类型",
    "经营范围",
    "注册资本",
    "成立日期",
    "营业期限",
    "法定代表人",
    "负责人",
    "登记机关",
    "统一社会信用代码",
    "社会信用代码",
    "组成形式",
    "住所",
    "名称",
)
# OCR 常把下一栏目标签粘在公司名末尾：…有限公司类型
NAME_TRAILING_LABELS = (
    "统一社会信用代码",
    "社会信用代码",
    "法定代表人",
    "负责人",
    "经营范围",
    "注册资本",
    "成立日期",
    "营业期限",
    "登记机关",
    "组成形式",
    "类型",
    "住所",
)
# 成立日期正文；执照上绝不会是字号的一部分
LICENSE_DATE_PATTERN = re.compile(r"\d{4}年\d{1,2}月\d{1,2}日")
# 名称后粘上「负责人张三 / 法定代表人李四」时，尾部为人名
PERSON_IN_CHARGE_LABELS = ("法定代表人", "负责人")
PERSON_NAME_TAIL_PATTERN = re.compile(r"^[\u4e00-\u9fa5·•]{0,4}$")
# 「类型」被水印切成单字时，仍视为栏目标签
TYPE_LABEL_REMNANTS = ("类型", "型", "类")
# 这些短后缀可能是字号续写（如北京分公司），不能当「粘上的类型栏」
NAME_CONTINUATION_SUFFIXES = frozenset(
    {"分公司", "公司", "企业", "中心", "商行", "工作室", "合作社"}
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


def _is_pure_org_type_description(compact: str) -> bool:
    """仅组织形式、无字号：有限责任公司（自然人投资或控股）。"""
    for suffix in COMPANY_SUFFIXES:
        if not compact.startswith(suffix):
            continue
        rest = compact[len(suffix) :]
        if not rest or re.fullmatch(r"[\(（].+[\)）]", rest):
            return True
        # 分公司执照：有限责任公司分公司(自然人独资)
        if re.fullmatch(r"分公司[\(（].+[\)）]", rest):
            return True
    return False


def _is_name_noise(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return True
    if NAME_NOISE_PATTERN.search(compact):
        return True
    if any(compact.startswith(prefix) for prefix in NAME_FIELD_PREFIXES):
        return True
    # 仅组织形式描述，无字号：如「有限责任公司(自然人投资或控股)」
    if _is_pure_org_type_description(compact):
        return True
    if compact in TYPE_LABEL_REMNANTS:
        return True
    # 成立日期正文被拆成独立框时，不能拼进企业名称
    if LICENSE_DATE_PATTERN.match(compact) and not _has_company_suffix(compact):
        return True
    # OCR 水印/模糊常把「类型」读成「型…」或「类…」，仍是组织形式行
    for remnant in TYPE_LABEL_REMNANTS:
        if compact.startswith(remnant) and _is_pure_org_type_description(compact[len(remnant) :]):
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
        if label not in normalized:
            continue
        idx = normalized.find(label)
        return _normalize_company_name(normalized[idx + len(label) :].strip(" :："))
    return ""


def _is_glued_name_tail(rest: str) -> bool:
    """公司名后是否粘上了下一栏（成立日期、类型残字、组织形式等）。"""
    compact = re.sub(r"\s+", "", rest or "").strip(" :：")
    if not compact:
        return False
    if compact in TYPE_LABEL_REMNANTS:
        return True
    for label in sorted(NAME_TRAILING_LABELS, key=len, reverse=True):
        if compact == label or compact.startswith(label):
            return True
    if LICENSE_DATE_PATTERN.match(compact):
        return True
    for remnant in TYPE_LABEL_REMNANTS:
        if not compact.startswith(remnant):
            continue
        after = compact[len(remnant) :]
        if (
            not after
            or LICENSE_DATE_PATTERN.match(after)
            or _is_pure_org_type_description(after)
        ):
            return True
    if _is_pure_org_type_description(compact) and compact not in NAME_CONTINUATION_SUFFIXES:
        return True
    return False


def _strip_glued_name_tail(value: str) -> str:
    """
    检测框合并时，名称后常带上成立日期和「类」：
    深圳市全一智简信息科技有限公司2018年09月20日类
    """
    cut: Optional[int] = None
    for suffix in COMPANY_SUFFIXES:
        start = 0
        while True:
            idx = value.find(suffix, start)
            if idx < 0:
                break
            end = idx + len(suffix)
            rest = value[end:]
            if rest and _is_glued_name_tail(rest):
                cut = end if cut is None else max(cut, end)
            start = idx + 1
    if cut is not None:
        return value[:cut]

    m = LICENSE_DATE_PATTERN.search(value)
    if m and m.start() >= 4:
        stem = value[: m.start()].rstrip(" :：")
        if len(stem) >= 4 and re.search(r"[\u4e00-\u9fa5]{4,}", stem):
            value = stem
    for remnant in TYPE_LABEL_REMNANTS:
        if value.endswith(remnant) and len(value) > len(remnant) + 2:
            stem = value[: -len(remnant)].rstrip(" :：")
            if len(stem) >= 4:
                return stem
    return value


def _strip_person_in_charge_from_name(value: str) -> str:
    """
    分公司执照栏是「负责人」而非「法定代表人」。
    OCR 常把「…世纪大道站负责人马志威」整段读进名称。
    """
    for label in PERSON_IN_CHARGE_LABELS:
        idx = value.find(label)
        if idx < 4:
            continue
        stem = value[:idx].rstrip(" :：")
        tail = value[idx + len(label) :].strip(" :：")
        if not stem or (tail and not PERSON_NAME_TAIL_PATTERN.match(tail)):
            continue
        if _has_company_suffix(stem) or (
            len(stem) >= 4 and bool(re.search(r"[\u4e00-\u9fa5]{4,}", stem))
        ):
            return stem
    return value


def _normalize_company_name(value: str) -> str:
    value = value.strip(" :：\"'「」『』“”")
    if not value:
        return value
    if value.startswith("称") and len(value) > 3 and _has_company_suffix(value[1:]):
        value = value[1:]
    for prefix in ("企业名称", "公司名称", "名称"):
        if value.startswith(prefix):
            value = value[len(prefix) :].strip(" :：")
            break
    value = _strip_person_in_charge_from_name(value)
    # …有限公司类型 / …有限公司法定代表人
    for label in sorted(NAME_TRAILING_LABELS, key=len, reverse=True):
        if not value.endswith(label) or len(value) <= len(label) + 2:
            continue
        candidate = value[: -len(label)].rstrip(" :：")
        if _has_company_suffix(candidate) or (
            len(candidate) >= 4 and bool(re.search(r"[\u4e00-\u9fa5]{4,}", candidate))
        ):
            value = candidate
            break
    return _strip_glued_name_tail(value)


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
        if normalized.startswith("法定代表人") or normalized.startswith("负责人"):
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
    # 「住」「所」拆框时，值框常以「所」开头粘在地址前
    if value.startswith("所") and len(value) > 2 and re.match(r"[\u4e00-\u9fa5]", value[1]):
        rest = value[1:]
        if (
            ADDRESS_HINT_PATTERN.search(rest)
            or STREET_ADDRESS_PATTERN.search(rest)
            or "号" in rest
            or "路" in rest
        ):
            value = rest
    # 执照页脚「（副本）/副本号/年报提示」常被拼进住所行
    for marker in (
        "本）（副本",
        "（副本",
        "(副本",
        "副本号",
        "市场主体",
        "年度报告",
        "每年1月1日",
    ):
        idx = value.find(marker)
        if idx >= 0:
            value = value[:idx].rstrip(" :：")
            break
    # 尾部英文/混杂噪声（如印章旁 OCR 出的 1906os），中文地址不应以拉丁字母收尾
    value = re.sub(r"[\dA-Za-z]*[A-Za-z]+[\dA-Za-z]*$", "", value).rstrip(" :：-—_")
    return value


def _is_address_noise(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return True
    if ADDRESS_NOISE_PATTERN.search(compact):
        return True
    # 截断后残留的年报碎片：如「通过」「报送」
    if compact in {"通过", "报送", "登录", "了解更多"}:
        return True
    return False


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
    """须带行政区划/道路门牌特征；纯中文长句（年报说明等）不算地址。"""
    value = _clean_address_text(text)
    if len(value) < 4:
        return False
    if _is_address_noise(value):
        return False
    if STREET_ADDRESS_PATTERN.search(value):
        return True
    return bool(ADDRESS_HINT_PATTERN.search(value))


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
        # 下一栏目标签：停止；页脚噪声：已有正文则停，否则跳过继续找真地址
        if any(stop in next_norm for stop in ADDRESS_STOP_LABELS) or _is_field_label_line(next_norm):
            if _is_address_noise(next_norm) and not parts:
                continue
            break
        piece = _truncate_address_at_stops(_strip_address_label_prefix(next_norm))
        if not piece:
            continue
        if _is_address_noise(piece):
            if parts:
                break
            continue
        # 续行本身不必「像完整地址」，直接拼上（如「理所院内」）
        parts.append(piece)
        taken += 1
        if taken >= continuation_budget:
            break

    addr = _clean_address_text("".join(parts))
    if addr and _looks_like_address(addr):
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
        if _looks_like_address(addr):
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


def _norm_box_text(box: Dict[str, Any]) -> str:
    return re.sub(r"\s+", "", box.get("text") or "")


def _boundary_label_compact(norm: str) -> Optional[str]:
    """若该框是栏目标签（可带同行正文），返回命中的标签。"""
    if not norm:
        return None
    # 「类型」常被水印切成「类」「型」，不拦住就会把成立日期拼进名称
    if norm in TYPE_LABEL_REMNANTS:
        return "类型"
    for remnant in TYPE_LABEL_REMNANTS:
        if norm.startswith(remnant) and _is_pure_org_type_description(norm[len(remnant) :]):
            return "类型"
    if norm in NAME_LABEL_FRAGMENTS or norm in ADDRESS_LABEL_FRAGMENTS:
        if norm in {"名", "名称"}:
            return "名称"
        return "住所" if norm in {"住", "所", "住所"} else norm
    # 长标签优先，避免「信用代码」抢在「统一社会信用代码」前误切
    for label in sorted(
        (x.replace(" ", "") for x in SPATIAL_BOUNDARY_LABELS),
        key=len,
        reverse=True,
    ):
        if norm == label:
            return label
        if norm.startswith(label):
            if label == "名称" and "法定代表人" in norm:
                return None
            return label
    return None


def _is_boundary_label_box(box: Dict[str, Any]) -> bool:
    return _boundary_label_compact(_norm_box_text(box)) is not None


def _match_target_label(norm: str, labels: Tuple[str, ...]) -> Optional[Tuple[str, str]]:
    """
    匹配目标字段标签。
    返回 (compact_label, inline_value)；碎片标签 inline 为空。
    """
    if not norm:
        return None
    compact_labels = tuple(label.replace(" ", "") for label in labels)
    fragments = NAME_LABEL_FRAGMENTS if any(l in NAME_LABELS for l in labels) else frozenset()
    if any(l in ADDRESS_LABELS for l in labels):
        fragments = ADDRESS_LABEL_FRAGMENTS

    if norm in fragments:
        if norm in {"名", "名称"} and any(c == "名称" or c.endswith("名称") for c in compact_labels):
            return "名称", ""
        if norm in ADDRESS_LABEL_FRAGMENTS and any(
            c in ADDRESS_LABEL_FRAGMENTS or c in compact_labels for c in compact_labels
        ):
            return (norm if norm in compact_labels else "住所"), ""

    for label in sorted(compact_labels, key=len, reverse=True):
        if label not in norm:
            continue
        if label == "名称" and "法定代表人" in norm:
            continue
        idx = norm.find(label)
        # 标签应在开头附近（栏目标签列），避免正文误命中
        if idx > 2:
            continue
        inline = norm[idx + len(label) :].strip(" :：")
        return label, inline
    return None


def _next_boundary_y(boxes: List[Dict[str, Any]], label_box: Dict[str, Any]) -> float:
    """当前标签下方最近栏目标签的 y0，作为续写硬上界。"""
    limit = float("inf")
    for box in boxes:
        if box is label_box:
            continue
        # 同行或更上的标签不构成下方边界
        if box["y0"] <= label_box["y1"] - 2:
            continue
        if not _is_boundary_label_box(box):
            continue
        limit = min(limit, float(box["y0"]))
    return limit


def _piece_is_skippable_noise(piece: str) -> bool:
    return _is_address_noise(piece) or _is_name_noise(piece)


def _extract_field_by_label(
    boxes: List[Dict[str, Any]],
    labels: Tuple[str, ...],
    *,
    max_below_rows: int,
    clean_fn: Callable[[str], str],
    accept_fn: Callable[[str], bool],
) -> Optional[str]:
    """
    坐标优先：定位标签框 → 同行右侧 + 下方续行，直到下一栏目标签 Y。
    """
    if not boxes:
        return None

    best: Optional[str] = None
    for box in boxes:
        matched = _match_target_label(_norm_box_text(box), labels)
        if not matched:
            continue
        _label_compact, inline = matched

        parts: List[str] = []
        if inline:
            head = clean_fn(inline)
            if head and not _piece_is_skippable_noise(head):
                parts.append(head)

        row_h = max(12.0, float(box["y1"] - box["y0"]))
        stop_y = _next_boundary_y(boxes, box)

        same_row: List[Dict[str, Any]] = []
        for other in boxes:
            if other is box:
                continue
            if abs(other["y"] - box["y"]) > row_h * 0.7:
                continue
            if other["x0"] < box["x0"] + 8:
                continue
            # 同行其他栏目标签不并入
            other_boundary = _boundary_label_compact(_norm_box_text(other))
            if other_boundary and _match_target_label(_norm_box_text(other), labels) is None:
                continue
            same_row.append(other)
        same_row.sort(key=lambda b: b["x0"])

        value_x0 = float(box["x1"])
        anchor_y1 = float(box["y1"])
        line_h = row_h
        for other in same_row:
            piece = clean_fn(other["text"])
            if not piece or _piece_is_skippable_noise(piece):
                continue
            parts.append(piece)
            value_x0 = min(value_x0, float(other["x0"]))
            anchor_y1 = max(anchor_y1, float(other["y1"]))
            line_h = max(line_h, float(other["y1"] - other["y0"]))

        below = [
            other
            for other in boxes
            if other is not box
            and other["y0"] >= box["y1"] - 4
            and other["y0"] < stop_y
            and other["x0"] >= min(box["x0"] - 5, value_x0 - 10)
        ]
        below.sort(key=lambda b: (b["y"], b["x0"]))

        # 标签独占一行时，下一行才是正文，续写额度 +1
        below_budget = max_below_rows + (1 if not parts else 0)
        taken = 0
        last_y = None
        for other in below:
            # 下方若出现其他栏目标签，硬停（stop_y 已拦，双保险）
            if _is_boundary_label_box(other) and _match_target_label(
                _norm_box_text(other), labels
            ) is None:
                break
            # 隔着大段空白不再跳着拼（公章顶上的「市」常在登记机关之上）
            if float(other["y0"]) - anchor_y1 > line_h * CONTINUATION_MAX_GAP_RATIO:
                break
            piece = clean_fn(other["text"])
            if not piece:
                continue
            if _piece_is_skippable_noise(piece):
                if parts:
                    break
                continue
            if last_y is None or abs(other["y"] - last_y) > row_h * 0.6:
                taken += 1
                last_y = other["y"]
            if taken > below_budget:
                break
            parts.append(piece)
            anchor_y1 = max(anchor_y1, float(other["y1"]))
            line_h = max(line_h, float(other["y1"] - other["y0"]))

        if not parts:
            continue

        value = clean_fn("".join(parts))
        if value and accept_fn(value):
            if best is None or len(value) > len(best):
                best = value
    return best


def _clean_spatial_name_piece(raw: str) -> str:
    value = _normalize_company_name(re.sub(r"\s+", "", raw or ""))
    for label in ("企业名称", "公司名称", "名称"):
        if value.startswith(label):
            value = value[len(label) :].strip(" :：")
    return value


def _clean_spatial_address_piece(raw: str) -> str:
    value = _strip_address_label_prefix(re.sub(r"\s+", "", raw or ""))
    return _truncate_address_at_stops(_clean_address_text(value))


def _extract_company_name_from_boxes(boxes: List[Dict[str, Any]]) -> Optional[str]:
    return _extract_field_by_label(
        boxes,
        NAME_LABELS,
        max_below_rows=1,
        clean_fn=_clean_spatial_name_piece,
        accept_fn=lambda v: _looks_like_company_name(v, require_suffix=False),
    )


def _extract_address_from_boxes(boxes: List[Dict[str, Any]]) -> Optional[str]:
    """坐标版：住所标签 → 右侧/下方，下一栏目标签 Y 硬停。"""
    return _extract_field_by_label(
        boxes,
        ADDRESS_LABELS,
        max_below_rows=ADDRESS_MAX_CONTINUATION_LINES,
        clean_fn=_clean_spatial_address_piece,
        accept_fn=_looks_like_address,
    )


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
            if addr and _looks_like_address(addr) and (best is None or len(addr) > len(best)):
                best = addr
    return best


def _extract_address_text_fallback(
    lines: List[str],
    full_text: str,
    ocr_rows: Optional[List[str]] = None,
) -> Optional[str]:
    """文本规则兜底（不含坐标路径）。"""
    labeled_candidates: List[str] = []
    for source in (
        _extract_address_from_lines(lines),
        _extract_address_from_lines(ocr_rows or []),
        _extract_address_from_full_text(full_text),
    ):
        if source and _looks_like_address(source) and not _is_address_noise(source):
            labeled_candidates.append(source)
    if labeled_candidates:
        return max(labeled_candidates, key=len)

    fallback: List[str] = []
    for line in lines + (ocr_rows or []):
        value = _clean_address_text(line)
        if _looks_like_address(value) and not _is_address_noise(value) and not _is_field_label_line(line):
            fallback.append(value)
    if not fallback:
        return None
    return max(fallback, key=len)


def _extract_address(
    lines: List[str],
    full_text: str,
    ocr_rows: Optional[List[str]] = None,
    boxes: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """坐标优先；失败才走文本兜底。空间结果不再与文本比最长。"""
    spatial = _extract_address_from_boxes(boxes or [])
    if spatial:
        return spatial
    return _extract_address_text_fallback(lines, full_text, ocr_rows)


def _parse_lines(
    lines: List[str],
    ocr_rows: Optional[List[str]] = None,
    boxes: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Optional[str]]:
    dedup = _dedup_lines(lines)
    full_text = "".join(dedup)
    box_list = boxes or []
    data = _empty_result()
    data["credit_code"] = _extract_credit_code(dedup, full_text)
    data["company_name"] = _extract_company_name_from_boxes(box_list) or _extract_company_name(
        dedup
    )
    data["address"] = _extract_address(dedup, full_text, ocr_rows, box_list)
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


def _resolve_upright_image(
    image_path: str, temp_paths: List[str]
) -> Tuple[str, int, Optional[List[Any]]]:
    """
    EXIF 转正 + 根据首轮 OCR 框估计文字方向，必要时整图旋转。
    返回 (正图路径, 额外顺时针旋转角度, 可复用的 probe OCR 结果)。
    已正图时复用 probe，避免多跑一遍；旋转后 probe 作废，需对正图重跑 OCR。
    """
    oriented = apply_exif_orientation(image_path)
    if oriented != image_path:
        temp_paths.append(oriented)

    probe = run_ocr(oriented, low_score=False)
    cw = estimate_upright_rotation_cw(probe)
    if cw == 0:
        logger.info("license orientation upright_cw=0 boxes=%d", len(probe))
        return oriented, 0, probe

    upright = rotate_image_cw(oriented, cw)
    if upright != oriented:
        temp_paths.append(upright)
    logger.info(
        "license orientation upright_cw=%d probe_boxes=%d path=%s",
        cw,
        len(probe),
        os.path.basename(upright),
    )
    return upright, cw, None


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
        # 先正图再坐标取值：用户上传角度不固定，未转正时「取右侧」会错
        working_path, upright_cw, reusable_ocr = _resolve_upright_image(
            image_path, temp_paths
        )

        variants = [
            ("upright" if upright_cw else "original", working_path, False),
            ("enhanced", None, False),
            # 住所换行小字（如「院内」）常被高阈值丢掉，补一轮低阈值
            ("upright_low_score" if upright_cw else "original_low_score", working_path, True),
        ]
        for variant_name, variant_path, low_score in variants:
            if variant_path is None:
                variant_path = enhance_for_embossed_text(working_path)
                temp_paths.append(variant_path)
            variants_run += 1

            # 无需旋转时复用方向探测 OCR，少一次推理
            if (
                reusable_ocr is not None
                and variant_path == working_path
                and not low_score
            ):
                lines = _extract_lines(reusable_ocr)
                rows = _rows_from_ocr(reusable_ocr)
                boxes = _boxes_from_ocr(reusable_ocr)
                reusable_ocr = None
            else:
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
                    "parse_business_license early_exit variant=%s variants_run=%d upright_cw=%d "
                    "total_ms=%d company_name=%s credit_code=%s address=%s",
                    variant_name,
                    variants_run,
                    upright_cw,
                    total_ms,
                    result.get("company_name"),
                    _mask_credit_code(result.get("credit_code")),
                    result.get("address"),
                )
                return result

        total_ms = int((time.perf_counter() - total_started) * 1000)
        logger.info(
            "parse_business_license done variants_run=%d upright_cw=%d total_ms=%d "
            "company_name=%s credit_code=%s address=%s",
            variants_run,
            upright_cw,
            total_ms,
            result.get("company_name"),
            _mask_credit_code(result.get("credit_code")),
            result.get("address"),
        )
        return result
    finally:
        cleanup_temp_paths(temp_paths, keep=image_path)


parse_license = parse_business_license
