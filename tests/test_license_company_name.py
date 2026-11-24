"""营业执照企业名称 / 住所解析回归。"""

from app.parsers.license_parser import (
    _extract_address,
    _extract_address_from_boxes,
    _extract_company_name,
    _parse_lines,
)


def test_prefer_real_company_over_qr_instruction():
    lines = [
        "营业执照",
        "扫描二维码登录“国",
        "统一社会信用代码 91410526553164590B",
        "名称",
        "滑县立得安保全服务有限公司",
        "类型有限责任公司(自然人投资或控股)",
        "法定代表人魏义利",
        "住所滑县道口镇解放中路滑县文物保护管理所院内",
    ]
    assert _extract_company_name(lines) == "滑县立得安保全服务有限公司"
    data = _parse_lines(lines)
    assert data["company_name"] == "滑县立得安保全服务有限公司"
    assert data["credit_code"] == "91410526553164590B"


def test_qr_instruction_alone_is_not_company_name():
    lines = [
        "扫描二维码登录“国",
        "统一社会信用代码91410526553164590B",
        "住所滑县道口镇解放中路滑县文物保护管理所院内",
    ]
    assert _extract_company_name(lines) is None


def test_reject_org_type_as_company_name():
    lines = [
        "有限责任公司(自然人投资或控股)",
        "统一社会信用代码91410526553164590B",
        "名称张三水果店",
    ]
    assert _extract_company_name(lines) == "张三水果店"


def test_company_name_inline_with_label():
    lines = [
        "名称滑县立得安保全服务有限公司",
        "统一社会信用代码91410526553164590B",
    ]
    assert _extract_company_name(lines) == "滑县立得安保全服务有限公司"


def test_address_label_plus_next_line():
    """通用规则：命中住所后取本行 + 下一行，直到下一栏目。"""
    lines = [
        "名称某某科技有限公司",
        "统一社会信用代码91410526553164590B",
        "住所滑县道口镇解放中路滑县文物保护管",
        "理所院内",
        "经营范围保安服务",
    ]
    addr = _extract_address(lines, "".join(lines), lines)
    assert addr == "滑县道口镇解放中路滑县文物保护管理所院内"


def test_address_label_alone_then_two_value_lines():
    lines = [
        "住所",
        "河南省安阳市文峰区某某路1号",
        "创业大厦A座801室",
        "登记机关安阳市市场监督管理局",
    ]
    addr = _extract_address(lines, "".join(lines), lines)
    assert addr == "河南省安阳市文峰区某某路1号创业大厦A座801室"


def test_address_split_continuation_two_fragments():
    """续行被拆成两段时，按续写额度依次拼上。"""
    lines = [
        "住所滑县道口镇解放中路滑县文物保护管",
        "理所",
        "院内",
        "经营范围保安服务",
    ]
    addr = _extract_address(lines, "".join(lines), lines)
    assert addr == "滑县道口镇解放中路滑县文物保护管理所院内"


def test_registered_address_label_alias():
    lines = [
        "注册地址北京市海淀区中关村大街1号",
        "3号楼2单元",
        "经营范围技术开发",
    ]
    addr = _extract_address(lines, "".join(lines), lines)
    assert addr == "北京市海淀区中关村大街1号3号楼2单元"


def test_spatial_boxes_label_plus_below():
    boxes = [
        {"text": "住所", "y": 100, "x0": 10, "x1": 40, "y0": 90, "y1": 110},
        {
            "text": "滑县道口镇解放中路滑县文物保护管",
            "y": 100,
            "x0": 80,
            "x1": 400,
            "y0": 90,
            "y1": 110,
        },
        {"text": "理所院内", "y": 130, "x0": 80, "x1": 160, "y0": 120, "y1": 140},
    ]
    assert _extract_address_from_boxes(boxes) == "滑县道口镇解放中路滑县文物保护管理所院内"


def test_reject_gsxt_url_as_address():
    lines = [
        "名称滑县立得安保全服务有限公司",
        "统一社会信用代码91410526553164590B",
        "住所滑县道口镇解放中路滑县文物保护管理所院内",
        "国家企业信用信息公示系统网址：http://www.gsxt.gov.cn",
    ]
    addr = _extract_address(lines, "".join(lines), lines)
    assert addr == "滑县道口镇解放中路滑县文物保护管理所院内"
    assert "gsxt" not in (addr or "").lower()


def test_gsxt_url_alone_is_not_address():
    lines = [
        "国家企业信用信息公示系统网址：http://www.gsxt.gov.cn",
        "统一社会信用代码91410526553164590B",
    ]
    assert _extract_address(lines, "".join(lines), lines) is None
