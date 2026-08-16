"""图表渲染工具的单元测试。"""

import pytest
from fakes import FakeObjectStorage
from langchain_core.tools import BaseTool
from pydantic import ValidationError

from agent_app.errors import AppError, ErrorCode
from agent_app.tools.chart_tools import create_chart_agent_tools


def _storage() -> FakeObjectStorage:
    return FakeObjectStorage()


def _invoke(tool: BaseTool, **kwargs) -> dict:
    return tool.invoke(kwargs)


def test_returns_empty_list_when_storage_missing() -> None:
    """storage 未配置时不注册图表工具（agent 降级纯 markdown）。"""
    assert create_chart_agent_tools(None) == []


def test_renders_line_chart_and_uploads_png() -> None:
    storage = _storage()
    (tool,) = create_chart_agent_tools(storage)
    result = _invoke(
        tool,
        chart_type="line",
        title="广东螺纹钢 HRB400 信息价走势",
        x_labels=["2026-01", "2026-02", "2026-03"],
        series=[{"name": "广东", "values": [3800.0, 3820.0, 3850.0]}],
    )
    assert result["url"].startswith("https://fake-s3.test/charts/")
    key = result["key"]
    assert key.startswith("charts/") and key.endswith(".png")
    # 上传记录：PNG 字节、image/png 类型。
    uploaded_key, uploaded_type, size = storage.uploads[0]
    assert uploaded_key == key
    assert uploaded_type == "image/png"
    assert size > 0


def test_renders_grouped_bar_with_multiple_series() -> None:
    storage = _storage()
    (tool,) = create_chart_agent_tools(storage)
    result = _invoke(
        tool,
        chart_type="grouped_bar",
        title="三地水泥价格对比",
        x_labels=["2026-05", "2026-06"],
        series=[
            {"name": "广东", "values": [430.0, 435.0]},
            {"name": "江苏", "values": [420.0, 425.0]},
            {"name": "浙江", "values": [425.0, 430.0]},
        ],
    )
    assert result["key"].startswith("charts/")


def test_rejects_series_length_mismatch() -> None:
    storage = _storage()
    (tool,) = create_chart_agent_tools(storage)
    with pytest.raises(ValidationError) as error:
        _invoke(
            tool,
            chart_type="line",
            title="长度不一致",
            x_labels=["2026-01", "2026-02"],
            series=[{"name": "广东", "values": [3800.0]}],
        )
    assert "expected 2" in str(error.value)


def test_rejects_grouped_bar_with_single_series() -> None:
    storage = _storage()
    (tool,) = create_chart_agent_tools(storage)
    with pytest.raises(ValidationError) as error:
        _invoke(
            tool,
            chart_type="grouped_bar",
            title="单序列分组柱状图",
            x_labels=["2026-01"],
            series=[{"name": "广东", "values": [3800.0]}],
        )
    assert "at least 2 series" in str(error.value)


def test_storage_error_returned_as_dict_not_raised() -> None:
    """上传失败返回错误字典，不抛异常（模型可自行降级为表格）。"""

    class _FailingStorage(FakeObjectStorage):
        def put(self, data, *, key, content_type):
            raise AppError(ErrorCode.UPSTREAM_UNAVAILABLE, "storage unavailable")

    (tool,) = create_chart_agent_tools(_FailingStorage())
    result = _invoke(
        tool,
        chart_type="bar",
        title="失败场景",
        x_labels=["2026-01"],
        series=[{"name": "广东", "values": [3800.0]}],
    )
    assert result == {"error": {"code": "UPSTREAM_UNAVAILABLE", "message": "storage unavailable"}}
