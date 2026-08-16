"""图表渲染 Agent 工具的已校验输入契约。"""

from typing import Literal

from pydantic import BaseModel, Field, model_validator

ChartType = Literal["line", "bar", "grouped_bar"]


class ChartSeries(BaseModel):
    """图表中的一条数据序列。"""

    name: str = Field(min_length=1, max_length=100)
    values: list[float] = Field(min_length=1)


class ChartRenderInput(BaseModel):
    """render_chart 接收的已校验输入。"""

    chart_type: ChartType
    title: str = Field(min_length=1, max_length=200)
    x_labels: list[str] = Field(min_length=1)
    series: list[ChartSeries] = Field(min_length=1)
    y_label: str = Field(default="价格（元）", min_length=1, max_length=100)

    @model_validator(mode="after")
    def check_series_lengths(self) -> "ChartRenderInput":
        """每个序列的长度必须与 x_labels 一致；grouped_bar 需多序列。"""
        expected = len(self.x_labels)
        for item in self.series:
            if len(item.values) != expected:
                raise ValueError(
                    f"series '{item.name}' has {len(item.values)} values, "
                    f"expected {expected} to match x_labels",
                )
        if self.chart_type == "grouped_bar" and len(self.series) < 2:
            raise ValueError("grouped_bar requires at least 2 series; use bar instead")
        return self
