"""不依赖应用配置的共享 pytest fixture。"""

import pytest
from fakes import FakeSummaryModel


@pytest.fixture
def fake_summary_model() -> FakeSummaryModel:
    """返回结果确定的结构化输出模型替身。

    返回值:
        不访问网络的共享摘要模型替身。
    """
    return FakeSummaryModel()
