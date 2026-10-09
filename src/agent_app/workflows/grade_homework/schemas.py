"""批改作业工作流的数据模型与状态。"""

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field, model_validator


class ExpressionItem(BaseModel):
    """学生书写的一个算式(供后端计算工具复算)。"""

    raw: str
    claimed: str | None = None


class ErrorDetail(BaseModel):
    """错题的错因描述。"""

    error_type: str | None = None
    expected: str | float | int | None = None
    correct_form: str | None = None
    reason: str | None = None


class QuestionItem(BaseModel):
    """单题批改结果。"""

    no: int
    question_text: str | None = None
    student_answer: str | None = None
    process_correct: bool | None = None
    result_correct: bool = True
    verdict: Literal["correct", "wrong", "partial", "unreadable"] = "correct"
    expressions: list[ExpressionItem] | None = None
    error_detail: ErrorDetail | None = None
    comment: str | None = None
    score: float | int | None = None
    full_score: float | int | None = None
    confidence: Literal["high", "low"] = "high"


class GradeResult(BaseModel):
    """整份作业的批改结果契约(前端直接消费)。"""

    subject: Literal["math", "chinese", "english", "other"] = "other"
    subject_confidence: float | None = None
    total_questions: int | None = None
    correct_count: int | None = None
    wrong_count: int | None = None
    accuracy: float | None = None
    score: float | int | None = None
    full_score: float | int | None = None
    summary: str = ""
    questions: list[QuestionItem] = Field(default_factory=list)
    degraded: bool = False


class GradeInput(BaseModel):
    """工作流输入: 同一份作业的一张或多张图片 URL(跨页拍摄时连续多图)。

    兼容旧契约: 传入单个 ``url`` 字符串时自动转为单元素列表。
    """

    urls: list[str] = Field(min_length=1, max_length=4)

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_url(cls, data: Any) -> Any:
        """接受旧的 ``url: str`` 单图输入,以及裸列表。"""
        if isinstance(data, str):
            return {"urls": [data]}
        if isinstance(data, dict) and "urls" not in data and data.get("url"):
            data = dict(data)
            return {**data, "urls": [data.pop("url")]}
        return data


class GradeState(TypedDict, total=False):
    """在批改图中流转的状态。"""

    urls: list[str]
    images_b64: list[str]
    result: dict[str, Any]
