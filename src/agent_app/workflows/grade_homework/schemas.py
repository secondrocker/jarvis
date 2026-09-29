"""批改作业工作流的数据模型与状态。"""

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


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
    """工作流输入: 可下载的作业图片 URL。"""

    url: str


class GradeState(TypedDict, total=False):
    """在批改图中流转的状态。"""

    url: str
    image_b64: str
    result: dict[str, Any]
