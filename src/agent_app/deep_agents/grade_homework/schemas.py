"""批改作业 agent 的数据模型与对外契约。

``GradeResult`` 是小程序直接消费的稳定契约：agent 的最终回答必须给出
符合该契约的 JSON，服务端在返回前用它做校验、统计重算与复算纠错。
``DiagnosisResult`` / ``FeedbackReport`` 是职能子代理的结构化输出契约，
同时作为 ``GradeResult`` 的可选字段回传，保持"一个契约、两处复用"。
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# 错因大类（两级 taxonomy 的一级）。子类写在 ErrorDetail.error_type 里。
# 顺序与命名对齐教育领域常见错因分类（知识性/策略性/计算/审题/概念/习惯）。
ErrorCategory = Literal[
    "knowledge",  # 知识性：概念、公式、规则记错
    "concept",  # 概念混淆：形近字、相似公式、易混术语
    "method",  # 策略/方法性：解题思路或方法选择错误
    "calculation",  # 计算错误：单纯运算失误
    "careless",  # 粗心：抄错数字、漏写单位、跳步
    "reading",  # 审题不清：看错要求、漏条件
    "habit",  # 习惯与规范：书写、格式、不写答句
    "language",  # 语言表达：拼写、语法、用词
    "other",
]

# 单题判定；unreadable 表示题干/字迹无法确认，须人工复核。
Verdict = Literal["correct", "wrong", "partial", "unreadable"]

# 客观题可直接判定；开放性/主观题（作文、说理、观点类）AI 只出预批。
AnswerType = Literal["objective", "open"]

Confidence = Literal["high", "low"]


class ExpressionItem(BaseModel):
    """学生书写的一个算式（供后端计算工具复算）。"""

    raw: str
    claimed: str | None = None


class ErrorDetail(BaseModel):
    """错题的错因描述。"""

    category: ErrorCategory | None = None
    error_type: str | None = None
    expected: str | float | int | None = None
    correct_form: str | None = None
    reason: str | None = None


class QuestionItem(BaseModel):
    """单题批改结果。"""

    no: int
    question_text: str | None = None
    student_answer: str | None = None
    knowledge_point: str | None = None
    process_correct: bool | None = None
    result_correct: bool = True
    verdict: Verdict = "correct"
    answer_type: AnswerType = "objective"
    expressions: list[ExpressionItem] | None = None
    error_detail: ErrorDetail | None = None
    comment: str | None = None
    score: float | int | None = None
    full_score: float | int | None = None
    confidence: Confidence = "high"
    needs_review: bool = False
    review_reason: str | None = None


class DiagnosisItem(BaseModel):
    """错因诊断子代理给出的单题诊断。"""

    no: int
    category: ErrorCategory
    error_type: str | None = None
    knowledge_point: str | None = None
    reason: str | None = None
    advice: str | None = None


class DiagnosisResult(BaseModel):
    """错因诊断与学情汇总（error_analyst 子代理输出）。"""

    items: list[DiagnosisItem] = Field(default_factory=list)
    top_errors: list[str] = Field(default_factory=list)
    weak_points: list[str] = Field(default_factory=list)
    teaching_advice: str = ""


class FeedbackReport(BaseModel):
    """面向学生的评语（feedback_writer 子代理输出，Glow-Grow-Go 结构）。"""

    glow: list[str] = Field(default_factory=list)
    grow: list[str] = Field(default_factory=list)
    go: str = ""
    parent_note: str | None = None


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
    diagnosis: DiagnosisResult | None = None
    feedback: FeedbackReport | None = None
    # 合规字段：AI 只出预批，需教师确认后才作为最终评价（教育部 AI 应用指引）。
    needs_human_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    degraded: bool = False


class GradeInput(BaseModel):
    """批改输入: 同一份作业的一张或多张图片 URL(跨页拍摄时连续多图)。

    兼容旧契约: 传入单个 ``url`` 字符串时自动转为单元素列表。
    """

    urls: list[str] = Field(min_length=1, max_length=8)

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
