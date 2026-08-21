"""Web 工具调用的全任务级预算（contextvar 隔离，防子代理无限重试）。

Deep Agent 的子代理（如 researcher）在 web 渠道失败时可能陷入
"换站点/换关键词重试"的循环，把整个任务的超时预算耗尽。prompt 级
约束（"最多重试 2 次"）不可靠，此处提供硬限制：

- TaskService 在每个任务开始时 ``set_web_call_budget(limit)``；
- web_search/web_fetch 每次执行前 ``consume_web_call()`` 扣减；
- 超限工具返回错误字典（而非抛异常），提示模型停止查询并基于
  已有数据收敛。

预算放在 contextvar 中：asyncio 请求天然按 task 隔离，子代理嵌套
子图与父图共享同一个任务上下文，因此同一次任务内所有 web 工具
调用共享同一份预算，跨请求互不影响。工具必须以 async 形式执行
（在事件循环线程内读写 contextvar）；同步网关客户端调用通过
线程池下发，不影响预算计数。

为什么不用官方 ToolCallLimitMiddleware 的 run_limit 做这件事：
run 语义是"每次子代理 invoke 重置"，主 agent 每派一次 task 工具
子代理就重新满额，无法表达"全任务共享"的硬顶；contextvar 挂在
任务上下文上，天然跨多次子代理调用共享，语义正确。
"""

from contextvars import ContextVar, Token
from typing import Any

# 当前任务请求的 web 调用预算；值为可变 dict {"used": int, "limit": int}。
_web_call_budget: ContextVar[dict[str, int] | None] = ContextVar("web_call_budget", default=None)


def set_web_call_budget(limit: int) -> Token[dict[str, int] | None]:
    """为当前任务初始化调用预算，返回用于恢复的 token。

    参数:
        limit: 本任务允许的 web_search + web_fetch 合计次数上限。

    返回值:
        传给 reset_web_call_budget 的 contextvar token。
    """
    return _web_call_budget.set({"used": 0, "limit": limit})


def reset_web_call_budget(token: Token[dict[str, int] | None]) -> None:
    """任务结束后恢复预算上下文，避免泄漏到下一个请求。"""
    _web_call_budget.reset(token)


def consume_web_call() -> bool:
    """扣减一次调用预算。

    返回值:
        允许继续调用时 True；已达上限时返回 False。预算未初始化
        （如 MCP 面或单元测试直接调用）时不做限制。
    """
    budget = _web_call_budget.get()
    if budget is None:
        return True
    budget["used"] += 1
    return budget["used"] <= budget["limit"]


def current_web_call_budget() -> dict[str, int] | None:
    """返回当前任务的预算快照；未初始化时为 None（供测试与诊断）。"""
    return _web_call_budget.get()


def budget_exceeded_payload() -> dict[str, Any]:
    """返回预算超限的错误字典（返回给模型，促使其停止查询并收敛）。"""
    budget = _web_call_budget.get()
    limit = budget["limit"] if budget is not None else 0
    return {
        "error": {
            "code": "QUERY_BUDGET_EXCEEDED",
            "message": (
                f"Web 查询次数已达上限（{limit} 次）。"
                "请停止搜索与抓取，基于已获取的数据生成结果；"
                '缺失的数据如实标注"未获取到"。'
            ),
        }
    }
