"""数学表达式归一化与安全精确求值。

兼容常见手写/印刷形式: ``* × x · ✕ ＊ / ÷ ： ／ - − ＋`` 全角数字、
中文括号、分数、百分号、幂记号等;整数分数用 ``Fraction`` 精确计算,
求值基于 ``ast`` 白名单,禁止任意代码。
"""

from __future__ import annotations

import ast
import ipaddress
import re
from fractions import Fraction
from typing import Any

MAX_EXPR_LEN = 200
MAX_NODES = 60

# 数字间的 x/X 视为乘号
_MUL_X = re.compile(r"(?<=[\d%.])\s*[xX]\s*(?=[\d(])")

_SUBS: list[tuple[str, str]] = [
    ("×", "*"),
    ("✕", "*"),
    ("·", "*"),
    ("＊", "*"),
    ("∗", "*"),
    ("÷", "/"),
    ("／", "/"),
    ("：", "/"),
    ("−", "-"),
    ("－", "-"),
    ("＋", "+"),
    ("（", "("),
    ("）", ")"),
    ("［", "["),
    ("］", "]"),
    ("｛", "("),
    ("｝", ")"),
    ("【", "("),
    ("】", ")"),
    ("^", "**"),
    ("²", "**2"),
    ("³", "**3"),
    ("½", "(1/2)"),
    ("¼", "(1/4)"),
]

_WS = re.compile(r"\s+")

_ALLOWED_BIN = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.FloorDiv, ast.Mod)
_ALLOWED_UNARY = (ast.UAdd, ast.USub)


class CalcError(ValueError):
    """表达式无法解析或超出支持范围。"""


def normalize(raw: str) -> str:
    """把各种书写形式归一化为标准 ASCII 表达式。"""
    s = raw.strip()
    s = s.split("=")[0].strip() if "=" in s else s
    s = "".join(chr(ord(c) - 0xFEE0) if "０" <= c <= "９" else c for c in s)
    s = _MUL_X.sub("*", s)
    for a, b in _SUBS:
        s = s.replace(a, b)
    s = _WS.sub("", s)
    s = re.sub(r"(\d+(?:\.\d+)?)%", r"(\1/100)", s)
    return s


def _eval(node: ast.expr) -> Fraction | int | float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, _ALLOWED_UNARY):
        value = _eval(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(node.op, _ALLOWED_BIN):
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow):
            if abs(float(right)) > 64 or abs(float(left)) > 10**12:
                raise CalcError("exponent out of range")
            return left**right
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if float(right) == 0:
                raise CalcError("division by zero")
            if isinstance(left, float) or isinstance(right, float):
                return left / right
            return Fraction(int(left), int(right))
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
    raise CalcError(f"unsupported node: {type(node).__name__}")


def _validate(tree: ast.Expression) -> None:
    count = 0
    for node in ast.walk(tree):
        count += 1
        if count > MAX_NODES:
            raise CalcError("expression too complex")
        if isinstance(node, (ast.Name, ast.Call, ast.Attribute, ast.Subscript, ast.Lambda)):
            raise CalcError(f"forbidden node: {type(node).__name__}")


def evaluate(raw: str) -> Fraction | int | float:
    """归一化并求值;失败抛 :class:`CalcError`。"""
    expr = normalize(raw)
    if not expr or len(expr) > MAX_EXPR_LEN:
        raise CalcError("empty or too long")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as error:
        raise CalcError(f"syntax error: {error.msg}") from error
    _validate(tree)
    return _eval(tree)


def _to_jsonable(value: Fraction | int | float) -> Any:
    if isinstance(value, Fraction):
        if value.denominator == 1:
            return value.numerator
        return {"fraction": f"{value.numerator}/{value.denominator}", "decimal": round(float(value), 10)}
    return value


def calc_one(raw: str) -> dict[str, Any]:
    """计算单条表达式,返回 ``{raw, ok, value|error}``。"""
    try:
        value = evaluate(raw)
    except CalcError as error:
        return {"raw": raw, "ok": False, "error": str(error)}
    return {"raw": raw, "ok": True, "value": _to_jsonable(value)}


def numbers_match(a: Any, b: Any, tol: float = 1e-9) -> bool:
    """比较两个数值(或 Fraction/dict/分数字符串)是否相等。"""

    def to_float(x: Any) -> float | None:
        if isinstance(x, dict):
            x = x.get("decimal", x.get("fraction"))
        if isinstance(x, Fraction):
            return float(x)
        if isinstance(x, bool) or x is None:
            return None
        if isinstance(x, str):
            stripped = x.strip()
            if stripped.endswith("%"):
                stripped = "(" + stripped[:-1] + "/100)"
            try:
                return float(Fraction(stripped.replace("×", "*").replace("÷", "/")))
            except (ValueError, ZeroDivisionError):
                try:
                    return float(stripped)
                except ValueError:
                    return None
        try:
            return float(x)
        except (TypeError, ValueError):
            return None

    fa, fb = to_float(a), to_float(b)
    if fa is None or fb is None:
        return str(a) == str(b)
    return abs(fa - fb) <= tol * max(1.0, abs(fa), abs(fb))


def assert_public_http_url(url: str) -> None:
    """拒绝内网/环回地址,防止 fetch 节点被用于 SSRF。"""
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise CalcError(f"unsupported url: {url!r}")
    try:
        addr = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        return  # 域名交给 httpx 解析;生产在网关层收敛
    if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_reserved:
        raise CalcError("private network address not allowed")
