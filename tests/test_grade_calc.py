"""grade_homework.calc 单元测试。"""

from fractions import Fraction

import pytest

from agent_app.workflows.grade_homework.calc import (
    CalcError,
    assert_public_http_url,
    calc_one,
    evaluate,
    normalize,
    numbers_match,
)


def test_normalize_symbols():
    assert normalize("25 × 4") == "25*4"
    assert normalize("25 x 4") == "25*4"
    assert normalize("10 ÷ 2") == "10/2"
    assert normalize("（１＋２）＊4") == "(1+2)*4"
    assert normalize("3 − 1") == "3-1"
    assert normalize("50%") == "(50/100)"
    assert normalize("2^3") == "2**3"
    assert normalize("2²") == "2**2"
    assert normalize("6÷2=3") == "6/2"


def test_evaluate_exact():
    assert evaluate("25 × 4") == 100
    assert evaluate("3/4+1/2") == Fraction(5, 4)
    assert evaluate("2³") == 8
    assert evaluate("（１＋２）＊4") == 12
    assert evaluate("-5 + 3") == -2
    assert evaluate("2^10") == 1024


def test_errors():
    for bad in ["__import__('os')", "abc", "1/0", "x+1", "a*b()"]:
        with pytest.raises(CalcError):
            evaluate(bad)


def test_calc_one():
    assert calc_one("25 × 4") == {"raw": "25 × 4", "ok": True, "value": 100}
    result = calc_one("1÷3")
    assert result["ok"] and result["value"]["fraction"] == "1/3"


def test_numbers_match():
    assert numbers_match(100, "100")
    assert numbers_match(Fraction(5, 4), "5/4")
    assert numbers_match({"decimal": 1.25}, "5/4")
    assert not numbers_match(100, 99)


def test_public_url_guard():
    with pytest.raises(CalcError):
        assert_public_http_url("http://127.0.0.1/x.png")
    with pytest.raises(CalcError):
        assert_public_http_url("http://192.168.1.4/x.png")
    assert_public_http_url("https://example.com/x.png")
