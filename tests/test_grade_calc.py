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
    numeric_value,
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


def test_calc_one_ignores_trailing_unit():
    # raw 原样保留单位,取第一个 "=" 前求值
    assert calc_one("5-1=4（份）")["value"] == 4
    assert calc_one("（156-12）÷（11+5）=144÷16=9（名）")["value"] == 9
    assert calc_one("4×5+4=24（名）")["value"] == 24


def test_numbers_match_with_units():
    # 学生结果带单位(全/半角括号包住或直接缀后缀)时按纯数值比对
    assert numbers_match(4, "4（份）")
    assert numbers_match(24, "24（名）")
    assert numbers_match(57, "57名")
    assert numbers_match(9, "共9（名）")
    assert numbers_match(0.5, "0.5小时")
    assert numbers_match(Fraction(5, 4), "5/4（份）")
    assert numbers_match("24名", 24)
    assert numbers_match(0.5, "50%")
    # 数值不同仍判不符
    assert not numbers_match(24, "25（名）")
    # 非单位后缀不剥离,交由上层按“无法复算”处理
    assert not numbers_match(24, "24个苹果")
    assert not numbers_match(24, "24（答案）")


def test_numeric_value_unit_awareness():
    assert numeric_value("4（份）") == 4.0
    assert numeric_value("２４名") == 24.0
    assert numeric_value("57（名）") == 57.0
    assert numeric_value("四份") is None
    assert numeric_value("第1步") is None


def test_public_url_guard():
    with pytest.raises(CalcError):
        assert_public_http_url("http://127.0.0.1/x.png")
    with pytest.raises(CalcError):
        assert_public_http_url("http://192.168.1.4/x.png")
    assert_public_http_url("https://example.com/x.png")
