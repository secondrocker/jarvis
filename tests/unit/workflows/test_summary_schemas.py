import pytest
from pydantic import ValidationError

from agent_app.workflows.summary.schemas import SummaryInput


def test_summary_input_defaults_and_bounds() -> None:
    value = SummaryInput(text="A useful source text", language=None)

    assert value.max_words == 200

    with pytest.raises(ValidationError):
        SummaryInput(text="A useful source text", max_words=49)

    with pytest.raises(ValidationError):
        SummaryInput(text="A useful source text", max_words=1001)


def test_summary_input_strips_text_and_rejects_whitespace_only_text() -> None:
    value = SummaryInput(text="  A useful source text  ")

    assert value.text == "A useful source text"

    with pytest.raises(ValidationError):
        SummaryInput(text=" \t\n ")
