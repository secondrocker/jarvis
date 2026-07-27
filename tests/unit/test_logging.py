import json

import structlog

from agent_app.logging import configure_logging
from agent_app.schemas.tasks import TaskRequest


class SensitiveMappingKey:
    def __str__(self) -> str:
        return "sensitive mapping key"


def test_configure_logging_emits_json_with_timestamp_and_level(capsys) -> None:
    configure_logging("INFO")

    structlog.get_logger("contract-test").info("contract_event")

    record = json.loads(capsys.readouterr().out)
    assert record["event"] == "contract_event"
    assert record["level"] == "info"
    assert record["timestamp"]


def test_configure_logging_redacts_request_models_without_losing_json_metadata(capsys) -> None:
    configure_logging("INFO")

    structlog.get_logger("contract-test").info(
        "task_received",
        request=TaskRequest(message="sensitive input"),
        metadata={"task_id": "task-1", "attempt": 1},
    )

    output = capsys.readouterr().out
    record = json.loads(output)
    assert "sensitive input" not in output
    assert record["request"] == "[REDACTED_NON_JSON]"
    assert record["metadata"] == {"task_id": "task-1", "attempt": 1}


def test_configure_logging_redacts_metadata_with_non_string_mapping_keys(capsys) -> None:
    configure_logging("INFO")

    structlog.get_logger("contract-test").info(
        "task_received",
        metadata={SensitiveMappingKey(): "safe value"},
    )

    output = capsys.readouterr().out
    record = json.loads(output)
    assert "sensitive mapping key" not in output
    assert record["metadata"] == "[REDACTED_NON_JSON]"
