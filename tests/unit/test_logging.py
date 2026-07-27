import json

import structlog

from agent_app.logging import configure_logging


def test_configure_logging_emits_json_with_timestamp_and_level(capsys) -> None:
    configure_logging("INFO")

    structlog.get_logger("contract-test").info("contract_event")

    record = json.loads(capsys.readouterr().out)
    assert record["event"] == "contract_event"
    assert record["level"] == "info"
    assert record["timestamp"]
