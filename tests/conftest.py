"""Shared pytest fixtures that do not depend on application configuration."""

import pytest

from fakes import FakeSummaryModel


@pytest.fixture
def fake_summary_model() -> FakeSummaryModel:
    """Return a deterministic structured-output model fake."""
    return FakeSummaryModel()
