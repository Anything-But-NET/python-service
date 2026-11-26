import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from python_service.services.grounded_answer_service import GroundedAnswerService
from python_service.config import Settings

@pytest.mark.asyncio
async def test_get_current_prompt_config_caches_prompt():
    # Arrange
    settings = Settings()
    dummy_llm = MagicMock()  # No real LLM needed
    db = MagicMock()
    # Simulate empty DB so that get_or_create_prompt_version is called
    db.get_latest_prompt_version = AsyncMock(return_value=None)
    expected_id = "test-id-1234"
    db.get_or_create_prompt_version = AsyncMock(
        return_value=SimpleNamespace(
            id=expected_id,
            prompt_text="custom test prompt",
            metadata={"model": settings.llm_model},
        )
    )
    vectorizer = MagicMock()
    service = GroundedAnswerService(db, vectorizer, dummy_llm, settings)

    # Act
    prompt_id, prompt_text, metadata = await service._get_current_prompt_config()

    # Assert
    assert prompt_id == expected_id
    assert prompt_text == "custom test prompt"
    assert metadata == {"model": settings.llm_model}
    # Verify cache fields are set
    assert service._cached_prompt_id == expected_id
    assert service._cached_prompt == "custom test prompt"
    assert service._cached_metadata == {"model": settings.llm_model}
