import uuid
import pytest
from python_service.db.service import _generate_prompt_id, _scrub_metadata

def test_generate_prompt_id_is_deterministic():
    """Checks that the same text and metadata yield the same UUID."""
    prompt = "You are a helpful assistant."
    metadata = {"temperature": 0.7, "model": "gpt-4"}

    id1 = _generate_prompt_id(prompt, metadata)
    id2 = _generate_prompt_id(prompt, metadata)

    assert id1 == id2
    assert isinstance(id1, uuid.UUID)

def test_generate_prompt_id_differs_on_content_change():
    """Verify that changing content generates a new UUID."""
    prompt1 = "You are an expert."
    prompt2 = "You are a helper."
    metadata = {"temperature": 0.7}

    assert _generate_prompt_id(prompt1, metadata) != _generate_prompt_id(prompt2, metadata)

def test_generate_prompt_id_differs_on_metadata_change():
    """Verify that a configuration change generates a new UUID."""
    prompt = "Fix this code."
    assert _generate_prompt_id(prompt, {"temp": 0.5}) != _generate_prompt_id(prompt, {"temp": 0.9})

def test_scrub_metadata_allows_safe_data():
    """Verify that secure data is passing through."""
    safe_meta = {"temperature": 0.7, "model": "gpt-4", "max_tokens": 100}
    assert _scrub_metadata(safe_meta) == safe_meta

def test_scrub_metadata_blocks_forbidden_keys():
    """Checks that keys with secrets are blocked."""
    with pytest.raises(ValueError, match="Detected potential secret in metadata key"):
        _scrub_metadata({"OPENAI_API_KEY": "some-value"})

    with pytest.raises(ValueError, match="Detected potential secret in metadata key"):
        _scrub_metadata({"db_password": "123"})

def test_scrub_metadata_blocks_secret_patterns():
    """Verify that token-like values are blocked even in secure keys."""
    with pytest.raises(ValueError, match="Detected potential API token in metadata value"):
        _scrub_metadata({"custom_string": "sk-123456789012345678901234567890123456789012345678"})
