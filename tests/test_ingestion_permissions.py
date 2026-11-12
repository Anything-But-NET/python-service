import pytest
from src.permissions import mock_ingest_document, check_access


def test_permission_metadata_retained():
    """AC 1: Permission metadata is retained with the document record."""
    raw_meta = {
        "acl": {
            "allowed_groups": ["Engineering", "HR", "taras.ivaniuk"]
        }
    }

    record = mock_ingest_document("doc_001", raw_meta)
    saved_acl = record["metadata"]["acl"]

    assert "Engineering" in saved_acl["allowed_groups"]
    assert "taras.ivaniuk" in saved_acl["allowed_groups"]


def test_missing_metadata_handled_explicitly():
    """AC 2: Missing or unknown permission metadata is handled explicitly (default deny)."""
    record = mock_ingest_document("doc_002", None)
    saved_acl = record["metadata"]["acl"]

    assert saved_acl["allowed_groups"] == []


def test_access_allowed_and_denied():
    """AC 3: Test access control for allowed and denied scenarios."""
    raw_meta = {
        "acl": {
            "allowed_groups": ["Engineering", "admin_user"]
        }
    }
    record = mock_ingest_document("doc_003", raw_meta)

    # Allowed scenarios
    assert check_access(record, user_id="admin_user", user_groups=["Sales"]) is True
    assert check_access(record, user_id="taras.ivaniuk", user_groups=["Engineering"]) is True

    # Denied scenarios
    assert check_access(record, user_id="vasyl.khrystiuk", user_groups=["Marketing"]) is False
    assert check_access(record, user_id="hacker", user_groups=[]) is False