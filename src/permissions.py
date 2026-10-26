from datetime import datetime
from typing import Any, Dict, List, Optional


def process_document_permissions(raw_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Process document permission metadata.
    If permissions are missing or unknown, apply explicit default-deny policy.
    """
    permissions = {
        "is_public": False,
        "allowed_groups": [],
        "allowed_users": []
    }

    if not raw_metadata or "permissions" not in raw_metadata:
        return permissions

    raw_perms = raw_metadata["permissions"]
    permissions["is_public"] = raw_perms.get("is_public", False)
    permissions["allowed_groups"] = raw_perms.get("allowed_groups", [])
    permissions["allowed_users"] = raw_perms.get("allowed_users", [])

    return permissions


def mock_ingest_document(doc_id: str, raw_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Simulate document ingestion and storage in PostgreSQL with permission metadata.
    """
    processed_permissions = process_document_permissions(raw_metadata)

    return {
        "document_id": doc_id,
        "status": "COMPLETED",
        "metadata": {
            "ingested_at": datetime.utcnow().isoformat() + "Z",
            "permissions": processed_permissions
        }
    }


def check_access(document_record: Dict[str, Any], user_id: str, user_groups: List[str]) -> bool:
    """
    Verify whether a user has access to the document based on permission metadata.
    """
    perms = document_record["metadata"]["permissions"]

    if perms["is_public"]:
        return True
    if user_id in perms["allowed_users"]:
        return True
    for group in user_groups:
        if group in perms["allowed_groups"]:
            return True

    return False