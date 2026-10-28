from datetime import datetime
from typing import Any, Dict, List, Optional


def process_document_permissions(raw_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Process document permission metadata according to the SCRUM-87 contract.
    If metadata is missing, apply explicit default-deny policy.
    """
    acl = {
        "allowed_groups": []
    }

    if not raw_metadata or "acl" not in raw_metadata:
        return acl

    raw_acl = raw_metadata["acl"]
    acl["allowed_groups"] = raw_acl.get("allowed_groups", [])

    return acl


def mock_ingest_document(doc_id: str, raw_metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Simulate document ingestion and storage in PostgreSQL with acl metadata.
    """
    processed_acl = process_document_permissions(raw_metadata)

    return {
        "document_id": doc_id,
        "status": "COMPLETED",
        "metadata": {
            "ingested_at": datetime.utcnow().isoformat() + "Z",
            "acl": processed_acl
        }
    }


def check_access(document_record: Dict[str, Any], user_id: str, user_groups: List[str]) -> bool:
    """
    Verify whether a user has access to the document based on the acl object.
    Both user_id and groups are checked against allowed_groups.
    """
    allowed = document_record["metadata"]["acl"]["allowed_groups"]

    if user_id in allowed:
        return True
    for group in user_groups:
        if group in allowed:
            return True

    return False