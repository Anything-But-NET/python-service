import unittest
from datetime import datetime

# --- 1. Логіка Ingestion (Імітація обробки прав) ---

def process_document_permissions(raw_metadata):
    """
    Обробляє метадані доступу. Якщо прав немає (Missing metadata), 
    діє політика Explicit Deny (повна заборона за замовчуванням).
    """
    # Дефолтні права (дуже суворі)
    permissions = {
        "is_public": False,
        "allowed_groups": [],
        "allowed_users": []
    }
    
    if not raw_metadata or "permissions" not in raw_metadata:
        return permissions # Missing or unknown -> Explicitly handled
        
    raw_perms = raw_metadata["permissions"]
    
    # Зберігаємо передані права (Permission metadata is retained)
    permissions["is_public"] = raw_perms.get("is_public", False)
    permissions["allowed_groups"] = raw_perms.get("allowed_groups", [])
    permissions["allowed_users"] = raw_perms.get("allowed_users", [])
    
    return permissions

def mock_ingest_document(doc_id, raw_metadata):
    """
    Імітує збереження документа в PostgreSQL разом з правами доступу.
    """
    processed_permissions = process_document_permissions(raw_metadata)
    
    record = {
        "document_id": doc_id,
        "status": "COMPLETED",
        "metadata": {
            "ingested_at": datetime.utcnow().isoformat() + "Z",
            "permissions": processed_permissions
        }
    }
    return record

def check_access(document_record, user_id, user_groups):
    """
    Перевіряє, чи має юзер доступ до документа.
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

# --- 2. Unit Tests (Перевірка Acceptance Criteria) ---

class TestPermissionMetadata(unittest.TestCase):
    
    def test_permission_metadata_retained(self):
        """AC 1: Permission metadata is retained with the document record."""
        raw_meta = {
            "permissions": {
                "is_public": False,
                "allowed_groups": ["Engineering", "HR"],
                "allowed_users": ["taras.ivaniuk"]
            }
        }
        
        record = mock_ingest_document("doc_001", raw_meta)
        saved_perms = record["metadata"]["permissions"]
        
        self.assertFalse(saved_perms["is_public"])
        self.assertIn("Engineering", saved_perms["allowed_groups"])
        self.assertIn("taras.ivaniuk", saved_perms["allowed_users"])

    def test_missing_metadata_handled_explicitly(self):
        """AC 2: Missing or unknown permission metadata is handled explicitly."""
        # Передаємо порожні метадані (або None)
        record = mock_ingest_document("doc_002", None)
        saved_perms = record["metadata"]["permissions"]
        
        # Має заблокувати все за замовчуванням
        self.assertFalse(saved_perms["is_public"])
        self.assertEqual(saved_perms["allowed_groups"], [])
        self.assertEqual(saved_perms["allowed_users"], [])

    def test_access_allowed_and_denied(self):
        """AC 3: Tests cover allowed and denied examples."""
        raw_meta = {
            "permissions": {
                "is_public": False,
                "allowed_groups": ["Engineering"],
                "allowed_users": ["admin_user"]
            }
        }
        record = mock_ingest_document("doc_003", raw_meta)
        
        # Allowed examples (Дозволено)
        self.assertTrue(check_access(record, user_id="admin_user", user_groups=["Sales"])) # Пускає по user_id
        self.assertTrue(check_access(record, user_id="taras.ivaniuk", user_groups=["Engineering"])) # Пускає по групі
        
        # Denied examples (Заборонено)
        self.assertFalse(check_access(record, user_id="vasyl.khrystiuk", user_groups=["Marketing"])) # Немає в списках
        self.assertFalse(check_access(record, user_id="hacker", user_groups=[])) # Невідомий юзер

if __name__ == "__main__":
    unittest.main(verbosity=2)