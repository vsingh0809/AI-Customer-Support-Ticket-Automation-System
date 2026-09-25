import json
from pathlib import Path

KB_DIR = Path(__file__).resolve().parents[2] / "knowledge_base"


def test_manifest_points_to_all_kb_documents() -> None:
    manifest = json.loads((KB_DIR / "manifest.json").read_text(encoding="utf-8"))
    documents = manifest["documents"]

    assert len(documents) == 8

    for item in documents:
        path = KB_DIR / item["path"]
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        assert text.startswith("---\n")
        assert f"document_id: {item['document_id']}" in text
        assert f"category: {item['category']}" in text


def test_required_business_topics_are_present() -> None:
    manifest = json.loads((KB_DIR / "manifest.json").read_text(encoding="utf-8"))
    categories = {item["category"] for item in manifest["documents"]}

    assert {"faq", "product", "refunds", "cancellations", "shipping", "payments", "account", "support-guidelines"} <= categories
