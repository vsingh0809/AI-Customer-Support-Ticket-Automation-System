import json
from pathlib import Path

import pytest

from app.ai.rag.loader import (
    KnowledgeBaseDocumentError,
    load_document,
    load_documents,
)

KB_DIR = Path(__file__).resolve().parents[2] / "knowledge_base"


def test_load_document_parses_metadata_and_strips_front_matter() -> None:
    document = load_document(KB_DIR / "refund_policy.md")

    assert document.metadata.document_id == "refund-policy"
    assert document.metadata.category == "refunds"
    assert document.metadata.version == "1.0"
    assert document.content.startswith("# Refund Policy")
    assert "document_id:" not in document.content
    assert "\r\n" not in document.content


def test_load_documents_is_deterministic() -> None:
    documents = load_documents(KB_DIR)

    manifest = json.loads((KB_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert [doc.source_path.name for doc in documents] == sorted(
        item["path"] for item in manifest["documents"]
    )
    assert len(documents) == 8


def test_loader_normalizes_excess_blank_lines(tmp_path: Path) -> None:
    document = tmp_path / "sample.md"
    document.write_text(
        "---\n"
        "document_id: sample\n"
        "title: Sample\n"
        "category: faq\n"
        "version: '1.0'\n"
        "status: active\n"
        "effective_date: 2026-01-01\n"
        "source: test\n"
        "---\n"
        "# Title\r\n\r\n\r\nBody   \r\n",
        encoding="utf-8",
    )

    loaded = load_document(document)

    assert loaded.content == "# Title\n\nBody"


def test_loader_rejects_missing_required_metadata(tmp_path: Path) -> None:
    document = tmp_path / "invalid.md"
    document.write_text(
        "---\n"
        "document_id: invalid\n"
        "title: Invalid\n"
        "---\n"
        "# Content\n",
        encoding="utf-8",
    )

    with pytest.raises(KnowledgeBaseDocumentError, match="Missing metadata fields"):
        load_document(document)


def test_loader_rejects_invalid_yaml(tmp_path: Path) -> None:
    document = tmp_path / "invalid-yaml.md"
    document.write_text(
        "---\n"
        "document_id: [broken\n"
        "---\n"
        "# Content\n",
        encoding="utf-8",
    )

    with pytest.raises(KnowledgeBaseDocumentError, match="Invalid YAML"):
        load_document(document)
