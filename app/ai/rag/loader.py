"""Load and normalize Markdown knowledge-base documents."""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

_FRONT_MATTER_PATTERN = re.compile(
    r"\A---\r?\n(?P<meta>.*?)\r?\n---\r?\n(?P<body>.*)\Z",
    re.DOTALL,
)
_REQUIRED_METADATA_FIELDS = {
    "document_id",
    "title",
    "category",
    "version",
    "status",
    "effective_date",
    "source",
}


class KnowledgeBaseDocumentError(ValueError):
    """Raised when a knowledge-base document is invalid."""


class KnowledgeDocumentMetadata(BaseModel):
    """Validated metadata stored in a KB document's YAML front matter."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    document_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    category: str = Field(min_length=1)
    version: str = Field(min_length=1)
    status: str = Field(min_length=1)
    effective_date: date
    source: str = Field(min_length=1)


class KnowledgeDocument(BaseModel):
    """Normalized document ready for downstream RAG processing."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    metadata: KnowledgeDocumentMetadata
    content: str = Field(min_length=1)
    source_path: Path


def load_document(path: Path | str) -> KnowledgeDocument:
    """Load one Markdown KB document, validate metadata, and normalize its body."""
    source_path = Path(path)
    if not source_path.is_file():
        raise KnowledgeBaseDocumentError(f"Knowledge document not found: {source_path}")

    if source_path.suffix.lower() != ".md":
        raise KnowledgeBaseDocumentError(f"Knowledge document must be Markdown: {source_path}")

    raw_text = source_path.read_text(encoding="utf-8")
    metadata, body = _parse_front_matter(raw_text, source_path)
    normalized_body = _normalize_markdown(body)

    if not normalized_body:
        raise KnowledgeBaseDocumentError(f"Knowledge document has empty content: {source_path}")

    return KnowledgeDocument(
        metadata=metadata,
        content=normalized_body,
        source_path=source_path,
    )


def load_documents(directory: Path | str) -> list[KnowledgeDocument]:
    """Load the corpus documents listed by the KB manifest in deterministic order."""
    kb_directory = Path(directory)
    if not kb_directory.is_dir():
        raise KnowledgeBaseDocumentError(f"Knowledge-base directory not found: {kb_directory}")

    manifest_path = kb_directory / "manifest.json"
    if not manifest_path.is_file():
        raise KnowledgeBaseDocumentError(f"Knowledge-base manifest not found: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest["documents"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise KnowledgeBaseDocumentError(
            f"Invalid knowledge-base manifest: {manifest_path}"
        ) from exc

    if not isinstance(entries, list) or not entries:
        raise KnowledgeBaseDocumentError(
            f"Knowledge-base manifest contains no documents: {manifest_path}"
        )

    paths = []
    for entry in entries:
        if not isinstance(entry, dict) or "path" not in entry:
            raise KnowledgeBaseDocumentError(
                f"Invalid document entry in knowledge-base manifest: {manifest_path}"
            )
        paths.append(kb_directory / entry["path"])

    return [load_document(path) for path in sorted(paths, key=lambda item: item.name)]


def _parse_front_matter(text: str, path: Path) -> tuple[KnowledgeDocumentMetadata, str]:
    match = _FRONT_MATTER_PATTERN.match(text)
    if not match:
        raise KnowledgeBaseDocumentError(
            f"Document must start with YAML front matter delimited by '---': {path}"
        )

    try:
        raw_metadata = yaml.safe_load(match.group("meta"))
    except yaml.YAMLError as exc:
        raise KnowledgeBaseDocumentError(f"Invalid YAML front matter in {path}: {exc}") from exc

    if not isinstance(raw_metadata, dict):
        raise KnowledgeBaseDocumentError(f"YAML front matter must be an object: {path}")

    missing = _REQUIRED_METADATA_FIELDS - raw_metadata.keys()
    if missing:
        fields = ", ".join(sorted(missing))
        raise KnowledgeBaseDocumentError(f"Missing metadata fields ({fields}) in {path}")

    try:
        metadata = KnowledgeDocumentMetadata.model_validate(raw_metadata)
    except ValidationError as exc:
        raise KnowledgeBaseDocumentError(f"Invalid document metadata in {path}: {exc}") from exc

    return metadata, match.group("body")


def _normalize_markdown(content: str) -> str:
    """Normalize whitespace while preserving Markdown structure and semantics."""
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    normalized = "\n".join(line.rstrip() for line in normalized.split("\n"))
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()
