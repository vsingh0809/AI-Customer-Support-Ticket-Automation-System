# NovaMart Knowledge Base

## Purpose

This directory contains the company-specific support knowledge used by the RAG pipeline.
The assignment does not name a real company, so **NovaMart** is the fictional company used for the implementation, demo data, and deterministic testing.

## Document contract

Every knowledge-base document contains YAML front matter with:

- `document_id`: stable identifier used in retrieval metadata
- `title`: human-readable document title
- `category`: business category
- `version`: content version
- `status`: publication status
- `effective_date`: date this policy/content becomes effective
- `source`: logical source of truth

The document body is Markdown and is the content that will later be extracted, chunked, embedded, and indexed.
