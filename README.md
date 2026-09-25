# AI Customer Support & Ticket Automation System

AI-powered customer support system using RAG, agent orchestration, business tools, conversation memory, ticketing, and human escalation.

## Current Phase

Phase 2 — Data & Application Foundation.

## Development Workflow

- `main` is the protected release branch.
- Create feature branches from `main`.
- Use Conventional Commit-style messages.
- Open a pull request for every change; do not push directly to `main`.
- Never commit secrets, local `.env` files, generated credentials, or production data.

## Planned Stack

- FastAPI
- PostgreSQL
- SQLAlchemy
- Alembic
- Pydantic Settings
- Pytest
- Ruff
- Docker Compose

## Phase 2 Goals

- Establish project structure and configuration.
- Run PostgreSQL locally through Docker Compose.
- Define application domain models.
- Add migrations with Alembic.
- Seed realistic development data.
- Implement repository/service boundaries.
- Add database-focused tests.

## Phase Documents

- [Phase 2 scope and acceptance criteria](docs/phase-2.md)
- [Local development](docs/development.md)
- [GitHub workflow](docs/git-workflow.md)
