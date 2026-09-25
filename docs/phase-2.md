# Phase 2 — Data & Application Foundation

## Status

Implementation prepared. PostgreSQL execution remains to be validated on a machine with Docker/PostgreSQL available.

## Locked objectives

1. Establish a maintainable Python application structure.
2. Run PostgreSQL locally through Docker Compose.
3. Use SQLAlchemy for the application data model.
4. Use Alembic for schema versioning and migrations.
5. Separate repository/database access from business services.
6. Seed deterministic development data for agent/tool testing.
7. Keep configuration and secrets outside source control.
8. Establish GitHub workflow before feature development expands.

## Domain model

```text
Customer
  ├── Orders
  │     ├── OrderItems
  │     └── Payments
  ├── Conversations
  │     ├── Messages
  │     └── Tickets
  └── Tickets
```

## Tables

| Table | Purpose |
|---|---|
| `customers` | Customer identity and contact data |
| `orders` | Business order state |
| `order_items` | Products belonging to orders |
| `payments` | Payment transactions and status |
| `conversations` | Persistent chat sessions |
| `messages` | Individual user/assistant messages |
| `tickets` | Support cases and escalation records |

## Layering rule

```text
API / Agent Tool
       ↓
Service
       ↓
Repository
       ↓
SQLAlchemy
       ↓
PostgreSQL
```

Business logic must not be embedded inside API routes or SQL queries inside AI tool implementations.

## Migration rule

All schema changes go through Alembic. Do not use `Base.metadata.create_all()` as the application migration strategy.

## GitHub rule

- `main` is the stable branch.
- Work happens on short-lived feature/fix/refactor/test/docs branches.
- Every change is reviewed through a pull request.
- CI must pass before merge.
- Conventional Commit-style messages are used.
- Secrets and customer data never enter Git history.

## Acceptance criteria

- [x] Repository structure exists.
- [x] Configuration is environment driven.
- [x] PostgreSQL Compose configuration exists.
- [x] Core domain models exist.
- [x] Initial Alembic migration exists.
- [x] Repository/service separation exists for core business operations.
- [x] Deterministic seed script exists.
- [x] GitHub workflow files exist.
- [x] Unit tests pass in the current environment.
- [ ] Run `docker compose up -d postgres`.
- [ ] Run `alembic upgrade head` against PostgreSQL.
- [ ] Run the seed script against PostgreSQL.
- [ ] Verify repository/service integration against PostgreSQL.
