# GitHub Workflow

## Branch model

`main` is the stable branch. Work is done on short-lived branches:

```text
main
 ├── feat/<name>
 ├── fix/<name>
 ├── refactor/<name>
 ├── test/<name>
 └── docs/<name>
```

## Commit rules

Use Conventional Commit-style messages and keep each commit focused on one logical change.

Examples:

```text
feat: add order repository
fix: handle missing order
refactor: separate ticket service from repository
test: add payment service coverage
 docs: document local database setup
```

## Pull request rules

- Open a PR for every branch.
- Explain the problem, implementation, and validation.
- Keep PRs small enough to review.
- Do not merge with failing CI.
- Review migrations before merging database changes.

## Local checks before pushing

```bash
ruff check .
pytest
```

For database changes:

```bash
alembic upgrade head
```

## Secrets

Never commit `.env`, API keys, credentials, tokens, customer data, or production database dumps.
