# Contributing

## Branching

Create one focused branch from `main` per change:

- `feat/<short-name>`
- `fix/<short-name>`
- `refactor/<short-name>`
- `docs/<short-name>`
- `test/<short-name>`

## Commit messages

Use Conventional Commit style:

- `feat: add order repository`
- `fix: handle missing payment`
- `test: add ticket service tests`
- `docs: update architecture notes`

Keep commits small and logically coherent.

## Pull requests

Every change goes through a pull request. The author must explain the change, validation performed, and any migration or operational impact.

## Secrets

Never commit `.env`, API keys, database credentials, tokens, or customer data. Use `.env.example` for non-secret configuration shape only.
