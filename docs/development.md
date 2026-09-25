# Local Development

## 1. Configure environment

Copy `.env.example` to `.env` and change values only when required.

## 2. Start PostgreSQL

```bash
docker compose up -d postgres
```

## 3. Apply migrations

```bash
alembic upgrade head
```

## 4. Seed development data

```bash
python -m scripts.seed_db
```

## 5. Run the API

```bash
uvicorn app.main:app --reload
```

## 6. Run checks

```bash
ruff check .
pytest
```
