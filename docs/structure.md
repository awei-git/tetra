# Tetra Legacy Structures + New Package Plan

## tetra-legacy (external repo)

Top-level structure (high level):
- backend/: backend FastAPI app + models + migrations
- frontend/: React app + build assets
- src/: core pipelines, strategies, simulators, API routers, services
- tests/: pipeline, simulator, ML tests
- data/: raw + scenario + metrics artifacts
- output/: pipeline outputs
- alembic/: schema migrations
- docker/: infra (Timescale, etc.)
- bin/: pipeline/service launchers
- config/: settings + launchd
- scripts/: utilities
- logs/, models/, old_docs/

Notes:
- It is a full-stack monolith with overlapping API/pipeline concerns.
- There is both backend/ and src/ API logic, which can create duplication.
- Large data/output trees live inside the repo, which is convenient but heavy.

## tetra-legacy-2 (snapshot moved into this repo)

Top-level structure (high level):
- src/: ingestion + db schema + definitions
- scripts/: ingestion CLI
- alembic/: migrations
- config/: settings
- docker/: infra
- docs/: planning notes

Notes:
- This snapshot is slimmer and ingestion-focused.
- It is a good baseline for rebuilding the pipeline layer without frontend/backend baggage.

## Structure sanity check

- The legacy repo makes sense for a mature, all-in-one platform, but has a few risky overlaps:
  - API responsibilities appear split across backend/ and src/.
  - Data artifacts sit in-repo, which can bloat version control over time.
  - Pipeline execution is spread across bin/ scripts and Python entrypoints.
- The tetra-legacy-2 snapshot is much cleaner and suitable for an ingestion-first rebuild.

## New full package implementation plan (current repo)

Phase 1: Foundation
- Repo layout (src/, scripts/, config/, docker/, frontend/)
- Minimal packaging and dependencies (pyproject.toml)
- Secrets template (config/secrets.example.yml)

Phase 2: Database
- Local Postgres service via docker-compose
- Init SQL for schemas (market/event/economic/news)
- Table creation script (scripts/init_db.py)

Phase 3: Data ingestion
- Shared ingestion utilities in src/utils/ingestion
- Per-type pipelines in src/pipelines/data
- Runner to execute all pipelines (src/pipelines/data/runner.py)
- Daily scheduler (scripts/schedule_daily.py)

Phase 4: API + UI
- FastAPI app for status + trigger ingestion (src/api/app.py)
- Frontend console to view counts/latest + refresh (frontend/*)

Phase 5: Ops + quality (next)
- Add structured logging + retry/backoff telemetry
- Data quality checks (freshness, row counts, missing symbols)
- Persist pipeline runs in a small metadata table
- Add a lightweight health endpoint for DB + provider checks

Phase 6: Testing + deployment (next)
- Smoke tests for DB connectivity and schema availability
- Integration tests with provider keys in CI/local (optional)
- Scheduler deployment option (systemd/cron) + containerization

Status
- Phases 1-4 are implemented in this repo.
- Phases 5-6 are the next engineering steps.
