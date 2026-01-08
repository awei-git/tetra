# Frontend

## FastAPI console
The console is served by `src/api/app.py` and exposes:
- `/`: UI
- `/api/status`: counts + latest timestamps
- `/api/ingest`: trigger pipelines

Run from `tetra/`:

```
uvicorn src.api.app:app --reload
```

Open `http://localhost:8000`.

## Frontend assets
Static assets live in:
- `tetra/frontend/index.html`
- `tetra/frontend/styles.css`
- `tetra/frontend/app.js`

The API mounts these at `/static`.
