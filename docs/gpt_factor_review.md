# GPT Factor Review

## Purpose
The factor review pass asks LLMs to critique the latest factor-based picks. It keeps the signal context and last prices, then returns verdicts (approve/watch/reject) with notes and confidence.

## Data Flow
- **Source**: `factors.daily_factors` (latest factor run)
- **Output**: `gpt.factor_reviews`

## Table
`gpt.factor_reviews`
- `provider`: openai, deepseek, gemini
- `session`: pre/post
- `run_time`: review run time
- `as_of`: factor date
- `payload`: merged pick + review payload
- `raw_text`, `error`, `created_at`

## API
- `GET /api/gpt/factor-reviews`
- `POST /api/gpt/factor-reviews/refresh`

## Launchd
`config/launchd/com.tetra.gpt-factor-review.plist` runs daily at **19:35 local time**.

## Manual Run
```bash
./.venv/bin/python scripts/run_gpt_factor_reviews.py
```
