# GPT Challenge Pass

## Purpose
The challenge pass critiques the latest GPT recommendations and proposes improved ideas for risk-adjusted portfolio fit. It runs daily and stores results for review.

## Data Flow
- **Source**: `gpt.recommendations` (latest successful run per provider)
- **Output**: `gpt.recommendation_challenges`

## Tables
`gpt.recommendation_challenges`
- `provider`: challenger model (openai, deepseek)
- `session`: pre/post
- `run_time`: challenge run time
- `source_provider`: provider used for the original recommendations
- `source_run_time`: timestamp of the source recommendations
- `source_payload`: JSON snapshot of the original recommendations
- `payload`: JSON challenge output (categories + ideas)
- `raw_text`, `error`, `created_at`

## API
- `GET /api/gpt/challenges`
- `POST /api/gpt/challenges/refresh`

## Launchd
`config/launchd/com.tetra.gpt-challenge.plist` runs daily at **18:30 local time**.

## Manual Run
```bash
./.venv/bin/python scripts/run_gpt_challenge.py
```
