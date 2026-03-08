<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://img.shields.io/badge/△-Tetra-white?style=for-the-badge&labelColor=333&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxMDAgMTAwIj48cG9seWdvbiBwb2ludHM9IjUwLDEwIDEwLDkwIDkwLDkwIiBmaWxsPSJ3aGl0ZSIgc3Ryb2tlPSJ3aGl0ZSIgc3Ryb2tlLXdpZHRoPSIyIi8+PC9zdmc+">
    <img src="https://img.shields.io/badge/△-Tetra-black?style=for-the-badge&labelColor=eee&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxMDAgMTAwIj48cG9seWdvbiBwb2ludHM9IjUwLDEwIDEwLDkwIDkwLDkwIiBmaWxsPSIjMzMzIiBzdHJva2U9IiMzMzMiIHN0cm9rZS13aWR0aD0iMiIvPjwvc3ZnPg==" alt="Tetra">
  </picture>
</p>

<h1 align="center">Tetra</h1>

<p align="center">
  <strong>LLM-powered market analysis system with novel signal extraction</strong>
</p>

<p align="center">
  <a href="#methods">Methods</a> &bull;
  <a href="#pipeline">Pipeline</a> &bull;
  <a href="#architecture">Architecture</a> &bull;
  <a href="#setup">Setup</a>
</p>

---

## Philosophy

**Don't do analysis anyone can get for free.**

Traditional technical analysis (RSI, SMA, momentum) and simple statistics are commodity — you can ask any LLM for that. Tetra does what others can't: **process 30K news articles, 10K prediction market contracts, 59 macro series, insider trading patterns, and analyst consensus shifts simultaneously** to extract novel signals.

The edge is not the model. The edge is the unique data combinations fed to the model.

---

## Methods

### 1. Narrative Regime Detection
Not sentiment scores. LLM analyzes **structural changes in news narratives** — when the market story shifts from "AI growth" to "antitrust risk", that's a leading indicator.

### 2. Polymarket Lead-Lag Scanner
Information propagates at different speeds across markets. Polymarket reacts in minutes, bonds in hours, equities in days. Tetra detects **information gaps** between prediction markets and traditional assets.

### 3. Insider Trading Signal
Not "3 insiders bought AAPL". Detects **pattern breaks** — non-routine timing, abnormal size, cluster buying by multiple insiders outside their historical patterns.

### 4. Earnings Network Cascade
Builds a company relationship graph (supplier → customer → peer). When TSMC guides strong AI demand, Tetra calculates the **implied probability shift** for NVDA's upcoming earnings.

### 5. Analyst Network Alpha
Tracks which analysts move first and which firms lead consensus shifts. **First-mover analysts** historically have 2-3x the signal value of the consensus.

### 6. LLM Adversarial Debate
Three LLMs with **genuine information asymmetry**:
- **Macro Analyst** (OpenAI): sees macro data, yield curves, credit spreads — no individual stocks
- **Micro Analyst** (DeepSeek): sees earnings, insider trades, fundamentals — no macro
- **Crowd Analyst** (Gemini): sees Polymarket, news narratives, sentiment — no fundamentals

They debate in 3 rounds. **Consensus trades** (all agree) are high-confidence. **Contrarian trades** (one analyst sees something others can't) are the real edge.

### 7. Forward Scenario Analysis
Not backward-looking stress tests. LLM generates **forward scenarios** using all current data — 3 base cases + 1 black swan, with specific portfolio impact calculations.

### 8. Meta-Signal Layer
LLM reasons about **which signals to trust** given the current regime. In risk-off, weight macro signals higher. In earnings season, weight micro signals higher.

---

## Pipeline

```
16:20 ET  Data Ingestion
          ├── Market OHLCV (205 symbols, Polygon)
          ├── News articles (301 sources, NewsAPI)
          ├── Economic data (59 FRED series)
          ├── Events (earnings, dividends, splits)
          ├── Polymarket (10K contracts)
          ├── Insider trades (Finnhub Form 4)
          └── Analyst consensus (Finnhub)

17:00 ET  Analysis Pipeline (9 stages)
          ├── 1. Narrative Fragmentation Index
          ├── 2. Polymarket Lead-Lag Scanner
          ├── 3. Insider Trading Signal
          ├── 4. Earnings Network Cascade
          ├── 5. Analyst Network Alpha
          ├── 6. LLM Adversarial Debate (3 rounds)
          ├── 7. Portfolio Mark-to-Market
          ├── 8. Forward Scenario Analysis
          └── 9. Meta-Signal Layer

18:00 ET  Report & Delivery
          ├── PDF report (WeasyPrint)
          ├── Email delivery
          ├── Mira bridge push (iPhone)
          └── Briefing artifact (markdown)
```

---

## Architecture

```
tetra/
├── src/
│   ├── analysis/          # Novel analysis methods
│   │   ├── narrative.py       # Method 1: Narrative regime detection
│   │   ├── polymarket.py      # Method 2: Polymarket lead-lag
│   │   ├── insider_signal.py  # Method 3: Insider pattern detection
│   │   ├── earnings_network.py # Method 4: Earnings cascade
│   │   ├── analyst_network.py # Method 5: Analyst network alpha
│   │   ├── debate.py          # Method 6: 3-LLM adversarial debate
│   │   ├── scenarios.py       # Method 7: Forward scenario analysis
│   │   └── meta_signal.py     # Method 8: Meta-signal layer
│   ├── portfolio/         # Position tracking & analytics
│   │   ├── manager.py         # Mark-to-market, recommendation tracker
│   │   └── track_record.py    # Performance analytics, self-critique
│   ├── report/            # Report generation
│   │   ├── generator.py       # Data assembly + LLM commentary
│   │   ├── delivery.py        # PDF generation + email
│   │   ├── llm_clients.py     # Multi-provider LLM abstraction
│   │   └── templates/         # Jinja2 HTML template
│   ├── mira/              # Mira agent integration
│   │   └── push.py            # Bridge push + feedback loop
│   ├── db/                # Database session + schema
│   └── utils/
│       ├── ingestion/         # Data pipeline (market, news, events)
│       ├── gpt/               # Legacy GPT analysis
│       ├── inference/         # Statistical inference
│       └── simulations/       # Monte Carlo, stress testing
├── scripts/               # Pipeline entry points
├── config/                # Configuration (secrets excluded)
└── frontend/              # FastAPI web console
```

### Feedback Loops

```
User asks question via Mira app
  → Analyst handler: briefing + claude_think → answer
  → If [GAP]: logs to feedback/gaps.jsonl
  → Next pipeline run: reads gaps → LLM commentary addresses them
  → Next briefing includes "Previously Asked" section

Debate generates recommendations
  → tracker.recommendations (entry, target, stop)
  → Daily mark-to-market → hit_target / hit_stop / expired
  → Next debate: sees track record → learns from losses
```

---

## Setup

### Prerequisites

- Python 3.11+
- PostgreSQL 15 (Docker recommended)
- API keys: Polygon, Finnhub, FRED, NewsAPI, OpenAI, DeepSeek, Gemini

### Quick Start

```bash
# Database
docker run -d --name tetra-db \
  -e POSTGRES_DB=tetra \
  -e POSTGRES_USER=tetra_user \
  -e POSTGRES_PASSWORD=yourpassword \
  -p 5432:5432 postgres:15

# Schema
docker exec tetra-db psql -U tetra_user -d tetra \
  -f /path/to/scripts/migrate_schemas.sql

# Python
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Config
cp config/secrets.example.yml config/secrets.yml
# Edit secrets.yml with your API keys

# Run
python scripts/run_analysis.py              # Full pipeline
python scripts/run_analysis.py --stage debate  # Just the debate
python scripts/run_report.py                # Generate report
```

### Scheduling

Register the LaunchAgent for automated daily runs at market close:

```bash
# Create run_daily.sh with your paths
# Register: launchctl load com.tetra.plist
```

---

## License

MIT
