---
title: CalRetail AI Platform
emoji: 🛍️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# 🛍️ CalRetail — Enterprise Retail AI Intelligence Platform

> **16 AI Capabilities · 4 Domains · 35 REST APIs · Dash Console**
> A retail AI platform for fashion retail, built on Python, FastAPI, Dash and
> sixteen pure-Python capability modules over one SQLite database. The console
> implements the Calsoft *Retail AI Solutions* deck.

---

## 📋 Table of Contents
- [Architecture Overview](#architecture-overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Capability Domains](#capability-domains)
- [Quick Start](#quick-start)
- [Running Tests](#running-tests)
- [Notes & Gotchas](#notes--gotchas)

---

## Architecture Overview

```
                     data/calretail.db  (SQLite, 31 tables)
                                  ▲
                                  │ backend/utils/db.py
                                  ▼
                  backend/capabilities/  (16 Python modules)
                                  ▲
                                  │ ordinary imports, built on first call
                                  ▼
                       FastAPI backend  (port 8000)
                                  ▲
                                  │ HTTP JSON
                                  ▼
                       Dash console     (port 8050)
```

Each capability is a **plain Python module** in `backend/capabilities/`. A
module's shared frames are built lazily by `_init()` on the first call, so
importing all sixteen costs ~100 MB and **0.8 s** and computes nothing for a
capability nobody asks for. `_registry` keeps only the most recently used few
warm (`CALRETAIL_WARM_CAPABILITIES`, default 3) and calls `reset()` on the rest,
which is what keeps the process inside a small memory budget.

These modules were ported from the notebooks in `notebooks/capabilities/`, which
remain as the readable narrative of each method. **They are no longer executed
by the application.** Running them at request time meant re-`exec`-ing every
cell — including demo cells that printed telemetry and drew matplotlib charts —
which cost seconds per capability, silently swallowed failing cells (a broken
cell surfaced much later as a missing attribute), and defeated bytecode caching.
Dropping that path made every capability faster, some dramatically:

| Capability | Notebook | Module |
|---|---|---|
| Conversational buying assistant | 7.4 s | 0.03 s |
| Personalised recommendations | 4.0 s | 0.2 s |
| 24×7 AI chatbot | 3.5 s | 0.03 s |
| Inventory health | 5.4 s | 2.7 s |

All sixteen were verified to return **byte-identical** results before and after
the port.

All data lives in one **SQLite database**, `data/calretail.db` — 31 tables, 38
indexes, ~68 MB. It is committed, so a clone runs immediately and the Hugging
Face Space deploys with no build step. Nothing reads CSVs any more.

---

## Tech Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| Database | SQLite | `data/calretail.db` — committed, read-only at runtime |
| Backend API | FastAPI + Uvicorn | Async, Pydantic-validated |
| Console | Dash (Plotly) | `frontend_dash/` — the current UI |
| ML & Analytics | pandas, NumPy, scikit-learn, XGBoost | cosine similarity, K-Means, ABC slotting, routing, forecasting |
| LLM | LangChain (Gemini / Groq / OpenAI) | falls back to a rule-based engine with no key |
| Tests | pytest | 32 checks over all 16 capability modules |

---

## Project Structure

```
CalRetail/
├── data/
│   └── calretail.db                # ← the database (committed, ~68 MB)
├── backend/
│   ├── main.py                     # FastAPI entry point
│   ├── utils/
│   │   ├── db.py                   # SQLite engine: connections, pushdown, caching
│   │   ├── llm_service.py          # provider detection + LangChain wrappers
│   │   └── data_loader.py          # table accessors + indexed lookups
│   ├── capabilities/               # the 16 capability modules (lazy _init)
│   ├── services/                   # thin proxies onto the capabilities
│   └── routers/                    # 33 REST routes
├── frontend_dash/                  # ← the console
│   ├── app.py                      # shell, routing, pre-paint theme bootstrap
│   ├── assets/
│   │   ├── style.css               # design system (tokens, cards, nav, dark mode)
│   │   ├── theme.js                # light/dark toggle + persistence
│   │   └── chart_theme.js          # keeps Plotly figures in step with the theme
│   ├── components/
│   │   ├── cards.py                # card / kpi / pill / bar / table / money
│   │   └── layout.py               # right-hand nav rail + page header
│   ├── services/
│   │   ├── api.py                  # cached FastAPI client (never raises)
│   │   └── capabilities.py         # the 16 deck capabilities + their qualifiers
│   ├── theme/                      # colors.py + Plotly chart theme
│   └── pages/                      # home, 4 domains, AI assistant
├── notebooks/
│   ├── capabilities/               # the 16 notebooks (reference, not executed)
│   ├── generate_data.py            # seeded synthetic generator (stage 1)
│   ├── clean_data.py               # cleaning rules (stage 2)
│   ├── feature_engineering.py      # feature_* tables (stage 3)
│   ├── pipeline_io.py              # build-time SQLite reads/writes
│   └── build_db.py                 # runs all three, indexes, VACUUMs
└── tests/test_capabilities.py
```

---

## Capability Domains

Sixteen capabilities across four domains. **Every one is served by a real
endpoint reading the database** — there is no seeded or illustrative data left
in the console.

| # | Domain | Capabilities | Live |
|---|--------|--------------|------|
| 01 | Customer Experience | Hyper-personalized Recommendations · Personalized Buying Assistants · Next-Best-Offer Engines · Communication Timing Optimiser | 4/4 |
| 02 | Merchandising | Demand Forecasting · Dynamic Pricing Engines · Promotion Optimization · Competitor Price Monitoring | 4/4 |
| 03 | Operational Efficiency | Smart Inventory Management · Automated Replenishment · Warehouse Optimization · Logistics, Route & Fleet Optimization | 4/4 |
| 04 | Customer Support | 24x7 AI Chatbots · Intelligent Ticket Triage · Agent Assist · Voice of Customer | 4/4 |

Each card carries the deck's own **Impact / Data / Speed** qualifiers and its
deployment **Wave** badge.

### Names, not identifiers

Nothing in the console shows a raw `C00001` / `P00489` / `W002`. Identifiers are
resolved to names in the **backend**, by [`backend/utils/naming.py`](backend/utils/naming.py),
so every consumer of an endpoint gets the name for free rather than each card
re-implementing its own lookup:

```python
from backend.utils import naming

naming.customer("C00001")     # 'Niharika Bhatti'
naming.warehouse("W002")      # 'Bengaluru DC 2'
naming.annotate(rows)         # adds *_name beside every known *_id
naming.location_label(row)    # store or warehouse, whichever the row has
```

`annotate` never overwrites a name a capability already supplied — that one's
is the more specific one. Maps are built once per process from the database and
memoised, so resolution is a dict hit, not a query per row.

---

## Quick Start

> **`PYTHONUTF8=1` is required.** Python 3.14 still defaults to cp1252 on Windows,
> and several capabilities print `≈` and `₹`. Without UTF-8 mode those writes
> raise `'charmap' codec can't encode character`.

The database is committed, so there is no data step. Install and run:

```bash
pip install -r requirements.txt
```

### 1. Backend API — port 8000
```bash
PYTHONUTF8=1 python -m uvicorn backend.main:app --port 8000
```

### 2. Dash console — port 8050
```bash
PYTHONUTF8=1 python frontend_dash/app.py
```

PowerShell equivalent:
```powershell
$env:PYTHONUTF8 = "1"
python -m uvicorn backend.main:app --port 8000
python frontend_dash\app.py
```

### 3. Open
- **Console** — <http://127.0.0.1:8050>
- **Swagger** — <http://127.0.0.1:8000/docs>

Set `DASH_DEBUG=1` for hot reload and the callback graph. It's off by default
because the dev-tools widget overlays the last card in the grid.

---

## The Database

`data/calretail.db` holds all 31 tables and is the only data artifact. The app
opens it **read-only**, one connection per thread, so a request can never mutate
the demo data.

### Rebuilding

```bash
python -m notebooks.build_db                # demo scale — what is committed
python -m notebooks.build_db --scale full   # full fidelity: 3.9M rows, 528 MB
python -m notebooks.build_db --scale 0.5    # anything in between
```

The pipeline runs `generate_data → clean_data → feature_engineering`, then
indexes and VACUUMs. The generator is seeded (`random.seed(42)`), so a given
scale always produces the same database.

Stop the backend and console first — Windows will not let the builder replace a
file those processes hold open.

### What `--scale` does

Scale multiplies **event-log** tables only. Customers (10,000), products
(5,000), stores, warehouses, suppliers and inventory (25,000) are always
generated at full size, so a demo build still shows a complete catalogue and
customer base — only the behavioural history behind them is thinner.

Every customer is guaranteed at least one transaction and one order at any
scale. Sampling customers independently would leave ~9% of them with no history
at demo scale, which makes the recommendation and chatbot cards look broken
rather than sparse.

### Why demo scale is the committed default

A full-fidelity build is 528 MB, which exceeds GitHub's 100 MB per-file limit
and cannot be deployed to a Hugging Face Space without LFS. The demo build is
68 MB: it commits normally, clones fast, and the Space boots straight into a
warm dataset with no build step.

To run against a full build without touching the committed one:

```bash
CALRETAIL_DB=/path/to/full.db python -m notebooks.build_db --scale full
CALRETAIL_DB=/path/to/full.db python -m uvicorn backend.main:app --port 8000
```

### Reading from it

```python
from backend.utils import db, data_loader as dl

dl.get_customers()                 # whole table, memoised (what services use)
dl.customer_transactions("C00001") # indexed lookup, ~2 ms
db.query("SELECT category, COUNT(*) FROM products GROUP BY category")
```

`load_df` parses date columns to datetimes; `load_table` (used by the capability
capability modules) leaves them as ISO-8601 strings, which is the contract they
were written against when they read CSVs.

---

## Deployment

The console and the API are hosted separately, because they have very different
appetites. The Dash console imports only `dash`, `plotly` and `requests` —
100 MB, comfortably inside Vercel's 250 MB function limit. The backend's
scientific stack is ~427 MB installed, roughly twice that limit, so it runs as a
container on Render.

| Piece | Host | Notes |
|---|---|---|
| Dash console | Vercel | `api/index.py` + `vercel.json`, `requirements-vercel.txt` |
| FastAPI backend | Render | same `Dockerfile`, `CALRETAIL_MODE=api` |

`start.sh` serves only the API when `CALRETAIL_MODE=api`, which keeps Dash's
~140 MB out of a process capped at 512 MiB. The console finds the backend
through `CALRETAIL_API_BASE`.

### Staying inside 512 MiB

The free tier is a hard 512 MiB and the process is killed the moment it crosses.
Three settings hold the line, all tunable:

| Variable | Default | What it bounds |
|---|---|---|
| `CALRETAIL_WARM_CAPABILITIES` | 3 | capabilities holding frames at once |
| `CALRETAIL_TABLE_CACHE` | 8 | whole tables memoised by `db.load_df` |
| `CALRETAIL_RESULT_TTL` | 1800 | seconds an endpoint result is reused |

The result cache is what makes a small warm set affordable: an answer outlives
the capability that produced it, so evicting one costs memory back without
costing speed. `CALRETAIL_WARM_CACHE=1` (the default) computes the expensive
reads once at boot in a background thread, so the first visitor does not pay
for them either.

### Sleep

Render's free tier stops a service after ~15 minutes without traffic, and waking
it costs a container cold start plus rebuilding the in-process cache.
`.github/workflows/keep-api-awake.yml` pings `/health` every 10 minutes to
prevent that — free on a public repository, and `schedule` runs on the default
branch. GitHub disables scheduled workflows after 60 days of repository
inactivity; re-enable from the Actions tab if the API starts sleeping again.

---

## Running Tests

```bash
PYTHONUTF8=1 python -m pytest tests/ -q
```

---

## Notes & Gotchas

- **Dependencies.** `requirements.txt` pins with `>=`, so a fresh install
  resolves to current majors (pandas 3.x, langchain-core 1.x). Two things that
  follow from that are already handled in code, but worth knowing if you re-pin.
- **`jupyter` metapackage.** Not installed, and not needed — the application no
  longer executes notebooks. Install it only to read them in JupyterLab, along
  with `matplotlib`, which the demo cells use and the app does not.
- **Capability state is bounded.** Only `CALRETAIL_WARM_CAPABILITIES` (default
  3) keep their frames in memory; the coldest is `reset()` when a fourth builds.
  A rebuild is a few seconds, so raise the limit wherever memory is not tight.
- **Rebuilding while the app runs** fails on Windows with `PermissionError`
  (`WinError 32`): uvicorn holds the database open. `build_db` detects this and
  tells you to stop the processes.
- **LLM responses.** In langchain-core 1.x, Gemini returns `content` as a list of
  blocks, not a string. `llm_service._response_text()` normalises both shapes —
  without it every call silently fell through to the rule-based engine while
  still logging `LLM: Gemini ✅`.
- **No API key?** Everything still runs; LLM-backed cards report
  `powered_by: Rule-Based Engine`.
