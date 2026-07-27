# CalRetail — how the whole thing works

One document covering the stack end to end: what runs where, how a number gets
from the database onto a card, and what every card on every page is actually
showing you.

---

## 1. What this is

A retail-AI console. Four business domains, sixteen capabilities, all served
from one SQLite database of synthetic-but-fixed retail data. Nothing on the
console is mocked at the UI layer — every figure you see came back from a real
HTTP call to the FastAPI backend, which computed it from the database.

Two processes:

| Process | Port | What it is |
| --- | --- | --- |
| `backend.main:app` | 8000 | FastAPI. Owns the data and all computation. |
| `frontend_dash.app` | 8050 | Dash (Flask + React). Owns layout only. |

Five routers: `overview` (the home dashboard), plus one per domain —
`customer-experience`, `merchandising`, `operations`, `support`.

The frontend has no database access and no business logic. If a number is wrong,
it is wrong in the backend or in the notebook behind it.

---

## 2. Running it

```bash
# Backend — the UTF-8 flags are NOT optional, see below
PYTHONIOENCODING=utf-8 PYTHONUTF8=1 python -m uvicorn backend.main:app --port 8000

# Frontend
python -m frontend_dash.app          # http://127.0.0.1:8050
```

**The UTF-8 gotcha.** Capability logic lives in Jupyter notebooks that are
executed at runtime (§5). Several of them `print()` rupee amounts. On a Windows
console defaulting to cp1252, printing `₹` raises `UnicodeEncodeError`, which
kills that notebook cell. The loader logs the error and moves to the next cell —
so the variables that cell was supposed to define never exist, and every later
cell fails with `NameError`. The visible symptom is unrelated-looking: every
merchandising endpoint returns HTTP 500 with `NameError: name 'pricing_merge' is
not defined`. Setting the two environment variables makes all of it go away.

Useful env vars:

- `CALRETAIL_API_BASE` — point the frontend at a backend on another host/port.
- `DASH_HOST` / `DASH_PORT` — bind address for containers.
- `DASH_DEBUG=1` — Dash dev tools + hot reload (off by default; the dev-tools
  widget anchors bottom-right and covers the last card in the grid).

---

## 3. The path a number takes

```
Browser
  │  user picks a product / clicks a button
  ▼
Dash callback                    frontend_dash/pages/*.py
  │  api_get(...) / api_post(...)     frontend_dash/services/api.py
  ▼
FastAPI route                    backend/routers/*.py
  │  thin: validate, delegate, translate errors to HTTP
  ▼
Service function                 backend/services/*.py
  │  either calls into a notebook module, or does its own SQL + pandas
  ▼
Data access                      backend/utils/db.py, data_loader.py
  │  sqlite3 → pandas DataFrame, memoised
  ▼
data/calretail.db                ~32 tables, ~68 MB
```

The return trip is JSON → a Python list of Dash components → React. The Dash
callback is the *only* place the two halves meet.

### The frontend HTTP client

`frontend_dash/services/api.py` is deliberately small:

- `api_get(path, params)` — 300-second TTL cache, returns `None` on any failure.
- `api_post(path, payload)` — never cached (these are compute calls).
- `last_failure(path)` — why the last call failed: `"missing"` (404, the route
  isn't implemented), `"http-<code>"`, `"unreachable"`, or `None`.

Both swallow every exception and return `None`. That keeps call sites simple but
loses information, which is what `last_failure` restores: without it a card
cannot tell "this endpoint does not exist" from "there genuinely is no data",
and ends up telling the reader *"no sales history for this product"* when the
truth is the route 404s. Operations uses `_unavailable()` to say which.

---

## 4. The data layer

Everything lives in `data/calretail.db`. Roughly 10,000 customers, 5,000
products, 36,000 transactions, ~32 tables.

Two families of table:

- **Raw** — `customers`, `products`, `orders`, `transactions`, `inventory`,
  `stores`, `warehouses`, `suppliers`, `customer_reviews`, `support_tickets`,
  `competitor_pricing`, `promotions`, `browsing_history`, `shipments`, …
- **Feature** — precomputed by the pipeline in `notebooks/`:
  `feature_inventory_health`, `feature_daily_sales`, `feature_customers`,
  `feature_products`, `feature_tickets`, `feature_buying_intent`.

The `feature_*` tables are why the console is fast. A health score or a daily
sales series is computed once by the pipeline, not per request.

**Caching.** `db.load_df(table)` is `@lru_cache`d, so each table is read from
SQLite once per process and thereafter served from memory. That is also why the
backend must be restarted to pick up a regenerated database.

**Dates are anchored to the data, not the clock.** This is a fixed dataset
ending in 2024. Anything windowed (`get_inventory_timeseries`, the forecast)
anchors to the last date *present in the data*. Anchoring to `today` would
return an empty series forever.

---

## 5. The notebook execution model

This is the most unusual part of the codebase and worth understanding before
debugging anything.

Each of the sixteen capabilities has a notebook in `notebooks/capabilities/`
(`01_personalised_recommendations.ipynb` … `16_voice_of_customer.ipynb`). At
runtime `backend/utils/notebook_loader.py`:

1. Reads the `.ipynb` as JSON.
2. Creates an empty `ModuleType`.
3. `exec()`s each code cell **in order**, into that module's `__dict__`.
4. Caches the resulting module in `_loaded_notebooks`.

So a notebook becomes an importable module whose top-level variables and
functions the service layer can call. `matplotlib` is forced to `Agg`, or
stubbed with a spec-compliant `MagicMock` if absent, so plotting cells don't
explode headless.

Consequences you need to know:

- **Cells share state, and failures cascade silently.** A failing cell is logged
  and skipped, not raised. Later cells then fail on missing names. Always read
  the *first* error in the backend log, not the last.
- **First call to a capability is slow** — the whole notebook runs. After that
  it's a cached module lookup. `main.py` starts a background thread that pre-warms
  the merchandising notebooks at startup for this reason.
- **Notebook edits need a backend restart**, because of the module cache.

Not everything goes through notebooks. Newer operations endpoints
(`markdown-candidates`, `inventory-timeseries`) are plain SQL + pandas in
`backend/services/operations.py`, which is faster and much easier to debug.

---

## 6. The frontend

```
frontend_dash/
  app.py                  Dash app, routing shell, dark-mode boot script
  components/
    layout.py             sidebar, page header, app shell
    cards.py              every UI primitive (card, split, kpi, table, pill…)
  pages/                  one module per route, auto-registered
  services/
    api.py                HTTP client + failure tracking
    capabilities.py       the 16-capability catalogue (titles, waves, endpoints)
  theme/                  colour tokens + Plotly figure defaults
  assets/
    style.css             the entire design system
    theme.js              light/dark toggle
    chart_theme.js        re-themes already-rendered Plotly figures on toggle
```

### Routing

`use_pages=True`. Each page module calls `dash.register_page(__name__, path=...)`
and exposes `layout()`. `app.py` renders `app_shell(pathname, dash.page_container)`.

### Two ways a page loads data

1. **In `layout()`** (Customer Experience, Operations, Support). Simple, but the
   page blocks until every call returns.
2. **Lazy, via callback** (Home, Merchandising). `layout()` returns instantly
   with a `dcc.Store` (`hm-load`, `mc-page-load`); callbacks fire off that Store
   and fill the cards in, showing `dcc.Loading` spinners meanwhile.

Pattern 2 is the better one and the direction the codebase is moving — Home and
Merchandising paint in ~0 ms where the other three wait on the network. The three
remaining blocking pages are the obvious next thing to convert.

### `Input` vs `State` — a real trap

A dropdown whose options are loaded lazily **must be an `Input`**, not `State`.
As `State`, the callback fires once at page load — before the options exist —
sees `None`, renders "pick a product", and then never fires again. The card sits
empty next to a populated dropdown until someone happens to click the button.
Dynamic Pricing had exactly this bug.

---

## 7. The design system

All in `assets/style.css`. The rules that matter:

**Grid.** `.grid-2` is two columns, `align-items: start`. Cards are exactly as
tall as their content. Stretching them to a common row height was tried and is
worse — Replenishment beside Warehouse differed by ~430 px, and a bordered card
padded out with that much blank reads as a rendering fault. `grid-auto-flow:
dense` is deliberately **not** set: with `span-2` cards in the mix it back-fills
gaps by pulling a later card forward, silently reordering the page.

**`.card-split` — the two-column card body.** A card carrying KPIs + a chart +
two tables is several screens tall stacked. `C.split(left, right)` lays it out as
two columns instead, turning a tall card into a wide one. It is driven by a
**container query on the card**, not a media query — the split turns on when the
*card* is wide enough (≥680 px), so a card in a half-width column stays stacked
and the same card spanning the grid splits. Pages never reason about viewport
width. Browsers without `@container` keep the single-column default.

> **Gotcha:** `container-type: inline-size` means a card's width is resolved
> *without consulting its contents*. Any card that becomes shrink-to-fit — e.g.
> via `margin-inline: auto` — then resolves to **zero**. That is why
> `.card-solo` carries an explicit `width: 100%`. Without it the AI Assistant
> card collapsed to its 2 px of border and every line inside rendered as one
> character per row.

**Tables.** `table-layout: auto`, so columns get the width their content needs
(fixed gave a "Risk" column the same width as "Product"). Cells use
`overflow-wrap: break-word`, **never `anywhere`** — `anywhere` also drops a
column's *minimum* width to one character, and auto-layout takes it: headers
collapse into vertical letter stacks, and the resulting header row is tall enough
to fill the scroll well and hide every data row. Numeric cells are `nowrap`, so
`₹1,24,500` never splits across lines.

**Keeping tables inside the card.** The real fix for a too-wide table is fewer
columns, not smaller text. Related columns fold into one cell via `.cell-stack`
(a health score above its risk pill; a product above its class badge). Every
table on the console now fits its card with zero horizontal overflow.

**KPI tiles.** `.kpi-grid` is flex-wrap-with-grow, not `grid auto-fit`. auto-fit
fixes the column count, so four tiles in a card that fits three per row leave two
dead cells; flex lets the last row expand. The `178px` basis is tuned: small
enough that two tiles still pair in the narrowest place they appear (the ~389 px
column of a 60/40 split at 1366 px), large enough that a half-width card still
lands on a tidy 2×2 for four KPIs.

**Dark mode.** `body.dark` swaps CSS custom properties. `color-scheme` is set on
both themes so *native* UI — number-input spin buttons, scrollbars, focus rings —
follows the theme; CSS cannot restyle those directly.

### Primitives in `components/cards.py`

| Call | Use |
| --- | --- |
| `C.card(title, body, caption, info, span, cls)` | the standard card; `span=2` spans the grid |
| `C.split(left, right, weight, ruled)` | two-column card body |
| `C.kpi_grid([C.kpi(label, value, delta, trend)])` | stat tiles |
| `C.table(headers, rows, numeric, wide, narrow, full)` | data table with column hints |
| `C.pill(text, level)` | status pill; `level` maps through `colors.risk_class` |
| `C.bar_row(label, value, pct, tone)` | labelled progress bar (empty label drops the column) |
| `C.stat_list([(k, v)])` | label/value rows; scrolls past 6 |
| `C.graph(fig, height)` | Plotly figure with shared config |
| `C.subhead(text)` / `C.empty(text)` / `C.caveat(text)` | section label / empty state / caveat |

Each card takes a `caption` (what the reader is looking at) and an `info`
(the hover "i" — how it was computed). `info` accepts `<b>…</b>`, rebuilt into
real `html.B` nodes by `rich()` so raw markup never reaches the DOM.

---

## 8. Card-by-card: what each one shows and where it comes from

### Home — `/`

A trading dashboard, then the capability index. Everything lazy-loads off a
`dcc.Store(id="hm-load")` so the page paints before any rollup finishes. All five
analytical cards are served by the `/api/v1/overview/*` router.

The reading order is deliberate — each row answers the question the row above
raises: *what is the position → how is it trending → where should money go next
→ when and what sells.*

**Trading position** (full width) — `GET /api/v1/overview/estate`
Revenue, gross margin, units, buyers, average basket and return rate for the
whole estate, aggregated from the transaction log rather than sampled. The line
underneath grounds it: how many SKUs sold out of the catalogue, across how many
stores, DCs and vendors.

**Revenue and gross margin by month** (full width) — `GET /api/v1/overview/revenue-trend`
36 months of both series in rupees on **one** scale, so the gap between the two
lines literally *is* cost of goods sold. Margin is revenue less
`quantity × cost_price` per line, joined from the product catalogue.

**Category momentum** (full width) — `GET /api/v1/overview/category-performance`
A four-encoding bubble quadrant: position carries revenue and growth, area
carries volume, colour carries margin on a single-hue ramp, and identity sits on
the label — so no categorical hue is needed. The split lines are the **medians of
what is currently on screen**, not fixed thresholds, so the quadrants stay
meaningful at any scale. Filters (category / channel / region) come from
`GET /api/v1/overview/filters`; choosing a category drills into its
sub-categories. There is no year control because growth is already a
year-on-year comparison.

**Trading seasonality** — `GET /api/v1/overview/seasonality`
Month × weekday heatmap. Each cell is revenue per **trading day**, not total
revenue — otherwise months with more selling days simply read as busier.

**Top movers** — `GET /api/v1/overview/top-movers`
The eight SKUs carrying the most revenue, with units and margin %.

**4 × domain cards** — from `capabilities.py`, no database. Each lists its
capabilities, how many are live, and links through to the domain console.

---

### Domain 01 — Customer Experience — `/customer-experience`

**Hyper-personalized Recommendations** (full width, split)
`POST /api/v1/customer-experience/recommendations` → notebook 01.
User–user collaborative filtering over the purchase matrix: products are ranked
by how many *similar* customers bought them, blended with average rating.
Cold-start falls back to bestsellers in the customer's preferred category.
Left column is the shopper-facing result — profile pills and the ranked product
list (scrolled, because top-N goes to 25). Right column is the admin evidence:
why each product was picked, which similar customers drove it (cosine
similarity), and the customer's own category history. Reasons are plain English;
raw scores are never shown.

**Next-Best-Offer Engines**
`GET /api/v1/customer-experience/next-best-offer/segment` → notebook 03.
A campaign-planning view: active promotions scored against a whole segment's
dominant preferred category and channel, plus whether they on-target that
segment, then ranked by predicted uplift. Segments come live from
`/segmentation`. An off-target row shows an "off-target" pill with the segment it
*does* target on a sub-line — inside the pill, a word like "Professional" set a
~100 px floor on the column and pushed the table past the card.

**Personalized Buying Assistants**
`POST /api/v1/customer-experience/buying-assistant` → notebook 02.
Free-text intent extraction: category, colour and price ceiling parsed out of a
sentence (LLM if a key is configured, deterministic rules otherwise), turned into
a catalogue filter. Suggestions come from the real catalogue.

**Communication Timing Optimiser** (full width, split)
`GET /api/v1/customer-experience/communication-timing` → notebook 04.
Aggregates the customer's session log by hour and day-of-week to find when they
are actually online; channel is inferred from device/session patterns. Left
column is the four recommendations, right column the hourly histogram that
justifies them — the answer and its evidence, side by side.

---

### Domain 02 — Merchandising — `/merchandising`

This page lazy-loads everything (§6).

**Dynamic Pricing Engines** (full width, split)
`POST /api/v1/merchandising/dynamic-pricing` → notebook 06.
An elasticity-weighted price search bounded by a hard **floor price**
(cost + minimum margin); competitor mean and stock cover pull the recommendation
up or down, and it never returns a price below the floor. Left column is what the
engine recommends (KPIs + the floor/current/recommended/competitor bar chart).
Right column is the what-if: override the price and the margin, projected revenue
and volume impact recompute client-side from the elasticity, with pills flagging
below-cost, below-floor and above-market. Whole rupees — the field declares
`step=1`, so a fractional seed would open the input `:invalid` and red.

**Competitor Price Monitoring**
`GET /api/v1/merchandising/competitor-monitoring` → notebook 08.
The whole catalogue is swept; each SKU's gap against the competitor mean becomes
a **z-score within its category**, and anything outside the band raises
`alert_flag` with a recommended action. Only breaches are listed. The category
dropdown filters client-side through the same shared builder used for the initial
render, so a fix lands in both paths.

**Promotion Optimization**
`GET /api/v1/merchandising/promotion-optimization` → notebook 07.
Treated vs. matched-control revenue for one promotion. **Cannibalization** is the
share of the lift stolen from full-price sales — above 30 % the promotion is
moving margin, not volume, and the bar turns red.

**Assortment Planning** (full width, split)
`GET /api/v1/merchandising/assortment-plan`.
Orders joined to products and to each store's region; Cancelled and Returned
orders excluded (neither leaves revenue on the books). Because regions differ in
size, a SKU is judged on its **share of its own region's revenue**, never on
rupees. **80/20** is a real Pareto — the share of selling SKUs earning the first
80 % of revenue. An **add** is proven in ≥2 other regions yet earning under half
that peer share here; a **drop** sits below a quarter of the region's median SKU
revenue, with inventory joined so stocked-but-static lines surface. Left column
reads the shape of the estate, right column is the actionable detail.

**Demand Forecasting** (full width, split, chart-weighted)
`GET /api/v1/merchandising/demand-forecast` → notebook 05.
A global multi-product regressor over calendar, price and promotion features.
The shaded band is the prediction interval the model will commit to over the
30-day horizon. Summary figures left, series right.

---

### Domain 03 — Operational Efficiency — `/operations`

All four cards are full width and split — the content on this page is dense
enough that half-width cards produced ~430 px height mismatches.

**Smart Inventory Management**
Left: `GET /api/v1/operations/inventory-health` → notebook 09. Days of cover,
stockout probability and supplier reliability collapsed into one 0–1 health
index; the risk mix is split by location type because a store stockout and a DC
stockout need different remedies. The buy-list is the weakest positions.
Right: `GET /api/v1/operations/markdown-candidates` — the opposite failure.
SKUs flagged `overstock_flag` (the pipeline's own `stock_qty > max_stock`, reused
deliberately so this endpoint agrees with the health score) **and** stockout risk
≤ 10 %, ranked by **capital tied up** rather than by cover: a year of cover on a
₹200 vest is not the problem a quarter of cover on a ₹9,000 coat is. The
suggested markdown deepens with excess cover, and "freed" is net of what the
discount costs.
Below that, `GET /api/v1/operations/inventory-timeseries` draws one SKU's real
daily units sold with a 7-day rolling mean. Days with no sale are emitted as
zero — `feature_daily_sales` only stores days that had a transaction, and
plotting that sparse frame directly overstates demand and averages over selling
days instead of calendar days.

**Automated Replenishment**
`POST /api/v1/operations/replenishment` → notebook 10.
A min/max policy: when stock falls to the **reorder point** (demand over lead
time plus safety stock), order back up to max. Supplier reliability widens the
safety buffer. Left column is the order to place; right column is the policy
thresholds it came from.

**Warehouse Optimization**
`GET /api/v1/operations/warehouse-optimization` → notebook 11.
SKUs ranked by movement and cut at the 80 %/95 % cumulative thresholds into
**A/B/C** classes, then mapped onto pick zones by walking distance. Savings are
modelled travel time, not headcount. Left column is the class mix and a
plain-language legend for what A/B/C mean; right column is the movers themselves
and a per-class SKU list.

**Logistics, Route & Fleet Optimization**
`POST /api/v1/operations/route-optimization` → notebook 12.
A nearest-neighbour tour with 2-opt improvement over the haversine distance
matrix, closing back at the depot. **Baseline** is the unordered
depot-and-back-again pattern the route replaces. The map is real MapLibre/Carto
tiles at the outlets' true coordinates; co-located stops are fanned out in a
tight golden-angle spiral (~4–6 km) so pins unstack without lying about
location. Pins are numbered in visiting order and sized by orders on board. The
map takes ~60 % of the card; the numbers stack beside it.

---

### Domain 04 — Customer Support — `/support`

**24×7 AI Chatbots**
`POST /api/v1/support/chatbot` → notebook 13. Intent classification, then an
answer built from that customer's own order and catalogue records. Shows the
detected intent and whether it would escalate.

**Intelligent Ticket Triage**
`POST /api/v1/support/ticket-triage` → notebook 14. Two classifiers — category
and priority — run over raw ticket text; overall confidence is their blend,
shown as-is rather than talked up. Also predicts the owning team.

**Agent Assist** (full width, split)
`POST /api/v1/support/agent-assist` → notebook 15.
Encodes the support knowledge base and retrieves the closest resolved tickets
from real ticket history. The **recommended next step** is the reply that worked
on the nearest of them; **match confidence** is that ticket's similarity score,
so a low number means the suggestion is a weak analogy, not a verified answer.
Left column is what to say now (plus knowledge-base links), right column the
tickets it was drawn from.

> This card read `matched_sop` / `suggested_response` / `similar_cases` /
> `confidence` — **none of which this endpoint has ever returned**. Every block
> fell through to its fallback and the card rendered essentially blank. The real
> contract is `recommended_sop`, `matched_tickets`, `knowledge_articles`.

**Voice of Customer** (full width, split)
`GET /api/v1/support/voice-of-customer` → notebook 16.
Aspect-level sentiment over the product-review dataset, filterable by product and
date range. Left column is volume and the monthly rating trend; right column is
what reviewers actually talk about — mentions, average rating and % positive per
aspect.

---

### AI Assistant — `/ai-assistant`

`POST /api/v1/support/assistant-router`, plus whichever capability endpoint the
question resolves to.

A deterministic keyword table built from every capability's real trigger phrases
is checked **first**, so the page works identically with or without an LLM key.
Only if nothing matches locally does it fall back to the router's own (narrower)
classifier, and only if that also returns `general_chat` does it show the
router's conversational reply. The answer is then produced by calling the same
endpoint the corresponding domain page would call.

Two standing rules on this page: it never renders a raw entity code (`C00001`,
`P00001` — ids are call arguments, never visible text), and it never shows a raw
model score. Where a question needs an entity it didn't name, a real one is
picked from the list endpoint and named.

---

## 9. Known gaps and things to watch

- **Backend must run with UTF-8 stdout.** See §2. This masquerades as a
  merchandising `NameError`.
- **Notebook cell failures are silent.** Read the *first* traceback in the log.
- **`db.load_df` is process-cached** — restart the backend after regenerating
  the database, and after editing any notebook.
- **Blocking `layout()`** on Customer Experience, Operations and Support. These
  three pages wait on the network before first paint. Merchandising shows the
  lazy-load pattern they should adopt.
- **Assortment "Add candidates" reads `+0`** with `₹0` opportunity while the
  per-region table shows `+15` per region. The per-region and total figures are
  computed differently in the backend; worth reconciling.
- **`overflow-wrap: anywhere` is banned** in this stylesheet. It looks harmless
  and silently destroys table and pill layout under pressure. Use `break-word`.
- **Adding a column to a table is not free.** Every column claims at least its
  header's longest word. Before adding one, check whether it can fold into an
  existing cell with `.cell-stack`.
