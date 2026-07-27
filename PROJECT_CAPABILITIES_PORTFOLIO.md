# 🛍️ CalRetail — Enterprise Retail AI Intelligence Platform
### Architecture, Capabilities & Algorithmic Specifications

---

## 🏛️ System Architecture

CalRetail is built as a production-grade, two-tier industrial AI architecture:

```
                                  SQLite Database (data/calretail.db)
                                           ▲
                                           │ backend/utils/db.py
                                           ▼
                            backend/capabilities/ (16 Python AI Modules)
                                           ▲
                                           │ Lazy initialization & LRU memory cache
                                           ▼
                                 FastAPI Backend (Port 8000)
                                           ▲
                                           │ REST JSON APIs
                                           ▼
                                 Dash Console (Port 7860)
```

- **Data Layer**: Single source of truth SQLite database (`data/calretail.db`) with 31 normalized tables, indexed queries, PyArrow compact string dtypes, and read-only connection pooling.
- **Compute Layer**: 16 native Python capability modules running inside FastAPI, lazy-loaded on demand with LRU cache controls (`CALRETAIL_TABLE_CACHE=2`).
- **Presentation Layer**: Dash UI console with industrial dark/light design system, reactive state callbacks, Plotly charts, and fallback resilience.

---

## 🚀 The 16 Retail AI Capabilities & Algorithmic Reference

### 📦 Domain 01: Customer Experience (Hyper-Personalization & Discovery)

#### 1. Hyper-Personalized Recommendations
- **Module**: `backend/capabilities/personalised_recommendations.py`
- **Algorithm**: **User-User & Item-Item Collaborative Filtering (Cosine Similarity)**
- **Technical Detail**: Builds an implicit feedback matrix from user purchases, cart additions, and wishlist items. Calculates pairwise cosine similarity:
  $$\text{Cosine Similarity}(u, v) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$$
  Generates explainable recommendation reasons (e.g., *"Shoppers similar to you also bought..."*) and handles cold-start users by falling back to category bestseller popularity scoring.

#### 2. Personalized Buying Assistants
- **Module**: `backend/capabilities/conversational_buying_assistant.py`
- **Algorithm**: **TF-IDF Vector Space Retrieval + LLM Prompt Chaining**
- **Technical Detail**: Extracts shopping intent, budget constraints, and product attributes from natural language user queries using TF-IDF feature extraction and cosine distance ranking over the product catalog. Integrates Gemini/Groq/OpenAI LLM APIs with deterministic rule-based fallback.

#### 3. Next-Best-Offer (NBO) Engines
- **Module**: `backend/capabilities/next_best_offer.py`
- **Algorithm**: **RFM Segmentation + Uplift Propensity Scoring**
- **Technical Detail**: Segments customers into personas (e.g., *Brand Loyalists*, *Price Sensitive*, *At-Risk*) based on Recency, Frequency, and Monetary (RFM) values and K-Means clustering. Computes conversion uplift probabilities for personalized promotion matching.

#### 4. Communication Timing Optimizer
- **Module**: `backend/capabilities/communication_timing.py`
- **Algorithm**: **Hourly Kernel Density Estimation (KDE)**
- **Technical Detail**: Analyzes customer browsing timestamps and order histories. Applies continuous Kernel Density Estimation across 24-hour day periods to pinpoint peak engagement probability hours per individual customer:
  $$\hat{f}(x) = \frac{1}{nh} \sum_{i=1}^{n} K\left(\frac{x - x_i}{h}\right)$$

---

### 📊 Domain 02: Merchandising (Pricing, Assortment & Placement)

#### 5. Demand Forecasting
- **Module**: `backend/capabilities/demand_forecasting.py`
- **Algorithm**: **Global XGBoost Regressor with Rolling Lag Features**
- **Technical Detail**: Predicts 30-day product demand using a gradient boosted decision tree ensemble. Features include 7-day, 14-day, and 30-day lag variables, rolling moving averages, day-of-week indicators, promotional flags, and holiday calendars. Evaluated via Mean Absolute Error (MAE) and MAPE.

#### 6. Dynamic Pricing Engines
- **Module**: `backend/capabilities/dynamic_pricing.py`
- **Algorithm**: **Log-Log Price Elasticity Linear Regression**
- **Technical Detail**: Estimates price elasticity of demand ($\epsilon$) per product category:
  $$\ln(Q) = \alpha + \epsilon \cdot \ln(P) + \gamma \cdot X$$
  Simulates revenue and margin curves under price variations while enforcing minimum gross margin percentage constraints.

#### 7. Promotion Optimization
- **Module**: `backend/capabilities/promotion_optimization.py`
- **Algorithm**: **Promotional Uplift Modeling & Cannibalization Penalty Factor**
- **Technical Detail**: Fits non-linear discount-to-uplift curves over historical promotion logs. Models product cannibalization penalties across complementary categories to calculate Net Margin Uplift.

#### 8. Competitor Price Monitoring
- **Module**: `backend/capabilities/competitor_price_monitoring.py`
- **Algorithm**: **IQR & Z-Score Anomaly Outlier Detection**
- **Technical Detail**: Tracks competitor pricing across external channels. Uses Interquartile Range (IQR) and Z-score statistical tests to flag pricing premium or undercut anomalies in real time:
  $$Z = \frac{X - \mu}{\sigma}$$

---

### 🚚 Domain 03: Operational Efficiency (Supply Chain & Fulfillment)

#### 9. Smart Inventory Health Monitoring
- **Module**: `backend/capabilities/inventory_health_monitoring.py`
- **Algorithm**: **Composite Inventory Risk Scoring & ABC/XYZ Matrix**
- **Technical Detail**: Combines Stock-out Probability, Days of Supply (DoS), and Overstock Holding Cost into a normalized composite risk score ($0–100$). Categorizes SKUs using ABC (revenue contribution) and XYZ (demand variability) matrices.

#### 10. Automated Replenishment
- **Module**: `backend/capabilities/automated_replenishment.py`
- **Algorithm**: **Dynamic Safety Stock & Continuous Review $(s, S)$ Inventory Control**
- **Technical Detail**: Calculates dynamic Reorder Points (ROP) considering lead time demand variance:
  $$\text{Safety Stock} = Z_{\alpha} \times \sqrt{\bar{L} \sigma_D^2 + \bar{D}^2 \sigma_L^2}$$
  $$\text{ROP} = (\bar{D} \times \bar{L}) + \text{Safety Stock}$$

#### 11. Warehouse Slotting Optimization
- **Module**: `backend/capabilities/warehouse_slotting.py`
- **Algorithm**: **Co-Location Pick Frequency Affinity Clustering**
- **Technical Detail**: Analyzes historical picking orders to group items frequently picked together. Assigns high-velocity (Class A) items closer to packing stations to minimize picker travel distance.

#### 12. Logistics, Route & Fleet Optimization
- **Module**: `backend/capabilities/route_optimisation.py`
- **Algorithm**: **Vehicle Routing Problem with Time Windows (VRPTW) + 2-Opt Local Search**
- **Technical Detail**: Computes pairwise Haversine distance matrices between distribution centers and stores. Optimizes vehicle delivery routes using 2-Opt iterative heuristic edge exchange to minimize total fleet travel kilometers.

---

### 🎧 Domain 04: Customer Support (AI Resolution Layer)

#### 13. 24x7 AI Chatbots
- **Module**: `backend/capabilities/ai_chatbot.py`
- **Algorithm**: **Conversational Finite State Machine + RAG**
- **Technical Detail**: Manages multi-turn conversation states for order tracking, refund status, and store locator queries. Integrates LLM Retrieval-Augmented Generation (RAG) with rule-based fallback.

#### 14. Intelligent Ticket Triage
- **Module**: `backend/capabilities/ticket_triage.py`
- **Algorithm**: **Multi-Class Intent Classification & Sentiment Analysis**
- **Technical Detail**: Classifies incoming customer support tickets by urgency (High/Med/Low), category (Logistics/Billing/Product), and sentiment score. Routes urgent tickets automatically to tier-2 human agents.

#### 15. Agent Assist
- **Module**: `backend/capabilities/agent_assist.py`
- **Algorithm**: **BM25 Search & Vector Cosine Similarity Retrieval**
- **Technical Detail**: Searches internal Standard Operating Procedure (SOP) documentation during live customer calls. Provides suggested response templates and policy guidelines to human agents in real time.

#### 16. Voice of Customer (VoC)
- **Module**: `backend/capabilities/voice_of_customer.py`
- **Algorithm**: **Aspect-Based Sentiment Analysis (ABSA)**
- **Technical Detail**: Mines product review datasets. Extracts product aspects (e.g., *Fit*, *Fabric Quality*, *Zipper Durability*) and computes aspect-level sentiment polarity scores using VADER lexicon / Transformer models.

---

## 🛠️ Deployment & Infrastructure Specification

- **Containerization**: Docker multi-stage image based on `python:3.11-slim` with `libgomp1` OpenMP support.
- **Process Supervision**: Watchdog loop in `start.sh` ensuring automatic process recovery.
- **Memory Optimization**: PyArrow string downcasting, bounded LRU caches, and 2GB virtual memory swap configuration.
- **Live AWS EC2 Instance**: `t3.micro` instance running at `http://16.192.165.12`.
