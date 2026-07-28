# 📊 CalRetail — Enterprise Retail AI Capabilities Presentation Deck

> **Executive Presentation Deck**: 20 Slides covering Platform Architecture, Testing Benchmarks, Deployment Specs, and deep-dive **What, Why, and How** for all 16 AI Capabilities.

---

## 📽️ Slide 1: Title & Executive Summary
- **Platform Name**: CalRetail Enterprise AI Platform
- **Presentation Title**: Retail AI Capabilities Deck — Complete Technical & Business Breakdown
- **Scope**: Comprehensive analysis of 16 retail AI capabilities across 4 business domains (Customer Experience, Merchandising, Operations, Support).

---

## 🏛️ Slide 2: Platform Architecture & Technical Pillars
- **Data Layer**: Single source of truth SQLite database (`data/calretail.db`) with 31 normalized tables, 38 indexes, PyArrow compact string downcasting (65% RAM savings).
- **Compute Layer**: 16 native Python capability modules in `backend/capabilities/` lazy-initialized on demand, served via FastAPI (Port 8000).
- **Presentation Layer**: Dash Web Console (Port 7860) with sage-green industrial theme, reactive callbacks, and top-bar loading animations.
- **Infrastructure**: Containerized Docker image running live on AWS EC2 (`t3.small`, 2 GB RAM, 2 GB Swap, Watchdog supervisor loop).

---

## 📦 Domain 01: Customer Experience (Personalization & Discovery)

---

### 🛈 Slide 3: 1. Hyper-Personalized Recommendations
- **📌 WHAT**: User-User & Item-Item Collaborative Filtering cosine similarity matrix built over shopper behavioral history.
- **💡 WHY**: Drives cross-sell revenue, increases average order value (AOV), and eliminates product discovery friction.
- **⚙️ HOW (Algorithm & Formula)**:
  - `Implicit Signal = (3.0 * Purchase Qty) + (2.0 * Cart Additions) + (1.0 * Wishlist Saves)`
  - `Similarity = CosineSimilarity(User_A_Signal, User_B_Signal)`
  - `Final Score = Sum(User_Similarity * OtherUser_Product_Signal) * Category_Boost_Multiplier`
- **🎯 REAL-WORLD EXAMPLE**: Shopper `C00001` bought Denim Jacket (Signal 6.0). Similar shopper `C00084` (Similarity 0.85) bought Leather Bag (Signal 3.0). System recommends Leather Bag with Score **2.93** (#1 Rank).

---

### 🛈 Slide 4: 2. Personalized Buying Assistants
- **📌 WHAT**: Natural language intent parsing & TF-IDF attribute retrieval matching customer budget and specifications.
- **💡 WHY**: Replaces rigid drop-down search with conversational product discovery, boosting sales conversion.
- **⚙️ HOW (Algorithm & Formula)**:
  - `TF-IDF = (Keyword Frequency in Item) * Log(Total Catalog Items / Items containing Keyword)`
  - `Relevance = CosineSimilarity(Query_TFIDF, Product_TFIDF) subject to Product_Price <= Maximum_Budget`
- **🎯 REAL-WORLD EXAMPLE**: Query *"red jacket under 3000"* extracts Max Price = ₹3,000, Color = Red. Matches SKU `P00102` (*Red Puffer Winter Jacket*, ₹2,499, TF-IDF 0.92). Assistant returns product recommendation.

---

### 🛈 Slide 5: 3. Next-Best-Offer (NBO) Engines
- **📌 WHAT**: RFM customer segmentation combined with uplift propensity scoring to select target promotional discounts.
- **💡 WHY**: Maximizes conversion probability while preventing unnecessary margin giveaway on price-insensitive shoppers.
- **⚙️ HOW (Algorithm & Formula)**:
  - `Incremental Margin = Average_Basket_Value * Conversion_Uplift_% * (1.0 - Discount_%)`
  - `RFM Persona Score = Recency + Frequency + Monetary`
- **🎯 REAL-WORLD EXAMPLE**: *Value Hunter* customer segment: Option A (10% Off) yields +6% Uplift (**₹120 Net Margin**). Option B (25% Off) yields +16% Uplift (**₹95 Net Margin**). System selects Option A to protect gross margin.

---

### 🛈 Slide 6: 4. Communication Timing Optimizer
- **📌 WHAT**: 24-hour continuous Kernel Density Estimation (KDE) over individual customer session timestamps.
- **💡 WHY**: Maximizes notification open rates by sending push messages during each customer's peak active attention window.
- **⚙️ HOW (Algorithm & Formula)**:
  - `Hourly Density(Hour) = Average( Smooth Curve over User Session Timestamps )`
  - `Optimal Send Hour = Peak Hour Density Maximum`
- **🎯 REAL-WORLD EXAMPLE**: Shopper session log shows 15 visits concentrated between 7:00 PM and 9:00 PM. KDE identifies peak at 20:00 (8:00 PM). Sends notification Thursdays at 8:00 PM (**34.2% predicted open rate** vs 8.1% store average).

---

## 📊 Domain 02: Merchandising (Pricing, Assortment & Placement)

---

### 🛈 Slide 7: 5. Demand Forecasting
- **📌 WHAT**: Global XGBoost Regressor with 7d/14d/30d rolling lag features, moving averages, and holiday calendars.
- **💡 WHY**: Prevents stock-outs, optimizes supplier procurement orders, and lowers excess warehouse holding costs.
- **⚙️ HOW (Algorithm & Formula)**:
  - `Predicted Sales(Day t) = XGBoost(Lag_7, Lag_14, MovingAverage_7d, DayOfWeek, IsHoliday)`
- **🎯 REAL-WORLD EXAMPLE**: Classic White Shirt (`P00001`): 7-day lag = 14 units, 7-day average = 12.5 units/day, Saturday multiplier = 1.2x. Forecasts Day 1 Demand = **16 units** (30-day cumulative forecast = 485 units).

---

### 🛈 Slide 8: 6. Dynamic Pricing Engines
- **📌 WHAT**: Log-Log price elasticity linear regression under margin floor constraints and competitor benchmarking.
- **💡 WHY**: Maximizes net gross profit by marking down overstocked SKUs and marking up high-demand low-stock items.
- **⚙️ HOW (Algorithm & Formula)**:
  - `Price Elasticity = (% Change in Quantity Demanded) / (% Change in Price)`
  - `Price Multiplier = 1.0 + Inventory_Factor + Stockout_Risk_Factor + Velocity_Factor`
  - `Final Price = Keep price between (Cost + 5%) and (Current Price +/- 25%)`
- **🎯 REAL-WORLD EXAMPLE**: Product `P00010` (Cost = ₹700, Competitor Avg = ₹1,150). Inventory ratio = 85% (High stock). Applies -5.6% markdown to **₹1,085** (+13.3% projected volume lift).

---

### 🛈 Slide 9: 7. Promotion Optimization
- **📌 WHAT**: Non-linear discount uplift modeling combined with cross-category product cannibalization penalties.
- **💡 WHY**: Ensures promotional campaigns generate true net margin gain rather than loss-making volume surges.
- **⚙️ HOW (Algorithm & Formula)**:
  - `Net Profit = (Promoted Units * [Discounted Price - Cost]) - Cannibalization Loss`
- **🎯 REAL-WORLD EXAMPLE**: Summer Sale on Tops (Cost = ₹400): 25% Off increases sales +65% but causes ₹4,200 cannibalization loss (₹53,550 profit). Optimizer adjusts discount to **18% Off** (₹64,200 max net profit).

---

### 🛈 Slide 10: 8. Competitor Price Monitoring
- **📌 WHAT**: Interquartile Range (IQR) & Z-Score anomaly detection across external retailer catalog scrapers.
- **💡 WHY**: Detects uncompetitive price premiums or undercuts in real time to maintain market share.
- **⚙️ HOW (Algorithm & Formula)**:
  - `Price Index = (Our Price / Average Competitor Price) * 100`
  - `Z-Score = (Our Price - Competitor Average) / Competitor Standard Deviation`
- **🎯 REAL-WORLD EXAMPLE**: Running Shoes `P00044` priced at ₹4,500. Competitor average = ₹3,666 (Std Dev = ₹76). Z-Score = **+10.97** (Overprice anomaly, Price Index = 122.7). System flags for price alignment.

---

## 🚚 Domain 03: Operational Efficiency (Supply Chain & Fulfillment)

---

### 🛈 Slide 11: 9. Smart Inventory Health Monitoring
- **📌 WHAT**: Composite Inventory Risk Scoring (0 to 100) combining Days of Supply, Stockout Risk, & ABC/XYZ matrices.
- **💡 WHY**: Gives supply chain managers real-time visibility into high-risk SKUs requiring immediate reordering or markdown.
- **⚙️ HOW (Algorithm & Formula)**:
  - `Days of Supply = Current Stock / Daily Sales Rate`
  - `Health Score = 100 - [ (0.45 * Stockout Risk) + (0.35 * Overstock Risk) + (0.20 * Expiry Risk) ] * 100`
- **🎯 REAL-WORLD EXAMPLE**: SKU `P00088` has 12 units in stock and daily demand of 4 units/day (3.0 Days of Supply vs 5 days lead time). Stockout Risk = 0.88. Inventory Health Score = **60.4/100** (Critical Stockout Alert).

---

### 🛈 Slide 12: 10. Automated Replenishment
- **📌 WHAT**: Dynamic Safety Stock & Reorder Point (ROP) calculation under $(s, S)$ continuous review policy.
- **💡 WHY**: Automates purchase order generation, preventing stock-outs while maintaining optimal working capital.
- **⚙️ HOW (Algorithm & Formula)**:
  - `Safety Stock = Service_Z_Factor * Sqrt( [Lead Time * Demand Variance^2] + [Daily Demand^2 * Lead Time Variance^2] )`
  - `Reorder Point (ROP) = (Daily Demand * Lead Time) + Safety Stock`
- **🎯 REAL-WORLD EXAMPLE**: Product `P00005` (Daily Sales = 25, Lead Time = 6 days, 95% Service Level). Safety Stock = 45 units. ROP = 195 units. Current Stock = 140 units. System triggers PO for **160 units**.

---

### 🛈 Slide 13: 11. Warehouse Slotting Optimization
- **📌 WHAT**: Co-location pick frequency velocity clustering mapping Class A items close to dispatch docks.
- **💡 WHY**: Reduces warehouse picker walking distance by up to 70%, accelerating order packing turnaround.
- **⚙️ HOW (Algorithm & Formula)**:
  - `Pick Velocity = Sum( Quantity Picked across All Orders )`
  - `Distance Objective = Min Sum( Monthly Picks * Distance from Slot to Shipping Dock )`
- **🎯 REAL-WORLD EXAMPLE**: Stockholm Warehouse `W002`: Class A item `P00012` (1,420 picks/month) currently in Zone C (85m from dock). Optimized move to Zone A (12m from dock) saves **103.6 km** picker walking/month (**72% reduction in pick travel time**).

---

### 🛈 Slide 14: 12. Logistics, Route & Fleet Optimization
- **📌 WHAT**: Vehicle Routing Problem with Time Windows (VRPTW) solved via Haversine distance & 2-Opt local search.
- **💡 WHY**: Minimizes delivery fleet transport mileage, fuel costs, and carbon emissions for store delivery runs.
- **⚙️ HOW (Algorithm & Formula)**:
  - `Haversine Distance = Geodesic GPS Distance between Warehouse & Stores`
  - `2-Opt Edge Swap = Swap Route Segment if Distance(A->C) + Distance(B->D) < Distance(A->B) + Distance(C->D)`
- **🎯 REAL-WORLD EXAMPLE**: Stockholm Hub `W002` delivering to 7 stores. Initial unoptimized route = 3,637 km. 2-Opt route optimization restructures path to **3,361 km** (**276 km fuel saved, -7.6% transport cost**).

---

## 🎧 Domain 04: Customer Support (AI Resolution Layer)

---

### 🛈 Slide 15: 13. 24x7 AI Chatbots
- **📌 WHAT**: Multi-turn conversational state machine integrated with database order lookup & LLM fallback.
- **💡 WHY**: Provides instant 24x7 order tracking and returns resolution, deflecting support ticket volume.
- **⚙️ HOW (Algorithm & Formula)**:
  - `State Machine = (Intent Recognition, Order Entity Extraction, Live DB Query)`
- **🎯 REAL-WORLD EXAMPLE**: Customer Query *"Where is my order ORD99281?"*. Bot extracts Order ID `ORD99281`, queries database (Shipped via Express, Tracking #TRK8821), and generates instant 24x7 response.

---

### 🛈 Slide 16: 14. Intelligent Ticket Triage
- **📌 WHAT**: Multi-class text classification & sentiment analysis routing tickets to specialized agent queues.
- **💡 WHY**: Ensures urgent customer complaints (damaged goods, missing refunds) are prioritized immediately.
- **⚙️ HOW (Algorithm & Formula)**:
  - `Urgency Score = (0.40 * Negative Sentiment) + (0.30 * Urgent Keyword Flag) + (0.30 * VIP Flag)`
- **🎯 REAL-WORLD EXAMPLE**: Ticket *"Parcel arrived damaged, box torn and jacket missing. Want immediate refund!"*. Sentiment = -0.85. Urgency Score = **0.94/1.0** (HIGH Priority). Automatically routed to **Claims Escalation Queue**.

---

### 🛈 Slide 17: 15. Agent Assist
- **📌 WHAT**: Real-time BM25 search & vector similarity policy lookup across standard operating procedures (SOPs).
- **💡 WHY**: Empowers human support agents during live calls with instant verified answer templates.
- **⚙️ HOW (Algorithm & Formula)**:
  - `BM25 Relevance Score = Sum( Word Rarity * [ Word Frequency / (Word Frequency + Length Penalty) ] )`
- **🎯 REAL-WORLD EXAMPLE**: Support Query *"What is return policy for worn footwear?"*. BM25 search retrieves SOP Section 4.2 (Relevance Score = 14.8). System auto-fills template for agent: *"Footwear returnable in 30 days..."*

---

### 🛈 Slide 18: 16. Voice of Customer (VoC) Aspect Mining
- **📌 WHAT**: Aspect-Based Sentiment Analysis (ABSA) mining customer reviews for specific product feedback.
- **💡 WHY**: Provides product engineering & sourcing teams with granular quality defect alerts.
- **⚙️ HOW (Algorithm & Formula)**:
  - `Net Aspect Polarity = (Positive Mentions - Negative Mentions) / Total Mentions for Aspect`
- **🎯 REAL-WORLD EXAMPLE**: 1,250 reviews for Dresses category: Aspect 'Fit' = +0.68 Net Score (Positive). Aspect 'Zipper' = **-0.73 Net Score** (Negative Defect Alert). Generates quality alert to replace zipper vendor.

---

## 🧪 Slide 19: Verification & Testing Benchmarks
- **Automated Pytest Suite**: `tests/test_capabilities.py` (32 / 32 test cases passed cleanly in 24 seconds).
- **Response Latency**: Sub-100ms API response time across all 16 capability endpoints.
- **Memory Footprint**: Bounded strictly under 1 GB RAM via PyArrow string downcasting and `CALRETAIL_TABLE_CACHE=2`.

---

## 🚀 Slide 20: Production Deployment & AWS EC2 Specifications
- **Cloud Platform**: AWS EC2 Ubuntu 24.04 LTS (`eu-north-1` Stockholm Region).
- **Instance Hardware**: `t3.small` instance (2 GB RAM, 2 VCPUs) with 2 GB Swap virtual memory protection.
- **Containerization**: Docker multi-stage container (`python:3.11-slim` with `libgomp1` OpenMP).
- **Process Supervision**: Automatic `uvicorn` watchdog restart loop in `start.sh`.
- **Live Web App**: `http://13.50.101.137`
- **GitHub Repository**: `https://github.com/calritik/Calretail_.git`
