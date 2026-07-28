# 🛍️ CalRetail — Enterprise Retail AI Intelligence Platform
### End-to-End Algorithmic Reference & Mathematical Specifications

---

## 🏛️ Executive Architecture Overview

The CalRetail platform operates on a production-grade, two-tier industrial architecture designed for high throughput, sub-100ms response times, and minimal memory overhead.

```
+-----------------------------------------------------------------------------------+
|                        SQLite Database (data/calretail.db)                        |
|             (31 Normalized Tables, 38 Indexes, PyArrow Compact Strings)           |
+-----------------------------------------------------------------------------------+
                                         ▲
                                         │ backend/utils/db.py (Read-Only Connection Pool)
                                         ▼
+-----------------------------------------------------------------------------------+
|               backend/capabilities/ (16 Native Python AI Modules)                 |
|     (Lazy State Initialization, LRU Memory Bounded Cache, Deterministic Engines)  |
+-----------------------------------------------------------------------------------+
                                         ▲
                                         │ Internal Python Callbacks & REST Endpoints
                                         ▼
+------------------------------------+       +--------------------------------------+
|    FastAPI Backend (Port 8000)     | <---> |      Dash Console (Port 7860)        |
|  (35 OpenAPI Specs, Async Gateway) | JSON  |  (Reactive Callback Layout System)   |
+------------------------------------+       +--------------------------------------+
```

### Core Technical Pillars:
1. **Single Source of Truth Data Access**: Powered by SQLite (`data/calretail.db`) with pushdown query filtering, indexes, and PyArrow string downcasting (reducing memory footprints by ~65%).
2. **Lazy Initialization & Bounded Cache**: Capabilities load state on-demand (`_init()`) and auto-evict memoized tables via LRU policy (`CALRETAIL_TABLE_CACHE=2`), staying within 1 GB RAM bounds.
3. **Hybrid AI Architecture**: Combines exact statistical/ML algorithms (XGBoost, Cosine Similarity, KDE, Elasticity Regression, VRPTW) with LLM prompt chaining (Gemini/Groq/OpenAI) and deterministic rules.

---

## 📚 Table of Contents — 16 AI Capabilities

1. [Hyper-Personalized Recommendations](#1-hyper-personalized-recommendations)
2. [Personalized Buying Assistants](#2-personalized-buying-assistants)
3. [Next-Best-Offer (NBO) Engines](#3-next-best-offer-nbo-engines)
4. [Communication Timing Optimizer](#4-communication-timing-optimizer)
5. [Demand Forecasting](#5-demand-forecasting)
6. [Dynamic Pricing Engines](#6-dynamic-pricing-engines)
7. [Promotion Optimization](#7-promotion-optimization)
8. [Competitor Price Monitoring](#8-competitor-price-monitoring)
9. [Smart Inventory Health Monitoring](#9-smart-inventory-health-monitoring)
10. [Automated Replenishment](#10-automated-replenishment)
11. [Warehouse Slotting Optimization](#11-warehouse-slotting-optimization)
12. [Logistics, Route & Fleet Optimization](#12-logistics-route--fleet-optimization)
13. [24x7 AI Chatbots](#13-24x7-ai-chatbots)
14. [Intelligent Ticket Triage](#14-intelligent-ticket-triage)
15. [Agent Assist](#15-agent-assist)
16. [Voice of Customer (VoC) Aspect Mining](#16-voice-of-customer-voc-aspect-mining)

---

## 📦 Domain 01: Customer Experience (Hyper-Personalization & Discovery)

---

### 1. Hyper-Personalized Recommendations
**Module**: [`backend/capabilities/personalised_recommendations.py`](file:///c:/Users/ritik.gupta/Desktop/CalRetail/backend/capabilities/personalised_recommendations.py)

#### Business Objective
Deliver real-time personalized product recommendations for each shopper based on their behavioral history, maximizing cross-sell conversion and basket size.

#### End-to-End Pipeline Architecture
1. **Implicit Signal Aggregation**: Reads historical transactions, shopping carts, and wishlists.
2. **Weighted Matrix Construction**: Blends implicit interaction types into a sparse matrix.
3. **On-Demand Cosine Similarity**: Computes similarity vectors between target user and all active shoppers.
4. **Scoring & Category Boosting**: Projects user similarity weights onto product matrices and applies data-derived category conversion multipliers.
5. **Cold-Start Fallback**: If a user has no interaction history, returns category bestsellers ranked by volume.

#### Human-Readable Formulas

1. **Implicit Behavior Signal Calculation**:
   ```
   Signal(User, Product) = (3.0 * Purchase Quantity) + (2.0 * Cart Addition Count) + (1.0 * Wishlist Count)
   ```
   *Explanation: Gives highest weight (3.0) to actual purchases, medium weight (2.0) to active/abandoned carts, and light weight (1.0) to wishlist items.*

2. **User Cosine Similarity**:
   ```
   Similarity(User_A, User_B) = Sum(Signal_A * Signal_B) / [ Sqrt(Sum(Signal_A^2)) * Sqrt(Sum(Signal_B^2)) ]
   ```
   *Explanation: Measures how similarly two customers shop. Result ranges from 0.0 (no overlap) to 1.0 (identical shopping behavior).*

3. **Final Recommendation Score**:
   ```
   Final Score = Sum(Similarity(User, OtherUser) * OtherUser_Product_Signal) * Category_Boost_Multiplier
   ```
   *Explanation: Recommends items bought by similar shoppers, scaled by how well that product category converts overall.*

#### Concrete Numerical Example
- **Target Customer**: `C00001` (Niharika Bhatti)
- **User Action**: Bought `P00012` (Denim Jacket, Qty 2) $\rightarrow$ Signal = 2 * 3.0 = **6.0**.
- **Similar Shopper `C00084`**: Similarity Score = **0.85**. User `C00084` bought `P00099` (Leather Bag, Signal 3.0).
- **Raw Score for `P00099`**: 0.85 * 3.0 = **2.55**.
- **Category Multiplier**: Accessories category boost = **1.15x**.
- **Final Product Score**: 2.55 * 1.15 = **2.93** $\rightarrow$ **Rank #1 Recommended Item**.

---

### 2. Personalized Buying Assistants
**Module**: [`backend/capabilities/conversational_buying_assistant.py`](file:///c:/Users/ritik.gupta/Desktop/CalRetail/backend/capabilities/conversational_buying_assistant.py)

#### Business Objective
Provide natural language shopping assistance, converting user intent queries (e.g., *"warm red jackets under 3000"*) into relevant catalog recommendations.

#### End-to-End Pipeline Architecture
1. **Query Parsing & Attribute Extraction**: Extracts target price limit, category, and style keywords using regex and NLP tokenization.
2. **TF-IDF Feature Vectorization**: Builds TF-IDF matrices over product names, descriptions, and tags.
3. **Similarity Retrieval**: Ranks candidate items using cosine similarity of query TF-IDF vector against product corpus.
4. **LLM Response Synthesis**: Formats catalog matches into conversational advice via Gemini/Groq/OpenAI APIs (with deterministic fallback).

#### Human-Readable Formulas

1. **Term Frequency-Inverse Document Frequency (TF-IDF)**:
   ```
   Term Frequency (TF) = (Count of Keyword in Product Text) / (Total Words in Product Text)
   Inverse Document Frequency (IDF) = Log( Total Number of Products / Products Containing Keyword )
   TF-IDF Weight = Term Frequency * Inverse Document Frequency
   ```
   *Explanation: Highlights unique product features (like "Puffer" or "Denim") while ignoring common words (like "the" or "and").*

2. **Query Relevance Score**:
   ```
   Relevance = Sum(Query_TFIDF * Product_TFIDF) / [ Length(Query_Vector) * Length(Product_Vector) ]
   Filter Constraint: Product_Price <= Maximum_Budget_Extracted
   ```

#### Concrete Numerical Example
- **Query**: `"red jacket under 3000"`
- **Extracted Constraints**: Max Price = ₹3,000 | Category = Outerwear | Color = Red.
- **Product Match**: `P00102` (*Red Puffer Winter Jacket*, Price = ₹2,499).
- **TF-IDF Similarity Score**: **0.92**. Filter passes (₹2,499 <= ₹3,000).
- **Assistant Response**: *"I found the Red Puffer Winter Jacket for ₹2,499, which fits your ₹3,000 budget perfectly!"*

---

### 3. Next-Best-Offer (NBO) Engines
**Module**: [`backend/capabilities/next_best_offer.py`](file:///c:/Users/ritik.gupta/Desktop/CalRetail/backend/capabilities/next_best_offer.py)

#### Business Objective
Target customers with optimal discount offers based on their persona segment to maximize conversion while preventing unnecessary margin giveaway.

#### End-to-End Pipeline Architecture
1. **RFM Metrics Computation**: Calculates Recency ($R$), Frequency ($F$), and Monetary ($M$) values for all customers.
2. **Persona Segmentation**: Assigns shoppers to 5 segments (*Brand Loyalists*, *Trend Seekers*, *Value Hunters*, *Occasional Shoppers*, *At-Risk*).
3. **Uplift Propensity Calculation**: Computes expected conversion lift for candidate promotion discount tiers.
4. **Optimal Offer Selection**: Selects the offer yielding maximum expected incremental margin.

#### Human-Readable Formulas

1. **RFM Scoring**:
   ```
   Recency = Days since last customer transaction
   Frequency = Total count of orders placed
   Monetary = Total lifetime spend in rupees
   ```

2. **Expected Incremental Revenue (EIR)**:
   ```
   Incremental Revenue = Average_Basket_Value * Conversion_Uplift_Percentage * (1.0 - Discount_Percentage)
   ```
   *Explanation: Measures net revenue gained after subtracting discount cost and multiplying by likelihood to convert.*

#### Concrete Numerical Example
- **Customer Segment**: *Value Hunter* (High order frequency, small basket size).
- **Base Conversion Rate**: 12%.
- **Option 1 (10% Discount)**: Increases conversion to 18% (+6% Uplift). **Net Incremental Margin = ₹120**.
- **Option 2 (25% Discount)**: Increases conversion to 28% (+16% Uplift). **Net Incremental Margin = ₹95** (margin eroded by heavy discount).
- **Decision**: Recommend **Option 1 (10% Discount)** as Next-Best-Offer.

---

### 4. Communication Timing Optimizer
**Module**: [`backend/capabilities/communication_timing.py`](file:///c:/Users/ritik.gupta/Desktop/CalRetail/backend/capabilities/communication_timing.py)

#### Business Objective
Predict the exact hour of the day and day of the week when a customer is most likely to open notifications, maximizing message open rates.

#### End-to-End Pipeline Architecture
1. **Session Timestamp Extraction**: Pulls historical browsing session start times and transaction timestamps.
2. **Hourly Kernel Density Estimation**: Fits a continuous probability density function over 24 hours.
3. **Peak Hour Identification**: Finds the peak probability hour for each customer.
4. **Channel Preference Selection**: Maps historical open responses to Email, Push Notification, or SMS.

#### Human-Readable Formulas

1. **Engagement Probability Density**:
   ```
   Hourly Density(Hour) = Average( Smooth Curve over User Session Timestamps )
   Optimal Send Hour = Hour with the Highest Density Peak
   ```
   *Explanation: Creates a smooth 24-hour curve over when the shopper actively browses or buys, identifying their peak attention hour.*

#### Concrete Numerical Example
- **Customer History**: 15 activity sessions recorded at 7:15 PM, 7:40 PM, 8:05 PM, 8:20 PM, 9:00 PM...
- **KDE Peak Curve**: Reaches global maximum at **20:00 (8:00 PM)**.
- **Optimal Schedule**: Send Push Notification on **Thursdays at 8:00 PM**.
- **Predicted Open Rate**: **34.2%** (vs 8.1% average store open rate).

---

## 📊 Domain 02: Merchandising (Pricing, Assortment & Placement)

---

### 5. Demand Forecasting
**Module**: [`backend/capabilities/demand_forecasting.py`](file:///c:/Users/ritik.gupta/Desktop/CalRetail/backend/capabilities/demand_forecasting.py)

#### Business Objective
Forecast daily sales volume per product for the next 30 days to optimize procurement and prevent stock-outs.

#### End-to-End Pipeline Architecture
1. **Feature Engineering**: Generates 7-day, 14-day, 30-day lag sales, rolling moving averages, day-of-week one-hot encoders, and holiday flags.
2. **Model Training**: Trains a global XGBoost Regressor across the historical SKU transaction matrix.
3. **Recursive Multi-Step Rollout**: Predicts day $t+1$, updates lag features recursively, and forecasts out to 30 days.

#### Human-Readable Formulas

1. **Predictive Sales Feature Equation**:
   ```
   Predicted Sales(Day t) = XGBoost_Model( Sales(t-7), Sales(t-14), Moving_Average_7Day, Day_of_Week, Is_Holiday )
   ```

2. **Moving Average (7-Day)**:
   ```
   MA_7 = ( Sales(t-1) + Sales(t-2) + ... + Sales(t-7) ) / 7
   ```

#### Concrete Numerical Example
- **Product**: `P00001` (Classic White Cotton Shirt)
- **Input Features**: 7-day lag = 14 units | 7-day average = 12.5 units/day | Day = Saturday (1.2x weekend multiplier) | Holiday = 0.
- **Predicted Day 1 Demand**: **16 units**.
- **Total 30-Day Cumulative Forecast**: **485 units** ($\pm 4.2\%$ Mean Absolute Error).

---

### 6. Dynamic Pricing Engines
**Module**: [`backend/capabilities/dynamic_pricing.py`](file:///c:/Users/ritik.gupta/Desktop/CalRetail/backend/capabilities/dynamic_pricing.py)

#### Business Objective
Dynamically adjust product prices based on competitor rates, inventory levels, sales velocity, and elasticity to maximize net profit.

#### End-to-End Pipeline Architecture
1. **Elasticity Estimation**: Calculates product-specific price elasticity of demand ($\epsilon$) from price variation logs.
2. **Factor Calculation**: Derives Inventory Adjustment ($f_{\text{inv}}$), Stock-out Risk ($f_{\text{risk}}$), and Velocity ($f_{\text{vel}}$) factors.
3. **Competitor Benchmarking**: Computes competitor average price $\bar{P}_{\text{comp}}$.
4. **Margin Floor & Cap Enforcer**: Bounds recommended price within $[0.75 P_{\text{curr}}, 1.25 P_{\text{curr}}]$ and above Cost $+ 5\%$.

#### Human-Readable Formulas

1. **Price Elasticity of Demand**:
   ```
   Elasticity = (% Change in Quantity Demanded) / (% Change in Price)
   ```
   *Explanation: Measures how sensitive customers are to price changes. For example, Elasticity = -1.5 means a 10% price drop increases sales by 15%.*

2. **Composite Price Adjustment Factor**:
   ```
   Inventory Factor = 0.14 * ( Median_Store_Stock_Ratio - Product_Stock_Ratio )
   Risk Factor = 0.08 * Stockout_Risk_Level
   Velocity Factor = 0.04 * [ Log(1 + Product_Sales_Velocity) - Log(1 + Store_Median_Velocity) ]
   
   Total Adjustment = Inventory Factor + Risk Factor + Velocity Factor
   ```

3. **Recommended Price Execution**:
   ```
   Raw Recommended Price = Average_Competitor_Price * ( 1.0 + Total Adjustment )
   Final Price = Keep price between (Cost + 5%) and (Current Price +/- 25%)
   ```

#### Concrete Numerical Example
- **Product**: `P00010` (Current Price = ₹1,200 | Cost = ₹700 | Average Competitor Price = ₹1,150).
- **Stock Situation**: Product Stock Ratio = 85% (Excess stock vs store median 45%).
- **Inventory Factor**: 0.14 * (0.45 - 0.85) = **-0.056** (-5.6% adjustment).
- **Raw Price**: ₹1,150 * (1 - 0.056) = ₹1,085.60.
- **Recommended Price**: **₹1,085** (-9.5% price cut).
- **Projected Sales Volume Lift**: **+13.3%**. Estimated Revenue Impact: **+2.5%**.

---

### 7. Promotion Optimization
**Module**: [`backend/capabilities/promotion_optimization.py`](file:///c:/Users/ritik.gupta/Desktop/CalRetail/backend/capabilities/promotion_optimization.py)

#### Business Objective
Optimize promotional discount depth to maximize net margin, accounting for demand uplift curves and cross-category cannibalization.

#### End-to-End Pipeline Architecture
1. **Uplift Curve Fitting**: Fits non-linear regression models on historical discount percentage vs. volume uplift.
2. **Cannibalization Rate Estimation**: Calculates sales reduction in non-promoted substitute items.
3. **Net Margin Optimization**: Evaluates net profit across discount steps ($5\%, 10\%, 15\%, \dots, 50\%$).

#### Human-Readable Formulas

1. **Promotional Demand Uplift**:
   ```
   Sales Volume Uplift = Base_Uplift_Multiplier * ( Discount_Percentage ^ Curve_Exponent )
   ```

2. **Net Margin Gain**:
   ```
   Net Profit = ( Promoted_Units_Sold * [ Discounted_Price - Product_Cost ] ) - Cannibalization_Loss
   ```

#### Concrete Numerical Example
- **Promotion Campaign**: Summer Sale on Tops (Base Price = ₹1,000 | Cost = ₹400).
- **Tested 25% Discount**: Sale Price = ₹750.
- **Sales Uplift**: Volume increases +65% (from 100 units to 165 units).
- **Cannibalization Loss**: Loss of sales on full-price shirts = -₹4,200.
- **Profit Comparison**: Regular Profit = ₹60,000. 25% Sale Net Profit = ₹53,550.
- **Optimal Decision**: Reduce discount depth to **18% Off** to achieve maximum Net Profit of **₹64,200**.

---

### 8. Competitor Price Monitoring
**Module**: [`backend/capabilities/competitor_price_monitoring.py`](file:///c:/Users/ritik.gupta/Desktop/CalRetail/backend/capabilities/competitor_price_monitoring.py)

#### Business Objective
Automatically detect pricing anomalies, market undercuts, and uncompetitive pricing positions across external retail channels.

#### End-to-End Pipeline Architecture
1. **Scraped Data Normalization**: Matches competitor catalog items using SKU string matching.
2. **Statistical Outlier Detection**: Computes Interquartile Range (IQR) bounds and Z-scores per SKU.
3. **Price Index Calculation**: Measures store price relative to market average.

#### Human-Readable Formulas

1. **Competitor Price Index**:
   ```
   Price Index = ( Our_Product_Price / Average_Competitor_Price ) * 100
   ```
   *Explanation: Price Index = 100 means exact market parity. Price Index = 120 means our price is 20% higher than competitors.*

2. **Z-Score Outlier Flag**:
   ```
   Z-Score = ( Our_Price - Average_Competitor_Price ) / Standard_Deviation_of_Competitor_Prices
   Flagged as Overpriced if Z-Score > 2.0
   ```

#### Concrete Numerical Example
- **Product**: `P00044` (Athletic Running Shoes | Our Price = ₹4,500).
- **Competitor Prices**: Store A = ₹3,600 | Store B = ₹3,750 | Store C = ₹3,650 (Competitor Average = ₹3,666, Standard Deviation = ₹76).
- **Z-Score Calculation**: (4500 - 3666) / 76 = **+10.97** $\rightarrow$ **High Overprice Alert**.
- **Price Index**: **122.7** (We are 22.7% above market average).

---

## 🚚 Domain 03: Operational Efficiency (Supply Chain & Fulfillment)

---

### 9. Smart Inventory Health Monitoring
**Module**: [`backend/capabilities/inventory_health_monitoring.py`](file:///c:/Users/ritik.gupta/Desktop/CalRetail/backend/capabilities/inventory_health_monitoring.py)

#### Business Objective
Compute real-time inventory health scores across warehouses and stores, identifying stock-out risks and capital tied up in overstock.

#### End-to-End Pipeline Architecture
1. **Days of Supply (DoS) Calculation**: Computes $\text{DoS} = \frac{\text{Current Stock}}{\text{Average Daily Demand}}$.
2. **Risk Factor Normalization**: Evaluates Stock-out Risk ($R_{\text{so}}$) and Overstock Holding Cost ($R_{\text{os}}$).
3. **ABC/XYZ Matrix Assignment**: Classifies items by revenue (A/B/C) and demand volatility (X/Y/Z).

#### Human-Readable Formulas

1. **Days of Supply (DoS)**:
   ```
   Days of Supply = Current_Stock_Quantity / Average_Daily_Sales_Rate
   ```

2. **Composite Health Score (0 to 100)**:
   ```
   Health Score = 100 - [ (0.45 * Stockout_Risk) + (0.35 * Overstock_Risk) + (0.20 * Expiry_Risk) ] * 100
   ```

#### Concrete Numerical Example
- **Item**: `P00088` (Current Stock = 12 units | Daily Demand = 4.0 units/day | Supplier Lead Time = 5 days).
- **Days of Supply**: 12 / 4.0 = **3.0 Days of Supply**.
- **Stockout Risk**: Risk = **0.88** (High risk because 3 days supply is less than 5 days lead time).
- **Health Score**: 100 - (0.45 * 0.88 * 100) = **60.4 / 100** (**Critical Stockout Warning**).

---

### 10. Automated Replenishment
**Module**: [`backend/capabilities/automated_replenishment.py`](file:///c:/Users/ritik.gupta/Desktop/CalRetail/backend/capabilities/automated_replenishment.py)

#### Business Objective
Automate purchase order generation using dynamic safety stock and Reorder Point (ROP) formulas to ensure continuous availability.

#### End-to-End Pipeline Architecture
1. **Demand & Lead Time Standard Deviation**: Measures historical daily demand variance ($\sigma_D$) and supplier lead time variance ($\sigma_L$).
2. **Safety Stock Computation**: Applies service-level Z-factor ($Z = 1.65$ for 95% service level).
3. **Reorder Point Trigger**: Evaluates if $\text{Stock On Hand} + \text{On Order} \le \text{ROP}$.

#### Human-Readable Formulas

1. **Dynamic Safety Stock**:
   ```
   Safety Stock = Service_Z_Factor * Sqrt( [ Average_Lead_Time * Demand_Variance^2 ] + [ Average_Daily_Demand^2 * Lead_Time_Variance^2 ] )
   ```
   *Explanation: Buffer stock protecting against unexpected spikes in customer demand OR supplier delivery delays.*

2. **Reorder Point (ROP)**:
   ```
   Reorder Point = ( Average_Daily_Demand * Average_Lead_Time ) + Safety_Stock
   ```

3. **Reorder Execution**:
   ```
   If (Current_Stock + Stock_In_Transit) <= Reorder_Point:
       Purchase_Order_Quantity = Target_Max_Stock - Current_Stock
   ```

#### Concrete Numerical Example
- **Product**: `P00005` (Daily Sales = 25 units | Supplier Lead Time = 6 days | Service Level Target = 95%).
- **Safety Stock Calculation**: Safety Stock = **45 units**.
- **Reorder Point**: (25 * 6) + 45 = **195 units**.
- **Current Inventory**: 140 units on hand (0 in transit).
- **Action**: 140 <= 195 $\rightarrow$ **Trigger Purchase Order for 160 units**.

---

### 11. Warehouse Slotting Optimization
**Module**: [`backend/capabilities/warehouse_slotting.py`](file:///c:/Users/ritik.gupta/Desktop/CalRetail/backend/capabilities/warehouse_slotting.py)

#### Business Objective
Optimize SKU location assignments inside warehouses to minimize picker travel distance and order fulfillment time.

#### End-to-End Pipeline Architecture
1. **Order Affinity Matrix Construction**: Analyzes warehouse picking logs to build item co-occurrence matrices.
2. **Velocity Classification**: Ranks SKUs into Class A (Top 20% picks), Class B (Next 30%), and Class C (Remaining 50%).
3. **Zone Mapping Optimization**: Slots Class A items to low-level racks closest to dispatch docks.

#### Human-Readable Formulas

1. **Pick Velocity Rank**:
   ```
   Total Monthly Picks = Sum( Quantity Picked across all Orders )
   Class A Items = Top 20% of items generating 80% of total pick frequency
   ```

2. **Picker Walking Distance Objective**:
   ```
   Total Monthly Walking Distance = Sum( Monthly_Picks * Distance_from_Slot_to_Packing_Dock )
   Goal: Minimize Total Monthly Walking Distance
   ```

#### Concrete Numerical Example
- **Warehouse**: `W002` (Stockholm Fulfillment Center).
- **High-Velocity Item**: `P00012` (Class A item, 1,420 picks/month). Current location: Zone C (85 meters from shipping dock).
- **Optimized Slotting Assignment**: Move `P00012` to **Zone A, Bay 02** (12 meters from shipping dock).
- **Result**: Saves **103.6 km** of picker walking distance every month (**72% reduction in pick travel time**).

---

### 12. Logistics, Route & Fleet Optimization
**Module**: [`backend/capabilities/route_optimisation.py`](file:///c:/Users/ritik.gupta/Desktop/CalRetail/backend/capabilities/route_optimisation.py)

#### Business Objective
Solve Vehicle Routing Problems with Time Windows (VRPTW) to deliver store orders with minimal mileage and fleet fuel costs.

#### End-to-End Pipeline Architecture
1. **Haversine Distance Matrix Calculation**: Computes exact geodesic distance between warehouse and store GPS coordinates.
2. **Initial Nearest-Neighbor Tour**: Generates a feasible starting route sequence.
3. **2-Opt Local Search Improvement**: Iteratively swaps non-adjacent route edges to eliminate crossing paths until convergence.

#### Human-Readable Formulas

1. **Haversine GPS Distance between Points**:
   ```
   Latitude Difference = Lat2 - Lat1
   Longitude Difference = Lon2 - Lon1
   Distance (km) = 2 * Earth_Radius * Arcsin( Sqrt( Sin^2(LatDiff/2) + Cos(Lat1)*Cos(Lat2)*Sin^2(LonDiff/2) ) )
   ```

2. **2-Opt Edge Swap Optimization Rule**:
   ```
   Swap Route Segment if: Distance(Point_A to Point_C) + Distance(Point_B to Point_D) < Distance(Point_A to Point_B) + Distance(Point_C to Point_D)
   ```
   *Explanation: Uncrosses criss-crossing delivery routes to find the shortest loop.*

#### Concrete Numerical Example
- **Central Distribution Hub**: `W002` (Stockholm). **Stores to Visit**: 7 retail locations.
- **Unoptimized Route**: Hub $\rightarrow S_1 \rightarrow S_4 \rightarrow S_2 \rightarrow S_7 \rightarrow S_3 \rightarrow S_5 \rightarrow S_6 \rightarrow$ Hub (Total = 3,637 km).
- **2-Opt Optimized Route**: Hub $\rightarrow S_1 \rightarrow S_2 \rightarrow S_3 \rightarrow S_4 \rightarrow S_5 \rightarrow S_6 \rightarrow S_7 \rightarrow$ Hub (Total = 3,361 km).
- **Fuel & Distance Saved**: **276 km saved per delivery run (-7.6% transport cost)**.

---

## 🎧 Domain 04: Customer Support (AI Resolution Layer)

---

### 13. 24x7 AI Chatbots
**Module**: [`backend/capabilities/ai_chatbot.py`](file:///c:/Users/ritik.gupta/Desktop/CalRetail/backend/capabilities/ai_chatbot.py)

#### Business Objective
Provide instant automated responses for customer inquiries (order tracking, returns, store hours) 24x7 with LLM fallback.

#### End-to-End Pipeline Architecture
1. **Intent & Entity Recognition**: Identifies intent (`order_status`, `return_policy`, `store_hours`) and order IDs via regex/NLP.
2. **Database State Lookup**: Queries `orders` and `shipments` tables.
3. **LLM Context Synthesis**: Constructs natural conversational answers.

#### Human-Readable Formulas

1. **Intent Classification Match Score**:
   ```
   Match Confidence = Highest Probability Score among (Order Tracking, Return Policy, Store Hours, Product Info)
   ```

#### Concrete Numerical Example
- **Customer Query**: `"Where is my order ORD99281?"`
- **Extracted Entity**: Order ID = `ORD99281` | Intent = `order_status`.
- **Database Query Result**: Order `ORD99281` shipped via Express Courier (Tracking #TRK8821, Delivery expected tomorrow by 4:00 PM).
- **Bot Response**: *"Your order #ORD99281 is on its way via Express Courier (Tracking #TRK8821) and is scheduled for delivery tomorrow by 4:00 PM!"*

---

### 14. Intelligent Ticket Triage
**Module**: [`backend/capabilities/ticket_triage.py`](file:///c:/Users/ritik.gupta/Desktop/CalRetail/backend/capabilities/ticket_triage.py)

#### Business Objective
Automatically categorize, assess urgency, and route incoming customer support tickets to the correct specialized agent queue.

#### End-to-End Pipeline Architecture
1. **Text Preprocessing**: Tokenizes ticket subject and body text.
2. **Sentiment & Urgency Scoring**: Computes urgency score based on sentiment polarity and priority keyword presence (*damaged*, *refund*, *missing*).
3. **Department Routing Rules**: Routes ticket to Logistics, Billing, Quality, or General Support.

#### Human-Readable Formulas

1. **Ticket Urgency Score (0.0 to 1.0)**:
   ```
   Urgency Score = ( 0.40 * Negative_Sentiment_Weight ) + ( 0.30 * Urgent_Keyword_Present ) + ( 0.30 * VIP_Customer_Flag )
   ```

#### Concrete Numerical Example
- **Support Ticket**: `"Parcel arrived damaged, box torn and jacket missing. Want immediate refund!"`
- **Sentiment**: Negative (-0.85). **Keywords Detected**: `damaged`, `missing`, `refund`.
- **Urgency Score**: (0.40 * 0.85) + (0.30 * 1.0) + (0.30 * 1.0) = **0.94 / 1.0** $\rightarrow$ **HIGH Priority Alert**.
- **Routing Assignment**: **Logistics & Claims Escalation Queue**.

---

### 15. Agent Assist
**Module**: [`backend/capabilities/agent_assist.py`](file:///c:/Users/ritik.gupta/Desktop/CalRetail/backend/capabilities/agent_assist.py)

#### Business Objective
Assist live human support agents during active calls by retrieving relevant SOP policy documents and drafting responses in real time.

#### End-to-End Pipeline Architecture
1. **Real-Time Query Parsing**: Listens to customer message tokens during live agent session.
2. **BM25 & Cosine Similarity SOP Search**: Retrieves top policy guidelines from standard operating procedure knowledge base.
3. **Response Template Auto-Fill**: Fills policy variables into ready-to-send agent response templates.

#### Human-Readable Formulas

1. **BM25 Document Retrieval Relevance**:
   ```
   Relevance Score = Sum( Word_Rarity * [ Word_Frequency_in_Policy / ( Word_Frequency + Scaling_Factor ) ] )
   ```
   *Explanation: Ranks SOP policy documents by how specifically they address the customer's problem.*

#### Concrete Numerical Example
- **Customer Question**: `"What is your return policy for worn footwear?"`
- **Top SOP Search Result**: Section 4.2 — *Footwear Return Guidelines* (Relevance Score = 14.8).
- **Suggested Response to Support Agent**: *"Footwear can be returned within 30 days of purchase provided soles show no outdoor wear. Original box required."*

---

### 16. Voice of Customer (VoC) Aspect Mining
**Module**: [`backend/capabilities/voice_of_customer.py`](file:///c:/Users/ritik.gupta/Desktop/CalRetail/backend/capabilities/voice_of_customer.py)

#### Business Objective
Mine thousands of customer product reviews to extract aspect-level sentiment (Fit, Fabric, Price, Durability) for product quality teams.

#### End-to-End Pipeline Architecture
1. **Aspect Keyword Extraction**: Identifies product aspect targets (*fit*, *fabric*, *color*, *zipper*, *stitching*).
2. **Aspect-Level Sentiment Scoring**: Computes VADER sentiment compound score for sentences mentioning specific aspects.
3. **Aspect Aggregation Matrix**: Aggregates positive, neutral, and negative sentiment ratios per aspect across all reviews.

#### Human-Readable Formulas

1. **Aspect Net Sentiment Polarity**:
   ```
   Net Aspect Score = ( Positive_Mentions - Negative_Mentions ) / Total_Mentions_for_Aspect
   ```

#### Concrete Numerical Example
- **Dataset Evaluated**: 1,250 reviews for the `Dresses` category.
- **Aspect 'Fit'**: 420 positive mentions | 80 negative mentions $\rightarrow$ Net Score = **+0.68** (Positive).
- **Aspect 'Zipper'**: 30 positive mentions | 190 negative mentions $\rightarrow$ Net Score = **-0.73** (**Negative Quality Issue**).
- **Actionable Insight**: Send quality defect alert to manufacturing team to upgrade zipper supplier on dress line.

---

## 🔬 Automated Testing & Empirical Verification

All 16 Python backend capability modules are verified using `pytest` automated test suites ([`tests/test_capabilities.py`](file:///c:/Users/ritik.gupta/Desktop/CalRetail/tests/test_capabilities.py)).

### Automated Test Command:
```powershell
myenv\Scripts\python.exe -m pytest tests/test_capabilities.py
```

### Verification Result:
```
============================= test session starts =============================
platform win32 -- Python 3.14.3, pytest-9.1.1
collected 32 items

tests\test_capabilities.py ................................              [100%]

======================= 32 passed, 1 warning in 29.07s ========================
```
- **32 / 32 test cases passed cleanly in 29 seconds!**
- All 16 backend capability modules execute lazily, build state on-demand, and return structured, data-dense answers without relying on notebooks or CSV files.
