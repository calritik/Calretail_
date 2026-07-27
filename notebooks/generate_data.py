"""
Nexalyze — Synthetic Data Generation Script
Generates all 25 datasets for the Enterprise Retail AI Intelligence Platform.
Run via: python generate_data.py  OR  from Synthetic_Data_Generation.ipynb
"""

import random
import math
import os
import warnings
from pathlib import Path
from datetime import datetime, timedelta, date

import numpy as np
import pandas as pd
from faker import Faker

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

fake = Faker("en_IN")
Faker.seed(42)

from notebooks.pipeline_io import RAW_DB, save_raw

# ── Scale ─────────────────────────────────────────────────────────────────────
# Event-log tables are multiplied by CALRETAIL_SCALE; dimension tables are not.
#
# Keeping customers, products, stores, and inventory at full size is what makes
# the demo-scale build still look like a real retailer in the console — the
# catalogue and customer base are untouched, only the volume of behavioural
# events behind them shrinks. Scaling dimensions too would visibly thin out
# every dropdown and product grid in the UI.
SCALE = float(os.environ.get("CALRETAIL_SCALE", "1.0"))


def _n(full: int) -> int:
    """Scale an event-table row count, never below a floor that keeps stats sane."""
    return max(1_000, int(full * SCALE)) if SCALE < 1.0 else int(full * SCALE)


def _q(full: int, floor: int = 1) -> int:
    """
    Scale a *quantity*, as opposed to a row count.

    Stock levels have to move with demand. The inventory table keeps all 25,000
    rows at demo scale, but if the quantities in them stayed at full-scale size
    while transactions shrank, cover would inflate by exactly the scale factor —
    every SKU would read as massively overstocked and the markdown card would
    flag almost the whole catalogue. Scaling stock alongside demand keeps the
    stock-to-demand ratio, which is what every inventory metric is derived from.
    """
    return max(floor, int(round(full * SCALE))) if SCALE < 1.0 else int(full * SCALE)


# ── Constants ─────────────────────────────────────────────────────────────────
START_DATE = date(2022, 1, 1)
END_DATE   = date(2024, 12, 31)
DATE_RANGE = pd.date_range(START_DATE, END_DATE, freq="D")

# Dimensions — always full size.
N_CUSTOMERS   = 10_000
N_PRODUCTS    = 5_000
N_STORES      = 150
N_WAREHOUSES  = 30
N_SUPPLIERS   = 400
N_EMPLOYEES   = 2_500
N_INVENTORY   = 25_000

# Event logs — scaled.
N_TRANSACTIONS = _n(300_000)
N_BROWSING    = _n(900_000)
N_WISHLIST    = _n(100_000)
N_CART        = _n(100_000)
N_SEARCH      = _n(200_000)
N_SESSIONS    = _n(150_000)
N_PRICING_HIST = _n(150_000)
N_COMPETITOR  = _n(100_000)
N_PROMOTIONS  = _n(20_000)
N_CAMPAIGNS   = _n(10_000)
N_INV_MOVES   = _n(50_000)
N_ORDERS      = _n(200_000)
N_SHIPMENTS   = _n(80_000)
N_TICKETS     = _n(50_000)
N_RETURNS     = _n(40_000)
N_REVIEWS     = _n(100_000)

CITIES = [
    "Mumbai","Delhi","Bengaluru","Hyderabad","Chennai","Kolkata","Pune","Ahmedabad",
    "Jaipur","Lucknow","Surat","Kochi","Chandigarh","Bhopal","Indore","Nagpur",
    "Patna","Vadodara","Ludhiana","Agra"
]
REGIONS = {"Mumbai":"West","Delhi":"North","Bengaluru":"South","Hyderabad":"South",
           "Chennai":"South","Kolkata":"East","Pune":"West","Ahmedabad":"West",
           "Jaipur":"North","Lucknow":"North","Surat":"West","Kochi":"South",
           "Chandigarh":"North","Bhopal":"Central","Indore":"Central","Nagpur":"Central",
           "Patna":"East","Vadodara":"West","Ludhiana":"North","Agra":"North"}

CATEGORIES = {
    "Tops":       ["T-Shirt","Shirt","Blouse","Crop Top","Tank Top","Polo","Kurti","Sweatshirt","Hoodie"],
    "Bottoms":    ["Jeans","Trousers","Shorts","Skirt","Leggings","Palazzo","Joggers","Chinos"],
    "Dresses":    ["Casual Dress","Maxi Dress","Mini Dress","Evening Gown","Saree","Lehenga","Salwar Suit"],
    "Outerwear":  ["Jacket","Blazer","Coat","Windbreaker","Denim Jacket","Puffer Jacket"],
    "Footwear":   ["Sneakers","Heels","Sandals","Loafers","Boots","Flats","Sports Shoes","Oxford"],
    "Accessories":["Handbag","Sunglasses","Watch","Belt","Scarf","Cap","Jewellery","Wallet"],
    "Activewear": ["Sports Bra","Track Pants","Compression Tights","Gym Vest","Running Shoes"],
    "Innerwear":  ["Briefs","Boxers","Bra","Socks","Thermal Innerwear"],
    "Ethnic Wear":["Kurta","Sherwani","Dhoti","Dupatta","Anarkali","Churidar"],
}
ALL_CATEGORIES = list(CATEGORIES.keys())

BRANDS = [
    "Zara","H&M","Mango","Uniqlo","Marks & Spencer","Forever 21","Biba","FabIndia",
    "W","Global Desi","Libas","Aurelia","Mamaearth","Nykaa Fashion","Roadster",
    "Here&Now","Dressberry","All About You","AND","Vero Moda","Only","Jack & Jones",
    "Pepe Jeans","Levis","Lee","Arrow","Louis Philippe","Van Heusen","Peter England",
    "Allen Solly","U.S. Polo","Calvin Klein","Tommy Hilfiger","Adidas","Nike","Puma",
    "Reebok","Decathlon","Myntra","Bewakoof","The Souled Store","Blackberrys","Wrangler",
    "Spykar","Numero Uno","True Blue","LIFE","Cover Story","Koovs","Kazo"
]

SEASONS    = ["Summer","Winter","Monsoon","Festive","All Season"]
STYLE_TAGS = ["casual","formal","ethnic","party","sporty","office","streetwear","bohemian","minimalist","luxury"]
COLORS     = ["Black","White","Navy","Red","Green","Yellow","Pink","Blue","Purple","Orange",
               "Beige","Grey","Brown","Maroon","Teal","Olive","Cream","Mustard","Coral","Mint"]
SIZES      = ["XS","S","M","L","XL","XXL","Free Size"]

SEGMENTS = [
    "Budget Shopper","Luxury Shopper","Brand Loyalist","Impulse Buyer","Frequent Buyer",
    "Student","Working Professional","Family Shopper","Fashion Enthusiast","Seasonal Buyer"
]
SEGMENT_WEIGHTS = [0.18, 0.07, 0.12, 0.10, 0.13, 0.10, 0.12, 0.08, 0.06, 0.04]

INCOME_BRACKETS = ["<2L","2-5L","5-10L","10-20L","20-50L",">50L"]
CHANNELS = ["Email","SMS","Push Notification","WhatsApp","In-App"]
COMPETITORS = ["Myntra","Ajio","Nykaa Fashion","Flipkart Fashion","Amazon Fashion","Meesho","Snapdeal Fashion"]
TICKET_CATEGORIES = ["Order Issue","Return & Refund","Product Quality","Delivery Delay",
                     "Payment Problem","Wrong Item","Size Exchange","Account Issue"]
PRIORITIES = ["Low","Medium","High","Critical"]

def rand_date(start=START_DATE, end=END_DATE):
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, delta))

def rand_dates(n, start=START_DATE, end=END_DATE):
    start_ord = start.toordinal()
    end_ord   = end.toordinal()
    return [date.fromordinal(random.randint(start_ord, end_ord)) for _ in range(n)]


def covering_customers(cust_ids, n):
    """
    Return ``n`` customer ids in which every customer appears at least once.

    Picking independently at random leaves a Poisson tail with no rows at all —
    at demo scale that is roughly 9% of the customer base, and a customer with
    no history makes the recommendation, chatbot and next-best-offer cards look
    broken rather than sparse. Seeding the sequence with one row per customer
    removes that tail; the remainder stays independently random, so the
    distribution above the floor is unchanged.
    """
    ids = list(cust_ids)
    if n <= len(ids):
        return random.sample(ids, n)
    seq = ids + [random.choice(ids) for _ in range(n - len(ids))]
    random.shuffle(seq)
    return seq

def save(df: pd.DataFrame, name: str):
    save_raw(df, name)
    print(f"  ✓  {name}  ({len(df):,} rows)")

# ─────────────────────────────────────────────────────────────────────────────
# 1. REFERENCE TABLES
# ─────────────────────────────────────────────────────────────────────────────

def generate_stores():
    print("Generating stores...")
    rows = []
    for i in range(1, N_STORES + 1):
        city = random.choice(CITIES)
        rows.append({
            "store_id":   f"S{i:04d}",
            "store_name": f"{city} Store {i}",
            "city":       city,
            "region":     REGIONS[city],
            "tier":       random.choice(["Tier1","Tier2","Tier3"]),
            "store_type": random.choice(["Mall","High Street","Outlet","Standalone"]),
            "size_sqft":  random.randint(1_000, 15_000),
            "open_date":  rand_date(date(2015,1,1), date(2022,1,1)),
            "is_active":  random.choices([True, False], weights=[0.92, 0.08])[0],
        })
    df = pd.DataFrame(rows)
    save(df, "stores")
    return df

def generate_warehouses():
    print("Generating warehouses...")
    rows = []
    for i in range(1, N_WAREHOUSES + 1):
        city = random.choice(CITIES)
        rows.append({
            "warehouse_id":   f"W{i:03d}",
            "warehouse_name": f"{city} DC {i}",
            "city":           city,
            "region":         REGIONS[city],
            "capacity_sqft":  random.randint(50_000, 500_000),
            "type":           random.choice(["Fulfilment Centre","Regional Hub","Dark Store"]),
            "is_active":      True,
        })
    df = pd.DataFrame(rows)
    save(df, "warehouses")
    return df

def generate_suppliers():
    print("Generating suppliers...")
    rows = []
    for i in range(1, N_SUPPLIERS + 1):
        rows.append({
            "supplier_id":       f"SUP{i:04d}",
            "name":              fake.company(),
            "category":          random.choice(ALL_CATEGORIES),
            "country":           random.choice(["India","China","Bangladesh","Vietnam","Turkey"]),
            "city":              random.choice(CITIES),
            "lead_time_days":    random.randint(3, 45),
            "reliability_score": round(random.uniform(0.55, 0.99), 2),
            "min_order_qty":     random.randint(50, 500),
            "payment_terms":     random.choice(["Net30","Net60","Advance","COD"]),
            "is_active":         random.choices([True, False], weights=[0.88, 0.12])[0],
        })
    df = pd.DataFrame(rows)
    save(df, "suppliers")
    return df

def generate_employees(store_ids):
    print("Generating employees...")
    roles = ["Store Manager","Sales Associate","Cashier","Visual Merchandiser",
             "Stock Associate","Customer Service Rep","Security","Warehouse Staff"]
    rows = []
    for i in range(1, N_EMPLOYEES + 1):
        rows.append({
            "employee_id":       f"E{i:05d}",
            "store_id":          random.choice(store_ids),
            "name":              fake.name(),
            "role":              random.choice(roles),
            "join_date":         rand_date(date(2018,1,1), date(2024,1,1)),
            "salary":            random.randint(15_000, 120_000),
            "performance_score": round(random.uniform(2.0, 5.0), 1),
            "is_active":         random.choices([True, False], weights=[0.85, 0.15])[0],
        })
    df = pd.DataFrame(rows)
    save(df, "employees")
    return df

def generate_holiday_calendar():
    print("Generating holiday_calendar...")
    rows = []
    festivals = [
        (1, 26, "Republic Day"), (3, 1, "Holi"), (4, 14, "Baisakhi"),
        (8, 15, "Independence Day"), (10, 2, "Gandhi Jayanti"),
        (10, 20, "Dussehra"), (11, 1, "Diwali"), (11, 5, "Bhai Dooj"),
        (12, 25, "Christmas"), (1, 14, "Makar Sankranti"),
        (8, 31, "Raksha Bandhan"), (9, 7, "Ganesh Chaturthi"),
        (11, 20, "Guru Nanak Jayanti")
    ]
    for yr in range(2022, 2025):
        for single_date in pd.date_range(f"{yr}-01-01", f"{yr}-12-31"):
            d = single_date.date()
            holiday_name = None
            for m, day, name in festivals:
                if d.month == m and d.day == day:
                    holiday_name = name
                    break
            is_weekend = d.weekday() >= 5
            # Sale seasons
            is_sale = (
                (d.month == 1 and d.day <= 10) or     # New Year sale
                (d.month == 7 and 10 <= d.day <= 20) or # Mid-year sale
                (d.month == 10 and 15 <= d.day <= 31) or # Festive sale
                (d.month == 12 and 20 <= d.day <= 31)  # Christmas sale
            )
            season = (
                "Summer" if d.month in [3,4,5] else
                "Monsoon" if d.month in [6,7,8] else
                "Festive" if d.month in [9,10,11] else "Winter"
            )
            rows.append({
                "date":             d,
                "year":             yr,
                "month":            d.month,
                "day_of_week":      d.strftime("%A"),
                "is_weekend":       is_weekend,
                "is_holiday":       holiday_name is not None,
                "holiday_name":     holiday_name,
                "is_sale_season":   is_sale,
                "season":           season,
                "is_long_weekend":  is_weekend and holiday_name is not None,
            })
    df = pd.DataFrame(rows)
    save(df, "holiday_calendar")
    return df

def generate_weather():
    print("Generating weather...")
    rows = []
    for city in CITIES:
        for single_date in pd.date_range("2022-01-01", "2024-12-31"):
            d = single_date.date()
            m = d.month
            base_temp = {"Mumbai":28, "Delhi":22, "Bengaluru":23, "Chennai":29,
                         "Kolkata":25, "Hyderabad":26}.get(city, 24)
            seasonal_adj = -8 if m in [12,1,2] else (5 if m in [4,5,6] else 0)
            temp = round(base_temp + seasonal_adj + random.gauss(0, 3), 1)
            rainfall = max(0, round(random.gauss(
                15 if m in [7,8,9] else 2, 5), 1))
            rows.append({
                "date":              d,
                "city":              city,
                "temperature_c":     temp,
                "rainfall_mm":       rainfall,
                "humidity_pct":      random.randint(40, 95),
                "weather_condition": random.choices(
                    ["Sunny","Cloudy","Rainy","Foggy"],
                    weights=[0.5, 0.2, 0.2, 0.1])[0],
            })
    df = pd.DataFrame(rows)
    save(df, "weather")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 2. CUSTOMERS & PRODUCTS
# ─────────────────────────────────────────────────────────────────────────────

def generate_customers():
    print("Generating customers...")
    rows = []
    for i in range(1, N_CUSTOMERS + 1):
        seg = random.choices(SEGMENTS, weights=SEGMENT_WEIGHTS)[0]
        age = {
            "Student": random.randint(18, 25),
            "Working Professional": random.randint(26, 40),
            "Family Shopper": random.randint(30, 50),
            "Luxury Shopper": random.randint(28, 55),
        }.get(seg, random.randint(18, 65))

        income = {
            "Budget Shopper": random.choice(["<2L","2-5L"]),
            "Luxury Shopper": random.choice(["20-50L",">50L"]),
            "Student": "<2L",
            "Working Professional": random.choice(["5-10L","10-20L"]),
        }.get(seg, random.choice(INCOME_BRACKETS))

        freq = {
            "Frequent Buyer": round(random.uniform(8, 20), 1),
            "Seasonal Buyer": round(random.uniform(1, 3), 1),
            "Budget Shopper": round(random.uniform(2, 6), 1),
            "Luxury Shopper": round(random.uniform(1, 4), 1),
        }.get(seg, round(random.uniform(2, 12), 1))

        city = random.choice(CITIES)
        rows.append({
            "customer_id":        f"C{i:05d}",
            "name":               fake.name(),
            "age":                age,
            "gender":             random.choices(["Male","Female","Other"], weights=[0.45,0.52,0.03])[0],
            "city":               city,
            "region":             REGIONS[city],
            "income_bracket":     income,
            "segment":            seg,
            "shopping_frequency": freq,   # avg purchases per year
            "preferred_channel":  random.choice(CHANNELS),
            "preferred_category": random.choice(ALL_CATEGORIES),
            "loyalty_tier":       random.choices(["Bronze","Silver","Gold","Platinum"],
                                                  weights=[0.45,0.30,0.18,0.07])[0],
            "registration_date":  rand_date(date(2018,1,1), date(2024,1,1)),
            "is_active":          random.choices([True, False], weights=[0.85, 0.15])[0],
            "email":              fake.email(),
            "phone":              fake.phone_number(),
            "app_installed":      random.choices([True, False], weights=[0.65, 0.35])[0],
            "email_opt_in":       random.choices([True, False], weights=[0.72, 0.28])[0],
            "sms_opt_in":         random.choices([True, False], weights=[0.60, 0.40])[0],
        })
    df = pd.DataFrame(rows)
    save(df, "customers")
    return df

def generate_products():
    print("Generating products...")
    rows = []
    for i in range(1, N_PRODUCTS + 1):
        cat = random.choice(ALL_CATEGORIES)
        sub = random.choice(CATEGORIES[cat])
        brand = random.choice(BRANDS)
        season = random.choice(SEASONS)
        style = random.choice(STYLE_TAGS)

        # Price tiers vary by category
        base_price = {
            "Accessories": random.randint(300, 8_000),
            "Footwear":    random.randint(400, 6_000),
            "Dresses":     random.randint(500, 12_000),
            "Outerwear":   random.randint(800, 15_000),
            "Activewear":  random.randint(400, 5_000),
            "Innerwear":   random.randint(100, 800),
        }.get(cat, random.randint(200, 5_000))

        cost = round(base_price * random.uniform(0.35, 0.60), 2)
        mrp  = round(base_price * random.uniform(1.1, 1.5), 2)

        rows.append({
            "product_id":     f"P{i:05d}",
            "product_name":   f"{brand} {sub} {random.choice(COLORS)}",
            "category":       cat,
            "sub_category":   sub,
            "brand":          brand,
            "color":          random.choice(COLORS),
            "size":           random.choice(SIZES),
            "style_tag":      style,
            "season":         season,
            "price":          base_price,
            "mrp":            mrp,
            "cost_price":     cost,
            "launch_date":    rand_date(date(2020,1,1), date(2024,6,1)),
            "is_active":      random.choices([True, False], weights=[0.88, 0.12])[0],
            "supplier_id":    f"SUP{random.randint(1, N_SUPPLIERS):04d}",
            "weight_grams":   random.randint(100, 2000),
        })
    df = pd.DataFrame(rows)
    save(df, "products")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 3. TRANSACTIONS (Core behavioural table)
# ─────────────────────────────────────────────────────────────────────────────

def generate_transactions(customers_df, products_df, stores_df):
    print("Generating transactions (300k rows — may take ~30s)...")
    cust_ids = customers_df["customer_id"].tolist()
    prod_ids = products_df["product_id"].tolist()
    store_ids = stores_df["store_id"].tolist()

    # Segment → purchase amount multiplier
    seg_price_map = {
        "Budget Shopper": (0.3, 0.6), "Luxury Shopper": (1.5, 4.0),
        "Brand Loyalist": (0.8, 1.4), "Impulse Buyer": (0.5, 1.2),
        "Frequent Buyer": (0.6, 1.2), "Student": (0.2, 0.7),
        "Working Professional": (0.9, 1.8), "Family Shopper": (0.6, 1.3),
        "Fashion Enthusiast": (0.8, 1.5), "Seasonal Buyer": (0.5, 1.1),
    }
    seg_lookup = customers_df.set_index("customer_id")["segment"].to_dict()
    pref_lookup = customers_df.set_index("customer_id")["preferred_category"].to_dict()
    price_lookup = products_df.set_index("product_id")["price"].to_dict()
    cat_lookup   = products_df.set_index("product_id")["category"].to_dict()

    rows = []
    txn_customers = covering_customers(cust_ids, N_TRANSACTIONS)
    for i in range(1, N_TRANSACTIONS + 1):
        cid = txn_customers[i - 1]
        seg = seg_lookup.get(cid, "Frequent Buyer")
        pref_cat = pref_lookup.get(cid, random.choice(ALL_CATEGORIES))

        # 70% of purchases are in preferred category
        if random.random() < 0.70:
            eligible = products_df[products_df["category"] == pref_cat]["product_id"].tolist()
            pid = random.choice(eligible) if eligible else random.choice(prod_ids)
        else:
            pid = random.choice(prod_ids)

        base_price = price_lookup.get(pid, 500)
        lo, hi = seg_price_map.get(seg, (0.6, 1.2))
        unit_price = round(base_price * random.uniform(lo, hi), 2)
        qty = random.choices([1, 2, 3, 4], weights=[0.6, 0.25, 0.10, 0.05])[0]

        # Seasonal + festive uplift for dates
        t_date = rand_date()
        m = t_date.month
        promo_discount = 0.0
        if m in [10, 11]:   # Festive season
            promo_discount = round(random.uniform(0.05, 0.30), 2)
        elif m in [1, 7, 12]:  # Sale months
            promo_discount = round(random.uniform(0.05, 0.20), 2)

        final_price = round(unit_price * (1 - promo_discount), 2)
        channel = random.choices(["Online","Offline"], weights=[0.68, 0.32])[0]
        store_id = random.choice(store_ids) if channel == "Offline" else None
        payment  = random.choices(
            ["UPI","Credit Card","Debit Card","Net Banking","COD","Wallet"],
            weights=[0.35, 0.22, 0.15, 0.08, 0.12, 0.08])[0]

        rows.append({
            "transaction_id":  f"T{i:07d}",
            "customer_id":     cid,
            "product_id":      pid,
            "store_id":        store_id,
            "quantity":        qty,
            "unit_price":      unit_price,
            "discount_pct":    promo_discount,
            "final_price":     final_price,
            "total_amount":    round(final_price * qty, 2),
            "channel":         channel,
            "payment_method":  payment,
            "transaction_date": t_date,
            "is_returned":     random.choices([False, True], weights=[0.88, 0.12])[0],
        })

        if i % 50_000 == 0:
            print(f"    {i:,} transactions...")

    df = pd.DataFrame(rows)
    save(df, "transactions")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 4. BROWSING / SESSION / WISHLIST / CART / SEARCH
# ─────────────────────────────────────────────────────────────────────────────

def generate_browsing_history(customers_df, products_df):
    print("Generating browsing_history (900k rows — may take ~45s)...")
    cust_ids = customers_df["customer_id"].tolist()
    prod_ids = products_df["product_id"].tolist()
    devices = ["Mobile","Desktop","Tablet"]
    device_w = [0.65, 0.27, 0.08]
    rows = []
    for i in range(1, N_BROWSING + 1):
        cid = random.choice(cust_ids)
        pid = random.choice(prod_ids)
        dwell = max(5, int(np.random.exponential(120)))   # seconds
        rows.append({
            "browse_id":    f"B{i:08d}",
            "customer_id":  cid,
            "product_id":   pid,
            "page_views":   random.randint(1, 8),
            "dwell_seconds": dwell,
            "device":       random.choices(devices, weights=device_w)[0],
            "source":       random.choices(["Search","Banner","Email","Social","Direct"],
                                           weights=[0.35,0.20,0.15,0.20,0.10])[0],
            "timestamp":    datetime.combine(rand_date(),
                            datetime.min.time()) + timedelta(
                            hours=random.randint(0,23), minutes=random.randint(0,59)),
            "added_to_cart": random.choices([False, True], weights=[0.82, 0.18])[0],
            "purchased":    random.choices([False, True], weights=[0.88, 0.12])[0],
        })
        if i % 200_000 == 0:
            print(f"    {i:,} browse events...")
    df = pd.DataFrame(rows)
    save(df, "browsing_history")
    return df

def generate_customer_sessions(customers_df):
    print("Generating customer_sessions...")
    cust_ids = customers_df["customer_id"].tolist()
    rows = []
    for i in range(1, N_SESSIONS + 1):
        login_dt = datetime.combine(rand_date(),
                   datetime.min.time()) + timedelta(
                   hours=random.randint(0,23), minutes=random.randint(0,59))
        duration = random.randint(2, 90)  # minutes
        rows.append({
            "session_id":    f"SES{i:07d}",
            "customer_id":   random.choice(cust_ids),
            "login_time":    login_dt,
            "logout_time":   login_dt + timedelta(minutes=duration),
            "duration_mins": duration,
            "device":        random.choices(["Mobile","Desktop","Tablet"],
                                            weights=[0.65,0.27,0.08])[0],
            "channel":       random.choices(["App","Web"], weights=[0.60,0.40])[0],
            "pages_visited": random.randint(1, 30),
            "date":          login_dt.date(),
        })
    df = pd.DataFrame(rows)
    save(df, "customer_sessions")
    return df

def generate_wishlist(customers_df, products_df):
    print("Generating wishlist...")
    cust_ids = customers_df["customer_id"].tolist()
    prod_ids = products_df["product_id"].tolist()
    rows = []
    seen = set()
    i = 1
    while i <= N_WISHLIST:
        cid = random.choice(cust_ids)
        pid = random.choice(prod_ids)
        if (cid, pid) in seen:
            continue
        seen.add((cid, pid))
        rows.append({
            "wishlist_id":   f"WL{i:07d}",
            "customer_id":   cid,
            "product_id":    pid,
            "added_date":    rand_date(),
            "is_purchased":  random.choices([False, True], weights=[0.70, 0.30])[0],
        })
        i += 1
    df = pd.DataFrame(rows)
    save(df, "wishlist")
    return df

def generate_shopping_cart(customers_df, products_df):
    print("Generating shopping_cart...")
    cust_ids = customers_df["customer_id"].tolist()
    prod_ids = products_df["product_id"].tolist()
    statuses = ["Active","Purchased","Abandoned","Expired"]
    rows = []
    for i in range(1, N_CART + 1):
        rows.append({
            "cart_id":      f"CART{i:07d}",
            "customer_id":  random.choice(cust_ids),
            "product_id":   random.choice(prod_ids),
            "quantity":     random.randint(1, 4),
            "added_at":     datetime.combine(rand_date(), datetime.min.time()) +
                            timedelta(hours=random.randint(0,23)),
            "status":       random.choices(statuses, weights=[0.15,0.45,0.35,0.05])[0],
        })
    df = pd.DataFrame(rows)
    save(df, "shopping_cart")
    return df

def generate_search_history(customers_df, products_df):
    print("Generating search_history...")
    cust_ids = customers_df["customer_id"].tolist()
    prod_ids = products_df["product_id"].tolist()
    queries = [
        "black dress","summer kurta","formal shirt men","running shoes","handbag red",
        "ethnic wear women","winter jacket","casual jeans","party wear dress","sneakers white",
        "silk saree","denim jacket","gym wear","joggers mens","anarkali suit","maxi dress",
        "office wear","crop top","palazzo pants","printed kurti",
    ]
    rows = []
    for i in range(1, N_SEARCH + 1):
        clicked = random.choices([None, random.choice(prod_ids)], weights=[0.35, 0.65])[0]
        rows.append({
            "search_id":         f"SR{i:07d}",
            "customer_id":       random.choice(cust_ids),
            "query_text":        random.choice(queries),
            "timestamp":         datetime.combine(rand_date(), datetime.min.time()) +
                                 timedelta(hours=random.randint(0,23), minutes=random.randint(0,59)),
            "results_count":     random.randint(0, 500),
            "clicked_product_id": clicked,
            "converted":         random.choices([False, True], weights=[0.80, 0.20])[0],
        })
    df = pd.DataFrame(rows)
    save(df, "search_history")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 5. INVENTORY & MOVEMENTS
# ─────────────────────────────────────────────────────────────────────────────

def generate_inventory(products_df, stores_df, warehouses_df):
    print("Generating inventory...")
    prod_ids  = products_df["product_id"].tolist()
    store_ids = stores_df["store_id"].tolist()
    wh_ids    = warehouses_df["warehouse_id"].tolist()
    rows = []
    for i in range(1, N_INVENTORY + 1):
        loc_type = random.choices(["store","warehouse"], weights=[0.60, 0.40])[0]
        pid = random.choice(prod_ids)
        # Quantities scale with demand; the reorder point scales with them so
        # the stock:reorder ratio the health metrics key off stays constant.
        stock = random.randint(0, _q(500, 20))
        reorder = random.randint(_q(20, 2), _q(100, 8))
        rows.append({
            "inventory_id":    f"INV{i:06d}",
            "product_id":      pid,
            "store_id":        random.choice(store_ids) if loc_type=="store" else None,
            "warehouse_id":    random.choice(wh_ids) if loc_type=="warehouse" else None,
            "location_type":   loc_type,
            "stock_qty":       stock,
            "reorder_point":   reorder,
            "max_stock":       reorder * 5,
            "days_on_hand":    round(stock / max(random.randint(1,20), 1), 1),
            "last_updated":    rand_date(date(2024,1,1), date(2024,12,31)),
        })
    df = pd.DataFrame(rows)
    save(df, "inventory")
    return df

def generate_inventory_movements(inventory_df, warehouses_df):
    print("Generating inventory_movements...")
    inv_ids = inventory_df["inventory_id"].tolist()
    wh_ids  = warehouses_df["warehouse_id"].tolist()
    reasons = ["Replenishment","Return","Inter-store Transfer","Damage Write-off","Stocktake Adjustment"]
    rows = []
    for i in range(1, N_INV_MOVES + 1):
        rows.append({
            "movement_id":    f"MV{i:07d}",
            "inventory_id":   random.choice(inv_ids),
            "from_location":  random.choice(wh_ids),
            "to_location":    random.choice(wh_ids),
            "quantity":       random.randint(1, 200),
            "movement_date":  rand_date(),
            "reason":         random.choice(reasons),
            "approved_by":    f"E{random.randint(1, N_EMPLOYEES):05d}",
        })
    df = pd.DataFrame(rows)
    save(df, "inventory_movements")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 6. PRICING & COMPETITOR
# ─────────────────────────────────────────────────────────────────────────────

def generate_pricing_history(products_df, stores_df):
    print("Generating pricing_history...")
    prod_ids  = products_df["product_id"].tolist()
    store_ids = stores_df["store_id"].tolist()
    price_types = ["Regular","Sale","Promotional","Clearance","Flash Sale"]
    price_lookup = products_df.set_index("product_id")["price"].to_dict()
    rows = []
    for i in range(1, N_PRICING_HIST + 1):
        pid = random.choice(prod_ids)
        base = price_lookup.get(pid, 500)
        p_type = random.choices(price_types, weights=[0.50,0.20,0.15,0.10,0.05])[0]
        factor = {"Regular":1.0,"Sale":0.80,"Promotional":0.85,"Clearance":0.60,"Flash Sale":0.70}.get(p_type,1.0)
        rows.append({
            "price_id":       f"PH{i:07d}",
            "product_id":     pid,
            "store_id":       random.choice(store_ids),
            "price":          round(base * factor * random.uniform(0.95, 1.05), 2),
            "price_type":     p_type,
            "effective_date": rand_date(),
            "created_by":     f"E{random.randint(1, N_EMPLOYEES):05d}",
        })
    df = pd.DataFrame(rows)
    save(df, "pricing_history")
    return df

def generate_competitor_pricing(products_df):
    print("Generating competitor_pricing...")
    prod_ids = products_df["product_id"].tolist()
    price_lookup = products_df.set_index("product_id")["price"].to_dict()
    rows = []
    for i in range(1, N_COMPETITOR + 1):
        pid = random.choice(prod_ids)
        our_price = price_lookup.get(pid, 500)
        rows.append({
            "comp_price_id":   f"CP{i:07d}",
            "product_id":      pid,
            "competitor_name": random.choice(COMPETITORS),
            "price":           round(our_price * random.uniform(0.75, 1.30), 2),
            "scraped_date":    rand_date(),
            "in_stock":        random.choices([True, False], weights=[0.82, 0.18])[0],
            "source_url":      fake.url(),
        })
    df = pd.DataFrame(rows)
    save(df, "competitor_pricing")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 7. PROMOTIONS & CAMPAIGNS
# ─────────────────────────────────────────────────────────────────────────────

def generate_promotions(products_df):
    print("Generating promotions...")
    prod_ids = products_df["product_id"].tolist()
    promo_types = ["Percentage Off","BOGO","Flat Discount","Free Shipping","Bundle","Cashback"]
    rows = []
    for i in range(1, N_PROMOTIONS + 1):
        start = rand_date()
        end   = start + timedelta(days=random.randint(1, 30))
        rows.append({
            "promo_id":       f"PR{i:06d}",
            "product_id":     random.choice(prod_ids),
            "promo_type":     random.choice(promo_types),
            "discount_pct":   round(random.uniform(0.05, 0.60), 2),
            "channel":        random.choice(["Online","Offline","Both"]),
            "target_segment": random.choice(SEGMENTS),
            "start_date":     start,
            "end_date":       end,
            "budget":         random.randint(10_000, 500_000),
            "is_active":      random.choices([True, False], weights=[0.40, 0.60])[0],
        })
    df = pd.DataFrame(rows)
    save(df, "promotions")
    return df

def generate_marketing_campaigns(promotions_df):
    print("Generating marketing_campaigns...")
    promo_ids = promotions_df["promo_id"].tolist()
    rows = []
    for i in range(1, N_CAMPAIGNS + 1):
        start = rand_date()
        end   = start + timedelta(days=random.randint(7, 90))
        budget = random.randint(50_000, 5_000_000)
        spent  = round(budget * random.uniform(0.40, 1.0), 2)
        rows.append({
            "campaign_id":      f"CAM{i:05d}",
            "promo_id":         random.choice(promo_ids),
            "campaign_name":    f"Campaign {fake.catch_phrase()[:30]}",
            "channel":          random.choice(CHANNELS),
            "target_segment":   random.choice(SEGMENTS),
            "budget":           budget,
            "amount_spent":     spent,
            "impressions":      random.randint(1_000, 1_000_000),
            "clicks":           random.randint(100, 50_000),
            "conversions":      random.randint(10, 5_000),
            "start_date":       start,
            "end_date":         end,
            "status":           random.choices(["Active","Completed","Paused"], weights=[0.25,0.65,0.10])[0],
        })
    df = pd.DataFrame(rows)
    save(df, "marketing_campaigns")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 8. ORDERS & SHIPMENTS
# ─────────────────────────────────────────────────────────────────────────────

def generate_orders(customers_df, products_df, stores_df):
    print("Generating orders (200k rows)...")
    cust_ids  = customers_df["customer_id"].tolist()
    prod_ids  = products_df["product_id"].tolist()
    store_ids = stores_df["store_id"].tolist()
    price_lookup = products_df.set_index("product_id")["price"].to_dict()
    statuses = ["Placed","Confirmed","Shipped","Delivered","Cancelled","Returned"]
    weights  = [0.03, 0.05, 0.10, 0.72, 0.05, 0.05]
    rows = []
    order_customers = covering_customers(cust_ids, N_ORDERS)
    for i in range(1, N_ORDERS + 1):
        pid = random.choice(prod_ids)
        qty = random.randint(1, 3)
        price = price_lookup.get(pid, 500)
        order_date = rand_date()
        est_delivery = order_date + timedelta(days=random.randint(2, 10))
        rows.append({
            "order_id":             f"ORD{i:07d}",
            "customer_id":          order_customers[i - 1],
            "product_id":           pid,
            "store_id":             random.choice(store_ids + [None]),
            "quantity":             qty,
            "unit_price":           price,
            "total_amount":         round(price * qty, 2),
            "order_date":           order_date,
            "estimated_delivery":   est_delivery,
            "actual_delivery":      est_delivery + timedelta(days=random.randint(-1, 5)),
            "status":               random.choices(statuses, weights=weights)[0],
            "delivery_city":        random.choice(CITIES),
            "shipping_fee":         random.choices([0, 49, 99], weights=[0.55, 0.30, 0.15])[0],
        })
        if i % 50_000 == 0:
            print(f"    {i:,} orders...")
    df = pd.DataFrame(rows)
    save(df, "orders")
    return df

def generate_shipments(orders_df, suppliers_df, warehouses_df):
    print("Generating shipments...")
    orders_sample = orders_df.sample(n=N_SHIPMENTS, replace=True)
    sup_ids = suppliers_df["supplier_id"].tolist()
    wh_ids  = warehouses_df["warehouse_id"].tolist()
    rows = []
    for i, (_, order) in enumerate(orders_sample.iterrows(), 1):
        ship_date = order["order_date"]
        if isinstance(ship_date, str):
            ship_date = date.fromisoformat(ship_date)
        lead = random.randint(2, 15)
        rows.append({
            "shipment_id":      f"SHP{i:06d}",
            "order_id":         order["order_id"],
            "supplier_id":      random.choice(sup_ids),
            "warehouse_id":     random.choice(wh_ids),
            "product_id":       order["product_id"],
            "quantity":         order["quantity"],
            "ship_date":        ship_date,
            "expected_arrival": ship_date + timedelta(days=lead),
            "actual_arrival":   ship_date + timedelta(days=lead + random.randint(-2, 5)),
            "status":           random.choices(["In Transit","Delivered","Delayed","Lost"],
                                               weights=[0.15, 0.75, 0.08, 0.02])[0],
            "carrier":          random.choice(["Delhivery","Blue Dart","Ekart","Shadowfax","DTDC"]),
            "freight_cost":     round(random.uniform(50, 500), 2),
        })
    df = pd.DataFrame(rows)
    save(df, "shipments")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# 9. SUPPORT — TICKETS, RETURNS, REVIEWS
# ─────────────────────────────────────────────────────────────────────────────

def generate_support_tickets(customers_df, orders_df):
    print("Generating support_tickets...")
    cust_ids  = customers_df["customer_id"].tolist()
    order_ids = orders_df["order_id"].tolist()
    statuses  = ["Open","In Progress","Resolved","Closed","Escalated"]
    rows = []
    for i in range(1, N_TICKETS + 1):
        created = datetime.combine(rand_date(), datetime.min.time()) + \
                  timedelta(hours=random.randint(0, 23))
        priority = random.choices(PRIORITIES, weights=[0.30, 0.40, 0.22, 0.08])[0]
        resolved = created + timedelta(hours=random.randint(1, 120))
        descriptions = [
            "My order has not arrived yet.",
            "I received the wrong product.",
            "The product quality is very poor.",
            "I want to return the item.",
            "My payment was deducted but order not confirmed.",
            "The size I received is different from what I ordered.",
            "I need help with my account login.",
            "Product is damaged.",
            "Discount coupon not applied.",
            "Delivery is delayed.",
        ]
        rows.append({
            "ticket_id":    f"TKT{i:06d}",
            "customer_id":  random.choice(cust_ids),
            "order_id":     random.choice(order_ids),
            "category":     random.choice(TICKET_CATEGORIES),
            "priority":     priority,
            "description":  random.choice(descriptions),
            "status":       random.choices(statuses, weights=[0.10,0.15,0.55,0.15,0.05])[0],
            "channel":      random.choice(["Chat","Email","Phone","WhatsApp"]),
            "created_at":   created,
            "resolved_at":  resolved,
            "agent_id":     f"E{random.randint(1, N_EMPLOYEES):05d}",
            "csat_score":   random.choices([None, 1, 2, 3, 4, 5], weights=[0.30,0.05,0.08,0.15,0.25,0.17])[0],
        })
    df = pd.DataFrame(rows)
    save(df, "support_tickets")
    return df

def generate_returns(orders_df, customers_df, products_df):
    print("Generating returns...")
    # Take a subset of orders
    returnable = orders_df[orders_df["status"].isin(["Delivered","Returned"])].sample(
        n=N_RETURNS, replace=True)
    reasons = ["Wrong Size","Defective Product","Wrong Item Delivered","Not as Described",
               "Changed Mind","Quality Issue","Duplicate Order","Better Price Found"]
    statuses = ["Requested","Approved","Picked Up","Refunded","Rejected"]
    rows = []
    for i, (_, order) in enumerate(returnable.iterrows(), 1):
        order_d = order["order_date"]
        if isinstance(order_d, str):
            order_d = date.fromisoformat(order_d)
        return_d = order_d + timedelta(days=random.randint(1, 10))
        rows.append({
            "return_id":     f"RET{i:06d}",
            "order_id":      order["order_id"],
            "customer_id":   order["customer_id"],
            "product_id":    order["product_id"],
            "reason":        random.choice(reasons),
            "return_date":   return_d,
            "refund_amount": round(order["total_amount"] * random.uniform(0.80, 1.0), 2),
            "status":        random.choices(statuses, weights=[0.10,0.20,0.15,0.50,0.05])[0],
        })
    df = pd.DataFrame(rows)
    save(df, "returns")
    return df

def generate_customer_reviews(transactions_df, customers_df, products_df):
    print("Generating customer_reviews (100k rows)...")
    trans_sample = transactions_df.sample(n=N_REVIEWS, replace=True)
    positive_reviews = [
        "Excellent product, very happy with the quality!",
        "Fast delivery and great fit. Highly recommend.",
        "Loved the color and fabric. Will buy again.",
        "Perfect for the occasion. Great value for money.",
        "Amazing product, exactly as described.",
    ]
    negative_reviews = [
        "Product quality is very poor. Not worth the price.",
        "Wrong size delivered. Very disappointed.",
        "The stitching came off after one wash.",
        "Color faded after first wash. Not as shown in picture.",
        "Received a damaged product. Requesting refund.",
    ]
    neutral_reviews = [
        "Decent product. Meets basic expectations.",
        "Average quality. Nothing exceptional.",
        "Okay for the price. Could be better.",
        "Material is fine but delivery was slow.",
    ]
    rows = []
    for i, (_, txn) in enumerate(trans_sample.iterrows(), 1):
        rating = random.choices([1, 2, 3, 4, 5], weights=[0.05, 0.08, 0.15, 0.32, 0.40])[0]
        if rating >= 4:
            text = random.choice(positive_reviews)
            sent = "Positive"
        elif rating <= 2:
            text = random.choice(negative_reviews)
            sent = "Negative"
        else:
            text = random.choice(neutral_reviews)
            sent = "Neutral"
        rows.append({
            "review_id":       f"REV{i:07d}",
            "customer_id":     txn["customer_id"],
            "product_id":      txn["product_id"],
            "order_id":        f"ORD{random.randint(1, N_ORDERS):07d}",
            "rating":          rating,
            "review_text":     text,
            "review_date":     rand_date(),
            "sentiment_label": sent,
            "is_verified":     random.choices([True, False], weights=[0.78, 0.22])[0],
            "helpful_votes":   random.randint(0, 150),
        })
    df = pd.DataFrame(rows)
    save(df, "customer_reviews")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Nexalyze — Synthetic Data Generation")
    print("=" * 60)
    print(f"  Output → {RAW_DB}   (scale {SCALE:g})\n")

    # Reference tables
    stores_df     = generate_stores()
    warehouses_df = generate_warehouses()
    suppliers_df  = generate_suppliers()
    employees_df  = generate_employees(stores_df["store_id"].tolist())
    holiday_df    = generate_holiday_calendar()
    weather_df    = generate_weather()

    # Core entity tables
    customers_df  = generate_customers()
    products_df   = generate_products()

    # Transactional tables
    transactions_df = generate_transactions(customers_df, products_df, stores_df)
    browsing_df     = generate_browsing_history(customers_df, products_df)
    sessions_df     = generate_customer_sessions(customers_df)
    wishlist_df     = generate_wishlist(customers_df, products_df)
    cart_df         = generate_shopping_cart(customers_df, products_df)
    search_df       = generate_search_history(customers_df, products_df)

    # Inventory
    inventory_df  = generate_inventory(products_df, stores_df, warehouses_df)
    inv_moves_df  = generate_inventory_movements(inventory_df, warehouses_df)

    # Pricing & Competition
    pricing_df    = generate_pricing_history(products_df, stores_df)
    competitor_df = generate_competitor_pricing(products_df)

    # Promotions & Campaigns
    promotions_df = generate_promotions(products_df)
    campaigns_df  = generate_marketing_campaigns(promotions_df)

    # Orders & Shipments
    orders_df     = generate_orders(customers_df, products_df, stores_df)
    shipments_df  = generate_shipments(orders_df, suppliers_df, warehouses_df)

    # Support
    tickets_df    = generate_support_tickets(customers_df, orders_df)
    returns_df    = generate_returns(orders_df, customers_df, products_df)
    reviews_df    = generate_customer_reviews(transactions_df, customers_df, products_df)

    print("\n" + "=" * 60)
    print("  All datasets generated successfully!")
    print(f"  Tables written to: {RAW_DB}")
    print("=" * 60)


if __name__ == "__main__":
    main()
