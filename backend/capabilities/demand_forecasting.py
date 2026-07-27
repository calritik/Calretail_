"""
CalRetail — Demand forecasting.

Ported from ``notebooks/capabilities/05_demand_forecasting.ipynb``. The notebook remains the readable
narrative of the method; this module is what the API actually runs.

State is built lazily by :func:`_init` on the first call, so importing this
module is free and nothing is computed for a capability nobody asks for.
:func:`reset` drops it again, which is how the process stays inside a small
memory budget without re-executing a notebook.
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import json
import re
import math
from backend.utils.db import load_table
import warnings

warnings.filterwarnings('ignore')
from backend.capabilities import _registry

_READY = False
_BUILDING = False


def _init() -> None:
    """
    Build this capability's shared frames. Idempotent and cheap once warm.

    The _BUILDING guard matters: helpers lifted out of the setup block call
    _init() like every other function, and the setup itself calls those helpers.
    Without the guard that is unbounded recursion. Re-entering during the build
    simply returns, which leaves the helper reading the partially-built state —
    exactly what it saw when these were sequential notebook cells.
    """
    global _READY, _BUILDING, xgb, mean_absolute_error, DOW_MAP, daily, prod_ref, holiday_cal, holiday_lookup, c, product_avg, category_avg, FEATURES, model_df, split_idx, X_train, X_test, y_train, y_test, model, test_preds, GLOBAL_MAE, GLOBAL_MAPE, _product_mape_cache, _test_pid_lookup
    if _READY or _BUILDING:
        return
    _BUILDING = True
    try:
        import xgboost as xgb
        from sklearn.metrics import mean_absolute_error

        DOW_MAP = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3, 'Friday': 4, 'Saturday': 5, 'Sunday': 6}

        # Use the pre-engineered daily sales feature table (real lags, rolling means and
        # calendar flags computed in notebooks/feature_engineering.py) instead of
        # recomputing a thin, single-product subset of features from raw transactions.
        daily = load_table('feature_daily_sales')
        daily['transaction_date'] = pd.to_datetime(daily['transaction_date'])
        prod_ref = load_table('products')
        daily = daily.merge(prod_ref[['product_id', 'category']], on='product_id', how='left')

        holiday_cal = load_table('holiday_calendar')
        holiday_cal['date'] = pd.to_datetime(holiday_cal['date'])
        holiday_lookup = holiday_cal.set_index('date')[['is_holiday', 'is_sale_season']].to_dict('index')

        daily['dow_num'] = daily['day_of_week'].map(DOW_MAP).fillna(0).astype(int)
        for c in ['is_holiday', 'is_weekend', 'is_sale_season']:
            daily[c] = daily[c].astype(bool).astype(int)

        # Per-product / per-category baseline demand levels. This lets ONE global
        # model generalise across all 5,000 SKUs instead of fitting (and mis-applying)
        # a separate small model per product.
        product_avg = daily.groupby('product_id')['daily_qty'].mean().rename('product_avg_qty')
        category_avg = daily.groupby('category')['daily_qty'].mean().rename('category_avg_qty')
        daily = daily.merge(product_avg, on='product_id', how='left')
        daily = daily.merge(category_avg, on='category', how='left')

        FEATURES = ['lag_7', 'lag_14', 'rolling_7_mean', 'rolling_30_mean', 'dow_num', 'month',
                    'day_of_month', 'week_number', 'quarter', 'is_holiday', 'is_weekend',
                    'is_sale_season', 'product_avg_qty', 'category_avg_qty']

        model_df = daily.dropna(subset=FEATURES + ['daily_qty']).sort_values('transaction_date')

        # Time-based split (train on the past, test on the most recent 15%) — the only
        # honest way to validate a forecasting model.
        split_idx = int(len(model_df) * 0.85)
        X_train, X_test = model_df[FEATURES].iloc[:split_idx], model_df[FEATURES].iloc[split_idx:]
        y_train, y_test = model_df['daily_qty'].iloc[:split_idx], model_df['daily_qty'].iloc[split_idx:]

        model = xgb.XGBRegressor(
            n_estimators=250, max_depth=5, learning_rate=0.06,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
        )
        model.fit(X_train, y_train)

        test_preds = np.clip(model.predict(X_test), 0, None)
        GLOBAL_MAE = float(mean_absolute_error(y_test, test_preds))
        GLOBAL_MAPE = float(np.mean(np.abs((y_test - test_preds) / np.clip(y_test, 1, None))) * 100)

        print(f"Global XGBoost demand model trained on {len(X_train):,} rows across {daily['product_id'].nunique():,} products.")
        print(f"Held-out test MAE: {GLOBAL_MAE:.2f} units | MAPE: {GLOBAL_MAPE:.2f}%")

        _product_mape_cache = {}
        _test_pid_lookup = model_df.loc[X_test.index, 'product_id']

        _READY = True
    finally:
        _BUILDING = False

    # Registering last bounds how many capabilities hold frames at once; the
    # coldest is reset when this one pushes the count over the limit.
    _registry.touch(__name__)


def __getattr__(name: str):
    """
    Build the state on first attribute access (PEP 562).

    Callers that reach past the public functions for a shared frame — the
    recommendations debug view reads the feedback matrix directly — would
    otherwise see an AttributeError, because nothing exists until _init() runs.
    This is only consulted for names *missing* from the module, so it costs
    nothing once warm.
    """
    if not name.startswith("__"):
        _init()
        if name in globals():
            return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def reset() -> None:
    """
    Release the cached frames so the next call rebuilds them.

    The names are *deleted*, not set to None. __getattr__ above only fires for
    names missing from the module, so leaving a None behind would hand a caller
    that None forever instead of triggering a rebuild — the frames would look
    released while every read of them silently broke.
    """
    global _READY
    _READY = False
    for _name in ('xgb', 'mean_absolute_error', 'DOW_MAP', 'daily', 'prod_ref', 'holiday_cal', 'holiday_lookup', 'c', 'product_avg', 'category_avg', 'FEATURES', 'model_df', 'split_idx', 'X_train', 'X_test', 'y_train', 'y_test', 'model', 'test_preds', 'GLOBAL_MAE', 'GLOBAL_MAPE', '_product_mape_cache', '_test_pid_lookup'):
        globals().pop(_name, None)


def _product_test_mape(product_id):
    """Real MAPE computed from this product's own rows in the held-out test
    split when there are enough of them; otherwise the overall model MAPE.
    Never a fabricated/hashed number."""
    _init()
    if product_id in _product_mape_cache:
        return _product_mape_cache[product_id]
    mask = _test_pid_lookup == product_id
    if mask.sum() >= 5:
        actual = y_test[mask]
        preds = np.clip(model.predict(X_test[mask]), 0, None)
        mape = float(np.mean(np.abs((actual - preds) / np.clip(actual, 1, None))) * 100)
    else:
        mape = GLOBAL_MAPE
    _product_mape_cache[product_id] = mape
    return mape


def get_demand_forecast(product_id, forecast_days=7):
    _init()
    p_hist = daily[daily['product_id'] == product_id].sort_values('transaction_date')
    cat_series = prod_ref.loc[prod_ref['product_id'] == product_id, 'category']
    cat = cat_series.iloc[0] if len(cat_series) else None
    c_avg_qty = float(category_avg.get(cat, daily['daily_qty'].mean()))

    if p_hist.empty:
        # No sales history at all for this SKU -> category baseline is the
        # most honest estimate available (still real data, not a guess).
        forecast = [{"day": i, "forecast_qty": round(c_avg_qty, 1)} for i in range(1, forecast_days + 1)]
        return {
            "product_id": product_id, "forecast_horizon_days": forecast_days,
            "forecast": forecast, "historical": [],
            "mape": round(GLOBAL_MAPE, 2), "model": "Category Baseline (no sales history)",
        }

    last_date = p_hist['transaction_date'].iloc[-1]
    p_avg_qty = float(product_avg.get(product_id, daily['daily_qty'].mean()))
    recent_lags = list(p_hist['daily_qty'].tail(30))

    forecast = []
    for i in range(1, forecast_days + 1):
        future_date = last_date + pd.Timedelta(days=i)
        cal_info = holiday_lookup.get(future_date.normalize(), {})
        row = {
            'lag_7':  recent_lags[-7]  if len(recent_lags) >= 7  else recent_lags[-1],
            'lag_14': recent_lags[-14] if len(recent_lags) >= 14 else recent_lags[-1],
            'rolling_7_mean':  float(np.mean(recent_lags[-7:])),
            'rolling_30_mean': float(np.mean(recent_lags[-30:])),
            'dow_num': DOW_MAP.get(future_date.day_name(), 0),
            'month': future_date.month,
            'day_of_month': future_date.day,
            'week_number': int(future_date.isocalendar()[1]),
            'quarter': future_date.quarter,
            'is_holiday': int(cal_info.get('is_holiday', False)),
            'is_weekend': int(future_date.dayofweek >= 5),
            'is_sale_season': int(cal_info.get('is_sale_season', False)),
            'product_avg_qty': p_avg_qty,
            'category_avg_qty': c_avg_qty,
        }
        X_future = pd.DataFrame([row])[FEATURES]
        pred_qty = float(np.clip(model.predict(X_future)[0], 0, None))
        forecast.append({"day": i, "forecast_qty": round(pred_qty, 1)})
        recent_lags.append(pred_qty)

    historical = [
        {"date": str(r['transaction_date'].date()), "actual_qty": float(r['daily_qty'])}
        for _, r in p_hist.tail(14).iterrows()
    ]

    return {
        "product_id": product_id,
        "forecast_horizon_days": forecast_days,
        "forecast": forecast,
        "historical": historical,
        "mape": round(_product_test_mape(product_id), 2),
        "model": "XGBoost (Global Multi-Product Regressor)",
    }
