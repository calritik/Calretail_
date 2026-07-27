"""
CalRetail — Route and fleet optimisation.

Ported from ``notebooks/capabilities/12_route_optimisation.ipynb``. The notebook remains the readable
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
    global _READY, _BUILDING, math, stores, whs, city_coords, wh_lat, wh_lon, coords, all_locations, dist_matrix, i, loc1, j, loc2
    if _READY or _BUILDING:
        return
    _BUILDING = True
    try:
        import math

        # Load store coordinates
        stores = load_table('stores')
        whs = load_table('warehouses')

        # Real coordinates for all 20 Indian cities used in this dataset (see
        # notebooks/generate_data.py CITIES). Previously only 9 cities were mapped
        # (and under a misspelling — "Bangalore" instead of the real "Bengaluru" —
        # so more than half of all cities silently fell back to Bangalore's coords).
        city_coords = {
            'Mumbai': (19.0760, 72.8777), 'Delhi': (28.7041, 77.1025), 'Bengaluru': (12.9716, 77.5946),
            'Hyderabad': (17.3850, 78.4867), 'Chennai': (13.0827, 80.2707), 'Kolkata': (22.5726, 88.3639),
            'Pune': (18.5204, 73.8567), 'Ahmedabad': (23.0225, 72.5714), 'Jaipur': (26.9124, 75.7873),
            'Lucknow': (26.8467, 80.9462), 'Surat': (21.1702, 72.8311), 'Kochi': (9.9312, 76.2673),
            'Chandigarh': (30.7333, 76.7794), 'Bhopal': (23.2599, 77.4126), 'Indore': (22.7196, 75.8577),
            'Nagpur': (21.1458, 79.0882), 'Patna': (25.5941, 85.1376), 'Vadodara': (22.3072, 73.1812),
            'Ludhiana': (30.9010, 75.8573), 'Agra': (27.1767, 78.0081),
        }

        stores['latitude'] = stores['city'].apply(get_lat)
        stores['longitude'] = stores['city'].apply(get_lon)
        whs['latitude'] = whs['city'].apply(get_lat)
        whs['longitude'] = whs['city'].apply(get_lon)

        wh_lat, wh_lon = whs.iloc[0]['latitude'], whs.iloc[0]['longitude']
        coords = stores[['store_id', 'latitude', 'longitude']].values.tolist()

        # Build distance matrix from warehouse origin
        all_locations = [('Warehouse', wh_lat, wh_lon)] + [(c[0], c[1], c[2]) for c in coords[:6]] # subset
        dist_matrix = {}
        for i, loc1 in enumerate(all_locations):
            for j, loc2 in enumerate(all_locations):
                dist_matrix[(loc1[0], loc2[0])] = haversine(loc1[1], loc1[2], loc2[1], loc2[2])
        
        print(f"Computed travel matrix. Size: {len(all_locations)} stops.")

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
    for _name in ('math', 'stores', 'whs', 'city_coords', 'wh_lat', 'wh_lon', 'coords', 'all_locations', 'dist_matrix', 'i', 'loc1', 'j', 'loc2'):
        globals().pop(_name, None)


def get_lat(c): return city_coords.get(c, (22.9734, 78.6569))[0]  # fallback: geographic centre of India


def get_lon(c): return city_coords.get(c, (22.9734, 78.6569))[1]


# Haversine function
def haversine(lat1, lon1, lat2, lon2):
    _init()
    r = 6371.0 # Earth radius km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def two_opt(route, dist_matrix):
    """Standard 2-opt local search: repeatedly reverses a segment of the route
    whenever doing so shortens total distance, until no improving swap remains."""
    _init()
    best = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                a, b, c, d = best[i - 1], best[i], best[j], best[j + 1]
                delta = (dist_matrix[(a, c)] + dist_matrix[(b, d)]) - (dist_matrix[(a, b)] + dist_matrix[(c, d)])
                if delta < -1e-9:
                    best[i:j + 1] = best[i:j + 1][::-1]
                    improved = True
    return best


def solve_delivery_route(warehouse_id=None):
    _init()
    global city_coords, stores, whs
    import pandas as pd
    import numpy as np
    import math
    
    # Earth radius
    r_earth = 6371.0
    
    def haversine(lat1, lon1, lat2, lon2):
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        return 2 * r_earth * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
    # Get warehouse coords
    if warehouse_id:
        wh_row = whs[whs['warehouse_id'] == warehouse_id]
    else:
        wh_row = whs
        
    if wh_row.empty:
        wh_row = whs
        
    wh_lat = wh_row.iloc[0]['latitude']
    wh_lon = wh_row.iloc[0]['longitude']
    wh_actual_id = wh_row.iloc[0]['warehouse_id']
    wh_city = wh_row.iloc[0]['city']
    
    # Load shipments and orders to find destination stores
    base_path = Path().resolve()
    while not (base_path / 'data').exists() and base_path.parent != base_path:
        base_path = base_path.parent
    processed_dir = base_path / 'data'
    
    sh = load_table('shipments')
    ord_df = load_table('orders')
    
    merged = pd.merge(sh[['warehouse_id', 'order_id']], ord_df[['order_id', 'store_id']], on='order_id', how='inner')
    wh_shipments = merged[merged['warehouse_id'] == wh_actual_id]
    
    # Find stores with the most shipments from this warehouse
    # Ties are broken by store id, not left to chance. Shipment counts bunch
    # heavily — the sixth-busiest store for a warehouse sits in a group of a
    # dozen sharing the same count — so head(6) on an unordered tie returns
    # whichever six pandas happened to hash first. The route, its distance and
    # its saving all moved when an unrelated change altered that order. Sorting
    # by count then id makes the answer reproducible.
    store_counts = (wh_shipments['store_id'].value_counts()
                    .rename_axis('store_id').reset_index(name='n')
                    .sort_values(['n', 'store_id'], ascending=[False, True]))
    top_store_ids = store_counts['store_id'].head(6).tolist()
    
    if not top_store_ids:
        top_store_ids = stores['store_id'].head(6).tolist()
        
    route_stores = stores[stores['store_id'].isin(top_store_ids)].copy()
    
    # Build locations list
    all_locations = [('Warehouse', wh_lat, wh_lon, wh_city)]
    for idx, row in route_stores.iterrows():
        all_locations.append((row['store_id'], row['latitude'], row['longitude'], row['city']))
        
    # Calculate distance matrix between all locations
    dist_matrix = {}
    for i, loc1 in enumerate(all_locations):
        for j, loc2 in enumerate(all_locations):
            dist_matrix[(loc1[0], loc2[0])] = haversine(loc1[1], loc1[2], loc2[1], loc2[2])
            
    # 1. Nearest-neighbour construction
    visited = ['Warehouse']
    stops_left = [loc[0] for loc in all_locations if loc[0] != 'Warehouse']
    
    current = 'Warehouse'
    while stops_left:
        next_stop = min(stops_left, key=lambda s: dist_matrix[(current, s)])
        visited.append(next_stop)
        stops_left.remove(next_stop)
        current = next_stop
    visited.append('Warehouse')

    nn_distance = sum(dist_matrix[(visited[i], visited[i + 1])] for i in range(len(visited) - 1))

    # 2. 2-opt local-search improvement over the nearest-neighbour tour
    visited = two_opt(visited, dist_matrix)
    total_dist = sum(dist_matrix[(visited[i], visited[i + 1])] for i in range(len(visited) - 1))
    
    # Calculate baseline distance (unoptimized alphabetical visiting sequence)
    baseline_stores = sorted([loc[0] for loc in all_locations if loc[0] != 'Warehouse'])
    baseline_visited = ['Warehouse'] + baseline_stores + ['Warehouse']
    baseline_dist = 0
    current_b = 'Warehouse'
    for next_b in baseline_visited[1:]:
        baseline_dist += dist_matrix[(current_b, next_b)]
        current_b = next_b
        
    # Calculate total active orders/shipments
    total_orders = int(wh_shipments[wh_shipments['store_id'].isin(top_store_ids)].shape[0])
    if total_orders == 0:
        total_orders = 75
        
    # Build route nodes coordinate dict list for UI
    route_nodes = []
    for stop_id in visited:
        if stop_id == 'Warehouse':
            continue
        st_row = route_stores[route_stores['store_id'] == stop_id]
        if not st_row.empty:
            store_order_count = int(wh_shipments[wh_shipments['store_id'] == stop_id].shape[0])
            if store_order_count == 0:
                store_order_count = 12
            route_nodes.append({
                "lat": float(st_row.iloc[0]['latitude']),
                "lng": float(st_row.iloc[0]['longitude']),
                "city": str(st_row.iloc[0]['city']),
                "items": store_order_count
            })
            
    origin_coords = {
        "lat": float(wh_lat),
        "lng": float(wh_lon)
    }
    
    return {
        "origin_warehouse": wh_actual_id,
        "distance_km": round(total_dist, 2),
        "nearest_neighbour_distance_km": round(nn_distance, 2),
        "baseline_distance_km": round(baseline_dist, 2),
        "stops_count": len(visited) - 2,
        "route_order": visited,
        "total_orders": total_orders,
        "route": route_nodes,
        "origin": origin_coords
    }
