from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_RADIUS_M = 6371000
HALF_LIFE_DAYS = 257
BANDWIDTH_M = 400
RADIUS_M = 1200
GRID_DECIMALS = 2
HOUR_BANDWIDTH = 4

REFERENCE_TZ = ZoneInfo("America/Chicago")

SEVERITY_DEFAULT_BY_TYPE = {
    "HOMICIDE": 100, "CRIMINAL SEXUAL ASSAULT": 92, "SEX OFFENSE": 60,
    "KIDNAPPING": 88, "HUMAN TRAFFICKING": 88, "OFFENSE INVOLVING CHILDREN": 70,
    "ROBBERY": 68, "ARSON": 65, "ASSAULT": 50, "BATTERY": 45,
    "STALKING": 45, "WEAPONS VIOLATION": 45, "INTIMIDATION": 40, "BURGLARY": 40,
    "MOTOR VEHICLE THEFT": 32, "THEFT": 28, "NARCOTICS": 22, "OTHER NARCOTIC VIOLATION": 20,
    "CRIMINAL DAMAGE": 18, "DECEPTIVE PRACTICE": 18, "CRIMINAL TRESPASS": 14,
    "OTHER OFFENSE": 14, "PROSTITUTION": 12, "PUBLIC PEACE VIOLATION": 12,
    "PUBLIC INDECENCY": 12, "OBSCENITY": 15, "INTERFERENCE WITH PUBLIC OFFICER": 25,
    "GAMBLING": 6, "LIQUOR LAW VIOLATION": 6, "CONCEALED CARRY LICENSE VIOLATION": 8,
    "NON-CRIMINAL": 5,
}

SEVERITY_OVERRIDE = {
    ("BATTERY", "DOMESTIC BATTERY SIMPLE"): 42,
    ("BATTERY", "SIMPLE"): 38,
    ("BATTERY", "AGGRAVATED - OTHER DANGEROUS WEAPON"): 75,
    ("BATTERY", "AGGRAVATED - HANDGUN"): 85,
    ("THEFT", "$500 AND UNDER"): 16,
    ("THEFT", "OVER $500"): 30,
    ("THEFT", "RETAIL THEFT"): 12,
    ("THEFT", "FROM BUILDING"): 25,
    ("THEFT", "THEFT FROM MOTOR VEHICLE"): 22,
    ("THEFT", "FROM PERSON"): 40,
    ("CRIMINAL DAMAGE", "TO VEHICLE"): 14,
    ("CRIMINAL DAMAGE", "TO PROPERTY"): 16,
    ("ASSAULT", "SIMPLE"): 28,
    ("ASSAULT", "AGGRAVATED - HANDGUN"): 78,
    ("ASSAULT", "AGGRAVATED - KNIFE / CUTTING INSTRUMENT"): 70,
    ("WEAPONS VIOLATION", "UNLAWFUL POSSESSION - HANDGUN"): 50,
    ("BURGLARY", "FORCIBLE ENTRY"): 52,
    ("BURGLARY", "UNLAWFUL ENTRY"): 38,
    ("BURGLARY", "BURGLARY FROM MOTOR VEHICLE"): 30,
    ("OTHER OFFENSE", "TELEPHONE THREAT"): 20,
    ("OTHER OFFENSE", "HARASSMENT BY TELEPHONE"): 15,
    ("OTHER OFFENSE", "HARASSMENT BY ELECTRONIC MEANS"): 15,
    ("OTHER OFFENSE", "VIOLATE ORDER OF PROTECTION"): 35,
    ("DECEPTIVE PRACTICE", "FINANCIAL IDENTITY THEFT OVER $ 300"): 25,
    ("DECEPTIVE PRACTICE", "CREDIT CARD FRAUD"): 20,
    ("ROBBERY", "ARMED - HANDGUN"): 88,
    ("ROBBERY", "STRONG ARM - NO WEAPON"): 55,
    ("MOTOR VEHICLE THEFT", "ATTEMPT - AUTOMOBILE"): 20,
}

GLOBAL_FALLBACK_SEVERITY = 15

CLIP_BOUND = 9.9818

LOW_MAX = 25
MEDIUM_MAX = 50
HIGH_MAX = 75


def score_severity(primary_type, description):
    key = (primary_type, description)
    if key in SEVERITY_OVERRIDE:
        return SEVERITY_OVERRIDE[key]
    return SEVERITY_DEFAULT_BY_TYPE.get(primary_type, GLOBAL_FALLBACK_SEVERITY)


def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def hour_circular_distance(hour_a, hour_b):
    diff = np.abs(hour_a - hour_b) % 24
    return np.minimum(diff, 24 - diff)


def to_reference_naive(dt):
    if dt.tzinfo is not None:
        return dt.astimezone(REFERENCE_TZ).replace(tzinfo=None)
    return dt


def compute_risk_raw(query_lat, query_lon, query_hour, events, reference_date,
                      half_life_days=HALF_LIFE_DAYS, bandwidth_m=BANDWIDTH_M,
                      radius_m=RADIUS_M, hour_bandwidth=HOUR_BANDWIDTH):
    reference_date = to_reference_naive(reference_date)
    candidates = events[events["Datetime"] <= reference_date]
    dist_m = haversine_m(query_lat, query_lon, candidates["lat_r"].values, candidates["lon_r"].values)
    nearby_mask = dist_m <= radius_m
    nearby = candidates[nearby_mask]
    nearby_dist = dist_m[nearby_mask]

    if len(nearby) == 0:
        return 0.0

    age_days = (reference_date - nearby["Datetime"]).dt.total_seconds() / 86400
    lam = np.log(2) / half_life_days
    w_time = np.exp(-lam * age_days)
    w_space = np.exp(-0.5 * (nearby_dist / bandwidth_m) ** 2)

    is_exact_midnight = (
        (nearby["Datetime"].dt.hour == 0)
        & (nearby["Datetime"].dt.minute == 0)
        & (nearby["Datetime"].dt.second == 0)
    )
    hour_dist = hour_circular_distance(nearby["hour"].values, query_hour)
    w_hour = np.exp(-0.5 * (hour_dist / hour_bandwidth) ** 2)
    w_hour = np.where(is_exact_midnight.values, 1.0, w_hour)

    return float((nearby["severity"].values * w_time.values * w_space * w_hour).sum())


def get_nearby_crime_count(query_lat, query_lon, events, reference_date, radius_m=RADIUS_M):
    reference_date = to_reference_naive(reference_date)
    candidates = events[events["Datetime"] <= reference_date]
    dist_m = haversine_m(query_lat, query_lon, candidates["lat_r"].values, candidates["lon_r"].values)
    return int((dist_m <= radius_m).sum())


def build_grid_risk_raw(events, reference_date, radius_m=RADIUS_M, bandwidth_m=BANDWIDTH_M,
                         half_life_days=HALF_LIFE_DAYS, hour_bandwidth=HOUR_BANDWIDTH):
    reference_date = to_reference_naive(reference_date)
    events_valid = events[events["Datetime"] <= reference_date]

    unique_cells = events_valid[["lat_r", "lon_r"]].drop_duplicates().reset_index(drop=True)
    unique_cells["cell_id"] = unique_cells["lat_r"].astype(str) + "_" + unique_cells["lon_r"].astype(str)

    coords_rad = np.radians(unique_cells[["lat_r", "lon_r"]].values)
    tree = BallTree(coords_rad, metric="haversine")
    neighbor_idx, neighbor_dist = tree.query_radius(
        coords_rad, r=radius_m / EARTH_RADIUS_M, return_distance=True
    )

    events_by_cell = {
        cell_id: group
        for cell_id, group in events_valid.assign(
            cell_id=events_valid["lat_r"].astype(str) + "_" + events_valid["lon_r"].astype(str)
        ).groupby("cell_id")
    }

    lam = np.log(2) / half_life_days
    rows = []

    for cell_index, cell_row in unique_cells.iterrows():
        neighbor_cell_ids = unique_cells.iloc[neighbor_idx[cell_index]]["cell_id"].values
        neighbor_dists_m = neighbor_dist[cell_index] * EARTH_RADIUS_M
        frames = []
        dist_lookup = {}
        for cid, dist in zip(neighbor_cell_ids, neighbor_dists_m):
            if cid in events_by_cell:
                group = events_by_cell[cid]
                frames.append(group)
                dist_lookup[cid] = dist

        if not frames:
            for h in range(24):
                rows.append({
                    "cell_id": cell_row["cell_id"], "lat_r": cell_row["lat_r"],
                    "lon_r": cell_row["lon_r"], "hour": h, "risk_raw": 0.0,
                    "nearby_crime_count": 0,
                })
            continue

        nearby = pd.concat(frames, ignore_index=False)
        nearby_dist_m = nearby["cell_id"].map(dist_lookup).values

        age_days = (reference_date - nearby["Datetime"]).dt.total_seconds().values / 86400
        w_time = np.exp(-lam * age_days)
        w_space = np.exp(-0.5 * (nearby_dist_m / bandwidth_m) ** 2)

        is_exact_midnight = (
            (nearby["Datetime"].dt.hour == 0)
            & (nearby["Datetime"].dt.minute == 0)
            & (nearby["Datetime"].dt.second == 0)
        ).values
        severity_vals = nearby["severity"].values
        candidate_hours = nearby["hour"].values

        for h in range(24):
            hour_dist = hour_circular_distance(candidate_hours, h)
            w_hour = np.exp(-0.5 * (hour_dist / hour_bandwidth) ** 2)
            w_hour = np.where(is_exact_midnight, 1.0, w_hour)
            risk_raw = float((severity_vals * w_time * w_space * w_hour).sum())
            rows.append({
                "cell_id": cell_row["cell_id"], "lat_r": cell_row["lat_r"],
                "lon_r": cell_row["lon_r"], "hour": h, "risk_raw": risk_raw,
                "nearby_crime_count": len(nearby),
            })

    return pd.DataFrame(rows)


def calibrate_clip_bound(events, reference_date, percentile=99):
    grid = build_grid_risk_raw(events, reference_date)
    return float(np.percentile(np.log1p(grid["risk_raw"]), percentile))


def predict_risk_score(query_lat, query_lon, query_datetime, events, reference_date,
                        clip_bound=CLIP_BOUND, half_life_days=HALF_LIFE_DAYS,
                        bandwidth_m=BANDWIDTH_M, radius_m=RADIUS_M, hour_bandwidth=HOUR_BANDWIDTH):
    query_datetime = to_reference_naive(query_datetime)
    risk_raw = compute_risk_raw(query_lat, query_lon, query_datetime.hour, events, reference_date,
                                 half_life_days, bandwidth_m, radius_m, hour_bandwidth)
    log_val = np.log1p(risk_raw)
    clipped = min(log_val, clip_bound)
    return float(100 * clipped / clip_bound)


def has_recent_nearby_incident(query_lat, query_lon, events, reference_date,
                                radius_m=300, recency_days=3, severity_threshold=50):
    reference_date = to_reference_naive(reference_date)
    candidates = events[events["Datetime"] <= reference_date]
    dist_m = haversine_m(query_lat, query_lon, candidates["lat_r"].values, candidates["lon_r"].values)
    age_days = (reference_date - candidates["Datetime"]).dt.total_seconds() / 86400

    mask = (
        (dist_m <= radius_m)
        & (age_days.values <= recency_days)
        & (candidates["severity"].values >= severity_threshold)
    )
    return bool(mask.sum() > 0)


def categorize_level(risk_score, recent_incident_flag):
    if recent_incident_flag:
        return "Very High"
    if risk_score <= LOW_MAX:
        return "Low"
    if risk_score <= MEDIUM_MAX:
        return "Medium"
    if risk_score <= HIGH_MAX:
        return "High"
    return "Very High"
