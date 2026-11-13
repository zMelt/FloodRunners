import math
import os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

# Toggle this to True when running on the GB10 workstation with vLLM active
USE_GB10_VLM = os.getenv("USE_GB10_VLM", "false").lower() == "true"

DATA_DIR = Path("data")
_CLIP_MODEL = None


def _get_clip():
    """Lazy-load lightweight CLIP on Mac (MPS/CPU) for fast dry vs. flooded scoring."""
    global _CLIP_MODEL
    if _CLIP_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _CLIP_MODEL = SentenceTransformer("sentence-transformers/clip-ViT-B-32")
    return _CLIP_MODEL


def _haversine_m(lat1, lon1, lat2, lon2):
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _normalize_coords(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize common lat/lon column names across CoSAGIS, TCEQ, and OpenDataSA CSVs."""
    cols = {c.lower().strip(): c for c in df.columns}
    lat_col = next((cols[k] for k in ["lat", "latitude", "y", "point_y"] if k in cols), None)
    lon_col = next((cols[k] for k in ["lon", "lng", "longitude", "x", "point_x"] if k in cols), None)
    if not lat_col or not lon_col:
        return pd.DataFrame()
    out = df.copy()
    out["lat"] = pd.to_numeric(out[lat_col], errors="coerce")
    out["lon"] = pd.to_numeric(out[lon_col], errors="coerce")
    return out.dropna(subset=["lat", "lon"])


def load_sa_datasets():
    """Load downloaded San Antonio CSVs from data/ with fallback Bexar County coordinates."""
    # 1. Low Water Crossings
    lwc_files = list(DATA_DIR.glob("*crossing*.csv")) + list(DATA_DIR.glob("*lwc*.csv"))
    if lwc_files:
        lwc = _normalize_coords(pd.read_csv(lwc_files[0]))
        name_col = next((c for c in lwc.columns if "name" in c.lower() or "street" in c.lower()), None)
        lwc["label"] = lwc[name_col] if name_col else "CoSAGIS Low Water Crossing"
    else:
        lwc = pd.DataFrame([
            {"label": "Leon Creek @ Hausman Rd", "lat": 29.5518, "lon": -98.6184},
            {"label": "Salado Creek @ Eisenhauer Rd", "lat": 29.4932, "lon": -98.4231},
            {"label": "Alazan Creek @ W Commerce St", "lat": 29.4275, "lon": -98.5236},
            {"label": "Martinez Creek @ Fredericksburg Rd", "lat": 29.4521, "lon": -98.5162},
        ])

    # 2. TCEQ Sanitary Sewer Overflows
    sso_files = list(DATA_DIR.glob("*sewer*.csv")) + list(DATA_DIR.glob("*sso*.csv"))
    if sso_files:
        sso = _normalize_coords(pd.read_csv(sso_files[0]))
        sso["label"] = "TCEQ Historical Sewer Overflow"
    else:
        sso = pd.DataFrame([
            {"label": "San Pedro Creek Lift Station SSO", "lat": 29.4189, "lon": -98.4981},
            {"label": "Apache Creek Sewer Main SSO", "lat": 29.4115, "lon": -98.5320},
            {"label": "Brooks Innovation Zone Main SSO", "lat": 29.3468, "lon": -98.4385},
        ])

    # 3. SA Smart Streetlight Pilot Sensors
    sl_files = list(DATA_DIR.glob("*street*.csv")) + list(DATA_DIR.glob("*light*.csv"))
    if sl_files:
        sl = _normalize_coords(pd.read_csv(sl_files[0]))
        sl["label"] = "SmartSA Streetlight Sensor"
        flood_col = next((c for c in sl.columns if "flood" in c.lower() or "water" in c.lower()), None)
        sl["flood_level_in"] = pd.to_numeric(sl[flood_col], errors="coerce").fillna(0.0) if flood_col else 1.2
    else:
        sl = pd.DataFrame([
            {"label": "Downtown Innovation Zone", "lat": 29.4241, "lon": -98.4936, "flood_level_in": 1.8},
            {"label": "Medical Center Innovation Zone", "lat": 29.5074, "lon": -98.5752, "flood_level_in": 0.0},
            {"label": "Brooks Innovation Zone", "lat": 29.3482, "lon": -98.4351, "flood_level_in": 3.4},
        ])

    return lwc, sso, sl


def score_image(image_path: str) -> dict:
    """Score a photo for Dry vs. Flooded either locally (Mac CLIP) or remotely (GB10 Qwen3-VL)."""
    if USE_GB10_VLM:
        from texas_flood import ask_image
        ans = ask_image(
            image_path,
            "Classify as DRY, PONDING, or FLOODED and give a 0-100 severity number plus one sentence why."
        )
        is_flood = "FLOOD" in ans.upper()
        sev = 85 if is_flood else (45 if "POND" in ans.upper() else 10)
        return {
            "status": "FLOODED" if is_flood else "DRY / MINOR",
            "visual_severity": sev,
            "sewage_risk_visual": "SEWAGE" in ans.upper() or "MURKY" in ans.upper(),
            "summary": ans,
            "engine": "Qwen3-VL-8B (GB10)",
        }

    # Fast local Mac execution using CLIP zero-shot image-text similarity
    model = _get_clip()
    prompts = [
        "a dry paved street or dry neighborhood with no standing water",
        "minor rainwater puddles on an asphalt street",
        "severe street flooding, submerged road, deep muddy floodwater",
        "murky brown contaminated sewage runoff or clogged storm drain with debris",
    ]
    with Image.open(image_path) as img:
        img_emb = model.encode(img.convert("RGB"), normalize_embeddings=True)
    txt_emb = model.encode(prompts, normalize_embeddings=True)
    sims = np.dot(txt_emb, img_emb)

    # Softmax over dry vs. ponding vs. flooded
    exp_s = np.exp((sims[:3] - np.max(sims[:3])) * 15.0)
    probs = exp_s / exp_s.sum()
    p_dry, p_pond, p_flood = float(probs[0]), float(probs[1]), float(probs[2])

    severity = int(round(p_pond * 45 + p_flood * 100))
    if p_flood >= 0.50:
        status = "FLOODED"
    elif p_pond >= 0.40 or severity >= 35:
        status = "MINOR PONDING"
    else:
        status = "DRY"

    murky_flag = bool(sims[3] > sims[0])
    summary = (
        f"Local Vision Confidence — Dry: {p_dry*100:.0f}%, Ponding: {p_pond*100:.0f}%, "
        f"Flooded: {p_flood*100:.0f}%."
    )
    return {
        "status": status,
        "visual_severity": severity,
        "sewage_risk_visual": murky_flag,
        "summary": summary,
        "engine": "CLIP ViT-B/32 (Local Mac)",
    }


def compute_disaster_score(lat: float, lon: float, image_path: str = None) -> dict:
    """Combine visual AI + San Antonio geospatial layers into a 0-100 disaster probability score."""
    lwc, sso, sl = load_sa_datasets()

    # 1. Visual Evidence (0 - 40 pts)
    if image_path:
        vis = score_image(image_path)
        vis_pts = (vis["visual_severity"] / 100.0) * 40.0
    else:
        vis = {
            "status": "NO PHOTO",
            "visual_severity": 0,
            "sewage_risk_visual": False,
            "summary": "No photo supplied; score computed from San Antonio floodplain & sensor data.",
            "engine": "Geospatial Only",
        }
        vis_pts = 8.0

    # 2. Low Water Crossing Prior (0 - 25 pts)
    lwc_dists = [_haversine_m(lat, lon, r.lat, r.lon) for _, r in lwc.iterrows()]
    min_lwc = min(lwc_dists)
    lwc_name = lwc.iloc[int(np.argmin(lwc_dists))]["label"]
    lwc_pts = 25.0 if min_lwc <= 250 else (15.0 if min_lwc <= 1000 else 4.0)

    # 3. Smart Streetlight Flood Sensor Telemetry (0 - 20 pts)
    sl_dists = [_haversine_m(lat, lon, r.lat, r.lon) for _, r in sl.iterrows()]
    min_sl = min(sl_dists)
    nearest_sl = sl.iloc[int(np.argmin(sl_dists))]
    water_in = float(nearest_sl["flood_level_in"]) if min_sl <= 2500 else 0.0
    sl_pts = min(20.0, (water_in / 4.0) * 20.0) if water_in > 0 else 4.0

    # 4. TCEQ Sanitary Sewer Overflow Prior (0 - 15 pts)
    sso_dists = [_haversine_m(lat, lon, r.lat, r.lon) for _, r in sso.iterrows()]
    min_sso = min(sso_dists)
    sso_name = sso.iloc[int(np.argmin(sso_dists))]["label"]
    sso_pts = 15.0 if min_sso <= 500 else (8.0 if min_sso <= 1500 else 2.0)
    sewage_alert = (min_sso <= 500) or vis["sewage_risk_visual"]

    total = round(min(100.0, vis_pts + lwc_pts + sl_pts + sso_pts), 1)
    tier = "HIGH RISK — AVOID AREA" if total >= 70 else ("MODERATE RISK" if total >= 40 else "LOW RISK")

    return {
        "total_score": total,
        "risk_tier": tier,
        "breakdown": {
            "Visual AI Evidence (40 max)": round(vis_pts, 1),
            "Low Water Crossing Prior (25 max)": round(lwc_pts, 1),
            "Streetlight Sensor Telemetry (20 max)": round(sl_pts, 1),
            "Sewage Overflow Prior (15 max)": round(sso_pts, 1),
        },
        "vlm": vis,
        "nearest_lwc": f"{lwc_name} ({int(min_lwc)}m)",
        "nearest_sensor": f"{nearest_sl['label']} ({water_in:.1f} in., {int(min_sl)}m)",
        "sewage_warning": sewage_alert,
        "nearest_sso": f"{sso_name} ({int(min_sso)}m)",
    }