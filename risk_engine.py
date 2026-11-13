import base64
import io
import json
import math
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from openai import OpenAI

from vlm_config import BASE_URL, API_KEY, MODEL

# Default to True to use the USAA GB10 VLM server; set USE_GB10_VLM=false for offline Mac CLIP
USE_GB10_VLM = os.getenv("USE_GB10_VLM", "true").lower() == "true"

DATA_DIR = Path("data")
_CLIENT = None
_CLIP_MODEL = None


def get_vlm_client() -> OpenAI:
    """Lazy-initialize the OpenAI client pointing to the USAA GB10 server."""
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=120)
    return _CLIENT


def get_image_root() -> Path:
    """Locate the flood image dataset on the server or fall back to local sample_images/ or datasets/."""
    candidates = [
        Path("/tmp/datathon/UTSA-flood-images-complete/images"),
        Path("tmp/datathon/UTSA-flood-images-complete/images"),
        Path("sample_images"),
        Path("datasets"),
    ]
    for path in candidates:
        if path.exists():
            # Check if directory contains images
            try:
                if any(path.rglob("*.jp*g")) or any(path.rglob("*.png")):
                    return path
            except Exception:
                pass
    Path("sample_images").mkdir(exist_ok=True)
    return Path("sample_images")


def _get_clip():
    """Lazy-load lightweight CLIP on Mac (MPS/CPU) only when offline fallback is enabled."""
    global _CLIP_MODEL
    if _CLIP_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _CLIP_MODEL = SentenceTransformer("sentence-transformers/clip-ViT-B-32")
    return _CLIP_MODEL


def state_plane_to_wgs84(x_ft: float, y_ft: float) -> tuple:
    """Convert NAD83 / Texas South Central (ftUS) (EPSG:2278) to WGS84 (lat, lon)."""
    a = 6378137.0
    f = 1.0 / 298.257222101
    e = math.sqrt(2 * f - f * f)
    to_meters = 1200.0 / 3937.0

    phi1 = math.radians(28 + 23 / 60)
    phi2 = math.radians(30 + 17 / 60)
    phi0 = math.radians(27 + 50 / 60)
    lambda0 = math.radians(-99.0)
    ef = 1968500.0 * to_meters
    nf = 13123333.33333333 * to_meters

    m1 = math.cos(phi1) / math.sqrt(1 - e * e * math.sin(phi1) ** 2)
    m2 = math.cos(phi2) / math.sqrt(1 - e * e * math.sin(phi2) ** 2)

    t1 = math.tan(math.pi / 4 - phi1 / 2) / (((1 - e * math.sin(phi1)) / (1 + e * math.sin(phi1))) ** (e / 2))
    t2 = math.tan(math.pi / 4 - phi2 / 2) / (((1 - e * math.sin(phi2)) / (1 + e * math.sin(phi2))) ** (e / 2))
    t0 = math.tan(math.pi / 4 - phi0 / 2) / (((1 - e * math.sin(phi0)) / (1 + e * math.sin(phi0))) ** (e / 2))

    n = math.log(m1 / m2) / math.log(t1 / t2)
    F = m1 / (n * (t1 ** n))
    rho0 = a * F * (t0 ** n)

    x_m = x_ft * to_meters
    y_m = y_ft * to_meters
    x_rel = x_m - ef
    y_rel = rho0 - (y_m - nf)

    rho = math.hypot(x_rel, y_rel) * (1 if n > 0 else -1)
    theta = math.atan2(x_rel, y_rel)

    t = (rho / (a * F)) ** (1.0 / n)
    lam = theta / n + lambda0

    phi = math.pi / 2 - 2 * math.atan(t)
    for _ in range(5):
        con = e * math.sin(phi)
        phi = math.pi / 2 - 2 * math.atan(t * (((1 - con) / (1 + con)) ** (e / 2)))

    return math.degrees(phi), math.degrees(lam)


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
    raw_lat = pd.to_numeric(out[lat_col], errors="coerce")
    raw_lon = pd.to_numeric(out[lon_col], errors="coerce")

    # If coordinates are in Texas State Plane (ftUS) (e.g. x ~ 2e6, y ~ 1.3e7), project to WGS84
    if (raw_lon.dropna().abs() > 1000).any() or (raw_lat.dropna().abs() > 1000).any():
        lats, lons = [], []
        for x_val, y_val in zip(raw_lon, raw_lat):
            if pd.notna(x_val) and pd.notna(y_val):
                lat_wgs, lon_wgs = state_plane_to_wgs84(float(x_val), float(y_val))
                lats.append(lat_wgs)
                lons.append(lon_wgs)
            else:
                lats.append(np.nan)
                lons.append(np.nan)
        out["lat"] = lats
        out["lon"] = lons
    else:
        out["lat"] = raw_lat
        out["lon"] = raw_lon

    return out.dropna(subset=["lat", "lon"])


def load_sa_datasets():
    """Load San Antonio CSVs (Low Water Crossings, TCEQ SSO, SmartSA Streetlights)."""
    # 1. CoSAGIS Low Water Crossings (check root dir and DATA_DIR)
    lwc_files = (
        list(Path(".").glob("*crossing*.csv"))
        + list(Path(".").glob("*lwc*.csv"))
        + list(Path(".").glob("*LowWater*.csv"))
        + list(DATA_DIR.glob("*crossing*.csv"))
        + list(DATA_DIR.glob("*lwc*.csv"))
    )
    if lwc_files:
        lwc = _normalize_coords(pd.read_csv(lwc_files[0]))
        name_col = next((c for c in lwc.columns if "name" in c.lower() or "street" in c.lower()), None)
        lwc["label"] = lwc[name_col] if name_col else "CoSAGIS Low Water Crossing"
        lwc["name"] = lwc["label"]
    else:
        lwc = pd.DataFrame([
            {"label": "Leon Creek @ Hausman Rd", "name": "Leon Creek @ Hausman Rd", "lat": 29.5518, "lon": -98.6184},
            {"label": "Salado Creek @ Eisenhauer Rd", "name": "Salado Creek @ Eisenhauer Rd", "lat": 29.4932, "lon": -98.4231},
            {"label": "Alazan Creek @ W Commerce St", "name": "Alazan Creek @ W Commerce St", "lat": 29.4275, "lon": -98.5236},
            {"label": "Martinez Creek @ Fredericksburg Rd", "name": "Martinez Creek @ Fredericksburg Rd", "lat": 29.4521, "lon": -98.5162},
        ])

    # 2. TCEQ Sanitary Sewer Overflows
    sso_files = list(DATA_DIR.glob("*sewer*.csv")) + list(DATA_DIR.glob("*sso*.csv")) + list(Path(".").glob("*sso*.csv"))
    if sso_files:
        sso = _normalize_coords(pd.read_csv(sso_files[0]))
        sso["label"] = "TCEQ Historical Sewer Overflow"
        sso["name"] = sso["label"]
    else:
        sso = pd.DataFrame([
            {"label": "San Pedro Creek Lift Station SSO", "name": "San Pedro Creek Lift Station SSO", "lat": 29.4189, "lon": -98.4981},
            {"label": "Apache Creek Sewer Main SSO", "name": "Apache Creek Sewer Main SSO", "lat": 29.4115, "lon": -98.5320},
            {"label": "Brooks Innovation Zone Main SSO", "name": "Brooks Innovation Zone Main SSO", "lat": 29.3468, "lon": -98.4385},
        ])

    # 3. SmartSA Streetlight Sensors (load DT, Medical, Brooks flood CSVs if present)
    dt_f = Path("DT_flood_data.csv")
    med_f = Path("Medical_flood_data.csv")
    brooks_f = Path("Brooks_flood_data.csv")
    brooks_zip = Path("Brooks_flood_data.csv.zip")

    if dt_f.exists() and med_f.exists():
        sensor_rows = []
        for name, p, default_lat, default_lon in [
            ("Downtown Innovation Zone", dt_f, 29.421882, -98.485166),
            ("Medical Center Innovation Zone", med_f, 29.511164, -98.586094),
            ("Brooks Innovation Zone", brooks_f, 29.345167, -98.451583),
        ]:
            water_in, lat_s, lon_s = 1.0, default_lat, default_lon
            try:
                if p.exists():
                    df_s = pd.read_csv(p)
                    last = df_s.dropna(subset=["DistToDF_ft", "DistToWL_ft"]).iloc[-1]
                    water_in = round(max(0.0, float(last["DistToDF_ft"] - last["DistToWL_ft"]) * 12.0), 1)
                    lat_s = float(last["LAT"]) if "LAT" in last and pd.notna(last["LAT"]) else default_lat
                    lon_s = float(last["LONG"]) if "LONG" in last and pd.notna(last["LONG"]) else default_lon
                elif p == brooks_f and brooks_zip.exists():
                    import zipfile
                    with zipfile.ZipFile(brooks_zip) as z:
                        with z.open("Brooks_flood_data.csv") as f:
                            df_s = pd.read_csv(f)
                            last = df_s.dropna(subset=["DistToDF_ft", "DistToWL_ft"]).iloc[-1]
                            water_in = round(max(0.0, float(last["DistToDF_ft"] - last["DistToWL_ft"]) * 12.0), 1)
                            lat_s = float(last["LAT"]) if "LAT" in last and pd.notna(last["LAT"]) else default_lat
                            lon_s = float(last["LONG"]) if "LONG" in last and pd.notna(last["LONG"]) else default_lon
            except Exception:
                pass

            sensor_rows.append({
                "label": name,
                "zone": name,
                "lat": lat_s,
                "lon": lon_s,
                "flood_level_in": water_in,
            })
        sl = pd.DataFrame(sensor_rows)
    else:
        sl_files = list(DATA_DIR.glob("*street*.csv")) + list(DATA_DIR.glob("*light*.csv")) + list(Path(".").glob("*street*.csv"))
        if sl_files:
            sl = _normalize_coords(pd.read_csv(sl_files[0]))
            sl["label"] = "SmartSA Streetlight Sensor"
            sl["zone"] = sl["label"]
            flood_col = next((c for c in sl.columns if "flood" in c.lower() or "water" in c.lower()), None)
            sl["flood_level_in"] = pd.to_numeric(sl[flood_col], errors="coerce").fillna(0.0) if flood_col else 1.2
        else:
            sl = pd.DataFrame([
                {"label": "Downtown Innovation Zone", "zone": "Downtown Innovation Zone", "lat": 29.4241, "lon": -98.4936, "flood_level_in": 1.0},
                {"label": "Medical Center Innovation Zone", "zone": "Medical Center Innovation Zone", "lat": 29.5074, "lon": -98.5752, "flood_level_in": 0.2},
                {"label": "Brooks Innovation Zone", "zone": "Brooks Innovation Zone", "lat": 29.3482, "lon": -98.4351, "flood_level_in": 12.4},
            ])

    return lwc, sso, sl


load_sa_layers = load_sa_datasets


def encode_resized_image(image_path: str, max_dim: int = 1024) -> str:
    """Resize image to max 1024px and encode as JPEG base64 data URL."""
    p = Path(image_path).expanduser().resolve()
    with Image.open(p) as img:
        img = img.convert("RGB")
        img.thumbnail((max_dim, max_dim))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def score_image(image_path: str) -> dict:
    """Score a photo for Dry vs. Flooded using Qwen3-VL on the USAA GB10 server."""
    if USE_GB10_VLM:
        client = get_vlm_client()
        data_url = encode_resized_image(image_path)
        prompt = (
            "Analyze this street or waterway photo for flood and public safety hazards. "
            "Respond ONLY with valid JSON using these exact keys:\n"
            '{"status": "DRY" | "MINOR_PONDING" | "FLOODED", '
            '"visual_severity": <integer 0 to 100>, '
            '"estimated_water_depth": "<string, e.g. None, 1-3 inches, >1 foot>", '
            '"sewage_or_debris_visible": <true or false>, '
            '"summary": "<one concise sentence explaining visible evidence>"}'
        )

        response = client.chat.completions.create(
            model=MODEL,
            temperature=0.001,
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )

        raw = (response.choices[0].message.content or "").strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
                return {
                    "status": parsed.get("status", "UNKNOWN"),
                    "visual_severity": int(parsed.get("visual_severity", 50)),
                    "estimated_water_depth": parsed.get("estimated_water_depth", "Uncertain"),
                    "sewage_risk_visual": bool(parsed.get("sewage_or_debris_visible", False)),
                    "summary": parsed.get("summary", raw),
                    "engine": f"{MODEL} (USAA GB10)",
                }
            except (json.JSONDecodeError, ValueError):
                pass

        return {
            "status": "UNKNOWN",
            "visual_severity": 50,
            "estimated_water_depth": "Uncertain",
            "sewage_risk_visual": False,
            "summary": raw,
            "engine": f"{MODEL} (USAA GB10)",
        }

    # Offline fallback: Local Mac CLIP zero-shot scoring
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

    exp_s = np.exp((sims[:3] - np.max(sims[:3])) * 15.0)
    probs = exp_s / exp_s.sum()
    p_dry, p_pond, p_flood = float(probs[0]), float(probs[1]), float(probs[2])

    severity = int(round(p_pond * 45 + p_flood * 100))
    status = "FLOODED" if p_flood >= 0.50 else ("MINOR PONDING" if severity >= 35 else "DRY")

    return {
        "status": status,
        "visual_severity": severity,
        "estimated_water_depth": "Estimated via CLIP",
        "sewage_risk_visual": bool(sims[3] > sims[0]),
        "summary": f"Local Vision — Dry: {p_dry*100:.0f}%, Ponding: {p_pond*100:.0f}%, Flooded: {p_flood*100:.0f}%.",
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
            "estimated_water_depth": "No photo uploaded",
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


SA_PRESET_AREAS = [
    {
        "name": "Alazan Creek @ W Commerce St",
        "zone": "West Side / Historic Flood Basin",
        "lat": 29.4275,
        "lon": -98.5236,
        "description": "Historic flash flood basin near downtown with high runoff risk.",
    },
    {
        "name": "Leon Creek @ Hausman Rd",
        "zone": "Northwest / UTSA Corridor",
        "lat": 29.5518,
        "lon": -98.6184,
        "description": "Critical low water crossing near UTSA and Loop 1604.",
    },
    {
        "name": "Salado Creek @ Eisenhauer Rd",
        "zone": "Northeast Corridor",
        "lat": 29.4932,
        "lon": -98.4231,
        "description": "Major urban creek basin subject to rapid surge from upstream hills.",
    },
    {
        "name": "Olmos Basin / Olmos Dam",
        "zone": "Midtown / Central Retention Basin",
        "lat": 29.4715,
        "lon": -98.4776,
        "description": "Primary flood control reservoir protecting downtown San Antonio.",
    },
    {
        "name": "Downtown River Walk & San Pedro Creek",
        "zone": "Downtown Innovation Zone",
        "lat": 29.4241,
        "lon": -98.4936,
        "description": "Urban core monitored by SmartSA smart streetlight flood sensors.",
    },
    {
        "name": "Medical Center Innovation Zone",
        "zone": "South Texas Medical Center",
        "lat": 29.5074,
        "lon": -98.5752,
        "description": "Critical healthcare corridor with SmartSA environmental sensors.",
    },
    {
        "name": "Brooks Innovation Zone",
        "zone": "South Side / Mission Reach",
        "lat": 29.3482,
        "lon": -98.4351,
        "description": "South San Antonio urban revitalization zone with active flood telemetry.",
    },
    {
        "name": "Apache Creek @ Elmendorf Lake",
        "zone": "West Side / Our Lady of the Lake",
        "lat": 29.4289,
        "lon": -98.5398,
        "description": "West side drainage creek with historical sewer overflow records.",
    },
    {
        "name": "Martinez Creek @ Fredericksburg Rd",
        "zone": "Deco District / Midtown",
        "lat": 29.4521,
        "lon": -98.5162,
        "description": "Dense urban intersection near low water crossing.",
    },
    {
        "name": "Espada Dam & Mission Espada",
        "zone": "South River Corridor",
        "lat": 29.3512,
        "lon": -98.4725,
        "description": "Historic aqueduct and river bend in southern Bexar County.",
    },
]


def score_all_preset_areas() -> pd.DataFrame:
    """Compute and compare probability scores across all major San Antonio zones."""
    records = []
    for area in SA_PRESET_AREAS:
        res = compute_disaster_score(area["lat"], area["lon"], image_path=None)
        records.append({
            "Area Name": area["name"],
            "Zone / Region": area["zone"],
            "Disaster Probability (%)": res["total_score"],
            "Risk Tier": res["risk_tier"],
            "Nearest Low Water Crossing": res["nearest_lwc"],
            "Nearest Streetlight Sensor": res["nearest_sensor"],
            "Sewage Contamination Alert": "⚠️ Alert" if res["sewage_warning"] else "Normal",
            "Latitude": area["lat"],
            "Longitude": area["lon"],
        })
    df = pd.DataFrame(records).sort_values(by="Disaster Probability (%)", ascending=False)
    return df
