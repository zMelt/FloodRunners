import uuid
from datetime import datetime
from pathlib import Path
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from risk_engine import compute_disaster_score, get_image_root, load_sa_datasets

st.set_page_config(page_title="311SA Flood & Hazard Guard", page_icon="🌊", layout="wide")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
IMAGE_ROOT = get_image_root()

if "tickets" not in st.session_state:
    st.session_state.tickets = [
        {
            "case_id": "311SA-2026-8841",
            "time": "11:42 AM",
            "location": "Alazan Creek @ W Commerce St",
            "lat": 29.4275,
            "lon": -98.5236,
            "score": 82.5,
            "status": "AI Verified — Public Works Dispatched",
            "sewage_alert": True,
            "summary": "Standing water covering curb lane near storm inlet.",
        }
    ]

st.title("🌊 311SA Flood & Hazard Guard — Citizen Portal")
st.caption("Powered by Qwen3-VL on USAA NVIDIA GB10 + San Antonio Open Data & Smart Streetlights")

col_map, col_form = st.columns([1.3, 1.0], gap="large")

# --- LEFT COLUMN: INTERACTIVE SAN ANTONIO RISK MAP ---
with col_map:
    st.subheader("Live San Antonio Flood & Infrastructure Map")
    lwc, sso, sl = load_sa_datasets()
    m = folium.Map(location=[29.4241, -98.4936], zoom_start=11, tiles="CartoDB positron")

    # CoSAGIS Low Water Crossings
    for _, r in lwc.head(150).iterrows():
        folium.CircleMarker(
            [r["lat"], r["lon"]],
            radius=5,
            color="#d9534f",
            fill=True,
            popup=f"<b>Low Water Crossing:</b> {r['label']}",
        ).add_to(m)

    # SmartSA Streetlight Flood Sensors
    for _, r in sl.head(150).iterrows():
        folium.CircleMarker(
            [r["lat"], r["lon"]],
            radius=6,
            color="#0275d8",
            fill=True,
            popup=f"<b>Smart Streetlight:</b> {r['label']}<br>Water Level: {r['flood_level_in']} in.",
        ).add_to(m)

    # TCEQ Sanitary Sewer Overflows
    for _, r in sso.head(150).iterrows():
        folium.CircleMarker(
            [r["lat"], r["lon"]],
            radius=5,
            color="#8a2be2",
            fill=True,
            popup=f"<b>TCEQ Sewer Site:</b> {r['label']}",
        ).add_to(m)

    # Active Citizen 311SA Tickets
    for t in st.session_state.tickets:
        color = "red" if t["score"] >= 70 else ("orange" if t["score"] >= 40 else "green")
        folium.Marker(
            [t["lat"], t["lon"]],
            icon=folium.Icon(color=color, icon="info-sign"),
            popup=f"<b>{t['case_id']}</b> ({t['score']}%)<br>{t['status']}<br>{t['summary']}",
        ).add_to(m)

    map_data = st_folium(m, height=460, width="100%")
    st.markdown(
        "🔴 **Low Water Crossing** &nbsp;&nbsp; 🔵 **Smart Streetlight** &nbsp;&nbsp; "
        "🟣 **TCEQ Sewer Site** &nbsp;&nbsp; 📍 **311SA Ticket**"
    )

# --- RIGHT COLUMN: CITIZEN REPORT & PROBABILITY SCORER ---
with col_form:
    st.subheader("Report Flooding or Check Area Risk")
    st.info("Click anywhere on the map to auto-fill coordinates, or enter them below.")

    clicked = map_data.get("last_clicked") if map_data else None
    c1, c2 = st.columns(2)
    lat = c1.number_input("Latitude", value=float(clicked["lat"] if clicked else 29.4275), format="%.5f")
    lon = c2.number_input("Longitude", value=float(clicked["lng"] if clicked else -98.5236), format="%.5f")
    location_desc = st.text_input("Intersection / Landmark", value="W Commerce St & Alazan Creek")

    # Recursive scan of server dataset (/tmp/datathon/UTSA-flood-images-complete/images) or local sample_images/
    sample_paths = sorted([
        p for p in IMAGE_ROOT.rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} and ".ipynb_checkpoints" not in str(p)
    ])
    sample_labels = {f"{p.parent.name}/{p.name}": str(p) for p in sample_paths[:150]}

    sample_choice = st.selectbox(
        f"Pick from dataset ({IMAGE_ROOT})",
        ["None"] + list(sample_labels.keys()),
    )
    uploaded_photo = st.file_uploader("Or upload a street/creek photo", type=["jpg", "jpeg", "png", "webp"])

    if st.button("Compute Disaster Probability & Submit 311SA", type="primary", use_container_width=True):
        img_path = None
        if uploaded_photo is not None:
            saved = UPLOAD_DIR / f"{uuid.uuid4().hex}_{uploaded_photo.name}"
            saved.write_bytes(uploaded_photo.getbuffer())
            img_path = str(saved)
        elif sample_choice != "None":
            img_path = sample_labels[sample_choice]

        with st.spinner("Running Qwen3-VL visual inspection & San Antonio spatial risk model..."):
            res = compute_disaster_score(lat, lon, img_path)

        if img_path:
            st.image(img_path, caption=f"Analyzed Image ({res['vlm']['engine']})", use_container_width=True)

        score = res["total_score"]
        if score >= 70:
            st.error(f"### Disaster Probability Score: {score}% — {res['risk_tier']}")
        elif score >= 40:
            st.warning(f"### Disaster Probability Score: {score}% — {res['risk_tier']}")
        else:
            st.success(f"### Disaster Probability Score: {score}% — {res['risk_tier']}")

        if res["sewage_warning"]:
            st.warning(
                "⚠️ **Sewage Overflow / Contamination Alert:** Location is near a historical TCEQ "
                "Sanitary Sewer Overflow site or murky runoff was detected. Avoid contact with water."
            )

        cols = st.columns(2)
        for idx, (k, v) in enumerate(res["breakdown"].items()):
            cols[idx % 2].metric(k, f"{v} pts")

        with st.expander("Detailed AI & Sensor Evidence", expanded=True):
            st.write(f"**Visual Classification:** `{res['vlm']['status']}` — {res['vlm']['summary']}")
            st.write(f"**Estimated Water Depth:** {res['vlm'].get('estimated_water_depth', 'N/A')}")
            st.write(f"**Nearest Low Water Crossing:** {res['nearest_lwc']}")
            st.write(f"**Nearest Smart Streetlight:** {res['nearest_sensor']}")
            st.write(f"**Nearest TCEQ Sewer Site:** {res['nearest_sso']}")

        case_id = f"311SA-2026-{uuid.uuid4().hex[:4].upper()}"
        st.session_state.tickets.insert(0, {
            "case_id": case_id,
            "time": datetime.now().strftime("%I:%M %p"),
            "location": location_desc,
            "lat": lat,
            "lon": lon,
            "score": score,
            "status": "AI Verified — Queued for Dispatch" if score >= 40 else "Logged — Monitoring",
            "sewage_alert": res["sewage_warning"],
            "summary": res["vlm"]["summary"],
        })
        st.toast(f"Created 311SA Ticket {case_id}")

# --- BOTTOM SECTION: CITIZEN 311SA STATUS TRACKER ---
st.divider()
st.subheader("Live 311SA Citizen Reports & Dispatch Status")
st.dataframe(
    pd.DataFrame(st.session_state.tickets)[
        ["case_id", "time", "location", "score", "sewage_alert", "status", "summary"]
    ],
    use_container_width=True,
    hide_index=True,
)