import uuid
from datetime import datetime
from pathlib import Path
import folium
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from citizen_risk_engine import compute_disaster_score, load_sa_layers

st.set_page_config(page_title="311SA Flood & Hazard Guard", page_icon="🌊", layout="wide")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Initialize session state for live citizen 311 tickets
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
st.caption("Powered by Qwen3-VL on NVIDIA GB10 + San Antonio Open Data & Smart Streetlights")

col_map, col_report = st.columns([1.3, 1.0], gap="large")

# --- LEFT COLUMN: INTERACTIVE SAN ANTONIO RISK MAP ---
with col_map:
    st.subheader("Live San Antonio Flood & Infrastructure Map")
    lwc, sso, streetlights = load_sa_layers()

    m = folium.Map(location=[29.4241, -98.4936], zoom_start=11, tiles="CartoDB positron")

    # Plot CoSAGIS Low Water Crossings
    for _, row in lwc.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=6,
            color="#d9534f",
            fill=True,
            popup=f"<b>Low Water Crossing:</b> {row['name']}",
        ).add_to(m)

    # Plot Smart Streetlight Sensors
    for _, row in streetlights.iterrows():
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=7,
            color="#0275d8",
            fill=True,
            popup=f"<b>Smart Streetlight:</b> {row['zone']}<br>Water Level: {row['flood_level_in']} in.",
        ).add_to(m)

    # Plot Citizen 311 Tickets
    for t in st.session_state.tickets:
        pin_color = "red" if t["score"] >= 70 else ("orange" if t["score"] >= 40 else "green")
        folium.Marker(
            location=[t["lat"], t["lon"]],
            icon=folium.Icon(color=pin_color, icon="info-sign"),
            popup=f"<b>{t['case_id']}</b> ({t['score']}%)<br>{t['status']}<br>{t['summary']}",
        ).add_to(m)

    map_data = st_folium(m, height=460, width="100%")
    st.markdown("🔴 **Low Water Crossing** &nbsp;&nbsp; 🔵 **Smart Streetlight Sensor** &nbsp;&nbsp; 📍 **Active 311SA Report**")

# --- RIGHT COLUMN: CITIZEN REPORT & PROBABILITY SCORER ---
with col_report:
    st.subheader("Report Flooding or Check Area Risk")
    st.info("Click anywhere on the map to auto-fill coordinates, or enter them below.")

    clicked = map_data.get("last_clicked") if map_data else None
    default_lat = clicked["lat"] if clicked else 29.4275
    default_lon = clicked["lng"] if clicked else -98.5236

    c1, c2 = st.columns(2)
    lat = c1.number_input("Latitude", value=float(default_lat), format="%.5f")
    lon = c2.number_input("Longitude", value=float(default_lon), format="%.5f")
    location_desc = st.text_input("Intersection or Landmark", value="W Commerce St & Alazan Creek")
    uploaded_photo = st.file_uploader("Upload Street / Creek Photo (Optional)", type=["jpg", "jpeg", "png", "webp"])

    if st.button("Analyze Risk & Submit 311SA Report", type="primary", use_container_width=True):
        saved_path = None
        if uploaded_photo is not None:
            saved_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{uploaded_photo.name}"
            saved_path.write_bytes(uploaded_photo.getbuffer())

        with st.spinner("Running Qwen3-VL visual inspection & San Antonio spatial risk model..."):
            result = compute_disaster_score(lat, lon, str(saved_path) if saved_path else None)

        # Display Risk Score Card
        score = result["total_score"]
        if score >= 70:
            st.error(f"### Disaster Probability Score: {score}% — {result['risk_tier']}")
        elif score >= 40:
            st.warning(f"### Disaster Probability Score: {score}% — {result['risk_tier']}")
        else:
            st.success(f"### Disaster Probability Score: {score}% — {result['risk_tier']}")

        if result["sewage_warning"]:
            st.warning("⚠️ **Sewage Overflow / Contamination Hazard:** Location is near a historical TCEQ Sanitary Sewer Overflow site or shows murky runoff. Avoid skin contact with floodwater.")

        # Breakdown metrics
        b_cols = st.columns(2)
        for idx, (label, val) in enumerate(result["breakdown"].items()):
            b_cols[idx % 2].metric(label, f"{val} pts")

        with st.expander("View AI Visual & Sensor Evidence", expanded=True):
            st.write(f"**AI Visual Assessment (`{result['vlm']['status']}`):** {result['vlm']['summary']}")
            st.write(f"**Estimated Depth:** {result['vlm']['estimated_water_depth']}")
            st.write(f"**Nearest Low Water Crossing:** {result['nearest_lwc']}")
            st.write(f"**Nearest Smart Streetlight:** {result['nearest_sensor']}")

        # Create 311SA Ticket
        case_id = f"311SA-2026-{uuid.uuid4().hex[:4].upper()}"
        st.session_state.tickets.insert(0, {
            "case_id": case_id,
            "time": datetime.now().strftime("%I:%M %p"),
            "location": location_desc,
            "lat": lat,
            "lon": lon,
            "score": score,
            "status": "AI Verified — Queued for Dispatch" if score >= 40 else "Logged — Monitoring",
            "sewage_alert": result["sewage_warning"],
            "summary": result["vlm"]["summary"],
        })
        st.toast(f"Created 311SA Ticket {case_id}")

# --- BOTTOM SECTION: CITIZEN 311SA STATUS TRACKER ---
st.divider()
st.subheader("Live 311SA Community Feed & Ticket Status")
st.dataframe(
    pd.DataFrame(st.session_state.tickets)[["case_id", "time", "location", "score", "sewage_alert", "status", "summary"]],
    use_container_width=True,
    hide_index=True,
)