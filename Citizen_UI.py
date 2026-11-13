"""Citizen UI Frontend - 311SA Flood & Hazard Guard."""
import sys

# Auto-launch with streamlit runner if executed directly with python / in PyCharm
try:
    import streamlit.runtime
    if not streamlit.runtime.exists():
        from streamlit.web import cli as stcli
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
except Exception:
    pass

import uuid
from datetime import datetime
from pathlib import Path
import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from risk_engine import (
    SA_PRESET_AREAS,
    compute_disaster_score,
    get_image_root,
    load_sa_datasets,
    score_all_preset_areas,
)

# --- PAGE SETUP ---
st.set_page_config(
    page_title="311SA Flood & Hazard Guard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
IMAGE_ROOT = get_image_root()

# --- CACHED DATASETS ---
@st.cache_data(show_spinner=False)
def get_cached_datasets():
    return load_sa_datasets()

@st.cache_data(show_spinner=False)
def get_cached_leaderboard():
    return score_all_preset_areas()

lwc_df, sso_df, sl_df = get_cached_datasets()

# --- INITIALIZE SESSION STATE ---
if "tickets" not in st.session_state:
    st.session_state.tickets = [
        {
            "case_id": "311SA-2026-8841",
            "time": "11:42 AM",
            "location": "Alazan Creek @ W Commerce St",
            "lat": 29.4275,
            "lon": -98.5236,
            "score": 82.5,
            "tier": "HIGH RISK — AVOID AREA",
            "status": "AI Verified — Public Works Dispatched",
            "sewage_alert": True,
            "summary": "Standing water covering curb lane near storm inlet.",
        },
        {
            "case_id": "311SA-2026-4102",
            "time": "12:15 PM",
            "location": "Brooks Innovation Zone",
            "lat": 29.3482,
            "lon": -98.4351,
            "score": 48.0,
            "tier": "MODERATE RISK",
            "status": "Logged — Monitoring Sensor",
            "sewage_alert": False,
            "summary": "Smart streetlight telemetry reporting 3.4 in. water accumulation.",
        },
    ]

if "selected_lat" not in st.session_state:
    st.session_state.selected_lat = 29.4275
if "selected_lon" not in st.session_state:
    st.session_state.selected_lon = -98.5236
if "selected_area_name" not in st.session_state:
    st.session_state.selected_area_name = "Alazan Creek @ W Commerce St"
if "selected_zone_desc" not in st.session_state:
    st.session_state.selected_zone_desc = "West Side / Historic Flood Basin"
if "current_result" not in st.session_state:
    st.session_state.current_result = compute_disaster_score(29.4275, -98.5236, None)
if "last_map_click" not in st.session_state:
    st.session_state.last_map_click = None

# --- HEADER & BANNER ---
st.title("🌊 311SA Flood & Hazard Guard — Citizen Portal")
st.markdown(
    "**San Antonio Flash Flood Probability & Infrastructure Intelligence System** &nbsp;|&nbsp; "
    "Powered by *Qwen3-VL on NVIDIA GB10* + *CoSAGIS 156 Low Water Crossings* + *SmartSA IoT Sensors*"
)

# Citywide Key Indicators
m1, m2, m3, m4 = st.columns(4)
m1.metric("Monitored Low Water Crossings", f"{len(lwc_df)} Sites", help="Bexar County FEMA Floodplain Crossings")
m2.metric("SmartSA IoT Flood Sensors", f"{len(sl_df)} Zones", help="Downtown, Brooks, Medical Center")
m3.metric("TCEQ Sewer Overflow Sites", f"{len(sso_df)} Lift Stations", help="San Pedro & Apache Creeks, Brooks")
m4.metric("Active 311SA Dispatches", f"{len(st.session_state.tickets)} Tickets", help="Live community hazard reports")

st.divider()

# --- TOP NAVIGATION TABS ---
tab_inspect, tab_leaderboard, tab_tickets = st.tabs([
    "📍 Area Risk Inspector & Probability Scorer",
    "📊 Citywide Risk Leaderboard (Compare Areas)",
    "📋 Live 311SA Community Feed & Dispatch Queue",
])

# ==============================================================================
# TAB 1: AREA RISK INSPECTOR & PROBABILITY SCORER
# ==============================================================================
with tab_inspect:
    col_map, col_controls = st.columns([1.25, 1.0], gap="large")

    # --- RIGHT PANEL: AREA SELECTION & PROBABILITY SCORING ---
    with col_controls:
        st.subheader("🎯 Select Area & Compute Probability")

        selection_mode = st.radio(
            "Choose Area Selection Method:",
            [
                "⭐ San Antonio Key Flood Zones (10 Preset Areas)",
                "⚠️ Search 156 Low Water Crossings (CoSAGIS)",
                "🗺️ Custom Coordinates or Click on Map",
            ],
            index=0,
            horizontal=False,
        )

        # Mode A: Key Presets
        if selection_mode.startswith("⭐"):
            preset_labels = [f"{a['name']} ({a['zone']})" for a in SA_PRESET_AREAS]
            default_idx = 0
            for idx, a in enumerate(SA_PRESET_AREAS):
                if a["name"] in st.session_state.selected_area_name:
                    default_idx = idx
                    break

            chosen_preset_label = st.selectbox(
                "Select a San Antonio Zone / Watershed:",
                preset_labels,
                index=default_idx,
            )
            chosen_preset = SA_PRESET_AREAS[preset_labels.index(chosen_preset_label)]
            if (
                st.session_state.selected_lat != chosen_preset["lat"]
                or st.session_state.selected_lon != chosen_preset["lon"]
            ):
                st.session_state.selected_lat = chosen_preset["lat"]
                st.session_state.selected_lon = chosen_preset["lon"]
                st.session_state.selected_area_name = chosen_preset["name"]
                st.session_state.selected_zone_desc = chosen_preset["zone"]
                st.session_state.current_result = compute_disaster_score(
                    chosen_preset["lat"], chosen_preset["lon"], None
                )

            st.caption(f"ℹ️ **Area Profile:** {chosen_preset['description']}")

        # Mode B: All 156 Low Water Crossings
        elif selection_mode.startswith("⚠️"):
            crossing_names = sorted(lwc_df["label"].dropna().unique().tolist())
            chosen_crossing_name = st.selectbox(
                "Select from 156 Official San Antonio Low Water Crossings:",
                crossing_names,
            )
            crossing_row = lwc_df[lwc_df["label"] == chosen_crossing_name].iloc[0]
            if (
                st.session_state.selected_lat != crossing_row["lat"]
                or st.session_state.selected_lon != crossing_row["lon"]
            ):
                st.session_state.selected_lat = float(crossing_row["lat"])
                st.session_state.selected_lon = float(crossing_row["lon"])
                st.session_state.selected_area_name = f"LWC: {crossing_row['label']}"
                st.session_state.selected_zone_desc = "Bexar County CoSAGIS Crossing"
                st.session_state.current_result = compute_disaster_score(
                    st.session_state.selected_lat, st.session_state.selected_lon, None
                )

        # Mode C: Custom Coordinates / Map Click
        else:
            c_lat, c_lon = st.columns(2)
            custom_lat = c_lat.number_input(
                "Latitude",
                value=float(st.session_state.selected_lat),
                format="%.5f",
                step=0.001,
            )
            custom_lon = c_lon.number_input(
                "Longitude",
                value=float(st.session_state.selected_lon),
                format="%.5f",
                step=0.001,
            )
            custom_name = st.text_input(
                "Intersection / Landmark Name",
                value=st.session_state.selected_area_name,
            )
            if (
                custom_lat != st.session_state.selected_lat
                or custom_lon != st.session_state.selected_lon
                or custom_name != st.session_state.selected_area_name
            ):
                st.session_state.selected_lat = custom_lat
                st.session_state.selected_lon = custom_lon
                st.session_state.selected_area_name = custom_name
                st.session_state.selected_zone_desc = "Custom Coordinates"

        # Optional Photo / Vision AI
        with st.expander("📸 Optional: Attach Street / Creek Photo (Qwen3-VL Vision AI)", expanded=False):
            use_photo = st.checkbox("Include Photo in Risk Assessment", value=False)
            photo_path = None
            if use_photo:
                photo_src = st.radio("Photo Source:", ["Pick from Dataset", "Upload Photo"], horizontal=True)
                if photo_src == "Pick from Dataset":
                    sample_paths = sorted([
                        p for p in IMAGE_ROOT.rglob("*")
                        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"} and ".ipynb_checkpoints" not in str(p)
                    ])
                    if sample_paths:
                        sample_dict = {f"{p.parent.name}/{p.name}": str(p) for p in sample_paths[:50]}
                        picked = st.selectbox("Select Sample Image:", list(sample_dict.keys()))
                        photo_path = sample_dict[picked]
                        st.image(photo_path, caption=f"Selected: {picked}", use_container_width=True)
                    else:
                        st.info("No sample images found in local directory. Upload a photo below.")
                else:
                    uploaded = st.file_uploader(
                        "Upload photo of street, creek, or drainage inlet",
                        type=["jpg", "jpeg", "png", "webp"],
                    )
                    if uploaded is not None:
                        saved = UPLOAD_DIR / f"{uuid.uuid4().hex}_{uploaded.name}"
                        saved.write_bytes(uploaded.getbuffer())
                        photo_path = str(saved)
                        st.image(photo_path, caption="Uploaded Street Photo", use_container_width=True)

        # Trigger Calculation Button
        if st.button("⚡ Calculate Area Probability Score", type="primary", use_container_width=True):
            with st.spinner("Analyzing geospatial risk & sensor telemetry..."):
                st.session_state.current_result = compute_disaster_score(
                    st.session_state.selected_lat,
                    st.session_state.selected_lon,
                    photo_path if use_photo else None,
                )

        # --- DISPLAY CURRENT AREA PROBABILITY SCORECARD ---
        res = st.session_state.current_result
        score = res["total_score"]
        tier = res["risk_tier"]

        st.markdown("---")
        st.markdown(f"### Area: **{st.session_state.selected_area_name}**")
        st.caption(f"Coordinates: `{st.session_state.selected_lat:.5f}, {st.session_state.selected_lon:.5f}` &nbsp;|&nbsp; {st.session_state.selected_zone_desc}")

        # Probability Gauge Card
        if score >= 70:
            st.error(f"### 🚨 Disaster Probability Score: **{score}%** — {tier}")
        elif score >= 40:
            st.warning(f"### ⚠️ Disaster Probability Score: **{score}%** — {tier}")
        else:
            st.success(f"### ✅ Disaster Probability Score: **{score}%** — {tier}")

        st.progress(score / 100.0)

        # Contextual Alerts
        if res.get("sewage_warning"):
            st.error(
                "☣️ **Sewage Contamination Alert:** Within 500m of a historical TCEQ Sanitary Sewer Overflow "
                "site or murky runoff detected. High biohazard risk; avoid contact with standing water!"
            )

        # 4-Factor Weighted Breakdown
        st.markdown("#### 📊 Risk Factor Breakdown")
        b1, b2 = st.columns(2)
        b3, b4 = st.columns(2)
        breakdown = res["breakdown"]
        b1.metric("Visual AI Evidence", f"{breakdown.get('Visual AI Evidence (40 max)', 0)} / 40 pts")
        b2.metric("Low Water Crossing", f"{breakdown.get('Low Water Crossing Prior (25 max)', 0)} / 25 pts")
        b3.metric("Streetlight Sensor", f"{breakdown.get('Streetlight Sensor Telemetry (20 max)', 0)} / 20 pts")
        b4.metric("Sewage Overflow", f"{breakdown.get('Sewage Overflow Prior (15 max)', 0)} / 15 pts")

        # Detailed Sensor Evidence
        with st.expander("🔍 Detailed Sensor & Spatial Telemetry", expanded=True):
            st.markdown(f"- **Nearest Low Water Crossing:** `{res['nearest_lwc']}`")
            st.markdown(f"- **Nearest Smart Streetlight Sensor:** `{res['nearest_sensor']}`")
            st.markdown(f"- **Nearest TCEQ Sewer Overflow Site:** `{res['nearest_sso']}`")
            st.markdown(f"- **AI Vision Assessment (`{res['vlm']['engine']}`):** {res['vlm']['summary']}")
            if res["vlm"].get("estimated_water_depth") != "No photo uploaded":
                st.markdown(f"- **Estimated Water Depth:** `{res['vlm'].get('estimated_water_depth', 'N/A')}`")

        # 311 Report Button
        st.markdown("---")
        if st.button("📢 Submit Official 311SA Citizen Report", use_container_width=True):
            new_case = f"311SA-2026-{uuid.uuid4().hex[:4].upper()}"
            st.session_state.tickets.insert(0, {
                "case_id": new_case,
                "time": datetime.now().strftime("%I:%M %p"),
                "location": st.session_state.selected_area_name,
                "lat": st.session_state.selected_lat,
                "lon": st.session_state.selected_lon,
                "score": score,
                "tier": tier,
                "status": "AI Verified — Queued for Dispatch" if score >= 40 else "Logged — Monitoring",
                "sewage_alert": res["sewage_warning"],
                "summary": res["vlm"]["summary"],
            })
            st.toast(f"✅ Ticket {new_case} logged for {st.session_state.selected_area_name}!")
            st.success(f"Report logged as **{new_case}**. City crews alerted.")

    # --- LEFT PANEL: INTERACTIVE SAN ANTONIO RISK MAP ---
    with col_map:
        st.subheader("🗺️ Live San Antonio Risk & Infrastructure Map")

        # Map Layer Toggles
        layer_col1, layer_col2, layer_col3 = st.columns(3)
        show_lwc = layer_col1.checkbox("Show 156 LWC Crossings (🔴)", value=True)
        show_sl = layer_col2.checkbox("Show Flood Sensors (🔵)", value=True)
        show_sso = layer_col3.checkbox("Show Sewer Sites (🟣)", value=True)

        # Build Folium Map
        m = folium.Map(
            location=[st.session_state.selected_lat, st.session_state.selected_lon],
            zoom_start=12,
            tiles="OpenStreetMap",
        )

        # 1. Plot Low Water Crossings (156 CoSAGIS)
        if show_lwc:
            for _, r in lwc_df.iterrows():
                folium.CircleMarker(
                    location=[r["lat"], r["lon"]],
                    radius=5,
                    color="#d9534f",
                    fill=True,
                    fill_opacity=0.7,
                    popup=f"<b>CoSAGIS Low Water Crossing:</b><br>{r['label']}",
                ).add_to(m)

        # 2. Plot Smart Streetlight Flood Sensors
        if show_sl:
            for _, r in sl_df.iterrows():
                folium.CircleMarker(
                    location=[r["lat"], r["lon"]],
                    radius=7,
                    color="#0275d8",
                    fill=True,
                    fill_opacity=0.9,
                    popup=f"<b>Smart Streetlight Flood Sensor:</b><br>{r['label']}<br>Water Level: <b>{r['flood_level_in']} in.</b>",
                ).add_to(m)

        # 3. Plot TCEQ Sewer Overflow Sites
        if show_sso:
            for _, r in sso_df.iterrows():
                folium.CircleMarker(
                    location=[r["lat"], r["lon"]],
                    radius=6,
                    color="#8a2be2",
                    fill=True,
                    fill_opacity=0.8,
                    popup=f"<b>TCEQ Sewer Overflow Site:</b><br>{r['label']}",
                ).add_to(m)

        # 4. Highlight the CURRENTLY INSPECTED AREA with a prominent pin
        pin_color = "red" if score >= 70 else ("orange" if score >= 40 else "green")
        folium.Marker(
            location=[st.session_state.selected_lat, st.session_state.selected_lon],
            icon=folium.Icon(color=pin_color, icon="flag"),
            popup=f"<b>CURRENT TARGET: {st.session_state.selected_area_name}</b><br>Disaster Probability: <b>{score}%</b><br>Risk Tier: <b>{tier}</b>",
        ).add_to(m)

        # Render Map
        map_out = st_folium(m, height=520, use_container_width=True)

        st.caption(
            "🔴 **156 CoSAGIS Low Water Crossings** &nbsp;|&nbsp; "
            "🔵 **SmartSA Streetlight Flood Sensors** &nbsp;|&nbsp; "
            "🟣 **TCEQ Sewer Sites** &nbsp;|&nbsp; "
            "🚩 **Selected Area Pin**"
        )

        # Handle Map Click to Update Area
        if map_out and map_out.get("last_clicked"):
            clicked = map_out["last_clicked"]
            c_lat, c_lon = round(clicked["lat"], 5), round(clicked["lng"], 5)
            if (c_lat, c_lon) != st.session_state.last_map_click:
                st.session_state.last_map_click = (c_lat, c_lon)
                st.session_state.selected_lat = c_lat
                st.session_state.selected_lon = c_lon
                st.session_state.selected_area_name = f"Map Pin ({c_lat:.4f}, {c_lon:.4f})"
                st.session_state.selected_zone_desc = "Selected via Interactive Map Click"
                st.session_state.current_result = compute_disaster_score(c_lat, c_lon, None)
                st.rerun()

# ==============================================================================
# TAB 2: CITYWIDE RISK LEADERBOARD & COMPARISON
# ==============================================================================
with tab_leaderboard:
    st.subheader("📊 San Antonio Key Sectors & Watershed Risk Comparison")
    st.markdown(
        "Evaluate and compare flash flood probabilities across major San Antonio corridors at a glance. "
        "Click any area below to immediately inspect it on the map and view detailed sensor telemetry."
    )

    leaderboard_df = get_cached_leaderboard()

    # Visual Bar Chart of Probability by Area
    chart_data = leaderboard_df.set_index("Area Name")[["Disaster Probability (%)"]]
    st.bar_chart(chart_data, color="#d9534f")

    # Interactive Table
    st.dataframe(
        leaderboard_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Disaster Probability (%)": st.column_config.ProgressColumn(
                "Disaster Probability (%)",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
        },
    )

    # Quick Switcher: Select an area to inspect in Tab 1
    st.markdown("#### 🔍 One-Click Area Inspector")
    col_sel, col_btn = st.columns([2, 1])
    target_area = col_sel.selectbox(
        "Select an area from the leaderboard to load into the Inspector Map:",
        leaderboard_df["Area Name"].tolist(),
    )
    if col_btn.button("Load Area into Inspector", use_container_width=True):
        row_target = leaderboard_df[leaderboard_df["Area Name"] == target_area].iloc[0]
        st.session_state.selected_lat = float(row_target["Latitude"])
        st.session_state.selected_lon = float(row_target["Longitude"])
        st.session_state.selected_area_name = row_target["Area Name"]
        st.session_state.selected_zone_desc = row_target["Zone / Region"]
        st.session_state.current_result = compute_disaster_score(
            st.session_state.selected_lat, st.session_state.selected_lon, None
        )
        st.toast(f"Loaded {target_area} into Inspector Map!")
        st.rerun()

# ==============================================================================
# TAB 3: LIVE 311SA COMMUNITY FEED & DISPATCH QUEUE
# ==============================================================================
with tab_tickets:
    st.subheader("📋 Live 311SA Citizen Reports & Public Works Queue")
    st.markdown("Real-time feed of AI-verified hazard reports submitted by citizens across Bexar County.")

    if st.session_state.tickets:
        tickets_df = pd.DataFrame(st.session_state.tickets)
        display_cols = ["case_id", "time", "location", "score", "tier", "sewage_alert", "status", "summary"]
        existing_cols = [c for c in display_cols if c in tickets_df.columns]

        st.dataframe(
            tickets_df[existing_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "score": st.column_config.NumberColumn("Risk Score (%)", format="%.1f%%"),
                "sewage_alert": st.column_config.CheckboxColumn("Sewage Alert"),
            },
        )
    else:
        st.info("No active 311 tickets logged yet.")
