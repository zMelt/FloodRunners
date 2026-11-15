# 🌊 FloodRunners: 311SA Flood & Hazard Guard
### Real-Time San Antonio Flash Flood Probability & Infrastructure Intelligence System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io)
[![NVIDIA GB10](https://img.shields.io/badge/Powered%20By-NVIDIA%20DGX%20Spark%20GB10-76B900.svg)](https://www.nvidia.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Developed for the **UTSA / USAA Social Impact AI Datathon**, **FloodRunners (311SA Flood & Hazard Guard)** is an end-to-end multi-modal flood risk intelligence platform designed for the City of San Antonio and Bexar County. It fuses real-time IoT municipal sensor telemetry, 156 official CoSAGIS low-water crossings, TCEQ sanitary sewer overflow hazard tracking, and zero-shot vision AI on an **NVIDIA DGX Spark / Dell Pro Max GB10** workstation to calculate explainable disaster probability scores ($0\% - 100\%$) across San Antonio watersheds.

---

## 📌 Table of Contents
- [Executive Overview & Social Impact](#-executive-overview--social-impact)
- [System Architecture & Tech Stack](#-system-architecture--tech-stack)
- [Mathematical Formulation & Risk Scoring](#-mathematical-formulation--risk-scoring)
- [Key Features](#-key-features)
- [Municipal & State Datasets](#-municipal--state-datasets)
- [Repository Structure](#-repository-structure)
- [Quickstart & Installation](#-quickstart--installation)
- [Configuration & VLM Setup](#-configuration--vlm-setup)
- [Hackathon Pitch & Use Cases](#-hackathon-pitch--use-cases)

---

## 🌍 Executive Overview & Social Impact

San Antonio and Central Texas sit in **"Flash Flood Alley"**, recognized as one of the most flash-flood-prone regions in North America due to steep Hill Country limestone topography, thin soils with poor absorption, and severe precipitation systems. In San Antonio:
1. **Low-Water Crossings (LWCs)** quickly become deadly traps for drivers during rapid water surges.
2. **Sanitary Sewer Overflows (SSOs)** occur when floodwaters infiltrate wastewater systems, contaminating street runoffs with hazardous bio-pathogens.
3. **Disjointed Telemetry**: Citizens, emergency services (SAFD/Bexar County OEM), and 311 municipal dispatchers previously had to cross-reference static maps, weather radars, and unverified phone calls.

**FloodRunners** unifies these disparate data streams into a single interactive intelligence platform:
- **For Citizens**: Point-and-click flood hazard assessment, pre-trip navigation checks across 156 low-water crossings, and immediate 311 photo reporting.
- **For First Responders & 311 Dispatch**: Automated risk scoring, real-time Folium heatmaps, bio-hazard sewer contamination warnings, and prioritized ticket queues.

---

## 🛠️ System Architecture & Tech Stack

```
                     ┌──────────────────────────────────────────────┐
                     │          CROWDSOURCED CITIZEN DATA           │
                     │  • Geotagged Photos (Street/Creek/Inlet)     │
                     │  • Citizen 311 Emergency Reports             │
                     └──────────────────────┬───────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────┐   ┌───────────────────────────────┐
│     MUNICIPAL TELEMETRY & GIS   │   │     NVIDIA GB10 WORKSTATION   │
│  • CoSAGIS 156 LW Crossings     │   │  • Qwen3-VL Vision-Language   │
│  • SmartSA IoT Streetlight Data │   │    Model (8B-Instruct)        │
│  • TCEQ Sanitary Sewer Overflows│   │  • Zero-Shot Flood Analysis   │
└────────────────┬────────────────┘   └───────────────┬───────────────┘
                 │                                    │
                 └──────────────────┬─────────────────┘
                                    ▼
       ┌─────────────────────────────────────────────────────────────┐
       │                 GEOSPATIAL RISK ENGINE                      │
       │  • Lambert Conformal Conic Projection (EPSG:2278 -> WGS84)  │
       │  • Haversine Geodesic Distance Matrix                       │
       │  • Exponential Decay Distance Modeling                      │
       │  • Weighted Multi-Factor Disaster Probability Formulation   │
       └────────────────────────────┬────────────────────────────────┘
                                    │
                                    ▼
       ┌─────────────────────────────────────────────────────────────┐
       │                 INTERACTIVE WEB DASHBOARD                   │
       │  • Streamlit Multi-Tab Operations Console                   │
       │  • Folium HeatMap Layer (172+ San Antonio Hazard Points)    │
       │  • Citywide Cross-Watershed Leaderboard                     │
       │  • 311 Citizen Hazard Dispatch Queue                        │
       └─────────────────────────────────────────────────────────────┘
```

- **Frontend & Visualization**: [Streamlit](https://streamlit.io), [Folium](https://python-visualization.github.io/folium/) with Leaflet.js `HeatMap` plugin, [Streamlit-Folium](https://github.com/randyzwitch/streamlit-folium).
- **Vision-Language Model (VLM)**: `Qwen/Qwen3-VL-8B-Instruct` served on Dell Pro Max GB10 via vLLM / GPUStack OpenAI-compatible API (`http://10.246.100.230/v1`), with local Mac/CPU fallback via CLIP ViT-B/32 (`sentence-transformers`).
- **Geospatial & Mathematical Computations**: `NumPy`, `SciPy`, `Pandas`, custom EPSG:2278 ellipsoidal projection algorithms, and spherical Haversine formulas.
- **Data Engineering**: Integration of City of San Antonio Open Data (CoSAGIS, SmartSA) and Texas Commission on Environmental Quality (TCEQ).

---

## 📐 Mathematical Formulation & Risk Scoring

### 1. Coordinate Transformation (EPSG:2278 $\rightarrow$ WGS84)
The City of San Antonio's official GIS datasets (`LowWaterCrossings_data.csv`) publish spatial coordinates in **NAD83 / Texas South Central (ftUS) (EPSG:2278)**. The risk engine converts these into global **WGS84 Latitude/Longitude** via the Lambert Conformal Conic 2SP projection formulas:

$$m_i = \frac{\cos \phi_i}{\sqrt{1 - e^2 \sin^2 \phi_i}}, \quad t_i = \frac{\tan(\pi/4 - \phi_i/2)}{\left(\frac{1 - e \sin \phi_i}{1 + e \sin \phi_i}\right)^{e/2}}$$

$$n = \frac{\ln(m_1 / m_2)}{\ln(t_1 / t_2)}, \quad F = \frac{m_1}{n \, t_1^n}, \quad \rho = \sqrt{(x - x_0)^2 + (\rho_0 - (y - y_0))^2}$$

$$\lambda = \frac{\theta}{n} + \lambda_0, \quad \phi = \frac{\pi}{2} - 2 \arctan(t) \quad \text{(solved iteratively)}$$

### 2. Geodesic Proximity Decay
For any inspected point $(\phi_1, \lambda_1)$ and hazard point $(\phi_2, \lambda_2)$, geodesic distance $d$ is computed using the Great-Circle Haversine formula:

$$d = 2 R \arcsin \left( \sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos \phi_1 \cos \phi_2 \sin^2\left(\frac{\Delta \lambda}{2}\right)} \right)$$

Hazard impacts decay exponentially with distance:
- **Low Water Crossing Score ($S_{lwc}$)**:
  $$S_{lwc}(d) = 100 \cdot \exp\left(-\frac{d}{800}\right)$$
- **Smart Streetlight Water Level Score ($S_{sensor}$)**:
  $$S_{sensor}(d, \Delta h) = \min(100, 20 + \Delta h \cdot 80) \cdot \exp\left(-\frac{d}{1200}\right)$$
- **Sanitary Sewer Overflow Hazard ($S_{sso}$)**:
  $$S_{sso}(d) = 100 \cdot \exp\left(-\frac{d}{1500}\right)$$

### 3. Total Disaster Probability Score ($S_{total}$)
When a citizen attaches a photo, the Qwen3-VL VLM assesses the visual evidence (water level, current speed, vehicle impassability) to generate a vision confidence score $S_{vlm} \in [0, 100]$. The components are fused into a normalized total disaster score:

$$S_{total} = \begin{cases} 
0.40 \, S_{lwc} + 0.35 \, S_{sensor} + 0.25 \, S_{sso} & \text{without photo} \\
0.30 \, S_{lwc} + 0.25 \, S_{sensor} + 0.20 \, S_{sso} + 0.25 \, S_{vlm} & \text{with photo}
\end{cases}$$

### 4. Risk Tiers
| Score Range | Risk Tier | Actionable Protocol |
| :---: | :---: | :--- |
| **70% – 100%** | 🚨 **High / Extreme Hazard** | Road closed, immediate flash flood warning, deploy barricades, avoid area. |
| **40% – 69%** | ⚠️ **Elevated Hazard** | High caution, rising creek levels, water over roadway possible. |
| **0% – 39%** | 🟢 **Low / Nominal** | Normal drainage flow, low-water crossing dry or passable. |

---

## ✨ Key Features

1. **Interactive Folium Heatmap**:
   - Citywide visual heatmap showing disaster score density across San Antonio.
   - 156 CoSAGIS Low Water Crossing markers with street name popups.
   - SmartSA streetlight flood sensor markers displaying real-time depth telemetry.
   - TCEQ sanitary sewer overflow hazard markers warning of bio-contamination.
   - Live marker pin for the currently inspected location.

2. **Multi-Mode Location Selection**:
   - **Preset Watershed Zones**: 10 high-risk San Antonio corridors (Olmos Basin & Dam, Leon Creek @ Hausman/UTSA, Salado Creek @ Eisenhauer, San Pedro Creek River Walk, Medical Center, Brooks City-Base, Apache Creek @ Elmendorf Lake, Martinez Creek, Mission Espada).
   - **Searchable Low Water Crossings**: All 156 official CoSAGIS crossings with auto-lookup and coordinate projection.
   - **Point-and-Click Map Inspection**: Click anywhere on the map or input custom coordinates.

3. **Multimodal Qwen3-VL Vision AI**:
   - Direct image upload (`.jpg`, `.jpeg`, `.png`, `.webp`).
   - Analyzes street flooding, submerged curbs, stormwater inlet debris, and water velocity.
   - Provides structured JSON analysis and elevates disaster probability accordingly.

4. **Citywide Risk Leaderboard**:
   - Instant multi-zone comparative matrix ranking all major San Antonio corridors by risk score.
   - Visual badges for sewage contamination alerts, nearest crossings, and active sensors.

5. **311 Citizen Hazard Dispatch Queue**:
   - Citizens can submit live emergency flood reports with photos and descriptions.
   - Real-time municipal dispatch table with ticket IDs, status indicators, and timestamps.

---

## 📊 Municipal & State Datasets

| Dataset | Source | Details | File in Repo |
| :--- | :--- | :--- | :--- |
| **Low Water Crossings** | City of San Antonio (CoSAGIS) | 156 official low-water crossings with Texas State Plane coordinates (EPSG:2278). | [`LowWaterCrossings_data.csv`](LowWaterCrossings_data.csv) |
| **Downtown Flood Telemetry** | SmartSA Innovation Zone | Smart streetlight ultrasonic flood sensors in the Downtown corridor. | [`DT_flood_data.csv`](DT_flood_data.csv) |
| **Medical Center Sensors** | SmartSA Innovation Zone | Smart streetlight flood sensors around the South Texas Medical Center. | [`Medical_flood_data.csv`](Medical_flood_data.csv) |
| **Brooks Sensor Telemetry** | SmartSA Innovation Zone | Smart streetlight environmental and water depth sensors in the Brooks area. | [`Brooks_flood_data.csv.zip`](Brooks_flood_data.csv.zip) |
| **Citywide Streetlight Sensors** | SmartSA Open Data | Historical and multi-zone pilot streetlight sensor telemetry across San Antonio. | [`SAStreetlight_Sensor_Data.csv.zip`](SAStreetlight_Sensor_Data.csv.zip) |
| **Sanitary Sewer Overflows (SSO)** | TCEQ (Texas Commission on Environmental Quality) | Historical and active unauthorized wastewater discharge records in Bexar County. | [`Texas_Commission_on_Environmental_Quality_-_Sanitary_Sewer_Overflows_(SSO)_20261113.csv.zip`](Texas_Commission_on_Environmental_Quality_-_Sanitary_Sewer_Overflows_(SSO)_20261113.csv.zip) |
| **Sensor Data Dictionary** | City of San Antonio | Definitions and metadata for SmartSA streetlight environmental telemetry. | [`datadictionary-flood.xlsx`](datadictionary-flood.xlsx) |

---

## 📂 Repository Structure

```
FloodRunners/
├── .streamlit/
│   └── config.toml                  # Streamlit theme & port configuration
├── app.py                           # Primary Streamlit Dashboard (Heatmap, Leaderboard, 311)
├── Citizen_UI.py                    # Citizen Portal Interface
├── risk_engine.py                   # Core Geospatial, Sensor, and VLM Risk Engine
├── citizen_risk_engine.py           # Compatibility wrapper
├── vlm_config.py                    # NVIDIA GB10 / Qwen3-VL connection settings
├── FLOODALERT.ipynb                 # Datathon experimentation & validation notebook
├── requirements.txt                 # Clean, pinned Python dependencies
├── UTSA_more_data.md                # Municipal dataset documentation & API references
├── datadictionary-flood.xlsx        # City of San Antonio sensor schema dictionary
├── LowWaterCrossings_data.csv       # CoSAGIS 156 Low Water Crossings (EPSG:2278)
├── DT_flood_data.csv                # Downtown Innovation Zone flood sensors
├── Medical_flood_data.csv           # Medical Center flood sensors
├── Brooks_flood_data.csv.zip        # Brooks Innovation Zone sensor telemetry
├── SAStreetlight_Sensor_Data.csv.zip# Full SmartSA pilot sensor dataset
├── Texas_..._SSO_20261113.csv.zip   # TCEQ Sanitary Sewer Overflow incident data
├── LICENSE                          # MIT License
└── README.md                        # Project documentation
```

---

## 🚀 Quickstart & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/zMelt/FloodRunners.git
cd FloodRunners
```

### 2. Create and Activate a Python Environment
```bash
# Using Python 3.11 or 3.12
python3 -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Streamlit Application
You can launch the web application directly:
```bash
streamlit run app.py
```
*(Or simply run `python3 app.py`, which includes built-in auto-bootstrapping for Streamlit).*

Open your browser to `http://localhost:8501`.

---

## ⚙️ Configuration & VLM Setup

The vision AI engine connects by default to the **USAA / UTSA NVIDIA GB10 workstation** running `Qwen3-VL-8B-Instruct`. Connection parameters are defined in [`vlm_config.py`](vlm_config.py):

```python
BASE_URL = "http://10.246.100.230/v1"
MODEL = "qwen3-vl-8b-instruct"
API_KEY = "usaadatathon"
```

- **Custom VLM Endpoint**: To point to a local vLLM or Ollama instance, update `BASE_URL` in `vlm_config.py` or export standard OpenAI environment variables:
  ```bash
  export OPENAI_BASE_URL="http://localhost:8000/v1"
  export OPENAI_API_KEY="your-key"
  ```
- **Offline / Local Fallback**: If running without access to the GB10 server, set `USE_GB10_VLM=false` to use the embedded local CLIP ViT model:
  ```bash
  export USE_GB10_VLM="false"
  ```

---

## 🏆 Hackathon Pitch & Use Cases

### 1. Civic Safety & Smart City Integration (311SA)
Instead of relying on delayed citizen telephone calls, San Antonio 311 can triage incoming reports in real time. The model correlates submitted street photos with live sensor telemetry and low-water crossing proximity to flag false alarms and prioritize critical life-safety hazards.

### 2. Public Health & Biohazard Early Warning
Flash floods are hazardous not only due to drowning risks, but also due to sewage contamination. By integrating TCEQ Sanitary Sewer Overflow (SSO) data, FloodRunners warns citizens and emergency responders if receding floodwaters contain untreated wastewater.

### 3. First Responder Dispatch & Resource Allocation
San Antonio Fire Department (SAFD) and Public Works can monitor the **Citywide Risk Leaderboard** to deploy flood barricades, close roadways, and stage high-water rescue vehicles *before* low water crossings become fully submerged.

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
