# How We Got Here: The Story Behind FloodRunners

This page explains how FloodRunners went from the datathon's flood theme to the finished app. It's based on the organizers' starter materials and the project's commit history.

---

## 1. The Challenge

The **Dell Social Impact AI Datathon** asked teams to build something useful around **floods**. Every team got:

- **A Dell Pro Max GB10 workstation** running **Qwen3-VL**, an AI model that can look at a photo and answer questions about it.
- **A [starter kit](https://github.com/jennifer-fowler/GB10competition)** with setup instructions and example notebooks.
- **A list of public datasets** to build on (see [`UTSA_more_data.md`](UTSA_more_data.md)).

## 2. What the Starter Kit Suggested

The example notebooks used **50 aerial photos of Houston from before and after Hurricane Harvey** (25 from 2016 and 25 from 2017). They showed three things the AI could do:

1. Answer a question about a single image ("Is floodwater covering normally dry land?")
2. Find images that match a text description ("water around houses")
3. Compare a before/after pair alongside rainfall records

The suggested projects included a "what happened near my home" photo viewer, a searchable archive of flood images, and reports residents could use to request drainage improvements. The organizers also advised treating the AI's answer as **a suggestion for a person to check**, not a final verdict.

## 3. Where We Went Instead

FloodRunners took the tools from the starter kit in a different direction, with three main choices.

**1. Our own city.** We focused on **San Antonio** instead of Houston. Central Texas is known as "Flash Flood Alley": steep terrain and thin soils mean water rises fast.

**2. Street level, not aerial.** Instead of satellite or airplane photos, the app takes the kind of **photo a resident would send to 311**: a flooded street, a creek, a clogged storm drain.

**3. City data first, AI second.** The risk score works **without any photo at all**, using data the city already collects. A photo adds extra evidence on top. This keeps the AI as one input among several, which fits the organizers' advice.

The project's README gives the reasoning behind these choices:

- **Low-water crossings** are where drivers get trapped when water rises.
- **Floodwater can carry sewage** when sewer systems overflow, so flooding is a health risk as well as a drowning risk.
- **The information was scattered.** Residents, first responders and 311 dispatchers had to piece together maps, weather reports and phone calls by hand.

## 4. Picking the Data

From the organizers' dataset list, the app uses the ones that are **specific to San Antonio and tied to exact locations on a map**:

| Dataset | What it tells the app | Source |
| :--- | :--- | :--- |
| Low Water Crossings (156 crossings) | How close a spot is to a known flood-prone road crossing | [City of San Antonio GIS](https://opendata-cosagis.opendata.arcgis.com/datasets/4089adb69af14608acfb02703b1dd439_0/about) |
| Smart streetlight flood sensors | Water-level readings in the Downtown, Medical Center and Brooks areas | [San Antonio Open Data](https://data.sanantonio.gov/dataset/street-light-sensor-data) |
| Sanitary Sewer Overflows | Where sewage has overflowed before, for the contamination warning | [Texas Commission on Environmental Quality](https://data.texas.gov/dataset/Texas-Commission-on-Environmental-Quality-Sanitary/8kc5-95uk/about_data) |
| Qwen3-VL on the GB10 | Reads a resident's photo for visible flooding, depth and debris | Provided by the organizers |

The list also included FEMA flood-zone data, national disaster records, the city's storm drain maps (inlets, manholes, channels and outfalls), drainage maintenance records, state complaint records, and Microsoft's satellite flood-detection model. This version of the app doesn't use them, but they're good candidates for future versions.

## 5. How the Build Came Together

Most of the code was committed on the afternoon of **November 13**, and the README was rewritten on **November 14**. Here's the order things happened in:

1. **Starting from the kit.** The repo began as a copy of the organizers' setup guide and example notebooks. These were removed once our app replaced them.
2. **First prototype** ([`FLOODALERT.ipynb`](FLOODALERT.ipynb)). This notebook connected to the AI, combined a photo score with three map layers into one 0–100 risk score, and plotted the results on a map. At this point it used a few sample locations.
3. **First app.** A simple web page called the "Citizen Portal": enter a location and an optional photo, get a risk score.
4. **Real data.** All 156 crossings were loaded and converted from the city's map coordinate system into regular latitude and longitude. The app also started reading the latest sensor readings from Downtown and the Medical Center. The AI was set up to answer in a fixed format so the app could read its responses reliably.
5. **A bigger audience.** We added preset high-risk areas, a citywide leaderboard ranking them, a heatmap and a 311 report queue. That made the app useful for dispatchers and first responders, not only residents.
6. **Cleanup.** Photo input was narrowed to uploads only, which matches how a real resident's report works. The workshop files were removed and the README was rewritten.

## 6. Where It Could Go Next

- **Put the sewer overflows on the map.** The state's file lists incidents by county and waterway but not exact locations, so each one needs to be matched to a spot on the map. For now, the app uses three sample locations.
- **Read every sensor area.** The Brooks sensor data is stored as a zip file that the app doesn't open yet.
- **Use live data.** Switch from saved sensor files to live readings.
- **Add rainfall,** the way the starter kit's third notebook paired images with rain records.
- **Offer Spanish and plain-language versions** of the warnings. This was one of the organizers' suggested ideas.
- **Measure how accurate the photo scoring is** by testing it on a set of labeled flood photos.
