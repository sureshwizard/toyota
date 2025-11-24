# GR PitSense Live – Real-Time Strategy Co-Driver (COTA Race 1)

## Overview

**GR PitSense Live** is a real-time race strategy console built for the  
**Hack the Track presented by Toyota GR** hackathon.

It uses official Toyota GR Cup data from **Circuit of the Americas – Race 1**  
to simulate a live race and answer:

- When should we **pit** this car?
- How many seconds can we **gain or lose** vs staying out?
- How does **tire age**, **track temp**, and **pace** interact over the stint?

Category: **Real-Time Analytics**

---

## Datasets Used (COTA R1)

From the provided Circuit of the Americas race package:

- Lap times: `cota_lap_time_r1.csv`
- Race results by class: `cota_results_by_class_r1.csv`
- Weather: `cota_weather_r1.csv`
- (Optional) Telemetry: `cota_telemetry_r1.csv`

These files are **not bundled** in this repo (per rules).  
Place them under `data/` as described in this README.

---

## How It Works

1. **Data Engineering**
   - Detects car, lap, lap time, gaps and track temperature columns.
   - Builds a unified lap-level table:
     - `car_id`, `lap`, `lap_time`, `tire_age_laps`, `gap_first`, `track_temp`, etc.
   - Optionally includes telemetry features (`speed`, `throttle`, `brake_f`, `brake_r`).

2. **Lap Time Model**
   - Trains a Gradient Boosting Regressor to predict `lap_time` from:
     - Lap number / tire age
     - Track temperature
     - Gap to leader
     - Optional telemetry features

3. **Pit Strategy Simulation**
   - For a chosen car & current lap:
     - Predicts next laps if the car **stays out**.
     - Evaluates candidate pit laps with:
       - One-time **pit lane loss**
       - Faster laps on **fresher tires**
     - Outputs an estimated **time gain vs no-pit** for each option.

4. **Interactive Dashboard (Streamlit)**
   - Select **car**, **current lap**, **candidate pit laps**, and **pit loss seconds**.
   - View:
     - Predicted future lap times
     - Strategy table (time gain vs no-pit)
     - Recommended pit window
     - Last 5 lap times and current gap to leader

---

## Tech Stack

- Python, Pandas, NumPy
- Scikit-learn
- Streamlit

---

## Running Locally

```bash
git clone <your-repo-url>
cd gr-pitsense-live
pip install -r requirements.txt

# Place COTA Race 1 CSVs here:
# data/cota_lap_time_r1.csv
# data/cota_results_by_class_r1.csv
# data/cota_weather_r1.csv
# data/cota_telemetry_r1.csv (optional)

streamlit run app.py

