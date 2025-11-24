# app.py
import streamlit as st
import pandas as pd

from backend.strategy_engine import StrategyEngine

st.set_page_config(
    page_title="GR PitSense Live – COTA R1",
    layout="wide",
)


@st.cache_resource
def load_engine():
    engine = StrategyEngine(data_dir="data")
    engine.load_data()
    engine.engineer_features()
    engine.train_lap_time_model()
    return engine


st.title("🏁 GR PitSense Live – Real-Time Strategy Co-Driver")
st.caption("Hack the Track presented by Toyota GR – Real-Time Analytics | Circuit of the Americas – Race 1")

engine = load_engine()
df = engine.strategy_df

st.sidebar.header("Race Control Panel")

# Car & lap selection
if "car_id" not in df.columns:
    st.error("Could not find 'car_id' column after processing. Please check your data mapping.")
    st.stop()

car_ids = sorted(df["car_id"].unique())
car_id = st.sidebar.selectbox("Select Car", car_ids)

laps_for_car = sorted(df[df["car_id"] == car_id]["lap"].unique())
current_lap = st.sidebar.slider(
    "Current Lap (Simulation)",
    min_value=int(min(laps_for_car)),
    max_value=int(max(laps_for_car)),
    value=int(min(laps_for_car)) + 3 if len(laps_for_car) > 3 else int(min(laps_for_car)),
)

# Candidate pit laps
pit_options = list(
    range(
        current_lap + 1,
        min(current_lap + 7, int(max(laps_for_car)) + 1),
    )
)
pit_laps_selected = st.sidebar.multiselect(
    "Candidate Pit Laps",
    pit_options,
    default=pit_options[:3],
)

pit_loss = st.sidebar.number_input(
    "Estimated Pit Lane Loss (seconds)",
    min_value=10.0,
    max_value=60.0,
    value=25.0,
    step=0.5,
)

col_main, col_side = st.columns([3, 2])

with col_main:
    st.subheader("📈 Predicted Pace – Stay Out vs Pit Window")
    future_df = engine.predict_future_laps(car_id, current_lap, horizon=8)

    if future_df is not None and not future_df.empty:
        chart_df = future_df.set_index("future_lap")["predicted_lap_time"]
        st.line_chart(chart_df, height=260)
    else:
        st.info("Not enough data to predict future laps for this car.")

    st.subheader("🧠 Strategy Comparison")
    if pit_laps_selected:
        strat_df = engine.simulate_pit_strategy(
            car_id, current_lap, pit_laps_selected, pit_loss_seconds=pit_loss
        )
        if strat_df is not None and not strat_df.empty:
            st.dataframe(strat_df, use_container_width=True)

            best_row = strat_df.sort_values(
                "estimated_time_gain_vs_no_pit", ascending=False
            ).head(1)
            best_pit_lap = int(best_row["pit_lap"].values[0])
            gain = float(best_row["estimated_time_gain_vs_no_pit"].values[0])

            if gain > 0:
                st.success(
                    f"🏆 Recommended pit around **Lap {best_pit_lap}** "
                    f"for an estimated **gain of {gain:.1f} s** vs staying out."
                )
            else:
                st.warning(
                    f"Staying out may be comparable or better – "
                    f"best tested pit (Lap {best_pit_lap}) only gains {gain:.1f} s vs no pit."
                )
        else:
            st.info("No strategy comparison available. Try selecting more candidate pit laps.")
    else:
        st.info("Select at least one candidate pit lap in the sidebar to see strategy comparison.")

with col_side:
    st.subheader("📊 Driver & Race Snapshot")

    cur_row = df[(df["car_id"] == car_id) & (df["lap"] == current_lap)]
    if not cur_row.empty:
        cur_row = cur_row.iloc[0]

        if "gap_first" in df.columns and not pd.isna(cur_row.get("gap_first")):
            st.metric("Gap to Leader", f"{float(cur_row['gap_first']):.1f} s")

        st.write(f"**Current Lap:** {int(cur_row['lap'])}")
        if "pos" in df.columns and not pd.isna(cur_row.get("pos")):
            st.write(f"**Race Position (final):** {int(cur_row['pos'])}")

    if "lap_time" in df.columns:
        recent = (
            df[(df["car_id"] == car_id) & (df["lap"] <= current_lap)]
            .sort_values("lap")
            .tail(5)
        )
        if not recent.empty:
            st.caption("Last 5 Lap Times")
            st.table(recent[["lap", "lap_time"]])

    st.markdown("---")
    st.caption(
        "PitSense Live replays Toyota GR Cup COTA Race 1 as if it were live, "
        "letting engineers test pit decisions, tire stints, and attack windows."
    )

