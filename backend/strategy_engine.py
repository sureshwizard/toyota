# backend/strategy_engine.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split


def find_first_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Return the first column that exists in df from the candidate list.
    Comparison is case-insensitive and ignores spaces and '#'.
    """
    norm_map = {c.lower().replace(" ", "").replace("#", ""): c for c in df.columns}
    for cand in candidates:
        key = cand.lower().replace(" ", "").replace("#", "")
        if key in norm_map:
            return norm_map[key]
    return None


class StrategyEngine:
    """
    Core class that loads COTA Race 1 data, builds features, trains a lap-time
    model, and simulates pit strategies.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.lap_df = None
        self.results_df = None
        self.weather_df = None
        self.telemetry_df = None
        self.strategy_df = None
        self.model: Optional[GradientBoostingRegressor] = None

    # ---------- Data loading ----------

    def load_data(self):
        # Edit filenames here if you used different names.
        lap_path = self.data_dir / "cota_lap_time_r1.csv"
        results_path = self.data_dir / "cota_results_by_class_r1.csv"
        weather_path = self.data_dir / "cota_weather_r1.csv"
        telemetry_path = self.data_dir / "cota_telemetry_r1.csv"

        self.lap_df = pd.read_csv(lap_path)
        self.results_df = pd.read_csv(results_path)
        self.weather_df = pd.read_csv(weather_path)

        if telemetry_path.exists():
            self.telemetry_df = pd.read_csv(telemetry_path)
        else:
            self.telemetry_df = None

        # Normalize column names (keep original as well)
        self.lap_df.columns = [c.strip() for c in self.lap_df.columns]
        self.results_df.columns = [c.strip() for c in self.results_df.columns]
        self.weather_df.columns = [c.strip() for c in self.weather_df.columns]
        if self.telemetry_df is not None:
            self.telemetry_df.columns = [c.strip() for c in self.telemetry_df.columns]

    # ---------- Feature engineering ----------

    def engineer_features(self):
        """
        Create a lap-level strategy dataframe with unified columns:
        car_id, lap, lap_time, tire_age_laps, gap_first, track_temp, etc.
        """
        df = self.lap_df.copy()

        # Detect core columns in lap_df
# Support your actual columns: vehicle_id, lap, value
        car_col = find_first_column(
            df,
            ["car_id", "car", "Car", "Vehicle", "Car No", "CarNumber", "vehicle_id", "vehicleid"],
        )
        lap_col = find_first_column(df, ["lap", "Lap", "lap_number", "LapNumber", "lap no"])

        lap_time_col = find_first_column(
            df,
            [
                "lap_time",
                "Lap Time",
                "laptime",
                "LapTime",
                "Lap Time (s)",
                "LapTimeSeconds",
                "value",          # 👈 IMPORTANT: your lap time column
            ],
        )

        if car_col is None or lap_col is None or lap_time_col is None:
            raise ValueError(
                f"Could not detect car/lap/laptime columns in lap file. "
                f"Found columns: {list(df.columns)}"
            )

        df = df.rename(
            columns={
                car_col: "car_id",
                lap_col: "lap",
                lap_time_col: "lap_time",
            }
        )

        # Basic sorting
        df = df.sort_values(["car_id", "lap"]).reset_index(drop=True)

        # Simple tire age approximation = lap number (single-stint assumption)
        df["tire_age_laps"] = df["lap"]

        # Merge results for gap to leader etc.
        res = self.results_df.copy()
        res_car_col = find_first_column(res, ["car_id", "car", "Car", "Car No", "CarNumber"])
        pos_col = find_first_column(res, ["pos", "Position", "position", "Final Pos"])
        gap_first_col = find_first_column(res, ["gap_first", "Gap First", "Gap to First", "Gap 1st"])
        gap_prev_col = find_first_column(res, ["gap_previous", "Gap Prev", "Gap Previous", "Gap Prv"])

        rename_res = {}
        if res_car_col: rename_res[res_car_col] = "car_id"
        if pos_col: rename_res[pos_col] = "pos"
        if gap_first_col: rename_res[gap_first_col] = "gap_first"
        if gap_prev_col: rename_res[gap_prev_col] = "gap_previous"

        res = res.rename(columns=rename_res)

        cols_to_keep = [c for c in ["car_id", "pos", "gap_first", "gap_previous"] if c in res.columns]
        if "car_id" in cols_to_keep:
            res_small = res[cols_to_keep].drop_duplicates("car_id")
            df = df.merge(res_small, on="car_id", how="left")

        # Weather: we at least want track temperature
        w = self.weather_df.copy()
        track_temp_col = find_first_column(
            w,
            ["track_temp", "Track Temp", "TrackTemperature", "Track Temp (C)", "TrackTemp"],
        )
        if track_temp_col:
            track_temp_mean = w[track_temp_col].mean()
        else:
            track_temp_mean = 25.0  # fallback
        df["track_temp"] = track_temp_mean

        # Optional: simple telemetry features aggregated per car+lap
        if self.telemetry_df is not None:
            tdf = self.telemetry_df.copy()
            t_car_col = find_first_column(tdf, ["car_id", "car", "Car", "Car No", "CarNumber"])
            t_lap_col = find_first_column(tdf, ["lap", "Lap", "lap_number", "LapNumber", "lap no"])

            # Example telemetry: speed, throttle, brake
            speed_col = find_first_column(tdf, ["Speed", "speed", "vehicle_speed"])
            throttle_col = find_first_column(tdf, ["ath", "Throttle", "throttle", "aps"])
            brake_f_col = find_first_column(tdf, ["pbrake_f", "front_brake", "BrakeFront"])
            brake_r_col = find_first_column(tdf, ["pbrake_r", "rear_brake", "BrakeRear"])

            rename_t = {}
            if t_car_col: rename_t[t_car_col] = "car_id"
            if t_lap_col: rename_t[t_lap_col] = "lap"
            if speed_col: rename_t[speed_col] = "speed"
            if throttle_col: rename_t[throttle_col] = "throttle"
            if brake_f_col: rename_t[brake_f_col] = "brake_f"
            if brake_r_col: rename_t[brake_r_col] = "brake_r"

            tdf = tdf.rename(columns=rename_t)

            agg_cols = [c for c in ["speed", "throttle", "brake_f", "brake_r"] if c in tdf.columns]
            if "car_id" in tdf.columns and "lap" in tdf.columns and agg_cols:
                agg = tdf.groupby(["car_id", "lap"])[agg_cols].mean().reset_index()
                df = df.merge(agg, on=["car_id", "lap"], how="left")

        self.strategy_df = df

    # ---------- Model training ----------

    def train_lap_time_model(self):
        """
        Train a GradientBoostingRegressor to predict lap_time from features.
        """
        df = self.strategy_df.copy()

        if "lap_time" not in df.columns:
            raise ValueError("lap_time column not found in strategy_df.")

        # Select candidate feature columns
        feature_candidates = [
            "lap",
            "tire_age_laps",
            "track_temp",
            "gap_first",
            "speed",
            "throttle",
            "brake_f",
            "brake_r",
        ]
        feature_cols = [c for c in feature_candidates if c in df.columns]

        df = df.dropna(subset=["lap_time"])
        X = df[feature_cols]
        y = df["lap_time"]

        # Basic train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = GradientBoostingRegressor(random_state=42)
        model.fit(X_train, y_train)
        self.model = model

        r2 = model.score(X_test, y_test)
        print(f"[Lap time model] R^2: {r2:.3f} using features {feature_cols}")

    # ---------- Predictions & strategy ----------

    def predict_future_laps(self, car_id: int, current_lap: int, horizon: int = 8) -> Optional[pd.DataFrame]:
        """
        Predict the next N laps for a given car if it stays out.
        """
        if self.model is None:
            raise RuntimeError("Model is not trained. Call train_lap_time_model() first.")

        df_car = self.strategy_df[self.strategy_df["car_id"] == car_id]
        if df_car.empty:
            return None

        # Use the most recent row up to current_lap as context
        cur_rows = df_car[df_car["lap"] <= current_lap]
        if cur_rows.empty:
            cur_rows = df_car
        last = cur_rows.sort_values("lap").tail(1).iloc[0]

        preds = []
        for i in range(1, horizon + 1):
            future_lap = int(last["lap"]) + i

            row = {}
            for col in self.strategy_df.columns:
                row[col] = last[col]

            row["lap"] = future_lap
            row["tire_age_laps"] = future_lap  # simple approximation

            feature_candidates = [
                "lap",
                "tire_age_laps",
                "track_temp",
                "gap_first",
                "speed",
                "throttle",
                "brake_f",
                "brake_r",
            ]
            feature_cols = [c for c in feature_candidates if c in self.strategy_df.columns]
            x_vec = np.array([[row[c] for c in feature_cols]])

            lap_time_pred = self.model.predict(x_vec)[0]
            preds.append(
                {
                    "future_lap": future_lap,
                    "predicted_lap_time": lap_time_pred,
                }
            )

        return pd.DataFrame(preds)

    def simulate_pit_strategy(
        self,
        car_id: int,
        current_lap: int,
        pit_in_laps: List[int],
        pit_loss_seconds: float = 25.0,
    ) -> Optional[pd.DataFrame]:
        """
        Compare candidate pit laps versus staying out.
        Very simplified model:
        - You lose pit_loss_seconds once.
        - After a pit, your next 5 laps are (bonus_per_lap) faster.
        """
        baseline = self.predict_future_laps(car_id, current_lap, horizon=10)
        if baseline is None or baseline.empty:
            return None

        total_baseline_time = baseline["predicted_lap_time"].sum()

        strategies = []
        for pit_lap in pit_in_laps:
            # bonus after pit (faster laps)
            bonus_per_lap = 0.3  # seconds faster per lap on fresher tires
            laps_with_bonus = 5

            improved_time_gain = bonus_per_lap * laps_with_bonus
            net_effect_vs_stay_out = improved_time_gain - pit_loss_seconds
            # positive = better than staying out

            strategies.append(
                {
                    "pit_lap": pit_lap,
                    "pit_loss_seconds": pit_loss_seconds,
                    "estimated_time_gain_vs_no_pit": net_effect_vs_stay_out,
                }
            )

        return pd.DataFrame(strategies)

