"""
f1db_json_ml.py

Unified Formula 1 ML data pipeline using a single F1DB JSON file (schema v6.x).

Features:
- Loads a single f1db.json (drivers, constructors, races, results)
- Flattens all practice, qualifying, and race result data
- Builds per-driver-per-race dataset with circuit metadata
- Optionally caches processed tables to speed up reruns

Usage:
    python f1db_json_ml.py ./f1db.json
"""

import json
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
json_path = "f1db.json"

# Circuit ID to numeric mapping for consistent encoding
CIRCUIT_ID_MAP = {
    "adelaide": 1, "aida": 2, "ain-diab": 3, "aintree": 4, "anderstorp": 5,
    "austin": 6, "avus": 7, "bahrain": 8, "baku": 9, "brands-hatch": 10,
    "bremgarten": 11, "buddh": 12, "buenos-aires": 13, "bugatti": 14, "caesars-palace": 15,
    "catalunya": 16, "clermont-ferrand": 17, "dallas": 18, "detroit": 19, "dijon": 20,
    "donington": 21, "east-london": 22, "estoril": 23, "fuji": 24, "hockenheimring": 25,
    "hungaroring": 26, "imola": 27, "indianapolis": 28, "interlagos": 29, "istanbul": 30,
    "jacarepagua": 31, "jarama": 32, "jeddah": 33, "jerez": 34, "kyalami": 35,
    "las-vegas": 36, "long-beach": 37, "lusail": 38, "magny-cours": 39, "marina-bay": 40,
    "melbourne": 41, "mexico-city": 42, "miami": 43, "monaco": 44, "monsanto": 45,
    "mont-tremblant": 46, "montjuic": 47, "montreal": 48, "monza": 49, "mosport": 50,
    "mugello": 51, "nivelles": 52, "nurburgring": 53, "paul-ricard": 54, "pedralbes": 55,
    "pescara": 56, "phoenix": 57, "portimao": 58, "porto": 59, "reims": 60,
    "riverside": 61, "rouen": 62, "sebring": 63, "sepang": 64, "shanghai": 65,
    "silverstone": 66, "sochi": 67, "spa-francorchamps": 68, "spielberg": 69, "suzuka": 70,
    "valencia": 71, "watkins-glen": 72, "yas-marina": 73, "yeongam": 74, "zandvoort": 75,
    "zeltweg": 76, "zolder": 77
}

feature_cols = [         # +ve if improved last race
        "form_avg_2races",
        "form_avg_3races",
        "form_avg_5races",
        "prev_race_time",
        "prev_laps_completed",
        "prev_race_gapMillis",
        "prev_race_intervalMillis",
        "prev_race_pos",
        "form_trend_exp", 
        "qual_pos",
        "qual_time",
        "qual_q1_time",
        "qual_q2_time",
        "qual_q3_time",
        "qual_gapMillis",
        "fp1_time",
        "fp2_time",
        "fp3_time",
        "fp1_pos",
        "fp2_pos",
        "fp3_pos",
        "fp1_gap",
        "fp2_gap",
        "fp3_gap",
        "fp1_interval",
        "fp2_interval",
        "fp3_interval",
        "year",
        "round",
        "circuit_id_numeric",
        "circuit_courseLength",
        "circuit_turns",
        "circuit_laps",
        "distance"
    ]

qual_feature_cols = [         # Base qualifying features (no practice data)
        "form_avg_2races",
        "form_avg_3races",
        "form_avg_5races",
        "prev_race_time",
        "prev_laps_completed",
        "prev_race_gapMillis",
        "prev_race_intervalMillis",
        "prev_race_pos",
        "prev_qual_pos",
        "prev_qual_q1_time",
        "prev_qual_q2_time",
        "prev_qual_q3_time",
        "prev_qual_gapMillis",
        "form_trend_exp", 
        "year",
        "round",
        "circuit_id_numeric",
        "circuit_courseLength",
        "circuit_turns",
        "circuit_laps",
        "distance"
    ]

qual_feature_cols_after_fp1 = qual_feature_cols + [
        "fp1_time",
        "fp1_pos", 
        "fp1_gap",
        "fp1_interval"
    ]

qual_feature_cols_after_fp2 = qual_feature_cols_after_fp1 + [
        "fp2_time",
        "fp2_pos",
        "fp2_gap", 
        "fp2_interval"
    ]

qual_feature_cols_after_fp3 = qual_feature_cols_after_fp2 + [
        "fp3_time",
        "fp3_pos",
        "fp3_gap",
        "fp3_interval"
    ]

cache_dir = "cache"

DRIVER_MAP = {"lando-norris":1,
             "oscar-piastri":2,
             "charles-leclerc":3,
             "carlos-sainz-jr":4,
             "lewis-hamilton":5,
             "george-russell":6,
             "max-verstappen":7,
             "lance-stroll":8,
             "fernando-alonso":9,
             "esteban-ocon":10,
             "pierre-gasly":11,
             "yuki-tsunoda":12,
             "kimi-antonelli":13,
             "alexander-albon":14,
             "nico-hulkenberg":15,
             "isack-hadjar":16,
             "oliver-bearman":17,
             "liam-lawson":18,
             "gabriel-bortoleto":19,
             "franco-colapinto":20}


target_drivers = ["lando-norris",
             "oscar-piastri",
             "charles-leclerc",
             "carlos-sainz-jr",
             "lewis-hamilton",
             "george-russell",
             "max-verstappen",
             "lance-stroll",
             "fernando-alonso",
             "esteban-ocon",
             "pierre-gasly",
             "yuki-tsunoda",
             "kimi-antonelli",
             "alexander-albon",
             "nico-hulkenberg",
             "isack-hadjar",
             "oliver-bearman",
             "liam-lawson",
             "gabriel-bortoleto",
             "franco-colapinto"
]
# ============================================================
#  HELPERS
# ============================================================

def load_tables(json_path: str, cache_dir: str = "cache") -> Dict[str, pd.DataFrame]:
    """Load cached tables if available, else extract from JSON and cache."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    json_file = Path(json_path)
    cache_pkl = cache_dir / f"tables.pkl"

    if cache_pkl.exists() and cache_pkl.stat().st_mtime > json_file.stat().st_mtime:
        print("Loading cached tables...")
        with open(cache_pkl, "rb") as f:
            tables = pickle.load(f)
    else:
        print("Flattening JSON data...")
        tables = extract_flat_tables(load_json(json_path))
        tables["driver_event_df"] = build_driver_event_dataset(tables)
        with open(cache_pkl, "wb") as f:
            pickle.dump(tables, f)
    return tables

def parse_position_number(item: dict) -> Optional[float]:
    """Parse numeric position from schema fields (positionNumber or positionText)."""
    pn = item.get("positionNumber")
    if pn is not None:
        try:
            return float(pn)
        except Exception:
            return np.nan

    pt = item.get("positionText")
    if isinstance(pt, str) and pt.isdigit():
        return float(pt)
    return np.nan

def load_json(path: str) -> Dict[str, Any]:
    """Load the F1DB single-file JSON dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_recency_weights(df: pd.DataFrame, date_col: str, half_life_days: float = 180.0) -> np.ndarray:
    """
    Compute exponentially decaying sample weights based on recency.
    - Recent races (near max_date) → weight ≈ 1
    - Older races → smaller weights (exponential decay)
    - Future races (after max_date) → weight = 1
    - half_life_days = 0 or None → uniform weights
    """
    if date_col not in df.columns:
        raise ValueError(f"Missing date column: {date_col}")

    # Convert to datetime safely
    dates = pd.to_datetime(df[date_col], errors="coerce")

    # Handle half_life = 0 → no bias
    if not half_life_days or half_life_days <= 0:
        return np.ones(len(df))

    # Compute day difference from the most recent known race
    valid_dates = dates.dropna()
    if valid_dates.empty:
        return np.ones(len(df))
    max_date = valid_dates.max()

    # Positive = older races; negative = future races
    days_diff = (max_date - dates).dt.days

    # Replace negatives (future races) with 0 → full weight
    days_diff = days_diff.clip(lower=0)

    # Exponential decay
    weights = 0.5 ** (days_diff / half_life_days)

    # Replace NaNs or infinities with neutral weight (1.0)
    weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
    return weights

# ============================================================
#  FLATTEN RAW JSON → SEPARATE TABLES
# ============================================================

def extract_flat_tables(single_json: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    """Flatten the F1DB single-file JSON into separate normalized DataFrames."""

    drivers = pd.DataFrame(single_json.get("drivers", []))
    constructors = pd.DataFrame(single_json.get("constructors", []))
    races_raw = single_json.get("races", [])

    races, practices, qualifying, race_results = [], [], [], []

    # ---- Flatten races and session results ----
    for race in races_raw:
        race_id = race.get("id")
        date = pd.to_datetime(race.get("date"), errors="coerce")

        races.append({
            "race_id": race_id,
            "year": race.get("year"),
            "round": race.get("round"),
            "date": date,
            "circuitId": race.get("circuitId"),
            "officialName": race.get("officialName"),
            "circuit_courseLength": race.get("courseLength"),
            "circuit_turns": race.get("turns"),
            "circuit_laps": race.get("laps"),
            "scheduledLaps": race.get("scheduledLaps"),
            "distance": race.get("distance"),
        })

        # Helper to flatten nested results
        def flatten_session(arr, session_name):
            if not arr:
                return
            for item in arr:
                entry = dict(item)
                entry.update({
                    "race_id": race_id,
                    "date": date,
                    "session": session_name,
                    "position_num": parse_position_number(item)
                })
                yield entry

        # Practice sessions (FP1–FP4)
        for i in range(1, 5):
            for row in flatten_session(race.get(f"freePractice{i}Results") or [], f"FP{i}"):
                practices.append(row)

        # Qualifying sessions (normal, sprint, etc.)
        for key in ["qualifyingResults", "qualifying1Results", "qualifying2Results", "sprintQualifyingResults"]:
            for row in flatten_session(race.get(key) or [], key.replace("Results", "")):
                qualifying.append(row)

        # Race + Sprint results
        for key in ["raceResults", "sprintRaceResults"]:
            for row in flatten_session(race.get(key) or [], key.replace("Results", "")):
                race_results.append(row)

    # ---- Convert to DataFrames ----
    races_df = pd.DataFrame(races)
    practice_df = pd.DataFrame(practices)
    qualifying_df = pd.DataFrame(qualifying)
    results_df = pd.DataFrame(race_results)

    # ---- Normalize ----
    def normalize(df: pd.DataFrame, time_fields: List[str]) -> pd.DataFrame:
        if df.empty:
            return df
        if "driver_id" in df.columns:
            df.rename(columns={"driver_id": "driverId"}, inplace=True)
        if "constructor_id" in df.columns:
            df.rename(columns={"constructor_id": "constructorId"}, inplace=True)

        for tf in time_fields:
            if tf in df.columns:
                df[tf] = pd.to_numeric(df[tf], errors="coerce")
        return df

    practice_df = normalize(practice_df, ["timeMillis"])
    qualifying_df = normalize(qualifying_df, ["timeMillis", "q1Millis", "q2Millis", "q3Millis"])
    results_df = normalize(results_df, ["timeMillis"])

    return {
        "drivers": drivers,
        "constructors": constructors,
        "races": races_df,
        "practice": practice_df,
        "qualifying": qualifying_df,
        "results": results_df
    }

# ============================================================
#  BUILD UNIFIED DRIVER–RACE DATASET
# ============================================================

def build_driver_event_dataset(
    tables: Dict[str, pd.DataFrame],
    min_year = 2020
) -> pd.DataFrame:
    """
    Build a unified per-driver-per-race dataset.

    Includes race, qualifying (pos, q1/q2/q3/gap), practice, and circuit metadata.
    Optionally filters to include only races from min_year and later.
    """

    races_df = tables["races"]
    results_df = tables["results"]
    qualifying_df = tables["qualifying"]
    practice_df = tables["practice"]

    # --- 0. Filter by year if requested ---
    if min_year is not None and "year" in races_df.columns:
        race_ids_to_keep = races_df.loc[races_df["year"] >= min_year, "race_id"]
        results_df = results_df[results_df["race_id"].isin(race_ids_to_keep)]
        qualifying_df = qualifying_df[qualifying_df["race_id"].isin(race_ids_to_keep)]
        practice_df = practice_df[practice_df["race_id"].isin(race_ids_to_keep)]
        races_df = races_df[races_df["race_id"].isin(race_ids_to_keep)]
        print(f"📅 Filtering dataset to races from {min_year} onward ({len(race_ids_to_keep)} races)")

    # --- 1. Base from race results ---
    base = results_df[[
        "race_id", "driverId", "constructorId",
        "position_num", "timeMillis", "laps", "gapMillis", "intervalMillis"
    ]].copy()

    base.rename(columns={
        "position_num": "race_pos",
        "timeMillis": "race_time",
        "laps": "laps_completed",
        "gapMillis": "race_gapMillis",
        "intervalMillis": "race_intervalMillis"
    }, inplace=True)
    
    # Check for duplicates in base race results
    base_dupes = base.duplicated(subset=["race_id", "driverId"]).sum()
    if base_dupes > 0:
        print(f"⚠️  Found {base_dupes} duplicate driver entries in race results (likely multiple race sessions)")
        # Keep the best result (lowest position) for each driver per race
        base = base.sort_values("race_pos").drop_duplicates(subset=["race_id", "driverId"], keep="first")
        print("✅ Removed duplicates, keeping best race result per driver per race")

    # --- 2. Add full qualifying data ---
    if not qualifying_df.empty:
        # Debug: Check for duplicates before aggregation
        dupe_check = qualifying_df.groupby(["race_id", "driverId"]).size()
        if (dupe_check > 1).any():
            max_dupes = dupe_check.max()
            print(f"⚠️  Found drivers with up to {max_dupes} qualifying entries per race (multiple sessions/formats)")
        
        # Use best qualifying position (minimum) and fastest time per driver per race
        qual = (
            qualifying_df.groupby(["race_id", "driverId"], as_index=False)
            .agg({
                "position_num": "min",  # Best qualifying position
                "timeMillis": lambda x: x.dropna().min() if x.dropna().any() else np.nan,  # Fastest lap
                "q1Millis": lambda x: x.dropna().min() if x.dropna().any() else np.nan,
                "q2Millis": lambda x: x.dropna().min() if x.dropna().any() else np.nan,
                "q3Millis": lambda x: x.dropna().min() if x.dropna().any() else np.nan,
                "gapMillis": lambda x: x.dropna().min() if x.dropna().any() else np.nan
            })
            .rename(columns={
                "position_num": "qual_pos",
                "timeMillis": "qual_time",
                "q1Millis": "qual_q1_time",
                "q2Millis": "qual_q2_time",
                "q3Millis": "qual_q3_time",
                "gapMillis": "qual_gapMillis"
            })
        )
        
        # Ensure no duplicates after aggregation
        qual_dupes = qual.duplicated(subset=["race_id", "driverId"]).sum()
        if qual_dupes > 0:
            print(f"❌ ERROR: {qual_dupes} duplicate driver entries found after qualifying aggregation!")
            qual = qual.drop_duplicates(subset=["race_id", "driverId"], keep="first")
            print("✅ Removed duplicates, keeping first occurrence")
        
        base = base.merge(qual, on=["race_id", "driverId"], how="left")
    else:
        for col in ["qual_pos", "qual_time", "qual_q1_time", "qual_q2_time", "qual_q3_time", "qual_gapMillis"]:
            base[col] = np.nan

    # --- 3. Add practice sessions (FP1–FP3) ---
    if not practice_df.empty and "session" in practice_df.columns:
        summary = (
            practice_df.groupby(["race_id", "driverId", "session"], as_index=False)
            .agg({"position_num": "min", 
                  "timeMillis": "min",
                  "gapMillis": "min",
                  "intervalMillis": "min"})
        )
        pivot = summary.pivot(index=["race_id", "driverId"], columns="session", values=["timeMillis", "position_num", "gapMillis", "intervalMillis"])

        pivot.columns = [
            f"{sess.lower()}_{'time' if m == 'timeMillis' else 'pos' if m == 'position_num' else 'gap' if m == 'gapMillis' else 'interval'}"
            for m, sess in zip(pivot.columns.get_level_values(0), pivot.columns.get_level_values(1))
        ]
        pivot.reset_index(inplace=True)

        base = base.merge(pivot, on=["race_id", "driverId"], how="left")

    # --- 4. Add race/circuit metadata ---
    meta_cols = [
        "race_id", "year", "round", "date", "circuitId",
        "circuit_courseLength", "circuit_turns", "circuit_laps",
        "scheduledLaps", "distance"
    ]
    races_meta = races_df[[c for c in meta_cols if c in races_df.columns]].copy()
    races_meta.rename(columns={"date": "date"}, inplace=True)

    # ensure datetime type
    if "date" in races_meta.columns:
        races_meta["date"] = pd.to_datetime(races_meta["date"], errors="coerce")
    base = base.merge(races_meta, on="race_id", how="left")

    # --- 4.1. Add numeric circuit ID ---
    if "circuitId" in base.columns:
        base["circuit_id_numeric"] = base["circuitId"].map(CIRCUIT_ID_MAP).fillna(0)

    # --- 5. Clean numeric fields ---
    for c in base.columns:
        if any(x in c for x in ["_pos", "_time", "_gap", "laps", "distance", "turns", "Length"]):
            base[c] = pd.to_numeric(base[c], errors="coerce")

    # --- 6. Fill missing with neutral defaults ---
    defaults = {
        "race_pos": 20.0,
        "qual_pos": 20.0,
        "race_time": np.nanmedian(base["race_time"].dropna()) if base["race_time"].notna().any() else 0.0,
        "qual_time": np.nanmedian(base["qual_time"].dropna()) if base["qual_time"].notna().any() else 0.0,
        "laps_completed": 50.0,
        "scheduledLaps": 50.0,
        "circuit_courseLength": 5000.0,
        "circuit_turns": 15.0,
        "distance": 300000.0,
    }
    for c, v in defaults.items():
        if c in base.columns:
            base[c].fillna(v, inplace=True)

    # --- 7. Final deduplication check ---
    final_dupes = base.duplicated(subset=["race_id", "driverId"]).sum()
    if final_dupes > 0:
        print(f"❌ CRITICAL: {final_dupes} duplicate driver entries found in final dataset!")
        print("Removing duplicates...")
        base = base.drop_duplicates(subset=["race_id", "driverId"], keep="first")
        print("✅ Final dataset deduplicated")
    
    print(f"📊 Final driver-event dataset: {len(base)} unique driver-race combinations")

    base.sort_values(["race_id", "driverId"], inplace=True, ignore_index=True)
    for n in [2, 3, 5]:
        base[f"form_avg_{n}races"] = (
            base.groupby("driverId")["race_pos"]
            .apply(lambda x: x.shift(1).rolling(n, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )

    # Rate of improvement: Δ position vs. previous race
    base["pos_change_from_last"] = (
        base.groupby("driverId")["race_pos"].diff(1) * -1
    )  # negative diff means improvement

    # Weighted trend (exponential smoothing)
    base["form_trend_exp"] = (
        base.groupby("driverId")["race_pos"]
        .apply(lambda x: x.shift(1).ewm(span=3, adjust=False).mean())
        .reset_index(level=0, drop=True)
    )

    lag_cols = ["race_time", "laps_completed", "race_gapMillis", "race_intervalMillis", "race_pos"]

    for col in lag_cols:
        base[f"prev_{col}"] = (
            base.sort_values(["driverId", "year", "round"])
            .groupby("driverId")[col]
            .shift(1)
        )

    # Optional: drop current-race versions to avoid leakage
    # base.drop(columns=lag_cols, inplace=True)

    # Fill NaNs for drivers' first races
    for col in lag_cols:
        base[f"prev_{col}"] = base[f"prev_{col}"].fillna(0)

    # Add qualifying lag features
    qual_lag_cols = ["qual_pos", "qual_q1_time", "qual_q2_time", "qual_q3_time", "qual_gapMillis"]
    
    for col in qual_lag_cols:
        if col in base.columns:
            base[f"prev_{col}"] = (
                base.sort_values(["driverId", "year", "round"])
                .groupby("driverId")[col]
                .shift(1)
            )
            base[f"prev_{col}"] = base[f"prev_{col}"].fillna(0)

        # -----------------------------
        # Return cleaned dataset
        # -----------------------------
        base.sort_values(["race_id", "driverId"], inplace=True, ignore_index=True)
        return base


# ============================================================
#  MAIN PIPELINE
# ============================================================
def train_and_select_best_model(df: pd.DataFrame, feature_cols: list, target_col: str = "race_pos", cv_splits: int = 10):
    """
    Train models using temporal (date-based) cross-validation — ensuring no future race data is used.
    """

    # Ensure chronological order
    df = df.sort_values("date").reset_index(drop=True)
    unique_dates = sorted(df["date"].unique())


    # Define splits based on race dates
    date_splits = np.linspace(0, len(unique_dates) - 1, cv_splits + 1, dtype=int)
    # Candidate models
    model_grid = {
        "gbr": {
            "model": GradientBoostingRegressor,
            "params": {"n_estimators": [200, 400], "learning_rate": [0.025,0.05, 0.1], "max_depth": [3, 5]}
        }
    }

    best_rmse = float("inf")
    best_config = None

    # Loop over each model and param combination
    for model_name, model_info in model_grid.items():
        for params in ParameterGrid(model_info["params"]):
            rmses = []

            for i in range(1, len(date_splits) - 1):
                train_dates = unique_dates[:date_splits[i]]
                val_dates = unique_dates[date_splits[i]:date_splits[i+1]]
                train_df = df[df["date"].isin(train_dates)]
                val_df = df[df["date"].isin(val_dates)]

                if train_df.empty or val_df.empty:
                    continue

                X_train = train_df[feature_cols].fillna(0.0)
                y_train = train_df[target_col]
                X_val = val_df[feature_cols].fillna(0.0)
                y_val = val_df[target_col]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)

                model = model_info["model"](**params)
                model.fit(X_train_scaled, y_train)
                preds = model.predict(X_val_scaled)

                rmse = root_mean_squared_error(y_val, preds)
                rmses.append(rmse)

            if rmses:
                mean_rmse = np.mean(rmses)
                print(f"{model_name} {params} → mean RMSE={mean_rmse:.4f}")

                if mean_rmse < best_rmse:
                    best_rmse = mean_rmse
                    best_config = {
                        "model_name": model_name,
                        "params": params,
                        "rmse": best_rmse,
                        "scaler": scaler,
                        "model": model
                    }

    if not best_config:
        raise RuntimeError("No valid model trained. Check your date or feature setup.")

    # Save best model
    with open("best_race_model.pkl", "wb") as f:
        pickle.dump(best_config, f)

    print(f"\n✅ Best model: {best_config['model_name']} ({best_config['params']}) | RMSE={best_config['rmse']:.4f}")
    return best_config

def run_pipeline(json_path: str, cache_dir: str = "cache") -> Dict[str, pd.DataFrame]:
    """Load F1DB JSON, flatten to tables, and build unified driver–event dataset."""

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(exist_ok=True)
    json_file = Path(json_path)
    cache_pkl = cache_dir / f"tables.pkl"

    if cache_pkl.exists() and cache_pkl.stat().st_mtime > json_file.stat().st_mtime:
        print("Loading cached tables...")
        with open(cache_pkl, "rb") as f:
            tables = pickle.load(f)
    else:
        print("Flattening JSON data...")
        tables = extract_flat_tables(load_json(json_path))
        tables["driver_event_df"] = build_driver_event_dataset(tables)
        with open(cache_pkl, "wb") as f:
            pickle.dump(tables, f)

    target_col = "race_pos"

    best_config = train_and_select_best_model(
        df=tables["driver_event_df"],
        feature_cols=feature_cols,
        target_col=target_col,        # ensure your races_df merge added a 'date' field
    )

    print(f"\n✅ Best model: {best_config['model_name']} ({best_config['params']}) | RMSE={best_config['rmse']:.4f}")
    with open("best_race_model.pkl", "wb") as f:
        pickle.dump(best_config, f)
    return tables

def build_future_race_features(race_id: int, driver_event_df: pd.DataFrame, race_meta: pd.DataFrame, drivers: list) -> pd.DataFrame:
    """
    Build feature data for a future race by calculating historical statistics up to the race date.
    """
    race_date = race_meta["date"].iloc[0]
    
    # Get all historical data before this race
    historical_df = driver_event_df[driver_event_df["date"] < race_date].copy()
    
    # Create base dataframe for this race with all target drivers
    race_df_list = []
    
    for driver in drivers:
        # Get driver's historical data
        driver_hist = historical_df[historical_df["driverId"] == driver].copy()
        
        if driver_hist.empty:
            # New driver with no history - use defaults
            driver_features = {
                "race_id": race_id,
                "driverId": driver,
                "constructorId": "unknown",  # Will need to be filled
                "form_avg_2races": 15.0,
                "form_avg_3races": 15.0,
                "form_avg_5races": 15.0,
                "form_avg_3qual": 15.0,
                "form_avg_5qual": 15.0,
                "prev_race_time": 0.0,
                "prev_laps_completed": 50.0,
                "prev_race_gapMillis": 0.0,
                "prev_race_intervalMillis": 0.0,
                "prev_race_pos": 15.0,
                "prev_qual_pos": 15.0,
                "prev_q1_time": 0.0,
                "prev_q2_time": 0.0,
                "prev_q3_time": 0.0,
                "prev_qual_gapMillis": 0.0,
                "form_trend_exp": 15.0,
                "qual_pos": 15.0,
                "qual_time": 0.0,
                "qual_q1_time": 0.0,
                "qual_q2_time": 0.0,
                "qual_q3_time": 0.0,
                "qual_gapMillis": 0.0,
                "fp1_time": 0.0, "fp2_time": 0.0, "fp3_time": 0.0,
                "fp1_pos": 15.0, "fp2_pos": 15.0, "fp3_pos": 15.0,
                "fp1_gap": 0.0, "fp2_gap": 0.0, "fp3_gap": 0.0,
                "fp1_interval": 0.0, "fp2_interval": 0.0, "fp3_interval": 0.0,
            }
        else:
            # Calculate features from historical data
            driver_hist = driver_hist.sort_values("date").reset_index(drop=True)
            latest_data = driver_hist.iloc[-1] if len(driver_hist) > 0 else None
            
            # Calculate form averages
            race_positions = driver_hist["race_pos"].dropna()
            qual_positions = driver_hist["qual_pos"].dropna()
            
            driver_features = {
                "race_id": race_id,
                "driverId": driver,
                "constructorId": latest_data["constructorId"] if latest_data is not None else "unknown",
                "form_avg_2races": race_positions.tail(2).mean() if len(race_positions) >= 1 else 15.0,
                "form_avg_3races": race_positions.tail(3).mean() if len(race_positions) >= 1 else 15.0,
                "form_avg_5races": race_positions.tail(5).mean() if len(race_positions) >= 1 else 15.0,
                "form_avg_3qual": qual_positions.tail(3).mean() if len(qual_positions) >= 1 else 15.0,
                "form_avg_5qual": qual_positions.tail(5).mean() if len(qual_positions) >= 1 else 15.0,
                "prev_race_time": latest_data["race_time"] if latest_data is not None else 0.0,
                "prev_laps_completed": latest_data["laps_completed"] if latest_data is not None else 50.0,
                "prev_race_gapMillis": latest_data["race_gapMillis"] if latest_data is not None else 0.0,
                "prev_race_intervalMillis": latest_data["race_intervalMillis"] if latest_data is not None else 0.0,
                "prev_race_pos": latest_data["race_pos"] if latest_data is not None else 15.0,
                "prev_qual_pos": latest_data["qual_pos"] if latest_data is not None else 15.0,
                "prev_q1_time": latest_data["qual_q1_time"] if latest_data is not None else 0.0,
                "prev_q2_time": latest_data["qual_q2_time"] if latest_data is not None else 0.0,
                "prev_q3_time": latest_data["qual_q3_time"] if latest_data is not None else 0.0,
                "prev_qual_gapMillis": latest_data["qual_gapMillis"] if latest_data is not None else 0.0,
                "form_trend_exp": race_positions.tail(3).mean() if len(race_positions) >= 1 else 15.0,
                # Current race data - set to defaults since this is prediction
                "qual_pos": 15.0,
                "qual_time": 0.0,
                "qual_q1_time": 0.0,
                "qual_q2_time": 0.0,
                "qual_q3_time": 0.0,
                "qual_gapMillis": 0.0,
                "fp1_time": 0.0, "fp2_time": 0.0, "fp3_time": 0.0,
                "fp1_pos": 15.0, "fp2_pos": 15.0, "fp3_pos": 15.0,
                "fp1_gap": 0.0, "fp2_gap": 0.0, "fp3_gap": 0.0,
                "fp1_interval": 0.0, "fp2_interval": 0.0, "fp3_interval": 0.0,
            }
        
        # Add race metadata
        driver_features.update({
            "year": race_meta["year"].iloc[0],
            "round": race_meta["round"].iloc[0],
            "date": race_meta["date"].iloc[0],
            "circuitId": race_meta["circuitId"].iloc[0] if "circuitId" in race_meta.columns else "unknown",
            "circuit_courseLength": race_meta.get("circuit_courseLength", {}).iloc[0] if "circuit_courseLength" in race_meta.columns else 0.0,
            "circuit_turns": race_meta.get("circuit_turns", {}).iloc[0] if "circuit_turns" in race_meta.columns else 0.0,
            "circuit_laps": race_meta.get("circuit_laps", {}).iloc[0] if "circuit_laps" in race_meta.columns else 0.0,
            "distance": race_meta.get("distance", {}).iloc[0] if "distance" in race_meta.columns else 0.0,
        })
        
        race_df_list.append(driver_features)
    
    # Convert to DataFrame
    race_df = pd.DataFrame(race_df_list)
    
    # Add numeric circuit ID
    if "circuitId" in race_df.columns:
        race_df["circuit_id_numeric"] = race_df["circuitId"].map(CIRCUIT_ID_MAP).fillna(0)
    
    return race_df


def predict_race_positions(race_id: int) -> pd.DataFrame:
    """
    Predict race results for a given race_id, ensuring only data from *before* that race is used.
    Can handle future races by building features dynamically from historical data.
    """
    tables = load_tables(json_path)
    with open("best_race_model.pkl", "rb") as f:
        best_config = pickle.load(f)
        
    driver_event_df = tables["driver_event_df"]
    races_df = tables["races"]
    model = best_config["model"]
    scaler = best_config["scaler"]

    # Get race metadata and date
    race_meta = races_df.loc[races_df["race_id"] == race_id]
    if race_meta.empty:
        raise ValueError(f"Race ID {race_id} not found in races table.")
    race_date = race_meta["date"].iloc[0]

    # Check if race data exists in driver_event_df
    existing_race_df = driver_event_df[driver_event_df["race_id"] == race_id].copy()
    
    if not existing_race_df.empty:
        # Race data exists, use it directly
        race_df = existing_race_df
    else:
        # Race data doesn't exist (future race), build features dynamically
        print(f"🔮 Building features dynamically for future race {race_id}")
        race_df = build_future_race_features(race_id, driver_event_df, race_meta, target_drivers)

    # Ensure all feature columns are present
    for col in feature_cols:
        if col not in race_df.columns:
            race_df[col] = 0.0

    X = race_df[feature_cols].fillna(0.0)
    X_scaled = scaler.transform(X)

    race_df["predicted_race_pos_raw"] = model.predict(X_scaled)
    race_df["predicted_race_pos"] = race_df["predicted_race_pos_raw"].rank(method="first", ascending=True).astype(int)

    race_df = race_df.merge(
        race_meta[["race_id", "officialName"]],
        on="race_id",
        how="left"
    )

    race_df.sort_values("predicted_race_pos", inplace=True)
    return race_df[["race_id", "officialName", "driverId", "constructorId", "predicted_race_pos_raw"]]


def train_and_select_best_qualifying_model(df: pd.DataFrame, feature_cols: list, target_col: str = "qual_pos", cv_splits: int = 10):
    """
    Train models for qualifying position using temporal (date-based) cross-validation.
    """

    # Ensure chronological order
    df = df.sort_values("date").reset_index(drop=True)
    unique_dates = sorted(df["date"].unique())

    # Define splits based on race dates
    date_splits = np.linspace(0, len(unique_dates) - 1, cv_splits + 1, dtype=int)
    
    # Candidate models
    model_grid = {
        "gbr": {
            "model": GradientBoostingRegressor,
            "params": {"n_estimators": [200], "learning_rate": [0.025], "max_depth": [3, 5]}
        }
    }

    best_rmse = float("inf")
    best_config = None

    # Loop over each model and param combination
    for model_name, model_info in model_grid.items():
        for params in ParameterGrid(model_info["params"]):
            rmses = []

            for i in range(1, len(date_splits) - 1):
                train_dates = unique_dates[:date_splits[i]]
                val_dates = unique_dates[date_splits[i]:date_splits[i+1]]
                train_df = df[df["date"].isin(train_dates)]
                val_df = df[df["date"].isin(val_dates)]

                if train_df.empty or val_df.empty:
                    continue

                X_train = train_df[feature_cols].fillna(0.0)
                y_train = train_df[target_col]
                X_val = val_df[feature_cols].fillna(0.0)
                y_val = val_df[target_col]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)

                model = model_info["model"](**params)
                model.fit(X_train_scaled, y_train)
                preds = model.predict(X_val_scaled)

                rmse = root_mean_squared_error(y_val, preds)
                rmses.append(rmse)

            if rmses:
                mean_rmse = np.mean(rmses)
                print(f"{model_name} {params} → mean RMSE={mean_rmse:.4f}")

                if mean_rmse < best_rmse:
                    best_rmse = mean_rmse
                    best_config = {
                        "model_name": model_name,
                        "params": params,
                        "rmse": best_rmse,
                        "scaler": scaler,
                        "model": model
                    }

    if not best_config:
        raise RuntimeError("No valid qualifying model trained. Check your date or feature setup.")

    # Save best model
    with open("best_qualifying_model.pkl", "wb") as f:
        pickle.dump(best_config, f)

    print(f"\n✅ Best qualifying model: {best_config['model_name']} ({best_config['params']}) | RMSE={best_config['rmse']:.4f}")
    return best_config


def train_and_select_best_qualifying_model_after_fp1(df: pd.DataFrame, feature_cols: list, target_col: str = "qual_pos", cv_splits: int = 10):
    """Train qualifying model with FP1 data included."""
    return _train_progressive_qualifying_model(df, feature_cols, target_col, cv_splits, "fp1")

def train_and_select_best_qualifying_model_after_fp2(df: pd.DataFrame, feature_cols: list, target_col: str = "qual_pos", cv_splits: int = 10):
    """Train qualifying model with FP1 + FP2 data included.""" 
    return _train_progressive_qualifying_model(df, feature_cols, target_col, cv_splits, "fp2")

def train_and_select_best_qualifying_model_after_fp3(df: pd.DataFrame, feature_cols: list, target_col: str = "qual_pos", cv_splits: int = 10):
    """Train qualifying model with FP1 + FP2 + FP3 data included."""
    return _train_progressive_qualifying_model(df, feature_cols, target_col, cv_splits, "fp3")

def _train_progressive_qualifying_model(df: pd.DataFrame, feature_cols: list, target_col: str, cv_splits: int, practice_stage: str):
    """
    Common training logic for progressive qualifying models.
    """
    # Ensure chronological order
    df = df.sort_values("date").reset_index(drop=True)
    unique_dates = sorted(df["date"].unique())

    # Define splits based on race dates
    date_splits = np.linspace(0, len(unique_dates) - 1, cv_splits + 1, dtype=int)
    
    # Candidate models
    model_grid = {
        "gbr": {
            "model": GradientBoostingRegressor,
            "params": {"n_estimators": [200], "learning_rate": [0.025], "max_depth": [3, 5]}
        }
    }

    best_rmse = float("inf")
    best_config = None

    # Loop over each model and param combination
    for model_name, model_info in model_grid.items():
        for params in ParameterGrid(model_info["params"]):
            rmses = []

            for i in range(1, len(date_splits) - 1):
                train_dates = unique_dates[:date_splits[i]]
                val_dates = unique_dates[date_splits[i]:date_splits[i+1]]
                train_df = df[df["date"].isin(train_dates)]
                val_df = df[df["date"].isin(val_dates)]

                if train_df.empty or val_df.empty:
                    continue

                X_train = train_df[feature_cols].fillna(0.0)
                y_train = train_df[target_col]
                X_val = val_df[feature_cols].fillna(0.0)
                y_val = val_df[target_col]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)

                model = model_info["model"](**params)
                model.fit(X_train_scaled, y_train)
                preds = model.predict(X_val_scaled)

                rmse = root_mean_squared_error(y_val, preds)
                rmses.append(rmse)

            if rmses:
                mean_rmse = np.mean(rmses)
                print(f"{model_name} {params} → mean RMSE={mean_rmse:.4f}")

                if mean_rmse < best_rmse:
                    best_rmse = mean_rmse
                    best_config = {
                        "model_name": model_name,
                        "params": params,
                        "rmse": best_rmse,
                        "scaler": scaler,
                        "model": model
                    }

    if not best_config:
        raise RuntimeError(f"No valid qualifying model trained for {practice_stage}. Check your date or feature setup.")

    # Save best model with practice stage identifier
    model_filename = f"best_qualifying_model_after_{practice_stage}.pkl"
    with open(model_filename, "wb") as f:
        pickle.dump(best_config, f)

    print(f"\n✅ Best qualifying model after {practice_stage.upper()}: {best_config['model_name']} ({best_config['params']}) | RMSE={best_config['rmse']:.4f}")
    return best_config


def predict_qualifying_positions(race_id: int) -> pd.DataFrame:
    """
    Predict qualifying results for a given race_id, ensuring only data from *before* that race is used.
    Can handle future races by building features dynamically from historical data.
    """
    tables = load_tables(json_path)
    with open("best_qualifying_model.pkl", "rb") as f:
        best_config = pickle.load(f)
        
    driver_event_df = tables["driver_event_df"]
    races_df = tables["races"]
    model = best_config["model"]
    scaler = best_config["scaler"]

    # Get race metadata and date
    race_meta = races_df.loc[races_df["race_id"] == race_id]
    if race_meta.empty:
        raise ValueError(f"Race ID {race_id} not found in races table.")
    race_date = race_meta["date"].iloc[0]

    # Check if race data exists in driver_event_df
    existing_race_df = driver_event_df[driver_event_df["race_id"] == race_id].copy()
    
    if not existing_race_df.empty:
        # Race data exists, use it directly
        race_df = existing_race_df
    else:
        # Race data doesn't exist (future race), build features dynamically
        print(f"🔮 Building qualifying features dynamically for future race {race_id}")
        race_df = build_future_race_features(race_id, driver_event_df, race_meta, target_drivers)

    # Ensure all feature columns are present
    for col in qual_feature_cols:
        if col not in race_df.columns:
            race_df[col] = 0.0

    X = race_df[qual_feature_cols].fillna(0.0)
    X_scaled = scaler.transform(X)

    race_df["predicted_qual_pos_raw"] = model.predict(X_scaled)
    race_df["predicted_qual_pos"] = race_df["predicted_qual_pos_raw"].rank(method="first", ascending=True).astype(int)

    race_df = race_df.merge(
        race_meta[["race_id", "officialName"]],
        on="race_id",
        how="left"
    )

    race_df.sort_values("predicted_qual_pos", inplace=True)
    return race_df[["race_id", "officialName", "driverId", "constructorId", "predicted_qual_pos_raw"]]


def predict_qualifying_positions_after_fp1(race_id: int) -> pd.DataFrame:
    """Predict qualifying results after FP1 session."""
    return _predict_progressive_qualifying(race_id, qual_feature_cols_after_fp1, "fp1")

def predict_qualifying_positions_after_fp2(race_id: int) -> pd.DataFrame:
    """Predict qualifying results after FP1 + FP2 sessions."""
    return _predict_progressive_qualifying(race_id, qual_feature_cols_after_fp2, "fp2")

def predict_qualifying_positions_after_fp3(race_id: int) -> pd.DataFrame:
    """Predict qualifying results after FP1 + FP2 + FP3 sessions."""
    return _predict_progressive_qualifying(race_id, qual_feature_cols_after_fp3, "fp3")

def _predict_progressive_qualifying(race_id: int, feature_cols: list, practice_stage: str) -> pd.DataFrame:
    """
    Common prediction logic for progressive qualifying models.
    """
    tables = load_tables(json_path)
    model_filename = f"best_qualifying_model_after_{practice_stage}.pkl"
    
    with open(model_filename, "rb") as f:
        best_config = pickle.load(f)
        
    driver_event_df = tables["driver_event_df"]
    races_df = tables["races"]
    model = best_config["model"]
    scaler = best_config["scaler"]

    # Get race metadata and date
    race_meta = races_df.loc[races_df["race_id"] == race_id]
    if race_meta.empty:
        raise ValueError(f"Race ID {race_id} not found in races table.")
    race_date = race_meta["date"].iloc[0]

    # Check if race data exists in driver_event_df
    existing_race_df = driver_event_df[driver_event_df["race_id"] == race_id].copy()
    
    if not existing_race_df.empty:
        # Race data exists, use it directly
        race_df = existing_race_df
    else:
        # Race data doesn't exist (future race), build features dynamically
        print(f"🔮 Building qualifying features dynamically for future race {race_id} after {practice_stage.upper()}")
        race_df = build_future_race_features(race_id, driver_event_df, race_meta, target_drivers)

    # Ensure all feature columns are present
    for col in feature_cols:
        if col not in race_df.columns:
            race_df[col] = 0.0

    X = race_df[feature_cols].fillna(0.0)
    X_scaled = scaler.transform(X)

    race_df["predicted_qual_pos_raw"] = model.predict(X_scaled)
    race_df["predicted_qual_pos"] = race_df["predicted_qual_pos_raw"].rank(method="first", ascending=True).astype(int)

    race_df = race_df.merge(
        race_meta[["race_id", "officialName"]],
        on="race_id",
        how="left"
    )

    race_df.sort_values("predicted_qual_pos", inplace=True)
    return race_df[["race_id", "officialName", "driverId", "constructorId", "predicted_qual_pos_raw"]]


# ============================================================
#  CLI USAGE
# ============================================================

if __name__ == "__main__":
    """json_path = "f1db.json"
    tables = run_pipeline(json_path)

    # Save all tables to CSV for inspection
    outputs = {
        "drivers": "Driver information",
        "constructors": "Constructor/team information",
        "races": "Race schedule and circuits",
        "practice": "Free practice results",
        "qualifying": "Qualifying results",
        "results": "Race results",
        "driver_event_df": "Unified driver–event dataset"
    }

    for name, desc in outputs.items():
        df = tables.get(name)
        if df is None or df.empty:
            print(f"❌ Skipping {name}: no data")
            continue
        filename = f"{name}_data.csv"
        df.to_csv(filename, index=False)
        print(f"✅ {filename} ({len(df):,} rows) — {desc}")

    print("\n📊 All tables saved successfully.")"""
    tables = load_tables(json_path)
    # Train models
    driver_event_df = tables["driver_event_df"]
    
    # Train race position model
    print("\n🏁 Training race position model...")
    #train_and_select_best_model(driver_event_df, feature_cols, "race_pos")
    
    # Train qualifying position models
    print("\n🏎️ Training base qualifying position model...")
    train_and_select_best_qualifying_model(driver_event_df, qual_feature_cols, "qual_pos")
    
    print("\n🏁 Training qualifying model after FP1...")
    train_and_select_best_qualifying_model_after_fp1(driver_event_df, qual_feature_cols_after_fp1, "qual_pos")
    
    print("\n🏁 Training qualifying model after FP2...")
    train_and_select_best_qualifying_model_after_fp2(driver_event_df, qual_feature_cols_after_fp2, "qual_pos")
    
    print("\n🏁 Training qualifying model after FP3...")
    train_and_select_best_qualifying_model_after_fp3(driver_event_df, qual_feature_cols_after_fp3, "qual_pos")
    
    # Example predictions for a specific race
    example_race_id = 1146  # Change this to any race ID you want to predict
    print(f"\n🔮 Example predictions for race ID {example_race_id}:")
    
    # Race predictions
    race_predictions = predict_race_positions(example_race_id)
    if not race_predictions.empty:
        race_predictions_filename = f"race_predictions_{example_race_id}.csv"
        race_predictions.to_csv(race_predictions_filename, index=False)
        print(f"💾 Race predictions saved to {race_predictions_filename}")
    
    # Progressive qualifying predictions
    try:
        # Base qualifying (no practice data)
        qual_predictions = predict_qualifying_positions(example_race_id)
        if not qual_predictions.empty:
            qual_predictions_filename = f"qualifying_predictions_{example_race_id}.csv"
            qual_predictions.to_csv(qual_predictions_filename, index=False)
            print(f"💾 Base qualifying predictions saved to {qual_predictions_filename}")
            
        # After FP1
        qual_predictions_fp1 = predict_qualifying_positions_after_fp1(example_race_id)
        if not qual_predictions_fp1.empty:
            qual_predictions_fp1_filename = f"qualifying_predictions_after_fp1_{example_race_id}.csv"
            qual_predictions_fp1.to_csv(qual_predictions_fp1_filename, index=False)
            print(f"💾 Qualifying predictions after FP1 saved to {qual_predictions_fp1_filename}")
            
        # After FP2
        qual_predictions_fp2 = predict_qualifying_positions_after_fp2(example_race_id)
        if not qual_predictions_fp2.empty:
            qual_predictions_fp2_filename = f"qualifying_predictions_after_fp2_{example_race_id}.csv"
            qual_predictions_fp2.to_csv(qual_predictions_fp2_filename, index=False)
            print(f"💾 Qualifying predictions after FP2 saved to {qual_predictions_fp2_filename}")
            
        # After FP3
        qual_predictions_fp3 = predict_qualifying_positions_after_fp3(example_race_id)
        if not qual_predictions_fp3.empty:
            qual_predictions_fp3_filename = f"qualifying_predictions_after_fp3_{example_race_id}.csv"
            qual_predictions_fp3.to_csv(qual_predictions_fp3_filename, index=False)
            print(f"💾 Qualifying predictions after FP3 saved to {qual_predictions_fp3_filename}")
            
    except Exception as e:
        print(f"⚠️ Could not generate qualifying predictions: {e}")
