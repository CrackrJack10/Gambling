import pandas as pd
import numpy as np
from personalv2 import predict_race_positions, predict_qualifying_positions
from pathlib import Path
import pickle

json_path = "f1db.json"
cache_dir = "cache"
def compare_props_to_predictions(
    race_id: int,
    prop_bets: list[tuple[int, int]]
) -> pd.DataFrame:
    """
    Compare a list of (driverId, proposed_placement) tuples to model predictions
    for a given race. Returns a DataFrame sorted by largest absolute difference.
    """
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    json_file = Path(json_path)
    cache_pkl = cache_dir / f"tables.pkl"

    if cache_pkl.exists() and cache_pkl.stat().st_mtime > json_file.stat().st_mtime:
        print("Loading cached tables...")
        with open(cache_pkl, "rb") as f:
            tables = pickle.load(f)

    # Step 1: Get model predictions for the race
    pred_df = predict_race_positions(race_id)

    # Step 2: Convert props to a DataFrame
    props_df = pd.DataFrame(prop_bets, columns=["driverId", "proposed_position"])

    # Step 3: Merge predictions with props
    merged = pred_df.merge(props_df, on="driverId", how="inner")

    # Step 4: Compute difference
    merged["difference"] = merged["predicted_race_pos_raw"] - merged["proposed_position"]
    merged["abs_difference"] = merged["difference"].abs()

    # Step 5: Sort by biggest disagreement
    merged = merged.sort_values("abs_difference", ascending=False)

    # Optional: add direction interpretation
    merged["model_says"] = np.where(
        merged["difference"] < 0,
        "Model thinks better finish",
        np.where(merged["difference"] > 0, "Model thinks worse finish", "Exact match")
    )

    # Step 6: Return summary
    return merged[
        [
            "officialName",
            "driverId",
            "proposed_position",
            "predicted_race_pos_raw",
            "difference",
            "abs_difference",
            "model_says"
        ]
    ]


def compare_qualifying_props_to_predictions(
    race_id: int,
    prop_bets: list[tuple[int, int]]
) -> pd.DataFrame:
    """
    Compare a list of (driverId, proposed_qualifying_placement) tuples to model predictions
    for a given race's qualifying session. Returns a DataFrame sorted by largest absolute difference.
    """
    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    json_file = Path(json_path)
    cache_pkl = cache_dir / f"tables.pkl"

    if cache_pkl.exists() and cache_pkl.stat().st_mtime > json_file.stat().st_mtime:
        print("Loading cached tables...")
        with open(cache_pkl, "rb") as f:
            tables = pickle.load(f)

    # Step 1: Get model predictions for the qualifying session
    pred_df = predict_qualifying_positions(race_id)

    # Step 2: Convert props to a DataFrame
    props_df = pd.DataFrame(prop_bets, columns=["driverId", "proposed_position"])

    # Step 3: Merge predictions with props
    merged = pred_df.merge(props_df, on="driverId", how="inner")

    # Step 4: Compute difference
    merged["difference"] = merged["predicted_qual_pos_raw"] - merged["proposed_position"]
    merged["abs_difference"] = merged["difference"].abs()

    # Step 5: Sort by biggest disagreement
    merged = merged.sort_values("abs_difference", ascending=False)

    # Optional: add direction interpretation
    merged["model_says"] = np.where(
        merged["difference"] < 0,
        "Model thinks better qualifying",
        np.where(merged["difference"] > 0, "Model thinks worse qualifying", "Exact match")
    )

    # Step 6: Return summary
    return merged[
        [
            "officialName",
            "driverId",
            "proposed_position",
            "predicted_qual_pos_raw",
            "difference",
            "abs_difference",
            "model_says"
        ]
    ]


if __name__ == "__main__":
    
    """    # Example race predictions comparison
    print("🏁 RACE PREDICTIONS COMPARISON:")
    print("="*50)
    race_comparison = compare_props_to_predictions(1145,
                                       [("lando-norris", 5),
                                        ("oscar-piastri", 6),
                                        ("charles-leclerc", 7),
                                        ("carlos-sainz-jr", 8)])
    print(race_comparison)
    """
    print("\n🏎️ QUALIFYING PREDICTIONS COMPARISON:")
    print("="*50)
    # Example qualifying predictions comparison
    try:
        qual_comparison = compare_qualifying_props_to_predictions(1146,
                                           [("lando-norris", 2),
                                            ("oscar-piastri", 3.5),
                                            ("charles-leclerc", 4.5),
                                            ("carlos-sainz-jr", 10.5),
                                            ("lewis-hamilton", 6),
                                            ("george-russell", 4.5),
                                            ("kimi-antonelli", 7.5),
                                            ("alexander-albon", 11),
                                            ("nico-hulkenberg", 13.5),
                                            ("isack-hadjar", 11.5),
                                            ("liam-lawson", 12.5),
                                            ("gabriel-bortoleto", 14.5),
                                            ("fernando-alonso", 9.5),
                                            ("yuki-tsunoda", 10.5),
                                            ("lance-stroll", 16.5),
                                            ])
        print(qual_comparison)
    except Exception as e:
        print(f"⚠️ Could not generate qualifying comparison: {e}")
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