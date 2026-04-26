"""
update_features_xlsx.py

Reads the existing data/features.csv and rewrites data/features.xlsx with:
  - 'features'           tab : all rows and columns from features.csv
  - 'data_dictionary'    tab : auto-generated column descriptions (bug-fixed)
  - 'prediction_outputs' tab : data dictionary for predict_playoff_series.py outputs

Run from the NBAprediction/ directory:
    python update_features_xlsx.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from build_rolling_features import build_data_dictionary

DATA_DIR    = Path(__file__).parent / "data"
INPUT_CSV   = DATA_DIR / "features.csv"
OUTPUT_XLSX = DATA_DIR / "features.xlsx"

# Data dictionary for the two output files from predict_playoff_series.py
# game_level_win_probabilities.csv columns (original columns + home_win_prob)
# series_simulation_results.csv columns
PREDICTION_OUTPUT_DICT = [
    # --- game_level_win_probabilities.csv ---
    {
        "file": "game_level_win_probabilities.csv",
        "column": "home_win_prob",
        "category": "Model output — game level",
        "description": (
            "XGBoost-predicted probability that the home team wins (0–1). "
            "Model trained on regular season games; applied to playoff games. "
            "Scoring metric: neg_log_loss to ensure calibrated probabilities."
        ),
    },
    # --- series_simulation_results.csv ---
    {
        "file": "series_simulation_results.csv",
        "column": "series_key",
        "category": "Simulation output — series level",
        "description": (
            "Identifier for the playoff series. Formed from seriesId, gameLabel, "
            "or team IDs depending on which fields are present in the data."
        ),
    },
    {
        "file": "series_simulation_results.csv",
        "column": "games_4",
        "category": "Simulation output — series level",
        "description": (
            "Simulated probability the series ends in exactly 4 games (sweep). "
            "Based on 10,000 Monte Carlo iterations using home_win_prob per game "
            "and the standard 2-2-1-1-1 home court schedule."
        ),
    },
    {
        "file": "series_simulation_results.csv",
        "column": "games_5",
        "category": "Simulation output — series level",
        "description": "Simulated probability the series ends in exactly 5 games.",
    },
    {
        "file": "series_simulation_results.csv",
        "column": "games_6",
        "category": "Simulation output — series level",
        "description": "Simulated probability the series ends in exactly 6 games.",
    },
    {
        "file": "series_simulation_results.csv",
        "column": "games_7",
        "category": "Simulation output — series level",
        "description": "Simulated probability the series ends in exactly 7 games.",
    },
    {
        "file": "series_simulation_results.csv",
        "column": "competitiveness_6_or_7",
        "category": "Simulation output — series level",
        "description": (
            "P(series goes 6 or 7 games). Primary competitiveness score used to rank "
            "series for broadcaster ad-inventory planning. A sweep produces ~0; "
            "an evenly matched series can reach 0.60+. "
            "Every extra game is worth ~$35M–$40M in broadcast ad revenue."
        ),
    },
    {
        "file": "series_simulation_results.csv",
        "column": "home_win_rate",
        "category": "Simulation output — series level",
        "description": (
            "Fraction of 10,000 Monte Carlo iterations in which the home team "
            "(higher seed / series host) wins the series."
        ),
    },
    {
        "file": "series_simulation_results.csv",
        "column": "away_win_rate",
        "category": "Simulation output — series level",
        "description": (
            "Fraction of 10,000 Monte Carlo iterations in which the away team "
            "(lower seed) wins the series. Equals 1 - home_win_rate."
        ),
    },
    {
        "file": "series_simulation_results.csv",
        "column": "predicted_games",
        "category": "Simulation output — series level",
        "description": (
            "Number of actual playoff game rows found for this series in the input data "
            "(up to 7). Used to confirm how many per-game probabilities fed into the simulation."
        ),
    },
]


def main():
    print(f"Reading {INPUT_CSV} ...")
    games = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"  {len(games):,} rows x {games.shape[1]} columns")

    print("Building data dictionary ...")
    data_dict = build_data_dictionary(list(games.columns))
    print(f"  {len(data_dict)} dictionary rows")

    pred_dict = pd.DataFrame(PREDICTION_OUTPUT_DICT)

    # Round floats to 6 dp before writing — openpyxl writes full Python float
    # precision (17 digits) which inflates sheet XML to 270+ MB and breaks Excel.
    # xlsxwriter produces a clean, compact file.
    float_cols = games.select_dtypes(include=float).columns
    games[float_cols] = games[float_cols].round(6)

    print(f"Writing {OUTPUT_XLSX} ...")
    with pd.ExcelWriter(OUTPUT_XLSX, engine="xlsxwriter") as writer:
        games.to_excel(writer, sheet_name="features", index=False)
        data_dict.to_excel(writer, sheet_name="data_dictionary", index=False)
        pred_dict.to_excel(writer, sheet_name="prediction_outputs", index=False)

    print("Done.")
    print(f"  Sheets: features ({len(games):,} rows), "
          f"data_dictionary ({len(data_dict)} rows), "
          f"prediction_outputs ({len(pred_dict)} rows)")


if __name__ == "__main__":
    main()
