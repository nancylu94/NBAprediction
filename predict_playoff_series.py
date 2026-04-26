"""
predict_playoff_series.py

Trains an XGBoost classifier on regular season games, predicts NBA playoff
game outcomes, then runs a Monte Carlo simulation of each playoff series.

Inputs:
  data/features.csv              — feature matrix from build_rolling_features.py
  xgb_random_search_results.json — tuned XGBoost hyperparameters

Outputs:
  data/game_level_win_probabilities.csv — playoff games with home_win_prob
  data/series_simulation_results.csv    — series length distribution + competitiveness
"""

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURES_FILE = 'data/features.csv'
RESULTS_FILE = 'xgb_random_search_results.json'
GAME_LEVEL_OUTPUT = 'data/game_level_win_probabilities.csv'
SERIES_OUTPUT = 'data/series_simulation_results.csv'
RANDOM_STATE = 42
SIMULATION_ITERATIONS = 10_000
HOME_COURT_ORDER = [True, True, False, False, True, False, True]

# Playoff game types used as the test set — excludes play-in games
PLAYOFF_TEST_TYPES = {
    "Playoffs", "NBA Finals",
    "First Round", "East First Round", "West First Round",
    "Conference Semifinals", "East Second Round", "West Second Round",
    "Conference Finals", "East Conference Finals", "West Conference Finals",
}

_BASE_DIR = Path(__file__).parent

_METADATA_COLS = frozenset({
    'game_id', 'game_date', 'season_year', 'season_type', 'matchup',
    'home', 'away', 'team_id_home', 'team_id_away', 'team_name_home',
    'team_name_away', 'min',
})
_OUTCOME_COLS = frozenset({'winner', 'pts_home', 'pts_away', 'margin'})
_SERIES_CONTEXT_COLS = frozenset({'game_num_in_series', 'home_series_wins', 'away_series_wins'})
_PLAYER_NAME_RE = re.compile(r'^(home|away)_p\d+_PLAYER$')


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_features() -> pd.DataFrame:
    path = _BASE_DIR / FEATURES_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Features file not found: {path}\n"
            "Run build_rolling_features.py first."
        )
    print(f"Loading features from {path} ...")
    df = pd.read_csv(path, low_memory=False)
    print(f"  {len(df):,} rows x {df.shape[1]} columns loaded")
    return df


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------

def split_train_test(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if 'season_type' not in df.columns:
        raise ValueError(
            "Column 'season_type' missing — was features.csv produced by build_rolling_features.py?"
        )
    train = df[df['season_type'] == 'Regular Season'].copy()
    test = df[df['season_type'].isin(PLAYOFF_TEST_TYPES)].copy()
    if train.empty:
        raise ValueError("No regular season rows found in features.csv.")
    if test.empty:
        raise ValueError(
            "No playoff test rows found. "
            f"Check that PLAYOFF_TEST_TYPES matches season_type values in features.csv."
        )
    print(f"  Train (regular season): {len(train):,} rows")
    print(f"  Test  (playoff games):  {len(test):,} rows")
    return train, test


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------

def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Drop metadata, outcome, target, series context, and player-name string columns.
    Restrict to numeric columns. Fill NaN with 0 (with warning) if any remain.
    Returns (X, y) where y is the home_win target.
    """
    if 'home_win' not in df.columns:
        raise ValueError("Column 'home_win' not found — cannot build feature matrix.")

    player_name_cols = {c for c in df.columns if _PLAYER_NAME_RE.match(c)}
    drop_cols = _METADATA_COLS | _OUTCOME_COLS | {'home_win'} | _SERIES_CONTEXT_COLS | player_name_cols

    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    X = X.select_dtypes(include=['number']).copy()

    nan_counts = X.isna().sum()
    nan_cols = nan_counts[nan_counts > 0]
    if not nan_cols.empty:
        print(
            f"  WARNING: {nan_cols.sum():,} NaN values in {len(nan_cols)} columns — filling with 0:"
        )
        for col, cnt in nan_cols.items():
            print(f"    {col}: {cnt:,} NaNs")
        X = X.fillna(0)

    y = df['home_win']
    return X, y


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    results_path = _BASE_DIR / RESULTS_FILE
    if not results_path.exists():
        raise FileNotFoundError(
            f"Hyperparameter results not found: {results_path}\n"
            "Run xgb_randomized_search.py first."
        )
    payload = json.loads(results_path.read_text(encoding='utf-8'))
    best_params: dict = payload.get('best_params', {})
    if not best_params:
        raise ValueError("'best_params' missing or empty in results JSON.")
    print(f"  Hyperparameters: {best_params}")

    model = XGBClassifier(
        objective='binary:logistic',
        tree_method='hist',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
        **best_params,
    )
    model.fit(X_train, y_train)
    print(f"  Trained on {len(X_train):,} rows x {X_train.shape[1]} features")
    return model


# ---------------------------------------------------------------------------
# Game-level predictions
# ---------------------------------------------------------------------------

def predict_playoff_games(
    model: XGBClassifier,
    test_df: pd.DataFrame,
    X_test: pd.DataFrame,
) -> pd.DataFrame:
    probs = model.predict_proba(X_test)[:, 1]
    result = test_df.copy()
    result['home_win_prob'] = probs

    out_path = _BASE_DIR / GAME_LEVEL_OUTPUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"  Saved {len(result):,} game-level predictions -> {out_path}")
    return result


# ---------------------------------------------------------------------------
# Monte Carlo series simulation
# ---------------------------------------------------------------------------

def simulate_series(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Groups playoff games into series, builds a 7-probability vector for each,
    and runs SIMULATION_ITERATIONS Monte Carlo draws to estimate series length
    distribution and competitiveness score.

    full_probs[g] = P(game g's home team wins). HOME_COURT_ORDER determines
    whether the game-1 home team is home (True) or away (False) for slot g,
    so wins for the game-1 home team are tracked accordingly.
    """
    np.random.seed(RANDOM_STATE)

    required = {'season_year', 'home', 'away', 'game_num_in_series',
                'game_date', 'home_win_prob', 'home_win'}
    missing_cols = required - set(predictions.columns)
    if missing_cols:
        raise ValueError(f"simulate_series: missing required columns: {missing_cols}")

    # Build series key exactly as add_series_context() does in build_rolling_features.py
    preds = predictions.copy()
    preds['_series_key'] = preds.apply(
        lambda r: (
            f"{r['season_year']}"
            f"_{min(r['home'], r['away'])}"
            f"_{max(r['home'], r['away'])}"
        ),
        axis=1,
    )

    rows = []
    for series_key, group in preds.groupby('_series_key', sort=False):
        group = group.sort_values(['game_num_in_series', 'game_date'])
        played_probs = group['home_win_prob'].tolist()
        n_played = len(played_probs)

        avg_prob = float(np.mean(played_probs))

        # Fill unplayed game slots (series ended early) using HOME_COURT_ORDER
        full_probs: list[float] = list(played_probs)
        for g in range(n_played, 7):
            full_probs.append(avg_prob if HOME_COURT_ORDER[g] else 1.0 - avg_prob)

        # Simulate: track game-1-home-team wins via HOME_COURT_ORDER orientation
        length_counts: dict[int, int] = {4: 0, 5: 0, 6: 0, 7: 0}
        home_team_win_count = 0

        for _ in range(SIMULATION_ITERATIONS):
            g1h = 0  # game-1 home team cumulative wins
            g1a = 0  # game-1 away team cumulative wins
            n_games = 0
            for g, prob in enumerate(full_probs):
                n_games += 1
                r = np.random.rand()
                if HOME_COURT_ORDER[g]:
                    # game-1 home team is home this slot
                    if r < prob:
                        g1h += 1
                    else:
                        g1a += 1
                else:
                    # game-1 away team is home this slot
                    if r < prob:
                        g1a += 1
                    else:
                        g1h += 1
                if g1h == 4 or g1a == 4:
                    break
            if g1h == 4:
                home_team_win_count += 1
            length_counts[n_games] += 1

        total = SIMULATION_ITERATIONS
        season_year = group['season_year'].iloc[0]
        team_a = min(group['home'].iloc[0], group['away'].iloc[0])
        team_b = max(group['home'].iloc[0], group['away'].iloc[0])

        # Actual winner: count wins per team across played games
        team_wins: dict[str, int] = {}
        for _, row in group.iterrows():
            winning_team = row['home'] if row['home_win'] == 1 else row['away']
            team_wins[winning_team] = team_wins.get(winning_team, 0) + 1

        actual_winner: Optional[str] = None
        if team_wins:
            max_wins = max(team_wins.values())
            leaders = [t for t, w in team_wins.items() if w == max_wins]
            actual_winner = leaders[0] if len(leaders) == 1 else None

        rows.append({
            'series_key': series_key,
            'season_year': season_year,
            'team_a': team_a,
            'team_b': team_b,
            'games_4': length_counts[4] / total,
            'games_5': length_counts[5] / total,
            'games_6': length_counts[6] / total,
            'games_7': length_counts[7] / total,
            'competitiveness_6_or_7': (length_counts[6] + length_counts[7]) / total,
            'home_team_win_rate': home_team_win_count / total,
            'predicted_games': n_played,
            'actual_games': n_played,
            'actual_winner': actual_winner,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Step 1: Loading features...")
    df = load_features()

    print("Step 2: Splitting train/test...")
    train_df, test_df = split_train_test(df)

    print("Step 3: Building feature matrices...")
    X_train, y_train = build_feature_matrix(train_df)
    X_test, _ = build_feature_matrix(test_df)
    print(f"  {X_train.shape[1]} feature columns passed to model")

    print("Step 4: Training XGBoost model...")
    model = train_model(X_train, y_train)

    print("Step 5: Predicting playoff game outcomes...")
    predictions = predict_playoff_games(model, test_df, X_test)

    print("Step 6: Running Monte Carlo series simulation...")
    series_results = simulate_series(predictions)
    series_path = _BASE_DIR / SERIES_OUTPUT
    series_path.parent.mkdir(parents=True, exist_ok=True)
    series_results.to_csv(series_path, index=False)
    print(f"  Saved {len(series_results):,} series -> {series_path}")

    summary = {
        'train_rows': int(len(X_train)),
        'test_rows': int(len(X_test)),
        'feature_columns': int(X_train.shape[1]),
        'series_simulated': int(len(series_results)),
        'game_level_output': GAME_LEVEL_OUTPUT,
        'series_output': SERIES_OUTPUT,
    }
    print("\nJSON Summary:")
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
