import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

TRAIN_FILE = 'Games__1___1_.csv'
RESULTS_FILE = 'xgb_random_search_results.json'
OUTPUT_FILE = 'game_level_win_probabilities.csv'
SIMULATION_OUTPUT_FILE = 'series_simulation_results.csv'
RANDOM_STATE = 42
SIMULATION_ITERATIONS = 10_000
HOME_COURT_ORDER = [True, True, False, False, True, False, True]


def raise_if_missing(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f'Missing required file: {path}')


def make_home_win_target(df: pd.DataFrame) -> pd.Series:
    if 'winner' not in df.columns or 'hometeamId' not in df.columns:
        raise SystemExit('Source data must contain winner and hometeamId columns.')
    return (df['winner'] == df['hometeamId']).astype(int)


def clean_series_game_number(df: pd.DataFrame) -> pd.DataFrame:
    if 'seriesGameNumber' not in df.columns:
        return df
    df = df.copy()
    df['seriesGameNumber'] = (
        df['seriesGameNumber']
        .astype(str)
        .str.extract(r'(\d+)')
        .astype(float)
        .astype('Int64')
    )
    return df


def numeric_feature_matrix(df: pd.DataFrame, drop_columns: Iterable[str]) -> pd.DataFrame:
    features = df.drop(columns=[col for col in drop_columns if col in df.columns])
    features = features.select_dtypes(include=['number']).copy()
    if features.isna().any().any():
        raise SystemExit('Feature matrix contains missing values; clean data before training.')
    return features


def load_training_data() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    train_path = Path(TRAIN_FILE)
    raise_if_missing(train_path)

    df = pd.read_csv(train_path, low_memory=False)
    if 'gameType' not in df.columns:
        raise SystemExit('Training file must contain gameType column.')

    df = df.loc[df['gameType'] == 'Regular Season'].copy()
    if df.empty:
        raise SystemExit('No regular season games found in training data.')

    y = make_home_win_target(df)
    feature_exclusions = [
        'gameId', 'gameDateTimeEst', 'gameType', 'winner',
        'homeScore', 'awayScore', 'attendance', 'home_win',
        'gameLabel', 'seriesGameNumber', 'gameStatus', 'gameStatusText',
    ]
    X = numeric_feature_matrix(df, feature_exclusions)
    feature_columns = list(X.columns)

    return X, y, feature_columns


def load_best_params() -> dict:
    results_path = Path(RESULTS_FILE)
    raise_if_missing(results_path)

    payload = json.loads(results_path.read_text(encoding='utf-8'))
    best_params = payload.get('best_params')
    if not isinstance(best_params, dict):
        raise SystemExit('best_params not found in JSON results file.')
    return best_params


def prepare_prediction_frame(feature_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    predict_path = Path(TRAIN_FILE)
    raise_if_missing(predict_path)

    df = pd.read_csv(predict_path, low_memory=False)
    df = df.loc[df['gameType'] == 'Playoffs'].copy()
    if df.empty:
        raise SystemExit('No playoff games found for prediction.')

    df = clean_series_game_number(df)
    feature_exclusions = [
        'gameId', 'gameDateTimeEst', 'gameType', 'winner',
        'homeScore', 'awayScore', 'attendance', 'gameLabel',
        'seriesGameNumber', 'gameStatus', 'gameStatusText',
    ]
    features = numeric_feature_matrix(df, feature_exclusions)

    missing = [col for col in feature_columns if col not in features.columns]
    extra = [col for col in features.columns if col not in feature_columns]
    for column in missing:
        features[column] = 0
    if extra:
        features = features.drop(columns=extra)
    features = features[feature_columns]

    return df, features


def simulate_series(predictions: pd.DataFrame) -> pd.DataFrame:
    np.random.seed(RANDOM_STATE)

    if 'seriesGameNumber' in predictions.columns:
        predictions = predictions.sort_values(['seriesGameNumber', 'gameDateTimeEst'])

    if 'seriesId' in predictions.columns:
        group_keys = ['seriesId']
    elif 'gameLabel' in predictions.columns:
        group_keys = ['gameLabel', 'hometeamId', 'awayteamId']
    else:
        group_keys = ['hometeamId', 'awayteamId']

    rows = []
    for series_id, group in predictions.groupby(group_keys, dropna=False):
        ordered = group.sort_values(by=['seriesGameNumber', 'gameDateTimeEst'] if 'seriesGameNumber' in group.columns else ['gameDateTimeEst'])
        played_probs = ordered['home_win_prob'].tolist()[:7]
        if len(played_probs) == 0:
            continue

        # Build exactly 7 probabilities: played games use model output; unplayed
        # games use the series average home_win_prob, flipped when the higher seed
        # is away for that slot (HOME_COURT_ORDER[g] = False).
        avg_prob = float(np.mean(played_probs))
        full_probs = list(played_probs)
        for g in range(len(played_probs), 7):
            full_probs.append(avg_prob if HOME_COURT_ORDER[g] else 1.0 - avg_prob)

        length_counts = {length: 0 for length in range(4, 8)}
        winner_counts = {'home': 0, 'away': 0}
        for _ in range(SIMULATION_ITERATIONS):
            home_wins = 0
            away_wins = 0
            games_played = 0
            for prob in full_probs:
                games_played += 1
                if np.random.rand() < prob:
                    home_wins += 1
                else:
                    away_wins += 1
                if home_wins == 4 or away_wins == 4:
                    break
            winner = 'home' if home_wins == 4 else 'away'
            winner_counts[winner] += 1
            length_counts[games_played] += 1

        total = SIMULATION_ITERATIONS
        rows.append({
            'series_key': '|'.join(map(str, series_id)) if isinstance(series_id, tuple) else str(series_id),
            'games_4': length_counts[4] / total,
            'games_5': length_counts[5] / total,
            'games_6': length_counts[6] / total,
            'games_7': length_counts[7] / total,
            'competitiveness_6_or_7': (length_counts[6] + length_counts[7]) / total,
            'home_win_rate': winner_counts['home'] / total,
            'away_win_rate': winner_counts['away'] / total,
            'predicted_games': len(played_probs),
        })

    return pd.DataFrame(rows)


def main() -> None:
    X_train, y_train, feature_columns = load_training_data()
    best_params = load_best_params()
    original_df, predict_features = prepare_prediction_frame(feature_columns)

    model = XGBClassifier(
        objective='binary:logistic',
        tree_method='hist',
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
        **best_params,
    )
    model.fit(X_train, y_train)

    predictions = model.predict_proba(predict_features)[:, 1]
    output_df = original_df.copy()
    output_df['home_win_prob'] = predictions

    output_path = Path(OUTPUT_FILE)
    output_df.to_csv(output_path, index=False, encoding='utf-8')

    simulation_results = simulate_series(output_df)
    simulation_results.to_csv(Path(SIMULATION_OUTPUT_FILE), index=False, encoding='utf-8')

    print(json.dumps({
        'training_rows': int(len(X_train)),
        'playoff_rows': int(len(output_df)),
        'feature_columns': int(len(feature_columns)),
        'game_level_output': str(output_path),
        'simulation_output': str(Path(SIMULATION_OUTPUT_FILE)),
    }, indent=2))


if __name__ == '__main__':
    main()
