Create a new file `predict_playoff_series.py` that trains an XGBoost model to predict NBA game outcomes and runs a Monte Carlo simulation of playoff series. Read `PROJECT_NOTES.md` and `build_rolling_features.py` first to understand the project context and the input data schema.

## Inputs

- **Feature matrix:** `data/features.csv`, produced by `build_rolling_features.py`. Each row is one completed game with team-level rolling features, player-level rolling features, differential features, and a `home_win` target.
- **Tuned hyperparameters:** `xgb_random_search_results.json`, containing a `best_params` dict for XGBoost.

## Outputs

- `data/game_level_win_probabilities.csv` — every playoff game with its predicted `home_win_prob`
- `data/series_simulation_results.csv` — one row per playoff series with predicted series length distribution and competitiveness score
- Print a JSON summary at the end with row counts and output paths

## Train/Test Split

- **Training set:** rows where `season_type == 'Regular Season'`
- **Test set:** rows where `season_type` is in the playoff set defined in `build_rolling_features.py` (`PLAYOFF_TYPES`), but **excluding** play-in games (`'SoFi Play-In Tournament'`, `'Play-In Tournament'`)
- Define a `PLAYOFF_TEST_TYPES` constant at module level for the test set filter
- Train on **all available regular season seasons**; do not subset by year

## Feature Matrix Construction

Build the feature matrix by dropping these columns from `features.csv`:

- **Metadata:** `game_id`, `game_date`, `season_year`, `season_type`, `matchup`, `home`, `away`, `team_id_home`, `team_id_away`, `team_name_home`, `team_name_away`, `min`
- **Outcome / leakage:** `winner`, `pts_home`, `pts_away`, `margin`
- **Target:** `home_win` (this is `y`, not a feature)
- **Series context:** `game_num_in_series`, `home_series_wins`, `away_series_wins` — drop from features but keep in the dataframe so the simulation can group by series
- **Player name strings:** any column matching the regex `^(home|away)_p\d+_PLAYER$`

After dropping, restrict to numeric columns only. If any NaN values remain, fill with 0 and print a warning listing how many NaNs were filled per column (do not silently mask the problem).

## Model

- `XGBClassifier` with `objective='binary:logistic'`, `tree_method='hist'`, `random_state=42`, `n_jobs=-1`, `verbosity=0`
- Load hyperparameters from `xgb_random_search_results.json` and unpack into the constructor
- **Do not** apply sample weighting — train all regular season games with equal weight
- Fit on the full training set (no internal train/val split inside this script — the hyperparameter tuning script handles validation)

## Predictions

- Run `predict_proba` on the test set; take the second column as `home_win_prob`
- Attach `home_win_prob` to a copy of the playoff dataframe (with all original columns preserved) and save to `data/game_level_win_probabilities.csv`

## Monte Carlo Series Simulation

Implement a `simulate_series(predictions: pd.DataFrame) -> pd.DataFrame` function with this exact logic:

1. **Series grouping:** group rows by `season_year` plus the two team abbreviations sorted alphabetically (so home/away orientation doesn't split a series across two groups). Build the series key the same way `add_series_context()` in `build_rolling_features.py` does.

2. **Game ordering within a series:** sort by `game_num_in_series`, then `game_date` as a tiebreaker.

3. **Build a 7-probability vector for each series:**
   - For games actually played, use the model's predicted `home_win_prob`.
   - For unplayed games (when a series ended in fewer than 7 games), fill in using the average `home_win_prob` from played games, flipped by `HOME_COURT_ORDER`. Specifically: `HOME_COURT_ORDER = [True, True, False, False, True, False, True]` — if `True`, use the average; if `False`, use `1 - average`.
   - Every series must have exactly 7 probabilities, regardless of how many games were played.

4. **Simulation:** 10,000 iterations. Set `np.random.seed(42)` once at the top of the function for reproducibility. For each iteration, walk through the 7 probabilities, drawing wins until one team reaches 4. Record series winner and length (4/5/6/7).

5. **Per-series output row:**
   - `series_key`
   - `season_year`, `team_a`, `team_b` (the two teams)
   - `games_4`, `games_5`, `games_6`, `games_7` — fraction of simulations ending in each length
   - `competitiveness_6_or_7` — sum of `games_6` and `games_7`
   - `home_team_win_rate` — fraction of simulations where the team currently labeled "home" in game 1 won
   - `predicted_games` — count of actually-played games used as input
   - `actual_games` — same as `predicted_games` (for clarity in reporting)
   - `actual_winner` — the team that actually won the series (derived from played games — whichever team has 4 wins, or the team leading if the series was incomplete)

## Constants

Define at the top of the file:

```python
FEATURES_FILE = 'data/features.csv'
RESULTS_FILE = 'xgb_random_search_results.json'
GAME_LEVEL_OUTPUT = 'data/game_level_win_probabilities.csv'
SERIES_OUTPUT = 'data/series_simulation_results.csv'
RANDOM_STATE = 42
SIMULATION_ITERATIONS = 10_000
HOME_COURT_ORDER = [True, True, False, False, True, False, True]
```

## Code Quality

- Use type hints
- Functions should be small and single-purpose: `load_features`, `split_train_test`, `build_feature_matrix`, `train_model`, `predict_playoff_games`, `simulate_series`, `main`
- Raise informative errors if input files are missing or required columns are absent
- Print progress messages at each major step

After writing the file, summarize: (1) the final list of feature columns being passed to the model, (2) row counts for train and test, (3) any NaN handling that triggered.
