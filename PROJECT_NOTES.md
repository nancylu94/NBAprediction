# NBA Playoff Series Prediction — Project Notes

## Business Case
Broadcasters (ESPN, ABC, TNT) allocate ad inventory between the **upfront market** (committed months in advance) and the **scatter market** (sold closer to airtime at 40–80% premium). They currently make series competitiveness judgments on gut instinct. The stakes are large — the difference between a sweep and a 7-game NBA Finals is worth **$100M+ in ad revenue to a single broadcaster**. Our model gives that judgment a quantitative foundation.

Key figures for the report:
- 2024-25 NBA postseason generated $845M in total media buys
- TNT: $486M playoff revenue, ABC: $142M, ESPN: $217M
- ABC average ad rate: $157,200 per spot in playoffs
- 2025 Thunder-Pacers 7-game Finals: $288M vs prior year's 5-game series: $184.7M — $103M difference from 2 extra games
- A 4-game sweep generates ~$150M; a 7-game series ~$265M for a single broadcaster

---

## What We're Building
A two-stage prediction pipeline:

**Stage 1 — Game-level model:** Given two teams entering a specific playoff game, predict the probability the home team wins (binary classification).

**Stage 2 — Series simulation:** Feed game-level win probabilities into a Monte Carlo simulation (10,000 iterations) using the correct 2-2-1-1-1 home court schedule to output a series length distribution and a **competitiveness score** (probability the series goes 6 or 7 games).

### 2-2-1-1-1 Home Court Schedule
| Game | Host |
|---|---|
| G1 | Higher seed (home) |
| G2 | Higher seed (home) |
| G3 | Lower seed (home) |
| G4 | Lower seed (home) |
| G5 | Higher seed (home) |
| G6 | Lower seed (home) |
| G7 | Higher seed (home) |

### Monte Carlo Simulation Logic
For each of 10,000 iterations:
1. For each game, draw a random number against the home-adjusted win probability from Stage 1
2. Continue until one team reaches 4 wins
3. Record series winner and length (4/5/6/7 games)

Output: probability distribution over series lengths + competitiveness score + predicted winner.

---

## Data

### Available Files

| File | Description | Use |
|---|---|---|
| `team_boxscores.csv` | 30 seasons (1996-2026), ~38K games, per-team per-game stats with pre-computed advanced metrics (off/def rating, pace, eFG%, TS%, net rating) | **Primary feature source** |
| `player_boxscores.csv` | One row per player per game, traditional + advanced stats | **For player-level rolling features** |
| `games_index.csv` | Lightweight master table — one row per game with teams, scores, margin, date, odds | **Join spine** |
| `games_schedule.csv` | Game metadata with `season_type` field (Regular Season / Playoffs / NBA Finals) | **Train/test split source** |
| `game_odds.csv` | 513K rows of betting line movement (open + close + intraday timestamps) | **Vegas benchmark only — NOT a feature** |
| `play_by_play.csv` | Possession-level events | **Excluded — out of scope for this project** |

### Train/Test Split
- **Training:** All Regular Season games across all available seasons (~36K games)
- **Test:** Playoff games from most recent 1–2 completed seasons
- Use `season_type` field in `games_schedule.csv` to filter cleanly

### Feature Construction — Rolling Stats (Option 2)
With `team_boxscores.csv` and `player_boxscores.csv`, construct truly pre-game features for every row:

**Team-level rolling features (per game):**
- Last-5-game rolling averages: offensive rating, defensive rating, pace, eFG%, TS%
- Last-10-game rolling averages: same metrics
- Season-to-date averages through game N: same metrics
- Win % in last 10 games (momentum)

**Player-level rolling features (per game):**
- Last-5-game rolling averages for top 8–10 rotation players: minutes, PTS, AST, REB, +/-
- Active flag — did this player play in any of the last 5 games?

**Critical implementation rule:** All rolling stats computed using ONLY games where `game_date < target_game_date`. Build a strict temporal cutoff into the rolling-stat function and unit-test it.

### Other Features
- `is_home` — binary
- `rest_days` — derived from team's previous game date
- Series context (playoffs only): game number in series, series record at time of game
- Round (First Round, Conference Semis, Conference Finals, Finals)

### Confirmed Drops
- `attendance` — consequence of game quality, near-zero variance in playoffs
- `officials` — not known before tip-off
- `play_by_play.csv` — wrong granularity for this problem

---

## Vegas Odds — Benchmark Only

**`game_odds.csv` is used exclusively as a benchmark, not as a feature.**

Reason: including closing odds as a model feature would essentially be predicting wins using the market's prediction of wins. The model would look great but the contribution would be questionable.

**How to use it:**
1. For each playoff test game, extract the closing moneyline → convert to implied probability
2. Calculate log-loss and accuracy of "Vegas closing line" as a standalone predictor
3. Compare your model's log-loss and accuracy against this benchmark
4. Report whether your model — using only public box score data — meets, beats, or trails Vegas

**Report framing:** *"We benchmark our model against the closing Vegas moneyline, which represents the market's aggregated information including injuries, lineups, and public sentiment. Beating Vegas is a high bar; matching it using only box score data would be a meaningful result."*

---

## Model

### Stage 1 — Win Prediction
**Baseline:** Logistic regression — interpretable, calibrated probabilities, fast to run

**Primary:** XGBoost classifier
- `objective='binary:logistic'`
- Tune via `xgb_randomized_search.py` (adapted from teammate's MLB attendance model)
- Scoring metric: `neg_log_loss` (calibrated probabilities needed for simulation)

**Vegas benchmark:** Closing moneyline → implied probability, no model required

**Evaluation metrics:** Log-loss (primary), accuracy (secondary)

### Distribution Shift Mitigation
Training on regular season, testing on playoffs is a known domain transfer problem. Mitigations:
1. **Sample weighting** — upweight late-season games between playoff-bound teams in training. XGBoost accepts `sample_weight` natively
2. **Late-season validation** — hold out last 2 weeks of regular season as internal validation before evaluating on playoffs

**Report framing:** *"We acknowledge that regular season and playoff games differ systematically in intensity and lineup management. To mitigate this, we applied higher sample weights to late-season games between playoff-caliber teams, and validated on late-season games prior to testing on playoff data. Distribution shift remains a limitation and future work could apply formal domain adaptation techniques."*

---

## Codebase (adapted from teammate's MLB attendance model)

| File | Purpose | Changes needed for NBA |
|---|---|---|
| `xgb_randomized_search.py` | Hyperparameter tuning | Change target to `home_win`, swap to `XGBClassifier`, change scoring to `neg_log_loss` |
| `predict_missing_attendance.py` | Train model + generate predictions | Same swaps as above, rename outputs |
| `make_waterfall_chart.py` | SHAP waterfall for single game | Minimal changes, update labels |
| `make_more_charts.py` | Global feature importance + second waterfall | Minimal changes |

**New scripts to write:**
- `build_rolling_features.py` — for each game in the dataset, compute rolling team and player features using only games before that date
- `simulate_series.py` — Monte Carlo simulation taking game-level probabilities and returning series length distribution
- `vegas_benchmark.py` — converts closing moneylines to implied probabilities, computes log-loss and accuracy

**Execution order:**
1. Run `build_rolling_features.py` → produces feature-enriched dataset
2. Run `xgb_randomized_search.py` → produces `xgb_random_search_results.json`
3. Run adapted `predict_missing_attendance.py` → produces game-level win probabilities
4. Run `vegas_benchmark.py` → produces Vegas baseline metrics
5. Feed probabilities into `simulate_series.py` → produces competitiveness table
6. Run chart scripts for SHAP explainability

**Note:** Neither script saves a trained model file — the model is retrained from scratch each run using the best params from the JSON. This is intentional.

---

## Key Risks and Fallbacks

| Risk | Fallback |
|---|---|
| Rolling feature construction has temporal leakage bug | Unit test with hand-checked example before training; visualize feature distribution on early-season vs. late-season games |
| `season_type` field doesn't cleanly separate playoffs in old seasons | Use playoff round labels in matchup metadata as backup |
| XGBoost doesn't beat logistic regression | Frame as finding about feature engineering vs. model complexity |
| Model trails Vegas significantly | This is expected — frame as "we recover X% of the market's information using only box score data" |
| Simulation doesn't integrate in time | Use simple home team win probability as competitiveness score |
| Small playoff test set | Report confidence intervals on metrics, not just point estimates |

---

## Report Structure
1. Business problem + market quantification (~0.5 pages)
2. Data and features — sources, rolling feature construction, leakage prevention (~1 page)
3. Methodology — classification model, temporal split rationale, distribution shift mitigation (~1 page)
4. Results — logistic regression vs. XGBoost vs. Vegas benchmark, log-loss and accuracy, SHAP findings (~1 page)
5. Business output — competitiveness ranking table, simulation validation (~0.5 pages)
6. Limitations + future work (~0.5 pages)

---

## Key Analytical Questions to Answer
1. Does the model meaningfully beat a naive baseline (always predict higher seed wins)?
2. How does the model perform relative to the Vegas closing line benchmark?
3. Does home court advantage matter more or less in playoffs than the model learns from regular season?
4. Which features contribute most to predicted win probability — does it match basketball intuition?
5. For the most recent playoffs: did our competitiveness scores correctly rank which series went long?
6. Pick one series the model got badly wrong — why?

---

## What This Project Is Not
Explicitly out of scope (legitimate future work items):
- Neural networks
- Using Vegas odds as a model feature (used as benchmark only)
- Possession-level / play-by-play modeling
- In-game win probability updates (different problem)
- Formal domain adaptation
- Player-level injury API integration
