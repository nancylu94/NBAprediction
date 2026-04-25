"""
build_rolling_features.py

Combines team_boxscores, player_boxscores, games_index, and games_schedule
into one row per game with pre-game rolling features for both teams.

Key design rule: all rolling stats use shift(1) before rolling so that
the current game's stats are never included (zero temporal leakage).

Outputs:
  data/features.csv   — machine-readable feature matrix for model training
  data/features.xlsx  — same data + data_dictionary tab for human review
"""

import re
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_CSV  = DATA_DIR / "features.csv"
OUTPUT_XLSX = DATA_DIR / "features.xlsx"

# Metrics to roll (must exist in team_boxscores)
TEAM_METRICS = ["off_rating", "def_rating", "net_rating", "pace", "efg_pct", "ts_pct"]

# Metrics to aggregate from top rotation players
PLAYER_METRICS = ["pts", "ast", "reb", "plus_minus"]
TOP_N_PLAYERS = 10

PLAYOFF_TYPES = {
    "Playoffs", "NBA Finals",
    "First Round", "East First Round", "West First Round",
    "Conference Semifinals", "East Second Round", "West Second Round",
    "Conference Finals", "East Conference Finals", "West Conference Finals",
    "SoFi Play-In Tournament", "Play-In Tournament",
}

# NBA game_id third character encodes game type
_GAMEID_TYPE = {"1": "Preseason", "2": "Regular Season", "3": "All-Star", "4": "Playoffs"}

DROP_TYPES = {"Preseason", "All-Star", "Unknown"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _parse_dates(df: pd.DataFrame, col: str = "game_date") -> pd.DataFrame:
    df[col] = pd.to_datetime(df[col], utc=True).dt.tz_localize(None)
    return df


def load_data():
    gi = _parse_dates(pd.read_csv(DATA_DIR / "games_index.csv"))
    gs = _parse_dates(pd.read_csv(DATA_DIR / "games_schedule.csv"))
    tb = _parse_dates(pd.read_csv(DATA_DIR / "team_boxscores.csv"))
    pb = _parse_dates(pd.read_csv(DATA_DIR / "player_boxscores.csv"))

    print(f"  games_index:     {len(gi):>7,} rows")
    print(f"  games_schedule:  {len(gs):>7,} rows")
    print(f"  team_boxscores:  {len(tb):>7,} rows")
    print(f"  player_boxscores:{len(pb):>7,} rows")
    return gi, gs, tb, pb


# ---------------------------------------------------------------------------
# Team-level rolling features
# ---------------------------------------------------------------------------

def _shift_roll(series: pd.Series, window: int) -> pd.Series:
    """Shift by 1 then rolling mean — uses only games strictly before current."""
    return series.shift(1).rolling(window, min_periods=1).mean()


def _shift_expand(series: pd.Series) -> pd.Series:
    """Shift by 1 then expanding mean — season-to-date through game N-1."""
    return series.shift(1).expanding().mean()


def build_team_features(tb: pd.DataFrame) -> pd.DataFrame:
    tb = tb.sort_values(["team_id", "game_date"]).copy()
    tb["win"] = (tb["wl"] == "W").astype(float)

    keep = ["team_id", "team_abbreviation", "game_id", "game_date", "season_year", "is_home"]
    out = tb[keep].copy()

    for metric in TEAM_METRICS:
        if metric not in tb.columns:
            continue
        by_team = tb.groupby("team_id")[metric]
        out[f"{metric}_l5"]  = by_team.transform(lambda x: _shift_roll(x, 5))
        out[f"{metric}_l10"] = by_team.transform(lambda x: _shift_roll(x, 10))

        # Season-to-date resets each season_year
        by_team_season = tb.groupby(["team_id", "season_year"])[metric]
        out[f"{metric}_std"] = by_team_season.transform(_shift_expand)

    # Win % — last 10 games and season-to-date
    by_team_win = tb.groupby("team_id")["win"]
    out["win_pct_l10"] = by_team_win.transform(lambda x: _shift_roll(x, 10))
    out["win_pct_std"]  = tb.groupby(["team_id", "season_year"])["win"].transform(_shift_expand)

    # Rest days: calendar days since the team's previous game
    out["rest_days"] = tb.groupby("team_id")["game_date"].transform(lambda x: x.diff().dt.days)

    return out


# ---------------------------------------------------------------------------
# Player-level rolling features (aggregated to team-game)
# ---------------------------------------------------------------------------

def build_player_features(pb: pd.DataFrame) -> pd.DataFrame:
    """
    For each team-game, compute L5 rolling stats per player (strictly pre-game),
    rank by rolling minutes, keep top 10, and return in wide format matching
    the games 2024-2026.csv layout:
      p1_PLAYER, p1_MIN_l5, p1_PTS_l5, p1_AST_l5, p1_REB_l5, p1_PM_l5, p1_active,
      p2_PLAYER, ..., p10_active
    Players are ordered p1 (most minutes) → p10 (least minutes in rotation).
    """
    pb = pb.sort_values(["player_id", "game_date"]).copy()

    min_col = "min_sec" if "min_sec" in pb.columns else "min"

    pb["roll_min_l5"] = pb.groupby("player_id")[min_col].transform(lambda x: _shift_roll(x, 5))
    pb["roll_pts_l5"] = pb.groupby("player_id")["pts"].transform(lambda x: _shift_roll(x, 5))
    pb["roll_ast_l5"] = pb.groupby("player_id")["ast"].transform(lambda x: _shift_roll(x, 5))
    pb["roll_reb_l5"] = pb.groupby("player_id")["reb"].transform(lambda x: _shift_roll(x, 5))
    pb["roll_pm_l5"]  = pb.groupby("player_id")["plus_minus"].transform(lambda x: _shift_roll(x, 5))

    # Active: 1 if player has positive rolling minutes (played in recent games)
    pb["active"] = (pb["roll_min_l5"].fillna(0) > 0).astype(int)

    # Rank within team-game by rolling minutes descending (NaN → treated as 0)
    pb["_min_fill"] = pb["roll_min_l5"].fillna(0)
    pb["_rank"] = (
        pb.groupby(["team_id", "game_id"])["_min_fill"]
        .rank(ascending=False, method="first")
        .astype(int)
    )

    rotation = pb[pb["_rank"] <= TOP_N_PLAYERS].copy()

    keep = ["team_id", "game_id", "_rank", "player_name",
            "roll_min_l5", "roll_pts_l5", "roll_ast_l5", "roll_reb_l5", "roll_pm_l5", "active"]
    rotation = rotation[keep].set_index(["team_id", "game_id", "_rank"])

    # Unstack rank into columns: (stat, rank) pairs
    wide = rotation.unstack("_rank")

    # Flatten: ("roll_min_l5", 1) -> "p1_MIN_l5"
    stat_rename = {
        "player_name": "PLAYER",
        "roll_min_l5": "MIN_l5",
        "roll_pts_l5": "PTS_l5",
        "roll_ast_l5": "AST_l5",
        "roll_reb_l5": "REB_l5",
        "roll_pm_l5":  "PM_l5",
        "active":      "active",
    }
    wide.columns = [f"p{rank}_{stat_rename[stat]}" for stat, rank in wide.columns]

    # Reorder so each player's stats are grouped: p1_PLAYER, p1_MIN_l5, ..., p2_PLAYER, ...
    stat_order = ["PLAYER", "MIN_l5", "PTS_l5", "AST_l5", "REB_l5", "PM_l5", "active"]
    ordered = [f"p{r}_{s}" for r in range(1, TOP_N_PLAYERS + 1) for s in stat_order
               if f"p{r}_{s}" in wide.columns]
    wide = wide[ordered].reset_index()

    return wide


# ---------------------------------------------------------------------------
# Assemble one row per game
# ---------------------------------------------------------------------------

def assemble_game_matrix(
    gi: pd.DataFrame,
    gs: pd.DataFrame,
    team_feats: pd.DataFrame,
    player_feats: pd.DataFrame,
) -> pd.DataFrame:

    # Infer season_type for all games from game_id prefix (works for all 30 seasons)
    gi["season_type"] = gi["game_id"].astype(str).str.zfill(10).str[2].map(_GAMEID_TYPE).fillna("Unknown")

    # Override with specific round labels from games_schedule (current season only)
    sched = gs[["game_id", "season_type"]].drop_duplicates("game_id").rename(
        columns={"season_type": "season_type_detail"}
    )
    gi = gi.merge(sched, on="game_id", how="left")
    has_detail = gi["season_type_detail"].notna()
    gi.loc[has_detail, "season_type"] = gi.loc[has_detail, "season_type_detail"]
    gi = gi.drop(columns=["season_type_detail"])

    # Drop future / unplayed games (winner is null) and non-game types
    gi = gi[gi["winner"].notna()].copy()
    gi = gi[~gi["season_type"].isin(DROP_TYPES)].copy()

    # Drop odds columns — benchmark only, not model features (per project spec)
    gi = gi.drop(columns=[c for c in ("odds_home", "odds_away") if c in gi.columns])

    # Merge player features onto team features
    tf = team_feats.merge(player_feats, on=["team_id", "game_id"], how="left")

    home_tf = tf[tf["is_home"] == 1].copy()
    away_tf = tf[tf["is_home"] == 0].copy()

    # Columns to prefix (everything except the join keys)
    skip = {"team_id", "team_abbreviation", "game_id", "game_date", "season_year", "is_home"}
    feat_cols = [c for c in tf.columns if c not in skip]

    home_tf = home_tf.rename(columns={c: f"home_{c}" for c in feat_cols})
    away_tf = away_tf.rename(columns={c: f"away_{c}" for c in feat_cols})

    home_keep = ["game_id"] + [f"home_{c}" for c in feat_cols]
    away_keep = ["game_id"] + [f"away_{c}" for c in feat_cols]

    games = (
        gi
        .merge(home_tf[home_keep], on="game_id", how="left")
        .merge(away_tf[away_keep], on="game_id", how="left")
    )

    # Binary target: 1 if home team won
    games["home_win"] = (games["winner"] == games["home"]).astype(int)

    # Differential features (home minus away) for key metrics
    for metric in TEAM_METRICS:
        for suffix in ("_l5", "_l10", "_std"):
            h_col = f"home_{metric}{suffix}"
            a_col = f"away_{metric}{suffix}"
            if h_col in games.columns and a_col in games.columns:
                games[f"diff_{metric}{suffix}"] = games[h_col] - games[a_col]

    if "home_win_pct_l10" in games.columns and "away_win_pct_l10" in games.columns:
        games["diff_win_pct_l10"] = games["home_win_pct_l10"] - games["away_win_pct_l10"]
    if "home_rest_days" in games.columns and "away_rest_days" in games.columns:
        games["diff_rest_days"] = games["home_rest_days"] - games["away_rest_days"]

    return games


# ---------------------------------------------------------------------------
# Playoff series context
# ---------------------------------------------------------------------------

def add_series_context(games: pd.DataFrame) -> pd.DataFrame:
    games = games.copy()
    for col in ("game_num_in_series", "home_series_wins", "away_series_wins"):
        games[col] = np.nan

    mask = games["season_type"].isin(PLAYOFF_TYPES)
    if mask.sum() == 0:
        return games

    playoff = games[mask].sort_values("game_date").copy()

    # Series key: season + the two teams (order-independent)
    playoff["series_key"] = playoff.apply(
        lambda r: f"{r['season_year']}_{min(r['home'], r['away'])}_{max(r['home'], r['away'])}",
        axis=1,
    )

    playoff["game_num_in_series"] = playoff.groupby("series_key").cumcount() + 1

    playoff["home_series_wins"] = 0.0
    playoff["away_series_wins"] = 0.0

    for _, grp in playoff.groupby("series_key"):
        grp = grp.sort_values("game_date")
        hw, aw = 0, 0
        for idx, row in grp.iterrows():
            playoff.at[idx, "home_series_wins"] = hw
            playoff.at[idx, "away_series_wins"] = aw
            if row["home_win"] == 1:
                hw += 1
            else:
                aw += 1

    for col in ("game_num_in_series", "home_series_wins", "away_series_wins"):
        games.loc[playoff.index, col] = playoff[col]

    return games


# ---------------------------------------------------------------------------
# Data dictionary
# ---------------------------------------------------------------------------

_METRIC_LABELS = {
    "off_rating":  "Offensive Rating — points scored per 100 possessions",
    "def_rating":  "Defensive Rating — points allowed per 100 possessions (lower = better defense)",
    "net_rating":  "Net Rating — offensive rating minus defensive rating",
    "pace":        "Pace — estimated possessions per 48 minutes",
    "efg_pct":     "Effective Field Goal % — weights 3-pointers at 1.5x (range 0-1)",
    "ts_pct":      "True Shooting % — efficiency accounting for 3-pointers and free throws (range 0-1)",
    "win_pct":     "Win percentage",
    "rest_days":   "Calendar days since team's previous game (NaN for first game of season)",
    "player_pts":  "Sum of top-8 rotation players' points per game",
    "player_ast":  "Sum of top-8 rotation players' assists per game",
    "player_reb":  "Sum of top-8 rotation players' rebounds per game",
    "player_pm":   "Average plus/minus of top-8 rotation players per game",
    "active_players": "Count of rotation players with a positive rolling-minutes average (injury proxy)",
}

_WINDOW_LABELS = {
    "l5":  "5-game rolling average — strictly pre-game (no leakage)",
    "l10": "10-game rolling average — strictly pre-game (no leakage)",
    "std": "Season-to-date average — resets each season, strictly pre-game",
}

_FIXED_DESCRIPTIONS = {
    "game_id":        ("Metadata", "Unique 10-digit NBA game identifier"),
    "game_date":      ("Metadata", "Date the game was played"),
    "season_year":    ("Metadata", "NBA season label (e.g., '2024-25')"),
    "matchup":        ("Metadata", "Matchup string (e.g., 'TOR vs. CLE')"),
    "home":           ("Metadata", "Home team abbreviation"),
    "away":           ("Metadata", "Away team abbreviation"),
    "team_id_home":   ("Metadata", "NBA internal numeric ID for home team"),
    "team_id_away":   ("Metadata", "NBA internal numeric ID for away team"),
    "team_name_home": ("Metadata", "Full name of home team"),
    "team_name_away": ("Metadata", "Full name of away team"),
    "winner":         ("Metadata", "Abbreviation of winning team"),
    "pts_home":       ("Metadata", "Final points scored by home team"),
    "pts_away":       ("Metadata", "Final points scored by away team"),
    "margin":         ("Metadata", "Home score minus away score at end of game"),
    "min":            ("Metadata", "Game length in minutes (48 regulation; more in OT)"),
    "season_type":    ("Metadata", "Game category: Regular Season / Playoffs / NBA Finals / Play-In etc."),
    "home_win":       ("Target",   "Binary target: 1 = home team won, 0 = away team won"),
    "game_num_in_series":  ("Playoff context", "Which game in the playoff series (1-7). Null for regular season."),
    "home_series_wins":    ("Playoff context", "Home team wins in this series before this game. Null for regular season."),
    "away_series_wins":    ("Playoff context", "Away team wins in this series before this game. Null for regular season."),
}


_PLAYER_STAT_DESC = {
    "PLAYER": "Player name",
    "MIN_l5":  "Minutes per game, 5-game rolling average (pre-game, no leakage)",
    "PTS_l5":  "Points per game, 5-game rolling average (pre-game, no leakage)",
    "AST_l5":  "Assists per game, 5-game rolling average (pre-game, no leakage)",
    "REB_l5":  "Rebounds per game, 5-game rolling average (pre-game, no leakage)",
    "PM_l5":   "Plus/minus per game, 5-game rolling average (pre-game, no leakage)",
    "active":  "1 if player has positive rolling minutes (recent availability proxy), else 0",
}


def build_data_dictionary(columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        # 1. Fixed metadata / target / playoff context columns
        if col in _FIXED_DESCRIPTIONS:
            category, description = _FIXED_DESCRIPTIONS[col]
            rows.append({"column": col, "category": category, "description": description})
            continue

        # 2. Individual player slot columns: home_p{N}_STAT or away_p{N}_STAT
        #    Must be checked before the general home/away parser below.
        m = re.match(r"^(home|away)_(p\d+)_(.+)$", col)
        if m:
            side, slot, stat = m.group(1), m.group(2), m.group(3)
            rank_num   = slot[1:]  # "1", "2", ...
            team_label = "Home" if side == "home" else "Away"
            stat_desc  = _PLAYER_STAT_DESC.get(stat, stat)
            rows.append({
                "column":      col,
                "category":    f"{team_label} team — individual player features",
                "description": f"{team_label} team player {rank_num} (ranked by rolling minutes): {stat_desc}.",
            })
            continue

        # 3. Team rolling / differential columns: {home|away|diff}_{metric}_{window}
        parts = col.split("_", 1)
        side  = parts[0]

        if side in ("home", "away"):
            rest       = parts[1]
            window     = next((w for w in ("_l5", "_l10", "_std") if rest.endswith(w)), None)
            team_label = "Home" if side == "home" else "Away"
            if window:
                metric_key  = rest[: -len(window)]
                metric_desc = _METRIC_LABELS.get(metric_key, metric_key)
                window_desc = _WINDOW_LABELS[window.lstrip("_")]
                category    = f"{team_label} team — rolling features"
                description = f"{team_label} team {metric_desc}. {window_desc}."
            else:
                metric_desc = _METRIC_LABELS.get(rest, rest)
                category    = f"{team_label} team — rolling features"
                description = f"{team_label} team {metric_desc}."

        elif side == "diff":
            rest   = parts[1]
            window = next((w for w in ("_l5", "_l10", "_std") if rest.endswith(w)), None)
            if window:
                metric_key  = rest[: -len(window)]
                metric_desc = _METRIC_LABELS.get(metric_key, metric_key)
                window_desc = _WINDOW_LABELS[window.lstrip("_")]
                category    = "Differential (home minus away)"
                description = f"Home minus away: {metric_desc}. {window_desc}. Positive = home team advantage."
            else:
                metric_desc = _METRIC_LABELS.get(rest, rest)
                category    = "Differential (home minus away)"
                description = f"Home minus away: {metric_desc}. Positive = home team advantage."

        else:
            category    = "Other"
            description = col

        rows.append({"column": col, "category": category, "description": description})

        rows.append({"column": col, "category": category, "description": description})

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    gi, gs, tb, pb = load_data()

    season_types = gs["season_type"].value_counts()
    print(f"\n  season_type breakdown in games_schedule:\n{season_types.to_string()}\n")

    print("Building team rolling features...")
    team_feats = build_team_features(tb)

    print("Building player rolling features...")
    player_feats = build_player_features(pb)

    print("Assembling game-level matrix...")
    games = assemble_game_matrix(gi, gs, team_feats, player_feats)

    print("Adding playoff series context...")
    games = add_series_context(games)

    reg  = (games["season_type"] == "Regular Season").sum()
    post = games["season_type"].isin(PLAYOFF_TYPES).sum()
    print(f"\n  Total rows:       {len(games):,}")
    print(f"  Regular season:   {reg:,}")
    print(f"  Playoff games:    {post:,}")
    print(f"  Columns:          {games.shape[1]}")
    print(f"  Missing home_win: {games['home_win'].isna().sum()}")
    print(f"  home_win rate:    {games['home_win'].mean():.3f}")

    # CSV — used by model pipeline
    games.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved -> {OUTPUT_CSV}")

    # Excel — features tab + data dictionary tab for human review
    data_dict = build_data_dictionary(list(games.columns))
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        games.to_excel(writer, sheet_name="features", index=False)
        data_dict.to_excel(writer, sheet_name="data_dictionary", index=False)
    print(f"Saved -> {OUTPUT_XLSX}")


if __name__ == "__main__":
    main()
