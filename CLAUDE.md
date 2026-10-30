# Player Performance ML

The global `CLAUDE.md` in `OS/` still applies.

## What this is
Models that predict youth hitters' OPS and pitchers' ERA from GameChanger process stats, to see which stats matter most for player development. Public portfolio repo (github.com/jamesmarkham03/player-performance-ml). Grew out of the scripts in `Desktop/Projects/Practice/HPPDataAnalysis` (originals left untouched).

## Rules
- **Real player data never gets committed.** It lives in `data/real/` (gitignored), and so do its results (`results/real/`). Only aggregate numbers (MAE, importances, correlations) may appear in the README. Never player names or individual rows.
- `data/sample/` is synthetic, from `src/make_synthetic.py`. If the real data's shape changes, retune the generator from summary stats only.
- Paths are relative to the repo (`ROOT` in each script). Never hard-code `C:\Users\...`.
- When numbers in the README change, re-run `python -m src.models --data data/real` and copy the aggregate results.

## Commands
- `python -m src.models` (sample) / `python -m src.models --data data/real` (real).
- `python -m src.make_synthetic`: rebuild the sample data.
