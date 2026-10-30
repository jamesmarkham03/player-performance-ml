"""Which process stats best predict results for youth players?

Hitters: predict OPS from approach stats (quality at-bats, contact rate, batted-ball mix).
Pitchers: predict ERA from strike-throwing stats (strike %, first-pitch strikes, walks).

For each feature set, a Random Forest and an XGBoost model are scored with 5-fold
cross-validation (mean absolute error). A "predict the average" baseline shows whether
the stats carry any signal at all: a model is only useful if it beats the baseline.

Usage:
  python -m src.models                    # synthetic sample data (what's in the repo)
  python -m src.models --data data/real   # your own GameChanger exports (gitignored)
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save charts to files; no window needed
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

from .load import load_stats

ROOT = Path(__file__).resolve().parent.parent

STUDIES = {
    "batting": {
        "file": "batting_stats.csv",
        "target": "OPS",
        "keep": lambda df: df[df["PA"] >= 30],  # too few plate appearances = noisy OPS
        "feature_sets": [
            ["QAB%", "C%"],
            ["QAB%", "C%", "LD%"],
            ["QAB%", "C%", "LD%", "FB%", "GB%"],
        ],
    },
    "pitching": {
        "file": "pitching_stats.csv",
        "target": "ERA",
        "keep": lambda df: df[df["IP"] >= 5],
        "feature_sets": [
            ["<3%", "S%"],
            ["<3%", "S%", "FPS%"],
            ["<3%", "S%", "FPS%", "K/BB", "K/BF"],
            ["<3%", "S%", "FPS%", "BB/INN"],
            ["<3%", "S%", "FPS%", "BB/INN", "LD%", "FB%", "GB%"],
        ],
    },
}

MODELS = {
    "Baseline (average)": lambda: DummyRegressor(strategy="mean"),
    "Random Forest": lambda: RandomForestRegressor(n_estimators=250, random_state=0),
    "XGBoost": lambda: XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8, random_state=0
    ),
}


def cv_mae(model, X, y) -> float:
    # Imputer inside the pipeline, so each fold only learns fill-in values from its own training rows.
    pipeline = make_pipeline(SimpleImputer(strategy="mean"), model)
    folds = KFold(n_splits=5, shuffle=True, random_state=0)
    return -cross_val_score(pipeline, X, y, cv=folds, scoring="neg_mean_absolute_error").mean()


def run_study(name: str, data_dir: Path, out_dir: Path) -> pd.DataFrame:
    study = STUDIES[name]
    df = study["keep"](load_stats(data_dir / study["file"]))
    df = df[df[study["target"]].notna()]
    y = df[study["target"]]

    rows = []
    for features in study["feature_sets"]:
        X = df[features]
        row = {"features": " + ".join(features)}
        for model_name, make in MODELS.items():
            row[model_name] = round(cv_mae(make(), X, y), 3)
        rows.append(row)
    table = pd.DataFrame(rows)

    print(f"\n{name.upper()}: predict {study['target']} for {len(df)} players (5-fold CV, mean absolute error)")
    print(table.to_string(index=False))

    # Which stats the forest leaned on most, using the largest feature set.
    features = study["feature_sets"][-1]
    forest = make_pipeline(SimpleImputer(strategy="mean"), MODELS["Random Forest"]())
    forest.fit(df[features], y)
    importance = pd.Series(forest[-1].feature_importances_, index=features).sort_values(ascending=False)
    print("Feature importance (Random Forest):", ", ".join(f"{k} {v:.2f}" for k, v in importance.items()))

    ax = table.set_index("features").plot.barh(figsize=(12, 1.6 + 0.7 * len(table)))
    ax.set_xlabel(f"Mean absolute error in {study['target']} (lower is better)")
    ax.set_ylabel("")
    ax.set_title(f"Predicting {study['target']} ({name}, n={len(df)})")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False)  # beside the bars, not on top of them
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}_model_comparison.png", dpi=120)
    plt.close()
    table.to_csv(out_dir / f"{name}_model_comparison.csv", index=False)
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/sample", help="folder with batting_stats.csv and pitching_stats.csv")
    args = parser.parse_args()

    data_dir = ROOT / args.data
    # Results from real data go to results/real/, which is gitignored like the data itself.
    out_dir = ROOT / "results" / ("real" if "real" in args.data else "sample")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in STUDIES:
        run_study(name, data_dir, out_dir)
    print(f"\nCharts and tables saved to {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
