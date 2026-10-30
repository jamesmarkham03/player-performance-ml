"""Generate the synthetic sample data in data/sample/.

The real stats are youth players (mostly minors), so they never leave this computer.
These files have the same columns and roughly the same spread and relationships as a
real GameChanger export (tuned by hand from summary numbers only), with invented
players. Results on them show how the pipeline works, not real findings.

Usage: python -m src.make_synthetic
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TEAMS = ["14U", "15U", "16U", "17U"]


def players(rng, n):
    return pd.DataFrame({
        "Number": rng.integers(1, 99, n),
        "Last": [f"Player{i:03d}" for i in range(1, n + 1)],
        "First": "Sample",
        "Team": rng.choice(TEAMS, n),
    })


def batting(rng, n=120):
    df = players(rng, n)
    df["PA"] = rng.integers(8, 190, n)
    qab = np.clip(rng.normal(47, 11, n), 10, 85)
    contact = np.clip(rng.normal(76, 11, n) + 0.25 * (qab - 47), 35, 98)
    ld = np.clip(rng.normal(15, 7, n), 0, 40)
    fb = np.clip(rng.normal(34, 11, n), 5, 70)
    gb = np.clip(100 - ld - fb + rng.normal(0, 3, n), 5, 85)
    # OPS rises with contact, approach and line drives, plus plenty of noise (it's baseball).
    ops = 0.74 + 0.009 * (contact - 76) + 0.004 * (qab - 47) + 0.006 * (ld - 15) + rng.normal(0, 0.14, n)
    ops = np.clip(ops, 0.15, 1.6)
    obp = np.clip(ops * 0.47 + rng.normal(0, 0.02, n), 0.1, 0.7)
    df["AVG"] = np.round(np.clip(obp - rng.uniform(0.05, 0.12, n), 0.05, 0.6), 3)
    df["OBP"] = np.round(obp, 3)
    df["SLG"] = np.round(ops - obp, 3)
    df["OPS"] = np.round(df["OBP"] + df["SLG"], 3)
    df["QAB%"], df["C%"] = qab.round(2), contact.round(2)
    df["LD%"], df["FB%"], df["GB%"] = ld.round(2), fb.round(2), gb.round(2)
    return df


def pitching(rng, n=70):
    df = players(rng, n)
    df["IP"] = np.round(rng.uniform(3, 65, n), 1)
    strike = np.clip(rng.normal(58, 6.5, n), 40, 75)
    fps = np.clip(rng.normal(55, 8, n) + 0.5 * (strike - 58), 30, 78)
    under3 = np.clip(rng.normal(47, 8, n) + 0.4 * (strike - 58), 25, 68)
    bb_inn = np.clip(0.74 - 0.045 * (strike - 58) + rng.normal(0, 0.25, n), 0.05, 2.2)
    k_bf = np.clip(rng.normal(0.18, 0.06, n) + 0.004 * (strike - 58), 0.03, 0.45)
    ld = np.clip(rng.normal(16, 8, n), 0, 35)
    fb = np.clip(rng.normal(33, 12, n), 5, 65)
    # ERA falls with strikes and strikeouts and climbs with walks.
    era = 4.3 - 0.12 * (strike - 58) + 1.6 * (bb_inn - 0.74) - 6 * (k_bf - 0.18) + rng.normal(0, 1.2, n)
    df["ERA"] = np.round(np.clip(era, 0.2, 12), 2)
    df["<3%"], df["S%"], df["FPS%"] = under3.round(2), strike.round(2), fps.round(2)
    df["BB/INN"], df["K/BF"] = bb_inn.round(3), k_bf.round(3)
    # About 4.3 batters face a youth pitcher per inning, so strikeouts per inning = K/BF * 4.3.
    df["K/BB"] = np.round(k_bf * 4.3 / np.maximum(bb_inn, 0.05), 2)
    df["LD%"], df["FB%"], df["GB%"] = ld.round(2), fb.round(2), np.clip(100 - ld - fb, 5, 90).round(2)
    return df


def main():
    rng = np.random.default_rng(14)  # fixed seed: same files every time
    out = ROOT / "data" / "sample"
    out.mkdir(parents=True, exist_ok=True)
    batting(rng).to_csv(out / "batting_stats.csv", index=False)
    pitching(rng).to_csv(out / "pitching_stats.csv", index=False)
    print(f"Wrote synthetic batting_stats.csv and pitching_stats.csv to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
