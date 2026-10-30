"""Load a GameChanger season-stats export into a clean, numeric DataFrame.

GameChanger stacks one block per team in a single CSV: each block has its own header row,
and blocks are separated by blank rows. Missing stats are written as "-".
"""
from pathlib import Path

import pandas as pd

ID_COLUMNS = ["Number", "Last", "First", "Team"]


def load_stats(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    # Drop the separator rows and the repeated header rows between team blocks.
    df = df[df["Last"].notna() & (df["Last"] != "Last")]
    df = df[~df["Last"].str.contains("Totals|Glossary", case=False)]
    # GameChanger repeats some column names (pandas renames them "GP.1"...). Keep the first.
    df = df[[c for c in df.columns if "." not in c or c.split(".")[0] not in df.columns]]

    for col in df.columns:
        if col not in ID_COLUMNS:
            df[col] = pd.to_numeric(df[col].replace("-", None), errors="coerce")
    return df.reset_index(drop=True)
