"""Tiny Markdown helpers shared by the data-quality and metrics reports."""

import pandas as pd


def pct(x: float) -> str:
    return f"{x:.1%}"


def md_table(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    divider = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, divider, *body])
