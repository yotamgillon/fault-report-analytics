"""Near-duplicate detection and the rule for choosing which copy to keep.

Two reports describe the same fault when they share a system, an opening date and a category
and their descriptions are almost identical. Blocking on (system, date) keeps the comparison
cheap and avoids merging genuine repeat faults, which happen on different days.
"""

import re
from itertools import combinations

import pandas as pd
from rapidfuzz import fuzz

DEDUPE_SIMILARITY = 90  # description similarity (0-100) required to call two reports the same
UNKNOWN = "Unknown"

# Fields that count towards "most complete". Unknown / missing values do not count.
COMPLETENESS_FIELDS = [
    "date_opened", "date_closed", "site", "system_type", "system_id", "vendor",
    "fault_category", "severity", "description", "reported_by", "status",
]  # fmt: skip


def description_key(text: str) -> str:
    """Lower-case, punctuation-free form used only for similarity comparison."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text.lower())).strip()


def completeness(row: pd.Series) -> int:
    """Number of key fields that hold a real value."""
    score = 0
    for col in COMPLETENESS_FIELDS:
        value = row[col]
        if pd.isna(value) or value in ("", UNKNOWN):
            continue
        score += 1
    return score


def _id_number(report_id: str) -> int:
    digits = re.sub(r"\D", "", str(report_id))
    return int(digits) if digits else 0


def choose_survivor(cluster: pd.DataFrame):
    """Return the index label of the row to keep from a cluster of duplicates.

    Keep rule, in priority order:
    1. a Closed report that has a close date,
    2. the most complete report,
    3. the earliest report (lowest report ID, since IDs follow entry order).
    """

    def sort_key(label):
        row = cluster.loc[label]
        closed_with_date = row["status"] == "Closed" and pd.notna(row["date_closed"])
        return (0 if closed_with_date else 1, -completeness(row), _id_number(row["report_id"]))

    return min(cluster.index, key=sort_key)


def _categories_compatible(a: str, b: str) -> bool:
    return a == b or UNKNOWN in (a, b)


def near_duplicate_clusters(rows: pd.DataFrame, similarity: int = DEDUPE_SIMILARITY) -> pd.Series:
    """Label every row with a cluster id (the lowest row index in its cluster)."""
    parent = {i: i for i in rows.index}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    candidates = rows[rows["date_opened"].notna() & (rows["system_id"] != UNKNOWN)]
    for _, block in candidates.groupby(["system_id", "date_opened"]):
        if len(block) < 2:
            continue
        for i, j in combinations(block.index, 2):
            if not _categories_compatible(
                block.at[i, "fault_category"], block.at[j, "fault_category"]
            ):
                continue
            if (
                fuzz.ratio(block.at[i, "description_key"], block.at[j, "description_key"])
                >= similarity
            ):
                parent[find(max(i, j))] = find(min(i, j))

    return pd.Series({i: find(i) for i in rows.index})
