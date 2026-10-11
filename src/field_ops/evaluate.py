"""Score the cleaning step against the ground-truth files.

Only this module (and the data-quality report that calls it) reads ``data/truth``. The cleaning
code never does, so the scores are an honest measure and not something it was tuned against.
Scores are only possible for the synthetic data; on real data the report simply omits them.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from field_ops import config

UNKNOWN = "Unknown"


@dataclass
class Truth:
    tickets: pd.DataFrame  # one row per real ticket
    duplicates: pd.DataFrame  # report_id -> original_report_id, kind (exact / near)


def load_truth(truth_dir: Path = config.TRUTH_DIR) -> Truth | None:
    tickets_path, dups_path = truth_dir / "ground_truth.csv", truth_dir / "duplicates.csv"
    if not (tickets_path.exists() and dups_path.exists()):
        return None
    tickets = pd.read_csv(tickets_path, dtype=str, keep_default_na=False)
    return Truth(tickets=tickets, duplicates=pd.read_csv(dups_path, dtype=str))


def _ticket_ids(rows: pd.DataFrame, truth: Truth) -> pd.Series:
    """The real ticket every raw row belongs to (a near-duplicate maps to its original).

    Exact duplicates share their report_id with the row they copy, so only the "near" entries
    are needed to resolve a report_id to its real ticket.
    """
    near = truth.duplicates[truth.duplicates["kind"] == "near"]
    original = dict(zip(near["report_id"], near["original_report_id"], strict=True))
    return rows["report_id"].map(original).fillna(rows["report_id"])


def _expected(rows: pd.DataFrame, truth: Truth) -> pd.DataFrame:
    """Truth values aligned row by row with the cleaned rows."""
    by_id = truth.tickets.set_index("report_id")
    return by_id.loc[_ticket_ids(rows, truth)].reset_index(drop=True)


def _pairs(sizes: pd.Series) -> int:
    return int((sizes * (sizes - 1) // 2).sum())


def dedupe_scores(rows: pd.DataFrame, truth: Truth) -> dict:
    """Pairwise precision/recall: do the rows we grouped together belong to the same ticket?

    Pairwise scoring is used because the kept row can legitimately be either copy, so
    "did we remove the right row" is not well defined, but "did we group the right rows" is.
    """
    frame = pd.DataFrame({"cluster": rows["cluster_id"], "ticket": _ticket_ids(rows, truth)})
    predicted = _pairs(frame.groupby("cluster").size())
    actual = _pairs(frame.groupby("ticket").size())
    both = _pairs(frame.groupby(["cluster", "ticket"]).size())
    return {
        "raw_rows": len(rows),
        "true_tickets": frame["ticket"].nunique(),
        "kept_rows": frame["cluster"].nunique(),
        "true_duplicate_pairs": actual,
        "predicted_duplicate_pairs": predicted,
        "correct_pairs": both,
        "precision": both / predicted if predicted else 1.0,
        "recall": both / actual if actual else 1.0,
    }


def category_scores(rows: pd.DataFrame, raw: pd.DataFrame, truth: Truth) -> dict:
    """Did cleaning recover the category the reporter logged?

    The target is ``logged_category`` (what the reporter chose), not the category implied by the
    text: cleaning fixes spelling, it does not second-guess the reporter (that is Phase 3).
    Blank source labels are excluded from precision and recall because there is nothing to
    recover; they are reported separately.
    """
    expected = _expected(rows, truth)["logged_category"]
    blank = raw["fault_category"].fillna("").str.strip() == ""
    recoverable = ~blank
    assigned = recoverable & (rows["fault_category"] != UNKNOWN)
    correct = assigned & (rows["fault_category"] == expected)

    by_method = (
        pd.DataFrame({"method": rows["m_category"], "correct": correct})[assigned]
        .groupby("method")["correct"]
        .agg(rows="size", correct="sum")
        .reset_index()
    )
    by_method["accuracy"] = by_method["correct"] / by_method["rows"]
    return {
        "rows_with_label": int(recoverable.sum()),
        "assigned": int(assigned.sum()),
        "correct": int(correct.sum()),
        "wrongly_assigned": int((assigned & ~correct).sum()),
        "left_unknown": int((recoverable & ~assigned).sum()),
        "blank_in_source": int(blank.sum()),
        "precision": correct.sum() / assigned.sum() if assigned.sum() else 1.0,
        "recall": correct.sum() / recoverable.sum() if recoverable.sum() else 1.0,
        "by_method": by_method,
    }


def field_accuracy(rows: pd.DataFrame, truth: Truth) -> pd.DataFrame:
    """Share of rows whose cleaned value equals the true value, per field."""
    expected = _expected(rows, truth)
    is_near = rows["report_id"].isin(
        truth.duplicates.loc[truth.duplicates["kind"] == "near", "report_id"]
    )
    opened = rows["date_opened"].dt.strftime("%Y-%m-%d").fillna("")

    checks = {
        "vendor": (rows["vendor"], expected["vendor"], None),
        "site": (rows["site"], expected["site"], None),
        "system_id": (rows["system_id"], expected["system_id"], None),
        "system_type": (rows["system_type"], expected["system_type"], None),
        "severity": (rows["severity"], expected["severity"], None),
        "date_opened": (opened, expected["date_opened"], None),
        # Near-duplicate copies were re-keyed by a different person / with stale status.
        "reported_by": (rows["reported_by"], expected["reported_by"], ~is_near),
        "status": (rows["status"], expected["status"], ~is_near),
    }
    out = []
    for name, (got, want, subset) in checks.items():
        mask = subset if subset is not None else pd.Series(True, index=rows.index)
        unknown = (got[mask] == UNKNOWN) | (got[mask] == "")
        out.append(
            {
                "field": name,
                "rows": int(mask.sum()),
                "correct": int((got[mask] == want[mask]).sum()),
                "accuracy": float((got[mask] == want[mask]).mean()),
                "unknown_or_missing": int(unknown.sum()),
            }
        )
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------------------------
# Classification: scored per noise type so easy rows cannot hide the hard ones
# ---------------------------------------------------------------------------------------------
UNCLASSIFIED = config.UNCLASSIFIED
NOISE_TYPES = {
    "clean": "clean text",
    "vague": "vague text",
    "multi": "two categories in one sentence",
    "mismatch": "text contradicts logged category",
}


def classification_by_noise(survivors: pd.DataFrame, pred: pd.Series, truth: Truth) -> pd.DataFrame:
    """Accuracy of text -> category, separately for each kind of description.

    Target: the category the text describes (truth ``fault_category``). For vague text the
    correct answer is "Unclassified": guessing a category there counts as wrong.
    """
    expected = _expected(survivors, truth)
    pred = pred.reset_index(drop=True)
    rows = []
    for key, label in NOISE_TYPES.items():
        mask = expected["description_type"] == key
        p, want = pred[mask], expected.loc[mask, "fault_category"]
        unclassified = p == UNCLASSIFIED
        correct = unclassified if key == "vague" else (p == want)
        wrong = ~correct & (True if key == "vague" else ~unclassified)
        assigned = int((~unclassified).sum())
        trigger = (
            int((p == expected.loc[mask, "secondary_category"]).sum()) if key == "multi" else 0
        )
        rows.append(
            {
                "noise_type": label,
                "rows": int(mask.sum()),
                "correct": int(correct.sum()),
                "wrong": int(wrong.sum()),
                "unclassified": int(unclassified.sum()),
                "accuracy": float(correct.mean()) if mask.any() else float("nan"),
                "precision_when_assigned": (
                    float((p == want)[~unclassified].mean())
                    if key != "vague" and assigned
                    else float("nan")
                ),
                "picked_trigger_category": trigger,
            }
        )
    table = pd.DataFrame(rows)
    macro = {c: float("nan") for c in table.columns}
    macro.update(
        noise_type="macro average (equal weight per type)", accuracy=table["accuracy"].mean()
    )
    return pd.concat([table, pd.DataFrame([macro])], ignore_index=True)


def fill_scores(survivors: pd.DataFrame, pred: pd.Series, truth: Truth) -> dict:
    """How well blank categories are filled (rows whose cleaned category is Unknown)."""
    blank = (survivors["fault_category"] == UNKNOWN).to_numpy()
    expected = _expected(survivors, truth)[blank]
    p = pred.reset_index(drop=True)[blank]
    vague = expected["description_type"] == "vague"
    filled = p != UNCLASSIFIED
    right = filled & ~vague & (p == expected["fault_category"])
    return {
        "blank_rows": int(blank.sum()),
        "filled": int(filled.sum()),
        "filled_correct": int(right.sum()),
        "filled_wrong": int((filled & ~right).sum()),
        "vague_guessed": int((filled & vague).sum()),
        "unclassified": int((~filled).sum()),
        "unclassified_vague": int((~filled & vague).sum()),
        "precision": float(right.sum() / filled.sum()) if filled.any() else float("nan"),
    }


def mismatch_flag_scores(survivors: pd.DataFrame, pred: pd.Series, truth: Truth) -> dict:
    """Does the contradiction flag fire on the rows where the logged category is wrong?"""
    expected = _expected(survivors, truth)
    has_logged = (survivors["fault_category"] != UNKNOWN).to_numpy()
    p = pred.reset_index(drop=True)
    logged = survivors["fault_category"].reset_index(drop=True)
    flagged = has_logged & (p != UNCLASSIFIED) & (p != logged)
    actual = has_logged & (expected["logged_category"] != expected["fault_category"])
    both = flagged & actual
    false_flags = expected.loc[flagged & ~actual, "description_type"].value_counts()
    return {
        "rows_with_logged_category": int(has_logged.sum()),
        "truly_contradicting": int(actual.sum()),
        "flagged": int(flagged.sum()),
        "correctly_flagged": int(both.sum()),
        "precision": float(both.sum() / flagged.sum()) if flagged.any() else float("nan"),
        "recall": float(both.sum() / actual.sum()) if actual.any() else float("nan"),
        "false_flags_by_type": {NOISE_TYPES[k]: int(v) for k, v in false_flags.items()},
    }


def load_challenge(
    path: Path = config.REFERENCE_DIR / "challenge_descriptions.csv",
) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def challenge_by_kind(challenge: pd.DataFrame, pred: pd.Series) -> pd.DataFrame:
    """Score a hand-written set of descriptions that do not share the generator's wording.

    ``wrong`` means a confident wrong category; ``unclassified`` on a classifiable text is a
    safe miss. For vague text, Unclassified is the correct answer.
    """
    pred = pred.reset_index(drop=True)
    rows = []
    for kind, block in challenge.reset_index(drop=True).groupby("kind", sort=False):
        p, want = pred[block.index], block["expected"]
        correct = p == want
        unclassified = p == UNCLASSIFIED
        rows.append(
            {
                "kind": kind,
                "rows": len(block),
                "correct": int(correct.sum()),
                "wrong": int((~correct & ~unclassified).sum()),
                "unclassified": int((unclassified & ~correct).sum()),
                "accuracy": float(correct.mean()),
            }
        )
    table = pd.DataFrame(rows)
    total = {
        "kind": "all hand-written rows",
        "rows": int(table["rows"].sum()),
        "correct": int(table["correct"].sum()),
        "wrong": int(table["wrong"].sum()),
        "unclassified": int(table["unclassified"].sum()),
        "accuracy": float(table["correct"].sum() / table["rows"].sum()),
    }
    return pd.concat([table, pd.DataFrame([total])], ignore_index=True)


# ---------------------------------------------------------------------------------------------
# Planted patterns: do the metrics surface what the generator hid in the data?
# ---------------------------------------------------------------------------------------------
def load_scenario(truth_dir: Path = config.TRUTH_DIR) -> dict | None:
    path = truth_dir / "generation_summary.json"
    return json.loads(path.read_text())["scenario"] if path.exists() else None


def planted_pattern_checks(m, scenario: dict) -> pd.DataFrame:
    """Test the pre-registered criteria for each planted pattern against the metrics."""
    from field_ops import metrics  # local import: metrics does not depend on evaluate

    rows = []

    vendors = m.mttr["vendor"]
    adequate = vendors[~vendors["low_sample"]].set_index("vendor")
    slow = [name for name, _ in scenario["slow_vendors"]]
    middle = adequate["median_days"].median()
    ok = all(
        adequate.at[v, "rank_slowest"] <= 4 and adequate.at[v, "median_days"] > middle for v in slow
    )
    detail = ", ".join(
        f"{v.removeprefix('Vendor ')}: rank {int(adequate.at[v, 'rank_slowest'])} "
        f"(median {adequate.at[v, 'median_days']:.1f}d)"
        for v in slow
    )
    rows.append(
        ("slow vendors", f"{', '.join(slow)} in top 4 by median MTTR and above the median vendor",
         ok, f"{detail}; median vendor {middle:.1f}d")
    )  # fmt: skip

    gap = (adequate["mean_days"] - adequate["median_days"]).sort_values(ascending=False)
    tail = scenario["long_tail_vendor"]
    rows.append(
        ("long-tail vendor", f"{tail} has the largest mean-minus-median gap", gap.index[0] == tail,
         f"largest gap: {gap.index[0]} (+{gap.iloc[0]:.1f}d); {tail} +{gap[tail]:.1f}d")
    )  # fmt: skip

    site = m.aging_by_site.sort_values("over_30_days", ascending=False)
    backlog = scenario["backlog_site"]
    unique_top = site.iloc[0]["over_30_days"] > site.iloc[1]["over_30_days"]
    rows.append(
        ("backlog site", f"{backlog} has the most open tickets older than 30 days",
         bool(site.iloc[0]["site"] == backlog and unique_top),
         ", ".join(f"{r.site}: {r.over_30_days}" for r in site.head(3).itertuples()))
    )  # fmt: skip

    by_system = metrics.repeat_by(m.prepared, m.repeat, "system_id")
    by_system = by_system[~by_system["low_sample"]].sort_values(
        ["repeat_rate", "repeats"], ascending=False
    )
    lemons = set(scenario["lemon_systems"])
    top = by_system.head(len(lemons))
    ranks = {s: int(i + 1) for i, s in enumerate(by_system["system_id"]) if s in lemons}
    rows.append(
        ("lemon systems", f"the {len(lemons)} lemon systems = top {len(lemons)} by repeat rate",
         set(top["system_id"]) == lemons,
         f"top: {', '.join(f'{r.system_id} {r.repeat_rate:.0%}' for r in top.itertuples())}; "
         f"lemon ranks: {ranks}")
    )  # fmt: skip

    return pd.DataFrame(rows, columns=["pattern", "criterion", "found", "detail"]).assign(
        found=lambda d: d["found"].map({True: "yes", False: "NO"})
    )
