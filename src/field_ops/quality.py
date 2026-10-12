"""Build the data-quality report: what was fixed, what stayed Unknown and why, how accurate."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from field_ops import evaluate
from field_ops.classify import ClassifyResult, ml_apply, rule_classify
from field_ops.clean import (
    FUZZY_CONFIDENT,
    FUZZY_CUTOFF,
    UNKNOWN,
    CleanResult,
    clean_description,
)
from field_ops.markdown import md_table, pct

# field -> (column holding the cleaned value, column holding how it was resolved)
LABEL_FIELDS = {
    "vendor": ("vendor", "m_vendor"),
    "fault_category": ("fault_category", "m_category"),
    "site": ("site", "m_site"),
    "system_id": ("system_id", "m_system_id"),
    "system_type": ("system_type", "m_system_type"),
    "severity": ("severity", "m_severity"),
    "status": ("status", "m_status"),
    "reported_by": ("reported_by", "m_reporter"),
}
DATE_FIELDS = {"date_opened": "m_date_opened", "date_closed": "m_date_closed"}

UNKNOWN_REASONS = {
    "unknown_blank": "blank in source",
    "unknown_unparseable": "does not match the expected format",
    "unknown_not_a_vendor": "well-formed but not one of the 15 known vendors",
    "unknown_no_match": "no alias or fuzzy match above the cutoff",
    "unknown_ambiguous": "two canonical values matched almost equally well",
}
DATE_REASONS = {
    "unparseable": "date in an unknown format",
    "out_of_range": "date outside the plausible reporting window",
}


@dataclass
class DataQualityReport:
    summary: dict
    ledger: pd.DataFrame
    unknowns: pd.DataFrame
    assumptions: list[str]
    dedupe: dict | None = None
    category: dict | None = None
    fields: pd.DataFrame | None = None
    low_confidence: dict = field(default_factory=dict)
    classification: dict | None = None
    extras: dict = field(default_factory=dict)


def _unknown_table(final: pd.DataFrame) -> pd.DataFrame:
    records = []
    for name, (value_col, method_col) in LABEL_FIELDS.items():
        for method, count in (
            final.loc[final[value_col] == UNKNOWN, method_col].value_counts().items()
        ):
            records.append((name, UNKNOWN_REASONS.get(method, method), int(count)))
    for name, method_col in DATE_FIELDS.items():
        for method, count in final[method_col].value_counts().items():
            if method in DATE_REASONS:
                records.append((name, DATE_REASONS[method], int(count)))
        if name == "date_opened" and (final[method_col] == "blank").any():
            records.append((name, "blank in source", int((final[method_col] == "blank").sum())))
    table = pd.DataFrame(records, columns=["field", "reason", "rows"])
    table["share_of_rows"] = table["rows"] / len(final) if len(final) else 0.0
    return table


def _low_confidence(final: pd.DataFrame) -> dict:
    """Fuzzy matches scoring FUZZY_CUTOFF-FUZZY_CONFIDENT: accepted, but flagged for review."""
    return {
        "rows": int(final["dq_flags"].str.contains("fuzzy_low_confidence").sum()),
        "category": int((final["m_category"] == "fuzzy_low").sum()),
        "site": int((final["m_site"] == "fuzzy_low").sum()),
    }


def _challenge_scores(classified: ClassifyResult) -> dict | None:
    """Descriptions with different wording: a better estimate than the in-dataset numbers."""
    try:
        challenge = evaluate.load_challenge()
    except FileNotFoundError:
        return None
    texts = challenge["text"].map(clean_description)
    tables = {"Keyword rules": evaluate.challenge_by_kind(challenge, texts.map(
        lambda t: rule_classify(t)[0]))}  # fmt: skip
    if classified.ml_model is not None:
        tables["ML (TF-IDF + logistic regression)"] = evaluate.challenge_by_kind(
            challenge, ml_apply(classified.ml_model, texts)
        )
    return tables


def _classification_summary(classified: ClassifyResult, truth: evaluate.Truth | None) -> dict:
    final, survivors, pred = classified.final, classified.survivors, classified.predictions
    blank = survivors["fault_category"] == UNKNOWN
    out = {
        "source_counts": final["category_source"].value_counts().to_dict(),
        "mismatch_flagged": int(final["dq_flags"].str.contains("category_text_mismatch").sum()),
        "unclassified_reasons": pred.loc[blank & (pred["rule_category"] == "Unclassified"),
                                         "rule_reason"].value_counts().to_dict(),
        "ml_info": classified.ml_info,
        "methods": {},
    }  # fmt: skip
    out["challenge"] = _challenge_scores(classified)
    if truth is None:
        return out
    methods = {"Keyword rules": "rule_category"}
    if "ml_category" in pred:
        methods["ML (TF-IDF + logistic regression)"] = "ml_category"
    for name, column in methods.items():
        out["methods"][name] = {
            "by_noise": evaluate.classification_by_noise(survivors, pred[column], truth),
            "fill": evaluate.fill_scores(survivors, pred[column], truth),
            "flags": evaluate.mismatch_flag_scores(survivors, pred[column], truth),
        }
    return out


def build_report(
    result: CleanResult,
    raw: pd.DataFrame,
    truth: evaluate.Truth | None = None,
    classified: ClassifyResult | None = None,
) -> DataQualityReport:
    rows = result.rows
    final = rows[rows["is_survivor"]]
    any_unknown = (final[[v for v, _ in LABEL_FIELDS.values()]] == UNKNOWN).any(axis=1)
    ledger = result.ledger.to_frame()
    if classified is not None:
        ledger = pd.concat([ledger, classified.ledger.to_frame()], ignore_index=True)
    exact_removed = int(ledger.loc[ledger["rule"] == "exact duplicate rows removed", "rows"].sum())
    delivered = classified.final if classified is not None else result.clean
    summary = {
        "raw_rows": len(rows),
        "exact_duplicates_removed": exact_removed,
        "near_duplicates_removed": len(rows) - len(final) - exact_removed,
        "final_rows": len(final),
        "rows_with_any_unknown": int(any_unknown.sum()),
        "rows_with_flags": int((delivered["dq_flags"] != "").sum()),
    }

    report = DataQualityReport(
        summary=summary,
        ledger=ledger,
        unknowns=_unknown_table(final),
        assumptions=result.assumptions,
        low_confidence=_low_confidence(final),
    )
    if truth is not None:
        report.dedupe = evaluate.dedupe_scores(rows, truth)
        report.category = evaluate.category_scores(rows, raw, truth)
        report.fields = evaluate.field_accuracy(rows, truth)
    if classified is not None:
        report.classification = _classification_summary(classified, truth)
    return report


def _accuracy_section(d: dict, c: dict, fields: pd.DataFrame) -> list[str]:
    by_method = c["by_method"].assign(accuracy=c["by_method"]["accuracy"].map(pct))
    return [
        "## Accuracy against the ground truth (synthetic data only)",
        "",
        "### Deduplication (pairwise)",
        "",
        "A pair of raw rows is correct if both belong to the same real ticket.",
        "",
        f"- Raw rows **{d['raw_rows']}** -> kept **{d['kept_rows']}** "
        f"(truth: **{d['true_tickets']}** real tickets)",
        f"- Precision: **{pct(d['precision'])}** "
        f"({d['correct_pairs']} of {d['predicted_duplicate_pairs']} grouped pairs were correct)",
        f"- Recall: **{pct(d['recall'])}** "
        f"({d['correct_pairs']} of {d['true_duplicate_pairs']} true duplicate pairs were found)",
        "",
        "### Category cleaning",
        "",
        "Target: the category the reporter logged (spelling fixed, reporter not second-guessed).",
        "",
        f"- Rows with a label to recover: **{c['rows_with_label']}** "
        f"(plus {c['blank_in_source']} blank in source, left Unknown for the classifier)",
        f"- Precision: **{pct(c['precision'])}** "
        f"({c['correct']} of {c['assigned']} assigned labels correct; "
        f"{c['wrongly_assigned']} wrong)",
        f"- Recall: **{pct(c['recall'])}** "
        f"({c['correct']} of {c['rows_with_label']} recovered; {c['left_unknown']} left Unknown)",
        "",
        md_table(by_method),
        "",
        "### Other fields",
        "",
        md_table(fields.assign(accuracy=fields["accuracy"].map(pct))),
        "",
    ]


def _dash(value, fmt=lambda v: v) -> str:
    return "–" if pd.isna(value) else str(fmt(value))


def _noise_table(df: pd.DataFrame) -> str:
    shown = df.copy()
    for col in ("accuracy", "precision_when_assigned"):
        shown[col] = shown[col].map(lambda v: _dash(v, pct))
    for col in ("rows", "correct", "wrong", "unclassified", "picked_trigger_category"):
        shown[col] = shown[col].map(lambda v: _dash(v, int))
    return md_table(shown)


def _low_confidence_section(low: dict, category: dict | None) -> list[str]:
    out = [
        "## Low-confidence fuzzy matches",
        "",
        f"Fuzzy matches scoring {FUZZY_CUTOFF}-{FUZZY_CONFIDENT} are accepted but flagged "
        "`fuzzy_low_confidence` so a person can review them.",
        "",
        f"- Rows flagged: **{low['rows']}** (category: {low['category']}, site: {low['site']})",
    ]
    if category is not None:
        by_method = category["by_method"].set_index("method")
        for method in ("fuzzy", "fuzzy_low"):
            if method in by_method.index:
                hit = by_method.loc[method]
                out.append(
                    f"- `{method}` category matches correct: "
                    f"{int(hit['correct'])} of {int(hit['rows'])}"
                )
    if low["rows"] == 0:
        out += [
            "",
            "None occur in this dataset: every typo the generator produces scores 85 or higher "
            "(swapped letters in short words are what land in the band). The flag is covered "
            "by unit tests, so it will fire on real data that has them.",
        ]
    return [*out, ""]


def _challenge_section(challenge: dict | None) -> list[str]:
    if not challenge:
        return []
    out = [
        "### Challenge set: descriptions with different wording",
        "",
        "66 short descriptions in ordinary engineer language, written by the same AI assistant "
        "that wrote the rules, so this is not independent evidence. Written in one pass before "
        "being scored and not tuned afterwards (`data/reference/challenge_descriptions.csv`). "
        "`wrong` = confident wrong category; `unclassified` = safe miss on a text that has an "
        "answer.",
        "",
    ]
    for name, table in challenge.items():
        shown = table.assign(accuracy=table["accuracy"].map(pct))
        out += [f"**{name}**", "", md_table(shown), ""]
    return out


def _classification_section(c: dict) -> list[str]:
    src = c["source_counts"]
    out = [
        "## Classification of blank categories and contradiction flags",
        "",
        "- A logged category is **never overwritten**. Text only fills blanks and raises a flag.",
        "- Text with no clear signal stays `Unclassified` instead of being guessed.",
        "",
        f"- Category source: logged **{src.get('logged', 0)}**, filled from text "
        f"**{src.get('filled_from_text', 0)}**, unclassified **{src.get('unclassified', 0)}**",
        f"- Rows flagged `category_text_mismatch` (for review, not changed): "
        f"**{c['mismatch_flagged']}**",
    ]
    if c["unclassified_reasons"]:
        reasons = ", ".join(f"{k}: {v}" for k, v in c["unclassified_reasons"].items())
        out.append(f"- Why blanks stayed Unclassified: {reasons}")
    info = c["ml_info"]
    if "skipped" in info:
        out.append(f"- ML step skipped: {info['skipped']}")
    else:
        out.append(
            f"- ML: trained on {info['n_train']} rows with a logged category (noisy labels, no "
            f"ground truth), C={info['c']} chosen by cross-validated log-loss "
            f"({info['cv_log_loss']}), abstains below {info['threshold']:.0%} confidence. "
            "Predictions for logged rows are out-of-fold."
        )
    out.append("")
    if not c["methods"]:
        return out + _challenge_section(c.get("challenge"))

    out += [
        "Accuracy is shown **per noise type** against the ground truth. The overall number is "
        "dominated by the easy clean rows, so read the other rows first. For vague text the "
        "correct answer is `Unclassified`; guessing counts as wrong.",
        "",
    ]
    for name, m in c["methods"].items():
        out += [f"### {name}", "", _noise_table(m["by_noise"]), ""]

    fill_rows = []
    for name, m in c["methods"].items():
        f = m["fill"]
        fill_rows.append(
            {
                "method": name,
                "blank rows": f["blank_rows"],
                "filled": f["filled"],
                "filled correctly": f["filled_correct"],
                "precision": _dash(f["precision"], pct),
                "left Unclassified": f["unclassified"],
                "of which truly vague": f["unclassified_vague"],
                "vague text guessed": f["vague_guessed"],
            }
        )
    out += ["### Filling the blank categories", "", md_table(pd.DataFrame(fill_rows)), ""]

    flag_rows, false_flags = [], []
    for name, m in c["methods"].items():
        g = m["flags"]
        flag_rows.append(
            {
                "method": name,
                "rows with logged category": g["rows_with_logged_category"],
                "truly contradicting": g["truly_contradicting"],
                "flagged": g["flagged"],
                "correctly flagged": g["correctly_flagged"],
                "precision": _dash(g["precision"], pct),
                "recall": _dash(g["recall"], pct),
            }
        )
        wrong = ", ".join(f"{k}: {v}" for k, v in g["false_flags_by_type"].items()) or "none"
        false_flags.append(f"- {name}: false flags by noise type: {wrong}")
    out += [
        "### Contradiction flags (logged category vs text)",
        "",
        md_table(pd.DataFrame(flag_rows)),
        "",
        *false_flags,
        "",
        "**Read the in-dataset numbers as a regression check, not as expected real-world "
        "accuracy.** The keyword list was written by the same AI assistant that wrote the "
        "generator's phrase templates, so the rules know the vocabulary. The challenge set below "
        "is a better estimate, though it has the same author and is small.",
        "",
        *_challenge_section(c.get("challenge")),
    ]
    return out


def to_markdown(report: DataQualityReport) -> str:
    s = report.summary
    out = ["# Data quality report", ""]
    out += [
        "## Summary", "",
        f"- Raw rows: **{s['raw_rows']}**",
        f"- Exact duplicates removed: **{s['exact_duplicates_removed']}**",
        f"- Near-duplicate tickets merged: **{s['near_duplicates_removed']}**",
        f"- Clean rows delivered: **{s['final_rows']}**",
        "- Rows with at least one Unknown label (after cleaning, before classification): "
        f"**{s['rows_with_any_unknown']}** "
        f"({pct(s['rows_with_any_unknown'] / s['final_rows'])})",
        f"- Rows carrying a data-quality flag: **{s['rows_with_flags']}**",
        f"- Low-confidence fuzzy matches flagged: **{report.low_confidence['rows']}**",
        "",
    ]  # fmt: skip

    out += [
        "## What was fixed",
        "",
        "Rows affected = rows whose value changed (before the duplicates are removed).",
        "",
    ]
    out += [md_table(report.ledger), ""]

    out += ["## Unknown values and why (after cleaning, before classification)", ""]
    if report.unknowns.empty:
        out += ["No Unknown values.", ""]
    else:
        shown = report.unknowns.assign(share_of_rows=report.unknowns["share_of_rows"].map(pct))
        out += [md_table(shown), ""]
    fields_with_zero = [f for f in LABEL_FIELDS if f not in set(report.unknowns["field"])]
    if fields_with_zero:
        out += [f"No Unknown values for: {', '.join(fields_with_zero)}.", ""]

    out += _low_confidence_section(report.low_confidence, report.category)
    if report.classification is not None:
        out += _classification_section(report.classification)

    if report.dedupe and report.category is not None and report.fields is not None:
        out += _accuracy_section(report.dedupe, report.category, report.fields)

    out += ["## Assumptions", ""] + [f"- {a}" for a in report.assumptions] + [""]
    return "\n".join(out)


def write_tables(report: DataQualityReport, directory: Path) -> None:
    """Machine-readable copies of the report, so the Excel step can show the audit trail."""
    directory.mkdir(parents=True, exist_ok=True)
    report.ledger.to_csv(directory / "ledger.csv", index=False)
    report.unknowns.to_csv(directory / "unknowns.csv", index=False)
    classification = report.classification or {}
    summary = {
        "summary": report.summary,
        "low_confidence": report.low_confidence,
        "category_source": classification.get("source_counts", {}),
        "mismatch_flagged": classification.get("mismatch_flagged", 0),
        "assumptions": report.assumptions,
    }
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, default=int))


def write_report(report: DataQualityReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_markdown(report))
