"""Data-quality report and its scoring against the ground-truth files."""

import pytest

from field_ops import evaluate, quality
from field_ops.clean import UNKNOWN


@pytest.fixture(scope="module")
def scored(pipeline_run):
    p = pipeline_run
    return p.raw, p.result, p.truth, p.report


# Floors sit just under the values observed on the seeded dataset: they catch regressions
# rather than prove 100% on unseen data (see the unseen-typo test in test_clean.py).
def test_dedupe_precision_and_recall(scored):
    d = scored[3].dedupe
    assert d["precision"] >= 0.99
    assert d["recall"] >= 0.97


def test_dedupe_leaves_exactly_the_real_tickets(scored):
    d = scored[3].dedupe
    assert d["kept_rows"] == pytest.approx(d["true_tickets"], abs=3)


def test_category_cleaning_precision_and_recall(scored):
    c = scored[3].category
    assert c["precision"] >= 0.99
    assert c["recall"] >= 0.98


def test_vendor_cleaning_has_no_silent_errors(scored):
    fields = scored[3].fields.set_index("field")
    assert fields.loc["vendor", "accuracy"] >= 0.999


def test_dates_are_recovered(scored):
    assert scored[3].fields.set_index("field").loc["date_opened", "accuracy"] >= 0.999


def test_unknown_categories_are_only_blanks_in_source(scored):
    _, result, _, report = scored
    cat = report.unknowns.query("field == 'fault_category'")
    assert list(cat["reason"]) == ["blank in source"]
    final = result.rows[result.rows["is_survivor"]]
    assert cat["rows"].iloc[0] == (final["fault_category"] == UNKNOWN).sum()


def test_summary_adds_up(scored):
    s = scored[3].summary
    assert (
        s["raw_rows"] - s["exact_duplicates_removed"] - s["near_duplicates_removed"]
        == (s["final_rows"])
    )
    assert 0 < s["rows_with_any_unknown"] <= s["final_rows"]


def test_markdown_report_has_the_required_sections(scored):
    text = quality.to_markdown(scored[3])
    for heading in ("## What was fixed", "## Unknown values and why", "### Deduplication",
                    "### Category cleaning", "Precision", "Recall", "## Assumptions"):  # fmt: skip
        assert heading in text


def test_report_without_truth_omits_the_accuracy_section(scored):
    raw, result, _, _ = scored
    text = quality.to_markdown(quality.build_report(result, raw, truth=None))
    assert "Accuracy against the ground truth" not in text
    assert "## Unknown values and why" in text


def test_scorer_notices_when_cleaning_is_wrong(scored):
    """A scorer that cannot fail proves nothing: corrupt 10% of categories and check it drops."""
    raw, result, truth, _ = scored
    broken = result.rows.copy()
    broken.loc[broken.sample(frac=0.1, random_state=1).index, "fault_category"] = "Power"
    scores = evaluate.category_scores(broken, raw, truth)
    assert scores["precision"] < 0.95


def test_low_confidence_count_is_reported(scored):
    report = scored[3]
    assert set(report.low_confidence) == {"rows", "category", "site"}
    text = quality.to_markdown(report)
    assert "## Low-confidence fuzzy matches" in text
    assert "Low-confidence fuzzy matches flagged" in text


def test_report_includes_classification_per_noise_type(pipeline_run):
    text = quality.to_markdown(pipeline_run.report)
    for needle in ("### Keyword rules", "### ML (TF-IDF + logistic regression)", "vague text",
                   "two categories in one sentence", "text contradicts logged category",
                   "Challenge set", "never overwritten"):  # fmt: skip
        assert needle in text
