"""The README must stay true: its numbers come from the pipeline and its wording stays honest.

The numbers are the seeded dataset's, so a change to the generator, the cleaning or the metrics
that moves a figure fails here until the README is updated.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()


def section(title: str) -> str:
    match = re.search(rf"^## {re.escape(title)}\n(.*?)(?=^## |\Z)", README, re.S | re.M)
    assert match, f"README has no '## {title}' section"
    return re.sub(r"\s+", " ", match.group(1))  # one line: the checks survive re-wrapping


# --- structure ---------------------------------------------------------------------------------
def test_readme_opens_with_the_synthetic_data_note():
    body = README.split("\n", 1)[1].strip()  # after the title
    assert body.startswith("> **Synthetic data, demo values.**")
    assert body.index("Synthetic data, demo values") < body.index("## The problem")


def test_required_sections_exist_in_order():
    headings = re.findall(r"^## (.+)$", README, re.M)
    wanted = [
        "The problem",
        "Screenshots",
        "Architecture",
        "Limitations and findings",
        "How to run",
    ]
    positions = [headings.index(h) for h in wanted]
    assert positions == sorted(positions)


def test_problem_is_described_generically():
    problem = section("The problem")
    assert "multi-vendor field" in problem
    for term in ("role", "unit", "reserve", "command", "squadron"):
        assert term not in problem.lower()


def test_architecture_has_a_mermaid_diagram_covering_the_stages():
    diagram = re.search(r"```mermaid\n(.*?)```", README, re.S)
    assert diagram and diagram.group(1).lstrip().startswith("flowchart")
    for node in ("generate.py", "clean.py", "classify.py", "metrics.py", "excel_report.py",
                 "dashboard.py", "LibreOffice"):  # fmt: skip
        assert node in diagram.group(1)
    assert diagram.group(1).count("[") == diagram.group(1).count("]")


def test_every_image_in_the_readme_exists():
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", README)
    assert len(images) >= 9
    missing = [i for i in images if not (ROOT / i).exists()]
    assert not missing, missing
    assert sum("excel_" in i for i in images) == 6  # placeholders until captured by hand


def test_capture_checklist_covers_every_excel_image():
    checklist = (ROOT / "docs" / "img" / "CAPTURE.md").read_text()
    for image in re.findall(r"\(docs/img/(excel_[a-z_]+\.png)\)", README):
        assert image in checklist, image


def test_findings_section_states_the_honest_results():
    findings = section("Limitations and findings")
    for needle in ("59%", "42%", "66", "sensitive but not specific", "85%", "modest",
                   "not recovered", "authored by me", "not tuned afterwards",
                   "earlier version of the set scored 58% and 33%"):  # fmt: skip
        assert needle in findings, needle


# --- numbers ----------------------------------------------------------------------------------
def test_headline_numbers_match_the_pipeline(pipeline_run, metrics_run):
    k = metrics_run.kpis
    summary = pipeline_run.report.summary
    table = section("Key numbers (synthetic dataset)")
    for text in (
        f"{summary['raw_rows']:,} raw rows",
        f"{summary['exact_duplicates_removed']} exact duplicates",
        f"{summary['near_duplicates_removed']} re-entered",
        f"**{k['tickets']:,} tickets**",
        f"{k['resolved']:,} resolved",
        f"{k['open']} open",
        f"{k['open_over_30_days']} are older than 30 days",
        f"**{k['median_days']:.1f} days**",
        f"mean {k['mean_days']:.1f} days",
        f"**{k['repeat_rate']:.1%}**",
        f"**{k['sla_breach_rate']:.1%}**",
    ):
        assert text in table, text
    sensitivity = metrics_run.repeat_sensitivity.set_index("window_days")["repeat_rate"]
    assert f"{sensitivity[7]:.1%} at 7 days" in table
    assert f"{sensitivity[30]:.1%} at 30 days" in table


def test_findings_numbers_match_the_pipeline(pipeline_run, metrics_run):
    text = section("What the data shows") + section("Limitations and findings")
    challenge = pipeline_run.report.classification["challenge"]
    rules = challenge["Keyword rules"].set_index("kind").loc["all hand-written rows"]
    ml = (
        challenge["ML (TF-IDF + logistic regression)"]
        .set_index("kind")
        .loc["all hand-written rows"]
    )
    assert f"**{rules['accuracy']:.0%}** ({int(rules['correct'])} of {int(rules['rows'])}" in text
    assert f"**{ml['accuracy']:.0%}** ({int(ml['correct'])} of {int(ml['rows'])}" in text
    assert rules["wrong"] == 1 and f"{int(rules['unclassified'])} safe abstentions" in text
    assert f"four wrong, {int(ml['unclassified'])} abstentions" in text and ml["wrong"] == 4

    source = pipeline_run.report.classification["source_counts"]
    assert f"{source['filled_from_text']} were filled" in text
    assert f"{source['unclassified']} were left `Unclassified`" in text
    assert f"{pipeline_run.report.classification['mismatch_flagged']} logged categories" in text

    sites = metrics_run.aging_by_site.set_index("site").sort_values("over_30_days", ascending=False)
    top, runner_up = sites.iloc[0], sites.iloc[1]
    top_n, next_n = int(top["over_30_days"]), int(runner_up["over_30_days"])
    assert sites.index[0] == "Ridgeview" and "Ridgeview has" in text
    assert f"{top_n} open tickets older than 30 days" in text
    assert f"the next site has {next_n}" in text
    assert f"{top['median_age_days']:.0f} days old" in text

    monthly = metrics_run.monthly.set_index("month")["opened"]
    summer = monthly[["2026-07", "2026-08"]].mean()
    assert f"average about {summer:.0f} new reports a month against {monthly.mean():.0f}" in text
    assert round(summer / monthly.mean() - 1, 1) == 0.3

    vendors = metrics_run.scorecard.set_index("vendor")
    assert vendors.loc[["Vendor F", "Vendor N"], "median_days"].tolist() == [8.0, 8.0]
    assert vendors.at["Vendor C", "median_days"] == vendors.at["Vendor K", "median_days"] == 7.0
    assert f"mean of {vendors.at['Vendor C', 'mean_days']:.1f}" in text

    ambiguous = int(
        (pipeline_run.result.rows["m_date_opened"].eq("dmy_ambiguous")
         | pipeline_run.result.rows["m_date_closed"].eq("dmy_ambiguous")).sum()
    )  # fmt: skip
    assert f"({ambiguous} rows are ambiguous)" in text
    assert f"{metrics_run.repeat.summary['not_evaluable']} faults are left out" in text


def test_lemon_ranks_quoted_in_the_readme_are_current(metrics_run):
    ranked = metrics_run.repeat_by_system.set_index("system_id")["repeat_rate"].rank(
        ascending=False, method="first"
    )
    lemons = ["CTL-03", "FRM-05", "COM-04", "SEN-02", "PWR-03"]
    ranks = [int(ranked[s]) for s in lemons]
    assert sum(r <= 5 for r in ranks) == 2
    assert sorted(ranks)[2:] == [7, 14, 16]
    assert "7th, 14th and 16th" in section("Limitations and findings")
