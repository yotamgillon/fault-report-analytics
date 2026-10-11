"""End to end through the real pipeline functions and CSV files."""

import pandas as pd

from field_ops import metrics
from field_ops.mess import RAW_COLUMNS
from field_ops.pipeline import run_clean, run_metrics


def _raw(n: int, **overrides) -> dict:
    base = {
        "report_id": f"FR-{10000 + n}", "date_opened": "2026-03-01", "date_closed": "2026-03-05",
        "site": "Lakeside", "system_type": "Power Unit", "system_id": f"PWR-0{n}",
        "vendor": "Vendor A", "fault_category": "Power", "severity": "High",
        "description": "Battery pack drops voltage under load.", "reported_by": "Ruth Adler",
        "status": "Closed",
    }  # fmt: skip
    return {**base, **overrides}


def test_fuzzy_low_confidence_flows_from_raw_csv_to_flag_and_report(tmp_path):
    raw = pd.DataFrame(
        [
            _raw(1, fault_category="Poewr"),  # category typo scoring exactly 80
            _raw(2, site="Westbk"),  # site typo scoring exactly 80
            _raw(3, fault_category="Comunications", description="Radio loses signal at range."),
            _raw(4),  # clean row
        ],
        columns=RAW_COLUMNS,
    )
    raw_path = tmp_path / "raw.csv"
    raw.to_csv(raw_path, index=False)

    report = run_clean(raw_path, tmp_path / "out", tmp_path / "no_truth")

    clean = pd.read_csv(tmp_path / "out" / "clean_reports.csv", dtype=str, keep_default_na=False)
    by_id = clean.set_index("report_id")
    assert "fuzzy_low_confidence" in by_id.at["FR-10001", "dq_flags"]
    assert by_id.at["FR-10001", "fault_category"] == "Power"  # accepted, but marked for review
    assert "fuzzy_low_confidence" in by_id.at["FR-10002", "dq_flags"]
    assert by_id.at["FR-10002", "site"] == "Westbrook"
    assert "fuzzy_low_confidence" not in by_id.at["FR-10003", "dq_flags"]  # confident fuzzy match
    assert "fuzzy_low_confidence" not in by_id.at["FR-10004", "dq_flags"]

    assert report.low_confidence == {"rows": 2, "category": 1, "site": 1}
    text = (tmp_path / "out" / "data_quality.md").read_text()
    assert "Low-confidence fuzzy matches flagged: **2**" in text
    assert "Rows flagged: **2** (category: 1, site: 1)" in text


def test_metrics_step_runs_from_the_clean_csv(tmp_path):
    raw = pd.DataFrame([_raw(i, system_id=f"PWR-0{i}") for i in range(1, 5)], columns=RAW_COLUMNS)
    raw.to_csv(tmp_path / "raw.csv", index=False)
    run_clean(tmp_path / "raw.csv", tmp_path / "out", tmp_path / "no_truth")

    m = run_metrics(tmp_path / "out" / "clean_reports.csv", tmp_path / "out", tmp_path / "no_truth")

    assert m.kpis["tickets"] == 4 and m.kpis["resolved"] == 4 and m.kpis["median_days"] == 4.0
    assert (tmp_path / "out" / "metrics_report.md").exists()
    assert (tmp_path / "out" / "metrics" / "mttr_by_vendor.csv").exists()
    assert isinstance(m, metrics.Metrics)
