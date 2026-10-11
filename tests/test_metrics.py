"""Metrics on a tiny hand-computed fixture, then on the seeded dataset's planted patterns.

Fixture, reporting date 2026-03-31, repeat window 14 days. Every expected number was worked out
by hand. See _ROWS below; in words:
- CTL-01 Power: R01 (4d), R02 (10d, opened 15d after R01 closed: outside 14d, a repeat only at
  30d), R03 (2d, 39d after R02 closed: no), R05 (open, 8d after R03 closed: repeat).
  R04 is the only Sensor fault.
- COM-01 Communications: R06 (8d), R07 (20d, opened before R06 closed), R08 (open, opened before
  R07 closed), R09 (open, previous R08 still open): all three opened while the previous fault was
  open, so none is a repeat.
- PWR-01: R10 Unclassified and R11 flagged category_text_mismatch are excluded from repeats;
  R12 is therefore the first eligible Power fault (not a repeat of R11).
- SEN-01: R13 is Closed with no close date, so it is out of MTTR and R14 cannot be judged.
- FRM-01: six open Unclassified tickets aged 7, 8, 30, 31, 90 and 91 days (bucket edges).
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from field_ops import evaluate, metrics

AS_OF = date(2026, 3, 31)
SYSTEMS = {
    "CTL-01": "Control Unit", "COM-01": "Comms Module", "PWR-01": "Power Unit",
    "SEN-01": "Sensor Module", "FRM-01": "Mechanical Frame",
}  # fmt: skip

UNCLASSIFIED_FLAG = "category_unclassified"
_ROWS = [
    # id, system, category, opened, closed, status, flags
    ("R01", "CTL-01", "Power", "2026-01-01", "2026-01-05", "Closed", ""),
    ("R02", "CTL-01", "Power", "2026-01-20", "2026-01-30", "Closed", ""),
    ("R03", "CTL-01", "Power", "2026-03-10", "2026-03-12", "Closed", ""),
    ("R04", "CTL-01", "Sensor", "2026-01-02", "2026-01-04", "Closed", ""),
    ("R05", "CTL-01", "Power", "2026-03-20", "", "Open", ""),
    ("R06", "COM-01", "Communications", "2026-02-01", "2026-02-09", "Closed", ""),
    ("R07", "COM-01", "Communications", "2026-02-05", "2026-02-25", "Closed", ""),
    ("R08", "COM-01", "Communications", "2026-02-20", "", "In Progress", ""),
    ("R09", "COM-01", "Communications", "2026-03-25", "", "Open", ""),
    ("R10", "PWR-01", "Unclassified", "2026-02-01", "2026-02-11", "Closed", UNCLASSIFIED_FLAG),
    ("R11", "PWR-01", "Power", "2026-02-15", "2026-02-16", "Closed", "category_text_mismatch"),
    ("R12", "PWR-01", "Power", "2026-03-01", "2026-03-04", "Closed", ""),
    ("R13", "SEN-01", "Sensor", "2026-01-10", "", "Closed", "closed_no_close_date"),
    ("R14", "SEN-01", "Sensor", "2026-01-25", "2026-02-01", "Closed", ""),
    ("R16", "FRM-01", "Unclassified", "2026-03-24", "", "Open", "category_unclassified"),
    ("R17", "FRM-01", "Unclassified", "2026-03-23", "", "In Progress", "category_unclassified"),
    ("R18", "FRM-01", "Unclassified", "2026-03-01", "", "Open", "category_unclassified"),
    ("R19", "FRM-01", "Unclassified", "2026-02-28", "", "Open", "category_unclassified"),
    ("R21", "FRM-01", "Unclassified", "2025-12-31", "", "Open", "category_unclassified"),
    ("R22", "FRM-01", "Unclassified", "2025-12-30", "", "Open", "category_unclassified"),
]  # fmt: skip
_VENDOR_SITE = {"CTL-01": ("Vendor A", "North"), "FRM-01": ("Vendor A", "North")}


def _fixture() -> pd.DataFrame:
    rows = []
    for rid, system, category, opened, closed, status, flags in _ROWS:
        vendor, site = _VENDOR_SITE.get(system, ("Vendor B", "South"))
        rows.append(
            {
                "report_id": rid, "date_opened": opened, "date_closed": closed, "site": site,
                "system_type": SYSTEMS[system], "system_id": system, "vendor": vendor,
                "fault_category": category, "status": status, "dq_flags": flags,
            }
        )  # fmt: skip
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def prep():
    return metrics.prepare(_fixture(), AS_OF)


@pytest.fixture(scope="module")
def repeat(prep):
    return metrics.repeat_faults(prep)


# --- resolved / open ---------------------------------------------------------------------------
def test_resolved_excludes_open_and_closed_without_close_date(prep):
    resolved = set(prep.loc[prep["is_resolved"], "report_id"])
    assert resolved == {"R01", "R02", "R03", "R04", "R06", "R07", "R10", "R11", "R12", "R14"}
    assert "R13" not in resolved  # status Closed, but the close date is missing
    assert prep.set_index("report_id").at["R05", "days_to_resolve"] != prep.at[0, "days_to_resolve"]


def test_flag_match_is_exact_not_a_substring():
    flags = pd.Series(["a;category_text_mismatch_x;b", "x;closed_no_close_date"])
    assert metrics.has_flag(flags, "closed_no_close_date").tolist() == [False, True]
    assert metrics.has_flag(flags, "category_text_mismatch").tolist() == [False, False]


# --- MTTR --------------------------------------------------------------------------------------
def test_mttr_by_vendor_median_first_and_n(prep):
    table = metrics.mttr_by(prep, "vendor", min_sample=5).set_index("vendor")
    a, b = table.loc["Vendor A"], table.loc["Vendor B"]
    # A resolved: 4, 10, 2, 2 days -> median 3, mean 4.5
    assert (a["tickets"], a["n"], a["median_days"], a["mean_days"]) == (11, 4, 3.0, 4.5)
    # B resolved: 8, 20, 10, 1, 3, 7 days -> median 7.5, mean 49/6
    assert (b["tickets"], b["n"], b["median_days"]) == (9, 6, 7.5)
    assert b["mean_days"] == pytest.approx(49 / 6)
    assert list(metrics.mttr_by(prep, "vendor", 5).columns[:5]) == [
        "vendor", "tickets", "n", "median_days", "mean_days",
    ]  # fmt: skip


def test_low_sample_groups_are_marked_and_never_ranked(prep):
    table = metrics.mttr_by(prep, "vendor", min_sample=5).set_index("vendor")
    assert table.at["Vendor A", "low_sample"] and not table.at["Vendor B", "low_sample"]
    assert np.isnan(table.at["Vendor A", "rank_slowest"])
    assert table.at["Vendor B", "rank_slowest"] == 1
    default = metrics.mttr_by(prep, "vendor")  # default threshold is 10
    assert default["low_sample"].all()


def test_small_vendor_with_high_median_does_not_look_like_the_worst():
    def tickets(vendor, n, days):
        return [
            {"report_id": f"{vendor}{i}", "date_opened": "2026-01-01",
             "date_closed": f"2026-01-{1 + days:02d}", "vendor": vendor, "status": "Closed",
             "dq_flags": "", "site": "S", "system_id": "X", "system_type": "T",
             "fault_category": "Power"}
            for i in range(n)
        ]  # fmt: skip

    prep = metrics.prepare(pd.DataFrame(tickets("Big", 12, 3) + tickets("Tiny", 8, 20)), AS_OF)
    table = metrics.mttr_by(prep, "vendor")
    assert list(table["vendor"]) == ["Big", "Tiny"]  # adequate sample first
    assert table.set_index("vendor").at["Big", "rank_slowest"] == 1
    assert table.set_index("vendor").at["Tiny", "low_sample"]


def test_mttr_by_system_and_site(prep):
    systems = metrics.mttr_by(prep, "system_id", 1).set_index("system_id")
    assert systems.at["COM-01", "median_days"] == 14.0 and systems.at["COM-01", "n"] == 2
    assert systems.at["PWR-01", "mean_days"] == pytest.approx(14 / 3)
    assert systems.at["SEN-01", "n"] == 1 and systems.at["SEN-01", "median_days"] == 7.0
    assert systems.at["FRM-01", "n"] == 0 and np.isnan(systems.at["FRM-01", "median_days"])
    sites = metrics.mttr_by(prep, "site", 5).set_index("site")
    assert sites.at["North", "median_days"] == 3.0 and sites.at["South", "median_days"] == 7.5


def test_overall_kpis():
    m = metrics.compute_all(_fixture(), AS_OF)
    k = m.kpis
    # resolved days 4,10,2,2,8,20,10,1,3,7 -> sorted 1,2,2,3,4,7,8,10,10,20
    assert (k["tickets"], k["resolved"], k["open"], k["open_over_30_days"]) == (20, 10, 9, 4)
    assert k["median_days"] == 5.5 and k["mean_days"] == pytest.approx(6.7)


# --- repeat faults -----------------------------------------------------------------------------
def test_repeat_rate_after_close_only(repeat):
    s = repeat.summary
    assert s["window_days"] == 14
    assert (s["denominator"], s["repeats"], s["opened_while_previous_open"]) == (11, 1, 3)
    assert s["repeat_rate"] == pytest.approx(1 / 11)
    repeats = set(repeat.rows.index[repeat.rows["is_repeat"]])
    assert {_ROWS[i][0] for i in repeats} == {"R05"}  # R02 is 15 days after R01 closed


def test_window_is_configurable():
    wide = metrics.compute_all(_fixture(), AS_OF, repeat_window=30)
    assert (wide.repeat.summary["repeats"], wide.repeat.summary["window_days"]) == (2, 30)
    assert wide.repeat.rows["is_repeat"].sum() == 2  # R02 (15d) and R05 (8d)
    narrow = metrics.compute_all(_fixture(), AS_OF, repeat_window=7)
    assert narrow.repeat.summary["repeats"] == 0  # R05 opened 8 days after R03 closed


def _pair(prev_closed: str, next_opened: str, prev_status: str = "Closed") -> pd.DataFrame:
    """Two Power faults on one system: the first opens 2026-01-01 and closes on ``prev_closed``."""
    rows = []
    for rid, opened, closed, status in [
        ("A1", "2026-01-01", prev_closed, prev_status),
        ("A2", next_opened, "", "Open"),
    ]:
        rows.append(
            {"report_id": rid, "date_opened": opened, "date_closed": closed, "site": "S",
             "system_type": "T", "system_id": "X-01", "vendor": "V", "fault_category": "Power",
             "status": status, "dq_flags": ""}
        )  # fmt: skip
    return metrics.prepare(pd.DataFrame(rows), AS_OF)


@pytest.mark.parametrize(
    ("next_opened", "is_repeat", "while_open"),
    [
        ("2026-01-10", True, False),  # same day the previous fault closed: after-close, gap 0
        ("2026-01-24", True, False),  # exactly 14 days after close: inside the window
        ("2026-01-25", False, False),  # 15 days after close: outside
        ("2026-01-09", False, True),  # one day before close: opened while still open
    ],
)
def test_repeat_window_edges(next_opened, is_repeat, while_open):
    rows = metrics.repeat_faults(_pair("2026-01-10", next_opened), 14).rows.iloc[1]
    assert rows["counted"] and rows["is_repeat"] == is_repeat and rows["overlap"] == while_open


def test_fault_opened_while_previous_still_open_is_not_a_repeat():
    prep = _pair("", "2026-01-05", prev_status="In Progress")
    row = metrics.repeat_faults(prep, 14).rows.iloc[1]
    assert row["counted"] and not row["is_repeat"] and row["overlap"]


def test_repeat_exclusions_are_counted(repeat):
    s = repeat.summary
    assert s["total_rows"] == 20
    assert s["excluded_unclassified"] == 7  # R10 plus the six FRM-01 rows
    assert s["excluded_text_mismatch"] == 1  # R11
    assert s["excluded_no_open_date"] == 0
    assert s["not_evaluable"] == 1  # R14: previous fault closed with unknown date
    assert s["total_rows"] - s["denominator"] == 7 + 1 + 0 + 1


def test_excluded_rows_do_not_act_as_the_previous_fault(prep):
    """R12 would be a repeat of R11 (13 days after it closed) if R11 were not excluded."""
    rows = metrics.repeat_faults(prep).rows.set_index(prep["report_id"])
    assert not rows.at["R12", "is_repeat"] and rows.at["R12", "counted"]
    assert not rows.at["R11", "counted"] and rows.at["R11", "exclusion"] == "category_text_mismatch"
    assert rows.at["R14", "exclusion"] == "previous_close_unknown"


def test_repeat_by_system_and_vendor(prep, repeat):
    by_system = metrics.repeat_by(prep, repeat, "system_id", 1).set_index("system_id")
    assert by_system.at["CTL-01", "counted_faults"] == 5 and by_system.at["CTL-01", "repeats"] == 1
    assert by_system.at["COM-01", "repeat_rate"] == 0.0  # opened while the previous was open
    assert "FRM-01" not in by_system.index  # all its rows are excluded
    by_vendor = metrics.repeat_by(prep, repeat, "vendor", 1).set_index("vendor")
    assert by_vendor.at["Vendor B", "repeat_rate"] == 0.0
    assert by_vendor.at["Vendor A", "repeat_rate"] == pytest.approx(1 / 5)


def test_system_repeat_rate_vs_fleet_average(prep, repeat):
    fleet = 1 / 11  # all repeats / all counted faults, not the mean of the system rates
    by_system = metrics.repeat_by_system(prep, repeat, min_sample=1).set_index("system_id")
    ctl, com = by_system.loc["CTL-01"], by_system.loc["COM-01"]
    assert ctl["fleet_rate"] == pytest.approx(fleet)
    assert ctl["vs_fleet_pts"] == pytest.approx((0.2 - fleet) * 100)  # +10.9 points
    assert ctl["vs_fleet_x"] == pytest.approx(0.2 / fleet)  # 2.2x
    assert com["vs_fleet_pts"] == pytest.approx(-fleet * 100) and com["vs_fleet_x"] == 0.0
    assert {"vendor", "site", "tickets", "median_days"} <= set(by_system.columns)


def test_window_sensitivity_7_14_30(prep):
    table = metrics.repeat_sensitivity(prep).set_index("window_days")
    assert table.index.tolist() == [7, 14, 30]
    assert table["repeats"].tolist() == [0, 1, 2]
    assert table["repeat_rate"].tolist() == pytest.approx([0, 1 / 11, 2 / 11])
    assert (table["counted_faults"] == 11).all()


def test_sensitivity_always_includes_the_configured_window(prep):
    table = metrics.repeat_sensitivity(prep, configured=21)
    assert table["window_days"].tolist() == [7, 14, 21, 30]


def test_top_repeat_offenders_ranked_by_rate(prep, repeat):
    top = metrics.top_repeat_offenders(prep, repeat, n=3, min_sample=1)
    assert top["system_id"].tolist() == ["CTL-01", "COM-01", "PWR-01"]  # ties: system_id order
    assert top["rank"].tolist() == [1, 2, 3]
    first = top.iloc[0]
    assert (first["vendor"], first["site"], first["tickets"]) == ("Vendor A", "North", 5)
    assert (first["resolved_n"], first["median_days"]) == (4, 3.0)


# --- backlog aging -----------------------------------------------------------------------------
def test_aging_buckets_overall_with_edge_ages(prep):
    row = metrics.aging(prep).iloc[0]
    # open ages: 11, 39, 6 (R05, R08, R09) and 7, 8, 30, 31, 90, 91 (bucket edges)
    assert row["open"] == 9
    assert (row["0-7 days"], row["8-30 days"], row["31-90 days"], row["91+ days"]) == (2, 3, 3, 1)
    assert row["over_30_days"] == 4
    assert row["median_age_days"] == 30.0 and row["oldest_days"] == 91


def test_aging_by_site(prep):
    by_site = metrics.aging(prep, "site").set_index("site")
    north, south = by_site.loc["North"], by_site.loc["South"]
    assert (north["open"], north["0-7 days"], north["8-30 days"]) == (7, 1, 3)
    assert (north["31-90 days"], north["91+ days"], north["over_30_days"]) == (2, 1, 3)
    assert north["median_age_days"] == 30.0
    assert (south["open"], south["0-7 days"], south["31-90 days"]) == (2, 1, 1)
    assert south["median_age_days"] == 22.5 and south["oldest_days"] == 39


def test_aging_is_measured_against_the_fixed_as_of_date():
    later = metrics.prepare(_fixture(), date(2026, 4, 30))
    ages = later.loc[later["is_open"]].set_index("report_id")["age_days"]
    assert ages["R05"] == 41  # 11 days at 2026-03-31, 30 days later
    buckets = metrics.aging(later).iloc[0]
    # ages 30 days later: 41, 69, 36, 37, 38, 60, 61 (31-90) and 120, 121 (91+)
    assert (buckets["31-90 days"], buckets["91+ days"]) == (7, 2)


# --- monthly trend -----------------------------------------------------------------------------
def test_monthly_trend(prep):
    table = metrics.monthly_trend(prep, AS_OF, min_sample=3).set_index("month")
    assert table.index.tolist() == ["2025-12", "2026-01", "2026-02", "2026-03"]
    assert table["opened"].tolist() == [2, 5, 6, 7]
    assert table["closed"].tolist() == [0, 3, 5, 2]
    assert table["net_opened"].tolist() == [2, 2, 1, 5]
    assert np.isnan(table.at["2025-12", "median_days"])
    assert table["median_days"].tolist()[1:] == [4.0, 8.0, 2.5]
    assert table.at["2026-01", "mean_days"] == pytest.approx(16 / 3)
    assert table["open_at_month_end"].tolist() == [2, 3, 4, 9]
    assert table["low_sample"].tolist() == [True, False, False, True]


# --- planted patterns on the seeded dataset ----------------------------------------------------
@pytest.fixture(scope="module")
def checks(pipeline_run):
    m = metrics.compute_all(pipeline_run.classified.final)
    table = evaluate.planted_pattern_checks(m, pipeline_run.scenario).set_index("pattern")
    return m, table


@pytest.mark.parametrize(
    "pattern",
    [
        "slow vendors",
        "long-tail vendor",
        "backlog site",
        pytest.param(
            "lemon systems",
            marks=pytest.mark.xfail(
                strict=True,
                reason=(
                    "FINDING, not a bug: with the after-close, 14-day definition only 2 of the 5 "
                    "lemon systems reach the top 5. The metric catches ~52% of the planted "
                    "recurrences (38% of them open while the parent fault is still open, which "
                    "the rule excludes by design) and ~85% of what it flags is coincidence at "
                    "this ticket density. See the window table in the metrics report."
                ),
            ),
        ),
    ],
)
def test_planted_patterns_are_surfaced(checks, pattern):
    _, table = checks
    assert table.at[pattern, "found"] == "yes", table.at[pattern, "detail"]


def test_lemon_signal_exists_but_is_weak(pipeline_run, checks):
    m, _ = checks
    by_system = m.repeat_by_system.set_index("system_id")
    lemons = list(pipeline_run.scenario["lemon_systems"])
    gap = by_system.loc[lemons, "repeat_rate"].mean() - by_system.drop(lemons)["repeat_rate"].mean()
    assert 0.03 < gap < 0.15  # lemons repeat more (about +7 points), nowhere near separated
    top5 = set(by_system.sort_values("repeat_rate", ascending=False).head(5).index)
    assert 2 <= len(top5 & set(lemons)) <= 3


def test_metrics_report_states_exclusions_and_low_sample_rule(pipeline_run, checks):
    m, table = checks
    text = metrics.to_markdown(m, table.reset_index())
    for needle in (
        "Excluded from numerator **and** denominator",
        "category_text_mismatch",
        "low sample",
        "vs_fleet_pts",
        "is not a repeat",
        "window",
        "Did the metrics surface the planted patterns",
    ):
        assert needle in text
    s = m.repeat.summary
    assert f"{s['excluded_unclassified']} rows with an Unclassified/Unknown category" in text
