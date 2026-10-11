"""Keep rule and near-duplicate detection, tested on small hand-built frames."""

import pandas as pd

from field_ops.clean import UNKNOWN, clean_reports
from field_ops.dedupe import choose_survivor, completeness, near_duplicate_clusters
from field_ops.mess import RAW_COLUMNS


def _ticket(report_id, *, status="Closed", closed="2026-03-05", **overrides) -> dict:
    base = {
        "report_id": report_id,
        "date_opened": pd.Timestamp("2026-03-01"),
        "date_closed": pd.Timestamp(closed) if closed else pd.NaT,
        "site": "Lakeside",
        "system_type": "Sensor Module",
        "system_id": "SEN-01",
        "vendor": "Vendor A",
        "fault_category": "Sensor",
        "severity": "High",
        "description": "Camera module produces a blurry image.",
        "description_key": "camera module produces a blurry image",
        "reported_by": "Ruth Adler",
        "status": status,
    }
    return {**base, **overrides}


def _cluster(*tickets: dict) -> pd.DataFrame:
    return pd.DataFrame(tickets, index=range(len(tickets)))


# --- keep rule: closed with close date > most complete > earliest ----------------------------
def test_keep_rule_prefers_closed_with_close_date_over_more_complete_open():
    open_complete = _ticket("FR-10001", status="Open", closed=None)
    closed_sparse = _ticket("FR-10002", reported_by=UNKNOWN, severity=UNKNOWN, vendor=UNKNOWN)
    assert completeness(open_complete) > completeness(closed_sparse)
    assert choose_survivor(_cluster(open_complete, closed_sparse)) == 1


def test_keep_rule_closed_without_close_date_does_not_count_as_closed():
    closed_no_date = _ticket("FR-10001", closed=None)  # status Closed, date missing
    open_report = _ticket("FR-10002", status="Open", closed=None, reported_by=UNKNOWN)
    # Neither qualifies for rule 1, so completeness decides and the earlier one wins.
    assert choose_survivor(_cluster(closed_no_date, open_report)) == 0


def test_keep_rule_among_closed_prefers_most_complete():
    sparse = _ticket("FR-10001", fault_category=UNKNOWN, reported_by=UNKNOWN)
    complete = _ticket("FR-10002")
    assert choose_survivor(_cluster(sparse, complete)) == 1


def test_keep_rule_tie_goes_to_earliest_report():
    later = _ticket("FR-10009")
    earlier = _ticket("FR-10003")
    assert choose_survivor(_cluster(later, earlier)) == 1


def test_keep_rule_orders_report_ids_numerically_not_as_text():
    assert choose_survivor(_cluster(_ticket("FR-9999"), _ticket("FR-10000"))) == 0


def test_keep_rule_applies_all_three_criteria_in_order():
    cluster = _cluster(
        _ticket("FR-10001", status="Open", closed=None),
        _ticket("FR-10002", reported_by=UNKNOWN),
        _ticket("FR-10003"),
        _ticket("FR-10004"),
    )
    assert choose_survivor(cluster) == 2  # closed + complete, earlier of the two ties


# --- near-duplicate detection ----------------------------------------------------------------
def _clusters(*tickets: dict) -> list[int]:
    return near_duplicate_clusters(_cluster(*tickets)).tolist()


def test_same_system_day_category_and_text_are_merged():
    twin = _ticket("FR-10002", description_key="camera module produces a blurry imag")
    assert _clusters(_ticket("FR-10001"), twin) == [0, 0]


def test_different_day_is_a_repeat_fault_not_a_duplicate():
    later = _ticket("FR-10002", date_opened=pd.Timestamp("2026-03-08"))
    assert _clusters(_ticket("FR-10001"), later) == [0, 1]


def test_different_system_is_not_merged():
    assert _clusters(_ticket("FR-10001"), _ticket("FR-10002", system_id="SEN-02")) == [0, 1]


def test_conflicting_known_categories_are_not_merged():
    assert _clusters(_ticket("FR-10001"), _ticket("FR-10002", fault_category="Power")) == [0, 1]


def test_unknown_category_is_compatible_with_any_category():
    assert _clusters(_ticket("FR-10001"), _ticket("FR-10002", fault_category=UNKNOWN)) == [0, 0]


def test_different_text_is_not_merged():
    other = _ticket("FR-10002", description_key="operator reports voltage regulator overheats")
    assert _clusters(_ticket("FR-10001"), other) == [0, 1]


def test_chain_of_duplicates_forms_one_cluster():
    assert _clusters(_ticket("FR-10001"), _ticket("FR-10002"), _ticket("FR-10003")) == [0, 0, 0]


# --- end to end through clean_reports --------------------------------------------------------
def _raw_row(report_id, **overrides) -> dict:
    base = {
        "report_id": report_id, "date_opened": "2026-03-01", "date_closed": "2026-03-05",
        "site": "Lakeside", "system_type": "Sensor Module", "system_id": "SEN-01",
        "vendor": "Vendor A", "fault_category": "Sensor", "severity": "High",
        "description": "Camera module produces a blurry image.", "reported_by": "Ruth Adler",
        "status": "Closed",
    }  # fmt: skip
    return {**base, **overrides}


def test_exact_duplicate_rows_are_removed():
    raw = pd.DataFrame([_raw_row("FR-10001")] * 2, columns=RAW_COLUMNS)
    result = clean_reports(raw)
    assert len(result.clean) == 1
    assert result.clean.iloc[0]["dup_count"] == 2


def test_reentered_ticket_keeps_the_closed_copy_even_if_it_is_later():
    original = _raw_row("FR-10001", status="Open", date_closed="")
    reentry = _raw_row("FR-10002", description="camera module produces a blurry image")
    result = clean_reports(pd.DataFrame([original, reentry], columns=RAW_COLUMNS))
    kept = result.clean.iloc[0]
    assert len(result.clean) == 1
    assert kept["report_id"] == "FR-10002"
    assert kept["status"] == "Closed"
    assert kept["merged_report_ids"] == "FR-10001;FR-10002"
    assert "duplicate_merged" in kept["dq_flags"]
