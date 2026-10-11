from datetime import date

import numpy as np
import pandas as pd
import pytest

from field_ops import config
from field_ops.clean import (
    UNKNOWN,
    build_roster,
    clean_category,
    clean_reporter,
    clean_severity,
    clean_site,
    clean_status,
    clean_system_id,
    clean_vendor,
    derive_system_type,
    normalize_name,
    parse_date,
)
from field_ops.mess import RAW_COLUMNS, typo


# --- vendors: pattern only, never similarity -------------------------------------------------
@pytest.mark.parametrize(
    "raw",
    ["Vendor A", "vendor a", "VENDOR A", "Vendor-A", "Vndr A", "VendorA", " vendor a",
     "Vendor  A", "Vendor A ", "Vendor A Ltd", "Vendr A"],
)  # fmt: skip
def test_vendor_variants_resolve_to_canonical(raw):
    assert clean_vendor(raw)[0] == "Vendor A"


@pytest.mark.parametrize(
    ("raw", "reason"),
    [("", "unknown_blank"), ("A", "unknown_unparseable"), ("Vendor", "unknown_unparseable"),
     ("Vednor A", "unknown_unparseable"), ("Vendor Z", "unknown_not_a_vendor")],
)  # fmt: skip
def test_unparseable_vendor_becomes_unknown_with_reason(raw, reason):
    assert clean_vendor(raw) == (UNKNOWN, reason)


def test_vendor_typos_never_turn_into_another_vendor():
    rng = np.random.default_rng(0)
    for vendor in config.VENDORS:
        for _ in range(100):
            assert clean_vendor(typo(rng, vendor))[0] in (vendor, UNKNOWN)


# --- categories: alias table + fuzzy ---------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Network", "Communications"), ("Detector", "Sensor"), ("SW", "Software"),
     ("Cal/Align", "Calibration"), ("Battery/Power", "Power")],
)  # fmt: skip
def test_alias_table_resolves_labels_spelling_cannot(raw, expected):
    assert clean_category(raw) == (expected, "alias")


@pytest.mark.parametrize(
    ("raw", "expected", "method"),
    [
        ("Comunications", "Communications", "fuzzy"),
        ("Mechancial", "Mechanical", "fuzzy"),
        ("Powr", "Power", "fuzzy"),
        ("Snsor", "Sensor", "fuzzy"),
        ("Poewr", "Power", "fuzzy_low"),  # one swapped letter in a 5-letter word scores 80
        ("Snesor", "Sensor", "fuzzy_low"),  # 83.3
    ],
)
def test_fuzzy_fixes_spelling_typos_and_grades_confidence(raw, expected, method):
    assert clean_category(raw) == (expected, method)


def test_fuzzy_confidence_band_is_80_to_85():
    from rapidfuzz import fuzz

    from field_ops.clean import FUZZY_CONFIDENT, FUZZY_CUTOFF

    assert (FUZZY_CUTOFF, FUZZY_CONFIDENT) == (80, 85)
    assert fuzz.ratio("poewr", "power") == 80
    assert 80 <= fuzz.ratio("snesor", "sensor") < 85


def test_category_normalization_without_fuzzy():
    assert clean_category("COMMUNICATIONS") == ("Communications", "pattern")
    assert clean_category("Power") == ("Power", "exact")


@pytest.mark.parametrize("raw", ["Hardware", "Other", "TBD", "Radio gear"])
def test_unmatched_category_is_unknown_not_guessed(raw):
    assert clean_category(raw) == (UNKNOWN, "unknown_no_match")


def test_blank_category_is_unknown_blank():
    assert clean_category("  ") == (UNKNOWN, "unknown_blank")


def test_fuzzy_recovers_unseen_typos_without_wrong_matches():
    """Typos generated independently of the generator's variant list."""
    rng = np.random.default_rng(123)
    results = {"correct": 0, "unknown": 0, "wrong": 0}
    for category in config.CATEGORIES:
        for _ in range(200):
            value = clean_category(typo(rng, category))[0]
            key = "correct" if value == category else "unknown" if value == UNKNOWN else "wrong"
            results[key] += 1
    assert results["wrong"] == 0
    assert results["correct"] / sum(results.values()) > 0.98


def test_site_suffix_and_typo():
    assert clean_site("Northfield Site")[0] == "Northfield"
    assert clean_site("  harbor point")[0] == "Harbor Point"
    assert clean_site("Lakesde") == ("Lakeside", "fuzzy")
    assert clean_site("Mars Base")[0] == UNKNOWN


# --- severity / status / system -------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [("low", "Low"), ("L", "Low"), ("Med", "Medium"), ("Moderate", "Medium"), ("HIGH", "High"),
     ("Hi", "High"), ("Crit", "Critical")],
)  # fmt: skip
def test_severity_mapping(raw, expected):
    assert clean_severity(raw)[0] == expected


def test_unknown_severity_and_status_are_not_guessed():
    assert clean_severity("urgent") == (UNKNOWN, "unknown_no_match")
    assert clean_status("pending")[0] == UNKNOWN


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("closed", "Closed"), ("Resolved", "Closed"), ("Done", "Closed"), ("New", "Open"),
     ("WIP", "In Progress"), ("in-progress", "In Progress")],
)  # fmt: skip
def test_status_mapping(raw, expected):
    assert clean_status(raw)[0] == expected


@pytest.mark.parametrize("raw", ["ctl-03", "CTL03", "CTL-3", "CTL 03", " CTL-03 "])
def test_system_id_variants(raw):
    assert clean_system_id(raw)[0] == "CTL-03"


def test_system_id_with_unknown_prefix_is_unknown():
    assert clean_system_id("XYZ-01") == (UNKNOWN, "unknown_unparseable")


def test_system_type_comes_from_id_prefix():
    assert derive_system_type("COM-04", "garbage")[0] == "Comms Module"
    assert derive_system_type(UNKNOWN, "sensor module")[0] == "Sensor Module"
    assert derive_system_type(UNKNOWN, "garbage")[0] == UNKNOWN


# --- reporters -------------------------------------------------------------------------------
def test_name_formats_and_initial_expansion():
    assert normalize_name("ADLER, ruth") == "Ruth Adler"
    roster = build_roster(pd.Series(["Ruth Adler", "Tom Reyes"]))
    assert clean_reporter("R. Adler", roster) == ("Ruth Adler", "initial_expanded")
    assert clean_reporter("t. reyes", roster)[0] == "Tom Reyes"


def test_ambiguous_initial_is_not_expanded():
    roster = build_roster(pd.Series(["Ruth Adler", "Rita Adler"]))
    assert clean_reporter("R. Adler", roster) == ("R. Adler", "partial_name")


# --- dates -----------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected", "method"),
    [
        ("2026-03-04", date(2026, 3, 4), "iso"),
        ("2026-03-04 14:05", date(2026, 3, 4), "iso"),
        ("04/03/2026", date(2026, 3, 4), "dmy_ambiguous"),
        ("25/03/2026", date(2026, 3, 25), "dmy"),
        ("Mar 4 2026", date(2026, 3, 4), "text"),
        ("45932", date(2025, 10, 2), "excel_serial"),
    ],
)
def test_date_formats(raw, expected, method):
    assert parse_date(raw) == (expected, method)


@pytest.mark.parametrize(
    "raw", ["31/02/2026", "03/25/2026", "yesterday", "2026-13-01", "Foo 4 2026"]
)
def test_unparseable_dates_return_none(raw):
    assert parse_date(raw) == (None, "unparseable")


def test_blank_date():
    assert parse_date("  ") == (None, "blank")


# --- validation rules through the full pipeline ---------------------------------------------
def _row(**overrides) -> dict:
    base = {
        "report_id": "FR-10001", "date_opened": "2026-03-01", "date_closed": "2026-03-05",
        "site": "Lakeside", "system_type": "Sensor Module", "system_id": "SEN-01",
        "vendor": "Vendor A", "fault_category": "Sensor", "severity": "High",
        "description": "Camera module produces a blurry image.", "reported_by": "Ruth Adler",
        "status": "Closed",
    }  # fmt: skip
    return {**base, **overrides}


def _clean(*rows: dict):
    from field_ops.clean import clean_reports

    return clean_reports(pd.DataFrame(rows, columns=RAW_COLUMNS))


def test_close_before_open_removes_close_date_and_flags():
    out = _clean(_row(date_closed="2026-02-20")).clean.iloc[0]
    assert out["date_closed"] == ""
    assert "close_before_open" in out["dq_flags"]
    assert "closed_no_close_date" in out["dq_flags"]  # status stays Closed, excluded from MTTR


def test_open_status_with_close_date_becomes_closed():
    out = _clean(_row(status="Open")).clean.iloc[0]
    assert out["status"] == "Closed"
    assert "status_closed_from_date" in out["dq_flags"]


def test_closed_without_close_date_is_flagged_not_changed():
    out = _clean(_row(date_closed="")).clean.iloc[0]
    assert out["status"] == "Closed"
    assert "closed_no_close_date" in out["dq_flags"]


def test_date_outside_window_is_dropped():
    result = _clean(_row(date_opened="2031-01-01", date_closed=""))
    assert result.clean.iloc[0]["date_opened"] == ""
    assert result.rows.iloc[0]["m_date_opened"] == "out_of_range"


def test_ledger_counts_rows_changed():
    result = _clean(_row(vendor="vndr a"), _row(report_id="FR-10002", system_id="SEN-02"))
    entry = result.ledger.to_frame().query("rule.str.startswith('vendor')").iloc[0]
    assert entry["rows"] == 1
    assert "vndr a" in entry["example"]


def test_low_confidence_fuzzy_match_is_flagged_not_blocked():
    result = _clean(_row(fault_category="Poewr"), _row(report_id="FR-10002", system_id="SEN-02"))
    flagged, normal = result.clean.iloc[0], result.clean.iloc[1]
    assert flagged["fault_category"] == "Power"  # still resolved...
    assert "fuzzy_low_confidence" in flagged["dq_flags"]  # ...but marked for review
    assert "fuzzy_low_confidence" not in normal["dq_flags"]


def test_low_confidence_site_match_is_flagged_too():
    from field_ops.clean import clean_site

    assert clean_site("Ridgevew")[1] in ("fuzzy", "fuzzy_low")
    result = _clean(_row(site="Hbor Point"))
    flags = result.clean.iloc[0]["dq_flags"]
    method = result.rows.iloc[0]["m_site"]
    assert ("fuzzy_low_confidence" in flags) == (method == "fuzzy_low")


def test_confident_fuzzy_match_is_not_flagged():
    result = _clean(_row(fault_category="Comunications"))
    assert "fuzzy_low_confidence" not in result.clean.iloc[0]["dq_flags"]
